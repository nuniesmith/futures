# ==============================================================================
# deploy_nt8.ps1  --  NinjaTrader 8 Deploy Script
# ==============================================================================
#
# Pulls the latest CS source files, OnnxRuntime DLLs, and the champion ONNX
# model directly from github.com/nuniesmith/futures and deploys them to the
# local NinjaTrader 8 installation.
#
# Source layout in repo:
#   src/ninja/BreakoutStrategy.cs        -> Strategies\
#   src/ninja/RubyIndicator.cs           -> Indicators\
#   src/ninja/addons/Bridge.cs           -> AddOns\
#   src/ninja/addons/DataPreloader.cs    -> AddOns\
#   src/ninja/dll/*.dll                  -> bin\Custom\  (root, beside .csproj)
#   models/breakout_cnn_best.onnx        -> bin\Custom\Models\
#
# What the script does:
#   1. Resolves the NT8 Custom directory (auto-detected or via -NtCustomDir)
#   2. Downloads all files from the repo (or uses local copies if -LocalRepo)
#   3. Copies CS files into the correct NT8 subdirectories
#   4. Copies DLLs into bin\Custom\
#   5. Copies the ONNX model into bin\Custom\Models\
#   6. Patches NinjaTrader.Custom.csproj with the required <Reference> entries
#   7. Optionally launches NinjaTrader 8
#
# Usage:
#   .\scripts\deploy_nt8.ps1
#   .\scripts\deploy_nt8.ps1 -DryRun
#   .\scripts\deploy_nt8.ps1 -NoDlls
#   .\scripts\deploy_nt8.ps1 -NoModel
#   .\scripts\deploy_nt8.ps1 -NtCustomDir "D:\NinjaTrader 8\bin\Custom"
#   .\scripts\deploy_nt8.ps1 -LocalRepo "C:\code\futures"
#   .\scripts\deploy_nt8.ps1 -Branch "dev"
#   .\scripts\deploy_nt8.ps1 -Launch
#
# Requirements:
#   - PowerShell 5.1+ (ships with Windows 10/11)
#   - Internet access (or -LocalRepo for offline deploy)
#   - NinjaTrader 8 must be CLOSED before running
#
# ==============================================================================

[CmdletBinding(SupportsShouldProcess)]
param(
    # Dry-run: show what would happen without writing anything
    [switch]$DryRun,

    # Skip DLL copy (use if DLLs are already deployed)
    [switch]$NoDlls,

    # Skip ONNX model copy
    [switch]$NoModel,

    # Skip CS source copy
    [switch]$NoSource,

    # Skip patching NinjaTrader.Custom.csproj
    [switch]$NoPatch,

    # Override NT8 Custom directory (auto-detected by default)
    [string]$NtCustomDir = "",

    # Use a local clone of the futures repo instead of downloading from GitHub
    [string]$LocalRepo = "",

    # GitHub branch to pull from (default: main)
    [string]$Branch = "main",

    # GitHub personal access token (for private repos or rate-limit avoidance)
    [string]$GitHubToken = $env:GITHUB_TOKEN,

    # Launch NinjaTrader 8 after deploy
    [switch]$Launch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ==============================================================================
# Configuration
# ==============================================================================

$RepoOwner  = "nuniesmith"
$RepoName   = "futures"
$RepoSlug   = "$RepoOwner/$RepoName"
$RawBase    = "https://raw.githubusercontent.com/$RepoSlug/$Branch"
$LfsApiUrl  = "https://github.com/$RepoSlug.git/info/lfs/objects/batch"

# OnnxRuntime version -- must match src/ninja/dll/OnnxFetch.csproj
$OnnxVersion = "1.24.2"

# Files to deploy: each entry has RepoPath and DestSubdir keys.
#   RepoPath   -- path inside the repo (relative to repo root)
#   DestSubdir -- subdirectory inside bin\Custom\ (empty string = root)
$SourceFiles = @(
    @{ Repo = "src/ninja/BreakoutStrategy.cs";     Dest = "Strategies" },
    @{ Repo = "src/ninja/RubyIndicator.cs";        Dest = "Indicators" },
    @{ Repo = "src/ninja/addons/Bridge.cs";        Dest = "AddOns"     },
    @{ Repo = "src/ninja/addons/DataPreloader.cs"; Dest = "AddOns"     }
)

$DllFiles = @(
    "src/ninja/dll/Microsoft.ML.OnnxRuntime.dll",
    "src/ninja/dll/System.Buffers.dll",
    "src/ninja/dll/System.Memory.dll",
    "src/ninja/dll/System.Numerics.Vectors.dll",
    "src/ninja/dll/System.Runtime.CompilerServices.Unsafe.dll",
    "src/ninja/dll/onnxruntime.dll",
    "src/ninja/dll/onnxruntime_providers_shared.dll"
)

# DLL names that need <Reference> entries in NinjaTrader.Custom.csproj
# (managed DLLs only -- native DLLs are loaded at runtime, not referenced)
$ManagedDlls = @(
    "Microsoft.ML.OnnxRuntime",
    "System.Buffers",
    "System.Memory",
    "System.Numerics.Vectors",
    "System.Runtime.CompilerServices.Unsafe"
)

$ModelRepoPath = "models/breakout_cnn_best.onnx"
$ModelDestName = "breakout_cnn_best.onnx"

# ==============================================================================
# Colours / logging
# ==============================================================================

function Write-Step  ([string]$msg) { Write-Host "  --> $msg" -ForegroundColor Cyan    }
function Write-Ok    ([string]$msg) { Write-Host "  [+] $msg" -ForegroundColor Green   }
function Write-Warn  ([string]$msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow  }
function Write-Fail  ([string]$msg) { Write-Host "  [x] $msg" -ForegroundColor Red     }
function Write-Dim   ([string]$msg) { Write-Host "      $msg" -ForegroundColor DarkGray }

function Write-Banner {
    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor DarkCyan
    Write-Host "   NinjaTrader 8 Deploy  --  github.com/$RepoSlug"              -ForegroundColor Cyan
    Write-Host "   Branch : $Branch"                                             -ForegroundColor DarkCyan
    if ($DryRun) {
        Write-Host "   Mode   : DRY RUN (no files will be written)"              -ForegroundColor Yellow
    } else {
        Write-Host "   Mode   : LIVE"                                            -ForegroundColor DarkCyan
    }
    Write-Host "  ============================================================" -ForegroundColor DarkCyan
    Write-Host ""
}

# ==============================================================================
# NT8 directory detection
# ==============================================================================

function Find-NtCustomDir {
    # Common install locations (Documents may be redirected to OneDrive)
    $candidates = @(
        [System.IO.Path]::Combine($env:USERPROFILE, "Documents", "NinjaTrader 8", "bin", "Custom"),
        [System.IO.Path]::Combine($env:ONEDRIVE,    "Documents", "NinjaTrader 8", "bin", "Custom"),
        "C:\Users\$env:USERNAME\Documents\NinjaTrader 8\bin\Custom",
        "D:\NinjaTrader 8\bin\Custom"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

# ==============================================================================
# LFS resolution
# ==============================================================================

function Test-LfsPointer([string]$filePath) {
    if (-not (Test-Path $filePath)) { return $false }
    $size = (Get-Item $filePath).Length
    if ($size -gt 1024) { return $false }
    $firstLine = Get-Content $filePath -TotalCount 1 -ErrorAction SilentlyContinue
    return ($firstLine -match "^version https://git-lfs.github.com")
}

function Resolve-LfsUrl([string]$oid, [long]$size) {
    $payload = @{
        operation = "download"
        transfer  = @("basic")
        objects   = @(@{ oid = $oid; size = $size })
    } | ConvertTo-Json -Depth 5

    $headers = @{
        "Accept"       = "application/vnd.git-lfs+json"
        "Content-Type" = "application/vnd.git-lfs+json"
    }
    if ($GitHubToken) { $headers["Authorization"] = "token $GitHubToken" }

    try {
        $response = Invoke-RestMethod -Uri $LfsApiUrl -Method Post `
            -Headers $headers -Body $payload -ErrorAction Stop
        $href = $response.objects[0].actions.download.href
        if (-not $href) { throw "No download href in LFS batch response" }
        return $href
    } catch {
        throw "LFS batch API failed: $_"
    }
}

# ==============================================================================
# Download helper
# ==============================================================================

function Get-RepoFile([string]$repoPath, [string]$destPath) {
    # Build dest directory
    $destDir = Split-Path $destPath -Parent
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    }

    if ($LocalRepo) {
        # -- Local repo copy --------------------------------------------------
        $src = Join-Path $LocalRepo ($repoPath -replace "/", "\")
        if (-not (Test-Path $src)) {
            throw "Local file not found: $src"
        }
        if ($DryRun) {
            Write-Dim "  [dry] copy $src  ->  $destPath"
        } else {
            Copy-Item $src $destPath -Force
        }
        return
    }

    # -- Download from GitHub raw --------------------------------------------
    $url = "$RawBase/$repoPath"
    $tmp = "$destPath.tmp"

    $dlHeaders = @{}
    if ($GitHubToken) { $dlHeaders["Authorization"] = "token $GitHubToken" }

    if ($DryRun) {
        Write-Dim "  [dry] download $url  ->  $destPath"
        return
    }

    try {
        Invoke-WebRequest -Uri $url -OutFile $tmp -Headers $dlHeaders `
            -UseBasicParsing -ErrorAction Stop
    } catch {
        Remove-Item $tmp -ErrorAction SilentlyContinue
        throw "Failed to download ${repoPath}: $_"
    }

    # Check for LFS pointer
    if (Test-LfsPointer $tmp) {
        Write-Dim "    Resolving Git LFS pointer..."
        $lines   = Get-Content $tmp
        $oid     = ($lines | Where-Object { $_ -match "^oid sha256:" }) -replace "^oid sha256:", ""
        $lfsSize = [long](($lines | Where-Object { $_ -match "^size " }) -replace "^size ", "")

        if (-not $oid -or -not $lfsSize) {
            Remove-Item $tmp -ErrorAction SilentlyContinue
            throw "Failed to parse LFS pointer for $repoPath"
        }

        $lfsUrl = Resolve-LfsUrl $oid $lfsSize
        Write-Dim "    Downloading from LFS storage..."
        try {
            Invoke-WebRequest -Uri $lfsUrl -OutFile $tmp -UseBasicParsing -ErrorAction Stop
        } catch {
            Remove-Item $tmp -ErrorAction SilentlyContinue
            throw "Failed to download LFS content for ${repoPath}: $_"
        }

        # Verify size
        $actualSize = (Get-Item $tmp).Length
        if ($actualSize -ne $lfsSize) {
            Remove-Item $tmp -ErrorAction SilentlyContinue
            throw "Size mismatch for ${repoPath}: expected $lfsSize bytes, got $actualSize"
        }
    }

    Move-Item $tmp $destPath -Force
}

# ==============================================================================
# csproj patching
# ==============================================================================

function Update-CsprojReferences([string]$csprojPath) {
    if (-not (Test-Path $csprojPath)) {
        Write-Warn "NinjaTrader.Custom.csproj not found at: $csprojPath"
        Write-Warn "Skipping csproj patch -- add <Reference> entries manually if needed"
        return
    }

    [xml]$xml = Get-Content $csprojPath -Encoding UTF8
    $ns = $xml.DocumentElement.NamespaceURI

    # Find or create the first <ItemGroup> for references
    $nsm = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    if ($ns) { $nsm.AddNamespace("ms", $ns) }

    $itemGroups = $xml.Project.ItemGroup
    # Use the first ItemGroup that already has a Reference, or the first one
    $refGroup = $null
    foreach ($ig in $itemGroups) {
        if ($ig.Reference) { $refGroup = $ig; break }
    }
    if (-not $refGroup) {
        $refGroup = $xml.CreateElement("ItemGroup", $ns)
        $xml.Project.AppendChild($refGroup) | Out-Null
    }

    $changed = $false
    foreach ($dllName in $ManagedDlls) {
        # Check if this reference already exists
        $existing = $null
        if ($refGroup.Reference) {
            foreach ($r in $refGroup.Reference) {
                if ($r.Include -eq $dllName) { $existing = $r; break }
            }
        }

        if ($existing) {
            Write-Dim "    <Reference Include=`"$dllName`"> already present"
            continue
        }

        # Add new reference
        $refElem = $xml.CreateElement("Reference", $ns)
        $refElem.SetAttribute("Include", $dllName)

        $hintElem = $xml.CreateElement("HintPath", $ns)
        $hintElem.InnerText = "$dllName.dll"
        $refElem.AppendChild($hintElem) | Out-Null

        $privElem = $xml.CreateElement("Private", $ns)
        $privElem.InnerText = "False"
        $refElem.AppendChild($privElem) | Out-Null

        $refGroup.AppendChild($refElem) | Out-Null
        Write-Ok "  Added <Reference Include=`"$dllName`">"
        $changed = $true
    }

    if ($changed -and -not $DryRun) {
        $xml.Save($csprojPath)
        Write-Ok "NinjaTrader.Custom.csproj patched"
    } elseif (-not $changed) {
        Write-Ok "NinjaTrader.Custom.csproj -- all references already present"
    } else {
        Write-Dim "  [dry] would save patched csproj"
    }
}

# ==============================================================================
# NT8 running check
# ==============================================================================

function Assert-NtNotRunning {
    $nt = Get-Process -Name "NinjaTrader" -ErrorAction SilentlyContinue
    if ($nt) {
        Write-Fail "NinjaTrader 8 is currently running."
        Write-Fail "Please close it before deploying -- NT8 holds a file lock on the OnnxRuntime DLLs."
        Write-Host ""
        exit 1
    }
}

# ==============================================================================
# Main
# ==============================================================================

Write-Banner

# -- Check NT8 is not running -------------------------------------------------
if (-not $DryRun) {
    Assert-NtNotRunning
}

# -- Resolve NT8 Custom dir ---------------------------------------------------
if (-not $NtCustomDir) {
    $NtCustomDir = Find-NtCustomDir
}

if (-not $NtCustomDir) {
    Write-Fail "Could not find NinjaTrader 8 Custom directory."
    Write-Host ""
    Write-Host "  Specify it manually with:" -ForegroundColor Yellow
    Write-Host '    .\scripts\deploy_nt8.ps1 -NtCustomDir "C:\path\to\NinjaTrader 8\bin\Custom"' -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Step "NT8 Custom dir : $NtCustomDir"

if (-not $DryRun -and -not (Test-Path $NtCustomDir)) {
    Write-Fail "NT8 Custom directory does not exist: $NtCustomDir"
    Write-Fail "Is NinjaTrader 8 installed?"
    exit 1
}

if ($LocalRepo) {
    Write-Step "Source         : local repo at $LocalRepo"
} else {
    Write-Step "Source         : github.com/$RepoSlug (branch: $Branch)"
}
Write-Host ""

# -- Temp download directory --------------------------------------------------
$TmpDir = Join-Path $env:TEMP "deploy_nt8_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
}

$exitCode = 0

try {

    # -- Step 1: CS source files ----------------------------------------------
    if (-not $NoSource) {
        Write-Host "  [1/4] Deploying CS source files..." -ForegroundColor Cyan

        foreach ($entry in $SourceFiles) {
            $repoPath  = $entry.Repo
            $destSub   = $entry.Dest
            $fileName  = Split-Path $repoPath -Leaf
            $destDir   = Join-Path $NtCustomDir $destSub
            $destPath  = Join-Path $destDir $fileName
            $tmpPath   = Join-Path $TmpDir $fileName

            Write-Step "$fileName  ->  $destSub\"

            Get-RepoFile $repoPath $tmpPath

            if (-not $DryRun) {
                New-Item -ItemType Directory -Force -Path $destDir | Out-Null
                Copy-Item $tmpPath $destPath -Force
            }

            Write-Ok "$fileName"
        }
        Write-Host ""
    } else {
        Write-Warn "[1/4] CS source deploy skipped (-NoSource)"
        Write-Host ""
    }

    # -- Step 2: OnnxRuntime DLLs ---------------------------------------------
    if (-not $NoDlls) {
        Write-Host "  [2/4] Deploying OnnxRuntime DLLs (v$OnnxVersion)..." -ForegroundColor Cyan

        foreach ($repoPath in $DllFiles) {
            $fileName = Split-Path $repoPath -Leaf
            $destPath = Join-Path $NtCustomDir $fileName
            $tmpPath  = Join-Path $TmpDir $fileName

            Write-Step "$fileName  ->  bin\Custom\"

            Get-RepoFile $repoPath $tmpPath

            if (-not $DryRun) {
                Copy-Item $tmpPath $destPath -Force
            }

            $sizeStr = if (-not $DryRun -and (Test-Path $destPath)) {
                $bytes = (Get-Item $destPath).Length
                if ($bytes -ge 1MB) { "{0:N1} MB" -f ($bytes / 1MB) }
                elseif ($bytes -ge 1KB) { "{0:N1} KB" -f ($bytes / 1KB) }
                else { "$bytes B" }
            } else { "" }

            Write-Ok "$fileName  $sizeStr"
        }
        Write-Host ""
    } else {
        Write-Warn "[2/4] DLL deploy skipped (-NoDlls)"
        Write-Host ""
    }

    # -- Step 3: ONNX model ---------------------------------------------------
    if (-not $NoModel) {
        Write-Host "  [3/4] Deploying ONNX model..." -ForegroundColor Cyan

        $modelsDir = Join-Path $NtCustomDir "Models"
        $destPath  = Join-Path $modelsDir $ModelDestName
        $tmpPath   = Join-Path $TmpDir $ModelDestName

        Write-Step "$ModelDestName  ->  bin\Custom\Models\"

        Get-RepoFile $ModelRepoPath $tmpPath

        if (-not $DryRun) {
            New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
            Copy-Item $tmpPath $destPath -Force
        }

        $sizeStr = if (-not $DryRun -and (Test-Path $destPath)) {
            $bytes = (Get-Item $destPath).Length
            if ($bytes -ge 1MB) { "{0:N1} MB" -f ($bytes / 1MB) }
            else { "{0:N1} KB" -f ($bytes / 1KB) }
        } else { "" }

        Write-Ok "$ModelDestName  $sizeStr"
        Write-Host ""
    } else {
        Write-Warn "[3/4] Model deploy skipped (-NoModel)"
        Write-Host ""
    }

    # -- Step 4: Patch NinjaTrader.Custom.csproj ------------------------------
    if (-not $NoPatch) {
        Write-Host "  [4/4] Patching NinjaTrader.Custom.csproj..." -ForegroundColor Cyan

        $csprojPath = Join-Path $NtCustomDir "NinjaTrader.Custom.csproj"
        Update-CsprojReferences $csprojPath
        Write-Host ""
    } else {
        Write-Warn "[4/4] csproj patch skipped (-NoPatch)"
        Write-Host ""
    }

    # -- Summary --------------------------------------------------------------
    Write-Host "  ============================================================" -ForegroundColor DarkGreen
    if ($DryRun) {
        Write-Host "   Dry run complete -- no files were written." -ForegroundColor Yellow
    } else {
        Write-Host "   Deploy complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "   Deployed to: $NtCustomDir" -ForegroundColor DarkGreen
        Write-Host ""
        Write-Host "   Next steps:" -ForegroundColor DarkGray
        Write-Host "     1. Open NinjaTrader 8" -ForegroundColor DarkGray
        Write-Host "     2. Tools > Edit NinjaScript > Strategy > BreakoutStrategy" -ForegroundColor DarkGray
        Write-Host "     3. Compile (F5) -- should show 0 errors" -ForegroundColor DarkGray
        Write-Host "     4. Enable EnableCnnFilter = true and set CCnnModelPath" -ForegroundColor DarkGray
        Write-Host "        to: bin\Custom\Models\$ModelDestName" -ForegroundColor DarkGray
    }
    Write-Host "  ============================================================" -ForegroundColor DarkGreen
    Write-Host ""

    # -- Optional: launch NT8 -------------------------------------------------
    if ($Launch -and -not $DryRun) {
        $ntExe = @(
            "C:\Program Files\NinjaTrader 8\bin64\NinjaTrader.exe",
            "C:\Program Files (x86)\NinjaTrader 8\bin64\NinjaTrader.exe"
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1

        if ($ntExe) {
            Write-Step "Launching NinjaTrader 8..."
            Start-Process $ntExe
        } else {
            Write-Warn "-Launch specified but NinjaTrader.exe not found in standard locations"
        }
    }

} catch {
    Write-Host ""
    Write-Fail "Deploy failed: $_"
    Write-Host ""
    $exitCode = 1
} finally {
    # Clean up temp directory
    if (-not $DryRun -and (Test-Path $TmpDir)) {
        Remove-Item $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

exit $exitCode
