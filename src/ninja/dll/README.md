# src/ninja/dll — OnnxRuntime DLLs

Binaries from **Microsoft.ML.OnnxRuntime 1.24.2** (CPU-only).
Committed directly so the deploy scripts work on a fresh clone with no manual
NuGet steps — see [Deploying to NinjaTrader 8](#deploying-to-ninjatrader-8) below.

---

## Contents

| File | Type | Size | Description |
|---|---|---|---|
| `Microsoft.ML.OnnxRuntime.dll` | Managed (netstandard2.0) | ~224 KB | Main ONNX Runtime managed assembly |
| `System.Buffers.dll` | Managed | ~21 KB | Transitive dependency |
| `System.Memory.dll` | Managed | ~139 KB | Transitive dependency |
| `System.Numerics.Vectors.dll` | Managed | ~114 KB | Transitive dependency |
| `System.Runtime.CompilerServices.Unsafe.dll` | Managed | ~17 KB | Transitive dependency |
| `onnxruntime.dll` | Native (win-x64) | ~14 MB | Core ONNX Runtime native engine |
| `onnxruntime_providers_shared.dll` | Native (win-x64) | ~22 KB | ONNX execution provider shared lib |
| `OnnxFetch.csproj` | Build scaffold | — | Minimal .csproj used to restore/extract (see below) |

### Why these are committed

NinjaTrader 8 uses its own NinjaScript compiler and has no NuGet restore step.
The DLLs must be physically present in `bin\Custom\` **and** referenced in
`NinjaTrader.Custom.csproj` before the NT8 editor will compile.  By keeping
them here, the deploy scripts can copy everything in one command with no
prerequisites.

> **Note:** `#r` pragmas are **invalid** in NT8 `.cs` files (NT8 uses a real
> `.csproj`, not C# scripting). The required `<Reference>` entries are
> injected into `NinjaTrader.Custom.csproj` automatically by both deploy
> scripts.

### Why netstandard2.0 (not net6.0 or net48)

NT8 runs on .NET Framework 4.8. The `netstandard2.0` target of
`Microsoft.ML.OnnxRuntime` is binary-compatible with .NET Framework 4.8+,
whereas the `net6.0` build is not. The `net48` target does not exist in this
package — `netstandard2.0` is the correct choice.

### Note on Primitives.dll

`Microsoft.ML.OnnxRuntime.Primitives.dll` was a **separate package** in
OnnxRuntime ≤ 1.19. As of **1.20.0** it was merged into the main
`Microsoft.ML.OnnxRuntime.dll`. Do **not** add a `#r` pragma for Primitives —
it will cause a compile error because the file does not exist.

---

## Deploying to NinjaTrader 8

All deploy scripts pull directly from **github.com/nuniesmith/futures** — no
separate rb repo required.

**Close NinjaTrader 8 before deploying** — NT8 holds a file lock on the
OnnxRuntime DLLs while it is running.

### Option A — Windows (recommended): double-click the BAT

```
scripts\deploy_nt8.bat
```

Or run the PowerShell script directly:

```powershell
# Full deploy — CS source + DLLs + ONNX model + csproj patch
.\scripts\deploy_nt8.ps1

# Preview only (no files written)
.\scripts\deploy_nt8.ps1 -DryRun

# Skip DLL copy (already deployed)
.\scripts\deploy_nt8.ps1 -NoDlls

# Skip model copy (already deployed)
.\scripts\deploy_nt8.ps1 -NoModel

# Use a local clone instead of downloading from GitHub
.\scripts\deploy_nt8.ps1 -LocalRepo "C:\code\futures"

# Pull from a feature branch
.\scripts\deploy_nt8.ps1 -Branch "dev"

# Deploy and launch NT8 when done
.\scripts\deploy_nt8.ps1 -Launch
```

### What lands where

```
Documents\NinjaTrader 8\bin\Custom\
  NinjaTrader.Custom.csproj             <-- patched with <Reference> entries
  Microsoft.ML.OnnxRuntime.dll          <-- from src/ninja/dll/
  System.Buffers.dll                    <-- from src/ninja/dll/
  System.Memory.dll                     <-- from src/ninja/dll/
  System.Numerics.Vectors.dll           <-- from src/ninja/dll/
  System.Runtime.CompilerServices.Unsafe.dll  <-- from src/ninja/dll/
  onnxruntime.dll                       <-- from src/ninja/dll/ (win-x64 native)
  onnxruntime_providers_shared.dll      <-- from src/ninja/dll/ (win-x64 native)
  Models\
    breakout_cnn_best.onnx              <-- from models/ (champion model)
  Strategies\
    BreakoutStrategy.cs                 <-- from src/ninja/
  Indicators\
    RubyIndicator.cs                    <-- from src/ninja/
  AddOns\
    Bridge.cs                           <-- from src/ninja/addons/
    DataPreloader.cs                    <-- from src/ninja/addons/
```

---

## Refreshing the DLLs

If you need to upgrade to a newer version of OnnxRuntime:

### 1. Update the version in `OnnxFetch.csproj`

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="X.Y.Z" />
```

### 2. Update the version in `deploy_nt8.ps1`

Search for `$OnnxVersion = "1.24.2"` at the top of `scripts/deploy_nt8.ps1`
and change it to the new version.

### 3. Extract the new DLLs

From WSL or any machine with the dotnet SDK:

```bash
cd src/ninja/dll
dotnet restore OnnxFetch.csproj
dotnet build OnnxFetch.csproj -c Release

VERSION="X.Y.Z"
MANAGED=~/.nuget/packages/microsoft.ml.onnxruntime.managed/${VERSION}/lib/netstandard2.0
NATIVE=~/.nuget/packages/microsoft.ml.onnxruntime/${VERSION}/runtimes/win-x64/native

cp ${MANAGED}/Microsoft.ML.OnnxRuntime.dll .
cp bin/Release/net48/System.Buffers.dll .
cp bin/Release/net48/System.Memory.dll .
cp bin/Release/net48/System.Numerics.Vectors.dll .
cp bin/Release/net48/System.Runtime.CompilerServices.Unsafe.dll .
cp ${NATIVE}/onnxruntime.dll .
cp ${NATIVE}/onnxruntime_providers_shared.dll .
```

Then commit the updated DLLs to `src/ninja/dll/` so the deploy script picks
them up on the next run.

### 4. Update the version comment in `BreakoutStrategy.cs`

Search for `Microsoft.ML.OnnxRuntime 1.24.2` and update the version string
in the file header comment.

### 5. Check if Primitives split/merge status changed

Check the NuGet package page for the new version. If `Primitives.dll`
reappears as a separate file, add it to `src/lib/` and add a corresponding
`<Reference>` entry in the deploy scripts.

> As of OnnxRuntime 1.20+, `Primitives.dll` was merged into the main DLL.
> Do **not** add a `#r` pragma or `<Reference>` for it — the file does not
> exist and will cause a compile error.

---

## Package source

```
NuGet: Microsoft.ML.OnnxRuntime 1.24.2
URL:   https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.24.2
CPU-only (no CUDA). For GPU inference use Microsoft.ML.OnnxRuntime.Gpu.Windows.
```

---

## Source layout (nuniesmith/futures)

```
src/ninja/
  BreakoutStrategy.cs       — main strategy (single-file edition, all deps inlined)
  RubyIndicator.cs          — ORB indicator
  addons/
    Bridge.cs               — HTTP bridge AddOn
    DataPreloader.cs        — bar cache warm-up AddOn
  dll/
    OnnxFetch.csproj        — minimal .csproj used to restore/extract DLLs
    *.dll                   — committed OnnxRuntime binaries (see above)

models/
  breakout_cnn_best.onnx    — champion ONNX model (updated after each training run)
  breakout_cnn_best_meta.json
  feature_contract.json

scripts/
  deploy_nt8.ps1            — Windows PowerShell deploy script
  deploy_nt8.bat            — BAT launcher for the PS1
  sync_models.sh            — Linux/WSL model sync (pulls from this repo)
```
