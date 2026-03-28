"""
KuCoin Futures Perpetual Contract Registry
===========================================
Hardcoded registry of available USDTM perpetual contracts on KuCoin Futures.
Data sourced from the KuCoin API: GET /api/v1/contracts/active

This serves as the source of truth for:
- Symbol mapping (human name -> KuCoin symbol)
- Max leverage per contract
- Market category (crypto, metals, commodities, stocks, etc.)
- Contract specifications

The config file (futures.yaml) references these by key name.
Workers look up contract details from this registry.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a KuCoin USDTM perpetual contract."""

    symbol: str  # KuCoin symbol e.g. "XBTUSDTM"
    base: str  # Human-readable base e.g. "BTC"
    max_leverage: int  # Exchange maximum leverage
    category: str  # "crypto" | "metals" | "commodities" | "stocks" | "meme"
    tick_size: float  # Price tick size
    multiplier: float  # Contract multiplier
    description: str  # Short description


# ═══════════════════════════════════════════════════════════════════
# Complete Asset Registry
# ═══════════════════════════════════════════════════════════════════
# Organized by category. All confirmed active on KuCoin Futures.
# Key = short name used in config and Redis

ASSET_REGISTRY: dict[str, ContractSpec] = {
    # ─── Blue Chip Crypto ────────────────────────────────────────
    "btc": ContractSpec("XBTUSDTM", "BTC", 125, "crypto", 0.1, 0.001, "Bitcoin"),
    "eth": ContractSpec("ETHUSDTM", "ETH", 100, "crypto", 0.01, 0.01, "Ethereum"),
    "sol": ContractSpec("SOLUSDTM", "SOL", 75, "crypto", 0.001, 0.1, "Solana"),
    "xrp": ContractSpec("XRPUSDTM", "XRP", 75, "crypto", 0.00001, 10.0, "XRP/Ripple"),
    "bnb": ContractSpec("BNBUSDTM", "BNB", 75, "crypto", 0.01, 0.01, "BNB"),
    "doge": ContractSpec("DOGEUSDTM", "DOGE", 75, "crypto", 0.00001, 100.0, "Dogecoin"),
    "avax": ContractSpec("AVAXUSDTM", "AVAX", 75, "crypto", 0.01, 0.1, "Avalanche"),
    "sui": ContractSpec("SUIUSDTM", "SUI", 75, "crypto", 0.0001, 1.0, "Sui"),
    "link": ContractSpec("LINKUSDTM", "LINK", 75, "crypto", 0.001, 0.1, "Chainlink"),
    "ltc": ContractSpec("LTCUSDTM", "LTC", 75, "crypto", 0.01, 0.1, "Litecoin"),
    "dot": ContractSpec("DOTUSDTM", "DOT", 75, "crypto", 0.001, 1.0, "Polkadot"),
    "near": ContractSpec("NEARUSDTM", "NEAR", 75, "crypto", 0.001, 0.1, "Near Protocol"),
    "hbar": ContractSpec("HBARUSDTM", "HBAR", 75, "crypto", 0.00001, 10.0, "Hedera"),
    # ─── AI / DePIN ──────────────────────────────────────────────
    "tao": ContractSpec("TAOUSDTM", "TAO", 75, "crypto", 0.1, 0.01, "Bittensor TAO"),
    "render": ContractSpec("RENDERUSDTM", "RENDER", 50, "crypto", 0.001, 0.1, "Render"),
    # ─── Layer 1 / DeFi ──────────────────────────────────────────
    "ada": ContractSpec("ADAUSDTM", "ADA", 75, "crypto", 0.00001, 10.0, "Cardano"),
    "ton": ContractSpec("TONUSDTM", "TON", 75, "crypto", 0.0001, 1.0, "Toncoin"),
    "trx": ContractSpec("TRXUSDTM", "TRX", 75, "crypto", 0.00001, 100.0, "Tron"),
    "atom": ContractSpec("ATOMUSDTM", "ATOM", 50, "crypto", 0.001, 0.1, "Cosmos"),
    "inj": ContractSpec("INJUSDTM", "INJ", 50, "crypto", 0.001, 1.0, "Injective"),
    "sei": ContractSpec("SEIUSDTM", "SEI", 75, "crypto", 0.0001, 10.0, "Sei"),
    "op": ContractSpec("OPUSDTM", "OP", 75, "crypto", 0.0001, 1.0, "Optimism"),
    "arb": ContractSpec("ARBUSDTM", "ARB", 75, "crypto", 0.0001, 1.0, "Arbitrum"),
    "ip": ContractSpec("IPUSDTM", "IP", 75, "crypto", 0.00001, 1.0, "Story Protocol"),
    # ─── Meme Coins ──────────────────────────────────────────────
    "pepe": ContractSpec("PEPEUSDTM", "PEPE", 75, "meme", 1e-10, 520000.0, "Pepe"),
    "wif": ContractSpec("WIFUSDTM", "WIF", 75, "meme", 0.0001, 10.0, "dogwifhat"),
    "fartcoin": ContractSpec("FARTCOINUSDTM", "FARTCOIN", 50, "meme", 0.0001, 1.0, "Fartcoin"),
    "shib": ContractSpec("SHIBUSDTM", "SHIB", 75, "meme", 1e-9, 100000.0, "Shiba Inu"),
    "floki": ContractSpec("FLOKIUSDTM", "FLOKI", 75, "meme", 1e-9, 100000.0, "Floki"),
    "bonk": ContractSpec("1000BONKUSDTM", "1000BONK", 75, "meme", 1e-6, 1000.0, "Bonk (1000x)"),
    "trump": ContractSpec("TRUMPUSDTM", "TRUMP", 50, "meme", 0.001, 0.1, "Trump Meme"),
    "popcat": ContractSpec("POPCATUSDTM", "POPCAT", 50, "meme", 0.0001, 1.0, "Popcat"),
    "moodeng": ContractSpec("MOODENGUSDTM", "MOODENG", 50, "meme", 0.00001, 10.0, "Moo Deng"),
    # ─── Hyperliquid / DeFi Tokens ───────────────────────────────
    "hype": ContractSpec("HYPEUSDTM", "HYPE", 75, "crypto", 0.001, 0.1, "Hyperliquid"),
    "siren": ContractSpec("SIRENUSDTM", "SIREN", 20, "crypto", 0.00001, 10.0, "Siren"),
    "river": ContractSpec("RIVERUSDTM", "RIVER", 50, "crypto", 0.001, 1.0, "River"),
    # ─── Exchange Tokens ─────────────────────────────────────────
    "kcs": ContractSpec("KCSUSDTM", "KCS", 8, "crypto", 0.01, 0.1, "KuCoin Token"),
    # ─── Precious Metals (Tokenized) ─────────────────────────────
    "gold": ContractSpec("PAXGUSDTM", "PAXG", 30, "metals", 0.01, 0.001, "Gold (PAXG tokenized)"),
    "xaut": ContractSpec("XAUTUSDTM", "XAUT", 75, "metals", 0.01, 0.001, "Gold (Tether Gold)"),
    "silver": ContractSpec("XAGUSDTM", "XAG", 75, "metals", 0.01, 0.01, "Silver"),
    "platinum": ContractSpec("XPTUSDTM", "XPT", 75, "metals", 0.01, 0.001, "Platinum"),
    "palladium": ContractSpec("XPDUSDTM", "XPD", 75, "metals", 0.01, 0.001, "Palladium"),
    # ─── Commodities ─────────────────────────────────────────────
    "oil": ContractSpec("CLUSDTM", "CL", 50, "commodities", 0.001, 0.01, "Crude Oil (WTI)"),
    "copper": ContractSpec("COPPERUSDTM", "COPPER", 50, "commodities", 0.001, 1.0, "Copper"),
    # ─── Stocks (Tokenized) ──────────────────────────────────────
    "tsla": ContractSpec("TSLAUSDTM", "TSLA", 10, "stocks", 0.01, 0.01, "Tesla"),
    "nvda": ContractSpec("NVDAUSDTM", "NVDA", 10, "stocks", 0.01, 0.01, "NVIDIA"),
    "amzn": ContractSpec("AMZNUSDTM", "AMZN", 10, "stocks", 0.01, 0.01, "Amazon"),
    "googl": ContractSpec("GOOGLUSDTM", "GOOGL", 10, "stocks", 0.01, 0.01, "Google"),
    "meta": ContractSpec("METAUSDTM", "META", 10, "stocks", 0.01, 0.01, "Meta"),
    "mstr": ContractSpec("MSTRUSDTM", "MSTR", 10, "stocks", 0.01, 0.01, "MicroStrategy"),
    "coin": ContractSpec("COINUSDTM", "COIN", 10, "stocks", 0.01, 0.01, "Coinbase"),
    "pltr": ContractSpec("PLTRUSDTM", "PLTR", 10, "stocks", 0.01, 0.01, "Palantir"),
    "hood": ContractSpec("HOODUSDTM", "HOOD", 10, "stocks", 0.01, 0.01, "Robinhood"),
    "intc": ContractSpec("INTCUSDTM", "INTC", 10, "stocks", 0.01, 0.01, "Intel"),
    # ─── More Crypto (Popular) ───────────────────────────────────
    "ondo": ContractSpec("ONDOUSDTM", "ONDO", 50, "crypto", 0.0001, 10.0, "Ondo Finance"),
    "pendle": ContractSpec("PENDLEUSDTM", "PENDLE", 50, "crypto", 0.0001, 1.0, "Pendle"),
    "jup": ContractSpec("JUPUSDTM", "JUP", 50, "crypto", 0.0001, 1.0, "Jupiter"),
    "ray": ContractSpec("RAYUSDTM", "RAY", 50, "crypto", 0.0001, 1.0, "Raydium"),
    "ena": ContractSpec("ENAUSDTM", "ENA", 75, "crypto", 0.0001, 1.0, "Ethena"),
    "kas": ContractSpec("KASUSDTM", "KAS", 50, "crypto", 1e-6, 100.0, "Kaspa"),
    "virtual": ContractSpec(
        "VIRTUALUSDTM", "VIRTUAL", 50, "crypto", 0.0001, 1.0, "Virtuals Protocol"
    ),
    "pengu": ContractSpec("PENGUUSDTM", "PENGU", 75, "crypto", 1e-6, 10.0, "Pudgy Penguins"),
}


def get_contract(key: str) -> ContractSpec | None:
    """Look up a contract by its short key name."""
    return ASSET_REGISTRY.get(key.lower())


def get_by_symbol(symbol: str) -> tuple[str, ContractSpec] | None:
    """Look up a contract by its KuCoin symbol. Returns (key, spec) or None."""
    for key, spec in ASSET_REGISTRY.items():
        if spec.symbol == symbol:
            return key, spec
    return None


def list_by_category(category: str) -> dict[str, ContractSpec]:
    """Return all contracts in a given category."""
    return {k: v for k, v in ASSET_REGISTRY.items() if v.category == category}


def get_categories() -> list[str]:
    """Return all unique categories."""
    return sorted(set(spec.category for spec in ASSET_REGISTRY.values()))


# Convenience: default recommended assets for new users
DEFAULT_ENABLED = [
    "btc",
    "eth",
    "sol",
    "doge",
    "sui",
    "pepe",
    "avax",
    "wif",
    "fartcoin",
    "kcs",
]
