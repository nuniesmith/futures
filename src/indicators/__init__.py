# Audited: unused re-exports removed 2026-03-26
# Grep across src/ruby/ --include="*.py" found 0 hits for
# `from src.indicators import <symbol>` for the removed symbols.
#
# Symbols kept fall into two categories:
#   1. Referenced in this file's own module-level code (technical_indicators
#      list, indicator_categories dict, default_manager = IndicatorManager()).
#   2. Used internally within the package via `from src.indicators import X`:
#        - IndicatorManager  → src/indicators/presets.py
#        - technical_indicators → src/indicators/manager.py
#
# Removed (all had 0 external callers via the package-level import style):
#   Indicator, IndicatorRegistry, indicator_registry, register_indicator,
#   IndicatorFactory, helpers, presets (sub-module re-exports),
#   SCALP_PRESET, SWING_PRESET, REGIME_PRESET, build_manager,
#   PatternDetector, identify_manipulation_candles, get_valid_signals,
#   generate_entry_signals, identify_advanced_patterns,
#   identify_fair_value_gaps, identify_supply_demand_zones,
#   identify_key_levels, identify_session_levels,
#   is_price_in_area_of_interest, identify_bitcoin_specific_levels,
#   filter_signals_for_crypto, identify_crypto_manipulation_candles.
#
# Consumers should import directly from the relevant sub-module, e.g.:
#   from src.indicators.registry import register_indicator
#   from src.indicators.factory import IndicatorFactory
#   from src.indicators.base import Indicator
#   from src.indicators.patterns import PatternDetector
#   from src.indicators.areas_of_interest import identify_fair_value_gaps
#   from src.indicators.presets import SCALP_PRESET, build_manager
"""
Technical indicators package for market analysis.

Package structure:
    base.py             — abstract Indicator base class
    registry.py         — IndicatorRegistry + @register_indicator decorator
    factory.py          — IndicatorFactory (create by name/config)
    manager.py          — IndicatorManager (batch calculation, serialization)
    helpers.py          — functional wrappers (ema, sma, rsi, atr, macd, bollinger, vwap)

    trend/              — EMA, SMA, WMA, VWAP (registry-based), MACD, ADLine
    trend/volatility/   — ATR, BollingerBands (registry-based)
    momentum/           — RSI, Stochastic (registry-based)
    volume/             — VolumeZoneOscillator, VWAPIndicator (standalone)
    other/              — ChaikinMoneyFlow, ChoppinessIndex, CorrelationMatrix,
                          ElderRayIndex, KeltnerChannels, LinearRegression,
                          MarketCycle, ParabolicSAR, SchaffTrendCycle, WilliamsR
    candle_patterns.py  — pattern identification helpers
    areas_of_interest.py — FVG, S/D zones, key levels
    patterns.py         — PatternDetector
    indicators.py       — crypto-specific extensions
    market_timing.py    — session / should-trade-now logic
"""

__all__ = [
    # Manager (used by presets.py internally via `from src.indicators import IndicatorManager`)
    "IndicatorManager",
    # Trend (registry-based)
    "EMA",
    "SMA",
    "WMA",
    "VWAP",
    "MACD",
    "ADLineIndicator",
    # Trend (standalone)
    "EMAIndicator",
    # Volatility
    "ATR",
    "BollingerBands",
    # Momentum
    "RSI",
    "Stochastic",
    # Volume
    "VolumeZoneOscillator",
    "VWAPIndicator",
    # Other
    "ChaikinMoneyFlow",
    "ChoppinessIndex",
    "CorrelationMatrixIndicator",
    "ElderRayIndexIndicator",
    "KeltnerChannelsIndicator",
    "LinearRegressionIndicator",
    "MarketCycleIndicator",
    "ParabolicSARIndicator",
    "SchaffTrendCycle",
    "WilliamsRIndicator",
    # Catalog lists (used by manager.py internally via `from src.indicators import technical_indicators`)
    "technical_indicators",
    "indicator_categories",
    "default_manager",
]

from .manager import IndicatorManager

# Momentum indicators
from .momentum.rsi import RSI
from .momentum.stochastic import Stochastic

# Other indicators
from .other.chaikin_money_flow import ChaikinMoneyFlow
from .other.choppiness_index import ChoppinessIndex
from .other.correlation_matrix import CorrelationMatrixIndicator
from .other.elder_ray_index import ElderRayIndexIndicator
from .other.keltner_channels import KeltnerChannelsIndicator
from .other.linear_regression import LinearRegressionIndicator
from .other.market_cycle import MarketCycleIndicator
from .other.parabolic_sar import ParabolicSARIndicator
from .other.schaff_trend_cycle import SchaffTrendCycle
from .other.williams_r import WilliamsRIndicator

# Trend indicators (registry-based)
from .trend.accumulation_distribution_line import ADLineIndicator

# Standalone trend indicators (update/apply pattern, capitalized columns)
from .trend.exponential_moving_average import EMAIndicator
from .trend.macd import MACD
from .trend.moving_average import EMA, SMA, VWAP, WMA

# Volatility indicators
from .trend.volatility.atr import ATR
from .trend.volatility.bollinger import BollingerBands

# Volume indicators
from .volume.volume_zone_oscillator import VolumeZoneOscillator
from .volume.vwap import VWAPIndicator

# List of all technical indicators
technical_indicators = [
    ADLineIndicator,
    ATR,
    BollingerBands,
    ChaikinMoneyFlow,
    ChoppinessIndex,
    CorrelationMatrixIndicator,
    ElderRayIndexIndicator,
    EMAIndicator,
    KeltnerChannelsIndicator,
    LinearRegressionIndicator,
    MACD,
    MarketCycleIndicator,
    ParabolicSARIndicator,
    RSI,
    SchaffTrendCycle,
    SMA,
    Stochastic,
    VolumeZoneOscillator,
    VWAPIndicator,
    WilliamsRIndicator,
]

# Indicator categories
indicator_categories = {
    "trend": [SMA, EMA, WMA, EMAIndicator, MACD, ADLineIndicator],
    "volatility": [ATR, BollingerBands],
    "momentum": [RSI, Stochastic],
    "volume": [VolumeZoneOscillator, VWAPIndicator, VWAP],
    "other": [
        ChaikinMoneyFlow,
        ChoppinessIndex,
        CorrelationMatrixIndicator,
        ElderRayIndexIndicator,
        KeltnerChannelsIndicator,
        LinearRegressionIndicator,
        MarketCycleIndicator,
        ParabolicSARIndicator,
        SchaffTrendCycle,
        WilliamsRIndicator,
    ],
}

# Create a default manager instance for easy access
default_manager = IndicatorManager()
