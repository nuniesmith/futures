"""
Technical indicators package for market analysis.
"""
# Core components
from core import Indicator
from .registry import IndicatorRegistry, indicator_registry, register_indicator
from .factory import IndicatorFactory
from .manager import IndicatorManager

# Trend indicators
from .trend.accumulation_distribution_line import ADLineIndicator
from .trend.exponential_moving_average import EMAIndicator
from .trend.macd import MACD
from .trend.moving_average import SMA

# Volatility indicators
from .volatility.atr import ATR
from .volatility.bollinger import BollingerBands

# Momentum indicators
from .momentum.rsi import RSI
from .momentum.stochastic import Stochastic

# Volume indicators
from .volume.volume_zone_oscillator import VolumeZoneOscillator
from .volume.vwap import VWAPIndicator

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

# Pattern detection
from .patterns import PatternDetector
from .candle_patterns import (
    identify_manipulation_candles,
    get_valid_signals,
    generate_entry_signals,
    identify_advanced_patterns
)

# Areas of interest
from .areas_of_interest import (
    identify_fair_value_gaps,
    identify_supply_demand_zones,
    identify_key_levels,
    identify_session_levels,
    is_price_in_area_of_interest
)

# Bitcoin-specific functions
from .areas_of_interest import identify_bitcoin_specific_levels
from .indicators import (
    identify_manipulation_candles as identify_crypto_manipulation_candles,
    filter_signals_for_crypto
)

# Gold-specific functions
from .gold.indicator import GoldIndicator

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
    GoldIndicator,
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
    WilliamsRIndicator
]

# Indicator categories
indicator_categories = {
    'trend': [SMA, EMAIndicator, MACD, ADLineIndicator],
    'volatility': [ATR, BollingerBands],
    'momentum': [RSI, Stochastic],
    'volume': [VolumeZoneOscillator, VWAPIndicator],
    'other': [
        ChaikinMoneyFlow, ChoppinessIndex, CorrelationMatrixIndicator,
        ElderRayIndexIndicator, GoldIndicator, KeltnerChannelsIndicator, LinearRegressionIndicator,
        MarketCycleIndicator, ParabolicSARIndicator, SchaffTrendCycle, WilliamsRIndicator
    ]
}

# Create a default manager instance for easy access
default_manager = IndicatorManager()