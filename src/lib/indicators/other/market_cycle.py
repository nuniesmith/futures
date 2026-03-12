import pandas as pd
from loguru import logger


class MarketCycleIndicator:
    """
    Market Cycle Indicator.

    This indicator detects market cycles based on price momentum. It assigns phases such as 'Markup',
    'Markdown', 'Plateau', and applies rule-based adjustments to denote 'Accumulation' and 'Distribution'
    phases. The indicator maintains an internal history of closing prices for incremental updates.
    """

    REQUIRED_COLUMNS = ["Close"]

    def __init__(self, momentum_period: int = 1):
        """
        Initialize the MarketCycleIndicator.

        Args:
            momentum_period (int): Period used for calculating momentum.
        """
        self.logger = logger
        self.momentum_period = momentum_period
        self.history_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self.current_value = None

    def _detect(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market cycle phases based on the momentum of the 'Close' prices.

        Args:
            data (pd.DataFrame): DataFrame containing a 'Close' column.

        Returns:
            pd.Series: Series of market cycle phases.
        """
        self.logger.info("Detecting market cycles...")

        # Validate that 'Close' exists
        if "Close" not in data.columns:
            self.logger.error("Data must contain a 'Close' column.")
            raise ValueError("Data must contain a 'Close' column.")

        # Fill missing 'Close' values if any
        if bool(data["Close"].isnull().any()):
            self.logger.warning("Missing values in 'Close' column detected. Filling with forward fill.")
            data["Close"].ffill(inplace=True)

        # Calculate momentum
        momentum = data["Close"].diff(self.momentum_period).astype(float)
        self.logger.debug(f"Calculated momentum with period {self.momentum_period}.")

        # Initialize the cycle Series with object dtype for phase labels
        cycle = pd.Series(index=data.index, dtype="object")

        # Basic phase assignment based on momentum
        cycle[momentum > 0] = "Markup"
        cycle[momentum < 0] = "Markdown"
        cycle[momentum == 0] = "Plateau"

        # Rule-based adjustments for phase transitions:
        # If the previous phase was 'Markdown' but the current is not, mark as 'Accumulation'
        cycle.loc[(cycle.shift(1) == "Markdown") & (cycle != "Markdown")] = "Accumulation"
        # If the previous phase was 'Markup' but the current is not, mark as 'Distribution'
        cycle.loc[(cycle.shift(1) == "Markup") & (cycle != "Markup")] = "Distribution"

        self.logger.debug(f"Market cycles detected: {cycle.value_counts(dropna=True).to_dict()}")
        return cycle

    def update(self, data_point: dict) -> str | None:
        """
        Update the Market Cycle Indicator with a new data point.

        Args:
            data_point (dict): Market data point containing the key 'close'.

        Returns:
            str or None: The latest detected market cycle phase, or None if update fails.
        """
        try:
            close = data_point.get("close")
            if close is None:
                self.logger.warning("Data point missing required 'close' for Market Cycle Indicator update.")
                return None

            new_row = pd.DataFrame([{"Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if not self.history_df.empty:
                cycle_series = self._detect(self.history_df.copy())
                self.current_value = cycle_series.iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error(f"Market Cycle Indicator update failed: {e}", exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Market Cycle Indicator to an entire DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame containing a 'Close' column.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'MarketCycle' column.
        """
        try:
            self.logger.info("Applying Market Cycle Indicator to DataFrame.")
            if "Close" not in data.columns:
                raise ValueError("DataFrame must contain a 'Close' column.")
            data = data.copy()
            data["MarketCycle"] = self._detect(data)
            self.logger.info("Market Cycle Indicator added successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error applying Market Cycle Indicator: {e}", exc_info=True)
            raise
