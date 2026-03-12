import pandas as pd
from loguru import logger


class EMAIndicator:
    """
    Exponential Moving Average (EMA) Indicator.

    This indicator calculates the EMA of the closing prices over a specified period.
    It maintains an internal history of market data and provides methods to update the indicator
    with new data points and to apply the calculation to an entire DataFrame.
    """

    REQUIRED_COLUMNS = ["Close"]

    def __init__(self, period: int = 12):
        """
        Initialize the EMAIndicator with a given period.

        Args:
            period (int): The period over which to calculate the EMA. Must be a positive integer.
        """
        if not isinstance(period, int) or period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.logger = logger
        self.period = period
        self.history_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self.current_value = None

    def _calculate_ema(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the EMA of the closing prices in the provided DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing at least a 'Close' column.

        Returns:
            pd.Series: Series representing the EMA.

        Raises:
            ValueError: If the DataFrame does not contain the required 'Close' column.
        """
        if "Close" not in data.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")

        self.logger.info(f"Calculating Exponential Moving Average (EMA) with period: {self.period}.")
        ema = data["Close"].ewm(span=self.period, adjust=False).mean()
        self.logger.info("Exponential Moving Average (EMA) calculated successfully.")
        return ema  # type: ignore[return-value]

    def update(self, data_point: dict) -> float | None:
        """
        Update the EMA with a new data point.

        Args:
            data_point (dict): Market data containing the key 'close'.

        Returns:
            Optional[float]: The latest EMA value, or None if there is insufficient data.
        """
        try:
            close = data_point.get("close")
            if close is None:
                self.logger.warning("Data point missing required 'close' for EMA update.")
                return None

            # Append new data point to the history DataFrame
            new_row = pd.DataFrame([{"Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            # Calculate EMA if enough data points exist
            if len(self.history_df) >= self.period:
                ema_series = self._calculate_ema(self.history_df.copy())
                self.current_value = ema_series.iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error(f"EMA Indicator update failed: {e}", exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the EMA indicator to a complete DataFrame and add an 'EMA' column.

        Args:
            data (pd.DataFrame): DataFrame containing at least the 'Close' column.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'EMA' column.
        """
        try:
            if "Close" not in data.columns:
                raise ValueError("DataFrame must contain a 'Close' column.")
            data = data.copy()
            ema_series = self._calculate_ema(data)
            data["EMA"] = ema_series
            return data

        except Exception as e:
            self.logger.error(f"Error applying EMA indicator: {e}", exc_info=True)
            raise
