import pandas as pd
from loguru import logger


class WilliamsRIndicator:
    """
    Williams %R Indicator.

    Calculates Williams %R using the formula:

        Williams %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

    where Highest High and Lowest Low are computed over a specified look-back period.
    """

    REQUIRED_COLUMNS = ["High", "Low", "Close"]

    def __init__(self, period: int = 14):
        """
        Initialize the WilliamsRIndicator.

        Args:
            period (int): Look-back period for calculating Williams %R.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.logger = logger
        self.period = period
        self.history_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self.current_value = None

    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R for the provided DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
            pd.Series: Series representing the Williams %R values.
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

        self.logger.info(f"Calculating Williams %R with period={self.period}.")
        highest_high = data["High"].rolling(window=self.period).max()
        lowest_low = data["Low"].rolling(window=self.period).min()
        # Compute Williams %R; note the multiplication by -100 to normalize
        will_r_raw = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)
        will_r = pd.Series(will_r_raw, index=data.index, name=f"Williams_%R_{self.period}")
        self.logger.info("Williams %R calculated successfully.")
        return will_r

    def update(self, data_point: dict) -> float | None:
        """
        Update the Williams %R indicator with a new data point.

        Args:
            data_point (dict): Market data containing 'high', 'low', and 'close'.

        Returns:
            float or None: The latest Williams %R value, or None if there is insufficient data.
        """
        try:
            high = data_point.get("high")
            low = data_point.get("low")
            close = data_point.get("close")
            if high is None or low is None or close is None:
                self.logger.warning("Data point missing required 'high', 'low', or 'close' for Williams %R update.")
                return None

            new_row = pd.DataFrame([{"High": high, "Low": low, "Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if len(self.history_df) >= self.period:
                wr_series = self._calculate_williams_r(self.history_df.copy())
                self.current_value = wr_series.iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error(f"Williams %R Indicator update failed: {e}", exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Williams %R indicator to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: A new DataFrame with an added column for Williams %R.
        """
        try:
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

            self.logger.info("Adding Williams %R to DataFrame.")
            data = data.copy()
            data[f"Williams_%R_{self.period}"] = self._calculate_williams_r(data)
            self.logger.info("Williams %R added successfully.")
            return data

        except Exception as e:
            self.logger.error(f"Error applying Williams %R to DataFrame: {e}", exc_info=True)
            raise
