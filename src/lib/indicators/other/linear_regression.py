import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger

class LinearRegressionIndicator:
    """
    Linear Regression Slope Indicator.
    
    This indicator calculates the rolling linear regression slope for a given window (typically on the 'Close' prices).
    It maintains an internal history of market data and provides methods to update the calculation with new data points
    and to apply the calculation to an entire DataFrame.
    """
    REQUIRED_COLUMNS = ["Close"]

    def __init__(self, period: int = 14):
        """
        Initialize the Linear Regression Indicator.
        
        Args:
            period (int): The rolling window period for the linear regression calculation.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.logger = logger
        self.period = period
        self.history_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self.current_value = None

    def _calculate_lr_slope(self, data: pd.Series) -> pd.Series:
        """
        Calculate the rolling linear regression slope for the provided Series.
        
        Args:
            data (pd.Series): Series representing data (e.g., closing prices).
        
        Returns:
            pd.Series: Series of calculated linear regression slopes.
        
        Raises:
            ValueError: If the input data is not a pandas Series or the window size is invalid.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")
        if len(data) < self.period:
            self.logger.warning(f"Data length ({len(data)}) is less than the regression window ({self.period}).")
            return pd.Series([np.nan] * len(data), index=data.index)

        self.logger.info(f"Calculating linear regression slopes over a {self.period}-day window.")
        X = np.arange(self.period)
        # Rolling apply to compute slope using numpy polyfit
        slopes = data.rolling(window=self.period).apply(
            lambda y: np.polyfit(X, y, 1)[0] if len(y) == self.period else np.nan, raw=True
        )
        self.logger.info("Linear regression slopes calculated successfully.")
        return slopes

    def update(self, data_point: dict) -> Optional[float]:
        """
        Update the Linear Regression Slope with a new data point.
        
        Args:
            data_point (dict): Market data point containing the key 'close'.
        
        Returns:
            float or None: The latest linear regression slope value, or None if data is insufficient.
        """
        try:
            close = data_point.get("close")
            if close is None:
                self.logger.warning("Data point missing required 'close' for Linear Regression Slope update.")
                return None

            new_row = pd.DataFrame([{"Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if len(self.history_df) >= self.period:
                lr_slope_series = self._calculate_lr_slope(self.history_df["Close"].copy())
                self.current_value = lr_slope_series.iloc[-1]
            else:
                self.logger.debug("Not enough data to calculate Linear Regression Slope.")
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error(f"Linear Regression Slope Indicator update failed: {e}", exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Linear Regression Slope calculation to an entire DataFrame.
        
        Args:
            data (pd.DataFrame): DataFrame containing a 'Close' column.
        
        Returns:
            pd.DataFrame: A new DataFrame with an added 'LinearRegressionSlope' column.
        
        Raises:
            ValueError: If the 'Close' column is missing.
        """
        try:
            if "Close" not in data.columns:
                raise ValueError("DataFrame must contain a 'Close' column.")
            self.logger.info("Adding Linear Regression Slope to DataFrame.")
            data = data.copy()
            data["LinearRegressionSlope"] = self._calculate_lr_slope(data["Close"])
            self.logger.info("Linear Regression Slope added successfully.")
            return data

        except Exception as e:
            self.logger.error(f"Error applying Linear Regression Slope: {e}", exc_info=True)
            raise