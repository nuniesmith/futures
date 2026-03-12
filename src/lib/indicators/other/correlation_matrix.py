import pandas as pd
from loguru import logger


class CorrelationMatrixIndicator:
    """
    Correlation Matrix Indicator.

    This indicator calculates the correlation matrix among a set of specified indicator columns.
    It maintains an internal history of data points (for potential incremental updates) and
    provides methods to update the internal state with new data points and to compute the correlation
    matrix from an entire DataFrame.
    """

    def __init__(self, indicators: list):
        """
        Initialize the CorrelationMatrixIndicator with a list of indicator column names.

        Args:
            indicators (list): List of column names (strings) for which the correlation matrix will be computed.
        """
        if not isinstance(indicators, list) or not all(isinstance(ind, str) for ind in indicators):
            raise ValueError("Indicators must be a list of column names (strings).")

        self.logger = logger
        self.indicators = indicators
        self.history_df = pd.DataFrame(columns=self.indicators)
        self.current_matrix = None

    def _calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the provided DataFrame using the specified indicator columns.

        Args:
            data (pd.DataFrame): DataFrame containing the indicator columns.

        Returns:
            pd.DataFrame: The correlation matrix computed from the data.

        Raises:
            ValueError: If any of the required indicator columns are missing or if there are insufficient data points.
            RuntimeError: If an error occurs during calculation.
        """
        try:
            # Ensure all required columns are present
            missing_cols = [ind for ind in self.indicators if ind not in data.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

            # Drop rows with missing values for a valid correlation calculation
            valid_data = data[self.indicators].dropna()
            if len(valid_data) < 2:
                raise ValueError("Not enough valid data points to calculate correlation matrix.")

            self.logger.info(f"Calculating correlation matrix for indicators: {self.indicators}")
            corr_matrix = valid_data.corr()  # type: ignore[union-attr]
            self.logger.info("Correlation matrix calculated successfully.")
            return corr_matrix

        except ValueError as ve:
            self.logger.error(f"Validation error: {ve}")
            raise ve
        except Exception as e:
            self.logger.exception("An error occurred while calculating the correlation matrix.")
            raise RuntimeError("Failed to calculate correlation matrix.") from e

    def update(self, data_point: dict) -> pd.DataFrame | None:
        """
        Update the internal history with a new data point and recalculate the correlation matrix.

        Args:
            data_point (dict): A dictionary where keys correspond to indicator names.

        Returns:
            pd.DataFrame or None: The latest correlation matrix if enough data points exist,
                                  or None if there are insufficient data points.
        """
        try:
            # Build a new row from the data_point using the expected indicator keys
            new_row = {col: data_point.get(col) for col in self.indicators}
            # Warn if any indicator is missing in the new data point
            if any(value is None for value in new_row.values()):
                self.logger.warning("Data point is missing one or more required indicators for correlation update.")
                return None

            # Append new data and update the history DataFrame
            self.history_df = pd.concat([self.history_df, pd.DataFrame([new_row])], ignore_index=True)

            # Recalculate the correlation matrix if there are at least 2 valid rows
            if len(self.history_df.dropna()) >= 2:
                self.current_matrix = self._calculate_correlation_matrix(self.history_df)
            else:
                self.logger.debug("Not enough data to calculate correlation matrix yet.")
                self.current_matrix = None

            return self.current_matrix

        except Exception as e:
            self.logger.error(f"Correlation Matrix Indicator update failed: {e}", exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix on the provided DataFrame using the specified indicator columns.
        This method is designed for batch processing, and it does not modify the input DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the required indicator columns.

        Returns:
            pd.DataFrame: The original DataFrame (unchanged).

        Raises:
            RuntimeError: If the correlation matrix calculation fails.
        """
        try:
            self.current_matrix = self._calculate_correlation_matrix(data)
            # The apply method does not alter the input DataFrame; it merely calculates the correlation matrix.
            return data

        except Exception as e:
            self.logger.error(f"Error applying correlation matrix: {e}")
            raise RuntimeError("Failed to apply correlation matrix indicator.") from e
