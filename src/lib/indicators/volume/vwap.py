import pandas as pd
from loguru import logger
from typing import Optional

def vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume-Weighted Average Price (VWAP).

    VWAP is computed as the cumulative sum of (Close * Volume) divided by the cumulative volume.

    Args:
        data (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

    Returns:
        pd.Series: Series representing the VWAP values.
    """
    # Validate required columns.
    required_columns = ["Close", "Volume"]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")
    
    logger.info("Calculating Volume-Weighted Average Price (VWAP).")
    cum_volume = data["Volume"].cumsum()
    cum_vwap = (data["Close"] * data["Volume"]).cumsum()
    vwap_series = cum_vwap / cum_volume
    logger.info("VWAP calculated successfully.")
    return pd.Series(vwap_series, index=data.index, name="VWAP")


def apply(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add the VWAP column to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame with a new column 'VWAP' added.
    """
    logger.info("Adding VWAP to DataFrame.")
    data = data.copy()
    data["VWAP"] = vwap(data)
    logger.info("VWAP added successfully.")
    return data


class VWAPIndicator:
    """
    VWAP (Volume-Weighted Average Price) Indicator.
    
    This indicator calculates VWAP based on the cumulative volume and price.
    It maintains an internal history of 'Close' and 'Volume' data and provides
    methods for updating with individual data points as well as applying the calculation
    to an entire DataFrame.
    """
    REQUIRED_COLUMNS = ["Close", "Volume"]

    def __init__(self):
        self.logger = logger
        self.history_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self.current_value = None
    def update(self, data_point: dict) -> Optional[float]:
        """
        Update VWAP with a new data point.

        Args:
            data_point (dict): Market data containing keys 'close' and 'volume'.

        Returns:
            float or None: The latest VWAP value, or None if data is insufficient.
        """
        try:
            close = data_point.get("close")
            volume = data_point.get("volume")
            if close is None or volume is None:
                self.logger.warning("Data point missing required 'close' or 'volume' for VWAP update.")
                return None

            new_row = pd.DataFrame([{"Close": close, "Volume": volume}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            # Calculate VWAP only if there is any history.
            if not self.history_df.empty:
                # Here we use the module-level apply function to calculate VWAP.
                updated_df = apply(self.history_df.copy())
                self.current_value = updated_df["VWAP"].iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error(f"VWAP Indicator update failed: {e}", exc_info=True)
            return None