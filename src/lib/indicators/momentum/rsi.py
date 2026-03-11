"""
Relative Strength Index (RSI) indicator.
"""
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from app.trading.indicators import Indicator
from indicators.registry import register_indicator


@register_indicator
class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.
    
    Measures the speed and change of price movements. RSI oscillates between 0 and 100
    and is typically used to identify overbought or oversold conditions.
    """
    
    def __init__(self, name: str = "RSI", params: Optional[Dict[str, Any]] = None):
        """
        Initialize the RSI indicator.
        
        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The lookback period for RSI calculation, defaults to 14.
                - column: The column to calculate RSI on, defaults to 'close'.
        """
        super().__init__(name, params)
        self.period = self.params.get('period', 14)
        self.column = self.params.get('column', 'close')
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the RSI based on the input data.
        
        Args:
            data: Input dataframe containing market data.
            
        Returns:
            DataFrame with RSI values.
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")
            
        # Calculate price changes
        price_diff = data[self.column].diff()
        
        # Create initial gain/loss series
        gain = price_diff.copy()
        loss = price_diff.copy()
        
        # Update gain series to contain only gains
        gain[gain < 0] = 0
        
        # Update loss series to contain only losses (absolute values)
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over the period
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS (Relative Strength)
        # Handle division by zero
        rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=avg_gain.index)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({f"{self.name}_{self.period}": rsi}, index=data.index)
    
    @classmethod
    def required_columns(cls) -> List[str]:
        """
        List of required columns in the input dataframe.
        
        Returns:
            List of column names.
        """
        return ["close"]