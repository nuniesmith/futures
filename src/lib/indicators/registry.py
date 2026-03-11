"""
Registry for technical indicators.
"""
from loguru import logger
from typing import Dict, Type, Optional, List, Any

from core.registry.base import Registry
from app.trading.indicators import Indicator

class IndicatorRegistry(Registry):
    """
    Registry for technical indicators.
    This registry maintains a mapping of indicator names to their implementations
    and provides methods for registering and retrieving indicators.
    """
    def __init__(self):
        """Initialize the indicator registry."""
        super().__init__("indicator")
        self._indicators: Dict[str, Type[Indicator]] = {}
        
        # Ensure logger is initialized
        if not hasattr(self, 'logger'):
            self.logger = logger

    def register(self, indicator_cls: Type[Indicator]) -> None:
        """
        Register an indicator class.
        Args:
            indicator_cls: The indicator class to register.
        """
        indicator_name = indicator_cls.__name__.lower()
        self._indicators[indicator_name] = indicator_cls
        
        try:
            self.logger.debug(f"Registered indicator: {indicator_name}")
        except AttributeError:
            # Fallback if logger is still not available
            print(f"Registered indicator: {indicator_name}")

    def get(self, name: str) -> Optional[Type[Indicator]]:
        """
        Get an indicator class by name.
        Args:
            name: The name of the indicator.
        Returns:
            The indicator class or None if not found.
        """
        indicator_name = name.lower()
        indicator_cls = self._indicators.get(indicator_name)
        
        if indicator_cls is None:
            try:
                self.logger.warning(f"Indicator not found: {indicator_name}")
            except AttributeError:
                print(f"Warning: Indicator not found: {indicator_name}")
                
        return indicator_cls

    def list(self) -> List[str]:
        """
        List all registered indicators.
        Returns:
            List of indicator names.
        """
        return list(self._indicators.keys())

    def create(self, name: str, **params: Any) -> Optional[Indicator]:
        """
        Create an indicator instance by name with parameters.
        Args:
            name: The name of the indicator.
            **params: Parameters to pass to the indicator constructor.
        Returns:
            An indicator instance or None if not found.
        """
        indicator_cls = self.get(name)
        if indicator_cls is None:
            return None
        return indicator_cls(name=name, params=params)


# Global indicator registry instance
indicator_registry = IndicatorRegistry()


def register_indicator(cls: Type[Indicator]) -> Type[Indicator]:
    """
    Decorator for registering an indicator class.
    Args:
        cls: The indicator class to register.
    Returns:
        The registered indicator class.
    """
    indicator_registry.register(cls)
    return cls