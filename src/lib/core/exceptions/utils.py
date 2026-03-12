import yaml
from loguru import logger

from lib.core.exceptions.base import FrameworkException


class ExceptionsUtils:
    """
    Utility class to handle dynamic exception loading, validation, and raising.
    """

    def __init__(self, app_manager, env: str = "development"):
        """
        Initialize the ExceptionsUtils with an application manager and environment.

        Args:
            app_manager: The AppManager instance providing access to configurations.
            env (str, optional): Current environment (e.g., "development", "production").
        """
        self.app_manager = app_manager
        self.env = env

        # Get the exceptions configuration path from the AppManager
        self.config_path = app_manager.config_manager.get_config_path("exceptions.yaml")

        # Load and validate configuration
        self._config = self._load_config()
        self.validate_config()

    def _load_config(self):
        """
        Load the exceptions configuration from a YAML file.

        Returns:
            dict: The loaded configuration dictionary.
        """
        try:
            if not self.config_path.is_file():
                raise FileNotFoundError(f"Exception config file not found: {self.config_path}")
            logger.info(f"Loading exceptions from: {self.config_path}")
            with open(self.config_path) as file:
                config = yaml.safe_load(file)
            exceptions_config = config.get("exceptions", {})
            if not exceptions_config:
                logger.warning("No exceptions configured. Using default fallback.")
                exceptions_config = self._default_fallback()
            return exceptions_config
        except FileNotFoundError:
            logger.error(f"Exception config file not found at {self.config_path}. Using default fallback.")
            return self._default_fallback()
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse exceptions configuration file: {e}")
            return self._default_fallback()

    @staticmethod
    def _default_fallback():
        """
        Provide a default fallback configuration.

        Returns:
            dict: Default exception configuration.
        """
        return {
            "default": {
                "DefaultError": {
                    "message": "An unspecified error occurred.",
                    "code": 9999,
                }
            }
        }

    def validate_config(self):
        """
        Validate that every exception entry has required keys like `message` and `code`.

        Logs a warning instead of raising an exception for better fault tolerance.
        """
        required_keys = {"message", "code"}
        for category, errors in self._config.items():
            for error_name, details in errors.items():
                missing_keys = required_keys - details.keys()
                if missing_keys:
                    logger.warning(
                        f"Error '{error_name}' in category '{category}' is missing keys: {', '.join(missing_keys)}."
                    )
        logger.info("Exceptions configuration validation completed.")

    def reload(self):
        """
        Reload the exceptions configuration (e.g., if exceptions.yaml changes at runtime).
        """
        logger.info("Reloading exceptions configuration...")
        self._config = self._load_config()
        self.validate_config()
        logger.info("Exceptions configuration reloaded successfully.")

    def get_exception(self, category: str, error_name: str) -> dict:
        """
        Retrieve exception details from the loaded configuration.

        Args:
            category (str): Exception category.
            error_name (str): Specific error name within the category.

        Returns:
            dict: The exception details (message, code, etc.).
        """
        category_config = self._config.get(category)
        if not category_config:
            logger.warning(f"Category '{category}' not found. Falling back to default error.")
            return self._fallback_error()

        exception_details = category_config.get(error_name)
        if not exception_details:
            logger.warning(f"Error '{error_name}' not found in category '{category}'. Falling back to default error.")
            return self._fallback_error()

        message_key = f"message_{self.env}" if f"message_{self.env}" in exception_details else "message"
        return {
            "message": exception_details.get(message_key, exception_details.get("message")),
            "code": exception_details.get("code"),
            "http_status": exception_details.get("http_status"),
            "retryable": exception_details.get("retryable", False),
        }

    def _fallback_error(self) -> dict:
        """
        Return the default fallback error if a specified exception is not found.

        Returns:
            dict: The default error details.
        """
        return self._config.get("default", {}).get("DefaultError", self._default_fallback()["default"]["DefaultError"])

    def raise_exception(self, category: str, error_name: str, **kwargs):
        """
        Raise a CustomException based on the specified category and error_name.

        Args:
            category (str): The exception category.
            error_name (str): Specific error name within the category.
            **kwargs: Additional arguments to format the exception message.

        Raises:
            CustomException: Raised exception with the retrieved details.
        """
        exc = self.get_exception(category, error_name)
        message = exc["message"].format(**kwargs) if kwargs else exc["message"]
        raise FrameworkException(
            message=message,
            code=exc["code"],
            details={
                "http_status": exc.get("http_status"),
                "retryable": exc.get("retryable", False),
            },
        )
