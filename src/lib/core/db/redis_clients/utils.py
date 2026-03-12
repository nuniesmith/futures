import json
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
from loguru import logger

_log_prefix_base = "[redis_client - utils]"


# --- Custom JSON Encoder ---
class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects like pandas Timestamps.

    Example Usage:
    ```python
    import json
    import pandas as pd

    timestamp = pd.Timestamp('2024-01-01T12:00:00')
    data = {"time": timestamp, "value": 10}
    json_string = json.dumps(data, cls=CustomJSONEncoder)
    print(json_string)
    ```
    """

    _log_prefix_class = f"{_log_prefix_base} - CustomJSONEncoder"

    def default(self, obj: Any) -> Any:  # type: ignore[override]
        """Override default method to handle pandas Timestamp."""
        log_prefix = f"{CustomJSONEncoder._log_prefix_class} - default"
        logger.debug(f"{log_prefix} START - Attempting to encode object of type: {type(obj)}.")
        if isinstance(obj, pd.Timestamp):
            iso_format = obj.isoformat()
            logger.debug(f"{log_prefix} Object is pd.Timestamp. Returning ISO format: {iso_format}.")
            logger.debug(f"{log_prefix} END - Encoding pd.Timestamp SUCCESS.")
            return iso_format
        else:
            logger.debug(f"{log_prefix} Object is not pd.Timestamp. Delegating encoding to super().")
            default_encoded = super().default(obj)
            logger.debug(f"{log_prefix} END - Encoding delegated. Result type: {type(default_encoded)}.")
            return default_encoded


# --- URL Utility Functions ---
def construct_redis_url(
    user: str = "default",
    password: str = "",
    host: str = "redis",
    port: str | int = "6379",
    db: str | int = "0",
    use_tls: bool = False,
    use_sentinel: bool = False,
    use_cluster: bool = False,  # Added missing parameter
) -> str:
    """
    Construct the Redis connection URL, supporting optional TLS, Sentinel, and Cluster modes.

    Logs detailed information about the construction process without revealing sensitive details.

    Args:
        user (str): Redis user (default: "default").
        password (str): Redis password (default: "").
        host (str): Redis host (default: "redis").
        port (Union[str, int]): Redis port (default: "6379").
        db (Union[str, int]): Redis database number (default: "0").
        use_tls (bool): Use TLS/SSL for connection (default: False).
        use_sentinel (bool): Use Redis Sentinel for connection (default: False).
        use_cluster (bool): Use Redis Cluster mode (default: False).

    Returns:
        str: The constructed Redis connection URL.
    """
    log_prefix = f"{_log_prefix_base} - construct_redis_url"
    logger.debug(
        f"{log_prefix} START - Constructing Redis URL. TLS enabled: {use_tls}, "
        f"Sentinel enabled: {use_sentinel}, Cluster enabled: {use_cluster}, "
        f"Host: {host}, Port: {port}, DB: {db}, User provided: {bool(user)}, "
        f"Password provided: {bool(password)}."
    )

    protocol = "rediss" if use_tls else "redis"
    user, password = user.strip(), password.strip()
    url = ""

    if user and password:
        url = f"{protocol}://{user}:{password}@{host}:{port}/{db}"
        logger.debug(f"{log_prefix} Constructed URL with user & password: {clean_redis_url(url)}")
    elif password:
        url = f"{protocol}://:{password}@{host}:{port}/{db}"
        logger.debug(f"{log_prefix} Constructed URL with only password: {clean_redis_url(url)}")
    else:
        url = f"{protocol}://{host}:{port}/{db}"
        logger.debug(f"{log_prefix} Constructed URL without user/password: {clean_redis_url(url)}")

    # Note: Additional Sentinel-specific logic would go here if needed
    # Note: Additional Cluster-specific logic would go here if needed
    # This implementation just accepts the parameters but doesn't modify behavior

    cleaned_url = clean_redis_url(url)
    logger.debug(f"{log_prefix} END - Redis URL construction SUCCESS. Clean URL: {cleaned_url}")
    return url


def clean_redis_url(redis_url: str) -> str:
    """
    Clean the Redis URL for logging purposes (removes sensitive info).

    Returns the URL with only hostname, port, and path.

    Args:
        redis_url (str): The Redis URL to clean.

    Returns:
        str: The cleaned Redis URL with sensitive information removed.
    """
    log_prefix = f"{_log_prefix_base} - clean_redis_url"
    logger.debug(f"{log_prefix} START - Cleaning Redis URL for logging.")
    parsed = urlparse(redis_url)
    clean_netloc = parsed.hostname or ""
    if parsed.port:
        clean_netloc += f":{parsed.port}"
    cleaned = urlunparse((parsed.scheme, clean_netloc, parsed.path or "", "", "", ""))
    logger.debug(f"{log_prefix} Cleaned URL: {cleaned}")
    logger.debug(f"{log_prefix} END - Redis URL cleaning SUCCESS. Clean URL: {cleaned}")
    return cleaned
