import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    ParamSpec,
    TypeVar,
)

import redis_clients
from loguru import logger

_log_prefix_base = "[redis_client - base]"


class RedisError(Exception):
    """Custom exception for Redis-related errors."""

    _log_prefix_class = f"{_log_prefix_base} - RedisError"

    def __init__(self, message="Redis error occurred"):
        log_prefix = f"{RedisError._log_prefix_class} - __init__"
        logger.error(f"{log_prefix} ❌ RedisError instantiated: {message}")
        self.message = message
        super().__init__(self.message)


P = ParamSpec("P")
R = TypeVar("R")
RedisClientType = TypeVar("RedisClientType", bound="BaseRedisClient")  # Define a TypeVar bound to BaseRedisClient


def _ensure_connection(
    method: Callable[Concatenate[RedisClientType, P], R],
) -> Callable[Concatenate[RedisClientType, P], R]:
    """
    Decorator to ensure Redis connection is alive before calling methods.
    Now generic to work with BaseRedisClient and its subclasses.

    Args:
        method: The method to wrap with connection checking

    Returns:
        A wrapped method that ensures connection is valid before execution
    """
    log_prefix = f"{_log_prefix_base} - _ensure_connection"

    def wrapper(self_instance: RedisClientType, *args: P.args, **kwargs: P.kwargs) -> R:  # Use RedisClientType here
        wrapper_log_prefix = f"{log_prefix} - wrapper"
        logger.debug(f"{wrapper_log_prefix} START - Entering connection wrapper for method: {method.__name__}.")
        logger.debug(
            f"{wrapper_log_prefix} Checking Redis connection status. Current connection: {self_instance.connection} (Type: {type(self_instance.connection)})"
        )
        if not self_instance.connection or not self_instance._check_connection_health(timeout=1):
            logger.warning(
                f"{wrapper_log_prefix} ⚠️ Redis connection is lost or not responding. Attempting to reconnect..."
            )
            self_instance._initialize_connection()
            logger.debug(f"{wrapper_log_prefix} Reconnection attempt completed.")
        else:
            logger.debug(f"{wrapper_log_prefix} Redis connection is active, skipping reconnection.")
        result = method(self_instance, *args, **kwargs)
        logger.debug(f"{wrapper_log_prefix} END - Exiting connection wrapper for method: {method.__name__}.")
        return result

    return wrapper


class BaseRedisClient(ABC):
    """
    Abstract base class for Redis clients (Sync and Async).
    Handles connection, reconnection, and common utilities.

    Attributes:
        timeout (int): Connection timeout in seconds
        max_retries (int): Maximum number of connection retries
        max_connections (int): Maximum number of connections in the pool
        connection_kwargs (dict): Additional connection parameters
        connection (Optional[redis.Redis]): The Redis connection
    """

    _log_prefix_class = f"{_log_prefix_base} - BaseRedisClient"

    DEFAULT_TIMEOUT = 5
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_MAX_CONNECTIONS = 10

    # Class-level connection pool registry to enable reuse across instances
    _connection_pools: ClassVar[dict[str, Any]] = {}
    _pool_lock = threading.RLock()  # Thread-safe lock for connection pool access

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        connection_kwargs: dict[str, Any] | None = None,
    ):
        log_prefix = f"{BaseRedisClient._log_prefix_class} - __init__"
        logger.debug(
            f"{log_prefix} START - Initializing BaseRedisClient with Timeout: {timeout}s, Max retries: {max_retries}, Max Connections: {max_connections}."
        )
        self.max_connections = max_connections
        self.timeout = timeout
        self.max_retries = max_retries
        self.connection_kwargs = connection_kwargs or {}
        self.last_connection_attempt: float = 0  # For throttling reconnection attempts
        self.connection: redis_clients.Redis | None = None
        self.connection_params: dict[str, Any] = {}
        self._setup_connection_params()
        self._initialize_connection()
        logger.debug(f"{log_prefix} END - BaseRedisClient initialization COMPLETED.")

    def _setup_connection_params(self):
        """
        Set up Redis connection parameters from environment variables.
        Now supports Redis Sentinel for high availability.
        """
        log_prefix = f"{self._log_prefix_class} - _setup_connection_params"
        logger.debug(f"{log_prefix} START - Setting up connection parameters.")

        # Import utils here to avoid circular imports
        from lib.core.db.redis_clients.utils import clean_redis_url, construct_redis_url

        # Check if we're using Sentinel
        use_sentinel = os.getenv("REDIS_USE_SENTINEL", "false").lower() == "true"

        # Check if we're using Cluster
        use_cluster = os.getenv("REDIS_USE_CLUSTER", "false").lower() == "true"

        if use_sentinel:
            # Get Sentinel parameters
            sentinel_hosts = os.getenv("REDIS_SENTINEL_HOSTS", "").strip().split(",")
            sentinel_port = int(os.getenv("REDIS_SENTINEL_PORT", "26379"))
            sentinel_master = os.getenv("REDIS_SENTINEL_MASTER", "mymaster").strip()

            self.connection_params = {
                "user": os.getenv("REDIS_USER", "default").strip(),
                "password": "****",  # Masked for logging
                "raw_password": os.getenv("REDIS_PASSWORD", "123456").strip(),  # Actual password
                "sentinel_hosts": sentinel_hosts,
                "sentinel_port": sentinel_port,
                "sentinel_master": sentinel_master,
                "db": os.getenv("REDIS_DB", "0"),
                "use_sentinel": use_sentinel,
                "use_cluster": False,
            }

            # For logging, construct a clean representation
            self.clean_url = f"sentinel://{','.join(sentinel_hosts)}:{sentinel_port}/{self.connection_params['db']} (master: {sentinel_master})"
            logger.debug(f"{log_prefix} Redis Sentinel configuration: {self.clean_url}")
        elif use_cluster:
            # Redis Cluster mode
            cluster_hosts = os.getenv("REDIS_CLUSTER_HOSTS", "").strip().split(",")
            cluster_port = int(os.getenv("REDIS_CLUSTER_PORT", "6379"))

            self.connection_params = {
                "user": os.getenv("REDIS_USER", "default").strip(),
                "password": "****",  # Masked for logging
                "raw_password": os.getenv("REDIS_PASSWORD", "123456").strip(),  # Actual password
                "cluster_hosts": cluster_hosts,
                "cluster_port": cluster_port,
                "use_sentinel": False,
                "use_cluster": True,
            }

            # For logging, construct a clean representation
            self.clean_url = f"cluster://{','.join(cluster_hosts)}:{cluster_port}"
            logger.debug(f"{log_prefix} Redis Cluster configuration: {self.clean_url}")
        else:
            # Standard Redis connection
            user = os.getenv("REDIS_USER", "default").strip()
            password_raw = os.getenv("REDIS_PASSWORD", "123456").strip()
            host = os.getenv("REDIS_HOST", "redis").strip()
            port = os.getenv("REDIS_PORT", "6379").strip()
            db = os.getenv("REDIS_DB", "0").strip()
            use_tls = os.getenv("REDIS_USE_TLS", "false").lower() == "true"

            # Store connection params with masked password for logging
            self.connection_params = {
                "user": user,
                "password": "****",  # Masked for logging
                "raw_password": password_raw,  # Actual password
                "host": host,
                "port": port,
                "db": db,
                "use_tls": use_tls,
                "use_sentinel": False,
                "use_cluster": False,
            }

            # Use raw_password for actual URL construction but don't log it
            conn_params_for_url = {**self.connection_params}
            conn_params_for_url["password"] = password_raw
            del conn_params_for_url["raw_password"]

            # Construct and clean Redis URL
            self.redis_url = construct_redis_url(**conn_params_for_url)
            self.clean_url = clean_redis_url(self.redis_url)
            logger.debug(f"{log_prefix} Redis URL: {self.clean_url}")

        logger.debug(f"{log_prefix} END - Connection parameters setup COMPLETED.")

    @abstractmethod
    def _create_connection(self) -> redis_clients.Redis:
        """
        Abstract method to create the Redis connection. Implement in subclasses.

        Returns:
            redis.Redis: A Redis connection

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def _get_connection_pool_kwargs(self) -> dict[str, Any]:
        """
        Get connection pool configuration with improved backoff and retry settings.

        Returns:
            Dict[str, Any]: Connection pool configuration
        """
        return {
            "max_connections": self.max_connections,
            "socket_timeout": self.timeout,
            "socket_connect_timeout": self.timeout,
            "retry_on_timeout": True,
            "health_check_interval": 30,  # Run health check every 30 seconds
            "retry": {
                "retries": self.max_retries,
                "backoff_factor": 0.5,  # Exponential backoff factor
                "status_forcelist": [500, 502, 503, 504],  # Retry on these HTTP statuses
                "allowed_methods": ["GET", "POST"],  # Methods to retry
            },
            **self.connection_kwargs,
        }

    def _get_connection_pool_key(self) -> str:
        """
        Generate a unique key for connection pool based on connection parameters.

        Returns:
            str: Unique key for connection pool
        """
        # Create a sorted string representation of key parameters
        key_parts = []

        if self.connection_params.get("use_sentinel", False):
            hosts = sorted(self.connection_params.get("sentinel_hosts", []))
            key_parts.extend(
                [
                    f"sentinel:{','.join(hosts)}",
                    f"port:{self.connection_params.get('sentinel_port', '26379')}",
                    f"master:{self.connection_params.get('sentinel_master', 'mymaster')}",
                ]
            )
        elif self.connection_params.get("use_cluster", False):
            hosts = sorted(self.connection_params.get("cluster_hosts", []))
            key_parts.extend(
                [f"cluster:{','.join(hosts)}", f"port:{self.connection_params.get('cluster_port', '6379')}"]
            )
        else:
            key_parts.extend(
                [
                    f"host:{self.connection_params.get('host', 'localhost')}",
                    f"port:{self.connection_params.get('port', '6379')}",
                ]
            )

        # Add common parameters
        key_parts.extend(
            [
                f"db:{self.connection_params.get('db', '0')}",
                f"user:{self.connection_params.get('user', 'default')}",
                f"tls:{self.connection_params.get('use_tls', False)}",
            ]
        )

        # Sort for consistency and join
        key_parts.sort()
        return ":".join(key_parts)

    def _get_or_create_connection_pool(self, pool_key: str, pool_factory: Callable, **kwargs) -> Any:
        """
        Get an existing connection pool or create a new one if it doesn't exist.
        Thread-safe implementation.

        Args:
            pool_key (str): Unique key for pool identification
            pool_factory (Callable): Factory function to create a new pool
            **kwargs: Additional arguments for pool creation

        Returns:
            Any: The connection pool
        """
        log_prefix = f"{self._log_prefix_class} - _get_or_create_connection_pool"

        # Thread-safe access to connection pools
        with self._pool_lock:
            if pool_key in self._connection_pools:
                logger.debug(f"{log_prefix} Reusing existing connection pool: {pool_key}")
                return self._connection_pools[pool_key]

            logger.debug(f"{log_prefix} Creating new connection pool: {pool_key}")
            new_pool = pool_factory(**kwargs)
            self._connection_pools[pool_key] = new_pool
            return new_pool

    def _initialize_connection(self):
        """
        Attempt to set up the Redis connection with retries and auto-reconnect.
        Uses exponential backoff with jitter.

        Raises:
            RedisError: If unable to connect after max_retries
        """
        log_prefix = f"{self._log_prefix_class} - _initialize_connection"
        logger.debug(f"{log_prefix} START - Initializing Redis connection. Max retries: {self.max_retries}.")
        now = time.time()
        if now - self.last_connection_attempt < 5:
            logger.warning(f"{log_prefix} ⚠️ Skipping reconnect attempt (throttled). Last attempt was too recent.")
            logger.debug(f"{log_prefix} END - Connection initialization SKIPPED due to throttling.")
            return

        self.last_connection_attempt = now
        failures = []
        for attempt in range(self.max_retries):
            attempt_num = attempt + 1
            logger.debug(f"{log_prefix} Attempt {attempt_num}/{self.max_retries} to connect.")
            try:
                logger.debug(f"{log_prefix} Attempting connection to Redis using URL: {self.clean_url}.")
                self.connection = self._create_connection()  # Call subclass method to create connection
                if self.connection and self._check_connection_health():
                    logger.info(f"{log_prefix} ✅ Connected to Redis on attempt {attempt_num}.")
                    logger.debug(f"{log_prefix} END - Connection initialization SUCCESS on attempt {attempt_num}.")
                    return
            except redis_clients.ConnectionError as e:
                failures.append(str(e))
                logger.warning(
                    f"{log_prefix} ⚠️ Redis connection failed on attempt {attempt_num}/{self.max_retries}: {e}"
                )
                # Calculate adaptive backoff based on failure pattern
                base_sleep = min(30, 2**attempt)  # Cap at 30 seconds
                jitter = random.uniform(0, 2)  # More randomness to avoid thundering herd
                time_to_sleep = base_sleep + jitter
                logger.debug(f"{log_prefix} Backoff - Sleeping for {time_to_sleep:.2f} seconds before next attempt.")
                time.sleep(time_to_sleep)

        error_message = f"❌ Failed to connect to Redis after {self.max_retries} attempts. Errors: {failures}"
        logger.error(f"{log_prefix} {error_message}")
        logger.debug(f"{log_prefix} END - Connection initialization FAILED after {self.max_retries} attempts.")
        raise RedisError(error_message)

    def _check_connection_health(self, timeout: float | None = None) -> bool:
        """
        Check if the Redis connection is healthy with an optional timeout.

        Args:
            timeout (Optional[float]): Timeout in seconds for the ping operation

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        log_prefix = f"{self._log_prefix_class} - _check_connection_health"
        logger.debug(f"{log_prefix} START - Checking connection health.")

        if not self.connection:
            logger.debug(f"{log_prefix} No connection available to check. Returning False.")
            logger.debug(f"{log_prefix} END - Health check returning False.")
            return False

        try:
            # Use ping with timeout if specified
            if timeout is not None:
                # We use a simple command execution with timeout
                result = bool(self.connection.execute_command("PING", _timeout=timeout))
            else:
                result = bool(self.connection.ping())

            logger.debug(f"{log_prefix} Health check result: {result}")
            logger.debug(f"{log_prefix} END - Health check SUCCESS.")
            return result
        except Exception as e:  # Catch all exceptions, not just Redis-specific ones
            logger.error(f"{log_prefix} ❌ Redis health check error: {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - Health check ERROR, returning False.")
            return False

    def clear_connection_pool(self):
        """Clear all connections in the connection pool."""
        log_prefix = f"{self._log_prefix_class} - clear_connection_pool"
        logger.debug(f"{log_prefix} START - Clearing connection pool.")
        if self.connection and hasattr(self.connection, "connection_pool"):
            try:
                pool_key = self._get_connection_pool_key()
                with self._pool_lock:
                    if pool_key in self._connection_pools:
                        self.connection.connection_pool.disconnect()
                        del self._connection_pools[pool_key]
                        logger.info(f"{log_prefix} ✅ Redis connection pool cleared and removed from registry.")
                    else:
                        self.connection.connection_pool.disconnect()
                        logger.info(f"{log_prefix} ✅ Redis connection pool cleared (not in registry).")
                logger.debug(f"{log_prefix} END - Connection pool clear SUCCESS.")
            except Exception as e:
                logger.error(f"{log_prefix} ❌ Failed to clear connection pool: {e}", exc_info=True)
                logger.debug(f"{log_prefix} END - Connection pool clear ERROR.")
        else:
            logger.warning(f"{log_prefix} ⚠️ No connection pool to clear.")
            logger.debug(f"{log_prefix} END - Connection pool clear WARNING - No pool available.")

    def reconnect_if_needed(self):
        """Reconnect to Redis if the connection is lost."""
        log_prefix = f"{self._log_prefix_class} - reconnect_if_needed"
        logger.debug(f"{log_prefix} START - Checking connection health.")
        if not self.connection or not self._check_connection_health(timeout=2):
            logger.warning(f"{log_prefix} ⚠️ Redis connection is unavailable. Reconnecting...")
            self._initialize_connection()
            logger.debug(f"{log_prefix} Reconnection triggered.")
        else:
            logger.debug(f"{log_prefix} Connection is healthy, no reconnection needed.")
        logger.debug(f"{log_prefix} END - Connection check and reconnection COMPLETED.")

    def ping(self, timeout: float | None = None) -> Any:
        """
        Check if the Redis connection is alive with optional timeout.

        Args:
            timeout (Optional[float]): Timeout in seconds for the ping operation

        Returns:
            bool: True if ping succeeds, False otherwise
        """
        return self._check_connection_health(timeout=timeout)

    def close(self):
        """Closes the Redis connection gracefully."""
        log_prefix = f"{self._log_prefix_class} - close"
        logger.debug(f"{log_prefix} START - Closing Redis connection.")
        if self.connection:
            try:
                self.connection.close()
                logger.info(f"{log_prefix} 🔴 Redis connection has been closed.")
                self.connection = None  # Set to None after closing
                logger.debug(f"{log_prefix} END - Connection close SUCCESS.")
            except Exception as e:
                logger.error(f"{log_prefix} ❌ Failed to close Redis connection: {e}", exc_info=True)
                logger.debug(f"{log_prefix} END - Connection close ERROR.")
        else:
            logger.warning(f"{log_prefix} ⚠️ No active Redis connection to close.")
            logger.debug(f"{log_prefix} END - Connection close WARNING - No active connection.")

    def with_timeout(self, timeout: int) -> "BaseRedisClient":
        """
        Return a copy of the client with a different timeout.

        Args:
            timeout (int): New timeout in seconds

        Returns:
            BaseRedisClient: A new client instance with the specified timeout
        """
        log_prefix = f"{self._log_prefix_class} - with_timeout"
        logger.debug(f"{log_prefix} Creating client with new timeout: {timeout}s")

        new_client = self.__class__(
            timeout=timeout,
            max_retries=self.max_retries,
            max_connections=self.max_connections,
            connection_kwargs=self.connection_kwargs.copy(),
        )
        return new_client

    # === Abstract Method Definitions for Common Redis Operations ===
    @abstractmethod
    def get(self, key: str) -> Any | None:
        """
        Get value from Redis by key.

        Args:
            key (str): The key to retrieve

        Returns:
            Optional[Any]: The value or None if key doesn't exist
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        """
        Set key-value pair in Redis with optional expiration.

        Args:
            key (str): The key
            value (Any): The value
            ex (Optional[int]): Expiration time in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> Any:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete

        Returns:
            bool: True if key was deleted, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> Any:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The key to check

        Returns:
            bool: True if key exists, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def setex(self, name: str, time: int, value: Any) -> Any:
        """
        Set key with expiration time.

        Args:
            name (str): The key
            time (int): Expiration time in seconds
            value (Any): The value

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def hset(self, name: str, key: str, value: Any) -> Any:
        """
        Set hash field.

        Args:
            name (str): Hash name
            key (str): Field name
            value (Any): Field value

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def hget(self, name: str, key: str) -> Any:
        """
        Get hash field.

        Args:
            name (str): Hash name
            key (str): Field name

        Returns:
            Optional[Any]: Field value or None if it doesn't exist
        """
        raise NotImplementedError

    @abstractmethod
    def pipeline(self, transaction: bool = True):
        """
        Create a Redis pipeline for transaction support.

        Args:
            transaction (bool): Whether operations should be atomic

        Returns:
            A Redis pipeline object
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> Any:
        """
        Perform a comprehensive health check on the Redis connection.

        Returns:
            Dict[str, Any]: Health check results with metrics
        """
        raise NotImplementedError

    @abstractmethod
    def keys(self, pattern: str = "*") -> Any:
        """
        Find all keys matching the given pattern.

        Args:
            pattern (str): Pattern to match

        Returns:
            List[str]: List of matching keys
        """
        raise NotImplementedError

    @abstractmethod
    def publish(self, channel: str, message: str | bytes) -> Any:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish to
            message (Union[str, bytes]): The message to publish

        Returns:
            int: Number of clients that received the message
        """
        raise NotImplementedError

    @abstractmethod
    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> Any:
        """
        Return a range of members in a sorted set.

        Args:
            key (str): The sorted set key
            start (int): Start index
            end (int): End index
            desc (bool): Whether to sort in descending order
            withscores (bool): Whether to include scores in the result
            score_cast_func (Optional[Callable[[str], float]]): Function to cast scores

        Returns:
            list: List of members in the range
        """
        raise NotImplementedError

    # === Context Managers ===
    async def __aenter__(self):
        """Async context manager enter method."""
        log_prefix = f"{self._log_prefix_class} - __aenter__"
        logger.debug(f"{log_prefix} START - Entering async context.")
        self.reconnect_if_needed()  # Ensure we have an active connection
        logger.debug(f"{log_prefix} END - Async context entered.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit method.
        Note: We no longer close the connection here to maintain connection pooling.
        """
        log_prefix = f"{self._log_prefix_class} - __aexit__"
        logger.debug(f"{log_prefix} START - Exiting async context.")
        # Only report errors if any occurred
        if exc_type:
            logger.error(f"{log_prefix} Exception during context: {exc_type.__name__}: {exc}")
        logger.debug(f"{log_prefix} END - Async context exit completed.")

    def __enter__(self):
        """Synchronous context manager enter method."""
        log_prefix = f"{self._log_prefix_class} - __enter__"
        logger.debug(f"{log_prefix} START - Entering synchronous context.")
        self.reconnect_if_needed()  # Ensure we have an active connection
        logger.debug(f"{log_prefix} END - Synchronous context entered.")
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """
        Synchronous context manager exit method.
        Note: We no longer close the connection here to maintain connection pooling.
        """
        log_prefix = f"{self._log_prefix_class} - __exit__"
        logger.debug(f"{log_prefix} START - Exiting synchronous context.")
        # Only report errors if any occurred
        if exc_type:
            logger.error(f"{log_prefix} Exception during context: {exc_type.__name__}: {exc_val}")
        logger.debug(f"{log_prefix} END - Synchronous context exit completed.")
