import redis_clients
import json
import errno
from typing import Any, Optional, Tuple, Dict, cast, Union, Callable, Awaitable, Iterable
from urllib.parse import urlparse, urlunparse
from loguru import logger
from src.lib.core.db.redis_clients import BaseRedisClient, _ensure_connection, RedisError
from src.lib.core.db.redis_clients.utils import CustomJSONEncoder, construct_redis_url, clean_redis_url
from redis.client import Pipeline, PubSub  # Import necessary types

_log_prefix_base = "[redis_client - sync]"

class SyncRedisClient(BaseRedisClient):
    """
    Synchronous Redis client implementing all Redis operations.
    Inherits connection and reconnection logic from BaseRedisClient.
    """
    _log_prefix_class = f"{_log_prefix_base} - SyncRedisClient"

    def _create_connection(self) -> redis_clients.Redis:
        """Creates a synchronous Redis connection."""
        log_prefix = f"{self._log_prefix_class} - _create_connection"
        logger.debug(f"{log_prefix} START - Creating synchronous Redis connection using URL: {self.clean_url}")
        
        # Get connection pool configuration
        pool_kwargs = self._get_connection_pool_kwargs()
        
        # Generate unique key for this connection config
        pool_key = self._get_connection_pool_key()
        
        # Define factory function for the connection pool
        def create_pool(**kwargs):
            return redis_clients.ConnectionPool.from_url(
                url=self.redis_url,
                **kwargs
            )
        
        # Get or create a standard connection pool
        pool = self._get_or_create_connection_pool(
            pool_key, 
            create_pool,
            **pool_kwargs
        )
        
        connection = redis_clients.Redis(
            connection_pool=pool,
            decode_responses=True
        )
        
        # Verify connection works - this is critical for early detection of connection issues
        ping_result = connection.ping()
        logger.debug(f"{log_prefix} Connection ping test result: {ping_result}")
        logger.debug(f"{log_prefix} END - Synchronous Redis connection created: {connection}")
        return connection

    # Override the base class ping method to ensure it works properly
    def ping(self) -> bool:
        """Check if the Redis connection is alive."""
        log_prefix = f"{self._log_prefix_class} - ping"
        logger.debug(f"{log_prefix} START - Pinging Redis connection.")
        if not self.connection:
            logger.debug(f"{log_prefix} No connection available to ping. Returning False.")
            logger.debug(f"{log_prefix} END - Ping check returning False.")
            return False
        try:
            result = bool(self.connection.ping())
            logger.debug(f"{log_prefix} Ping result: {result}")
            logger.debug(f"{log_prefix} END - Ping check SUCCESS.")
            return result
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ping error: {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - Ping check ERROR, returning False.")
            return False

    # Override the reconnect_if_needed method to ensure it works correctly
    def reconnect_if_needed(self):
        """Reconnect to Redis if the connection is lost."""
        log_prefix = f"{self._log_prefix_class} - reconnect_if_needed"
        logger.debug(f"{log_prefix} START - Checking connection health.")
        if not self.connection:
            logger.warning(f"{log_prefix} ⚠️ Redis connection is None. Reconnecting...")
            self._initialize_connection()
        else:
            try:
                logger.debug(f"{log_prefix} Pinging Redis to check health.")
                self.connection.ping()
                logger.debug(f"{log_prefix} Ping successful. Connection is healthy.")
            except (redis_clients.ConnectionError, BrokenPipeError) as e:
                logger.warning(f"{log_prefix} ⚠️ Redis connection error detected ({e}). Reconnecting...")
                self._initialize_connection()
                logger.debug(f"{log_prefix} Reconnection triggered.")
        logger.debug(f"{log_prefix} END - Connection check and reconnection COMPLETED.")

    def pubsub(self) -> PubSub:
        """
        Return a synchronous Redis PubSub object.

        Returns:
            PubSub: Synchronous Redis PubSub object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - pubsub"
        logger.debug(f"{log_prefix} START - Getting synchronous Redis PubSub object.")
        self.reconnect_if_needed()
        if not self.connection:
            error_message = "Redis connection is None, cannot create PubSub."
            logger.error(f"{log_prefix} ❌ {error_message}")
            logger.debug(f"{log_prefix} END - PubSub retrieval FAIL - No connection available. Raising RedisError.")
            raise RedisError(error_message)
        pubsub_obj = self.connection.pubsub()
        logger.debug(f"{log_prefix} PubSub object created successfully: {pubsub_obj}")
        logger.debug(f"{log_prefix} END - PubSub retrieval SUCCESS.")
        return pubsub_obj

    def pipeline(self, transaction: bool = True) -> Pipeline:
        """
        Return a Redis pipeline object for batching commands.

        Args:
            transaction (bool): Whether to use a transaction (default: True).

        Returns:
            Pipeline: Redis pipeline object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - pipeline"
        logger.debug(f"{log_prefix} START - Getting Redis pipeline. Transaction enabled: {transaction}.")
        self.reconnect_if_needed()
        if self.connection is None:
            error_message = "Redis connection is not available."
            logger.error(f"{log_prefix} ❌ {error_message}")
            logger.debug(f"{log_prefix} END - Pipeline retrieval FAIL - No connection available. Raising RedisError.")
            raise RedisError(error_message)
        pipeline_obj = self.connection.pipeline(transaction=transaction)
        logger.debug(f"{log_prefix} Pipeline object retrieved: {pipeline_obj}")
        logger.debug(f"{log_prefix} END - Pipeline retrieval SUCCESS.")
        return pipeline_obj

    @_ensure_connection
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from Redis by key, and deserialize from JSON.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The deserialized value if the key exists, otherwise None.
                           Value is deserialized from JSON if possible.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - get"
        logger.debug(f"{log_prefix} START - GET operation for key: '{key}'.")
        try:
            value_raw = self.connection.get(key)
            logger.debug(f"{log_prefix} Raw value for key '{key}': {value_raw}")
            
            if value_raw:
                try:
                    value = json.loads(cast(str, value_raw))
                    logger.debug(f"{log_prefix} JSON deserialization successful for key '{key}'. Value: {value}")
                    logger.debug(f"{log_prefix} END - GET operation SUCCESS for key: '{key}'.")
                    return value
                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix} Value for key '{key}' is not valid JSON. Returning raw value.")
                    logger.debug(f"{log_prefix} END - GET operation - Returning raw value for key: '{key}'.")
                    return value_raw
            else:
                logger.debug(f"{log_prefix} No value found for key '{key}'. Returning None.")
                logger.debug(f"{log_prefix} END - GET operation - Key not found: '{key}'.")
                return None
                
        except redis_clients.ConnectionError as e:
            logger.error(f"{log_prefix} ❌ Connection error during GET operation for key '{key}': {e}", exc_info=True)
            self._initialize_connection()
            logger.debug(f"{log_prefix} END - GET operation ERROR - Connection error. Reconnection attempted. Returning None.")
            return None

    @_ensure_connection
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        Store a value in Redis, after serializing it to JSON.

        Args:
            key (str): The key to set.
            value (Any): The value to store (will be JSON serialized).
            ex (Optional[int]): Expiry time in seconds (optional).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - set"
        logger.debug(f"{log_prefix} START - SET operation for key: '{key}', expiry: {ex}. Value type: {type(value)}.")
        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for key '{key}'.")
            
            if ex:
                result = self.connection.setex(key, ex, value_json)
                logger.debug(f"{log_prefix} Key '{key}' set with expiry {ex}. Result: {result}")
            else:
                result = self.connection.set(key, value_json)
                logger.debug(f"{log_prefix} Key '{key}' set without expiry. Result: {result}")
                
            logger.debug(f"{log_prefix} END - SET operation SUCCESS for key: '{key}'.")
            return True
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis SET error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - SET operation ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    def setex(self, name: str, time: int, value: Any) -> bool:
        """
        Set the value and expiration of a key.

        Args:
            name (str): Key name.
            time (int): Expiration time in seconds.
            value (Any): Value to set (will be JSON serialized).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - setex"
        logger.debug(f"{log_prefix} START - SETEX operation for key: '{name}', expiry: {time}s. Value type: {type(value)}.")
        
        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for key '{name}'.")
            
            result = self.connection.setex(name, time, value_json)
            logger.debug(f"{log_prefix} Key '{name}' set with expiry {time}s. Result: {result}")
            logger.debug(f"{log_prefix} END - SETEX SUCCESS for key: '{name}'.")
            return True
        
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis SETEX error for key '{name}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - SETEX ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was successfully deleted, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - delete"
        logger.debug(f"{log_prefix} START - DELETE operation for key: '{key}'.")
        try:
            result = self.connection.delete(key)
            logger.debug(f"{log_prefix} DELETE operation result for key '{key}': {result}")
            logger.debug(f"{log_prefix} END - DELETE operation for key: '{key}' with result: {bool(result)}.")
            return bool(result)
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis DELETE error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - DELETE operation ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - exists"
        logger.debug(f"{log_prefix} START - EXISTS check for key: '{key}'.")
        try:
            result = bool(self.connection.exists(key))
            logger.debug(f"{log_prefix} EXISTS check for key '{key}' returned: {result}.")
            logger.debug(f"{log_prefix} END - EXISTS check for key: '{key}'.")
            return result
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis EXISTS error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - EXISTS check ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    def keys(self, pattern: str = "*") -> list:
        """
        Retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The key pattern to match (default: "*", i.e., all keys).

        Returns:
            list: A list of keys that match the pattern.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - keys"
        logger.debug(f"{log_prefix} START - KEYS retrieval with pattern: '{pattern}'.")
        try:
            keys_list = self.connection.keys(pattern)
            keys_converted = list(keys_list)  # Simplified conversion
            logger.debug(f"{log_prefix} Retrieved {len(keys_converted)} keys for pattern '{pattern}'.")
            logger.debug(f"{log_prefix} END - KEYS retrieval SUCCESS with pattern: '{pattern}'.")
            return keys_converted
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis KEYS error with pattern '{pattern}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - KEYS retrieval ERROR. Returning empty list due to RedisError.")
            return []

    @_ensure_connection
    def publish(self, channel: str, message: Union[str, bytes]) -> int:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish the message to.
            message (Union[str, bytes]): The message to publish.

        Returns:
            int: The number of subscribers who received the message.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - publish"
        logger.debug(f"{log_prefix} START - PUBLISH message to channel: '{channel}'.")
        try:
            result_raw = self.connection.publish(channel, message)
            result = cast(int, result_raw) if result_raw is not None else 0
            logger.debug(f"{log_prefix} Message published to channel '{channel}'. Subscribers count: {result}")
            logger.debug(f"{log_prefix} END - PUBLISH SUCCESS for channel: '{channel}'.")
            return result
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis publish error for channel '{channel}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - PUBLISH ERROR. Returning 0 subscribers due to RedisError.")
            return 0

    @_ensure_connection
    def hget(self, name: str, key: str) -> Optional[Any]:
        """
        Get the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field in the hash.

        Returns:
            Optional[Any]: Value of the field if it exists, None otherwise.
                           Value is deserialized from JSON if possible.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - hget"
        logger.debug(f"{log_prefix} START - HGET operation for hash: '{name}', key: '{key}'.")
        
        try:
            value_raw = self.connection.hget(name, key)
            logger.debug(f"{log_prefix} Raw value for hash '{name}', key '{key}': {value_raw}")
            
            if value_raw:
                try:
                    value = json.loads(cast(str, value_raw))
                    logger.debug(f"{log_prefix} JSON deserialization successful for hash '{name}', key '{key}'. Value: {value}")
                    logger.debug(f"{log_prefix} END - HGET SUCCESS for hash: '{name}', key: '{key}'.")
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug(f"{log_prefix} Raw value returned (not JSON) for hash '{name}', key '{key}'")
                    return value_raw
            else:
                logger.debug(f"{log_prefix} No value found for hash '{name}', key '{key}'. Returning None.")
                logger.debug(f"{log_prefix} END - HGET - Key not found in hash: '{name}', key: '{key}'.")
                return None
                
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis HGET error for hash '{name}', key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - HGET ERROR. Returning None due to RedisError.")
            return None

    @_ensure_connection
    def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Set the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field in the hash.
            value (Any): Value to set (will be JSON serialized).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - hset"
        logger.debug(f"{log_prefix} START - HSET operation for hash: '{name}', key: '{key}'. Value type: {type(value)}.")
        
        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for hash '{name}', key '{key}'.")
            
            result = self.connection.hset(name, key, value_json)
            logger.debug(f"{log_prefix} Hash field '{key}' set in hash '{name}'. Result: {result}")
            logger.debug(f"{log_prefix} END - HSET SUCCESS for hash: '{name}', key: '{key}'.")
            return True
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis HSET error for hash '{name}', key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - HSET ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the Redis connection is healthy.

        Returns:
            Dict[str, Any]: Health check information with status and details.
        """
        log_prefix = f"{self._log_prefix_class} - health_check"
        logger.debug(f"{log_prefix} START - Running health check on Redis connection")

        health_info = {
            "status": False,
            "details": {}
        }

        try:
            # Basic ping check
            ping_result = self.ping()
            health_info["status"] = ping_result
            health_info["details"]["ping"] = ping_result

            if ping_result:
                # Add additional stats if connection is working
                try:
                    info = self.connection.info()
                    health_info["details"]["version"] = info.get("redis_version", "unknown")
                    health_info["details"]["uptime_days"] = info.get("uptime_in_days", 0)
                    health_info["details"]["memory_used"] = info.get("used_memory_human", "unknown")
                    health_info["details"]["clients_connected"] = info.get("connected_clients", 0)
                except Exception as e:
                    logger.error(f"{log_prefix} Error getting Redis stats: {e}", exc_info=True)
                    health_info["details"]["stats_error"] = str(e)
            
            logger.debug(f"{log_prefix} END - Health check complete - Status: {health_info['status']}")
            return health_info
            
        except Exception as e:
            logger.error(f"{log_prefix} ❌ Error during health check: {e}", exc_info=True)
            health_info["details"]["error"] = str(e)
            logger.debug(f"{log_prefix} END - Health check FAIL with exception")
            return health_info

    @_ensure_connection
    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Optional[Callable[[str], float]] = None
    ) -> list:
        """
        Retrieve members from a sorted set within a range.

        Args:
            key (str): Sorted set key.
            start (int): Start of range (inclusive).
            end (int): End of range (inclusive).
            desc (bool): Retrieve in descending order (default: False).
            withscores (bool): Include scores in the result (default: False).
            score_cast_func (Optional[Callable[[str], float]]): Function to cast scores (optional).

        Returns:
            list: List of members (and scores if withscores=True) in the specified range.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - zrange"
        logger.debug(f"{log_prefix} START - ZRANGE operation for key: '{key}', start: {start}, end: {end}, desc: {desc}, withscores: {withscores}, score_cast_func provided: {score_cast_func is not None}.")
        try:
            if score_cast_func is None:
                result_raw = self.connection.zrange(key, start, end, desc=desc, withscores=withscores)
            else:
                result_raw = self.connection.zrange(key, start, end, desc=desc, withscores=withscores, score_cast_func=score_cast_func)
                
            result = cast(list, result_raw)
            logger.debug(f"{log_prefix} Retrieved {len(result)} members from sorted set for key '{key}'.")
            logger.debug(f"{log_prefix} END - ZRANGE SUCCESS for key: '{key}', count: {len(result)}.")
            return result
            
        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis zrange error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ZRANGE ERROR. Returning empty list due to RedisError.")
            return []

    def get_last_fetched_timestamp(self, asset: str, timeframe: str) -> Optional[int]:
        """
        Retrieve the last fetched timestamp for an asset and timeframe.

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").

        Returns:
            Optional[int]: Last fetched timestamp as integer, or None if not found/invalid.
        """
        log_prefix = f"{self._log_prefix_class} - get_last_fetched_timestamp"
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(f"{log_prefix} START - Retrieving last fetched timestamp for asset: '{asset}', timeframe: '{timeframe}'. Key: '{key}'.")
        value = self.get(key)
        if value is not None:
            try:
                ts = int(value)
                logger.debug(f"{log_prefix} Parsed timestamp from Redis value: {ts}.")
                logger.debug(f"{log_prefix} END - Last fetched timestamp retrieval SUCCESS. Timestamp: {ts}.")
                return ts
            except ValueError:
                logger.error(f"{log_prefix} ❌ Invalid last fetched timestamp format for {asset} {timeframe}: {value}.")
                logger.debug(f"{log_prefix} END - Last fetched timestamp retrieval WARNING - Invalid format. Returning None.")
        else:
            logger.debug(f"{log_prefix} No timestamp found in Redis for key '{key}'.")
            logger.debug(f"{log_prefix} END - Last fetched timestamp retrieval - Key not found. Returning None.")
        return None

    def set_last_fetched_timestamp(self, asset: str, timeframe: str, timestamp: int) -> bool:
        """
        Store the last fetched timestamp for an asset and timeframe.

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").
            timestamp (int): Timestamp to store.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        log_prefix = f"{self._log_prefix_class} - set_last_fetched_timestamp"
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(f"{log_prefix} START - Setting last fetched timestamp for asset: '{asset}', timeframe: '{timeframe}', timestamp: {timestamp}. Key: '{key}'.")
        result = self.set(key, timestamp)
        logger.debug(f"{log_prefix} SET operation result: {result}")
        logger.debug(f"{log_prefix} END - Set last fetched timestamp operation - Result: {result}.")
        return result

    def calculate_fetch_range(
        self, asset: str, timeframe: str, current_time: int, buffer_seconds: int = 120
    ) -> Tuple[int, int]:
        """
        Calculate the fetch range for the next API call.

        Args:
            asset (str): Asset symbol.
            timeframe (str): Timeframe.
            current_time (int): Current timestamp.
            buffer_seconds (int): Buffer in seconds to avoid duplicates (default: 120).

        Returns:
            Tuple[int, int]: Tuple of (start_timestamp, end_timestamp) for fetch range.
        """
        log_prefix = f"{self._log_prefix_class} - calculate_fetch_range"
        logger.debug(
            f"{log_prefix} START - Calculating fetch range for asset: '{asset}', timeframe: '{timeframe}', current_time: {current_time}, buffer_seconds: {buffer_seconds}."
        )
        resolution = 60  # seconds per data point for 1-minute data
        last_timestamp = self.get_last_fetched_timestamp(asset, timeframe)
        if last_timestamp is None:
            one_week = 7 * 24 * 3600  # 7 days in seconds
            start_timestamp = current_time - one_week
            logger.debug(f"{log_prefix} No last timestamp found in Redis.")
            logger.debug(f"{log_prefix} Using start_timestamp = current_time - one_week: {start_timestamp}")
        else:
            start_timestamp = last_timestamp + resolution
            logger.debug(f"{log_prefix} Last timestamp found in Redis: {last_timestamp}.")
            logger.debug(f"{log_prefix} Using start_timestamp = last_timestamp + resolution: {start_timestamp}")
        end_timestamp = current_time - buffer_seconds
        logger.debug(f"{log_prefix} Calculated end_timestamp: {end_timestamp}")
        if start_timestamp >= end_timestamp:
            logger.debug(f"{log_prefix} No new data needed as start_timestamp >= end_timestamp.")
            logger.debug(f"{log_prefix} END - Fetch range calculation - No new data needed. Returning interval: ({start_timestamp}, {end_timestamp}).")
        else:
            logger.debug(f"{log_prefix} Calculated fetch range: Start: {start_timestamp}, End: {end_timestamp}")
            logger.debug(f"{log_prefix} END - Fetch range calculation SUCCESS. Interval: ({start_timestamp}, {end_timestamp}).")
        return start_timestamp, end_timestamp