import asyncio
import json
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, cast

import redis.asyncio as redis_async
import redis_clients
from loguru import logger
from redis.asyncio.client import Pipeline, PubSub

from lib.core.db.redis_clients import BaseRedisClient, RedisError, _ensure_connection
from lib.core.db.redis_clients.utils import CustomJSONEncoder

_log_prefix_base = "[redis_client - async]"


class AsyncRedisClient(BaseRedisClient):
    """
    Asynchronous Redis client implementing Redis operations using redis.asyncio.
    Inherits connection and reconnection logic from BaseRedisClient.

    All data is stored in Redis as JSON strings. Values retrieved are deserialized from JSON.
    """

    _log_prefix_class = f"{_log_prefix_base} - AsyncRedisClient"

    def _create_connection(self) -> redis_async.Redis:
        """Creates an asynchronous Redis connection using connection pooling."""
        log_prefix = f"{self._log_prefix_class} - _create_connection"
        logger.debug(f"{log_prefix} START - Creating asynchronous Redis connection using URL: {self.clean_url}")

        # Get connection pool configuration
        pool_kwargs = self._get_connection_pool_kwargs()

        # Generate unique key for this connection config
        pool_key = self._get_connection_pool_key()

        if self.connection_params.get("use_cluster", False):
            # For cluster mode
            from redis.asyncio.cluster import RedisCluster  # type: ignore[attr-defined]

            # Define factory function to create a cluster connection pool
            def create_cluster_pool(**kwargs):  # type: ignore[return]
                startup_nodes = [
                    {"host": host, "port": self.connection_params["cluster_port"]}
                    for host in self.connection_params["cluster_hosts"]
                ]
                return RedisCluster(  # type: ignore[call-arg]
                    startup_nodes=startup_nodes,
                    username=self.connection_params.get("user"),
                    password=self.connection_params.get("password"),
                    ssl=self.connection_params.get("use_tls", False),
                    decode_responses=True,
                    **kwargs,
                )

            # Get or create a cluster connection pool
            connection: Any = self._get_or_create_connection_pool(pool_key, create_cluster_pool, **pool_kwargs)
            logger.debug(f"{log_prefix} END - Asynchronous Redis CLUSTER connection created: {connection}")
            return connection  # type: ignore[return-value]
        else:
            # For standard Redis or Sentinel
            # Define factory function for the connection pool
            def create_standard_pool(**kwargs):
                return redis_async.ConnectionPool.from_url(url=self.redis_url, **kwargs)

            # Get or create a standard connection pool
            pool = self._get_or_create_connection_pool(pool_key, create_standard_pool, **pool_kwargs)

            # Create Redis client with the pool
            connection = redis_async.Redis(connection_pool=pool, decode_responses=True)
            logger.debug(f"{log_prefix} END - Asynchronous Redis connection created: {connection}")
            return connection

    async def ping(self) -> bool:  # type: ignore[override]
        """
        Asynchronously ping the Redis server to verify the connection.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        log_prefix = f"{self._log_prefix_class} - ping"
        logger.debug(f"{log_prefix} START - Pinging Redis server")

        if not self.connection:
            logger.debug(f"{log_prefix} No connection available, returning False")
            return False

        try:
            result = await self.connection.ping()
            logger.debug(f"{log_prefix} END - Ping result: {result}")
            return result
        except Exception as e:
            logger.error(f"{log_prefix} ❌ Error pinging Redis: {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - Ping failed")
            return False

    async def _initialize_connection_async(self) -> None:
        """
        Asynchronously initialize the Redis connection with retry logic.
        """
        log_prefix = f"{self._log_prefix_class} - _initialize_connection_async"
        logger.debug(f"{log_prefix} START - Initializing async connection")

        now = time.time()
        last: float = self.last_connection_attempt  # type: ignore[has-type]
        if now - last < 5:
            logger.warning(f"{log_prefix} ⚠️ Skipping reconnect attempt (throttled). Last attempt was too recent.")
            logger.debug(f"{log_prefix} END - Connection initialization SKIPPED due to throttling.")
            return

        self.last_connection_attempt = now  # type: ignore[has-type]

        for attempt in range(self.max_retries):
            attempt_num = attempt + 1
            logger.debug(f"{log_prefix} Attempt {attempt_num}/{self.max_retries} to connect.")
            try:
                logger.debug(f"{log_prefix} Attempting async connection to Redis using URL: {self.clean_url}.")
                self.connection = self._create_connection()

                # Verify connection with ping
                if self.connection and await self.ping():
                    logger.info(f"{log_prefix} ✅ Connected to Redis on attempt {attempt_num}.")
                    logger.debug(
                        f"{log_prefix} END - Async connection initialization SUCCESS on attempt {attempt_num}."
                    )
                    return
            except redis_clients.RedisError as e:
                logger.warning(
                    f"{log_prefix} ⚠️ Redis connection failed on attempt {attempt_num}/{self.max_retries}: {e}"
                )
                base_sleep = 2**attempt
                jitter = random.uniform(0, 1)
                time_to_sleep = base_sleep + jitter
                logger.debug(f"{log_prefix} Backoff - Sleeping for {time_to_sleep:.2f} seconds before next attempt.")
                await asyncio.sleep(time_to_sleep)

        error_message = "❌ Failed to connect to Redis after multiple attempts."
        logger.error(f"{log_prefix} {error_message}")
        logger.debug(f"{log_prefix} END - Async connection initialization FAILED after {self.max_retries} attempts.")
        raise RedisError(error_message)

    async def pubsub(self) -> PubSub:
        """
        Return an asynchronous Redis PubSub object.

        Returns:
            PubSub: Asynchronous Redis PubSub object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - pubsub"
        logger.debug(f"{log_prefix} START - Getting asynchronous Redis PubSub object.")

        self.reconnect_if_needed()
        if not self.connection:
            error_message = "Redis connection is None, cannot create PubSub."
            logger.error(f"{log_prefix} ❌ {error_message}")
            logger.debug(f"{log_prefix} END - PubSub retrieval FAIL - No connection available. Raising RedisError.")
            raise RedisError(error_message)

        pubsub_obj = self.connection.pubsub()
        logger.debug(f"{log_prefix} Asynchronous PubSub object created successfully: {pubsub_obj}")
        logger.debug(f"{log_prefix} END - PubSub retrieval SUCCESS.")
        return cast("PubSub", pubsub_obj)

    @_ensure_connection
    async def get(self, key: str) -> Any | None:
        """
        Asynchronously retrieve data from Redis by key, and deserialize from JSON.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The deserialized value if the key exists, otherwise None.
                           Value is deserialized from JSON.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - get"
        logger.debug(f"{log_prefix} START - ASYNC GET operation for key: '{key}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(
                    f"{log_prefix} END - ASYNC GET operation FAIL - No connection available. Raising RedisError."
                )
                raise RedisError(error_message)

            value_raw = await self.connection.get(key)
            logger.debug(f"{log_prefix} Raw value for key '{key}': {value_raw}")

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug(f"{log_prefix} JSON deserialization successful for key '{key}'. Value: {value}")
                    logger.debug(f"{log_prefix} END - ASYNC GET operation SUCCESS for key: '{key}'.")
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug(f"{log_prefix} Raw value returned (not JSON) for key '{key}'")
                    return value_raw
            else:
                logger.debug(f"{log_prefix} No value found for key '{key}'. Returning None.")
                logger.debug(f"{log_prefix} END - ASYNC GET operation - Key not found: '{key}'.")
                return None

        except redis_clients.ConnectionError as e:
            logger.error(
                f"{log_prefix} ❌ Connection error during ASYNC GET operation for key '{key}': {e}", exc_info=True
            )
            await self._initialize_connection_async()
            logger.debug(
                f"{log_prefix} END - ASYNC GET operation ERROR - Connection error. Reconnection attempted. Returning None."
            )
            return None

    @_ensure_connection
    async def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """
        Asynchronously store a value in Redis, after serializing it to JSON.

        Args:
            key (str): The key to set.
            value (Any): The value to store (will be JSON serialized).
            expiry (Optional[int]): Expiry time in seconds (optional).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - set"
        logger.debug(
            f"{log_prefix} START - ASYNC SET operation for key: '{key}', expiry: {expiry}. Value type: {type(value)}."
        )

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for key '{key}'.")

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(
                    f"{log_prefix} END - ASYNC SET operation FAIL - No connection available. Raising RedisError."
                )
                raise RedisError(error_message)

            if expiry:
                result = await self.connection.setex(key, expiry, value_json)
                logger.debug(f"{log_prefix} Key '{key}' set with expiry {expiry}. Result: {result}")
            else:
                result = bool(await self.connection.set(key, value_json))
                logger.debug(f"{log_prefix} Key '{key}' set without expiry. Result: {result}")

            logger.debug(f"{log_prefix} END - ASYNC SET operation SUCCESS for key: '{key}'.")
            return True

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC SET error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC SET operation ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    async def delete(self, key: str) -> bool:
        """
        Asynchronously delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was successfully deleted, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - delete"
        logger.debug(f"{log_prefix} START - ASYNC DELETE operation for key: '{key}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(
                    f"{log_prefix} END - ASYNC DELETE operation FAIL - No connection available. Raising RedisError."
                )
                raise RedisError(error_message)

            result = await self.connection.delete(key)
            logger.debug(f"{log_prefix} DELETE operation result for key '{key}': {result}")
            logger.debug(f"{log_prefix} END - ASYNC DELETE operation for key: '{key}'. Result: {bool(result)}.")
            return bool(result)

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC DELETE error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC DELETE operation ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    async def exists(self, key: str) -> bool:
        """
        Asynchronously check if a key exists in Redis.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - exists"
        logger.debug(f"{log_prefix} START - ASYNC EXISTS check for key: '{key}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(
                    f"{log_prefix} END - ASYNC EXISTS check FAIL - No connection available. Raising RedisError."
                )
                raise RedisError(error_message)

            result = await self.connection.exists(key)
            logger.debug(f"{log_prefix} EXISTS check for key '{key}' returned: {result}")
            logger.debug(f"{log_prefix} END - ASYNC EXISTS check for key: '{key}'.")
            return bool(result)

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC EXISTS error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC EXISTS check ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    async def keys(self, pattern: str = "*") -> list:
        """
        Asynchronously retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The key pattern to match (default: "*", i.e., all keys).

        Returns:
            list: A list of keys that match the pattern.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - keys"
        logger.debug(f"{log_prefix} START - ASYNC KEYS retrieval with pattern: '{pattern}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(
                    f"{log_prefix} END - ASYNC KEYS retrieval FAIL - No connection available. Raising RedisError."
                )
                raise RedisError(error_message)

            keys_list = await self.connection.keys(pattern)
            keys_converted = list(keys_list) if isinstance(keys_list, list) else list(cast("Iterable", keys_list))
            logger.debug(f"{log_prefix} Retrieved {len(keys_converted)} keys for pattern '{pattern}'.")
            logger.debug(f"{log_prefix} END - ASYNC KEYS retrieval SUCCESS with pattern: '{pattern}'.")
            return keys_converted

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC KEYS error with pattern '{pattern}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC KEYS retrieval ERROR. Returning empty list due to RedisError.")
            return []

    @_ensure_connection
    async def publish(self, channel: str, message: str | bytes) -> int:
        """
        Asynchronously publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish the message to.
            message (Union[str, bytes]): The message to publish.

        Returns:
            int: The number of subscribers who received the message.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - publish"
        logger.debug(f"{log_prefix} START - ASYNC PUBLISH message to channel: '{channel}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(f"{log_prefix} END - ASYNC PUBLISH FAIL - No connection available. Raising RedisError.")
                raise RedisError(error_message)

            result_raw = await self.connection.publish(channel, message)
            result = int(result_raw) if result_raw is not None else 0  # type: ignore[redundant-cast]
            logger.debug(f"{log_prefix} Message published to channel '{channel}'. Subscribers count: {result}")
            logger.debug(f"{log_prefix} END - ASYNC PUBLISH SUCCESS to channel: '{channel}'. Subscribers: {result}.")
            return result

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC publish error for channel '{channel}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC PUBLISH ERROR. Returning 0 subscribers due to RedisError.")
            return 0

    def pipeline(self, transaction: bool = True) -> Pipeline:
        """
        Return a Redis pipeline for batching multiple commands (async version).

        Args:
            transaction (bool): Whether to use a transaction (default: True).

        Returns:
            Pipeline: Redis pipeline object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        log_prefix = f"{self._log_prefix_class} - pipeline"
        logger.debug(f"{log_prefix} START - Getting ASYNC Redis pipeline. Transaction enabled: {transaction}.")

        self.reconnect_if_needed()
        if self.connection is None:
            error_message = "Redis connection is not available."
            logger.error(f"{log_prefix} ❌ {error_message}")
            logger.debug(
                f"{log_prefix} END - ASYNC Pipeline retrieval FAIL - No connection available. Raising RedisError."
            )
            raise RedisError(error_message)

        pipeline_obj = self.connection.pipeline(transaction=transaction)
        logger.debug(f"{log_prefix} Redis ASYNC pipeline object retrieved: {pipeline_obj}")
        logger.debug(f"{log_prefix} END - ASYNC Pipeline retrieval SUCCESS.")
        return pipeline_obj  # type: ignore[return-value]

    @_ensure_connection
    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> list:
        """
        Retrieve members from a sorted set within a range (asynchronous).

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
        logger.debug(
            f"{log_prefix} START - ASYNC ZRANGE operation for key: '{key}', start: {start}, end: {end}, desc: {desc}, withscores: {withscores}, score_cast_func provided: {score_cast_func is not None}."
        )

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(f"{log_prefix} END - ASYNC ZRANGE FAIL - No connection available. Raising RedisError.")
                raise RedisError(error_message)

            if score_cast_func is None:
                result_raw = await self.connection.zrange(key, start, end, desc=desc, withscores=withscores)
            else:
                result_raw = await self.connection.zrange(
                    key, start, end, desc=desc, withscores=withscores, score_cast_func=score_cast_func
                )

            result: list = list(result_raw) if result_raw is not None else []  # type: ignore[redundant-cast]
            logger.debug(f"{log_prefix} Retrieved {len(result)} members from sorted set for key '{key}'.")
            logger.debug(f"{log_prefix} END - ASYNC ZRANGE SUCCESS for key: '{key}', count: {len(result)}.")
            return result

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC ZRANGE error for key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC ZRANGE ERROR. Returning empty list due to RedisError.")
            return []

    async def get_last_fetched_timestamp(self, asset: str, timeframe: str) -> int | None:
        """
        Retrieve the last fetched timestamp for an asset and timeframe (asynchronous).

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").

        Returns:
            Optional[int]: Last fetched timestamp as integer, or None if not found/invalid.
        """
        log_prefix = f"{self._log_prefix_class} - get_last_fetched_timestamp"
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(
            f"{log_prefix} START - Retrieving last fetched timestamp for asset: '{asset}', timeframe: '{timeframe}'. Key: '{key}'."
        )

        value = await self.get(key)
        if value is not None:
            try:
                ts = int(value)
                logger.debug(f"{log_prefix} Parsed timestamp from Redis value: {ts}.")
                logger.debug(f"{log_prefix} END - Last fetched timestamp retrieval SUCCESS. Timestamp: {ts}.")
                return ts
            except ValueError:
                logger.error(f"{log_prefix} ❌ Invalid last fetched timestamp format for {asset} {timeframe}: {value}.")
                logger.debug(
                    f"{log_prefix} END - Last fetched timestamp retrieval WARNING - Invalid format. Returning None."
                )
        else:
            logger.debug(f"{log_prefix} No timestamp found in Redis for key '{key}'.")
            logger.debug(f"{log_prefix} END - Last fetched timestamp retrieval - Key not found. Returning None.")

        return None

    async def set_last_fetched_timestamp(self, asset: str, timeframe: str, timestamp: int) -> bool:
        """
        Store the last fetched timestamp for an asset and timeframe (asynchronous).

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").
            timestamp (int): Timestamp to store.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        log_prefix = f"{self._log_prefix_class} - set_last_fetched_timestamp"
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(
            f"{log_prefix} START - Setting last fetched timestamp for asset: '{asset}', timeframe: '{timeframe}', timestamp: {timestamp}. Key: '{key}'."
        )

        result = await self.set(key, timestamp)
        logger.debug(f"{log_prefix} SET operation result: {result}")
        logger.debug(f"{log_prefix} END - Set last fetched timestamp operation - Result: {result}.")
        return result

    async def calculate_fetch_range(
        self, asset: str, timeframe: str, current_time: int, buffer_seconds: int = 120
    ) -> tuple[int, int]:
        """
        Calculate the fetch range for the next API call (asynchronous).

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
        last_timestamp = await self.get_last_fetched_timestamp(asset, timeframe)

        if last_timestamp is None:
            two_years = 7 * 24 * 3600  # using 7 days as placeholder; adjust as needed
            start_timestamp = current_time - two_years
            logger.debug(f"{log_prefix} No last timestamp found in Redis.")
            logger.debug(f"{log_prefix} Using start_timestamp = current_time - two_years: {start_timestamp}")
        else:
            start_timestamp = last_timestamp + resolution
            logger.debug(f"{log_prefix} Last timestamp found in Redis: {last_timestamp}.")
            logger.debug(f"{log_prefix} Using start_timestamp = last_timestamp + resolution: {start_timestamp}")

        end_timestamp = current_time - buffer_seconds
        logger.debug(f"{log_prefix} Calculated end_timestamp: {end_timestamp}")

        if start_timestamp >= end_timestamp:
            logger.debug(
                f"{log_prefix} No new data needed. start_timestamp ({start_timestamp}) >= end_timestamp ({end_timestamp})."
            )
            logger.debug(
                f"{log_prefix} END - Fetch range calculation - No new data needed. Returning interval: ({start_timestamp}, {end_timestamp})."
            )
        else:
            logger.debug(f"{log_prefix} Calculated fetch range: Start: {start_timestamp}, End: {end_timestamp}")
            logger.debug(
                f"{log_prefix} END - Fetch range calculation SUCCESS. Interval: ({start_timestamp}, {end_timestamp})."
            )

        return start_timestamp, end_timestamp

    # Added missing abstract methods

    @_ensure_connection
    async def health_check(self) -> bool:
        """
        Check if the Redis connection is healthy.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        log_prefix = f"{self._log_prefix_class} - health_check"
        logger.debug(f"{log_prefix} START - Running health check on Redis connection")

        try:
            if self.connection is None:
                logger.error(f"{log_prefix} ❌ No Redis connection available")
                logger.debug(f"{log_prefix} END - Health check FAIL - No connection available")
                return False

            # Perform a ping to check connection
            result = await self.ping()
            logger.debug(f"{log_prefix} END - Health check result: {result}")
            return result
        except Exception as e:
            logger.error(f"{log_prefix} ❌ Error during health check: {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - Health check FAIL due to exception")
            return False

    @_ensure_connection
    async def hget(self, name: str, key: str) -> Any | None:
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
        logger.debug(f"{log_prefix} START - ASYNC HGET operation for hash: '{name}', key: '{key}'.")

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(f"{log_prefix} END - ASYNC HGET FAIL - No connection available. Raising RedisError.")
                raise RedisError(error_message)

            value_raw = await self.connection.hget(name, key)
            logger.debug(f"{log_prefix} Raw value for hash '{name}', key '{key}': {value_raw}")

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug(
                        f"{log_prefix} JSON deserialization successful for hash '{name}', key '{key}'. Value: {value}"
                    )
                    logger.debug(f"{log_prefix} END - ASYNC HGET SUCCESS for hash: '{name}', key: '{key}'.")
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug(f"{log_prefix} Raw value returned (not JSON) for hash '{name}', key '{key}'")
                    return value_raw
            else:
                logger.debug(f"{log_prefix} No value found for hash '{name}', key '{key}'. Returning None.")
                logger.debug(f"{log_prefix} END - ASYNC HGET - Key not found in hash: '{name}', key: '{key}'.")
                return None

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC HGET error for hash '{name}', key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC HGET ERROR. Returning None due to RedisError.")
            return None

    @_ensure_connection
    async def hset(self, name: str, key: str, value: Any) -> bool:
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
        logger.debug(
            f"{log_prefix} START - ASYNC HSET operation for hash: '{name}', key: '{key}'. Value type: {type(value)}."
        )

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for hash '{name}', key '{key}'.")

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(f"{log_prefix} END - ASYNC HSET FAIL - No connection available. Raising RedisError.")
                raise RedisError(error_message)

            result = await self.connection.hset(name, key, value_json)
            logger.debug(f"{log_prefix} Hash field '{key}' set in hash '{name}'. Result: {result}")
            logger.debug(f"{log_prefix} END - ASYNC HSET SUCCESS for hash: '{name}', key: '{key}'.")
            return True

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC HSET error for hash '{name}', key '{key}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC HSET ERROR. Returning False due to RedisError.")
            return False

    @_ensure_connection
    async def setex(self, name: str, time: int, value: Any) -> bool:
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
        logger.debug(
            f"{log_prefix} START - ASYNC SETEX operation for key: '{name}', expiry: {time}s. Value type: {type(value)}."
        )

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug(f"{log_prefix} JSON serialization successful for key '{name}'.")

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error(f"{log_prefix} ❌ {error_message}")
                logger.debug(f"{log_prefix} END - ASYNC SETEX FAIL - No connection available. Raising RedisError.")
                raise RedisError(error_message)

            result = await self.connection.setex(name, time, value_json)
            logger.debug(f"{log_prefix} Key '{name}' set with expiry {time}s. Result: {result}")
            logger.debug(f"{log_prefix} END - ASYNC SETEX SUCCESS for key: '{name}'.")
            return True

        except redis_clients.RedisError as e:
            logger.error(f"{log_prefix} ❌ Redis ASYNC SETEX error for key '{name}': {e}", exc_info=True)
            logger.debug(f"{log_prefix} END - ASYNC SETEX ERROR. Returning False due to RedisError.")
            return False
