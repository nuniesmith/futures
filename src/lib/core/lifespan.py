"""
Application lifecycle management.
Handles application startup and shutdown events.

This module provides context managers for managing application lifecycle events,
particularly for FastAPI and other web frameworks that support lifespan events.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("core.lifecycle")

try:
    from fastapi import FastAPI

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore


@asynccontextmanager
async def get_lifespan_manager(app: Any | None = None) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events with proper error handling.

    Args:
        app: Optional FastAPI application instance

    Yields:
        None: This context manager doesn't yield any value
    """
    # Startup phase
    logger.info("Starting application lifecycle")

    # Try to get service registry if available
    try:
        from lib.core.registry import get_registry

        service_registry = get_registry()
    except ImportError:
        service_registry = None
        logger.debug("Service registry not available")

    try:
        # Connect to database if available
        try:
            from lib.core.db import database

            logger.info("Connecting to database")
            db_connected = await database.connect()
            if db_connected:
                logger.info("Database connection established")
            else:
                logger.error("Failed to connect to database")
        except ImportError:
            logger.debug("Database module not available, skipping connection")

        # Start all services if service registry is available
        if service_registry:
            logger.info("Starting registered services")
            await service_registry.start_all()
            logger.info("All services started")

    except Exception as e:
        # Log startup errors but allow app to start
        # so health endpoints can report the issues
        logger.error(f"Startup error: {str(e)}", exc_info=True)

    yield  # Application runs here

    # Shutdown phase
    logger.info("Shutting down application")

    try:
        # Stop all services gracefully if service registry is available
        if service_registry:
            logger.info("Stopping registered services")
            await service_registry.stop_all()
            logger.info("All services stopped")

        # Disconnect database if available
        try:
            from lib.core.db import database

            logger.info("Disconnecting from database")
            db_disconnected = await database.disconnect()
            if db_disconnected:
                logger.info("Database disconnected successfully")
            else:
                logger.warning("Database disconnection may not be complete")
        except ImportError:
            pass

    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)

    logger.info("Application lifecycle completed")


def get_sync_lifespan_manager() -> Callable:
    """
    Get a synchronous lifespan manager for non-async frameworks.

    Returns:
        A context manager function for managing application lifecycle
    """

    @contextlib.contextmanager
    def sync_lifespan_manager():
        # Startup phase
        logger.info("Starting application lifecycle (sync)")

        # Try to get service registry if available
        try:
            from lib.core.registry import get_service_registry

            service_registry = get_service_registry()
        except ImportError:
            service_registry = None
            logger.debug("Service registry not available")

        try:
            # Connect to database if available
            try:
                import importlib

                db_module = importlib.import_module("lib.core.db")
                if hasattr(db_module, "connect_sync"):
                    logger.info("Connecting to database")
                    db_connected = db_module.connect_sync()
                    if db_connected:
                        logger.info("Database connection established")
                    else:
                        logger.error("Failed to connect to database")
            except ImportError:
                logger.debug("Database module not available, skipping connection")

            # Start all services if service registry is available
            if service_registry and hasattr(service_registry, "start_all"):
                logger.info("Starting registered services")
                run_async(service_registry.start_all())
                logger.info("All services started")

        except Exception as e:
            # Log startup errors but allow app to start
            logger.error(f"Startup error: {str(e)}", exc_info=True)

        yield  # Application runs here

        # Shutdown phase
        logger.info("Shutting down application")

        try:
            # Stop all services gracefully if service registry is available
            if service_registry and hasattr(service_registry, "stop_all"):
                logger.info("Stopping registered services")
                run_async(service_registry.stop_all())
                logger.info("All services stopped")

            # Disconnect database if available
            try:
                import importlib

                db_module = importlib.import_module("lib.core.db")
                if hasattr(db_module, "disconnect_sync"):
                    logger.info("Disconnecting from database")
                    db_disconnected = db_module.disconnect_sync()
                    if db_disconnected:
                        logger.info("Database disconnected successfully")
                    else:
                        logger.warning("Database disconnection may not be complete")
            except ImportError:
                pass

        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}", exc_info=True)

        logger.info("Application lifecycle completed")

    return sync_lifespan_manager


def setup_fastapi_lifespan(app: Any) -> None:
    """
    Set up lifespan events for a FastAPI application.

    Args:
        app: FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI is not installed, cannot set up lifespan events")
        return

    logger.warning(
        "Assigning to 'app.lifespan' is not supported. "
        "Please pass 'lifespan=get_lifespan_manager' when creating the FastAPI app."
    )


# Helper function to run a coroutine from synchronous code
def run_async(coro):
    """
    Run an async coroutine from synchronous code.

    Args:
        coro: Coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop is available, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # This is a bit risky, but sometimes necessary
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return loop.run_until_complete(coro)


class ApplicationLifecycle:
    """
    Class to manage application lifecycle events.

    This class provides a unified interface for handling startup and shutdown
    events for both synchronous and asynchronous applications.
    """

    def __init__(self):
        """Initialize the application lifecycle manager."""
        self.services = []
        self.is_started = False

    def register_service(self, service: Any) -> None:
        """
        Register a service to be managed by the lifecycle manager.

        Args:
            service: Service object with start/stop methods
        """
        self.services.append(service)
        logger.debug(f"Registered service: {service.__class__.__name__}")

    def register_services(self, services: list[Any]) -> None:
        """
        Register multiple services to be managed by the lifecycle manager.

        Args:
            services: List of service objects with start/stop methods
        """
        for service in services:
            self.register_service(service)

    async def start_async(self) -> None:
        """Start all registered services asynchronously."""
        if self.is_started:
            logger.warning("Services are already started")
            return

        logger.info("Starting all services asynchronously")

        for service in self.services:
            try:
                # Check if service has async start method
                if hasattr(service, "start_async"):
                    await service.start_async()
                elif hasattr(service, "start"):
                    # Run synchronous start method in executor
                    await asyncio.to_thread(service.start)
                else:
                    logger.warning(f"Service {service.__class__.__name__} has no start method")
            except Exception as e:
                logger.error(f"Error starting service {service.__class__.__name__}: {str(e)}")

        self.is_started = True
        logger.info("All services started")

    def start_sync(self) -> None:
        """Start all registered services synchronously."""
        if self.is_started:
            logger.warning("Services are already started")
            return

        logger.info("Starting all services synchronously")

        for service in self.services:
            try:
                # Check if service has synchronous start method
                if hasattr(service, "start"):
                    service.start()
                elif hasattr(service, "start_async"):
                    # Run async start method in event loop
                    run_async(service.start_async())
                else:
                    logger.warning(f"Service {service.__class__.__name__} has no start method")
            except Exception as e:
                logger.error(f"Error starting service {service.__class__.__name__}: {str(e)}")

        self.is_started = True
        logger.info("All services started")

    async def stop_async(self) -> None:
        """Stop all registered services asynchronously."""
        if not self.is_started:
            logger.warning("Services are not started")
            return

        logger.info("Stopping all services asynchronously")

        # Stop services in reverse order
        for service in reversed(self.services):
            try:
                # Check if service has async stop method
                if hasattr(service, "stop_async"):
                    await service.stop_async()
                elif hasattr(service, "stop"):
                    # Run synchronous stop method in executor
                    await asyncio.to_thread(service.stop)
                else:
                    logger.warning(f"Service {service.__class__.__name__} has no stop method")
            except Exception as e:
                logger.error(f"Error stopping service {service.__class__.__name__}: {str(e)}")

        self.is_started = False
        logger.info("All services stopped")

    def stop_sync(self) -> None:
        """Stop all registered services synchronously."""
        if not self.is_started:
            logger.warning("Services are not started")
            return

        logger.info("Stopping all services synchronously")

        # Stop services in reverse order
        for service in reversed(self.services):
            try:
                # Check if service has synchronous stop method
                if hasattr(service, "stop"):
                    service.stop()
                elif hasattr(service, "stop_async"):
                    # Run async stop method in event loop
                    run_async(service.stop_async())
                else:
                    logger.warning(f"Service {service.__class__.__name__} has no stop method")
            except Exception as e:
                logger.error(f"Error stopping service {service.__class__.__name__}: {str(e)}")

        self.is_started = False
        logger.info("All services stopped")

    @asynccontextmanager
    async def lifespan_context(self):
        """
        Asynchronous context manager for application lifespan.

        This can be used as a lifespan manager for FastAPI applications.
        """
        try:
            await self.start_async()
            yield
        finally:
            await self.stop_async()

    @contextlib.contextmanager
    def sync_lifespan_context(self):
        """
        Synchronous context manager for application lifespan.

        This can be used for synchronous web frameworks.
        """
        try:
            self.start_sync()
            yield
        finally:
            self.stop_sync()


# Global application lifecycle instance
_app_lifecycle = ApplicationLifecycle()


def get_app_lifecycle() -> ApplicationLifecycle:
    """
    Get the global application lifecycle instance.

    Returns:
        ApplicationLifecycle: Global application lifecycle instance
    """
    global _app_lifecycle
    return _app_lifecycle
