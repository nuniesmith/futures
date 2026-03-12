"""
ORM engine management for the FKS Trading Systems.

This module provides SQLAlchemy ORM engine initialization, management
and cleanup functionality for the application.
"""

import threading
from typing import Dict, Any, Optional, List
from loguru import logger

# Try to import SQLAlchemy
try:
    import sqlalchemy
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, scoped_session
    from sqlalchemy.ext.declarative import declarative_base
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Module logger
_logger = logger.bind(name="core.database.orm")

# Base ORM model
Base = None
if HAS_SQLALCHEMY:
    Base = declarative_base()

# Engine and session tracking
_engine_lock = threading.RLock()
_engines: Dict[str, Any] = {}
_session_factories: Dict[str, Any] = {}
_active_sessions: List[Any] = []


def init_engine(
    db_url: str,
    engine_name: str = "default",
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    **engine_kwargs
) -> Any:
    """
    Initialize a SQLAlchemy engine.
    
    Args:
        db_url: Database connection URL
        engine_name: Name for this engine instance
        echo: Whether to echo SQL statements
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        pool_timeout: Pool timeout in seconds
        **engine_kwargs: Additional engine arguments
        
    Returns:
        SQLAlchemy engine or None if SQLAlchemy is not available
    """
    if not HAS_SQLALCHEMY:
        _logger.warning("SQLAlchemy is not installed, cannot initialize ORM engine")
        return None
    
    with _engine_lock:
        if engine_name in _engines:
            _logger.debug(f"Engine '{engine_name}' already initialized")
            return _engines[engine_name]
        
        try:
            # Create engine with connection pooling settings
            engine = create_engine(
                db_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                **engine_kwargs
            )
            
            _engines[engine_name] = engine
            
            # Create session factory
            session_factory = sessionmaker(bind=engine)
            _session_factories[engine_name] = scoped_session(session_factory)
            
            _logger.info(f"Initialized SQLAlchemy engine '{engine_name}'")
            return engine
            
        except Exception as e:
            _logger.error(f"Error initializing SQLAlchemy engine '{engine_name}': {e}")
            return None


def get_engine(engine_name: str = "default") -> Optional[Any]:
    """
    Get a SQLAlchemy engine by name.
    
    Args:
        engine_name: Name of the engine
        
    Returns:
        SQLAlchemy engine or None if not found
    """
    with _engine_lock:
        return _engines.get(engine_name)


def get_session(engine_name: str = "default") -> Optional[Any]:
    """
    Get a SQLAlchemy session.
    
    The session will be tracked for cleanup during system shutdown.
    
    Args:
        engine_name: Name of the engine to use
        
    Returns:
        SQLAlchemy session or None if engine not found
    """
    with _engine_lock:
        session_factory = _session_factories.get(engine_name)
        if not session_factory:
            return None
        
        session = session_factory()
        _active_sessions.append(session)
        return session


def close_session(session: Any) -> bool:
    """
    Close a SQLAlchemy session.
    
    Args:
        session: The session to close
        
    Returns:
        True if successful, False otherwise
    """
    if not session:
        return True
        
    try:
        session.close()
        with _engine_lock:
            if session in _active_sessions:
                _active_sessions.remove(session)
        return True
    except Exception as e:
        _logger.error(f"Error closing session: {e}")
        return False


def shutdown_engine(engine_name: Optional[str] = None) -> bool:
    """
    Shutdown SQLAlchemy engine(s).
    
    This function properly disposes of SQLAlchemy engines and ensures
    all database connections are closed.
    
    Args:
        engine_name: Name of engine to shut down, or None for all engines
        
    Returns:
        True if successful, False otherwise
    """
    if not HAS_SQLALCHEMY:
        return True  # Nothing to do
    
    success = True
    
    # Close all active sessions first
    with _engine_lock:
        active_sessions = _active_sessions.copy()
    
    for session in active_sessions:
        try:
            session.close()
            with _engine_lock:
                if session in _active_sessions:
                    _active_sessions.remove(session)
        except Exception as e:
            _logger.error(f"Error closing session during engine shutdown: {e}")
            success = False
    
    # Dispose engines
    with _engine_lock:
        engines_to_shutdown = {}
        
        if engine_name is not None:
            # Shut down specific engine
            if engine_name in _engines:
                engines_to_shutdown[engine_name] = _engines[engine_name]
            else:
                _logger.warning(f"Engine '{engine_name}' not found for shutdown")
        else:
            # Shut down all engines
            engines_to_shutdown = _engines.copy()
        
        # Dispose each engine
        for name, engine in engines_to_shutdown.items():
            try:
                _logger.info(f"Shutting down SQLAlchemy engine '{name}'")
                engine.dispose()
                
                # Remove from tracking
                if name in _engines:
                    del _engines[name]
                
                if name in _session_factories:
                    _session_factories[name].remove()
                    del _session_factories[name]
                    
            except Exception as e:
                _logger.error(f"Error shutting down SQLAlchemy engine '{name}': {e}")
                success = False
    
    if success:
        _logger.info("SQLAlchemy engine(s) shut down successfully")
    else:
        _logger.warning("Some errors occurred during SQLAlchemy engine shutdown")
    
    return success