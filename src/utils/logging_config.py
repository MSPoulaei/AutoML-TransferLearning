import sys
from pathlib import Path
from loguru import logger

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/orchestrator.log",
    rotation: str = "10 MB",
    retention: str = "1 week",
    logfire_token: str = None,
):
    """Configure loguru logger with file and console outputs, and optionally logfire."""

    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Logfire integration (optional)
    if LOGFIRE_AVAILABLE and logfire_token:
        _setup_logfire(logfire_token, log_level)
        logfire_status = "enabled"
    elif LOGFIRE_AVAILABLE and not logfire_token:
        logfire_status = "available but no token set (LOGFIRE_TOKEN)"
    else:
        logfire_status = "not installed"

    logger.info(
        f"Logging configured: level={log_level}, file={log_file}, logfire={logfire_status}"
    )

    return logger


def _setup_logfire(token: str, log_level: str) -> None:
    """Configure logfire and bridge loguru logs to it."""
    import logfire

    logfire.configure(token=token, service_name="automl-transfer-learning")

    # Instrument pydantic-ai to auto-trace every LLM call (prompts, completions,
    # token counts, latency) visible in the Logfire UI
    try:
        logfire.instrument_pydantic_ai()
        logger.info("Logfire: pydantic-ai instrumentation enabled")
    except Exception as e:
        logger.warning(f"Logfire: could not instrument pydantic-ai: {e}")

    # Map loguru level names to logfire functions
    _level_fns = {
        "TRACE": logfire.debug,
        "DEBUG": logfire.debug,
        "INFO": logfire.info,
        "SUCCESS": logfire.info,
        "WARNING": logfire.warning,
        "ERROR": logfire.error,
        "CRITICAL": logfire.error,
    }

    def _logfire_sink(message):
        record = message.record
        fn = _level_fns.get(record["level"].name, logfire.info)
        fn(
            "{log_message}",
            log_message=record["message"],
            logger_name=record["name"],
            function=record["function"],
            line=record["line"],
        )

    logger.add(_logfire_sink, level=log_level)
    logger.info("Logfire sink active — logs streaming to Logfire UI")


def get_logger(name: str = None):
    """Get a logger instance with optional name binding."""
    if name:
        return logger.bind(name=name)
    return logger
