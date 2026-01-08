"""
Request-scoped structured logging utility for affinetes environments

This module provides a reusable logging system for tracking requests across
different environments (lgc-v2, openspiel, trace, etc.)
"""

import time
import logging
import structlog
from contextvars import ContextVar
import re


# Configure structlog for human-readable console output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=False),
        structlog.dev.ConsoleRenderer(colors=False)  # Disable colors for Docker logs
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

base_logger = structlog.get_logger("affinetes")

# Context variable to hold the current request logger
_request_logger: ContextVar['RequestLogger'] = ContextVar('request_logger', default=None)


class RequestLogger:
    """Context manager for request-scoped structured logging"""

    def __init__(self, **context):
        """
        Initialize request logger with context information

        Args:
            **context: Arbitrary context fields. Common fields include:
                - task_id: Unique identifier for the task
                - task_type: Type of task (e.g., "dyck_language", "game", etc.)
                - seed: Random seed for reproducibility
                - model: Model name
                - base_url: API endpoint URL
                - game: Game type (for game environments)
                - trace: Trace identifier (for trace environments)
                - Any other custom fields
        """
        self.context = context
        self.start_time = time.time()

        # Extract miner slug from base_url if present
        if 'base_url' in context:
            self.context['miner'] = self._extract_miner_slug(context['base_url'])

        # Build compact request context string for easy grepping
        self.request_context = self._build_context_string()

        # Create logger bound with request context
        self.logger = base_logger.bind(req_ctx=self.request_context)

    def _extract_miner_slug(self, base_url):
        """Extract miner slug from base_url"""
        # e.g., https://silveraffine9-jrohner-affine-20261.chutes.ai/v1 -> silveraffine9-jrohner-affine-20261
        slug_match = re.search(r'https?://([^./]+)\.chutes\.ai', base_url)
        if slug_match:
            return slug_match.group(1)
        else:
            # Fallback: extract hostname or use first 40 chars
            hostname_match = re.search(r'https?://([^/]+)', base_url)
            return hostname_match.group(1)[:40] if hostname_match else base_url[:40]

    def _build_context_string(self):
        """Build compact context string from all fields"""
        # Define preferred order for common fields
        field_order = ['task_id', 'type', 'seed', 'miner', 'model', 'game', 'trace']

        ctx_parts = []

        # Add fields in preferred order
        for field in field_order:
            # Handle task_type -> type mapping
            if field == 'type' and 'task_type' in self.context:
                ctx_parts.append(f"type:{self.context['task_type']}")
            elif field in self.context and self.context[field] is not None:
                ctx_parts.append(f"{field}:{self.context[field]}")

        # Add any remaining fields not in preferred order
        for key, value in self.context.items():
            if key not in field_order and key != 'task_type' and key != 'base_url' and value is not None:
                ctx_parts.append(f"{key}:{value}")

        return "|".join(ctx_parts)

    def log(self, event, level='info', **details):
        """Log an event with elapsed time"""
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        details['elapsed_ms'] = elapsed_ms

        log_method = getattr(self.logger, level, self.logger.info)
        log_method(event, **details)

    def __enter__(self):
        self.token = _request_logger.set(self)
        self.log("request_start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.log("request_error", level='error', error=str(exc_val), error_type=exc_type.__name__)
        _request_logger.reset(self.token)


def log_event(event, level='info', **details):
    """
    Global logging function - automatically uses current request logger

    Args:
        event: Event name (e.g., "llm_call_start", "stream_complete")
        level: Log level ('info', 'warning', 'error')
        **details: Additional fields to log
    """
    logger_instance = _request_logger.get()
    if logger_instance:
        logger_instance.log(event, level=level, **details)
    else:
        # Fallback if no request context
        base_logger.info(event, **details)


def get_logger():
    """Get the current request logger or base logger"""
    logger_instance = _request_logger.get()
    return logger_instance.logger if logger_instance else base_logger
