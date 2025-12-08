"""
Utilities for resilient network operations.

Provides helpers to create retry-enabled HTTP sessions and to automatically
retry arbitrary callables when transient network errors occur. These helpers
are used across the pipeline to ensure that temporary connectivity issues do
not require manual restarts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

T = TypeVar("T")

DEFAULT_STATUS_FORCELIST: Tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_ALLOWED_METHODS = frozenset[str]({"HEAD", "GET", "OPTIONS", "POST"})


@dataclass(frozen=True)
class NetworkRetrySettings:
    """
    Configuration for retrying network operations.

    Attributes:
        max_retries: Maximum number of attempts for an operation. ``None`` means
            retry indefinitely until success or interruption.
        base_backoff_seconds: Initial wait time before retrying.
        max_backoff_seconds: Upper bound on the exponential backoff delay.
        session_retries: Number of retries configured on HTTP sessions.
    """

    max_retries: int | None = None
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    session_retries: int = 5

    @classmethod
    def from_config(cls, config: dict | None) -> "NetworkRetrySettings":
        """Build settings from a configuration dictionary."""
        if not config:
            return cls()

        return cls(
            max_retries=config.get("max_retries"),
            base_backoff_seconds=config.get("base_backoff_seconds", 2.0),
            max_backoff_seconds=config.get("max_backoff_seconds", 60.0),
            session_retries=config.get("session_retries", 5),
        )


class NetworkRetrier:
    """Retry helper that wraps network callables with exponential backoff."""

    def __init__(
        self, settings: NetworkRetrySettings, logger: logging.Logger | None = None
    ):
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        operation_name: str,
        operation: Callable[[], T],
        exceptions: Tuple[type[BaseException], ...] = (
            requests.exceptions.RequestException,
        ),
    ) -> T:
        """
        Execute an operation with retry and backoff.

        Args:
            operation_name: Human readable description used in log messages.
            operation: Callable to execute.
            exceptions: Exception types that should trigger a retry.

        Returns:
            The result of the successful operation.

        Raises:
            The last exception encountered if the maximum number of retries is
            exceeded (when ``max_retries`` is not ``None``).
        """
        attempt = 0

        while True:
            try:
                return operation()
            except exceptions as exc:  # type: ignore[misc]
                attempt += 1
                if (
                    self.settings.max_retries is not None
                    and attempt > self.settings.max_retries
                ):
                    self.logger.error(
                        "Network operation '%s' failed after %d attempts: %s",
                        operation_name,
                        attempt,
                        exc,
                    )
                    raise

                wait_time = min(
                    self.settings.base_backoff_seconds * (2 ** (attempt - 1)),
                    self.settings.max_backoff_seconds,
                )
                max_attempts_display = (
                    str(self.settings.max_retries)
                    if self.settings.max_retries is not None
                    else "∞"
                )
                self.logger.warning(
                    "Network operation '%s' failed (attempt %d/%s). Retrying in %.1f seconds. Error: %s",
                    operation_name,
                    attempt,
                    max_attempts_display,
                    wait_time,
                    exc,
                )
                time.sleep(wait_time)


def create_retry_session(
    settings: NetworkRetrySettings,
    *,
    allowed_methods: Sequence[str] | None = None,
    status_forcelist: Sequence[int] = DEFAULT_STATUS_FORCELIST,
) -> requests.Session:
    """
    Create a ``requests.Session`` configured with retry capabilities.

    Args:
        settings: Network retry configuration.
        allowed_methods: Optional iterable of HTTP methods that should be retried.
        status_forcelist: HTTP status codes that should trigger a retry.

    Returns:
        Configured ``requests.Session`` instance.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=settings.session_retries,
        read=settings.session_retries,
        connect=settings.session_retries,
        backoff_factor=1,
        status_forcelist=status_forcelist,
        allowed_methods=set(allowed_methods or DEFAULT_ALLOWED_METHODS),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
