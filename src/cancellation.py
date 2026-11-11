"""Cooperative cancellation of a computation, and its progress; ordinary CLI runs have neither."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event
from typing import Callable


class InferenceCancelled(Exception):
    """The user stopped this computation before its results were committed."""


_token: ContextVar[Event | None] = ContextVar("inference_cancellation", default=None)
_progress: ContextVar[Callable[[str], None] | None] = ContextVar("inference_progress", default=None)


def check_cancelled() -> None:
    event = _token.get()
    if event is not None and event.is_set():
        raise InferenceCancelled("Inference stopped. The previous session and results were kept.")


def report_progress(text: str) -> None:
    """Say what the computation is doing now, for whoever started it (the page's status line); else nothing."""
    report = _progress.get()
    if report is not None:
        report(text)


@contextmanager
def cancellable(event: Event, progress: Callable[[str], None] | None = None):
    token, reporter = _token.set(event), _progress.set(progress)
    try:
        check_cancelled()
        yield
    finally:
        _progress.reset(reporter)
        _token.reset(token)
