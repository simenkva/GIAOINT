"""Internal helpers for staged public-API deprecations."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    *, replacement: str, removal: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a callable for removal under the documented compatibility policy."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        message = (
            f"{function.__qualname__} is deprecated; use {replacement}. "
            f"Removal is planned for {removal}."
        )

        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, FutureWarning, stacklevel=2)
            return function(*args, **kwargs)

        wrapped.__deprecated__ = message  # type: ignore[attr-defined]
        return wrapped

    return decorate
