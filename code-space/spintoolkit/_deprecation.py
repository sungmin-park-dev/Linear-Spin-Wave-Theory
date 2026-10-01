"""Deprecation policy of the pre-0.2 API (D30, D43).

The ``SpinSystem``-based API (``SpinSystem``, ``LSWTSolver``,
``SpinOptimizer``, ``EnergyFunction``, the SI Hall method and the ``lswt``
package alias) keeps working in 0.2 with a DeprecationWarning and is removed
in :data:`REMOVAL_VERSION`. Code of the new API that still builds a
``SpinSystem`` internally does so inside :func:`internal_use`, so users of the
new API see no warning.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import warnings

#: First release without the deprecated API.
REMOVAL_VERSION = "0.3"

_INTERNAL: ContextVar[bool] = ContextVar("spintoolkit_internal_use", default=False)


def warn_deprecated(what: str, replacement: str, stacklevel: int = 3) -> None:
    """Emit the DeprecationWarning for ``what``, naming its ``replacement``."""
    if _INTERNAL.get():
        return
    warnings.warn(f"{what} is deprecated (D30) and will be removed in spintoolkit "
                  f"{REMOVAL_VERSION}; {replacement}", DeprecationWarning, stacklevel=stacklevel)


@contextmanager
def internal_use():
    """Silence :func:`warn_deprecated` for calls made by the package itself."""
    token = _INTERNAL.set(True)
    try:
        yield
    finally:
        _INTERNAL.reset(token)
