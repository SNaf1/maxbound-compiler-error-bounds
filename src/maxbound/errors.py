"""Exceptions raised by maxbound."""


class UnsupportedModelError(Exception):
    """The model uses an operation or a structure that maxbound cannot analyse
    (for example a residual connection, LayerNorm or attention)."""


class NumericalInconsistency(RuntimeError):
    """A soundness guard failed: two enclosures that must overlap do not, or a
    value left the range where our rounding analysis is valid. This points to a
    bug or an extreme model, so we stop instead of returning a bound."""
