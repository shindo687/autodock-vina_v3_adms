"""Load ChainRules and provide an import-only fallback for minimal installs."""

try:  # The declared runtime dependency.
    import chainrules as _ad
except ImportError:  # pragma: no cover - exercised in dependency-free smoke tests
    from ._mini_chainrules import (  # type: ignore[no-redef]
        NonDifferentiablePoint,
        RuleNotFound,
        UnsupportedWrt,
        ZERO,
        grad,
        jvp,
        value_and_grad,
        vjp,
        rules,
    )
else:
    NonDifferentiablePoint = _ad.NonDifferentiablePoint
    RuleNotFound = _ad.RuleNotFound
    UnsupportedWrt = _ad.UnsupportedWrt
    ZERO = _ad.ZERO
    grad = _ad.grad
    jvp = _ad.jvp
    value_and_grad = _ad.value_and_grad
    vjp = _ad.vjp
    rules = _ad.rules
