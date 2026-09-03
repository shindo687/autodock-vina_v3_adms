"""Dependency-free subset of the ChainRules 0.1.0 call protocol."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping


class _Zero:
    def __repr__(self) -> str:
        return "ZERO"


ZERO = _Zero()


def _function_name(function):
    return getattr(function, "__qualname__", getattr(function, "__name__", repr(function)))


class RuleNotFound(LookupError):
    pass


class UnsupportedWrt(ValueError):
    """Match ChainRules' contextual unsupported-active-input exception."""

    def __init__(self, function, requested: Iterable[str], *, supported: Iterable[str] | None = None):
        self.function = function
        self.requested = tuple(sorted(requested))
        self.supported = None if supported is None else tuple(sorted(supported))
        message = (
            f"{_function_name(function)} does not support differentiation "
            f"with respect to {self.requested!r}"
        )
        if self.supported is not None:
            message += f"; supported inputs are {self.supported!r}"
        super().__init__(message)


class NonDifferentiablePoint(RuntimeError):
    pass


_JVP, _VJP = {}, {}


class _Rules:
    def jvp_for(self, function):
        def decorator(rule):
            _JVP[id(function)] = (function, rule)
            return rule

        return decorator

    def vjp_for(self, function):
        def decorator(rule):
            _VJP[id(function)] = (function, rule)
            return rule

        return decorator


rules = _Rules()


def _bind(function, args, kwargs):
    signature = inspect.signature(function)
    signature.bind(*args, **kwargs)
    return signature


def _validate_names(signature, names, *, label):
    unknown = sorted(set(names) - set(signature.parameters))
    if unknown:
        raise TypeError(f"Unknown {label} parameter name(s): {unknown!r}")


def _names(wrt):
    if isinstance(wrt, str):
        return (wrt,)
    try:
        names = tuple(wrt)
    except TypeError as exc:
        raise TypeError("wrt must be a parameter name or iterable of names") from exc
    if not names or any(not isinstance(name, str) for name in names):
        raise TypeError("wrt must contain one or more string parameter names")
    if len(set(names)) != len(names):
        raise TypeError("wrt parameter names must be unique")
    return names


def jvp(function, *args, tangents, **kwargs):
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    signature = _bind(function, args, kwargs)
    if any(not isinstance(name, str) for name in tangents):
        raise TypeError("every tangent key must be a string parameter name")
    _validate_names(signature, tangents, label="tangent")
    active_tangents = dict(tangents)
    if not active_tangents or all(value is ZERO for value in active_tangents.values()):
        return function(*args, **kwargs), ZERO
    entry = _JVP.get(id(function))
    if entry is None or entry[0] is not function:
        raise RuleNotFound(function, "JVP")
    return entry[1](active_tangents, *args, **kwargs)


def vjp(function, *args, wrt, **kwargs):
    names = _names(wrt)
    # ChainRules lets the registered rule classify unsupported active inputs;
    # this preserves its contextual ``UnsupportedWrt`` error instead of
    # converting it into a constructor/signature TypeError.
    _bind(function, args, kwargs)
    entry = _VJP.get(id(function))
    if entry is None or entry[0] is not function:
        raise RuleNotFound(function, "VJP")
    value, candidate = entry[1](names, *args, **kwargs)
    if not callable(candidate):
        raise TypeError("A VJP rule must return a callable pullback")

    def pullback(cotangent):
        if cotangent is ZERO:
            return dict.fromkeys(names, ZERO)
        result = candidate(cotangent)
        if not isinstance(result, Mapping):
            raise TypeError("A pullback must return a mapping keyed by wrt names")
        if set(result) != set(names):
            missing = sorted(set(names) - set(result))
            extra = sorted(set(result) - set(names))
            raise TypeError(
                "Pullback keys must exactly match wrt; "
                f"missing={missing!r}, extra={extra!r}"
            )
        return {name: result[name] for name in names}

    return value, pullback


def grad(function, *args, wrt, **kwargs):
    return vjp(function, *args, wrt=wrt, **kwargs)[1](1.0)


def value_and_grad(function, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    return value, pullback(1.0)
