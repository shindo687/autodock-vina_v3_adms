"""Tiny protocol-compatible fallback used only when chainrules is unavailable."""

class _Zero:
    def __repr__(self): return "ZERO"

ZERO = _Zero()

class RuleNotFound(LookupError): pass
class UnsupportedWrt(ValueError): pass
class NonDifferentiablePoint(RuntimeError): pass

_JVP, _VJP = {}, {}

class _Rules:
    def jvp_for(self, fn):
        def deco(rule): _JVP[id(fn)] = (fn, rule); return rule
        return deco
    def vjp_for(self, fn):
        def deco(rule): _VJP[id(fn)] = (fn, rule); return rule
        return deco
rules = _Rules()

def _names(wrt): return (wrt,) if isinstance(wrt, str) else tuple(wrt)
def jvp(fn, *args, tangents, **kwargs):
    if not tangents or all(v is ZERO for v in tangents.values()):
        return fn(*args, **kwargs), ZERO
    entry = _JVP.get(id(fn))
    if entry is None or entry[0] is not fn: raise RuleNotFound(fn, "JVP")
    return entry[1](tangents, *args, **kwargs)
def vjp(fn, *args, wrt, **kwargs):
    names = _names(wrt); entry = _VJP.get(id(fn))
    if entry is None or entry[0] is not fn: raise RuleNotFound(fn, "VJP")
    value, pb = entry[1](names, *args, **kwargs)
    def pullback(c):
        if c is ZERO: return {n: ZERO for n in names}
        out = pb(c)
        if set(out) != set(names): raise TypeError("pullback keys must match wrt")
        return {n: out[n] for n in names}
    return value, pullback
def grad(fn, *args, wrt, **kwargs): return vjp(fn, *args, wrt=wrt, **kwargs)[1](1.0)
def value_and_grad(fn, *args, wrt, **kwargs):
    value, pb = vjp(fn, *args, wrt=wrt, **kwargs); return value, pb(1.0)
