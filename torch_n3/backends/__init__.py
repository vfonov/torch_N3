"""Choosing between the PyTorch blocks and the original C++ ones.

The pipeline is written once, against a set of block names, and either
implementation can supply them:

``"torch"``
    :mod:`torch_n3.blocks` -- the port, and the default.
``"legacy"``
    :mod:`torch_n3.backends.legacy` -- the original N3 C++ through a CFFI
    shim.  Slower, needs the extension built, and exists so that every block
    of the port has something exact to be tested against.

Neither is the installed ``nu_correct``.  Both run *this* pipeline, in one
process, on ``float64`` arrays; the original passes its intermediates between
programs as MINC files and is rounded at every step, which is why neither
backend reproduces it exactly.  See :mod:`torch_n3.backends.legacy`.
"""


def resolve(backend=None):
    """The module implementing ``backend``.

    Accepts a name, ``None`` for the default, or an already-imported module
    (returned unchanged), so callers can pass whichever they have.
    """
    if backend is None or backend == "torch":
        from torch_n3 import blocks
        return blocks
    if backend == "legacy":
        from torch_n3.backends import legacy
        return legacy
    if isinstance(backend, str):
        raise ValueError("unknown backend %r: expected 'torch' or 'legacy'"
                         % (backend,))
    return backend
