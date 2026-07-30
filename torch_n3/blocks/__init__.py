"""N3's blocks, in PyTorch.

This package is the port proper.  Each module is one step of the algorithm,
named for the mathematics it performs, and each is pinned against the original
C++ (through :mod:`torch_n3.backends.legacy`) by a test of its own:

============================  ==========================================
:mod:`~torch_n3.blocks.histogram`  the masked histogram, ``volume_hist``
:mod:`~torch_n3.blocks.sharpen`    the sharpened mapping, ``sharpen_hist``
:mod:`~torch_n3.blocks.spline`     the smooth field fit, ``spline_smooth``
:mod:`~torch_n3.blocks.field`      the field extension, ``correct_field``
============================  ==========================================

Everything takes and returns ``torch`` tensors and stays on whatever device
those tensors live on.  The names below are exactly the ones
:mod:`torch_n3.backends.legacy` exports, so the two are interchangeable:
:func:`torch_n3.backends.resolve` picks between them, and the pipeline is
written against the pair.
"""

from torch_n3.blocks.field import correct_field
from torch_n3.blocks.histogram import bin_centers, histogram, histogram_range
from torch_n3.blocks.sharpen import sharpen_lut
from torch_n3.blocks.spline import BSplineField

__all__ = ["bin_centers", "histogram", "histogram_range", "sharpen_lut",
           "BSplineField", "correct_field"]
