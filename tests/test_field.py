"""Stage 2: :mod:`torch_n3.blocks.field` against ``correct_field``.

Unlike the other blocks this one cannot be matched to machine precision, and
should not be expected to.  The legacy sweeps its relaxation in raster order
in ``float``; sweeping the two checkerboard colours in turn is the same
iteration reordered, which is what makes it a tensor operation, and both are
approximations to the same Laplace solution rather than to each other.  So the
tests below check that the two agree to about the accuracy the solve itself
has, and that the properties the pipeline depends on hold exactly.
"""

import pytest
import torch

from tests.conftest import assert_close, span
from tests.inputs import tilted_plane
from torch_n3 import blocks
from torch_n3.backends import legacy


@pytest.fixture(scope="module")
def ramp_in_a_mask(chunk, chunk_mask):
    """A field defined only inside the brain mask, as the spline leaves it."""
    inside = chunk_mask.data != 0
    return tilted_plane(chunk, inside), inside


def test_matches_the_legacy_solver(chunk, ramp_in_a_mask):
    field, inside = ramp_in_a_mask

    ours = blocks.correct_field(field, inside, chunk.step)
    theirs = legacy.correct_field(field, inside, chunk.step)

    assert_close(ours, theirs, atol=1e-4 * span(theirs))


def test_matches_the_correct_field_binary(legacy_output, chunk, ramp_in_a_mask):
    """`nu_correct_field_1`: the installed program, on the same input."""
    field, inside = ramp_in_a_mask
    recorded = legacy_output["correct_field.chunk"]

    extended = blocks.correct_field(field, inside, chunk.step)

    assert_close(extended, recorded, atol=1e-4 * span(recorded))


def test_the_masked_values_are_left_alone(chunk, ramp_in_a_mask):
    """The extension is only allowed to invent values it has none for."""
    field, inside = ramp_in_a_mask

    extended = blocks.correct_field(field, inside, chunk.step)

    assert_close(extended[inside], field[inside], atol=0.0)


def test_the_extension_stays_within_the_range_it_was_given(chunk,
                                                           ramp_in_a_mask):
    """A harmonic function attains its extrema on the boundary.

    That is the property ``nu_evaluate`` relies on: however far outside the
    head a voxel is, the field there cannot run away and make the division
    explode.
    """
    field, inside = ramp_in_a_mask

    extended = blocks.correct_field(field, inside, chunk.step)

    assert float(extended.min()) >= float(field[inside].min()) - 1e-9
    assert float(extended.max()) <= float(field[inside].max()) + 1e-9


def test_a_constant_field_extends_to_the_same_constant():
    """The one case with an exact answer, so it is worth being exact about."""
    shape = (16, 16, 16)
    inside = torch.zeros(shape, dtype=torch.bool)
    inside[6:10, 6:10, 6:10] = True
    field = torch.where(inside, torch.full(shape, 2.5), torch.zeros(shape))

    extended = blocks.correct_field(field, inside, (1.0, 1.0, 1.0))

    assert_close(extended, torch.full(shape, 2.5), atol=1e-6)
