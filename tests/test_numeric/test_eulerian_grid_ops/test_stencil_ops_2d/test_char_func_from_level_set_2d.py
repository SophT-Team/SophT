import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def char_func_from_level_set_via_sine_heaviside_reference(
    char_func_field,
    level_set_field,
    blend_width,
    real_t,
):
    char_func_field[...] = 0
    char_func_field[...] = char_func_field + (level_set_field >= blend_width)
    char_func_field[...] = char_func_field + (
        np.fabs(level_set_field) < blend_width
    ) * real_t(0.5) * (
        1
        + level_set_field / blend_width
        + np.sin(np.pi * level_set_field / blend_width) / np.pi
    )


class CharFuncFromLevelSetFuncSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.level_set_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.dx = real_t(0.1)
        self.blend_width = 2 * self.dx
        # later can add variations here...
        self.ref_char_func_field = np.zeros_like(self.level_set_field)
        char_func_from_level_set_via_sine_heaviside_reference(
            char_func_field=self.ref_char_func_field,
            level_set_field=self.level_set_field,
            blend_width=self.blend_width,
            real_t=real_t,
        )

    def check_equals(self, char_func_field):
        np.testing.assert_allclose(
            self.ref_char_func_field,
            char_func_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_char_func_from_level_set_via_sine_heaviside_pyst_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = CharFuncFromLevelSetFuncSolution(n_values, precision)
    char_func_field = np.zeros_like(solution.ref_char_func_field)
    char_func_from_level_set_via_sine_heaviside_pyst_kernel = (
        gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d(
            blend_width=solution.blend_width,
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    char_func_from_level_set_via_sine_heaviside_pyst_kernel(
        char_func_field=char_func_field, level_set_field=solution.level_set_field
    )
    solution.check_equals(char_func_field)
