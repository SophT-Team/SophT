"""Stencil based grid operations in 2D."""
from .advection_flux_2d import gen_advection_flux_conservative_eno3_pyst_kernel_2d
from .advection_timestep_2d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
)
from .brinkmann_penalise_2d import (
    gen_brinkmann_penalise_pyst_kernel_2d,
    gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d,
)
from .char_func_from_level_set_2d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d,
)
from .diffusion_flux_2d import gen_diffusion_flux_pyst_kernel_2d
from .diffusion_timestep_2d import gen_diffusion_timestep_euler_forward_pyst_kernel_2d
from .elementwise_ops_2d import (
    gen_add_fixed_val_pyst_kernel_2d,
    gen_elementwise_complex_product_pyst_kernel_2d,
    gen_elementwise_copy_pyst_kernel_2d,
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
    gen_elementwise_saxpby_pyst_kernel_2d,
)
from .inplane_field_curl_2d import gen_inplane_field_curl_pyst_kernel_2d
from .outplane_field_curl_2d import gen_outplane_field_curl_pyst_kernel_2d
from .penalise_field_boundary_2d import gen_penalise_field_boundary_pyst_kernel_2d
from .update_vorticity_from_velocity_forcing_2d import (
    gen_update_vorticity_from_penalised_velocity_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
)
