"""Stencil based grid operations in 3D."""
from .advection_flux_3d import gen_advection_flux_conservative_eno3_pyst_kernel_3d
from .advection_timestep_3d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
)
from .brinkmann_penalise_3d import gen_brinkmann_penalise_pyst_kernel_3d
from .char_func_from_level_set_3d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d,
)
from .curl_3d import gen_curl_pyst_kernel_3d
from .diffusion_flux_3d import gen_diffusion_flux_pyst_kernel_3d
from .diffusion_timestep_3d import gen_diffusion_timestep_euler_forward_pyst_kernel_3d
from .elementwise_ops_3d import (
    gen_add_fixed_val_pyst_kernel_3d,
    gen_elementwise_complex_product_pyst_kernel_3d,
    gen_elementwise_copy_pyst_kernel_3d,
    gen_elementwise_sum_pyst_kernel_3d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
    gen_elementwise_saxpby_pyst_kernel_3d,
    gen_elementwise_cross_product_pyst_kernel_3d,
)
from .penalise_field_boundary_3d import gen_penalise_field_boundary_pyst_kernel_3d
from .update_vorticity_from_velocity_forcing_3d import (
    gen_update_vorticity_from_penalised_velocity_pyst_kernel_3d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d,
)
from .vorticity_stretching_flux_3d import gen_vorticity_stretching_flux_pyst_kernel_3d
from .vorticity_stretching_timestep_3d import (
    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d,
    gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d,
)
from .laplacian_filter_3d import (
    gen_laplacian_filter_kernel_3d,
)
from .divergence_3d import gen_divergence_pyst_kernel_3d
from .update_passive_field_from_forcing_3d import (
    gen_update_passive_field_from_forcing_pyst_kernel_3d,
)
