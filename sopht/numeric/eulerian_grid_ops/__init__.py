"""Eulerian grid operations."""

from .poisson_solver_2d.FastDiagPoissonSolver2D import FastDiagPoissonSolver2D
from .poisson_solver_2d.FFTPyFFTW2D import FFTPyFFTW2D
from .poisson_solver_2d.scipy_fft_2d import fft_ifft_via_scipy_kernel_2d
from .poisson_solver_2d.UnboundedPoissonSolverPYFFTW2D import UnboundedPoissonSolverPYFFTW2D
from .poisson_solver_3d.FastDiagPoissonSolver3D import FastDiagPoissonSolver3D
from .poisson_solver_3d.FFTPyFFTW3D import FFTPyFFTW3D
from .poisson_solver_3d.scipy_fft_3d import fft_ifft_via_scipy_kernel_3d
from .poisson_solver_3d.UnboundedPoissonSolverPYFFTW3D import UnboundedPoissonSolverPYFFTW3D
from .stencil_ops_2d.advection_flux_2d import gen_advection_flux_conservative_eno3_pyst_kernel_2d
from .stencil_ops_2d.advection_timestep_2d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
)
from .stencil_ops_2d.brinkmann_penalise_2d import (
    gen_brinkmann_penalise_pyst_kernel_2d,
    gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d,
)
from .stencil_ops_2d.char_func_from_level_set_2d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d,
)
from .stencil_ops_2d.diffusion_flux_2d import gen_diffusion_flux_pyst_kernel_2d
from .stencil_ops_2d.diffusion_timestep_2d import (
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
)
from .stencil_ops_2d.elementwise_ops_2d import (
    gen_add_fixed_val_pyst_kernel_2d,
    gen_elementwise_complex_product_pyst_kernel_2d,
    gen_elementwise_copy_pyst_kernel_2d,
    gen_elementwise_saxpby_pyst_kernel_2d,
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)
from .stencil_ops_2d.inplane_field_curl_2d import gen_inplane_field_curl_pyst_kernel_2d
from .stencil_ops_2d.outplane_field_curl_2d import gen_outplane_field_curl_pyst_kernel_2d
from .stencil_ops_2d.penalise_field_boundary_2d import gen_penalise_field_boundary_pyst_kernel_2d
from .stencil_ops_2d.update_vorticity_from_velocity_forcing_2d import (
    gen_update_vorticity_from_penalised_velocity_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
)
from .stencil_ops_3d.advection_flux_3d import gen_advection_flux_conservative_eno3_pyst_kernel_3d
from .stencil_ops_3d.advection_timestep_3d import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
)
from .stencil_ops_3d.brinkmann_penalise_3d import gen_brinkmann_penalise_pyst_kernel_3d
from .stencil_ops_3d.char_func_from_level_set_3d import (
    gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d,
)
from .stencil_ops_3d.curl_3d import gen_curl_pyst_kernel_3d
from .stencil_ops_3d.diffusion_flux_3d import gen_diffusion_flux_pyst_kernel_3d
from .stencil_ops_3d.diffusion_timestep_3d import (
    gen_diffusion_timestep_euler_forward_pyst_kernel_3d,
)
from .stencil_ops_3d.divergence_3d import gen_divergence_pyst_kernel_3d
from .stencil_ops_3d.elementwise_ops_3d import (
    gen_add_fixed_val_pyst_kernel_3d,
    gen_elementwise_complex_product_pyst_kernel_3d,
    gen_elementwise_copy_pyst_kernel_3d,
    gen_elementwise_cross_product_pyst_kernel_3d,
    gen_elementwise_saxpby_pyst_kernel_3d,
    gen_elementwise_sum_pyst_kernel_3d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)
from .stencil_ops_3d.laplacian_filter_3d import (
    gen_laplacian_filter_kernel_3d,
)
from .stencil_ops_3d.penalise_field_boundary_3d import gen_penalise_field_boundary_pyst_kernel_3d
from .stencil_ops_3d.update_vorticity_from_velocity_forcing_3d import (
    gen_update_vorticity_from_penalised_velocity_pyst_kernel_3d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d,
)
from .stencil_ops_3d.vorticity_stretching_flux_3d import (
    gen_vorticity_stretching_flux_pyst_kernel_3d,
)
from .stencil_ops_3d.vorticity_stretching_timestep_3d import (
    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d,
    gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d,
)

__all__ = [
    "FFTPyFFTW2D",
    "FFTPyFFTW3D",
    "FastDiagPoissonSolver2D",
    "FastDiagPoissonSolver3D",
    "UnboundedPoissonSolverPYFFTW2D",
    "UnboundedPoissonSolverPYFFTW3D",
    "fft_ifft_via_scipy_kernel_2d",
    "fft_ifft_via_scipy_kernel_3d",
    "gen_add_fixed_val_pyst_kernel_2d",
    "gen_add_fixed_val_pyst_kernel_3d",
    "gen_advection_flux_conservative_eno3_pyst_kernel_2d",
    "gen_advection_flux_conservative_eno3_pyst_kernel_3d",
    "gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d",
    "gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d",
    "gen_brinkmann_penalise_pyst_kernel_2d",
    "gen_brinkmann_penalise_pyst_kernel_3d",
    "gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d",
    "gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d",
    "gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d",
    "gen_curl_pyst_kernel_3d",
    "gen_diffusion_flux_pyst_kernel_2d",
    "gen_diffusion_flux_pyst_kernel_3d",
    "gen_diffusion_timestep_euler_forward_pyst_kernel_2d",
    "gen_diffusion_timestep_euler_forward_pyst_kernel_3d",
    "gen_divergence_pyst_kernel_3d",
    "gen_elementwise_complex_product_pyst_kernel_2d",
    "gen_elementwise_complex_product_pyst_kernel_3d",
    "gen_elementwise_copy_pyst_kernel_2d",
    "gen_elementwise_copy_pyst_kernel_3d",
    "gen_elementwise_cross_product_pyst_kernel_3d",
    "gen_elementwise_saxpby_pyst_kernel_2d",
    "gen_elementwise_saxpby_pyst_kernel_3d",
    "gen_elementwise_sum_pyst_kernel_2d",
    "gen_elementwise_sum_pyst_kernel_3d",
    "gen_inplane_field_curl_pyst_kernel_2d",
    "gen_laplacian_filter_kernel_3d",
    "gen_outplane_field_curl_pyst_kernel_2d",
    "gen_penalise_field_boundary_pyst_kernel_2d",
    "gen_penalise_field_boundary_pyst_kernel_3d",
    "gen_set_fixed_val_at_boundaries_pyst_kernel_2d",
    "gen_set_fixed_val_at_boundaries_pyst_kernel_3d",
    "gen_set_fixed_val_pyst_kernel_2d",
    "gen_set_fixed_val_pyst_kernel_3d",
    "gen_update_vorticity_from_penalised_velocity_pyst_kernel_2d",
    "gen_update_vorticity_from_penalised_velocity_pyst_kernel_3d",
    "gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d",
    "gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d",
    "gen_vorticity_stretching_flux_pyst_kernel_3d",
    "gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d",
    "gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d",
]
