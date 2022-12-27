"""Kernels for penalising field boundary in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable, Literal


def gen_penalise_field_boundary_pyst_kernel_3d(  # noqa: C901
    width: int,
    dx: float,
    x_grid_field: np.ndarray,
    y_grid_field: np.ndarray,
    z_grid_field: np.ndarray,
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D penalise field boundary kernel generator."""
    assert width >= 0 and isinstance(width, int), "invalid zone width"
    match width:
        case 0:
            # bypass option to prevent penalisation, done this way since by
            # default to avoid artifacts one must use penalisation...
            match field_type:
                case "scalar":

                    def penalise_field_boundary_pyst_kernel_3d(
                        field: np.ndarray,
                    ) -> None:
                        pass

                    return penalise_field_boundary_pyst_kernel_3d
                case "vector":

                    def penalise_vector_field_boundary_pyst_kernel_3d(
                        vector_field: np.ndarray,
                    ) -> None:
                        pass

                    return penalise_vector_field_boundary_pyst_kernel_3d

                case _:
                    raise ValueError("Invalid field type")

        case _:
            pyst_dtype = spu.get_pyst_dtype(real_t)
            grid_info = (
                f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
                if type(fixed_grid_size) is tuple[int, ...]
                else "3D"
            )
            x_grid_field_start = x_grid_field[0, 0, 0]
            y_grid_field_start = y_grid_field[0, 0, 0]
            z_grid_field_start = z_grid_field[0, 0, 0]
            x_grid_field_end = x_grid_field[0, 0, -1]
            y_grid_field_end = y_grid_field[0, -1, 0]
            z_grid_field_end = z_grid_field[-1, 0, 0]

            sine_prefactor = (np.pi / 2) / (width * dx)

            x_front_boundary_slice = ps.make_slice[:, :, :width]
            x_front_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=x_front_boundary_slice,
            )
            x_back_boundary_slice = ps.make_slice[:, :, -width:]
            x_back_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=x_back_boundary_slice,
            )

            @ps.kernel
            def penalise_field_x_front_boundary_stencil_3d():
                field, x_grid_field = ps.fields(
                    f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (x_grid_field[0, 0, 0] - x_grid_field_start)
                )

            penalise_field_x_front_boundary_kernel_3d = ps.create_kernel(
                penalise_field_x_front_boundary_stencil_3d,
                config=x_front_boundary_kernel_config,
            ).compile()

            @ps.kernel
            def penalise_field_x_back_boundary_stencil_3d():
                field, x_grid_field = ps.fields(
                    f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (x_grid_field_end - x_grid_field[0, 0, 0])
                )

            penalise_field_x_back_boundary_kernel_3d = ps.create_kernel(
                penalise_field_x_back_boundary_stencil_3d,
                config=x_back_boundary_kernel_config,
            ).compile()

            y_front_boundary_slice = ps.make_slice[:, :width, :]
            y_front_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=y_front_boundary_slice,
            )
            y_back_boundary_slice = ps.make_slice[:, -width:, :]
            y_back_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=y_back_boundary_slice,
            )

            @ps.kernel
            def penalise_field_y_front_boundary_stencil_3d():
                field, y_grid_field = ps.fields(
                    f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (y_grid_field[0, 0, 0] - y_grid_field_start)
                )

            penalise_field_y_front_boundary_kernel_3d = ps.create_kernel(
                penalise_field_y_front_boundary_stencil_3d,
                config=y_front_boundary_kernel_config,
            ).compile()

            @ps.kernel
            def penalise_field_y_back_boundary_stencil_3d():
                field, y_grid_field = ps.fields(
                    f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (y_grid_field_end - y_grid_field[0, 0, 0])
                )

            penalise_field_y_back_boundary_kernel_3d = ps.create_kernel(
                penalise_field_y_back_boundary_stencil_3d,
                config=y_back_boundary_kernel_config,
            ).compile()

            z_front_boundary_slice = ps.make_slice[:width, :, :]
            z_front_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=z_front_boundary_slice,
            )
            z_back_boundary_slice = ps.make_slice[-width:, :, :]
            z_back_boundary_kernel_config = spu.get_pyst_kernel_config(
                real_t,
                num_threads,
                iteration_slice=z_back_boundary_slice,
            )

            @ps.kernel
            def penalise_field_z_front_boundary_stencil_3d():
                field, z_grid_field = ps.fields(
                    f"field, z_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (z_grid_field[0, 0, 0] - z_grid_field_start)
                )

            penalise_field_z_front_boundary_kernel_3d = ps.create_kernel(
                penalise_field_z_front_boundary_stencil_3d,
                config=z_front_boundary_kernel_config,
            ).compile()

            @ps.kernel
            def penalise_field_z_back_boundary_stencil_3d():
                field, z_grid_field = ps.fields(
                    f"field, z_grid_field : {pyst_dtype}[{grid_info}]"
                )
                field[0, 0, 0] @= field[0, 0, 0] * sp.sin(
                    sine_prefactor * (z_grid_field_end - z_grid_field[0, 0, 0])
                )

            penalise_field_z_back_boundary_kernel_3d = ps.create_kernel(
                penalise_field_z_back_boundary_stencil_3d,
                config=z_back_boundary_kernel_config,
            ).compile()

            def penalise_field_boundary_pyst_kernel_3d(field: np.ndarray) -> None:
                """3D penalise field boundary kernel.

                Penalises field on the boundaries in a sine wave fashion
                in the given width in X, Y and Z direction
                field: field to be penalised
                """
                # first along X
                # these parts involve broadcasting hence couldn't be pystencilized
                field[:, :, :width] = field[:, :, (width - 1) : width]
                field[:, :, -width:] = field[:, :, -width : (-width + 1)]
                penalise_field_x_front_boundary_kernel_3d(
                    field=field, x_grid_field=x_grid_field
                )
                penalise_field_x_back_boundary_kernel_3d(
                    field=field, x_grid_field=x_grid_field
                )

                # then along Y
                # these parts involve broadcasting hence couldn't be pystencilized
                field[:, :width, :] = field[:, (width - 1) : width, :]
                field[:, -width:, :] = field[:, -width : (-width + 1), :]
                penalise_field_y_front_boundary_kernel_3d(
                    field=field, y_grid_field=y_grid_field
                )
                penalise_field_y_back_boundary_kernel_3d(
                    field=field, y_grid_field=y_grid_field
                )

                # then along Z
                # these parts involve broadcasting hence couldn't be pystencilized
                field[:width, :, :] = field[(width - 1) : width, :, :]
                field[-width:, :, :] = field[-width : (-width + 1), :, :]
                penalise_field_z_front_boundary_kernel_3d(
                    field=field, z_grid_field=z_grid_field
                )
                penalise_field_z_back_boundary_kernel_3d(
                    field=field, z_grid_field=z_grid_field
                )

            match field_type:
                case "scalar":
                    return penalise_field_boundary_pyst_kernel_3d
                case "vector":
                    x_axis_idx = spu.VectorField.x_axis_idx()
                    y_axis_idx = spu.VectorField.y_axis_idx()
                    z_axis_idx = spu.VectorField.z_axis_idx()

                    def penalise_vector_field_boundary_pyst_kernel_3d(
                        vector_field: np.ndarray,
                    ) -> None:
                        """3D penalise vector field boundary kernel.

                        Penalises vector field on the boundaries in a sine wave
                        fashion in the given width in X, Y and Z direction
                        vector_field: vector field to be penalised
                        """
                        penalise_field_boundary_pyst_kernel_3d(
                            field=vector_field[x_axis_idx],
                        )
                        penalise_field_boundary_pyst_kernel_3d(
                            field=vector_field[y_axis_idx],
                        )
                        penalise_field_boundary_pyst_kernel_3d(
                            field=vector_field[z_axis_idx],
                        )

                    return penalise_vector_field_boundary_pyst_kernel_3d
                case _:
                    raise ValueError("Invalid field type")
