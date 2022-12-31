"""Module for Input/Output via HD5 format."""
import h5py
import numpy as np
from elastica.rod.cosserat_rod import CosseratRod


class IO:
    r"""IO class for field save and load.

    Currently, the XDMF files are written for visualization in Paraview.

    Attributes
    ----------
    dim: int
        Integer for specifying dimension of problem (2D or 3D only)
    real_dtype: data type
        Data type for typecasting real values and setting precision of XDMF description file.

    The HDF5 file roughly follows the hierarchy below:
                    __________________/root___________________
                   /                                          \
             Eulerian                               _______Lagrangian_______
            /   |                                  /                        \
          /     |     \                        Grid_1    ..............    Grid_N
     Scalar   Vector  Parameters              /  |  \                      /  |  \
      /|\       /|\                         /   |    \                    /   |   \
    (Fields)  (Fields)                Scalar Vector  Grid            Scalar Vector Grid
                                    /|\       /|\                   /|\       /|\
                                (Fields)   (Fields)             (Fields)   (Fields)
    """

    def __init__(self, dim: int, real_dtype: type = np.float64) -> None:
        """Class initializer."""
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "Invalid dimension (only 2D and 3D)"
        self.real_dtype = real_dtype
        self.precision = 8 if real_dtype is np.float32 else 16

        # Initialize dictionaries for fields for IO and their
        # corresponding field_type ('Scalar' or 'Vector') Eulerian grid
        self.eulerian_grid_defined = False
        self.eulerian_fields: dict = {}
        self.eulerian_fields_type: dict = {}

        # Lagrangian grid
        self.lagrangian_fields: dict = {}
        self.lagrangian_fields_type: dict = {}
        self.lagrangian_grids: dict = {}
        self.lagrangian_fields_with_grid_name: dict = {}
        self.lagrangian_grid_count = 0
        self.lagrangian_grid_connection: dict = {}

    def define_eulerian_grid(
        self,
        origin: np.ndarray,
        dx: np.ndarray,
        grid_size: np.ndarray,
    ) -> None:
        """
        Define the Eulerian grid mesh.

        Attributes
        ----------
        origin: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing origin position (min of coordinate values) in z-y-x ordering
        dx: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing dx in each dimension following z-y-x ordering.
        grid_size: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Array containing grid_size in each dimension following z-y-x ordering.
        """
        assert isinstance(origin, np.ndarray)
        assert isinstance(dx, np.ndarray)
        assert isinstance(grid_size, np.ndarray)
        self.eulerian_origin = origin
        self.eulerian_dx = dx
        self.eulerian_grid_size = grid_size  # z,y,x
        self.eulerian_grid_defined = True

    def add_as_eulerian_fields_for_io(self, **fields_for_io) -> None:
        """Add Eulerian fields to be saved/loaded.

        Eulerian grid needs to be defined first using `define_eulerian_grid(...)` call.

        Attributes
        ----------
        **fields_for_io: keyword arguments used for storing eulerian fields to file.

        Each field will be saved to the output file with its corresponding
        keyword name, similar to numpy savez function.
        https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
        """
        assert self.eulerian_grid_defined, "Eulerian mesh is not defined!"

        for field_name in fields_for_io:
            # Add each field into local dictionary
            field = fields_for_io[field_name]
            self.eulerian_fields[field_name] = field

            # Assign field types
            if field.shape == (*self.eulerian_grid_size,):
                self.eulerian_fields_type[field_name] = "Scalar"
            elif field.shape == (self.dim, *self.eulerian_grid_size):
                self.eulerian_fields_type[field_name] = "Vector"
            else:
                raise ValueError(
                    f"Unable to identify eulerian field type "
                    f"(scalar / vector) based on field dimension {field.shape}"
                )

    def add_as_lagrangian_fields_for_io(
        self,
        lagrangian_grid: np.ndarray,
        lagrangian_grid_name: str = None,
        lagrangian_grid_connect: bool = False,
        **fields_for_io,
    ) -> None:
        """
        Add lagrangian fields to be saved/loaded.

        Attributes
        ----------
        lagrangian_grid: numpy.ndarray
            2D (dim, N) array containing data with 'float' type.
            Array containing lagrangian grid positions.
        lagrangian_grid_name: str
            Optional naming for the lagrangian grid used by added
            fields to identify which grid they belong to.
            Otherwise default naming is used with lagrangian_grid_count.
        **fields_for_io: keyword arguments used for storing lagrangian fields to file.

        Each field will be saved to the output file with its corresponding
        keyword name, similar to numpy savez function.
        https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
        """
        assert (
            len(lagrangian_grid.shape) == 2
        ), "lagrangian grid has to be a 2D (dim, N) array."
        assert (
            lagrangian_grid.shape[0] == self.dim
        ), "Invalid lagrangian grid dimension (only 2D and 3D)"

        if lagrangian_grid_name is None:
            lagrangian_grid_name = f"Lagrangian_grid_{self.lagrangian_grid_count}"
            self.lagrangian_grid_count += 1

        if lagrangian_grid_connect:
            self.lagrangian_grid_connection[lagrangian_grid_name] = np.arange(
                lagrangian_grid.shape[1]
            )

        # Save `lagrangian_grid` with grid name `lagrangian_grid_name` to `lagrangian_grids`
        self.lagrangian_grids[lagrangian_grid_name] = lagrangian_grid
        # Create a list of to store names of fields that lie on
        # grid with grid name `lagrangian_grid_name`
        self.lagrangian_fields_with_grid_name[lagrangian_grid_name] = []
        for field_name in fields_for_io:
            # Add each field into local dictionary
            field = fields_for_io[field_name]
            self.lagrangian_fields[field_name] = field
            self.lagrangian_fields_with_grid_name[lagrangian_grid_name].append(
                field_name
            )

            # Assign field types
            if field.shape[0] == lagrangian_grid.shape[1]:
                self.lagrangian_fields_type[field_name] = "Scalar"
            elif field.shape == lagrangian_grid.shape:
                self.lagrangian_fields_type[field_name] = "Vector"
            else:
                raise ValueError(
                    f"Unable to identify lagrangian field type "
                    f"(scalar / vector) based on field dimension {field.shape}"
                )

    def save(self, h5_file_name: str, time: float = 0.0) -> None:  # noqa: C901
        """
        This is a wrapper function to call _save function.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """

        self._save(h5_file_name, time)

    def _save(self, h5_file_name: str, time: float = 0.0) -> None:  # noqa: C901
        """
        Save added fields to hdf5 file.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """
        # 1. Create hdf5 file.
        # 2. Initialize groups for Eulerian and Lagrangian grids.
        # 3. For Eulerian group, initialize groups for scalar and
        #    vector fields (grid is already defined).
        # 4. For Lagrangian group, initialize individual groups for different lagrangian grids.
        #    In each of the lagrangian grid group, store the grid information and initialize
        #    groups for scalar and vector field.
        # 5. Go over the fields in the dictionary and save them in their corresponding location.

        with h5py.File(h5_file_name, "w") as f:
            # Save time stamp
            f.attrs["time"] = time

            # Eulerian save
            if self.eulerian_grid_defined:
                eulerian_grp = f.create_group("Eulerian")
                # 'Scalar' and 'Vector' fields
                eulerian_scalar_grp = eulerian_grp.create_group("Scalar")
                eulerian_vector_grp = eulerian_grp.create_group("Vector")
                # Go over and save all fields that lie on the common eulerian grid
                # Note : Paraview renders 2D eulerian field on the YZ plane, inconsistently with
                # lagrangian fields rendered on XY plane. This could be a bug in Paraview and
                # I have opened up a question on Paraview's official forum
                # https://discourse.paraview.org/t/2dcorectmesh-displaying-on-yz-plane-instead-of-xy-plane/9535
                # As a workaround, here we extend the dimension of the 2D field, so that
                # it appears as a slice in a 3D space, and Paraview can
                # correctly render the field on the XY plane. We can remove the
                # reshaping when Paraview resolve the issue.
                for field_name in self.eulerian_fields:
                    field = self.eulerian_fields[field_name]
                    field_type = self.eulerian_fields_type[field_name]
                    if field_type == "Scalar":
                        eulerian_scalar_grp.create_dataset(
                            field_name,
                            data=field.reshape(1, *self.eulerian_grid_size),
                        )
                    elif field_type == "Vector":
                        # Decompose vector fields into individual component as scalar fields
                        for idx_dim in range(self.dim):
                            eulerian_vector_grp.create_dataset(
                                f"{field_name}_{idx_dim}",
                                data=field[idx_dim, ...].reshape(
                                    1, *self.eulerian_grid_size
                                ),
                            )
                    else:
                        raise ValueError(
                            "Unsupported eulerian_field_type ('Scalar' and 'Vector' only)"
                        )
                # Save eulerian simulation parameters
                eulerian_params_grp = eulerian_grp.create_group("Parameters")
                eulerian_params_grp.attrs["origin"] = self.eulerian_origin
                eulerian_params_grp.attrs["dx"] = self.eulerian_dx
                eulerian_params_grp.attrs["grid_size"] = self.eulerian_grid_size

            # Lagrangian save
            # Note: We need to reverse the order from (dim, ...) -> (..., dim) for Paraview.
            # For eulerian fields, we mitigate this by splitting each vector
            # component into scalar fields. For lagrangian fields, since N is small
            # compared to the N in Eulerian grid, I have decided to stick with the
            # tranpose/moveaxis approach for now. This pays off later as convenience
            # during post-processing and visualizing these lagrangian points in Paraview.
            lagrangian_grp = f.create_group("Lagrangian")
            # Go over all lagrangian grids
            for lagrangian_grid_name in self.lagrangian_grids:
                lagrangian_grid_grp = lagrangian_grp.create_group(lagrangian_grid_name)
                lagrangian_grid = self.lagrangian_grids[lagrangian_grid_name]
                lagrangian_grid_grp.create_dataset(
                    "Grid", data=np.transpose(lagrangian_grid)
                )
                if lagrangian_grid_name in self.lagrangian_grid_connection:
                    lagrangian_grid_grp.create_dataset(
                        "Connection",
                        data=self.lagrangian_grid_connection[lagrangian_grid_name],
                    )
                # 'Scalar' and 'Vector' fields
                lagrangian_scalar_grp = lagrangian_grid_grp.create_group("Scalar")
                lagrangian_vector_grp = lagrangian_grid_grp.create_group("Vector")
                # Go over and save all fields that lie on the current lagrangian grid
                for field_name in self.lagrangian_fields_with_grid_name[
                    lagrangian_grid_name
                ]:
                    field = self.lagrangian_fields[field_name]
                    field_type = self.lagrangian_fields_type[field_name]
                    if field_type == "Scalar":
                        lagrangian_scalar_grp.create_dataset(field_name, data=field)
                    elif field_type == "Vector":
                        lagrangian_vector_grp.create_dataset(
                            field_name, data=np.moveaxis(field, 0, -1)
                        )
                    else:
                        raise ValueError(
                            "Unsupported lagrangian_field_type ('Scalar' and 'Vector' only)"
                        )

        # Generate xdmf files for every grid
        if self.eulerian_fields:
            self.generate_xdmf_eulerian(h5_file_name=h5_file_name, time=time)
        if self.lagrangian_fields:
            self.generate_xdmf_lagrangian(h5_file_name=h5_file_name, time=time)

    def load(self, h5_file_name: str) -> None:  # noqa: C901
        """Load fields from hdf5 file.

        Field arrays need to be allocated and added to `eulerian_fields` and/or
        `lagrangian_fields` for proper loading and field recovery.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the hdf5 file.
        """
        with h5py.File(h5_file_name, "r") as f:
            keys: list = []
            f.visit(keys.append)

            # Load time
            time = f.attrs["time"]

            # Load Eulerian fields
            if self.eulerian_fields:
                assert self.eulerian_grid_defined, "Eulerian grid is not defined!"
                for field_name in self.eulerian_fields:
                    field_type = self.eulerian_fields_type[field_name]
                    if field_type == "Scalar":
                        assert (
                            f"Eulerian/{field_type}/{field_name}" in keys
                        ), f"Unable to find scalar field {field_name} in loaded file!"
                        # zero indexing here to account for additional dimension
                        # added during save
                        self.eulerian_fields[field_name][...] = f["Eulerian"][
                            field_type
                        ][field_name][0, ...]
                    elif field_type == "Vector":
                        for idx_dim in range(self.dim):
                            assert (
                                f"Eulerian/{field_type}/{field_name}_{idx_dim}" in keys
                            ), (
                                f"Unable to find vector field {field_name}_{idx_dim} "
                                f"in loaded file!"
                            )
                            # zero indexing here to account for additional dimension
                            # added during save
                            self.eulerian_fields[field_name][idx_dim, ...] = f[
                                "Eulerian"
                            ][field_type][f"{field_name}_{idx_dim}"][0, ...]
                    else:
                        raise ValueError(
                            "Unsupported lagrangian_field_type ('Scalar' and 'Vector' only)"
                        )

                # Load 'Parameters' and assert equals for restart consistency
                np.testing.assert_allclose(
                    self.eulerian_origin,
                    f["Eulerian"]["Parameters"].attrs["origin"],
                )
                np.testing.assert_allclose(
                    self.eulerian_dx, f["Eulerian"]["Parameters"].attrs["dx"]
                )
                np.testing.assert_allclose(
                    self.eulerian_grid_size,
                    f["Eulerian"]["Parameters"].attrs["grid_size"],
                )

            # Load Lagrangian fields
            if self.lagrangian_fields:
                # First loop over and load each of the lagrangian grids
                for lagrangian_grid_name in self.lagrangian_grids:
                    assert (
                        f"Lagrangian/{lagrangian_grid_name}/Grid" in keys
                    ), f"Unable to find grid '{lagrangian_grid_name}' in loaded file!"
                    self.lagrangian_grids[lagrangian_grid_name][...] = np.transpose(
                        f["Lagrangian"][lagrangian_grid_name]["Grid"][...]
                    )

                    if f"Lagrangian/{lagrangian_grid_name}/Connection" in keys:
                        self.lagrangian_grid_connection[lagrangian_grid_name] = f[
                            "Lagrangian"
                        ][lagrangian_grid_name]["Connection"][...]

                    # Load all the fields living on the current lagrangian grid
                    for field_name in self.lagrangian_fields_with_grid_name[
                        lagrangian_grid_name
                    ]:
                        field_type = self.lagrangian_fields_type[field_name]
                        if field_type == "Scalar":
                            assert (
                                f"Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}"
                                in keys
                            ), (
                                f"Unable to find scalar field {field_name} on "
                                f"grid {lagrangian_grid_name} in loaded file!"
                            )
                            self.lagrangian_fields[field_name][...] = f["Lagrangian"][
                                lagrangian_grid_name
                            ][field_type][field_name][...]
                        elif field_type == "Vector":
                            assert (
                                f"Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}"
                                in keys
                            ), (
                                f"Unable to find vector field {field_name} on grid "
                                f"{lagrangian_grid_name} in loaded file!"
                            )
                            self.lagrangian_fields[field_name][...] = np.moveaxis(
                                f["Lagrangian"][lagrangian_grid_name][field_type][
                                    field_name
                                ][...],
                                -1,
                                0,
                            )
                        else:
                            raise ValueError(
                                "Unsupported lagrangian_field_type "
                                "('Scalar' and 'Vector' only)"
                            )

        return time

    def generate_xdmf_eulerian(self, h5_file_name: str, time: float = 0.0) -> None:
        """Generate XDMF description file for Eulerian fields.

        Currently, the XDMF file is generated for Paraview visualization only.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the corresponding hdf5 file.
        time: real_dtype
            Time at which the fields are saved.

        """
        # We use 3DCORECTMESH for both 2D and 3D fields since paraview
        # does not render on XY plane in 2DCORECTMESH
        topology_type = "3DCORECTMesh"
        geometry_type = "ORIGIN_DXDYDZ"

        grid_size = (
            self.eulerian_grid_size
            if self.dim == 3
            else np.insert(self.eulerian_grid_size, 0, 1)
        )
        origin = (
            self.eulerian_origin
            if self.dim == 3
            else np.insert(self.eulerian_origin, 0, 0.0)
        )
        dx = self.eulerian_dx if self.dim == 3 else np.insert(self.eulerian_dx, 0, 0.0)

        grid_size_string = np.array2string(
            grid_size, precision=self.precision, separator="    "
        )[1:-1]
        origin_string = np.array2string(
            origin, precision=self.precision, separator="    "
        )[1:-1]
        dx_string = np.array2string(dx, precision=self.precision, separator="    ")[
            1:-1
        ]

        def generate_field_entry(file_name, field_name, field_type):
            if field_type == "Scalar":
                entry = f"""<Attribute Name="{field_name}" Active="1"
                AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{grid_size_string}"
                    NumberType="Float" Precision="{self.precision}" Format="HDF">
                        {file_name}:/Eulerian/{field_type}/{field_name}
                    </DataItem>
                </Attribute>

                """
            elif field_type == "Vector":
                entry = ""
                for idx_dim in range(self.dim):
                    entry += f"""<Attribute Name="{field_name}_{idx_dim}"
                    Active="1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{grid_size_string}" NumberType="Float"
                    Precision="{self.precision}" Format="HDF">
                        {file_name}:/Eulerian/{field_type}/{field_name}_{idx_dim}
                    </DataItem>
                </Attribute>

                """
            return entry

        field_entries = ""
        for field_name in self.eulerian_fields:
            field_type = self.eulerian_fields_type[field_name]
            field_entries += generate_field_entry(h5_file_name, field_name, field_type)

        xdmffile = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
    <Domain>
        <Grid GridType="Uniform">
            <Time Value="{time}"/>
            <Topology TopologyType="{topology_type}" Dimensions="{grid_size_string}"/>
            <Geometry GeometryType="{geometry_type}">
                <DataItem Name="Origin" Dimensions="{self.dim}"
                NumberType="Float" Precision="{self.precision}" Format="XML">
                    {origin_string}
                </DataItem>
                <DataItem Name="Spacing" Dimensions="{self.dim}"
                NumberType="Float" Precision="{self.precision}" Format="XML">
                    {dx_string}
                </DataItem>
            </Geometry>

            {field_entries}
        </Grid>
    </Domain>
</Xdmf>
"""
        with open(h5_file_name.replace(".h5", "_eulerian.xmf"), "w") as f:
            f.write(xdmffile)

    def generate_xdmf_lagrangian(self, h5_file_name: str, time: float) -> None:
        """Generate XDMF description file for Lagrangian fields.

        Currently, the XDMF file is generated for Paraview visualization only.

        Attributes
        ----------
        h5_file_name: str
            String containing name of the corresponding hdf5 file.
        time: real_dtype
            Time at which the fields are saved.
        """
        geometry_type = "XYZ" if self.dim == 3 else "XY"

        def generate_lagrangian_field_entry(
            h5_file_name,
            field_name,
            field_type,
            field_grid_size,
            lagrangian_grid_name,
        ):
            entry = f"""<Attribute Name="{field_name}" Active="1"
            AttributeType="{field_type}" Center="Node">
                <DataItem Dimensions="{field_grid_size}" NumberType="Float"
                Precision="{self.precision}" Format="HDF">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/{field_type}/{field_name}
                </DataItem>
            </Attribute>

                """
            return entry

        for lagrangian_grid_name in self.lagrangian_grids:
            xmf_file_name = h5_file_name.replace(".h5", f"_{lagrangian_grid_name}.xmf")
            field_entries = ""
            lagrangian_grid = self.lagrangian_grids[lagrangian_grid_name]
            lagrangian_grid_size = np.flip(np.array(lagrangian_grid.shape))
            lagrangian_grid_size_string = np.array2string(
                lagrangian_grid_size,
                precision=self.precision,
                separator="    ",
            )[1:-1]

            for field_name in self.lagrangian_fields_with_grid_name[
                lagrangian_grid_name
            ]:
                field_type = self.lagrangian_fields_type[field_name]

                if field_type == "Scalar":
                    field_grid_size_string = lagrangian_grid_size[0]
                elif field_type == "Vector":
                    field_grid_size_string = np.array2string(
                        lagrangian_grid_size,
                        precision=self.precision,
                        separator="    ",
                    )[1:-1]

                field_entries += generate_lagrangian_field_entry(
                    h5_file_name,
                    field_name,
                    field_type,
                    field_grid_size_string,
                    lagrangian_grid_name=lagrangian_grid_name,
                )

            if lagrangian_grid_name in self.lagrangian_grid_connection:
                topology = f"""<Topology TopologyType="Polyline">
                <DataItem DataType="Int" Dimensions="1
                {lagrangian_grid_size[0]}" Format="HDF" Precision="{self.precision}">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/Connection
                </DataItem>
            </Topology>"""
            else:
                topology = f"""<Topology TopologyType="Polyvertex"
                NumberOfElements="{lagrangian_grid_size[0]}"/>"""

            xdmffile = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
    <Domain>
        <Grid GridType="Uniform">
            <Time Value="{time}"/>
            {topology}
            <Geometry GeometryType="{geometry_type}">
                <DataItem Dimensions="{lagrangian_grid_size_string}"
                NumberType="Float" Precision="{self.precision}" Format="HDF">
                    {h5_file_name}:/Lagrangian/{lagrangian_grid_name}/Grid
                </DataItem>
            </Geometry>

            {field_entries}
        </Grid>
    </Domain>
</Xdmf>
"""

            with open(xmf_file_name, "w") as f:
                f.write(xdmffile)


class CosseratRodIO(IO):
    """
    Derived IO class for Cosserat rod IO.
    """

    def __init__(
        self, cosserat_rod: CosseratRod, dim: int, real_dtype: type = np.float64
    ) -> None:
        super().__init__(dim, real_dtype)
        self.cosserat_rod = cosserat_rod

        # Initialize rod element position
        self.rod_element_position = np.zeros((self.dim, cosserat_rod.n_elems))
        self._update_rod_element_position()

        # Add the element position to IO
        self.add_as_lagrangian_fields_for_io(
            lagrangian_grid=self.rod_element_position,
            lagrangian_grid_name="rod",
            scalar_3d=self.cosserat_rod.radius,
            lagrangian_grid_connect=True,
        )

    def save(self, h5_file_name: str, time: float = 0.0) -> None:
        self._update_rod_element_position()
        self._save(h5_file_name=h5_file_name, time=time)

    def _update_rod_element_position(self) -> None:
        self.rod_element_position[...] = 0.5 * (
            self.cosserat_rod.position_collection[: self.dim, 1:]
            + self.cosserat_rod.position_collection[: self.dim, :-1]
        )
