class VectorField:
    """
    Class representing vector fields.
    Useful for slicing individual axes.
    TODO can use later as dtype for vector fields in simulator
    """

    @staticmethod
    def x_axis_idx() -> int:
        """
        Returns index of X axis in a vector field

        Returns
        -------
        0, static value
        """
        return 0

    @staticmethod
    def y_axis_idx() -> int:
        """
        Returns index of Y axis in a vector field

        Returns
        -------
        1, static value
        """
        return 1

    @staticmethod
    def z_axis_idx() -> int:
        """
        Returns index of Z axis in a vector field

        Returns
        -------
        2, static value
        """
        return 2
