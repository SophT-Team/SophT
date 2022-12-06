import elastica as ea
from elastica._linalg import _batch_cross
from elastica.utils import MaxDimension
import logging
import numpy as np


class RectangularPlane(ea.RigidBodyBase):
    def __init__(
        self,
        origin: np.ndarray,
        plane_normal: np.ndarray,
        plane_tangent_along_length: np.ndarray,
        plane_length: float,
        plane_breadth: float,
    ) -> None:
        """
        Rigid rectangular plane initializer.

        Parameters
        ----------
        origin
        plane_normal
        plane_length
        plane_breadth
        plane_tangent_along_length

        Notes
        -----
        Currently only supports imposed modes, cannot track
        dynamics in pyelastica!
        """
        super(RectangularPlane, self).__init__()

        logger = logging.getLogger()
        logger.warning(
            "==============================================="
            "\nInitialising rectangular plane object. Note:"
            "\nCurrently tracking dynamics in pyelastica is"
            "\nnot supported, please do not add the plane"
            "\nto the pyelastica simulator!"
            "\n==============================================="
        )

        self.n_elems = 1
        self.length = plane_length
        self.breadth = plane_breadth
        normal = plane_normal.reshape(MaxDimension.value(), self.n_elems)
        tangent = plane_tangent_along_length.reshape(MaxDimension.value(), self.n_elems)
        binormal = _batch_cross(normal, tangent)
        self.director_collection = np.zeros(
            (MaxDimension.value(), MaxDimension.value(), 1)
        )
        # TODO put checks for invalid normals and tangents
        self.director_collection[0, ...] = tangent / np.linalg.norm(tangent)
        self.director_collection[1, ...] = binormal / np.linalg.norm(binormal)
        self.director_collection[2, ...] = normal / np.linalg.norm(normal)

        self.position_collection = origin.reshape(MaxDimension.value(), self.n_elems)
        self.velocity_collection = np.zeros((MaxDimension.value(), self.n_elems))
        self.omega_collection = np.zeros((MaxDimension.value(), self.n_elems))
