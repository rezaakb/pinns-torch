import numpy as np


class Interval:
    """Initialize a 1D spatial interval."""

    def __init__(self, x_interval, shape):
        """Initialize an Interval object to represent a 1D spatial interval.

        :param x_interval: A tuple or list representing the spatial interval [start_x, end_x].
        :param shape: The number of points in the spatial interval.
        """

        self.x_interval = x_interval
        self.shape = shape

    def generate_mesh(self, t_points: int):
        """Generate a mesh for the 1D spatial interval.

        :param t_points: The number of time points in the mesh.
        :return: A mesh for the 1D spatial interval.
        """

        x = np.linspace(self.x_interval[0], self.x_interval[1], num=self.shape[0])
        self.mesh = np.tile(x, (t_points, 1)).T[:, :, None]

        return self.mesh

    def __len__(self):
        """Get the length of the interval mesh.

        :return: The number of points in the interval mesh.
        """

        return len(self.mesh)

    def __getitem__(self, idx):
        """Get a specific point from the interval mesh using indexing.

        :param idx: The index of the desired point.
        :return: The point value at the specified index.
        """

        return self.mesh[idx, 0]


class Rectangle:
    """Initialize a 2D spatial domain."""

    def __init__(self, x_interval, y_interval, shape):
        """Initialize a Rectangle object to represent a 2D spatial rectangle.

        :param x_interval: A tuple representing the x-axis interval [start_x, end_x].
        :param y_interval: A tuple representing the y-axis interval [start_y, end_y].
        :param shape: The number of points along each axis in the rectangle.
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.shape = shape

    def generate_mesh(self, t_points):
        """Generate a mesh for the 2D spatial rectangle.

        :param t_points: The number of time points in the mesh.
        :return: A mesh for the 2D spatial rectangle.
        """

        x = np.linspace(self.x_interval[0], self.x_interval[1], num=self.shape[0])
        y = np.linspace(self.y_interval[0], self.y_interval[1], num=self.shape[1])

        self.xx, self.yy = np.meshgrid(x, y)
        self.spatial_mesh = np.stack((self.xx.flatten(), self.yy.flatten()), 1)

        self.mesh = np.zeros((np.prod(self.shape), t_points, 2))

        for i in range(2):
            self.mesh[:, :, i] = np.tile(self.spatial_mesh[:, i], (t_points, 1)).T

        return self.mesh

    def __len__(self):
        """Get the length of the rectangle mesh.

        :return: The number of points in the rectangle mesh.
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """Get a specific point from the rectangle mesh using indexing.

        :param idx: The index of the desired point.
        :return: The point value at the specified index.
        """

        return self.mesh[idx, 0]


class RectangularPrism:
    """Initialize a 3D spatial domain."""

    def __init__(self, x_interval, y_interval, z_interval, shape):
        """Initialize a Rectangular Prism object to represent a three-dimensional shape.

        :param x_interval: A tuple or list representing the x-axis interval [start_x, end_x].
        :param y_interval: A tuple or list representing the y-axis interval [start_y, end_y].
        :param z_interval: A tuple or list representing the z-axis interval [start_z, end_z].
        :param shape: The number of points along each axis in the cube.
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.z_interval = z_interval
        self.shape = shape

    def generate_mesh(self, t_points):
        """Generate a mesh for the 3D spatial cube.

        :param t_points: The number of time points in the mesh.
        :return: A mesh for the 3D spatial cube.
        """

        x = np.linspace(self.x_interval[0], self.x_interval[1], num=self.shape[0])
        y = np.linspace(self.y_interval[0], self.y_interval[1], num=self.shape[1])
        z = np.linspace(self.z_interval[0], self.z_interval[1], num=self.shape[2])

        self.xx, self.yy = np.meshgrid(x, y)
        self.spatial_mesh = np.stack((self.xx.flatten(), self.yy.flatten()), 1)

        self.mesh = np.zeros((np.prod(self.shape), t_points, 3))

        for i in range(3):
            self.mesh[:, :, i] = np.tile(self.spatial_mesh[:, i], (t_points, 1)).T

        return self.mesh

    def __len__(self):
        """Get the length of the cube mesh.

        :return: The number of points in the cube mesh.
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """Get a specific point from the cube mesh using indexing.

        :param idx: The index of the desired point.
        :return: The point value at the specified index.
        """
        return self.mesh[idx, 0]
