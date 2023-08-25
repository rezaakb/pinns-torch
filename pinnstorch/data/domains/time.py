import numpy as np


class TimeDomain:
    """Initialize a time domain."""

    def __init__(self, t_interval, t_points: int):
        """Initialize a TimeDomain object to represent a time domain.

        :param t_interval: A tuple or list representing the time interval [start_time, end_time].
        :param t_points: The number of time points to discretize the interval.
        """
        self.time_interval = t_interval
        self.time = np.linspace(self.time_interval[0], self.time_interval[1], num=t_points)

    def generate_mesh(self, spatial_points):
        """Generate a mesh of time based on number of spatial points.

        :param spatial_points: The number of spatial points.
        :return: A mesh of time and spatial points.
        """

        mesh = np.tile(self.time, (spatial_points, 1))[:, :, None]

        return mesh

    def __len__(self):
        """Get the length of the time domain.

        :return: The number of time points in the time domain.
        """
        return len(self.time)

    def __getitem__(self, idx):
        """Get a specific time point from the time domain using indexing.

        :param idx: The index of the desired time point.
        :return: The time value at the specified index.
        """
        return self.time[idx]
