import numpy as np


class TimeSeriesDomain:
    """Initialize a time domain."""

    def __init__(self, data_input: np.array, t_interval: int):
        """Initialize a TimeSeriesDomain object to represent a time domain.

        :param data_input: 
        :param t_interval: Sampling rate of the timeseries data.
        :param t_points: The number of time points to discretize the interval.
        """
        if len(data_input.shape) != 2:
            RuntimeError("The dimension of the data should be two!")

        self.data_input = data_input
        self.time_interval = t_interval
        self.t_points = data_input.shape[0]
        self.time = np.linspace(0, 0+t_interval*(self.t_points+1), num=self.t_points)

    def generate_mesh(self):
        """Generate a mesh of time based on number of spatial points.

        :param spatial_points: The number of spatial points.
        :return: A mesh of time and spatial points.
        """

        mesh = np.tile(self.time)[None, :, None]

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
