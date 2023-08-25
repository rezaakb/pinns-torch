from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PointCloudData:
    """A data class to hold point cloud data, including spatial points, time values, and solution
    data.

    This is an alternative class to spatial and time class that should be used along with `PointCloud` class.
    """

    spatial: List[np.array]  # List of spatial point data.
    time: np.array  # NumPy array containing time values.
    solution: Dict[str, np.array]  # Dictionary containing solution data.
