from typing import TypeAlias, Union

import numpy as np

from httomo.utils import xp


generic_array: TypeAlias = Union[np.ndarray, xp.ndarray]
