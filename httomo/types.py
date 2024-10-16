from typing import TypeAlias

import numpy as np

from httomo.utils import xp


generic_array: TypeAlias = np.ndarray | xp.ndarray
