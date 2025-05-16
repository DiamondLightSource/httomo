from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


class ParamSweepJsonLoader:
    """
    Loader for JSON pipelines containing parameter sweep
    """

    def __init__(self, json_string: str) -> None:
        self.json_string = json_string

    def load(self) -> List[Dict[str, Any]]:
        """
        Convert JSON data to python dict
        """
        data: List[Dict[str, Any]] = json.loads(self.json_string)
        res = self._find_range_sweep_param(data[1:])
        if res is not None:
            sweep_dict = data[res[1] + 1]["parameters"][res[0]]
            sweep_vals = tuple(
                np.arange(sweep_dict["start"], sweep_dict["stop"], sweep_dict["step"])
            )
            data[res[1] + 1]["parameters"][res[0]] = sweep_vals

        res = self._find_manual_sweep_param(data[1:])
        if res is not None:
            sweep_vals = data[res[1] + 1]["parameters"][res[0]]
            data[res[1] + 1]["parameters"][res[0]] = tuple(sweep_vals)

        return data

    def _find_range_sweep_param(
        self, methods: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, int]]:
        for idx, method in enumerate(methods):
            for name, value in method["parameters"].items():
                if isinstance(value, dict):
                    keys = value.keys()
                    has_keys_for_sweep = (
                        "start" in keys and "stop" in keys and "step" in keys
                    )
                    if has_keys_for_sweep and len(keys) == 3:
                        return name, idx

    def _find_manual_sweep_param(
        self, methods: List[Dict[str, Any]]
    ) -> Optional[Tuple[str, int]]:
        for idx, method in enumerate(methods):
            for name, value in method["parameters"].items():
                if isinstance(value, list):
                    return name, idx
