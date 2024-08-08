from typing import Any, Dict, List

from httomo.runner.method_wrapper import MethodWrapper


class SideOutputManager:
    """
    Provides helper functionality to manage side outputs produced by method wrappers
    """

    def __init__(self) -> None:
        self._side_outputs: Dict[str, Any] = dict()

    @property
    def labels(self) -> List[str]:
        """Get labels/names of all side outputs"""
        return list(self._side_outputs.keys())

    def append(self, side_outputs: Dict[str, Any]):
        """Merge given side outputs with existing side outputs"""
        self._side_outputs |= side_outputs

    def get(self, label: str) -> Any:
        """Get the value of the given side output"""
        return self._side_outputs[label]

    def update_params(self, wrapper: MethodWrapper):
        """Update parameters of given method wrapper with required side outputs"""
        for k, v in self._side_outputs.items():
            if k in wrapper.parameters:
                wrapper[k] = v
