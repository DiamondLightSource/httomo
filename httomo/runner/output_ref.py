from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # this is purely for type annotations. We import conditionally to
    # avoid circular imports (need to quote the type when used below)
    from httomo.runner.method_wrapper import MethodWrapper  # pragma: no cover


class OutputRef:
    """Class to reference an output from other methods lazily."""

    def __init__(self, method: "MethodWrapper", mapped_output_name: str):
        self.method = method
        self.mapped_output_name = mapped_output_name

    @property
    def value(self) -> Any:
        mapped_outputs = self.method.get_side_output()
        if self.mapped_output_name not in mapped_outputs:
            raise ValueError(
                f"Output name {self.mapped_output_name} not found in"
                + f" mapped outputs of method {self.method.method_name}."
                + f" The following mapped outputs are known: {mapped_outputs.keys()}"
            )
        return mapped_outputs[self.mapped_output_name]
