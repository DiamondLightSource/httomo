from typing import Tuple


__all__ = [
    "_calc_padding_remove_outlier",
]


def _calc_padding_remove_outlier(**kwargs) -> Tuple[int, int]:
    kernel_size = kwargs["kernel_size"]
    return (kernel_size // 2, kernel_size // 2)
