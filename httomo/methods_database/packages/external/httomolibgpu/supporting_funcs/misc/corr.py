from typing import Tuple


__all__ = [
    "_calc_padding_remove_outlier",
    "_calc_padding_median_filter",
]


def _calc_padding_remove_outlier(**kwargs) -> Tuple[int, int]:
    kernel_size = kwargs["kernel_size"]
    return (kernel_size // 2, kernel_size // 2)


def _calc_padding_median_filter(**kwargs) -> Tuple[int, int]:
    kernel_size = kwargs["kernel_size"]
    return (kernel_size // 2, kernel_size // 2)
