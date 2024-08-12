#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 18/July/2024
# ---------------------------------------------------------------------------
""" Modules for memory estimation, padding, data dims estimation """

from typing import Tuple

__all__ = [
    "_calc_padding_median_filter3d",
    "_calc_padding_remove_outlier3d",
]


def _calc_padding_median_filter3d(size: int = 3, **kwargs) -> Tuple[int, int]:
    return size // 2, size // 2


def _calc_padding_remove_outlier3d(size: int = 3, **kwargs) -> Tuple[int, int]:
    return size // 2, size // 2
