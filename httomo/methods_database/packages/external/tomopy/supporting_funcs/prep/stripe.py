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
    "_calc_padding_stripes_detect3d",
    "_calc_padding_stripes_mask3d",
]


def _calc_padding_stripes_detect3d(radius=3, **kwargs) -> Tuple[int, int]:
    # TODO: confirm if this correponds to the padding here:
    # https://github.com/tomopy/tomopy/blob/0c6d18da8a5b8fddde1a3e0d8864f5b3fefff8ae/source/tomopy/prep/stripe.py#L988C5-L988C19
    return radius, radius


def _calc_padding_stripes_mask3d(min_stripe_depth=10, **kwargs) -> Tuple[int, int]:
    # TODO: confirm if this correponds to the padding here:
    # https://github.com/tomopy/tomopy/blob/master/source/tomopy/prep/stripe.py#L1062
    return min_stripe_depth, min_stripe_depth
