from httomo.methods_database.packages.external.tomopy.supporting_funcs.misc.corr import (
    _calc_padding_median_filter3d,
    _calc_padding_remove_outlier3d,
)
from httomo.methods_database.packages.external.tomopy.supporting_funcs.prep.stripe import (
    _calc_padding_stripes_detect3d,
    _calc_padding_stripes_mask3d,
)


def test_calc_padding_median_filter3d() -> None:
    kwargs: dict = {"size": 5, "ncore": 10}
    assert _calc_padding_median_filter3d(**kwargs) == (2, 2)


def test_calc_padding_median_filter3d_defaults() -> None:
    # it defaults to size=3
    kwargs: dict = {}
    assert _calc_padding_median_filter3d(**kwargs) == (1, 1)


def test_calc_padding_remove_outlier3d() -> None:
    kwargs: dict = {"size": 5, "ncore": 10}
    assert _calc_padding_remove_outlier3d(**kwargs) == (2, 2)


def test_calc_padding_remove_outlier3d_defaults() -> None:
    # it defaults to size=3
    kwargs: dict = {}
    assert _calc_padding_remove_outlier3d(**kwargs) == (1, 1)

# TODO: unclear if the following function's padding behaviour is
# doing the right thing... 

def test_calc_padding_stripes_detect3d() -> None:
    kwargs: dict = {"size": 15, "radius": 2, "ncore": 10}
    assert _calc_padding_stripes_detect3d(**kwargs) == (2, 2)


def test_calc_padding_stripes_detect3d_defaults() -> None:
    # it defaults to radius=3
    kwargs: dict = {}
    assert _calc_padding_stripes_detect3d(**kwargs) == (3, 3)


def test_calc_padding_stripes_mask3d() -> None:
    kwargs: dict = {"threashold": .7, "min_stripe_depth": 5}
    assert _calc_padding_stripes_mask3d(**kwargs) == (5, 5)


def test_calc_padding_stripes_mask3d_defaults() -> None:
    # it defaults to min_stripe_depth=10
    kwargs: dict = {}
    assert _calc_padding_stripes_mask3d(**kwargs) == (10, 10)
