from pytest_mock import MockerFixture

from httomo.runner.pipeline import Pipeline
from httomo.utils import Pattern
from .testing_utils import make_test_loader, make_test_method


def test_pipeline_get_loader_properties(mocker: MockerFixture):
    loader = make_test_loader(mocker, pattern=Pattern.projection)
    loader.reslice = True
    p = Pipeline(loader=loader, methods=[])

    assert p.loader_pattern == loader.pattern
    assert p.loader_reslice == loader.reslice


def test_pipeline_set_loader_properties(mocker: MockerFixture):
    loader = make_test_loader(mocker, pattern=Pattern.projection)
    loader.reslice = True
    p = Pipeline(loader=loader, methods=[])

    p.loader_pattern = Pattern.sinogram
    p.loader_reslice = False

    assert loader.pattern == Pattern.sinogram
    assert loader.reslice is False


def test_pipeline_can_iterate(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[
            make_test_method(mocker, method_name="m1"),
            make_test_method(mocker, method_name="m2"),
            make_test_method(mocker, method_name="m3"),
        ],
    )

    assert len(p) == 3
    for i, m in enumerate(p):
        assert m.method_name == f"m{i+1}"
