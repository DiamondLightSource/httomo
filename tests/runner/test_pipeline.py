from pytest_mock import MockerFixture

from httomo.runner.pipeline import Pipeline
from httomo.utils import Pattern
from ..testing_utils import make_test_loader, make_test_method


def test_pipeline_get_loader_properties(mocker: MockerFixture):
    loader = make_test_loader(mocker, pattern=Pattern.projection)
    p = Pipeline(loader=loader, methods=[])

    assert p.loader_pattern == loader.pattern


def test_pipeline_set_loader_properties(mocker: MockerFixture):
    loader = make_test_loader(mocker, pattern=Pattern.projection)
    p = Pipeline(loader=loader, methods=[])

    p.loader_pattern = Pattern.sinogram

    assert loader.pattern == Pattern.sinogram


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


def test_pipeline_can_access_by_index(mocker: MockerFixture):
    p = Pipeline(
        loader=make_test_loader(mocker, pattern=Pattern.projection),
        methods=[
            make_test_method(mocker, method_name="m1"),
            make_test_method(mocker, method_name="m2"),
            make_test_method(mocker, method_name="m3"),
        ],
    )
    
    for i in range(3):
        assert p[i].method_name == f"m{i+1}"
