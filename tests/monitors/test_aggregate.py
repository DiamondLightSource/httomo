from io import StringIO
import pytest
from pytest_mock import MockerFixture
from httomo.monitors import make_monitors
from httomo.monitors.aggregate import AggregateMonitoring

from httomo.runner.monitoring_interface import MonitoringInterface


def test_aggregate_passes_on_method(mocker: MockerFixture):
    mon1 = mocker.create_autospec(MonitoringInterface, instance=True)
    mon2 = mocker.create_autospec(MonitoringInterface, instance=True)
    agg = AggregateMonitoring([mon1, mon2])
    args = ("method", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 
            42.0, 3.2, 1.2, 1.1)
    agg.report_method_block(*args)

    mon1.report_method_block.assert_called_once_with(*args)
    mon2.report_method_block.assert_called_once_with(*args)


def test_aggregate_passes_on_source_block(mocker: MockerFixture):
    mon1 = mocker.create_autospec(MonitoringInterface, instance=True)
    mon2 = mocker.create_autospec(MonitoringInterface, instance=True)
    agg = AggregateMonitoring([mon1, mon2])
    args = ("loader", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0)
    agg.report_source_block(*args)

    mon1.report_source_block.assert_called_once_with(*args)
    mon2.report_source_block.assert_called_once_with(*args)


def test_aggregate_passes_on_sink_block(mocker: MockerFixture):
    mon1 = mocker.create_autospec(MonitoringInterface, instance=True)
    mon2 = mocker.create_autospec(MonitoringInterface, instance=True)
    agg = AggregateMonitoring([mon1, mon2])
    args = ("save", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0)
    agg.report_sink_block(*args)

    mon1.report_sink_block.assert_called_once_with(*args)
    mon2.report_sink_block.assert_called_once_with(*args)


def test_aggregate_passes_on_total_time(mocker: MockerFixture):
    mon1 = mocker.create_autospec(MonitoringInterface, instance=True)
    mon2 = mocker.create_autospec(MonitoringInterface, instance=True)
    agg = AggregateMonitoring([mon1, mon2])
    args = (42.0,)
    agg.report_total_time(*args)

    mon1.report_total_time.assert_called_once_with(*args)
    mon2.report_total_time.assert_called_once_with(*args)


def test_aggregate_passes_on_write_results(mocker: MockerFixture):
    mon1 = mocker.create_autospec(MonitoringInterface, instance=True)
    mon2 = mocker.create_autospec(MonitoringInterface, instance=True)
    agg = AggregateMonitoring([mon1, mon2])
    dest = StringIO()
    args = (dest,)
    agg.write_results(*args)

    mon1.write_results.assert_called_once_with(*args)
    mon2.write_results.assert_called_once_with(*args)


def test_make_monitors_2(mocker: MockerFixture):
    moncls1 = mocker.create_autospec(MonitoringInterface, instance=False)
    moncls2 = mocker.create_autospec(MonitoringInterface, instance=False)
    moncls3 = mocker.create_autospec(MonitoringInterface, instance=False)
    mocker.patch(
        "httomo.monitors.MONITORS_MAP", {"m1": moncls1, "m2": moncls2, "m3": moncls3}
    )

    mon = make_monitors(["m1", "m2"])
    assert isinstance(mon, AggregateMonitoring)
    assert len(mon._monitors) == 2
    moncls1.assert_called_once()
    moncls2.assert_called_once()
    moncls3.assert_not_called()
    assert isinstance(mon._monitors[0], type(moncls1()))
    assert isinstance(mon._monitors[1], type(moncls2()))


def test_make_monitors_unknown(mocker: MockerFixture):
    mocker.patch("httomo.monitors.MONITORS_MAP", {})

    with pytest.raises(ValueError) as e:
        make_monitors(["m1"])

    assert "Unknown monitor 'm1'" in str(e)


def test_make_monitors_empty():
    assert make_monitors([]) is None