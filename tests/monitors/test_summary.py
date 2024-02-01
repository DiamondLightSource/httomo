from io import StringIO

import pytest
from mpi4py import MPI
from httomo.monitors.summary import SummaryMonitor


def test_summary_monitor_records_and_displays_data():
    mon = SummaryMonitor()
    mon.report_method_block(
        "method1", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0
    )
    mon.report_method_block(
        "method2", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0
    )
    mon.report_source_block(
        "loader", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 4.0
    )
    mon.report_source_block(
        "other", "method2", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 4.0
    )
    mon.report_sink_block("wrt1", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 3.0)
    mon.report_sink_block("dummy", "method2", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 3.0)
    mon.report_total_time(500.0)

    dest = StringIO()
    mon.write_results(dest)
    dest.flush()
    data = dest.getvalue()
    
    assert "across 1 process" in data
    assert "Summary Statistics" in data
    assert "methods CPU time" in data
    assert " 84.0" in data
    assert "methods GPU time" in data
    assert " 4.0" in data
    assert "sources time" in data
    assert " 8.0" in data
    assert "sinks time" in data
    assert " 6.0" in data
    assert "pipeline time" in data
    assert "500.0" in data
    assert "wall time" in data

@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_summary_monitor_records_and_displays_data_mpi():

    comm = MPI.COMM_WORLD
    mon = SummaryMonitor()
    # everything gets reported twice - once in each process - and the write_results should aggregate
    # in process 0
    mon.report_method_block(
        "method1", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0
    )
    mon.report_source_block(
        "loader", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 4.0
    )
    mon.report_sink_block("wrt1", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 3.0)
    mon.report_total_time(500.0)

    dest = StringIO()
    mon.write_results(dest)
    dest.flush()
    data = dest.getvalue()
    if comm.rank == 1:
        assert data == ""
    else:
        assert "Summary Statistics" in data
        assert "across 2 processes" in data
        assert "methods CPU time" in data
        assert " 84.0" in data
        assert "methods GPU time" in data
        assert " 4.0" in data
        assert "sources time" in data
        assert " 8.0" in data
        assert "sinks time" in data
        assert " 6.0" in data
        assert "pipeline time" in data
        assert "1000.0" in data
        assert "wall time" in data
        assert "500.0" in data
        
