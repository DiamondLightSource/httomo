from io import StringIO

import pytest
from mpi4py import MPI
from httomo.monitors.benchmark import BenchmarkMonitoring


def test_benchmark_monitor_records_and_displays_data():
    mon = BenchmarkMonitoring()
    mon.report_method_block(
        "method1", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0, 0.1, 0.2
    )
    mon.report_method_block(
        "method2", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0, 0.1, 0.2
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
    data = dest.getvalue().splitlines()
    assert len(data) == 8
    assert data[0] == ",".join([
        "Type",
        "Rank","Name",
        "Task id",
        "Module",
        "Slicing dim",
        "Block offset (chunk)",
        "Block offset (global)",
        "Block dim z",
        "Block dim y",
        "Block dim x",
        "CPU time",
        "GPU kernel time",
        "GPU H2D time",
        "GPU D2H time"
    ])

    assert "42.0,2.0,0.1,0.2" in data[1]
    assert "42.0,2.0,0.1,0.2" in data[2]
    assert "4.0,0.0" in data[3]
    assert "4.0,0.0" in data[4]
    assert "3.0,0.0" in data[5]
    assert "3.0,0.0" in data[6]
    assert "500.0,0.0" in data[7]


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_summary_monitor_records_and_displays_data_mpi():
    comm = MPI.COMM_WORLD
    mon = BenchmarkMonitoring()
    # everything gets reported twice - once in each process - and the write_results should aggregate
    # in process 0
    mon.report_method_block(
        "method1", "module", "task", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 42.0, 2.0, 0.1, 0.2
    )
    mon.report_source_block(
        "loader", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 4.0
    )
    mon.report_sink_block("wrt1", "method1", 0, (1, 2, 3), (0, 0, 0), (10, 0, 0), 3.0)
    mon.report_total_time(500.0)

    dest = StringIO()
    mon.write_results(dest)
    dest.flush()
    data = dest.getvalue().splitlines()
    if comm.rank == 1:
        assert len(data) == 0
    else:
        assert len(data) == 9
        assert data[0] == ",".join([
            "Type",
            "Rank","Name",
            "Task id",
            "Module",
            "Slicing dim",
            "Block offset (chunk)",
            "Block offset (global)",
            "Block dim z",
            "Block dim y",
            "Block dim x",
            "CPU time",
            "GPU kernel time",
            "GPU H2D time",
            "GPU D2H time"
        ])

        for rank in [0, 1]:
            assert "42.0,2.0,0.1,0.2" in data[1+rank*4]
            assert f"method,{rank},method1" in data[1+rank*4]
            assert "4.0,0.0" in data[2+rank*4]
            assert f"source,{rank},loader" in data[2+rank*4]
            assert "3.0,0.0" in data[3+rank*4]
            assert f"sink,{rank},wrt1" in data[3+rank*4]
            assert "500.0,0.0" in data[4+rank*4]
            assert f"total,{rank},," in data[4+rank*4]
