from httomo.data.param_sweep_store import ParamSweepWriter


def make_param_sweep_writer() -> ParamSweepWriter:
    NO_OF_SWEEPS = 5
    SWEEP_RES_SHAPE = (180, 3, 160)
    return ParamSweepWriter(
        no_of_sweeps=NO_OF_SWEEPS,
        single_shape=SWEEP_RES_SHAPE,
    )


def test_param_sweep_writer_get_no_of_sweeps():
    writer = make_param_sweep_writer()
    assert writer.no_of_sweeps == 5


def test_param_sweep_writer_get_concat_dim():
    writer = make_param_sweep_writer()
    assert writer.concat_dim == 1


def test_param_sweep_writer_get_single_shape():
    writer = make_param_sweep_writer()
    assert writer.single_shape == (180, 3, 160)


def test_param_sweep_writer_get_total_shape():
    writer = make_param_sweep_writer()
    assert writer.total_shape == (180, 3 * 5, 160)


def test_param_sweep_reader_get_no_of_sweeps():
    writer = make_param_sweep_writer()
    reader = writer.make_reader()
    assert reader.no_of_sweeps == 5


def test_param_sweep_reader_get_extract_dim():
    writer = make_param_sweep_writer()
    reader = writer.make_reader()
    assert reader.extract_dim == 1


def test_param_sweep_reader_get_single_shape():
    writer = make_param_sweep_writer()
    reader = writer.make_reader()
    assert reader.single_shape == (180, 3, 160)


def test_param_sweep_reader_get_total_shape():
    writer = make_param_sweep_writer()
    reader = writer.make_reader()
    assert reader.total_shape == (180, 3 * 5, 160)
