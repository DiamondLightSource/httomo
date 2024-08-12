from time import sleep
from httomo.utils import catchtime, xp, catch_gputime, gpu_enabled


def test_cachetime():
    with catchtime() as t:
        sleep(0.1)

    assert t.elapsed >= 0.1
    assert t.elapsed < 0.2


def test_catch_gputime():
    input = xp.ones((500, 200, 100), dtype=xp.float32)
    with catch_gputime() as t:
        xp.sum(input)

    if gpu_enabled:
        assert t.elapsed > 0.0
    else:
        assert t.elapsed == 0.0
