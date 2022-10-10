from typing import Any

from mpi4py.MPI import Comm


def print_once(output: Any, comm: Comm):
    """Print an output from rank zero only.

    Args:
        output: The item to be printed.
        comm: The comm used to determine the rank zero process.
    """
    if comm.rank == 0:
        print(output)


def print_rank(output: Any, comm: Comm):
    """Print an output with rank prefix.

    Args:
        output: The item to be printed.
        comm: The comm used to determine the process rank.
    """
    print(f"[{comm.rank}] {output}")
