from typing import Any
from mpi4py.MPI import Comm
 
def print_once(output: Any, comm: Comm) -> None:
    """Print an output from rank zero only.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the rank zero process.
    """
    if comm.rank == 0:
        print(output)


def print_rank(output: Any, comm: Comm) -> None:
    """Print an output with rank prefix.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the process rank.
    """
    print(f"[{comm.rank}] {output}")

