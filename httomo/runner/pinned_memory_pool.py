from ..utils import log_rank
import cupy as cp
import weakref
from cupy.cuda import PinnedMemoryPointer, PinnedMemory
from cupy.cuda.pinned_memory import _malloc
from time import perf_counter


class PooledPinnedMemory(PinnedMemory):
    """Memory allocation for a memory pool.

    As the instance of this class is created by memory pool allocator, users
    should not instantiate it manually.

    """

    def __init__(self, mem, pool):
        self.ptr = mem.ptr
        self.size = mem.size
        self.pool = pool
        self.mem = mem

    def free(self):
        """Releases the memory buffer and sends it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        pool = self.pool()
        if pool is not None and self.ptr != 0:
            pool.free(self.mem, self.size)
        self.ptr = 0
        self.size = 0

    __del__ = free


class PinnedMemoryPool:
    def __init__(self, comm) -> None:
        self._weakref = weakref.ref(self)
        self.comm = comm

    def malloc(self, size):
        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)
        start = perf_counter()
        pmem = PooledPinnedMemory(_malloc(size).mem, self._weakref)
        elapsed = perf_counter() - start
        log_rank(
            f"Pinned memory pool allocated {size} bytes (duration: {elapsed:.3f} s)",
            self.comm,
        )
        return PinnedMemoryPointer(pmem, 0)

    def free(self, mem, size):
        log_rank(f"Pinned memory pool freed {size} bytes", self.comm)
