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
        self.huge = None
        self.is_free = True

    def malloc(self, size):
        HUGE_THRESHOLD = 1024**3  # 1 GiB
        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)
        if size < HUGE_THRESHOLD or not self.is_free:
            start = perf_counter()
            ret = _malloc(size)
            elapsed = perf_counter() - start
            if size >= HUGE_THRESHOLD:
                log_rank(
                    f"PinnedMemoryPool raw huge alloc {size} bytes ({elapsed:.3f} s)",
                    self.comm,
                )
            return ret

        if (self.huge is None) or (self.huge.mem.size < size):
            alloc_size = size * 2
            start = perf_counter()
            self.huge = _malloc(alloc_size)
            elapsed = perf_counter() - start
            log_rank(
                f"PinnedMemoryPool huge alloc {alloc_size} bytes ({elapsed:.3f} s)",
                self.comm,
            )

        self.is_free = False
        return PinnedMemoryPointer(PooledPinnedMemory(self.huge.mem, self._weakref), 0)

    def free(self, mem, size):
        self.is_free = True
