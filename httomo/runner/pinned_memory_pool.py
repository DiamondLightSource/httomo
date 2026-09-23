from ..utils import log_rank
import cupy as cp
import weakref
from cupy.cuda import PinnedMemoryPointer, PinnedMemory
from cupy.cuda.pinned_memory import _malloc


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
        _, self.gpu_total_mem = cp.cuda.Device().mem_info
        self.pinned_memory_ptr = _malloc(self.gpu_total_mem)
        self.is_free = True
        log_rank(f"Pinned memory pool allocated {self.gpu_total_mem} bytes", self.comm)

    def malloc(self, size):
        if size == 0:
            return PinnedMemoryPointer(PinnedMemory(0), 0)
        if not self.is_free or size > self.gpu_total_mem:
            log_rank(f"Additional pinned memory allocation: {size} bytes", self.comm)
            return _malloc(size)
        self.is_free = False
        pmem = PooledPinnedMemory(self.pinned_memory_ptr.mem, self._weakref)
        return PinnedMemoryPointer(pmem, 0)

    def free(self, mem, size):
        self.is_free = True
