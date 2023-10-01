import numpy as np
from cython.parallel import parallel, prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline float cosine_similarity(float[:] vector_a, float[:] vector_b) nogil:
    """Calculate cosine similarity between two vectors."""
    cdef float dot_product = 0.0
    cdef float norm_a = 0.0
    cdef float norm_b = 0.0
    cdef Py_ssize_t i
    for i in range(vector_a.shape[0]):
        dot_product += vector_a[i] * vector_b[i]
        norm_a += vector_a[i] ** 2
        norm_b += vector_b[i] ** 2
    return dot_product / (norm_a**0.5 * norm_b**0.5)


def norm_cosine_similarity(float[:] vector_a, float[:] vector_b)->float:
    """Calculate normalized cosine similarity between two vectors."""
    return 0.5 * (1 + cosine_similarity(vector_a, vector_b))