import numpy as np
cimport numpy as np

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t


def wu_sum(np.ndarray [DTYPE_t, ndim=2] array, p1, p2, bint count=False, int width=0, bint debug=False):
    """This function is a wrapper around the Cython implementation of
    wu_sum. Its call signature is identical to the one of
    helpers.wu_sum. If possible, it is recommended to call
    wu_sum_cython() directly for increased performance.

    """
    return wu_sum_cython(array, p1[0], p1[1], p2[0], p2[1], count, width, debug)
    

def wu_sort_points(double p1_0, double p1_1, double p2_0, double p2_1):
    """This is a helper function for wu_sum in order to bring the points
    into the right order."""
    cdef bint switch_dim = abs(p1_0 - p2_0) < abs(p1_1 - p2_1)
    if switch_dim:
        p1_0, p1_1 = (p1_1, p1_0)
        p2_0, p2_1 = (p2_1, p2_0)
    if p1_0 > p2_0:
        (p1_0, p1_1, p2_0, p2_1) = (p2_0, p2_1, p1_0, p1_1)
    return (p1_0, p1_1, p2_0, p2_1, switch_dim)


def wu_sum_cython(np.ndarray [DTYPE_t, ndim=2] array, double p1_0, double p1_1, double p2_0, double p2_1, bint count=False, int width=0, bint debug=False):
    """Sum over the entries in array tha lie on the line between p1 and
    p2. Uses cheap aliasing for this sum.  Width changes this into a
    'fat' line that covers 1+width pixels. The algorithm is 'cheap'
    and not always completely accurate.
    This function is messy and slow. Refactor (maybe in Cython) at some
    point.

    """
    assert array.dtype == DTYPE
    cdef bint switch_dim = False
    cdef double x0 = 0
    cdef double x1 = 0
    cdef double y0 = 0
    cdef double y1 = 0
    (p1_0, p1_1, p2_0, p2_1, switch_dim) = wu_sort_points(p1_0, p1_1, p2_0, p2_1)
    cdef int max_x = array.shape[1] if switch_dim else array.shape[0]
    cdef int max_y = array.shape[0] if switch_dim else array.shape[1]
    cdef double s = 0
    (x0, y0) = (p1_0, p1_1)
    (x1, y1) = (p2_0, p2_1)
    cdef double dx = x1 - x0
    cdef double dy = y1 - y0

    cdef double dy_by_dx = abs(dy / dx)
    cdef short  dir      = int(np.sign(dy))
    cdef int proper_x0 = int(np.ceil(x0))
    cdef int proper_x1 = int(np.floor(x1))

    cdef double weight_first = proper_x0 - x0
    cdef double weight_last  = x1 - proper_x1
    cdef int proper_y0 = 0
    cdef int proper_y1 = 0
    cdef double fst_offset = 0
    cdef int fst_y = 0
    cdef double lst_offset = 0

    cdef double cur_offset = 0
    cdef double a = 0
    cdef int cur_y
    cdef int cur_x
    cdef int tmp_n = 0

    if dir > 0:
        proper_y0 = int(np.floor(y0))
        proper_y1 = int(np.floor(y1))
    else:
        proper_y0 = int(np.ceil(y0))
        proper_y1 = int(np.ceil(y1))

    if weight_first != 0.0:
        fst_offset = np.abs(proper_y0 - y0) - dy_by_dx * (1 - weight_first)
        fst_y0 = proper_y0
        assert(fst_offset < 1)
        if fst_offset < 0:
            fst_offset += 1
            fst_y0 -= dir
        try:
            if switch_dim:
                s += (array[fst_y0, proper_x0 - 1] * (1 - fst_offset) +
                      array[fst_y0 + dir, proper_x0 - 1] * fst_offset) * weight_first
            else:
                s += (array[proper_x0 - 1, fst_y0] * (1 - fst_offset) +
                      array[proper_x0 - 1, fst_y0 + dir] * fst_offset) * weight_first
        except IndexError:
            pass

    if weight_last != 0.0:
        lst_offset = np.abs(proper_y1 - y1) + dy_by_dx * (1 - weight_last)
        assert(lst_offset > 0)
        if lst_offset > 1:
            lst_offset -= 1
            proper_y1 += dir
        try:
            if switch_dim:
                s += (array[proper_y1, proper_x1 + 1] * (1 - lst_offset) +
                      array[proper_y1 + dir, proper_x1 + 1] * lst_offset) * weight_last
            else:
                s += (array[proper_x1 + 1, proper_y1] * (1 - lst_offset) +
                      array[proper_x1 + 1, proper_y1 + dir] * lst_offset) * weight_last
        except IndexError:
            pass

    cur_offset = np.abs(proper_y0 - y0) + dy_by_dx * weight_first
    cur_y = proper_y0

    for (ll, cur_x) in enumerate(range(proper_x0, proper_x1 + 1)):
        if cur_offset > 1:
            cur_offset -= 1
            cur_y += dir
        if max(0, - dir) <= cur_y < min(max_y, max_y - dir) and 0 <= cur_x < max_x:
            if switch_dim:
                if width == 0:
                    a = array[cur_y, cur_x] * (1 - cur_offset) + \
                        array[cur_y + dir, cur_x] * cur_offset
                else:
                    a = np.sum(array[cur_y - width: cur_y + width + 1, cur_x])
            else:
                if width == 0:
                    a = array[cur_x, cur_y] * (1 - cur_offset) + \
                        array[cur_x, cur_y + dir] * cur_offset
                else:
                    a = np.sum(array[cur_x, cur_y - width: cur_y + width + 1])
            if a != 0.0:
                tmp_n += 1
            s += a
        cur_offset += dy_by_dx

    return tmp_n if count else s
