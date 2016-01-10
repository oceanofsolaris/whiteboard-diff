import numpy as np

def points_match(pA, pB, rad):
    pA = np.array(pA)
    pB = np.array(pB)

    if len(pA) != len(pB):
        return False
    
    radSq = rad**2

    for val in pA:
        dists = (pB - val)**2
        euclid_dists = dists[:,0] + dists[:,1]
        best_match = np.argmin(euclid_dists)

        if euclid_dists[best_match] > radSq:
            return False
        else:
            pB[best_match] = np.array([np.NaN, np.NaN])
    return True


def wu_sort_points(p1, p2, dim):
    switch_dim = abs(p1[0] - p2[0]) < abs(p1[1] - p2[1])
    if switch_dim:
        p1 = p1[::-1]
        p2 = p2[::-1]
    if p1[0] > p2[0]:
        (p1, p2) = (p2, p1)
    return (p1, p2, switch_dim)


def wu_sum(array, p1, p2, count=False):
    """This function is messy, slow and does not do proper bounds-checking.
    Refactor (maybe in Cython) at some point."""
    (p1, p2, switch_dim) = wu_sort_points(p1, p2, array.shape)
    s = 0
    (x0, y0) = p1
    (x1, y1) = p2
    dx = x1 - x0
    dy = y1 - y0

    dy_by_dx = abs(dy / dx)
    dir = int(np.sign(dy))

    proper_x0 = int(np.ceil(x0))
    proper_x1 = int(np.floor(x1))

    weight_first = proper_x0 - x0
    weight_last  = x1 - proper_x1

    if dir > 0:
        proper_y0 = int(np.floor(y0))
        proper_y1 = int(np.ceil(y1))
    else:
        proper_y0 = int(np.ceil(y0))
        proper_y1 = int(np.floor(y1))

    if weight_first != 0.0:
        fst_offset = np.abs(proper_y0 - y0) - dy_by_dx * (1 - weight_first)
        fst_y0 = proper_y0
        if fst_offset > 1:
            fst_offset -= 1
            fst_y0 += dir
        if switch_dim:
            s += (array[fst_y0, proper_x0 - 1] * (1 - fst_offset) +
                  array[fst_y0 + dir, proper_x0 - 1] * fst_offset) * weight_first
        else:
            s += (array[proper_x0 - 1, fst_y0] * (1 - fst_offset) +
                  array[proper_x0 - 1, fst_y0 + dir] * fst_offset) * weight_first

    if weight_last != 0.0:
        lst_offset = np.abs(proper_y1 - y1) + dy_by_dx * weight_last
        if lst_offset > 1:
            lst_offset -= 1
            proper_y1 += dir
        if switch_dim:
            s += (array[proper_y1 - dir, proper_x1 + 1] * (1 - lst_offset) +
                  array[proper_y1, proper_x1 + 1] * lst_offset) * weight_last
        else:
            s += (array[proper_x1 + 1, proper_y1 - dir] * (1 - lst_offset) +
                  array[proper_x1 + 1, proper_y1] * lst_offset) * weight_last

    cur_offset = np.abs(proper_y0 - y0) + dy_by_dx * weight_first
    cur_y = proper_y0
    x0 = proper_x0
    x1 = proper_x1

    tmp_n = 0
    try:
        for (ll, cur_x) in enumerate(range(x0, x1+1)):
            a = 0
            if cur_offset > 1:
                cur_offset -= 1
                cur_y += dir
            if switch_dim:
                a = array[cur_y, cur_x] * (1 - cur_offset) + array[cur_y + dir, cur_x] * cur_offset
            else:
                a = array[cur_x, cur_y] * (1 - cur_offset) + array[cur_x, cur_y + dir] * cur_offset
            if a != 0.0:
                tmp_n += 1
            s += a
            cur_offset += dy_by_dx


    except IndexError:
        print("IndexError:", cur_x, cur_y, p1, p2, switch_dim, array.shape, s)

    return tmp_n if count else s
