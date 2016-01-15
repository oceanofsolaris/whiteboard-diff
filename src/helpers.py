import numpy as np


def points_match(pA, pB, tol=1):
    """Check whether two arrays of points contain (approximately) the same
    points. Two points are approximately the same if their distance is
    less than 'tol'. Behavior can get weird if multiple points are
    close together.
    """
    pA = np.array(pA)
    pB = np.array(pB)

    if len(pA) != len(pB):
        return False

    tolSq = tol ** 2

    for val in pA:
        dists = (pB - val) ** 2
        euclid_dists = dists[:, 0] + dists[:, 1]
        best_match = np.argmin(euclid_dists)

        if euclid_dists[best_match] > tolSq:
            return False
        else:
            pB[best_match] = np.array([np.NaN, np.NaN])
    return True


def wu_sort_points(p1, p2, dim):
    """This is a helper function for wu_sum in order to bring the points
    into the right order."""
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
        if switch_dim:
            s += (array[fst_y0, proper_x0 - 1] * (1 - fst_offset) +
                  array[fst_y0 + dir, proper_x0 - 1] * fst_offset) * weight_first
        else:
            s += (array[proper_x0 - 1, fst_y0] * (1 - fst_offset) +
                  array[proper_x0 - 1, fst_y0 + dir] * fst_offset) * weight_first

    if weight_last != 0.0:
        lst_offset = np.abs(proper_y1 - y1) + dy_by_dx * (1 - weight_last)
        assert(lst_offset > 0)
        if lst_offset > 1:
            lst_offset -= 1
            proper_y1 += dir
        if switch_dim:
            s += (array[proper_y1, proper_x1 + 1] * (1 - lst_offset) +
                  array[proper_y1 + dir, proper_x1 + 1] * lst_offset) * weight_last
        else:
            s += (array[proper_x1 + 1, proper_y1] * (1 - lst_offset) +
                  array[proper_x1 + 1, proper_y1 + dir] * lst_offset) * weight_last

    cur_offset = np.abs(proper_y0 - y0) + dy_by_dx * weight_first
    cur_y = proper_y0
    x0 = proper_x0
    x1 = proper_x1

    tmp_n = 0
    try:
        for (ll, cur_x) in enumerate(range(x0, x1 + 1)):
            a = 0
            if cur_offset > 1:
                cur_offset -= 1
                cur_y += dir
            if switch_dim:
                a = array[cur_y, cur_x] * (1 - cur_offset) + \
                    array[cur_y + dir, cur_x] * cur_offset
            else:
                a = array[cur_x, cur_y] * (1 - cur_offset) + \
                    array[cur_x, cur_y + dir] * cur_offset
            if a != 0.0:
                tmp_n += 1
            s += a
            cur_offset += dy_by_dx

    except IndexError:
        print("IndexError:", cur_x, cur_y, p1, p2, switch_dim, array.shape, s)

    return tmp_n if count else s


def find_intersection(line1, line2):
    """Find the intersection point between two lines. The lines are given
    by a pair of points on them."""
    x = line1[1] - line1[0]
    y = line2[1] - line2[0]
    c = line2[0] - line1[0]
    if(min(abs(x[0]), abs(y[1])) < min(abs(x[1]), abs(y[0]))):
        # To avoid divide by zero error, we switch the two lines
        return find_intersection(line2, line1)
    elif np.all([np.abs(a) < 1e-15 for a in [x[0], y[0]]]) or \
         np.all([np.abs(a) < 1e-15 for a in [x[1], y[1]]]):
        # This checks for parallel lines for which the following
        # formulas would produce nasty warnings
        return None
    # Small derivation of the used formula:
    # a * x + b * y = c
    # a = (c[0] - b * y[0]) / x[0]
    # b = (c[1] - a * x[1]) / y[1]
    # b = (c[1] - (c[0] - b * y[0]) / x[0] * x[1]) / y[1]
    # b * (1 - y[0] / x[0] * x[1] / y[1]) = c[1] / y[1] - c[0] * x[1] / x[0] / y[1]
    # Silence 'division by zero' errors
    old_err = np.geterr()['divide']
    np.seterr(divide='ignore')
    b = (c[1] / y[1] - c[0] * x[1] / x[0] / y[1]) / (1 - y[0] / x[0] * x[1] / y[1])
    a = (c[0] - b * y[0]) / x[0]
    np.seterr(divide=old_err)
    return np.array(line1[0] + a * (line1[1] - line1[0])) \
        if not np.any([np.isinf(coeff) or np.isnan(coeff) for coeff in [a, b]]) else None


def find_intersections(line1, otherlines):
    """Find intersections between a line and a set of other lines."""
    return [find_intersection(line1, ol) for ol in otherlines]


def order_points(points):
    """Orders points (given as list of coordinate tuples) that lie on a line.
    Tries to preserve order. Gives reordering as tuple of indices."""
    line_vec = points[1] - points[0]
    origin = points[0]
    distances = [np.dot(p - origin, line_vec) for p in points]
    return np.argsort(distances)


def get_corners(lines):
    """Find the corners of a rectangle given by four lines. Probably
    crashes if two lines are exactly parallel."""
    # Look for parallel lines first
    lines = [np.array(l) for l in lines]
    parallels = [np.argwhere([find_intersection(lines[jj], lines[ii]) is None
                             for jj in range(ii + 1, len(lines))])
                 for ii in range(len(lines))]
    if np.any([len(p) > 1 for p in parallels]):
        return None
    par_lines = np.argwhere([len(p) > 0 for p in parallels])
    if len(par_lines) > 0:
        l1 = par_lines[0, 0]
        l2 = parallels[l1][0, 0] + l1 + 1
        line1 = lines[l1]
        line2 = lines[l2]
        lines.pop(max(l1, l2))
        lines.pop(min(l1, l2))
        lines = [line1, line2] + lines

    # If we had any parallel lines, they are now in position 0 and 1
    l = lines[0]
    otherlines = lines[1:]
    for (ii, ol) in enumerate(otherlines):
        inter = find_intersection(l, ol)
        rest = otherlines.copy()
        del rest[ii]
        if inter is None:
            break
        inters1 = [inter] + find_intersections(l, rest)
        inters2 = [inter] + find_intersections(ol, rest)
        if np.all(order_points(inters1) == order_points(inters2)):
            # The two lines are opposite in the rectangle
            break
    lines = [l, rest[0], ol, rest[1]]
    corners = [find_intersection(lines[ii], lines[(ii + 1) % 4]) for ii in range(4)]
    return (corners, lines)
