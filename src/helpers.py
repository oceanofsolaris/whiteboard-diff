import numpy as np
import matplotlib.pyplot as plt


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
            pB[best_match] = np.array([np.Inf, np.Inf])
    return True


def points_contained(pA, pB, tol=1):
    """Check whether all points in pA are (approximately) contained in
    pB. Each point in pB might be matched to multiple points in pA (this
    is different to points_match)."""

    pA = np.array(pA)
    pB = np.array(pB)

    tolSq = tol ** 2

    for val in pA:
        dists = (pB - val) ** 2
        euclid_dists = np.sum(dists, axis=1)
        if np.min(euclid_dists) > tolSq:
            return False
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


def wu_sum(array, p1, p2, count=False, width=0, debug=False):
    """Sum over the entries in array tha lie on the line between p1 and
    p2. Uses cheap aliasing for this sum.  Width changes this into a
    'fat' line that covers 1+width pixels. The algorithm is 'cheap'
    and not always completely accurate.
    This function is messy and slow. Refactor (maybe in Cython) at some
    point.

    """
    (p1, p2, switch_dim) = wu_sort_points(p1, p2, array.shape)
    max_x = array.shape[1] if switch_dim else array.shape[0]
    max_y = array.shape[0] if switch_dim else array.shape[1]
    min_x = 0
    min_y = 0
    s = 0.0
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
        if width == 0:
            max_y -= 1
    else:
        proper_y0 = int(np.ceil(y0))
        proper_y1 = int(np.ceil(y1))
        if width == 0:
            min_y += 1

    if width != 0:
        min_y += width
        max_y -= width

    if weight_first != 0.0 and width == 0:
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

    if weight_last != 0.0 and width == 0:
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

    tmp_n = 0
    for (ll, cur_x) in enumerate(range(proper_x0, proper_x1 + 1)):
        a = 0
        if cur_offset > 1:
            cur_offset -= 1
            cur_y += dir
        if min_y <= cur_y < max_y and min_x <= cur_x < max_x:
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

            if debug and a == 0.0:
                if switch_dim:
                    plt.scatter([cur_x], [cur_y])
                else:
                    plt.scatter([cur_y], [cur_x])
        cur_offset += dy_by_dx

    return tmp_n if count else s


def wu_average(array, p1, p2, count=False, width=0, debug=False):
    line_length = np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    val = wu_sum(array, p1, p2, count, width, debug=debug)
    return val / line_length


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
    #  a * x + b * y = c
    #  a = (c[0] - b * y[0]) / x[0]
    #  b = (c[1] - a * x[1]) / y[1]
    #  b = (c[1] - (c[0] - b * y[0]) / x[0] * x[1]) / y[1]
    #  b * (1 - y[0] / x[0] * x[1] / y[1]) = c[1] / y[1] - c[0] * x[1] / x[0] / y[1]
    # Silence 'division by zero' and 'invalid value' errors, since
    # these result in NaN or Inf anyways
    old_err = {a:np.geterr()[a] for a in ['divide', 'invalid']}
    np.seterr(divide='ignore', invalid='ignore')
    b = (c[1] / y[1] - c[0] * x[1] / x[0] / y[1]) / (1 - y[0] / x[0] * x[1] / y[1])
    a = (c[0] - b * y[0]) / x[0]
    np.seterr(**old_err)
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
    """Find the corners of a rectangle given by four lines."""
    # How to find the corners of a rectangle?
    # Idea: find two opposite sides first.
    #       These can be defined in the following way: When starting
    #       from their intersection AB, the intersections AC and AD
    #       need to have the same distance-order as BC and BD (or the
    #       triplet (AB,AC,AD) must have the same order as the triplet
    #       (AB,BC,BD)).

    # Check for parallel lines first
    lines = [np.array(l) for l in lines]
    parallels = [np.argwhere([find_intersection(lines[jj], lines[ii]) is None
                             for jj in range(ii + 1, len(lines))])
                 for ii in range(len(lines))]
    if np.any([len(p) > 1 for p in parallels]):
        return (None, None)
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
            # The two lines are parallel and thus have to be opposite
            # in the rectangle
            break
        inters1 = [inter] + find_intersections(l, rest)
        inters2 = [inter] + find_intersections(ol, rest)
        o1 = order_points(inters1)
        o2 = order_points(inters2)
        if np.all(o1 == o2) and o1[0] == 0:
            # The two lines are opposite in the rectangle
            break
    lines = [l, rest[0], ol, rest[1]]
    corners = [find_intersection(lines[ii], lines[(ii + 1) % 4])
               for ii in range(4)]
    return (corners, lines)


def combinations(n, maxes):
    """This is a generator that returns all possible ways to distribute
    n tokens into len(maxes) buckets and where bucket n can only hold
    maxes[n] items."""
    if n <= sum(maxes):
        if len(maxes) == 1:
            yield [n]
        else:
            for ii in range(min(n + 1, maxes[-1] + 1)):
                for r in combinations(n - ii, maxes[:-1]):
                    yield r + [ii]


def allcombinations(maxes):
    """This is a generator that returns all possible lists L of length
    len(maxes) with only integer entries and where for entry n holds:
    0<=L[n]<=maxes[n].
    The values are returned in the order of increasing sum(L).
    This is e.g. useful if we want to look at all possible combinations of
    items from len(maxes) buckets, where each bucket either holds infinitely
    many items or the items are ordered by their likelyhood."""
    for ii in range(sum(maxes) + 1):
        for c in combinations(ii, maxes):
            yield c


def get_pairs(l):
    fst = next(l)
    try:
        while True:
            snd = next(l)
            yield (fst, snd)
            fst = snd
    except StopIteration:
        pass


def get_pairs_cycle(l):
    fst = next(l)
    veryfirst = fst
    try:
        while True:
            snd = next(l)
            yield(fst, snd)
            fst = snd
    except StopIteration:
        yield (fst, veryfirst)


def sino_to_line(offset, angle, shape):
    middle = np.array([shape[0] / 2, shape[1] / 2])
    max_offset = np.sqrt(2) * np.max(shape)
    offset_to_middle = offset - (max_offset / 2)
    phi = angle / 180 * np.pi
    offset_dir_x = -np.sin(phi)
    offset_dir_y = np.cos(phi)
    offset_v = np.array([offset_dir_x * offset_to_middle, offset_dir_y * offset_to_middle])
    point_of_line = middle + offset_v
    return [point_of_line, point_of_line + 20 * np.array([offset_dir_y, -offset_dir_x])]


def line_to_sino(line, shape):
    (y0, x0) = line[0]
    (y1, x1) = line[1]
    angle = np.arctan2(x1 - x0, y1 - y0)
    angle = angle % np.pi
    normal = np.array([-np.sin(angle), np.cos(angle)])
    middle = np.array([shape[0] / 2, shape[1] / 2])
    dist = np.sum((line[0] - middle) * normal)
    max_offset = np.sqrt(2) * np.max(shape)
    return_offset = dist + max_offset / 2
    assert(return_offset < max_offset)
    return (return_offset, angle * 180.0 / np.pi)


def in_bucket(coords, bucket_size):
    """This returns true if the 2d coords lie within the limits given by
    bucket_size. The second dimension is assumed to be cyclic (it will
    usually be the angle).

    """
    x_min = bucket_size[0][0]
    x_max = bucket_size[0][1]
    y_min = bucket_size[1][0]
    y_max = bucket_size[1][1]
    y_match = coords[0] < y_max and coords[0] > y_min
    x_match = coords[1] < x_max and coords[1] > x_min if x_min < x_max else \
              not (coords[1] > x_max and coords[1] < x_min)
    return x_match and y_match


def into_buckets(coord_list, bucket_sizes):
    """Sorts the coords in the list into the buckets given by the bucket
    sizes given by the bucket sizes. Uses in_bucket() to do
    this. Preserves order.

    """
    buckets = [[]] * len(bucket_sizes)
    for (ii, c) in enumerate(coord_list):
        for (jj, b) in enumerate(bucket_sizes):
            if in_bucket(c, b):
                buckets[jj] = buckets[jj] + [ii]
    return buckets


def clamp_range(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def clamp_sigmoid(val, min_val, max_val):
    scaled = (val - min_val) / (max_val - min_val)
    return clamp_range(scaled, 0, 1)


def rectangle_area(rectangle_points):
    """This function calculates the area of a rectangle. If it is given a
    non-rectangle, it will fail though."""
    points = [np.array(p) for p in rectangle_points]
    area1 = np.cross(points[1] - points[0],
                     points[2] - points[1])
    area2 = np.cross(points[2] - points[3],
                     points[3] - points[0])
    return (abs(area1) + abs(area2)) / 2
