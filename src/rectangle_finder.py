import pyximport; pyximport.install()
import cython_helpers
import numpy as np
import scipy
import cv2
import imutils
from skimage.transform import radon
import helpers
import itertools as it

size_small = 500

loadimage = lambda filename: cv2.imread(filename)

desaturate = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

radon_transform = lambda image: radon(image, circle=False)

gaussian_blur = lambda image, width: cv2.GaussianBlur(image, (width, width), 1)


def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower_threshold = int(max(0, 1.0 - sigma) * median)
    upper_threshold = int(min(255, 1.0 + sigma) * median)
    edged = cv2.Canny(image, lower_threshold, upper_threshold)
    return edged


def get_small_edge(image, target_height=size_small):
    ratio = image.shape[0] / target_height
    if ratio != 1.0:
        image_res = imutils.resize(image, height=target_height)
    else:
        image_res = image
    return [auto_canny(desaturate(image_res), sigma=0.5), ratio]


def poor_mans_sino(contour_image):
    orig_shape = np.array(contour_image.shape)
    sino_size = (int(np.ceil(np.sqrt(2) * np.max(orig_shape))), 180)
    sino = np.zeros(sino_size, dtype=np.float64)
    (_, cnts, _) = cv2.findContours(contour_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    for cnt in cnts:
        for (a, b) in helpers.get_pairs(iter(cnt)):
            pt1 = [a[0][1], a[0][0]]
            pt2 = [b[0][1], b[0][0]]
            (offset, angle) = helpers.line_to_sino([pt1, pt2], orig_shape)
            (angle_r, offset_r) = [int(np.floor(item)) for item in (angle, offset)]
            sino[offset_r, angle_r] += np.sqrt(np.sum((a - b) ** 2))
    return sino


def dilate(A, width):
    """This function replaces every value in the 2D array by the maximum in
    the region +- width. This function is slow as molasses. Use the
    cython implementation in cython_helpers instead.
    """
    x_l = A.shape[0]
    y_l = A.shape[1]
    B = A.copy()
    for ii in range(x_l):
        for jj in range(y_l):
            xrange_l = max(ii - width, 0)
            xrange_u = min(ii + width, x_l)
            yrange_l = max(jj - width, 0)
            yrange_u = min(jj + width, y_l)
            B[ii, jj] = np.amax(A[xrange_l:xrange_u, yrange_l:yrange_u])
    return B


def get_peaks(A, width=5):
    """Finds local peaks in a 2d-array. Peaks are the maximum in an area
    given by +-width.
    """
    flat = cython_helpers.dilate(A, width)
    peaks = np.transpose(np.nonzero((flat == A) * A))
    vals = [A[x, y] for (x, y) in peaks]
    return peaks[np.argsort(vals)][::-1], np.sort(vals)[::-1]


def get_blurred_peaks(A, width=5, blurwidth=5):
    """Same as get_peaks(), but blurs the array first (using a gaussian
    blur of width blur_width). This is more reliable if the data is a
    bit noisy.
    """
    A_b = gaussian_blur(A, blurwidth)
    return get_peaks(A_b, width)


def line_quality(l, contour_image, width=3, debug=False):
    if debug:
        quality = helpers.wu_average(contour_image, l[0], l[1],
                                     count=True, width=width, debug=debug)
    else:
        quality = cython_helpers.wu_average_cython(contour_image,
                                                   l[0][0], l[0][1],
                                                   l[1][0], l[1][1],
                                                   count=True,
                                                   width=width)
    return quality


def rectangle_quality(rectangle, contour_image, cutoffs=(0.4, 0.95), debug=False):
    lines = helpers.get_pairs_cycle(iter(rectangle))
    # TODO: test for good cutoff values (maybe make these more dynamic
    lower_cutoff = cutoffs[0]
    upper_cutoff = cutoffs[1]
    sigma = lambda x: helpers.clamp_sigmoid(x, lower_cutoff, upper_cutoff)
    line_qualities_raw = [line_quality(l, contour_image, 3, debug)
                          for l in lines]
    line_qualities = [sigma(q) for q in line_qualities_raw]
    area = helpers.rectangle_area(rectangle)
    quality = np.sum(line_qualities) * np.min(line_qualities) * np.sqrt(np.sqrt(area))
    if debug:
        print(rectangle)
        print(line_qualities)
        print(line_qualities_raw)
        print(area, quality)
    return quality


def find_rectangle(contour_image, max_evals=(4 ** 4)):
    sino = poor_mans_sino(contour_image)
    c_shape = contour_image.shape
    candidates, c_vals = get_blurred_peaks(sino)
    candidate_lines = [helpers.sino_to_line(c[0], c[1], c_shape)
                       for c in candidates]
    middle = sino.shape[0] / 2.0
    bucket_areas = [[[60, 120], [middle * 1.05, middle * 2]],
                    [[60, 120], [0, middle * 0.95]],
                    [[-1, 30], [middle * 1.05, middle * 2]],
                    [[-1, 30], [0, middle * 0.95]]]
    candidate_buckets = helpers.into_buckets(candidates, bucket_areas)
    bucket_depths = [len(b) - 1 for b in candidate_buckets]
    rectangles = []
    for (ii, c) in it.islice(enumerate(helpers.allcombinations(bucket_depths)), max_evals):
        rectangle_candidate = [candidate_buckets[ll][c[ll]]
                               for ll in range(len(bucket_depths))]
        rect_lines = [candidate_lines[c] for c in rectangle_candidate]
        corners, lines = helpers.get_corners(rect_lines)
        quality = rectangle_quality(corners, contour_image)
        rectangles.append((quality, corners))
    rectangles.sort(key=lambda x: x[0])
    return rectangles[-1][1]


def optimize_rectangle(rectangle, contour_image, max_offset=10):
    lines = helpers.get_pairs_cycle(iter(rectangle))
    opt_lines = np.zeros((4, 2, 2), np.float64)
    for (ii,line) in enumerate(lines):
        opt_lines[ii, :, :] = optimize_line(line, contour_image, max_offset)
    return helpers.get_corners(opt_lines)[0]


def optimize_line(endpoints, contour_image, max_offset):
    coor_a = np.array(endpoints[0])
    coor_b = np.array(endpoints[1])
    x0, y0 = coor_a
    x1, y1 = coor_b

    angle = np.arctan2(x1 - x0, y1 - y0)
    normal_vec = np.array([np.cos(angle), np.sin(angle)])

    def opt_target(offsets):
        x_a = coor_a + normal_vec * offsets[0]
        x_b = coor_b + normal_vec * offsets[1]
        return -cython_helpers.wu_sum(contour_image, x_a, x_b)

    n_tries = 5
    rands = np.random.random((n_tries, 2)) * 2 * max_offset - max_offset
    # Always have one try start at the initially estimated position
    rands[0, :] = 0
    results = [scipy.optimize.minimize(opt_target, rands[ii, :],
                                       bounds=[[-max_offset, max_offset],
                                               [-max_offset, max_offset]],
                                       tol=1e-3)
               for ii in range(n_tries)]
    vals = [res.fun for res in results]
    best = np.argmin(vals)
    best_offsets = results[best].x

    return np.array([coor_a + normal_vec * best_offsets[0],
                     coor_b + normal_vec * best_offsets[1]])


def get_rectangle_from_image(image):
    (contour_image, ratio) = get_small_edge(image)
    rectangle = find_rectangle(contour_image)
    rectangle = optimize_rectangle(rectangle, contour_image, size_small / 50)
    (contour_large, _) = get_small_edge(image, target_height=image.shape[0])
    rectangle_large = [p * ratio for p in rectangle]
    rectangle_large = optimize_rectangle(rectangle_large, contour_large,
                                         image.shape[0] / 50)
    return np.array(rectangle_large)
