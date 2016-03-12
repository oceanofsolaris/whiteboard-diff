import pyximport; pyximport.install()
import cython_helpers
import numpy as np
import scipy
import cv2
import imutils
from skimage.transform import radon
import helpers
import itertools as it
from math import sin, cos

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


def estimate_aspect_ratio(rectangle, image_shape):
    print(image_shape)
    u0, v0 = [a / 2.0 for a in image_shape]
    m1, m2, m3, m4 = [np.concatenate((a, [1])) for a in rectangle]
    s = 1
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    print(k2, k3, n2, n3)
    fsquared = - 1 / (n2[2] * n3[2] * s ** 2) * \
               ((n2[0]*n3[0] - (n2[0]*n3[2] + n2[2]*n3[0])*u0 + n2[2]*n3[2]*u0**2)*s**2 \
               +(n2[1]*n3[1] - (n2[1]*n3[2] + n2[2]*n3[1])*v0 + n2[2]*n3[2]*v0**2))
    print(fsquared)
    f = np.sqrt(fsquared)
    print(f)
    A = np.array([[f, 0, u0], [0, s * f, v0], [0, 0, 1]])
    A_inv = np.linalg.inv(A)
    M = np.transpose(A_inv) @ A_inv
    w_by_h_squared = np.dot(n2, M @ n2) / np.dot(n3, M @ n3)
    ratio = np.sqrt(w_by_h_squared)
    return ratio if ratio < 1 else 1.0 / ratio


def estimate_aspect_ratio_fit(rectangle, image_shape):
    u0, v0 = [a / 2.0 for a in image_shape]

    def fit_target(x):
        res = camera_transform(x, u0, v0)
        return np.sum((rectangle - res) ** 2)

    pih = np.pi / 2
    init_val = [2, 10, 0.5, -2, -pih, 0, 0, 5000]
    bounds = [(0.2, 5), (0, None), (None, None), (None, None), (None, None),
              (None, None), (None, None), (0, None)]
    result = scipy.optimize.minimize(fit_target, init_val, bounds=bounds)
    x = result.x
    w, d, t_x, t_y, phi1, phi2, phi3, f = x
    return w, camera_transform(x, u0, v0)


def camera_transform(x, u0, v0):
    w, d, t_x, t_y, phi1, phi2, phi3, f = x
    s = 1
    A = np.array([[f,     0, u0],
                  [0, s * f, v0],
                  [0, 0    , 1 ]])
    h = 1
    translation = np.array([t_x, t_y, d])
    R1 = np.array([[cos(phi1), -sin(phi1), 0, 0],
                   [sin(phi1), +cos(phi1), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]).transpose()
    R2 = np.array([[+cos(phi2), 0, sin(phi2), 0],
                   [0, 1, 0, 0],
                   [-sin(phi2), 0, cos(phi2), 0],
                   [0, 0, 0, 1]]).transpose()
    R3 = np.array([[1, 0, 0, 0],
                   [0, +cos(phi3), sin(phi3), 0],
                   [0, -sin(phi3), cos(phi3), 0],
                   [0, 0, 0, 1]]).transpose()
    rotation = R1 @ R2 @ R3
    T = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  translation]).transpose()
    oldpoints = np.array([[0, 0, 0, 1],
                          [0, h, 0, 1],
                          [w, 0, 0, 1],
                          [w, h, 0, 1]]).transpose()
    preimage_transform = T @ rotation
    projection_transform = A
    total_transform = projection_transform @ preimage_transform
    image_points = total_transform @ oldpoints
    rectangle = np.array([p[0:2] / p[2] for p in image_points.transpose()])
    return rectangle
