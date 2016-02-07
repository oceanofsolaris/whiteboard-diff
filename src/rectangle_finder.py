import numpy as np
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


def get_small_edge(image):
    ratio = image.shape[0] / size_small
    image_res = imutils.resize(image, height=size_small)
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
    "This function replaces every value in the 2D array by the maximum in"
    "the region +- width."
    # This function is slow as molasses. Rewrite in cython or so.
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
    flat = dilate(A, width)
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


def line_quality(l, contour_image):
    quality = helpers.wu_average(contour_image, l[0], l[1], count=False)
    return quality


def rectangle_quality(rectangle, contour_image, cutoffs=(0.4, 0.8)):
    lines = helpers.get_pairs_cycle(iter(rectangle))
    # TODO: test for good cutoff values (maybe make these more dynamic
    lower_cutoff = cutoffs[0]
    upper_cutoff = cutoffs[1]
    sigma = lambda x: helpers.clamp_sigmoid(x, lower_cutoff, upper_cutoff)
    line_qualities_raw = [line_quality(l, contour_image) for l in lines]
    line_qualities = [sigma(q) for q in line_qualities_raw]
    # print(rectangle)
    # print(line_qualities_raw)
    # print(line_qualities)
    area = helpers.rectangle_area(rectangle)
    quality = np.min(line_qualities) * area
    return quality


def find_rectangle(contour_image):
    sino = poor_mans_sino(contour_image)
    c_shape = contour_image.shape
    candidates, c_vals = get_blurred_peaks(sino)
    candidate_lines = [helpers.sino_to_line(c[0], c[1], c_shape) for c in candidates]
    middle = sino.shape[0] / 2.0
    bucket_areas = [[[60, 120], [middle * 1.05, middle * 2]],
                    [[60, 120], [0, middle * 0.7]],
                    [[-1, 30], [middle * 1.05, middle * 2]],
                    [[-1, 30], [0, middle * 0.7]]]
    candidate_buckets = helpers.into_buckets(candidates, bucket_areas)
    bucket_depths = [len(b) - 1 for b in candidate_buckets]
    max_evals = 30
    rectangles = []
    for (ii, c) in it.islice(enumerate(helpers.allcombinations(bucket_depths)), max_evals):
        rectangle_candidate = [candidate_buckets[ll][c[ll]]
                               for ll in range(len(bucket_depths))]
        rect_lines = [candidate_lines[c] for c in rectangle_candidate]
        corners, lines = helpers.get_corners(rect_lines)
        quality = rectangle_quality(corners, contour_image)
        rectangles.append((quality, corners))
    rectangles.sort(key=lambda x:x[0])
    return rectangles[0][1]
