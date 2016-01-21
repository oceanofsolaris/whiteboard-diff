import numpy as np
import cv2
import imutils
from skimage.transform import radon
import helpers

size_small = 500

loadimage = lambda filename: cv2.imread(filename)

desaturate = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

radon_transform = lambda image: radon(image, circle=False)

gaussian_blur = lambda image, width: cv2.GaussianBlur(image, (width, width), 1)


def sino_to_line(angle, offset, shape):
    middle = np.array([shape[1] / 2, shape[0] / 2])
    max_offset = np.sqrt(2) * np.max(shape)
    offset_to_middle = offset - (max_offset / 2)
    phi = angle / 180 * np.pi
    offset_dir_x = np.cos(phi)
    offset_dir_y = -np.sin(phi)
    offset_v = np.array([offset_dir_x * offset_to_middle, offset_dir_y * offset_to_middle])
    point_of_line = middle + offset_v
    return [point_of_line, point_of_line + 20 * np.array([offset_dir_y, -offset_dir_x])]


def line_to_sino(line, shape):
    (x0, y0) = line[0]
    (x1, y1) = line[1]
    angle = np.arctan2(x1 - x0, y1 - y0)
    angle = angle % np.pi
    normal = np.array([np.cos(angle), -np.sin(angle)])
    middle = np.array([shape[1] / 2, shape[0] / 2])
    dist = np.sum((line[0] - middle) * normal)

    max_offset = np.sqrt(2) * np.max(shape)
    return_offset = dist + max_offset / 2
    assert(return_offset < max_offset)
    return (angle * 180.0 / np.pi, return_offset)


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
            (angle, offset) = line_to_sino([a[0], b[0]], orig_shape)
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
    flat = dilate(A, width)
    print(A.shape, flat.shape)
    peaks = np.transpose(np.nonzero((flat == A) * A))
    vals = [A[x, y] for (x, y) in peaks]
    return peaks[np.argsort(vals)][::-1], np.sort(vals)[::-1]
