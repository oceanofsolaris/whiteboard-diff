import pyximport; pyximport.install()
import cython_helpers
import helpers
import rectangle_finder as rf
import unittest
import numpy as np
import itertools

shortonly = True

class test_helpers_python_and_cython(unittest.TestCase):
    def wu_sum_compare(self, array, p1, p2, count=False, width=0, debug=False):
        val1 = helpers.wu_sum(array, p1, p2, count, width, debug)
        val2 = cython_helpers.wu_sum(array, p1, p2, count, width, debug)
        self.assertAlmostEqual(val1, val2, delta=1e-13)
        return val2

    def test_points_match(self):
        """Test whether the match function does indeed return true if the
        points are matched within the tolerance"""
        values = [([[1.2, 1.4]], [[1.3, 1.4]], 0.11, True),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.2, 1.6]], 0.25, True),
                  ([[1.2, 1.4]], [[0.0, 3.0], [1.2, 1.6]], 0.25, False),
                  ([[1.2, 1.4], [0.0, 3.0]], [[1.2, 1.6], [1.0, 2.0]], 0.25, False),
                  ([[1.2, 1.4], [0.0, 3.0]], [[1.0, 2.0], [1.2, 1.6]], 0.25, False),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.3, 1.45]], 0.1, False)]
        for pA, pB, rad, res_val in values:
            result = helpers.points_match(pA, pB, rad)
            self.assertEqual(res_val, result)

    def test_points_contained(self):
        values = [([[1.2, 1.4]], [[1.3, 1.4]], 0.11, True),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.2, 1.6]], 0.25, True),
                  ([[1.2, 1.4]], [[0.0, 3.0], [1.2, 1.6]], 0.25, True),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.3, 1.45]], 0.1, False),
                  ([[1.0, 1.0], [1.5, 1.2]], [[1.2, 1.2]], 0.31, True),
                  ([[1.0, 1.0], [1.5, 1.5]], [[1.2, 1.2]], 0.2, False),
                  ([[1.0, 1.0], [1.5, 1.2]], [[1.2, 1.2], [100, 100]], 0.31, True),
                  ([[1.0, 1.0], [1.5, 1.2]], [[1.2, 1.2], [100, 100]], 0.2, False),
                  ([[2.0, 2.0]], [[1.95, 2.0], [2.05, 2.0]], 0.1, True)]
        for pA, pB, rad, res_val in values:
            result = helpers.points_contained(pA, pB, rad)
            self.assertEqual(res_val, result)

    def test_wu_sum(self):
        """Test whether the wu-sum does what it is supposed to do."""
        tol = 1e-11
        dtype = np.uint8
        test_array = (np.iinfo(dtype).max * np.random.random(size=(200, 200))).astype(dtype)

        # Test whether the cython implementation yields the same
        # results as the python implementation (we implicitly check
        # for all following calls, but this is probably a sensible
        # test on its own as well)
        random_points = [np.random.rand(2, 1) * (test_array.shape[ii] - 1)
                         for ii in range(2)]
        val0_c = cython_helpers.wu_sum_cython(test_array,
                                              random_points[0][0],
                                              random_points[0][1],
                                              random_points[1][0],
                                              random_points[1][1])
        val0_p = helpers.wu_sum(test_array, random_points[0], random_points[1])
        self.assertAlmostEqual(val0_c, val0_p, delta=tol)

        # Test a straightforward sum over a straight, pixel-aligned line
        val1 = self.wu_sum_compare(test_array, (10, 5), (110, 5))
        val1c = np.sum(test_array[10:111, 5])
        self.assertAlmostEqual(val1, val1c, delta=tol)

        # If both lines propagate in the opposite direction and only
        # include the same pixels, their sum should be the same as two
        # straight lines
        val2 = self.wu_sum_compare(test_array, (10, 5), (110, 6)) + \
               self.wu_sum_compare(test_array, (10, 6), (110, 5))
        val2c = np.sum(test_array[10:111, 5]) + np.sum(test_array[10:111, 6])
        self.assertAlmostEqual(val2, val2c, delta=tol)

        # Test non-integer start and stop y
        val5 = self.wu_sum_compare(test_array, (10, 5.1), (110, 5.9)) + \
               self.wu_sum_compare(test_array, (10, 5.9), (110, 5.1))
        val5c = val2c
        self.assertAlmostEqual(val5, val5c, delta=tol)

        # Test non-integer start and stop y as well as non-integer start x
        val6 = self.wu_sum_compare(test_array, (9.5, 5.1), (110, 5.9)) + \
               self.wu_sum_compare(test_array, (9.5, 5.9), (110, 5.1))
        val6c = val2c + np.sum(test_array[9, 5:7]) * 0.5
        self.assertAlmostEqual(val6, val6c, delta=tol)

        # Test non-integer stop x with non-integer start and stop y
        val7 = self.wu_sum_compare(test_array, (10, 5.1), (110.5, 5.9)) + \
               self.wu_sum_compare(test_array, (10, 5.9), (110.5, 5.1))
        val7c = val2c + np.sum(test_array[111, 5:7]) * 0.5
        self.assertAlmostEqual(val7, val7c, delta=tol)

        # Test whether summing either row-wise or column-wise makes a
        # difference in the case of almost completely 45 degree lines
        val_x_l = self.wu_sum_compare(test_array, (10.1, 20.1), (110.1 + tol ,120.1))
        val_y_l = self.wu_sum_compare(test_array, (10.1, 20.1), (110.1 - tol, 120.1))
        self.assertAlmostEqual(val_x_l, val_y_l, delta=255 * 10 * tol)

        val2_x_l = self.wu_sum_compare(test_array, (20.3, 10.3), (120.1, 110.1 + tol))
        val2_y_l = self.wu_sum_compare(test_array, (20.3, 10.3), (120.1, 110.1 - tol))
        self.assertAlmostEqual(val2_x_l, val2_y_l, delta=255 * 10 * tol)

        # Test whether transposing the array and the summation range yields the same results
        random_points = [np.random.rand(2, 1) * (test_array.shape[ii] - 1)
                         for ii in range(2)]
        val8_1 = self.wu_sum_compare(test_array, random_points[0], random_points[1])
        val8_2 = self.wu_sum_compare(test_array, random_points[1], random_points[0])
        random_points_t = [a[-1::-1] for a in random_points]
        val8_3 = self.wu_sum_compare(test_array.transpose(), random_points_t[0],
                                                        random_points_t[1])
        self.assertAlmostEqual(val8_1, val8_2, delta=tol)
        self.assertAlmostEqual(val8_1, val8_3, delta=tol)

        # Test that we can also walk out of the array without negative
        # consequences (points outside the array are not counted
        val9_1 = self.wu_sum_compare(test_array, (100, 0), (100, 100))
        val9_2 = self.wu_sum_compare(test_array, (100, -10), (100, 100))
        self.assertAlmostEqual(val9_1, val9_2, delta=tol)

        # Test that the cython implementation also correctly covers thick lines
        self.wu_sum_compare(test_array, (10, 100), (20, 120), width=4)

    def test_find_intersection(self):
        val_1_f = helpers.find_intersection(np.array([[1, 0], [0, 1]]),
                                            np.array([[0, 0], [2, 0]]))
        val_1_c = np.array([1, 0])
        self.assertTrue(np.sum(np.abs((val_1_f - val_1_c))) < 1e-13)

        val_2_f = helpers.find_intersection(np.array([[1, 0], [0, 1]]),
                                            np.array([[-1, 0], [0, -1]]))
        self.assertIsNone(val_2_f)

        val_3_f = helpers.find_intersection(np.array([[0, 1], [0, 2]]),
                                            np.array([[1, 0], [2, 0]]))
        val_3_c = np.array([0, 0])
        self.assertTrue(np.sum(np.abs((val_3_f - val_3_c))) < 1e-13)

        # The intersection_point should not depend on the order of the
        # two lines
        line_1_1 = np.random.rand(2, 2)
        line_1_2 = np.random.rand(2, 2)
        val_4_1 = helpers.find_intersection(line_1_1, line_1_2)
        val_4_2 = helpers.find_intersection(line_1_2, line_1_1)
        self.assertTrue(np.sum(np.abs(val_4_1 - val_4_2)) < 1e-13)

        # If two lines contain a common point, it has to be the
        # intersection point (barring that the other point is exactly
        # the same...which should never happen if it is choosen
        # randomly)
        line_2_1 = np.random.rand(2, 2)
        line_2_2 = np.random.rand(2, 2)
        line_2_2[1, :] = line_2_1[1, :]
        val_5 = helpers.find_intersection(line_2_1, line_2_2)
        self.assertTrue(np.sum(np.abs(val_5 - line_2_1[1, :])) < 1e-13)

    def get_corners_permutations(self, lines):
        ls_s = itertools.permutations(lines)
        [val, _] = helpers.get_corners(next(ls_s))
        for ls in ls_s:
            [val_p, _] = helpers.get_corners(ls)
            if val is None:
                self.assertIsNone(val_p)
            else:
                self.assertTrue(helpers.points_match(val, val_p, tol=1e-13))
        return val

    def test_get_corners(self):
        val_1 = self.get_corners_permutations([np.array(a) for a in
                                               [[[0, 0], [0, 1.1]],
                                                [[0, 1.1], [1, 1]],
                                                [[1, 1], [1.1, 0]],
                                                [[1.1, 0], [0, 0]]]])
        corners_c_1 = [[0, 0], [1.1, 0], [1, 1], [0, 1.1]]
        self.assertTrue(helpers.points_match(val_1, corners_c_1, tol=1e-13))

        val_2 = self.get_corners_permutations([np.array(a) for a in
                                               [[[0, 0], [0, 1]],
                                                [[0, 1], [1, 1]],
                                                [[1, 1], [1, 0]],
                                                [[1, 0], [0, 0]]]])
        corners_c_2 = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertTrue(helpers.points_match(val_2, corners_c_2, tol=1e-13))

        val_3 = self.get_corners_permutations([np.array(a) for a in
                                               [[[0, 1], [1, 0]],
                                                [[8, 2], [2, 1]],
                                                [[0, 1], [1, 0]],
                                                [[3, 4], [4, 3]]]])
        self.assertIsNone(val_3)

    def test_get_pairs(self):
        l = [1, 2, 3, 4]
        val_f = list(helpers.get_pairs(iter(l)))
        val_c = [(1, 2), (2, 3), (3, 4)]
        self.assertEqual(val_f, val_c)

    def test_get_pairs_cycle(self):
        l = [1, 2, 3, 4]
        val_f = list(helpers.get_pairs_cycle(iter(l)))
        val_c = [(1, 2), (2, 3), (3, 4), (4, 1)]
        self.assertEqual(val_f, val_c)

    def test_sino_line_conversions(self):
        shape = [int(a) for a in (np.random.rand(2) * 2000)]
        for nn in range(180):
            (phi, offset) = np.random.rand(2) * \
                            np.array([180, max(shape) * np.sqrt(2)])
            (offset_r, phi_r) = helpers.line_to_sino(helpers.sino_to_line(offset, phi, shape), shape)
            self.assertAlmostEqual(phi_r, phi, delta=10 * 1e-13)
            self.assertAlmostEqual(offset_r, offset, delta=10 * 1e-13)
        # We should definitely test some line to sino transformations
        # by hand as well

    def test_rectangle_area(self):
        rectangle_points_1 = [[0, 0],
                              [1, 1],
                              [1, 2],
                              [0, 1]]
        val1_f = helpers.rectangle_area(rectangle_points_1)
        val1_c = 1.0
        self.assertAlmostEqual(val1_f, val1_c, delta=1e-13)

        rectangle_points_2 = [[0, 0],
                              [1, 0],
                              [1, 1],
                              [0, 1]]
        val2_f = helpers.rectangle_area(rectangle_points_2)
        val2_c = 1.0
        self.assertAlmostEqual(val2_f, val2_c, delta=1e-13)

        rectangle_points_3 = [[0, 1],
                              [1, 2],
                              [1, 1],
                              [0, 0]]
        val3_f = helpers.rectangle_area(rectangle_points_3)
        val3_c = val1_c
        self.assertAlmostEqual(val3_f, val3_c, delta=1e-13)

    def test_clamp_sigmoid(self):
        val1_f = helpers.clamp_sigmoid(1.7, 1.0, 2.0)
        self.assertAlmostEqual(val1_f, 0.7, delta=1e-13)

        val2_f = helpers.clamp_sigmoid(1.7, 1.5, 2.0)
        self.assertAlmostEqual(val2_f, 0.4, delta=1e-13)


class test_rectangle_finder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.sample_image = rf.loadimage('../test_whiteboards/cellphone_samples/whiteboard_skewed.jpg')
        (self.contours, self.contour_ratio) = rf.get_small_edge(self.sample_image)
        if not shortonly:
            self.radon = rf.radon_transform(self.contours)
        pass

    def test_refine_line(self):
        pass

    def test_overlap_sum(self):
        pass

    def test_poor_mans_sino(self):
        # Check whether both sino methods result in similar enough peaks
        if shortonly:
            return
        radon_poor = rf.poor_mans_sino(self.contours)

        peaks_full, v_f = rf.get_blurred_peaks(self.radon, width=5)
        peaks_poor, v_p = rf.get_blurred_peaks(radon_poor, width=5)

        self.assertTrue(helpers.points_contained(peaks_full[0:6], peaks_poor[0:20], 7))
        self.assertTrue(helpers.points_contained(peaks_poor[0:6], peaks_full[0:20], 7))

    def test_find_rectangle(self):
        rectangle = rf.find_rectangle(self.contours)
        original_corners = np.array([[138, 2512],
                                     [2362, 2556],
                                     [1792, 511],
                                     [568, 436]])
        scaled_corners = original_corners / self.contour_ratio
        self.assertTrue(helpers.points_match(rectangle, scaled_corners,
                                             80 / self.contour_ratio))

    def dilate_compare(self, A, width):
        val_p = rf.dilate(A, width)
        val_c = cython_helpers.dilate(A, width)
        self.assertTrue(np.all(val_p == val_c))

    def test_dilate(self):
        # Test that dilate behaves the same in the cython and the
        # python implementation
        test_array = np.random.random(size=(200, 200))
        self.dilate_compare(test_array, width=5)
        
        # This is a shitty test. I should think of something
        # better. For now it works fine though, since I kind of trust
        # the python implementation to be correct and only use it to
        # make sure the cython implementation stays correct during a
        # rewrite

    def test_point_distance_sum(self):
        for ii in range(10):
            points = np.random.random((4, 2)) * 100
            dists = np.random.random((4, 2)) * 0.01
            points2 = points + dists
            totaldist = np.sum([np.sum(points ** 2) for points in dists])
            np.random.shuffle(points2)
            fctdist = rf.point_distance_sum(points, points2)
            self.assertAlmostEqual(fctdist, totaldist, delta=1e-13)

    def test_estimate_aspect_ratio_fit(self):
        square_points = [ [1, 0],
                          [0, 0],
                          [1.001, 1],
                          [0, 1.001]]
        square = np.array([np.array(a) for a in square_points])
        square_ratio, _ = rf.estimate_aspect_ratio_fit(square, (0.5, 0.5))
        self.assertAlmostEqual(square_ratio, 1.0, delta=1e-2)


if __name__ == "__main__":
    run_all_tests = True
    defaultTest  = None if run_all_tests else \
                   ['test_helpers']
    unittest.main(defaultTest=defaultTest)
