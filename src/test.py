import helpers
import rectangle_finder
import unittest
import numpy as np


class test_helpers(unittest.TestCase):
    def test_points_match(self):
        """Test whether the match function does indeed return true if the
        points are matched within the tolerance"""
        values = [([[1.2, 1.4]], [[1.3, 1.4]], 0.11, True),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.2, 1.6]], 0.25, True),
                  ([[1.2, 1.4]], [[0.0, 3.0], [1.2, 1.6]], 0.25, False),
                  ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.3, 1.45]], 0.1, False)]
        for pA, pB, rad, res_val in values:
            result = helpers.points_match(pA, pB, rad)
            self.assertEqual(res_val, result)

    def test_wu_sum(self):
        """Test whether the wu-sum does what it is supposed to do."""
        tol = 1e-13
        test_array = np.random.random(size=(200, 200))

        # Test a straightforward sum over a straight, pixel-aligned line
        val1 = helpers.wu_sum(test_array, (10, 5), (110, 5))
        val1c = np.sum(test_array[10:111, 5])
        self.assertAlmostEqual(val1, val1c, delta=tol)

        # If both lines propagate in the opposite direction and only
        # include the same pixels, their sum should be the same as two
        # straight lines
        val2 = helpers.wu_sum(test_array, (10, 5), (110, 6)) + \
               helpers.wu_sum(test_array, (10, 6), (110, 5))
        val2c = np.sum(test_array[10:111, 5]) + np.sum(test_array[10:111, 6])
        self.assertAlmostEqual(val2, val2c, delta=tol)

        # Test non-integer start and stop y
        val5 = helpers.wu_sum(test_array, (10, 5.1), (110, 5.9)) + \
               helpers.wu_sum(test_array, (10, 5.9), (110, 5.1))
        val5c = val2c
        self.assertAlmostEqual(val5, val5c, delta=tol)

        # Test non-integer start and stop y as well as non-integer start x
        val6 = helpers.wu_sum(test_array, (9.5, 5.1), (110, 5.9)) + \
               helpers.wu_sum(test_array, (9.5, 5.9), (110, 5.1))
        val6c = val2c + np.sum(test_array[9, 5:7]) * 0.5
        self.assertAlmostEqual(val6, val6c, delta=tol)

        # Test non-integer stop x with non-integer start and stop y
        val7 = helpers.wu_sum(test_array, (10, 5.1), (110.5, 5.9)) + \
               helpers.wu_sum(test_array, (10, 5.9), (110.5, 5.1))
        val7c = val2c + np.sum(test_array[111, 5:7]) * 0.5
        self.assertAlmostEqual(val7, val7c, delta=tol)

        # Test whether summing either row-wise or column-wise makes a
        # difference in the case of almost completely 45 degree lines
        val_x_l = helpers.wu_sum(test_array, (10.1, 20.1), (110.1 + tol ,120.1))
        val_y_l = helpers.wu_sum(test_array, (10.1, 20.1), (110.1 - tol, 120.1))
        self.assertAlmostEqual(val_x_l, val_y_l, delta=10 * tol)

        val2_x_l = helpers.wu_sum(test_array, (20.3, 10.3), (120.1, 110.1 + tol))
        val2_y_l = helpers.wu_sum(test_array, (20.3, 10.3), (120.1, 110.1 - tol))
        self.assertAlmostEqual(val2_x_l, val2_y_l, delta=10 * tol)

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

        line_1 = np.random.rand(2, 2)
        line_2 = np.random.rand(2, 2)
        val_4_1 = helpers.find_intersection(line_1, line_2)
        val_4_2 = helpers.find_intersection(line_2, line_1)
        self.assertTrue(np.sum(np.abs((val_4_1 - val_4_2))) < 1e-13)

    def test_get_corners(self):
        [val_1, l_s] = helpers.get_corners([np.array(a) for a in
                                     [[[0, 0], [0, 1.1]],
                                      [[0, 1.1], [1, 1]],
                                      [[1, 1], [1.1, 0]],
                                      [[1.1, 0], [0, 0]]]])
        corners_c_1 = [[0, 0], [1.1, 0], [1, 1], [0, 1.1]]
        self.assertTrue(helpers.points_match(val_1, corners_c_1, tol=1e-13))

        [val_2, l_s] = helpers.get_corners([np.array(a) for a in
                                     [[[0, 0], [0, 1]],
                                      [[0, 1], [1, 1]],
                                      [[1, 1], [1, 0]],
                                      [[1, 0], [0, 0]]]])
        corners_c_2 = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertTrue(helpers.points_match(val_2, corners_c_2, tol=1e-13))

        
# class test_helpers(unittest.TestCase):
#     def test_refine_line(self):
#         pass

#     def test_overlap_sum(self):
#         pass
        
if __name__ == "__main__":
    unittest.main()
