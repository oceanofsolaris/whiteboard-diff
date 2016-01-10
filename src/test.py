import helpers
import rectangle_finder
import unittest
import numpy as np

class test_helpers(unittest.TestCase):
    def test_pointers_match(self):
        '''Test whether the match function does indeed return true if the points are matched within the tolerance'''
        values = [ ([[1.2, 1.4]], [[1.3, 1.4]], 0.11, True),
                   ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.2, 1.6]], 0.25, True),
                   ([[1.2, 1.4]], [[0.0, 3.0], [1.2, 1.6]], 0.25, False),
                   ([[1.2, 1.4], [0.1, 2.9]], [[0.0, 3.0], [1.3, 1.45]], 0.1, False)]
        for pA, pB, rad, res_val in values:
            result = helpers.points_match(pA, pB, rad)
            self.assertEqual(res_val, result)

    def test_wu_sum(self):
        '''Test whether the wu-sum does what it is supposed to do.'''
        tol = 1e-13
        test_array = np.random.random(size=(200, 200))

        # Test a straightforward sum over a straight, pixel-aligned line
        # val1 = helpers.wu_sum(test_array, (10, 5), (110, 5))
        # val1c = np.sum(test_array[10:111, 5])
        # self.assertTrue(np.abs(val1 - val1c) < tol)

        # If both lines propagate in the opposite direction and only
        # include the same pixels, their sum should be the same as two
        # straight lines
        # val2 = helpers.wu_sum(test_array, (10, 5), (110, 6)) + \
        #        helpers.wu_sum(test_array, (10, 6), (110, 5))
        # val2c = np.sum(test_array[10:111, 5]) + np.sum(test_array[10:111, 6])
        # self.assertTrue(np.abs(val2 - val2c) < tol)

        # # Test whether everything still works if we have a non-integer start x
        # val3 = helpers.wu_sum(test_array, (9.5, 5), (110, 6)) + \
        #        helpers.wu_sum(test_array, (9.5, 6), (110, 5))
        # val3c = val2c + np.sum(test_array[9, 5:7]) * 0.5
        # self.assertTrue(np.abs(val3 - val3c) < tol)

        # # Test a non-integer stop x
        # val4 = helpers.wu_sum(test_array, (10, 5), (110.5, 6)) + \
        #        helpers.wu_sum(test_array, (10, 6), (110.5, 5))
        # val4c = val2c + np.sum(test_array[111, 5:7]) * 0.5
        # self.assertTrue(np.abs(val4 - val4c) < tol)

        # # Test non-integer start and stop y
        # val5 = helpers.wu_sum(test_array, (10, 5.1), (110, 5.9)) + \
        #        helpers.wu_sum(test_array, (10, 5.9), (110, 5.1))
        # val5c = val2c
        # self.assertTrue(np.abs(val5 - val5c) < tol)

        # # Test non-integer start and stop y as well as non-integer start x
        # val6 = helpers.wu_sum(test_array, (9.5, 5.1), (110, 5.9)) + \
        #        helpers.wu_sum(test_array, (9.5, 5.9), (110, 5.1))
        # val6c = val3c
        # self.assertTrue(np.abs(val6 - val6c) < tol)

        # # Test non-integer stop x with non-integer start and stop y
        # val7 = helpers.wu_sum(test_array, (10, 5.1), (110.5, 5.9)) + \
        #        helpers.wu_sum(test_array, (10, 5.9), (110.5, 5.1))
        # val7c = val4c
        # self.assertTrue(np.abs(val7 - val7c) < tol)

        # Test whether summing either row-wise or column-wise makes a
        # difference in the case of almost completely 45 degree lines
        val_x_l = helpers.wu_sum(test_array, (10.1, 20.1), (110.1 + tol ,120.1))
        val_y_l = helpers.wu_sum(test_array, (10.1, 20.1), (110.1 - tol, 120.1))
        self.assertAlmostEqual(val_x_l, val_y_l, delta=10 * tol)

        val2_x_l = helpers.wu_sum(test_array, (20.3, 10.3), (120.1, 110.1 + tol))
        val2_y_l = helpers.wu_sum(test_array, (20.3, 10.3), (120.1, 110.1 - tol))
        self.assertAlmostEqual(val2_x_l, val2_y_l, delta=10 * tol)

        
if __name__ == "__main__":
    unittest.main()
