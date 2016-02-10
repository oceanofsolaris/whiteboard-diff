import pyximport; pyximport.install()
import cython_helpers
import helpers
import unittest
import numpy as np
import timeit
import test

class test_cython_helpers(unittest.TestCase):
    def test_wu_sum(self):
        """Test whether the wu-sum does what it is supposed to do."""
        tol = 1e-13
        test_array = np.random.random(size=(200, 200))

        # Test whether the cython implementation yields the same
        # results as the python implementation
        random_points = [np.random.rand(2, 1) * (test_array.shape[ii] - 1)
                         for ii in range(2)]
        val0_c = cython_helpers.wu_sum(test_array,
                                       random_points[0][0],
                                       random_points[0][1],
                                       random_points[1][0],
                                       random_points[1][1])
        val0_p = helpers.wu_sum(test_array, random_points[0], random_points[1])
        self.assertAlmostEqual(val0_c, val0_p, delta=tol)
        
        # Test a straightforward sum over a straight, pixel-aligned line
        val1 = cython_helpers.wu_sum(test_array, 10, 5, 110, 5)
        val1c = np.sum(test_array[10:111, 5])
        self.assertAlmostEqual(val1, val1c, delta=tol)

        # If both lines propagate in the opposite direction and only
        # include the same pixels, their sum should be the same as two
        # straight lines
        val2 = cython_helpers.wu_sum(test_array, 10, 5, 110, 6) + \
               cython_helpers.wu_sum(test_array, 10, 6, 110, 5)
        val2c = np.sum(test_array[10:111, 5]) + np.sum(test_array[10:111, 6])
        self.assertAlmostEqual(val2, val2c, delta=tol)

        # Test non-integer start and stop y
        val5 = cython_helpers.wu_sum(test_array, 10, 5.1, 110, 5.9) + \
               cython_helpers.wu_sum(test_array, 10, 5.9, 110, 5.1)
        val5c = val2c
        self.assertAlmostEqual(val5, val5c, delta=tol)

        # Test non-integer start and stop y as well as non-integer start x
        val6 = cython_helpers.wu_sum(test_array, 9.5, 5.1, 110, 5.9) + \
               cython_helpers.wu_sum(test_array, 9.5, 5.9, 110, 5.1)
        val6c = val2c + np.sum(test_array[9, 5:7]) * 0.5
        self.assertAlmostEqual(val6, val6c, delta=tol)

        # Test non-integer stop x with non-integer start and stop y
        val7 = cython_helpers.wu_sum(test_array, 10, 5.1, 110.5, 5.9) + \
               cython_helpers.wu_sum(test_array, 10, 5.9, 110.5, 5.1)
        val7c = val2c + np.sum(test_array[111, 5:7]) * 0.5
        self.assertAlmostEqual(val7, val7c, delta=tol)

        # Test whether summing either row-wise or column-wise makes a
        # difference in the case of almost completely 45 degree lines
        val_x_l = cython_helpers.wu_sum(test_array, 10.1, 20.1, 110.1 + tol ,120.1)
        val_y_l = cython_helpers.wu_sum(test_array, 10.1, 20.1, 110.1 - tol, 120.1)
        self.assertAlmostEqual(val_x_l, val_y_l, delta=10 * tol)

        val2_x_l = cython_helpers.wu_sum(test_array, 20.3, 10.3, 120.1, 110.1 + tol)
        val2_y_l = cython_helpers.wu_sum(test_array, 20.3, 10.3, 120.1, 110.1 - tol)
        self.assertAlmostEqual(val2_x_l, val2_y_l, delta=10 * tol)

        # Test whether transposing the array and the summation range yields the same results
        random_points = [np.random.rand(2, 1) * (test_array.shape[ii] - 1)
                         for ii in range(2)]
        val8_1 = cython_helpers.wu_sum(test_array,
                                       random_points[0][0],
                                       random_points[0][1],
                                       random_points[1][0],
                                       random_points[1][1])
        val8_2 = cython_helpers.wu_sum(test_array,
                                       random_points[1][0],
                                       random_points[1][1],
                                       random_points[0][0],
                                       random_points[0][1])
        random_points_t = [a[-1::-1] for a in random_points]
        val8_3 = cython_helpers.wu_sum(test_array.transpose(),
                                       random_points_t[0][0],
                                       random_points_t[0][1],
                                       random_points_t[1][0],
                                       random_points_t[1][1])
        self.assertAlmostEqual(val8_1, val8_2, delta=tol)
        self.assertAlmostEqual(val8_1, val8_3, delta=tol)

if __name__ == "__main__":
    test_array = np.random.random(size=(2000, 2000))
    c_time = timeit.timeit("cython_helpers.wu_sum(test_array, 10, 6, 1910, 1005)",
                           number=1000, globals=globals())
    p_time = timeit.timeit("helpers.wu_sum(test_array, (10, 6), (1910, 1005))",
                           number=1000, globals=globals())
    print("Cython wu_sum: {}".format(c_time))
    print("Python wu_sum: {}".format(p_time))
    unittest.main()
