import pyximport; pyximport.install()
import cython_helpers
import helpers
import numpy as np
import timeit
import rectangle_finder as rf


if __name__ == "__main__":
    test_array = (np.iinfo(np.uint8).max * np.random.random(size=(2000, 2000))).astype(np.uint8)
    c_time = timeit.timeit("cython_helpers.wu_sum_cython(test_array, 10, 6, 1910, 1005, width=3)",
                           number=100, globals=globals())
    p_time = timeit.timeit("helpers.wu_sum(test_array, (10, 6), (1910, 1005), width=3)",
                           number=100, globals=globals())
    print("Cython wu_sum: {}".format(c_time))
    print("Python wu_sum: {}".format(p_time))

    test_array = np.random.random(size=(500, 500))
    c_time = timeit.timeit("cython_helpers.dilate(test_array, 5)",
                           number=10, globals=globals())
    p_time = timeit.timeit("rf.dilate(test_array, 5)",
                           number=10, globals=globals())
    dummy = 'super_dummy'
    tst_c = cython_helpers.dilate(test_array, 5)
    tst_p = rf.dilate(test_array, 5)
    print("Cython dilate: {}".format(c_time))
    print("Python dilate: {}".format(p_time))
    print(np.max(np.max(np.abs(tst_c - tst_p))))
