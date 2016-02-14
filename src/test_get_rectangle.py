import cProfile
import rectangle_finder as rf
import numpy as np
import matplotlib.pyplot as plt
import helpers
import itertools as it
import matplotlib.pyplot as plt
import time
import argparse as ap



def main_fnct(filen):
    if filen == "":
        filen='../test_whiteboards/cellphone_samples/big_blackboard_blank.jpg'
    sample_image = rf.loadimage(filen)
    # cannied_sample = rf.auto_canny(rf.desaturate(sample_image), sigma=0.1)
    # plt.subplot(221)
    # plt.imshow(cannied_sample[500:1000, 500:1000], cmap=plt.cm.Greys_r)
    # cannied_sample = rf.auto_canny(rf.desaturate(sample_image), sigma=0.05)
    # plt.subplot(222)
    # plt.imshow(cannied_sample[500:1000, 500:1000], cmap=plt.cm.Greys_r)
    # cannied_sample = rf.auto_canny(rf.desaturate(sample_image), sigma=0.2)
    # plt.subplot(223)
    # plt.imshow(cannied_sample[500:1000, 500:1000], cmap=plt.cm.Greys_r)
    # plt.subplot(224)
    # plt.imshow(contour_image, cmap=plt.cm.Greys_r)
    # plt.show()
    t = time.time()
    test_rect = rf.get_rectangle_from_image(sample_image)
    print(time.time() - t)
    print(test_rect)
    plt.imshow(sample_image)
    plt.scatter(test_rect[:, 1], test_rect[:, 0], c='b', s=80)
    plt.show()

if __name__ == "__main__":
    argparser = ap.ArgumentParser()
    argparser.add_argument("-f, --file", dest='filen', metavar='FILE',
                           default="")
    args = argparser.parse_args()
    cProfile.run('main_fnct("{}")'.format(args.filen), 'profile')
