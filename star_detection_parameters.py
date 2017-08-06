import sys
import os
sys.path.append('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/')
import glob
import copy
import warnings
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from skimage import morphology
from skimage import measure
from skimage import filters
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from utils import parallel_process
from Image_Analysis import detect_star

class Parameters():
    def __init__(self, bin_size=50, n_bins_avg=10, factor=1.6, data_file=None):
        if data_file is not None:
            self.data = np.array(pd.read_csv(data_file))
            # print('data file exists')
        else:
            self.data = None
            # print('no data_file')
        self.binsize = bin_size
        self.no_bins_avg = n_bins_avg
        self.threshold_factor = factor
        self.galaxies = None

    def get_params(self):
        return [self.binsize, self.no_bins_avg, self.threshold_factor]

    def get_asymmetry(self, image):
        return self.data[np.where(self.data[:, 0] == image.split('/')[-1])[0][0], 5]

    def star_detect(self, output_list):
        image, galaxy, maxima = output_list[0], output_list[-1], output_list[1]
        # detect_status = False
        """
        try adding a 2nd detection test for images with exactly 2 maximas. Erode the 
        image a couple more times and relabel to see if it there are small local maxima
        """
        if len(maxima) == 1:
            detect_status = False
        if len(maxima) == 2:
            im = copy.deepcopy(galaxy)
            blobs = im > 0
            labels = measure.label(blobs, neighbors=8)
            labels = np.where(ma.filled(morphology.erosion(labels), 0) != 0, 1, 0)

            size = labels.shape
            pic_plot = ma.masked_array(galaxy, labels != labels[int(size[1]/2), int(size[0]/2)])

            pic_erode = np.where(ma.filled(morphology.erosion(pic_plot), 0) != 0, 1, 0)
            for i in range(2):
                pic_erode = np.where(ma.filled(morphology.erosion(pic_erode), 0) != 0, 1, 0)

            pic_plot = ma.masked_array(galaxy, pic_erode == 0)
            pic_plot = ma.filled(pic_plot, 0)

            im = copy.deepcopy(pic_plot)
            blobs = im > 0
            labels = measure.label(blobs, neighbors=8)

            pic_plot = ma.masked_array(galaxy, labels != labels[int(size[1]/2), int(size[0]/2)])
            # plt.figure()
            # plt.imshow(pic_plot)
            
            new_maximas = find_local_maximum(pic_plot)
            if len(new_maximas) == 1:
                detect_status = False
            else:
                detect_status = detect_star(galaxy, *self.get_params())
        else:
            detect_status = detect_star(galaxy, *self.get_params())
        return [image.split('/')[-1], self.get_asymmetry(), detect_status]

def find_local_maximum(data):
    """
    Finds the location and the flux of all maxima in the image.

    Args:
        data (numpy array): 2d array that stores the data of the image.

    Returns:
        Output: An array containing arrays that store the x, y coordinate of the maxima and the
        flux at that location.
    """
    neighborhood_size = 20
    threshold = np.average(data[data > 0])

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    maxima_xy_loc = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)), dtype=np.int)
    maxima_data = [[i[1], i[0], data[i[0], i[1]]] for i in maxima_xy_loc]
    # print(maxima_data)

    # plt.figure()
    # plt.imshow(data)
    # plt.autoscale(False)
    # plt.plot(maxima_xy_loc[:, 1], maxima_xy_loc[:, 0], 'r.')

    return maxima_data

def write_detect_output(detect_output, filename):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in detect_output:
        out_file.write('{},{},{}\n'.format(*dat))
    out_file.close()

def parallel_parameter_check(data, filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for bin_sizes in range(51, 53):
            for n_bins in range(7, 9, 1):
                for thresh_factor in np.linspace(1.5, 1.75, 10):
                    parameter = Parameters(bin_size=bin_sizes, n_bins_avg=n_bins,
                                        factor=thresh_factor, data_file=filename)

                    detect_output = parallel_process(data, parameter.star_detect, 4)
                    # detect_output = []
                    # for out_list in out:
                    #     detect_output.append(parameter.star_detect(out_list))
                    write_detect_output(detect_output,
                                        'Detections_double_erosion/{}_{}_{:.2f}.csv'.format(*parameter.get_params()))


# parameters = Parameters(data_file='a_test.csv')
# names = parameters.data[:, 0]
# for n in names:
#     print(parameters.get_asymmetry(n))