import sys
import os
sys.path.append('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/')
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping
from Image_Analysis import detect_star, image_analysis
from utils import parallel_process

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
        image, galaxy = output_list[0], output_list[-1]
        # detect_status = False
        # if self.get_asymmetry(image) > 0.25:
        detect_status = detect_star(galaxy, *self.get_params())
        return [image.split('/')[-1], self.get_asymmetry(image), detect_status]

def write_detect_output(detect_output, filename):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in detect_output:
        out_file.write('{},{},{}\n'.format(*dat))
    out_file.close()

# parameters = Parameters(data_file='a_test.csv')
# names = parameters.data[:, 0]
# for n in names:
#     print(parameters.get_asymmetry(n))