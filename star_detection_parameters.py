import sys
import os
sys.path.append('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/')

import numpy as np
import pandas as pd
from Image_Analysis import detect_star

class parameters():
    def __init__(self, bin_size=50, n_bins_avg=10, factor=1.75, data_file=None):
        if data_file is not None:
            self.data = np.array(pd.read_csv(data_file))
            # print('data file exists')
        else:
            self.data = None
            # print('no data_file')
        self.binsize = bin_size
        self.no_bins_avg = n_bins_avg
        self.threshold_factor = factor

    def get_params(self):
        return [self.binsize, self.no_bins_avg, self.threshold_factor]

    def get_asymmetry(self, image):
        return self.data[np.where(self.data[:, 0] == image.split('/')[-1])[0][0], 5]

    def star_detect(self, output_list):
        image, galaxy = output_list[0], output_list[-1]
        detect_status = False
        if self.get_asymmetry(image) > 0.4:
            detect_status = detect_star(galaxy, *self.get_params())
        return [image.split('/')[-1], self.get_asymmetry(image), detect_status]


# parameters = parameters(data_file='a_test.csv')
# names = parameters.data[:, 0]
# for n in names:
#     print(parameters.get_asymmetry(n))