import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from Image_Analysis import image_analysis, write_asymetry_to_file, write_maxima_to_file, write_maxima_to_file_2, write_detections
# from Image_Analysis import detect_star
from scipy.optimize import minimize
from utils import parallel_process
from tqdm import trange
from star_detection_parameters import Parameters

# imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')
imgs = glob.glob('/shome/sahlr/summer_project_2017/Data/5*.fits')
# imgs = glob.glob('/disk1/ert/fits_images/*.fits')

def write_detect_output(detect_output, filename):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in detect_output:
        out_file.write('{},{},{}\n'.format(*dat))
    out_file.close()

step_size = 2000
nsteps = len(out)//step_size + 1
out = []

for k in trange(nsteps, desc="Blocks"):
    low_limit = k*step_size
    high_limit = (k+1)*step_size
    out += parallel_process(imgs, image_analysis, 11)

write_asymetry_to_file('Detections_2k_size_7/asymmetry_2k_s7.csv', out)

step_size = 2000
nsteps = len(out)//step_size + 1
res = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    parameter = Parameters(bin_size=52, n_bins_avg=8,
                        factor=1.72, data_file='Detections_2k_size_7/asymmetry_2k_s7.csv')
    for k in trange(nsteps, desc="Blocks"):
        low_limit = k*step_size
        high_limit = (k+1)*step_size
        res += parallel_process(out[low_limit:high_limit], parameter.star_detect,
                                n_jobs=11)

    write_detect_output(res, 'Detections_2k_size_7/{}_{}_{:.2f}.csv'.format(*parameter.get_params()))

# write_maxima_to_file_2('auto_test_maxima_alt2.txt', out)
# write_maxima_to_file('auto_test_maxima2.txt', out)
# write_asymetry_to_file('auto_test_asymetry2.txt', out)

# os.system('git add auto*.txt')
# os.system('git commit -m "Output auto upload"')
# os.system('git push')