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
from star_detection_parameters import Parameters, parallel_parameter_check

imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')
# imgs = glob.glob('/shome/sahlr/summer_project_2017/Data/5*.fits')
# imgs = glob.glob('/disk1/ert/fits_images/*.fits')

out = parallel_process(imgs, image_analysis, 4)
write_asymetry_to_file('Detections_best/asymmetry_2k.csv', out)

parallel_parameter_check(data=out, filename='Detections_best/asymmetry_2k.csv')
