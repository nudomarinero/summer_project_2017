import glob
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from Image_Analysis import image_analysis, write_asymetry_to_file, write_maxima_to_file, write_maxima_to_file_2, write_detections
# from Image_Analysis import detect_star
from utils import parallel_process
from star_detection_parameters import parameters

imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')

# imgs = glob.glob('/disk1/ert/fits_images/*.fits')

def write_detect_output(detect_output, filename):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in detect_output:
        out_file.write('{},{},{}\n'.format(*dat))
    out_file.close()

out = parallel_process(imgs, image_analysis, 3)
write_asymetry_to_file('a_test.csv', out)

# galaxies = out[-1]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for bin_sizes in range(50, 54):
        for n_bins in range(7, 10, 1):
            for thresh_factor in np.linspace(1.5, 1.75, 10):
                parameter = parameters(bin_size=bin_sizes, n_bins_avg=n_bins,
                                        factor=thresh_factor, data_file='a_test.csv')

                detect_output = parallel_process(out, parameter.star_detect, 3)
                write_detect_output(detect_output, 'Detections/{}_{}_{:.2f}_front.csv'.format(*parameter.get_params()))


# write_maxima_to_file_2('auto_test_maxima_alt2.txt', out)
# write_maxima_to_file('auto_test_maxima2.txt', out)
# write_asymetry_to_file('auto_test_asymetry2.txt', out)

# os.system('git add auto*.txt')
# os.system('git commit -m "Output auto upload"')
# os.system('git push')