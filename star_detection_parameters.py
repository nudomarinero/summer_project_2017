"""
Finds best parameters for :func:`~Image_Analysis.detect_star`. Does this by
comparing the detection outputs with known answers.
"""
import sys
import os
sys.path.append('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/')
import glob
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping
from Image_Analysis import detect_star, galaxy_isolation, find_local_maximum, remove_small_star
from Image_Analysis import minAsymmetry, determine_asymmetry_180, determine_asymmetry_90
from Image_Analysis import split_star_from_galaxy
from utils import parallel_process

class Parameters():
    """
    Creates an object that stores the parameters for
    :func:`~Image_Analysis.detect_star`.

    Args:
        binsize (int): The number of bins used for the histogram of the flux.
            [optional]
        no_of_previous_bins (int): Determines how many bins are used for
            calculating the average flux. [optional]
        threshold_factor (float): Determines how many counts are needed for a
            spike in flux at a given bin to register as a star. [optional]
        data_file (str): Data from of results from
            :func:`~Image_Analysis.image_analysis`.
        file_dir (str): Where to write new detections data to compare with
            correct detections.
        
    """
    def __init__(self, bin_size=50, n_bins_avg=10, factor=1.6, data_file=None, file_directory=''):
        if data_file is not None:
            self.data = np.array(pd.read_csv(data_file))
        else:
            self.data = None
        self.file_dir = file_directory
        self.binsize = bin_size
        self.no_bins_avg = n_bins_avg
        self.threshold_factor = factor
        self.galaxies = None

    def get_params(self):
        """
        Returns:
            :data:`bin_size`, :data:`n_bins_avg`, :data:`factor`.
        """
        return [self.binsize, self.no_bins_avg, self.threshold_factor]

    def get_asymmetry(self, image):
        """
        Gets the asymmetry using the :data:`data_file` provided.
        
        Args:
            image (str): The name of the galaxy.
        
        Returns:
            The asymmetry of the image (*float*).
        """
        return self.data[np.where(self.data[:, 0] == image.split('/')[-1])[0][0], 5]

    def star_detect(self, output_list):
        """
        Detects whether or not there is a star in the image for a particular set
        of parameters.

        Args:
            output_list (list): The output from :func:`~Image_Analysis.image_analysis`.
        
        Returns:
            list: * The galaxy name (*str*).
            * The asymmetry of the galaxy (*float*).
            * The detection of the star (*bool*).
        """
        image, maxima = output_list[0], output_list[1]
        galaxy = output_list[-1]
        # detect_status = False
        if len(maxima) == 1:
            detect_status = False
        else:
            # print(type(self.get_params()[0]), type(self.get_params()[1]), type(self.get_params()[2]))
            detect_status = detect_star(galaxy, *self.get_params())
        return [image.split('/')[-1], self.get_asymmetry(image), detect_status]

def write_detect_output(detect_output, filename):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in detect_output:
        out_file.write('{},{},{}\n'.format(*dat))
    out_file.close()

def image_analysis(image):
    try:
        galaxy, galaxy_name = galaxy_isolation(image, plot=False)
        # plot_image(galaxy, presentation=False, output_name='small_star'+galaxy_name.split('.')[0])
        galaxy = remove_small_star(galaxy, plot=False)
        # plot_image(galaxy, presentation=False, output_name='detect_star_'+galaxy_name.split('.')[0])
        maxima = find_local_maximum(galaxy, False)
        asymmetry_flux_180, asymmetry_binary_180 = determine_asymmetry_180(galaxy, plot=False)
        asymmetry_flux_90, asymmetry_binary_90 = determine_asymmetry_90(galaxy)
        min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy, maxima, plot=False)  
        detect_status = False
        # print(galaxy_name, min_asmmetry_flux, min_asmmetry_binary)
        # print(detect_status)
        # split_star_from_galaxy(galaxy, galaxy_name, plot=True)

        if len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy, plot=False)
            if detect_status:
                galaxy_split = split_star_from_galaxy(galaxy, galaxy_name, plot=False)
                galaxy_split = remove_small_star(galaxy_split, plot=False)
                # plot_image(galaxy_split, presentation=False, output_name='detect_star_'+galaxy_name.split('.')[0])
                min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy_split, maxima, plot=False)
                # print(galaxy_name, min_asmmetry_flux)

        return [galaxy_name, maxima, asymmetry_flux_180, asymmetry_binary_180,
                asymmetry_flux_90, asymmetry_binary_90, min_asmmetry_flux,
                min_asmmetry_binary, detect_status, galaxy]
    except Exception as err:
        traceback.print_exc()
        print(err)
        print(image.split('/')[-1])
        return [image.split('/')[-1], np.array([np.nan, np.nan, np.nan]),
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

def parallel_parameter_check(data, filename, file_dir):
    """
    Performs the detection analysis for numerous different :data:`bin_size`,
    :data:`n_bins_avg`, :data:`factor`; in order to help find the best parameters
    for :func:`~Image_Analysis.detect_star`.

    Args:
        data (list): List of outputs from :func:`~Image_Analysis.image_analysis`.
        filename (str): Filename of the asymmetries written by :func:`~Image_Analysis.write_asymetry_to_file`
        file_dir (str): Directory of where the images are stored.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for bin_sizes in range(50, 55):
            for n_bins in range(7, 11, 1):
                for thresh_factor in np.linspace(1.5, 1.85, 15):
                    parameter = Parameters(bin_size=bin_sizes, n_bins_avg=n_bins,
                                        factor=thresh_factor, data_file=filename, 
                                        file_directory=file_dir)

                    detect_output = parallel_process(data, parameter.star_detect, 4)
                    # detect_output = []
                    # for out_list in out:
                    #     detect_output.append(parameter.star_detect(out_list))
                    write_detect_output(detect_output,
                                        'Parameter_find/{}_{}_{:.2f}.csv'.format(*parameter.get_params()))


# file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'
# parameters = Parameters(data_file='Detections_best/asymmetry_2k.csv', file_directory=file_dir)
# names = parameters.data[3:10, 0]

# out = []
# for n in names:
#     # out.append(image_analysis(file_dir+n))
#     parameters.star_detect(image_analysis(file_dir+n))

# parallel_parameter_check(out, filename='Detections_best/asymmetry_2k.csv', file_dir=file_dir)
#     print(parameters.get_asymmetry(n))