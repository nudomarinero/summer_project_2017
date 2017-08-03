'''
Created on 21 Jul 2017

@author: Sahl
'''
import glob
import time
import copy
import os
import warnings
import traceback
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from utils import parallel_process

# img_file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'
# out_file = open('neigbour_test.txt', 'w')

# Type 'sphinx-apidoc -f -o source/ ../' to create documentation.

def plot_image(image_data, cmin=0, cmax=None, cmap='hot', axis=None, text=""):
    """
    Plots a 2d figure using matplotlib's imshow function.

    Args:
        image_data (numpy array): 2d array containing the data to plot using
            imshow.
        cmin (float): Low value of cmin for imshow. [Optional, default=0]
        cmax (float): High value of cmax for imshow. [Optional, default=None]
        cmap (str): Type of colour map for imshow. [Optional, default='hot']
        axis (list): List of the form, [xmin, xmax, ymin, ymax].
            [Optional, default=None]
        text (str): Text to write at bottom left corner of image. [Optional]
    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image_data, clim=[cmin, cmax], cmap=cmap)
    if axis is not None:
        plt.axis(axis)
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    ax.text(0.1, 0.05, text,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

def smooth_image(image, do_sigma_clipping=True, threshold=None):
    """
    Decreases the noise and boosts bright regions by performing a 3x3 running
    average on the image.

    Args:
        image (str): The name of the image.
        do_sigma_clipping (bool) [Optional]: If False, will not calculate a
            value of threshold from the image and will return the default/user
            threshold (None).
        threshold (float) [Optional]: If do_sigma_clipping=False, this parameter
            can be used to return an user defined threshold.
    """
    #pylint: disable=maybe-no-member
    image_data = fits.open(image)[0].data
    moving_avg_img = np.zeros_like(image_data)

    for y in range(image_data.shape[1]-1):
        for x in range(image_data.shape[0]-1):
            try:
                count = 0
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        moving_avg_img[x, y] += image_data[x+i, y+j]
                        count += 1
                moving_avg_img[x, y] = moving_avg_img[x, y]/count
            except:
                moving_avg_img[x, y] = image_data[x, y]

    if do_sigma_clipping:
        # image_data_test = image_data[image_data < 40]
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, iters=5)
        threshold = mean+1*std
        # print(threshold)

    return moving_avg_img, threshold

def galaxy_isolation(image):
    """
    Isolates the galaxy from the foreground and background objects in the image.

    Args:
        image (str): The name of the galaxy.

    Returns:
        Output: An image with only the galaxy and the name of the galaxy.
    """

    pic, threshold_value = smooth_image(image)
    im = copy.deepcopy(pic)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)
    labels = np.where(ma.filled(morphology.erosion(labels), 0) != 0, 1, 0)

    size = labels.shape
    pic_plot = ma.masked_array(pic, labels != labels[int(size[1]/2), int(size[0]/2)])

    pic_erode = np.where(ma.filled(morphology.erosion(pic_plot), 0) != 0, 1, 0)
    for i in range(2):
        pic_erode = np.where(ma.filled(morphology.erosion(pic_erode), 0) != 0, 1, 0)

    pic_plot = ma.masked_array(pic, pic_erode == 0)
    pic_plot = ma.filled(pic_plot, 0)

    im = copy.deepcopy(pic_plot)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)

    pic_plot = ma.masked_array(pic, labels != labels[int(size[1]/2), int(size[0]/2)])
    image_name = image.split('/')[-1]

    return ma.filled(pic_plot, 0), image_name

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

    # blobs = data > 0.8*threshold
    # labels = measure.label(blobs, neighbors=8)
    # plot_image(labels, cmax=None)

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

def determine_asymmetry_180(image_data, plot=False):
    """
    Determines the asymmetry coeffection by rotating the image 180 degrees
    around the center of the image (pixel [128, 128]) and comparing it to the
    original image.

    Args:
        image_data (numpy array): 2d array containing the data of the image used
            to find the asymmetry values.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.

    Return:
        Output: The values of the flux and binary asymmetry.
    """
    flipped_data = image_data[::-1, ::-1]
    try:
        diff = np.abs(image_data-flipped_data)
        asymmetry = np.sum(diff)/(2*np.sum(image_data))

        image_data_binary = np.where(image_data != 0, 1, 0)
        flipped_data_binary = np.where(flipped_data != 0, 1, 0)
        diff_binary = np.abs(image_data_binary-flipped_data_binary)
        asymmetry_binary = np.sum(diff_binary)/(2*np.sum(image_data_binary))
        # diff_binary = ma.masked_array(diff_binary, diff_binary == 0)
        mask = diff == 0
        if plot:
            plot_image(diff_binary, cmax=1, cmap='Greys',
                       text=np.round(asymmetry_binary, 3))
            plot_image(ma.masked_array(diff, mask=mask), cmax=np.max(diff),
                       cmin=0, cmap='Greys', text=np.round(asymmetry, 3))
            plot_image(image_data)

        return asymmetry, asymmetry_binary
    except:
        return 'nan'

def determine_asymmetry_90(image_data, plot=False):
    """
    Determines the asymmetry coeffection by rotating the image 90 degrees around
    the center of the image (pixel [128, 128]) and comparing it to the original
    image.

    Args:
        image_data (numpy array): 2d array containing the data of the image used
            to find the asymmetry values.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.

    Return:
        Output: The values of the flux and binary asymmetry.
    """
    rotate_data_90 = np.rot90(image_data)

    try:
        diff = np.abs(image_data-rotate_data_90)
        asymmetry = np.sum(diff)/(2*np.sum(image_data))

        image_data_binary = np.where(image_data != 0, 1, 0)
        rotate_data_90_binary = np.where(rotate_data_90 != 0, 1, 0)
        diff_binary = np.abs(image_data_binary-rotate_data_90_binary)
        asymmetry_binary = np.sum(diff_binary)/(2*np.sum(rotate_data_90_binary))
        # diff_binary = ma.masked_array(diff_binary, diff_binary == 0)
        mask = diff == 0
        if plot:
            plot_image(diff_binary, cmax=1, cmap='Greys',
                       text=str(asymmetry_binary))
            plot_image(ma.masked_array(diff, mask=mask), cmax=np.max(diff),
                       cmin=0, cmap='Greys', text=str(asymmetry))
            plot_image(image_data)
            plot_image(rotate_data_90)

        return asymmetry, asymmetry_binary
    except:
        return 'nan'

def shift_image(image_data, x, y):
    """
    Re-centers the image onto the given coordinates, x and y.

    Args:
        image_data (numpy array): 2d array containing the data that will be
            re-centered.
        x (int): The x coordinate in image_data to center on.
        y (int): The y coordinate in image_data to center on.

    Returns:
        Output: 2d array of image_data with the x and y coordinate now at the
        center.
    """
    size = image_data.shape
    pad = 150
    half_pad = int(pad/2)
    new_image = np.zeros([size[1]+pad, size[0]+pad])
    old_center, new_center = np.array([128, 128]), np.array([x, y])
    shift = old_center - new_center
    new_image[half_pad+shift[1]:-half_pad+shift[1], half_pad+shift[0]:-half_pad+shift[0]] = image_data

    return new_image

def minAsymmetry(image_data, plot=False, size=3):
    """
    Calculates the minimum value asymmetry of the image by choosing the center
    of rotation pixels neighbouring the center (128, 128).

    Args:
        image_data (numpy array): 2d array containing the data of the image.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.
        size (int): Optional parameter that determines how many pixels are used
            in the asymmetry calculation. A size of 3 will use the pixels with
            value 128±3 in all directions.

    Returns:
        Output: The minimum flux and binary asymmetry

    """
    min_asmmetry = 2
    for i in range(-size, size+1, 1):
        for j in range(-size, size+1, 1):
            new_image = shift_image(image_data, 128+i, 128+j)
            asymmetry = np.sum(np.abs(new_image-new_image[::-1, ::-1]))/(2*np.sum(new_image))
            if asymmetry < min_asmmetry:
                min_asmmetry, min_x, min_y = copy.deepcopy(asymmetry), 128+i, 128+j
                min_new_image = copy.deepcopy(new_image)
            # plt.plot(128+i, 128+j, 'b.')
    new_image_data_binary = np.where(min_new_image != 0, 1, 0)
    flipped_data_binary = np.where(min_new_image[::-1, ::-1] != 0, 1, 0)
    min_asymmetry_binary = np.sum(np.abs(new_image_data_binary-flipped_data_binary))/(2*np.sum(new_image_data_binary))
    # print('{}, {}, {}, {}'.format(min_asmmetry, min_asymmetry_binary, min_x-128, min_y-128))
    # out_file.write('{}, {}, {}, {}\n'.format(min_asmmetry, asymmetry_binary, min_x-128, min_y-128))

    if plot:
        # print(plot)
        diff_binary = np.abs(new_image_data_binary-flipped_data_binary)
        plot_image(diff_binary, cmap='Greys', cmax=1, text=np.round(min_asymmetry_binary, 3))
        diff = np.abs(new_image-new_image[::-1, ::-1])
        plot_image(ma.masked_array(diff, diff == 0),
                   cmax=np.max(diff), cmap='Greys', text=np.round(min_asmmetry, 3))

    return min_asmmetry, min_asymmetry_binary

def detect_star(galaxy, binsize=50, no_of_previous_bins=10, threshold_factor=1.75):
    galaxy_compressed = ma.masked_array(galaxy, galaxy == 0).compressed()
    detection = False
    # print(int(len(galaxy_compressed)/40))
    bins = np.min(np.array([int(len(galaxy_compressed)/50), binsize], dtype='int'))
    print(bins)
    # plt.figure()
    counts, __ = np.histogram(galaxy_compressed[galaxy_compressed > np.average(galaxy_compressed)],
                                  bins)
    for c in range(len(counts)-3):
        if counts[c] > 0:
            if c >= no_of_previous_bins:
                average_local_counts = np.average(counts[c-no_of_previous_bins:c])
                # print(counts[c]-1.75*np.average(counts[c-10:c]))
                if average_local_counts > 4:
                    if counts[c] > threshold_factor*average_local_counts:
                        # print(1.75*np.average(counts[c-10:c]), counts[c])
                        # print(True)
                        detection = True
                        break
            else:
                average_local_counts = np.average(counts[c-no_of_previous_bins:c])
                # print(np.average(counts[0:10]), counts[c])
                if average_local_counts > 4:
                    if counts[c] > threshold_factor*average_local_counts:
                        # print(1.75*np.average(counts[0:c]), counts[c])
                        # print('Diffraction Spikes detected.')
                        detection = True
                        break
    # print(detection)
    # plt.figure()
    # plt.hist(galaxy_compressed[galaxy_compressed > np.average(galaxy_compressed)],
    #                               bins)
    # plt.cla()
    return detection

def image_analysis(image, bin_size=50, n_bins_avg=10, factor=1.75):
    """
    Analysis of the image to give the number of maxima in the galaxy and it's
    asymmetry values.

    Args:
        image (str): The name of the image

    Returns:
        Output: An array containing the name of the image, the maximas of the
        image (stored in an array), the flux and binary asymmetry under a 180
        and 90 degree rotation at the center of the image, and the minimum value
        of the flux and binary asymmetry under a 180 degree rotation.

    Note:
        The array containing the maxima is stored as [[x1, y1, flux_1],
        [x2, y2, flux_2], ...] where x, y are ints and flux is a float.

    """
    try:
        galaxy, galaxy_name = galaxy_isolation(image)
        # plot_image(galaxy)
        maxima = find_local_maximum(galaxy)
        asymmetry_flux_180, asymmetry_binary_180 = determine_asymmetry_180(galaxy, plot=False)
        asymmetry_flux_90, asymmetry_binary_90 = determine_asymmetry_90(galaxy)
        min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy, plot=False)  
        # print(galaxy_name, end=' ')
        # print(maxima)
        return [galaxy_name, maxima, asymmetry_flux_180, asymmetry_binary_180,
                asymmetry_flux_90, asymmetry_binary_90, min_asmmetry_flux,
                min_asmmetry_binary, galaxy]
    except Exception as err:
        # traceback.print_exc()
        print(err)
        return [image.split('/')[-1], np.array([np.nan, np.nan, np.nan]),
                np.nan, np.nan, np.nan, np.nan, np.nan]

def write_asymetry_to_file(filename, data_to_write):
    """
    Writes the various asymmetry values calculated from image_analysis to a
    file.

    Args:
        filename (str): The name of the file.
        data_to_write (list): The data written to the file. The data is the
            output from image_analysis.
    """
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,A_flux_180,A_binary_180,A_flux_90,A_binary_90,Min_A_flux_180,Min_A_binary_180\n')
    for dat in data_to_write:
        out_file.write('{0},{2},{3},{4},{5},{6},{7}\n'.format(*dat))

def write_maxima_to_file(filename, data_to_write):
    """
    Writes the location and flux of each maxima in the image.

    Args:
        filename (str): The name of the file.
        data_to_write (list): The data written to the file. The data is the
            output from image_analysis.
    """
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,x,y,flux\n')
    for dat_img in data_to_write:
        try:
            out_file.write('{}'.format(dat_img[0]))
            if len(data_to_write[1]) == 1:
                for num, m in enumerate(dat_img[1]):
                    if num < len(dat_img[1])-1:
                        out_file.write('{}'.format(m))
                    else:
                        out_file.write(str(m))
            else:
                for dat in dat_img[1]:
                    for num, m in enumerate(dat):
                        if len(dat) != 1:
                            out_file.write('' + ',')
                        if num < len(dat)-1:
                            out_file.write('{}'.format(m))
                        else:
                            out_file.write(str(m))
                    out_file.write('\n')
        except:
            out_file.write(',')
            out_file.write('{},{},{}\n'.format(*dat_img[1]))

def write_maxima_to_file_2(filename, data_to_write):
    """
    Writes the location and flux of each maxima in the image.

    Args:
        filename (str): The name of the file.
        data_to_write (list): The data written to the file. The data is the
            output from image_analysis.
    """
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,x,y,flux\n')
    for dat_img in data_to_write:
        try:
            if len(data_to_write[1]) == 1:
                out_file.write(dat_img[0] + ',')
                for num, m in enumerate(dat_img[1]):
                    if num < len(dat_img[1])-1:
                        out_file.write(str(m) + ',')
                    else:
                        out_file.write(str(m))
            else:
                for dat in dat_img[1]:
                    out_file.write(dat_img[0] + ',')
                    for num, m in enumerate(dat):
                        if num < len(dat)-1:
                            out_file.write(str(m) + ',')
                        else:
                            out_file.write(str(m))
                    out_file.write('\n')
        except:
            out_file.write('{},{},{}\n'.format(*dat_img[1]))

def write_detections(filename, data_to_write):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,Min_A_flux_180,detection\n')
    for dat in data_to_write:
        out_file.write('{0},{6},{8}\n'.format(*dat))

def read_maxima_from_file(filename):
    with open(filename, encoding="utf-8") as file:
        my_list = file.readlines()

    maxima = []
    galaxy_names = []
    no_of_maxima = []
    galaxy_count = 0
    for num, x in enumerate(my_list):
        if not x.startswith('#'):
            x_split = x.strip().split('|')
            if len(x_split[0]) > 0:
                galaxy_names.append(x_split[0])
                maxima.append([x_split[1], x_split[2], x_split[3]])
                no_of_maxima.append(1)
                galaxy_count += 1
            else:
                no_of_maxima[galaxy_count-1] += 1
                maxima.append([x_split[1], x_split[2], x_split[3]])

    maxima_img = []
    count = 0
    for n in range(len(galaxy_names)):
        maxima_img.append([])
        if no_of_maxima[n] == 1:
            maxima_img[n] = maxima[count]
            maxima_img[n] = np.array(maxima_img[n])
            count += 1
        else:
            for n_max in range(no_of_maxima[n]):
                maxima_img[n].append(maxima[count])
                count += 1
                if n_max == no_of_maxima[n]-1:
                    maxima_img[n] = np.array(maxima_img[n])

    # for n in range(len(galaxy_names)):
    #     print(galaxy_names[n], maxima_img[n])

    # for img_no in range(10,20):
    #     # print(maxima_img[img_no].shape)
    #     try:
    #         x, y = maxima_img[img_no][:,0], maxima_img[img_no][:,1]
    #     except:
    #         x, y = maxima_img[img_no][0], maxima_img[img_no][1]
    #     data, __ = smooth_image(img_file_dir+galaxy_names[img_no], do_sigma_clipping=False)
    #     plt.figure()
    #     plt.imshow(data, cmap='hot')
    #     # plt.autoscale(False)
    #     plt.plot(x, y, 'b.')
    #     plt.show()

    return galaxy_names, maxima_img





if __name__ == "__main__":

    # 587742014909775877.fits wtf???

    # 587742061619839210.fits, 587742611346948260.fits ;
    #    high asymmetry with no diffraction spike and no spike in histogram

    # 587742775637311549.fits
    #   Diffraction spike with spike in histogram

    # 587727213348520295.fits : False positive (Low A value, A<0.25) too many empty bins (Now works)
    # 587733080273322124.fits : False positive (Low A value, A<0.25)
    # 587734621629513866.fits : False positive (only 1 maxima)
    # 587735696443310211.fits : False positive (Low A value, A<0.25)
    # 587739406805762069.fits : False positive (too many bins?) (Now works)
    # 587739720835399813.fits : False positive (Low A value, A<0.25, too many bins?)
    # 587744728761761895.fits : False positive (too many bins) (Now works)
    # 588007003649998869.fits : False positive (A<0.25)
    # 588017991239794937.fits : False positive (A<0.25)
    # 588298664655061021.fits : False positive (A=0.337)
    # 587742629070045469.fits : False positive (A=0.37)
    # 587739167310807244.fits : False positive (A = 0.74), too many bins? (Now works)
    # 587736619321655535.fits : False positive (A = 0.51, too many bins) (now works)
    # 587742572149080091.fits : False negative (A = 0.71) (Now works)
    # 587742061616758804.fits : False negative (A = 0.42, too many bins?) (Now works)
    # 587730847428968484.fits : False negative (A = 0.53) (Now works!)
    # 587733603734388952.fits : False negative (A = 0.62)
    # 587742903938908310.fits : Unknown
    # 588017977277480991.fits : Unknown
    # 587736542026858577.fits : ? Possibly with star, but no diffraction spikes in image (now identifies star)
    # 587736920509645063.fits : Similar to above (now identifies star)
    # 588016840705704048.fits : Above
    # 588017702403899406.fits : Above
    # 587738196659077271.fits : Identified as having a star, not sure.

    # 587737808501211272.fits : Weird image.
    # 587739609175031857.fits : Correctly identified, but good for testing (still works)
    # 587739811030761519.fits : Correctly identified, but good for testing
    # 587741532766142675.fits : Correctly identified, but good for testing
    # 587742566784106610.fits : Correctly identified, but good for testing
    # 587739720846934450.fits : Correctly identified, but good for testing
    # 587745243629617322.fits : Correctly identified, but good for testing
    # 588007006334943294.fits : Correctly identified, but good for testing


    imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')
    out =image_analysis('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/587736941449379858.fits')
    # image_analysis(imgs[773])

    min_asmmetry_flux, maxima, galaxy_name, galaxy = out[5], out[1], out[0], out[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if min_asmmetry_flux < 0.25 or len(maxima) == 1:
            detect_status = False
        else:
            detect_status = detect_star(galaxy, 52, 8, 1.62)  
    # print('Star in {}: {}'.format(galaxy_name, detect_status))    
    plt.show()
    # out = []

    # os.system('git add Detections/*.txt')
    # os.system('git commit -m "different detection parameters"')
    # os.system('git push')
    # plt.show()
    # print(a)
    # plt.show()
    # out = parallel_process(imgs[0:20], image_analysis)
    # out = parallel_process([imgs[138],  'test/5636.fits', imgs[773], imgs[241], imgs[345]], image_analysis)
    # write_maxima_to_file_2('maxima_alt.txt', out)
    # write_maxima_to_file('maxima.txt', out)
    # write_asymetry_to_file('asymetry.txt', out)

    # read_maxima_from_file('test_maxima_file.txt')
    # print(out)
    # 138, 773 interesting cases. 1910?

    # Draw circle around galaxies with only 1 maxima to determine how circular galaxies are?
