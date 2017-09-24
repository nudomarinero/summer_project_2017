'''
Analyses a galaxy
'''
# import pyximport
# pyximport.install()
import glob
import time
import copy
import os
import warnings
import traceback
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib import path
import numpy.ma as ma
import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage.morphology import dilation, erosion, square, ball, disk, diamond, star, watershed
from skimage.draw import polygon
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.interpolate import interp2d, rbf, Rbf
# from utils import parallel_process
# from star_detection_parameters import Parameters

plt.style.use(['black_fonts', 'presentation'])
TITLE_COLOR = 'black'

# TINY_SIZE = 12
# SMALL_SIZE = 16
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 24

# plt.rc('font', size=TINY_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# img_file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'
# out_file = open('neigbour_test.txt', 'w')

# Type 'sphinx-apidoc -f -o source/ ../' to create documentation.

def plot_image(image_data, cmin=0, cmax=None, cmap='hot', axis=None, text="", title="", presentation=False, output_name=None):
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
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image_data, clim=[cmin, cmax], cmap=cmap)
    if axis is not None:
        plt.axis(axis)
    # ax.set_xlabel('x pixel')
    # ax.set_ylabel('y pixel')
    plt.axis('off')
    ax.text(0.1, 0.05, text,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    # plt.title('test')
    # fig.savefig('Report/test.png', bbox_inches='tight')
    if presentation:
        title_obj = plt.title(title) #get the title property handler
        plt.setp(title_obj, color=TITLE_COLOR)
        fig.savefig('docs/_images/'+output_name+'.png', facecolor='none', bbox_inches='tight')

def smooth_image(image, do_sigma_clipping=True, threshold=None):
    """
    Decreases the noise and boosts bright regions by performing a 3x3 running
    average on the image. The original image is shown on the left, and the
    result of smoothing is shown on the right.

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/Before_smoothing.png
           :width: 45%
    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/After_smoothing.png
           :width: 45%

    Args:
        image (str): The name of the image.
        do_sigma_clipping (bool) [Optional]: If False, will not calculate a
            value of threshold from the image and will return the default/user
            threshold (None).
        threshold (float) [Optional]: If do_sigma_clipping=False, this parameter
            can be used to return an user defined threshold.
    
    Returns:
        * **moving_avg_img** (*numpy array*) - The smoothed image.
        * **threshold** (*float*) - The threshold value of the image which is
          equal to the mean + standard deviation of the image.
    """
    #pylint: disable=maybe-no-member
    image_data = fits.open(image)[0].data
    # plot_image(image_data, presentation=True, output_name='Before_smoothing')
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

    # plot_image(moving_avg_img, presentation=True, output_name='After_smoothing')
    if do_sigma_clipping:
        # image_data_test = image_data[image_data < 40]
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, iters=5)
        threshold = mean+1*std
        # threshold = -1.5*mean+2.5*median+1*std
        # print(threshold)

    return moving_avg_img, threshold

def galaxy_isolation(image, plot=False):
    """
    Isolates the galaxy using an 8 connectivity algorithm. The figure on the
    left is the smoothed image from :func:`smooth_image`, which is used to
    isolate the galaxy producing the image on the right.

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/After_smoothing.png
           :width: 45%
           :alt: Smoothed image of galaxy used to isolate galaxy.
    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/isolated_final_587739167310807244.png
           :width: 45%
           :alt: The isolated galaxy

    Args:
        image (str): The name of the galaxy.

    Returns:
        * *(numpy array)* - The isolated galaxy. 
        * *(str)* - The name of the galaxy.
    """
    image_name = image.split('/')[-1]
    pic, threshold_value = smooth_image(image)
    im = copy.deepcopy(pic)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)
    # labels = np.where(ma.filled(morphology.erosion(labels), 0) != 0, 1, 0)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(labels, cmap='hot', vmin=-20)

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('Report/labels_'+image_name.split('.')[0]+'.png', facecolor='none', bbox_inches='tight')

    size = labels.shape
    pic_plot = ma.masked_array(pic, labels != labels[int(size[1]/2), int(size[0]/2)])

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(ma.filled(pic_plot, 0), cmap='hot')

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('Report/isolated_initial_'+image_name.split('.')[0]+'.png', facecolor='none', bbox_inches='tight')

    pic_erode = np.where(ma.filled(morphology.erosion(pic_plot), 0) != 0, 1, 0)
    for i in range(3):
        pic_erode = np.where(ma.filled(morphology.erosion(pic_erode), 0) != 0, 1, 0)

    pic_plot = ma.masked_array(pic, pic_erode == 0)
    pic_plot = ma.filled(pic_plot, 0)

    im = copy.deepcopy(pic_plot)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)

    pic_plot = ma.masked_array(pic, labels != labels[int(size[1]/2), int(size[0]/2)])

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(ma.filled(pic_plot, 0), cmap='hot')

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('Report/isolated_final_'+image_name.split('.')[0]+'.png', facecolor='none', bbox_inches='tight')

    return ma.filled(pic_plot, 0), image_name

def find_local_maximum(data, plot=False):
    """
    Finds the location and the flux of all maxima in the image. Acheives this
    by using a combination of max and min filters and comparing to the original.
    An example of the maxima found in an image is shown below:

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/maxima_locations.png
           :width: 45%
           :align: center

    Args:
        data (numpy array): 2d array that stores the data of the image.

    Returns:
        list: A list containing lists that store the x, y coordinate of the maxima and the
        flux at that location.
    """
    neighborhood_size = 20
    threshold = np.average(data[data > 0])

    data_max = filters.maximum_filter(data, neighborhood_size)
    # plot_image(data_max, presentation=True, output_name='maximum_filter')
    maxima = (data == data_max)
    # plot_image(maxima, presentation=True, output_name='maxima')
    data_min = filters.minimum_filter(data, neighborhood_size)
    # plot_image(data_min, presentation=True, output_name='minimum_filter')
    diff = ((data_max - data_min) > threshold)
    # plot_image(diff, presentation=True, output_name='min_max_diff')
    maxima[diff == 0] = 0
    # plot_image(maxima, presentation=True, output_name='maxima_final')

    labeled, num_objects = ndimage.label(maxima)
    maxima_xy_loc = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)), dtype=np.int)
    maxima_data = [[i[1], i[0], data[i[0], i[1]]] for i in maxima_xy_loc]
    # print(maxima_data)
    if plot:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()
        plt.imshow(data, cmap='hot')
        plt.autoscale(False)
        plt.plot(maxima_xy_loc[:, 1], maxima_xy_loc[:, 0], 'b.')
        
        # ax.tick_params(axis='x', colors='white')
        # ax.xaxis.label.set_color('white')
        ax.set_xlabel('x pixel')
        # ax.tick_params(axis='y', colors='white')
        # ax.yaxis.label.set_color('white')
        ax.set_ylabel('y pixel')
        # fig.savefig('docs/_images/'+'maxima_locations'+'.png', facecolor='none', bbox_inches='tight')

    return maxima_data

def determine_asymmetry_180(image_data, plot=False):
    r"""
    Determines the asymmetry coeffection by rotating the image 180 degrees
    around the center of the image (pixel [128, 128]) and comparing it to the
    original image. The value of the asymmetry is found using

    .. math::
        A = \frac{\sum | I_{0} - I_{\theta} |}{2 \sum | I_{0} |}.

    Where :math:`I_{0}` is the flux of an individual pixel of the original
    image, and :math:`I_{\theta}` is the flux of the same pixel location as
    the original image after a :math:`180^{\circ}` rotation.

    Args:
        image_data (numpy array): 2d array containing the data of the image used
            to find the asymmetry values.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.

    Return:
        * The flux asymmetry under a 180 degree rotation.
        * The binary asymmetry under a 180 degree rotation.
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
    r"""
    Determines the asymmetry coeffection by rotating the image 90 degrees around
    the center of the image (pixel [128, 128]) and comparing it to the original
    image. The value of the asymmetry is found using

    .. math::
        A = \frac{\sum | I_{0} - I_{\theta} |}{2 \sum | I_{0} |}.

    Where :math:`I_{0}` is the flux of an individual pixel of the original
    image, and :math:`I_{\theta}` is the flux of the same pixel location as
    the original image after a :math:`90^{\circ}` rotation.

    Args:
        image_data (numpy array): 2d array containing the data of the image used
            to find the asymmetry values.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.

    Return:
        * The flux asymmetry under a 90 degree rotation.
        * The binary asymmetry under a 90 degree rotation.
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
        numpy array: 2d array of image_data with the x and y coordinate now at the
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



def minAsymmetry(image_data, maxima, plot=False, size=5):
    r"""
    Calculates the minimum value asymmetry of the image by choosing the center
    of rotation pixels neighbouring the center. The center is chosen as the 
    maximum associated with the galaxy or as (128, 128).  The value of the
    asymmetry is found using

    .. math::
        A = \frac{\sum | I_{0} - I_{\theta} |}{2 \sum | I_{0} |}.

    Args:
        image_data (numpy array): 2d array containing the data of the image.
        maxima: The maxima of the image found from :func:`find_local_maximum`.
        plot (bool): Optional parameter, will plot figures showing the flux and
            binary asymmetry when true.
        size (int): Optional parameter that determines how many pixels are used
            in the asymmetry calculation. A size of 5 will search a 10x10 box
            around the center.

    Returns:
        * The minimum flux asymmetry under a 180 degree rotation.
        * The minimum binary asymmetry under a 180 degree rotation.

    """
    # print(maxima)
    max_distance_from_center = np.inf
    for n, maximum in enumerate(maxima):
        x, y = maximum[0], maximum[1]
        if np.sqrt((x-128)**2+(y-128)**2) < max_distance_from_center:
            max_distance_from_center = np.sqrt((x-128)**2+(y-128)**2)
            maximum_idx = n
    # print(max_distance_from_center, maxima)
    if max_distance_from_center < 20:
        x_center, y_center = maxima[maximum_idx][0], maxima[maximum_idx][1]
    else:
        x_center, y_center = 128, 128
    # print(x_center, y_center, end=', ')
    min_asmmetry = np.inf
    for i in range(-size, size+1, 1):
        for j in range(-size, size+1, 1):
            new_image = shift_image(image_data, x_center+i, y_center+j)
            asymmetry = np.sum(np.abs(new_image-new_image[::-1, ::-1]))/(2*np.sum(new_image))
            if asymmetry < min_asmmetry:
                min_asmmetry, min_x, min_y = asymmetry, x_center+i, y_center+j
                min_new_image = new_image

    new_image_data_binary = np.where(min_new_image != 0, 1, 0)
    flipped_data_binary = np.where(min_new_image[::-1, ::-1] != 0, 1, 0)
    min_asymmetry_binary = np.sum(np.abs(new_image_data_binary-flipped_data_binary))/(2*np.sum(new_image_data_binary))

    if plot:
        # print(plot)
        plot_image(image_data, presentation=False, output_name='Flux_Image')
        plot_image(np.where(image_data == 0, 1, 0), presentation=False, output_name='Binary_Image')
        diff_binary = np.abs(new_image_data_binary-flipped_data_binary)
        diff = np.abs(new_image-new_image[::-1, ::-1])
        plot_image(diff_binary, cmap='Greys', cmax=1, text=np.round(min_asymmetry_binary, 3), presentation=False, output_name='Binary_asymmetry')
        plot_image(ma.masked_array(diff, diff == 0),
                   cmax=np.max(diff), cmap='Greys', text=np.round(min_asmmetry, 3), presentation=False, output_name='Flux_asymmetry')

    return min_asmmetry, min_asymmetry_binary

def detect_star(galaxy, binsize=50, no_of_previous_bins=8, threshold_factor=1.73, plot=False):
    """
    Detects whether or not there is a foreground star in the image. Does this by
    exploiting the fact that large stars tend to be saturated. Thus, plotting a
    histogram of the flux is used to detect a spike in flux which occurs due to
    the star. This is shown below.

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/detect_star_587739406795341869.png
           :width: 45%
    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/detect_star_587739406795341869_hist.png
           :width: 45%

    The green represents :data:`no_of_previous_bins`. The red bar represents
    that a star has been detected.

    Args:
        galaxy (numpy array): An array containing the data of the isolated galaxy from
            :func:`galaxy_isolation`.
        binsize (int): The number of bins used for the histogram of the flux.
            [optional]
        no_of_previous_bins (int): Determines how many bins are used for
            calculating the average flux. [optional]
        threshold_factor (float): Determines how many counts are needed for a
            spike in flux at a given bin to register as a star. [optional]
        plot (boolean): If true, will plot the image, and the histogram of the
            flux. [optional]
    
    Returns:
        True if star is detected. False otherwise.
    """
   
    galaxy_compressed = ma.masked_array(galaxy, galaxy == 0).compressed()
    detection = False
    # print(int(len(galaxy_compressed)/40))
    bins = np.min(np.array([int(len(galaxy_compressed)/70), binsize], dtype='int'))
    counts, edges = np.histogram(galaxy_compressed[galaxy_compressed > np.average(galaxy_compressed)],
                                bins)
    breakpoint = 0
    for c in range(len(counts)):
        if counts[c] > 0:
            if c >= no_of_previous_bins:
                counts_for_avg = counts[c-no_of_previous_bins:c]
                counts_for_avg = ma.masked_array(counts_for_avg, counts_for_avg < 1)
                average_local_counts = ma.average(counts_for_avg)
                # print(counts[c]-1.75*np.average(counts[c-10:c]))
                if average_local_counts > 4:
                    # print(counts[c], threshold_factor*average_local_counts)
                    if counts[c] >= np.floor(threshold_factor*average_local_counts):
                        # print(counts[c], edges[c])
                        detection = True
                        breakpoint = c
                        break
            else:
                counts_for_avg = counts[c-no_of_previous_bins:c]
                counts_for_avg = ma.masked_array(counts_for_avg, counts_for_avg < 1)
                average_local_counts = ma.average(counts_for_avg)
                # print(np.average(counts[0:10]), counts[c])
                if average_local_counts > 4:
                    if counts[c] > threshold_factor*average_local_counts:
                        # print(counts[c], edges[c])
                        detection = True
                        breakpoint = c
                        break
    # print(detection)
    # plot_image(galaxy)

    if plot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()

        ax.set_xlabel('Flux')
        ax.set_ylabel('No. of occurrences')

        # ax.spines['bottom'].set_color('white')
        # ax.spines['top'].set_color('white')
        # ax.spines['left'].set_color('white')
        # ax.spines['right'].set_color('white')

        edges = np.repeat(edges, 2)
        breakpoint *= 2
        hatch_from = edges[breakpoint-20]
        hatch_till = edges[breakpoint+2]
        hist = np.hstack((0, np.repeat(counts, 2), 0))
        fill_region = (hatch_from < edges)&(edges < hatch_till)
        # print(fill_region)

        ax.fill_between(edges[fill_region],
                        hist[fill_region], 0,
                        color='green', edgecolor='k',
                        hatch='///')

        hatch_from = edges[breakpoint-1]
        hatch_till = edges[breakpoint+4]
        # print(edges[breakpoint-1:breakpoint+4])
        fill_region = (hatch_from < edges)&(edges < hatch_till)
        # print(fill_region)

        ax.fill_between(edges[fill_region],
                        hist[fill_region], 0,
                        color='red', edgecolor='k',
                        hatch='xxx')
        # ax.plot(edges, hist, 'k')
        plt.hist(galaxy_compressed[galaxy_compressed > np.average(galaxy_compressed)],
                                    bins, color='b', zorder=-1,)

        # fig.savefig('docs/_images/detect_star_587739406795341869_hist.png', transparent=True, bbox_inches='tight')

    # plt.cla()
    return detection

def split_star_from_galaxy(galaxy, galaxy_name, plot=False):
    """
    Separates large stars from the galaxy only if :func:`detect_star` returns
    True. If it can locate the large star, the star is masked. If it can't, only
    the object associated with the center of the image is returned. The result
    is shown below.

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/split_star_contours.png
           :width: 45%
    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/split_star_galaxy.png
           :width: 45%

    Args:
        galaxy (numpy array): An array containing the flux data of the galaxy.
        galaxy_name (str): The name of the galaxy.
        plot (bool): Optional paramter, which when true will plot the galaxy
            separated from the star.

    Returns:
        * **galaxy** (*numpy array*) - The galaxy data.
    """

    img = np.zeros_like(galaxy)
    contours = measure.find_contours(galaxy, 1.*np.average(galaxy[galaxy > 0]))

    contour_avg = np.inf
    contour_idx = 0
    for c, contour in enumerate(contours):
        # print(len(contour))
        if len(contour) > 50:
            x, y = np.average(contour[:, 1]), np.average(contour[:, 0])
            if np.sqrt((x-128)**2+(y-128)**2) < contour_avg:
                contour_avg = np.sqrt((x-128)**2+(y-128)**2)
                contour_idx = c

    max_flux = np.max(galaxy)
    y, x = np.where(galaxy == max_flux)[0][0], np.where(galaxy == max_flux)[1][0]
    # print(max_flux, x, y)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(galaxy, cmap='hot')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # ax.plot(x, y, 'ob')
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        fig.savefig('Report/split_star_contours.png', facecolor='none', bbox_inches='tight')
    # plt.show()
    galaxy_label = 1
    label_count = 2
    max_c = []
    large_contours_labels = [] 
    for i, contour in enumerate(contours):
        if i != contour_idx:
            if len(contour) > 50:
                rr, cc = polygon(contour[:, 0], contour[:, 1], img.shape)
                img[rr, cc] = label_count+1
                # print(np.max(ma.masked_array(galaxy, img != i+1)))
                max_c.append(np.max(ma.masked_array(galaxy, img != label_count+1)))
                large_contours_labels.append(label_count+1)
                label_count += 1
        else:
            rr, cc = polygon(contour[:, 0], contour[:, 1], img.shape)
            img[rr, cc] = galaxy_label

    # plt.figure()
    # plt.imshow(img)

    max_c = np.array(max_c)
    mask_max = max_c < 200
    if np.sum(mask_max) != len(mask_max):
        for m, mask in enumerate(mask_max):
            # print(mask, m)
            if mask:
                galaxy_region = (img != 0) & (img == large_contours_labels[m])
                img[galaxy_region] = 1

            # print(np.where(img == large_contours_labels[m]))

    # plt.figure()
    # plt.imshow(img)
    # print(max_c)
    # print(large_contours_labels)

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='hot')
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('Report/split_star_contour_labels.png', facecolor='none', bbox_inches='tight')

    image_galaxy = np.where(galaxy != 0, 1, 0)
    labels = watershed(-img.astype(bool).astype(int), img, mask=image_galaxy)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(labels, cmap='hot')
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('Report/split_star_labels.png', facecolor='none', bbox_inches='tight')
    labels = measure.label(labels)
    split_galaxy = ma.masked_array(galaxy, labels != labels[128, 128]).filled(0)

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(split_galaxy, cmap='hot')
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)
        # fig.savefig('Report/split_star_galaxy.png', facecolor='none', bbox_inches='tight')

    return split_galaxy

def remove_small_star(galaxy, plot=False):
    """
    Separates small stars from the galaxy. Does this by locating the small stars
    using contours, and then masking it. The result is shown below.

    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/split_small_star_contours.png
           :width: 45%
    .. image:: //Users/Sahl/Desktop/University/Year_Summer_4/Summer_code/summer_project_2017/docs/_images/split_small_star_galaxy.png
           :width: 45%

    Args:
        galaxy (numpy array): An array containing the flux data of the galaxy.
        plot (bool): Optional paramter, which when true will plot the galaxy
            separated from the star.

    Returns:
        * **galaxy** (*numpy array*) - The galaxy data.
    """
    
    img = np.zeros_like(galaxy)
    contours = measure.find_contours(galaxy, 1*np.average(galaxy[galaxy > 0]))
    if len(contours) <= 1:
        # print(True)
        return galaxy
    contour_sizes = np.zeros(len(contours))
    contour_avg = np.inf
    contour_idx = 0
    for c, contour in enumerate(contours):
        # print(len(contour))
        contour_sizes[c] = len(contour)
        if len(contour) > 40:
            x, y = np.average(contour[:, 1]), np.average(contour[:, 0])
            if np.sqrt((x-128)**2+(y-128)**2) < contour_avg:
                contour_avg = np.sqrt((x-128)**2+(y-128)**2)
                contour_idx = c

    check_large_contours = contour_sizes > 105
    contain_small_star = False
    if (np.sum(check_large_contours) >= 2) :
        contain_small_star = False
    else:
        for c, contour in enumerate(contours):
            if c != contour_idx:
            # only check for contours not belonging to galaxy
                if 22 <= len(contour) <= 59:
                    contain_small_star = True
    
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(galaxy, cmap='hot')
        # ax.plot(contours[contour_idx][:, 1], contours[contour_idx][:, 0], linewidth=2)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)

        # fig.savefig('docs/_images/split_small_star_contours.png', facecolor='none', bbox_inches='tight')        

    # print(contain_small_star)
    if contain_small_star:
        galaxy_label = 1
        label_count = 2
        for i, contour in enumerate(contours):
            if i != contour_idx:
                if 22 <= len(contour) <= 59:
                    rr, cc = polygon(contour[:, 0], contour[:, 1], img.shape)
                    img[rr, cc] = label_count
                    label_count += 1
                if len(contour) > 59:
                    rr, cc = polygon(contour[:, 0], contour[:, 1], img.shape)
                    img[rr, cc] = galaxy_label
            if i == contour_idx:
                rr, cc = polygon(contour[:, 0], contour[:, 1], img.shape)
                img[rr, cc] = galaxy_label
        if plot:
            fig, ax = plt.subplots()
            plt.title('img')
            ax.imshow(img, cmap='hot')

        img_not_galaxy = ma.masked_array(img, img == galaxy_label).filled(0)
        # fig, ax = plt.subplots()
        # ax.imshow(img_not_galaxy, cmap='hot')
        regions = measure.regionprops(measure.label(img_not_galaxy), intensity_image=galaxy)
        for i in range(len(regions)):
            possible_star = regions[i]
            x0, y0 = possible_star.centroid

            # fimg = possible_star.intensity_image
            # fimg = ma.masked_array(fimg, fimg == 0)
            # print(ma.std(fimg), ma.mean(fimg))
            # plt.figure()
            # plt.imshow(fimg, cmap='hot')

            # print('eccentricity:', possible_star.eccentricity)
            # print('max intensity:', possible_star.max_intensity)
            if possible_star.eccentricity < 0.75:
                ## then it's circular
                image_galaxy = np.where(galaxy != 0, 1, 0)
                labels = watershed(-img.astype(bool).astype(int), img, mask=image_galaxy)
                # print(labels[int(np.ceil(x0)), int(np.ceil(y0))])
                if labels[int(x0), int(y0)] == 1:
                    galaxy = ma.masked_array(galaxy, labels == labels[int(np.ceil(x0)), int(np.ceil(y0))]).filled(0)
                else:
                    galaxy = ma.masked_array(galaxy, labels == labels[int(x0), int(y0)]).filled(0)
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(labels, cmap='hot')

            fig, ax = plt.subplots()
            # plt.plot(y0, x0, 'bo')
            ax.imshow(galaxy, cmap='hot')
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
            # fig.savefig('docs/_images/split_small_star_galaxy.png', facecolor='none', bbox_inches='tight')
        # print(galaxy)
        return galaxy

    else:
        return galaxy            

def image_analysis(image):
    """
    Analyses the image as follows. First, the galaxy is isolated using
    :func:`galaxy_isolation`. Then it removes small stars using
    :func:`remove_small_star`. The maxima are calculated using
    :func:`find_local_maximum`. Then it removes large stars using
    :func:`split_star_from_galaxy`. Finally the asymmetries are calculated using
    :func:`asymmetry_flux_180`, :func:`asymmetry_binary_180`,
    :func:`asymmetry_flux_90`,  :func:`asymmetry_binary_90`, and 
    :func:`minAsymmetry`.

    Args:
        image (str): The name of the image

    Returns:
        * **Name** (*str*) - The name of the galaxy.
        * **Maxima** (*list*) - The location and value of flux of each maxima.
        * **Various asymmetries**.

          * **flux_180** (*float*) - The asymmetry at the center of the
            image under 180 degree rotation.
          * **binary_180** (*float*) - The asymmetry at the center of
            the binary image under 180 degree rotation.
          * **flux_90** (*float*) - The asymmetry at the center of the
            image under 90 degree rotation.
          * **binary_90** (*float*) - The asymmetry at the center of
            the binary image under 90 degree rotation.
          * **min_flux_180** (*float*) - The minimum asymmetry of the
            image under 180 degree rotation.
          * **min_binary_180** (*float*) - The minimum asymmetry of the
            the binary image under 180 degree rotation.
        * **detect_status** (*bool*) - Whether or not there is a star in the image.
    Note:
        The list containing the maxima is stored as [[x1, y1, flux_1],
        [x2, y2, flux_2], ...] where x, y are ints and flux is a float.

    """
    try:
        galaxy, galaxy_name = galaxy_isolation(image, plot=False)
        # plot_image(galaxy, presentation=False, output_name='detect_star_'+galaxy_name.split('.')[0])
        # print(galaxy_name, end=' ')
        galaxy = remove_small_star(galaxy, plot=True)
        # plot_image(galaxy, presentation=False, output_name='Problem_'+galaxy_name.split('.')[0])
        maxima = find_local_maximum(galaxy, plot=False)
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
            # print(detect_status)
            if detect_status:
                galaxy_split = split_star_from_galaxy(galaxy, galaxy_name, plot=False)
                galaxy_split = remove_small_star(galaxy_split, plot=False)
                # plot_image(galaxy_split, presentation=False, output_name='detect_star_'+galaxy_name.split('.')[0])
                min_asmmetry_flux, min_asmmetry_binary = minAsymmetry(galaxy_split, maxima, plot=False)
                # print(galaxy_name, min_asmmetry_flux)

        return [galaxy_name, maxima, asymmetry_flux_180, asymmetry_binary_180,
                asymmetry_flux_90, asymmetry_binary_90, min_asmmetry_flux,
                min_asmmetry_binary, detect_status]
    except Exception as err:
        traceback.print_exc()
        print(err)
        print(image.split('/')[-1])
        return [image.split('/')[-1], np.array([np.nan, np.nan, np.nan]),
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

def write_asymetry_to_file(filename, data_to_write):
    out_file = open(filename, 'w')
    out_file.write('Galaxy_name,A_flux_180,A_binary_180,A_flux_90,A_binary_90,Min_A_flux_180,Min_A_binary_180\n')
    for dat in data_to_write:
        out_file.write('{0},{2},{3},{4},{5},{6},{7}\n'.format(*dat))

def write_maxima_to_file(filename, data_to_write):
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







if __name__ == "__main__":

    file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'
    imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')
    test_imgs = ['588013382727958540', '588013382727958540', '587739609175031857',
                 '587727213348520295', '587742061616758804', '587736619321655535',
                 '587739167310807244', '587733603734388952', '587739406805762069',
                 '587734621629513866', '587744728761761895', '587742566784106610',
                 '587730847428968484', '587742572149080091', '588017111293296655',
                 '588007006334943294']
    
    imgs_affect_by_size = ['587725469052371011.fits', '587732156314747278.fits', '587733196234096696.fits',
                           '587734621629513866.fits', '587735695912009805.fits', '587735743692996864.fits',
                           '587741421638451429.fits', '587741708879921260.fits', '587742012751282716.fits',
                           '587742062171521094.fits', '588015508212220022.fits', '588016890639941781.fits',
                           '588017725480108223.fits']

    small_star_tests = ['587739507159990339.fits', '587739609175031857.fits', '587744873715597513.fits',
                        '587734622710792236.fits', '587732483820093568.fits', '587726016692093083.fits',
                        '587739131348582565.fits', '587729748448182892.fits', '588017990703251464.fits',
                        '587742616175182087.fits', '587739406801961158.fits', '587742577534304450.fits',
                        '588017703484391495.fits', '587728918985375911.fits', '587733429234237590.fits',
                        '587732583130136762.fits']

    other_small_star_tests = ['587735743693848946.fits', '587738196113686666.fits', '587738409249734815.fits',
                              '587738570319986831.fits', '587739379920208077.fits', '587742009508102509.fits',
                              '587742060531744839.fits', '587726877798301756.fits', '587734861611204737.fits',
                              '587734893284884556.fits']

    # 587728906102309121 | star sandwich -> adapt to large star removal?
    # 587731887350284565 | To see changes showing new labeling (small stars given
    #       different label to large objects): change large contour -> 105 and eccentricity to 0.87 

    diff = ['587741532766142675.fits', '587744873715597513.fits', '588007004165701671.fits',
            '587729773680984272.fits', '587742577525457035.fits', '587736941449379858.fits',
            '587736542023122964.fits', '587729971254198276.fits', '587739827674022085.fits',
            '587739815315964115.fits', '587735743693848946.fits', '587727213348520295.fits',
            '587726879421956195.fits',]

    weird_267k = ['587734863758819334.fits', '587725470665277729.fits', '587738065663754413.fits', '588017702935527502.fits']

    # t1 = time.clock()
    # print(len(imgs))
    # out = image_analysis('/Users/Sahl/Desktop/5877/588010136268046361.fits')
    out = image_analysis('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/588017703484391495.fits')
    # print(time.clock()-t1)

    large_star_diff = ['587728666115506288.fits', '587741532766142675.fits', '587735660477743159.fits',
                       '587729386076176389.fits', '587742572149080091.fits', '588017111293296655.fits',
                       '588017702403899406.fits', '587742611067568140.fits', '587739811030761519.fits',
                       '587729228223217760.fits', '588017978910769274.fits', '587736753543905454.fits',
                       '587745540508745734.fits', '587730022799114306.fits', '587741816768889103.fits']

    # out = image_analysis(imgs[269])
    # image_analysis(file_dir+small_star_tests[11])

    # t1 = time.clock()
    # for i in np.random.randint(0,1998,size=10):
    #     # print(i)
    #     image_analysis(imgs[i])
    # print(time.clock()-t1)

    # for index, img in enumerate(imgs[260:272]):
    #     print(index)
    #     out = image_analysis(img)
    #     plt.show()

    # t1 = time.clock()
    # for index, img in enumerate(small_star_tests):
    #     out = image_analysis(file_dir+img)
    #     print()
    #     plt.show()
    # print(time.clock()-t1)
    #     print()
        # print('Image {} processed'.format(index+1))
    
    plt.show()
    
    # for t_img in imgs_affect_by_size:
    #     galaxy, galaxy_name = galaxy_isolation(file_dir+t_img)
    #     plot_image(galaxy)
    #     plt.title(t_img)
    #     plt.savefig('docs/_images/Figure_'+galaxy_name.split('.')[0]+'.png')
    #     plt.cla()
    # plt.show()
    # out = []

