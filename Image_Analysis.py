'''
Created on 21 Jul 2017

@author: Sahl
'''
import glob
import time
import copy
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from skimage import morphology
from skimage import measure
from skimage import filters
import pandas as pd
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from utils import parallel_process

# img_file_dir = '/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'
imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')

def plot_image(image_data, cmin=0, cmax=None, cmap='hot', axis=None, text=""):
    """
    Plots a 2d figure using matplotlib's imshow function.
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
        image_data_test = image_data[image_data < 40]
        mean, median, std = sigma_clipped_stats(image_data_test, sigma=3.0, iters=5)
        threshold = mean+1*std

    return moving_avg_img, threshold

def galaxy_isolation(image):

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
    Determines the asymmetry coeffection by rotating the image 180 degrees and comparing it to the 
    original image.
    """
    flipped_data = image_data[::-1, ::-1]
    try:
        diff = np.abs(image_data-flipped_data)
        asymmetry = np.round(np.sum(diff)/(2*np.sum(image_data)), 2)

        image_data_binary = np.where(image_data != 0, 1, 0)
        flipped_data_binary = np.where(flipped_data != 0, 1, 0)
        diff_binary = np.abs(image_data_binary-flipped_data_binary)
        asymmetry_binary = np.round(np.sum(diff_binary)/(2*np.sum(image_data_binary)), 2)
        # diff_binary = ma.masked_array(diff_binary, diff_binary == 0)
        mask = diff == 0
        if plot:
            plot_image(diff_binary, cmax=1, cmap='Greys', text=str(asymmetry_binary))
            plot_image(ma.masked_array(diff, mask=mask), cmax=np.max(diff), cmin=0, cmap='Greys',
                       text=str(asymmetry))
            plot_image(image_data)

        return asymmetry, asymmetry_binary
    except:
        return 'nan'

def determine_asymmetry_90(image_data, plot=False):
    rotate_data_90 = np.rot90(image_data)

    try:
        diff = np.abs(image_data-rotate_data_90)
        asymmetry = np.round(np.sum(diff)/(2*np.sum(image_data)), 2)

        image_data_binary = np.where(image_data != 0, 1, 0)
        rotate_data_90_binary = np.where(rotate_data_90 != 0, 1, 0)
        diff_binary = np.abs(image_data_binary-rotate_data_90_binary)
        asymmetry_binary = np.round(np.sum(diff_binary)/(2*np.sum(rotate_data_90_binary)), 2)
        # diff_binary = ma.masked_array(diff_binary, diff_binary == 0)
        mask = diff == 0
        if plot:
            plot_image(diff_binary, cmax=1, cmap='Greys', text = str(asymmetry_binary))
            plot_image(ma.masked_array(diff, mask=mask),cmax=np.max(diff), cmin=0, cmap='Greys',
                        text = str(asymmetry))
            plot_image(image_data)
            plot_image(rotate_data_90)

        return asymmetry, asymmetry_binary
    except:
        return 'nan'

def image_analysis(image):
    try:
        galaxy, galaxy_name = galaxy_isolation(image)
        maxima = find_local_maximum(galaxy)
        asymmetry_flux_180, asymmetry_binary_180 = determine_asymmetry_180(galaxy, plot=False)
        asymmetry_flux_90, asymmetry_binary_90 = determine_asymmetry_90(galaxy)
        # print(maxima, asymmetry_binary_180, asymmetry_flux_180, galaxy_name)
        return [galaxy_name, maxima, asymmetry_flux_180, asymmetry_binary_180, asymmetry_flux_90, asymmetry_binary_90]
    except:
        return[image.split('/')[-1], np.array([np.nan, np.nan, np.nan]), np.nan, np.nan, np.nan, np.nan]

def write_asymetry_to_file(filename, data_to_write):
    out_file = open(filename, 'w')
    out_file.write('# Galaxy_name | A_flux_180 | A_binary_180 | A_flux_90 | A_binary_90 \n')
    for dat in data_to_write:
        # print(dat[0], dat[2], dat[3])
        out_file.write(dat[0] + '|' + str(dat[2]) + '|' + str(dat[3]))
        out_file.write('|' + str(dat[4]) + '|' + str(dat[5]) + '\n')

def write_maxima_to_file(filename, data_to_write):
    out_file = open(filename, 'w')
    out_file.write('# Galaxy_name | x | y | flux \n')
    print(data_to_write)

    try:
        for dat_img in data_to_write:
            out_file.write(dat_img[0] + '|')
            if len(data_to_write[1]) == 1:
                for num, m in enumerate(dat_img[1]):
                    if num < len(dat_img[1])-1:
                        out_file.write(str(m) + '|')
                    else:
                        out_file.write(str(m))
            else:
                for dat in dat_img[1]:
                    print(dat)
                    for num, m in enumerate(dat):
                        if len(dat) != 1:
                            out_file.write('' + '|')
                        if num < len(dat)-1:
                            out_file.write(str(m) + '|')
                        else:
                            out_file.write(str(m))
                    out_file.write('\n')
    except:
        out_file.write('nan|nan|nan')


def write_maxima_to_file_2(filename, data_to_write):
    out_file = open(filename, 'w')
    out_file.write('# Galaxy_name | x | y | flux \n')
    print(data_to_write)

    try:
        for dat_img in data_to_write:
            if len(data_to_write[1]) == 1:
                out_file.write(dat_img[0] + '|')
                for num, m in enumerate(dat_img[1]):
                    if num < len(dat_img[1])-1:
                        out_file.write(str(m) + '|')
                    else:
                        out_file.write(str(m))
            else:
                for dat in dat_img[1]:
                    out_file.write(dat_img[0] + '|')
                    print(dat)
                    for num, m in enumerate(dat):
                        if num < len(dat)-1:
                            out_file.write(str(m) + '|')
                        else:
                            out_file.write(str(m))
                    out_file.write('\n')
    except:
        out_file.write('nan|nan|nan')
        



    # for dat in data_to_write:
    #     print(dat)
    #     for num, m in enumerate(data_to_write[1]):
    #         out_file.write(dat[0] + '|')
    #         print(m, 'here')
    #         try:
    #             out_file.write(str(m) + '|'+ str(m) + '|'+ str(m)+'\n')
    #         except:
    #             out_file.write(m[0] + '|'+ m[1] + '|'+ m[2]+'\n')


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

# a = image_analysis(0)
# print(a)
# plt.show()
# out = parallel_process(imgs[0:20], image_analysis)
out = parallel_process([imgs[138], imgs[773], 'test/5636.fits'], image_analysis)
write_maxima_to_file_2('maxima_alt.txt', out)
write_maxima_to_file('maxima.txt', out)
write_asymetry_to_file('asymetry.txt', out)

# read_maxima_from_file('test_maxima_file.txt')
# print(out)
# 138, 773 interesting cases. 1910?

# Draw circle around galaxies with only 1 maxima to determine how circular galaxies are? 
