'''
Created on 21 Jul 2017

@author: Sahl
'''
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import glob
import cv2
import scipy.misc
import matplotlib.mlab as mlab
from astropy.stats import sigma_clipped_stats
from skimage import morphology
import pandas as pd
from skimage import measure
import os
import time
import copy
from skimage import filters
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')

def plot_image(image_data, cmin=0, cmax=None, cmap='hot',axis=None, text=""):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image_data, clim=[cmin,cmax], cmap=cmap)
    if axis is not None:
        plt.axis(axis)
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    ax.text(0.1, 0.05,text,
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)

def smooth_image(image, do_sigma_clipping=True, threshold=None):

    image_data = fits.open(image)[0].data[:,:]
    moving_avg_img = np.zeros_like(image_data)

    for y in range(image_data.shape[1]-1):
        for x in range(image_data.shape[0]-1):
            try:
                count = 0
                for i in range(-1,2,1):
                    for j in range(-1,2,1):
                        moving_avg_img[x,y]+=image_data[x+i,y+j]
                        count+=1
                moving_avg_img[x,y] = moving_avg_img[x,y]/count
            except:
                moving_avg_img[x,y]=image_data[x,y]

    if(do_sigma_clipping):
        image_data_test = image_data[image_data<40]
        mean, median, std = sigma_clipped_stats(image_data_test, sigma=3.0, iters=5)
        threshold = mean+1*std

    return moving_avg_img, threshold

def GalaxyIsolation(image):

    pic, threshold_value = smooth_image(image)
    im = copy.deepcopy(pic)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)
    labels = np.where(ma.filled(morphology.erosion(labels),0)!=0,1,0)

    pic_plot = ma.masked_array(pic, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])

    pic_erode = np.where(ma.filled(morphology.erosion(pic_plot),0)!=0,1,0)
    for i in range(2):
        pic_erode = np.where(ma.filled(morphology.erosion(pic_erode),0)!=0,1,0)

    pic_plot = ma.masked_array(pic, pic_erode==0)
    pic_plot = ma.filled(pic_plot,0)

    im = copy.deepcopy(pic_plot)
    blobs = im > threshold_value
    labels = measure.label(blobs, neighbors=8)

    pic_plot = ma.masked_array(pic, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])

#     plot_image(pic_plot)
#     plt.show()

    image_name = image.split('/')[8]

    return ma.filled(pic_plot,0), image_name

def find_local_maximum(data):
    neighborhood_size = 20 #threshold based on number of pixels? np.bincount?
#     print(np.count_nonzero(data))
    threshold = np.average(data[data>0])

    # Perhaps relabel images with multiple images with a much higher threshold to separate the more distant
    # maxima, whilst keeping close local maxima like multiple nuclei
    blobs = data > 0.8*threshold
    labels = measure.label(blobs, neighbors=8)
    plot_image(labels,cmax=None)

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    plt.figure()
    plt.imshow(data)
    plt.autoscale(False)
    plt.plot(xy[:, 1], xy[:, 0], 'r.')

    return num_objects

def determine_Asymmetry_180(image_data):
    flipped_data = image_data[::-1,::-1]

    try:
        diff = np.abs(image_data-flipped_data)
        asymmetry = np.round(np.sum(diff)/(2*np.sum(image_data)),2)

        image_data_binary = np.where(image_data!=0,1,0)
        flipped_data_binary = np.where(flipped_data!=0,1,0)
        diff_binary = np.abs(image_data_binary-flipped_data_binary)
        asymmetry_binary = np.round(np.sum(diff_binary)/(2*np.sum(image_data_binary)),2)
    #     diff_binary = ma.masked_array(diff_binary, diff_binary==0)
        plot_image(diff_binary, cmax=1, cmap='Greys', text = str(asymmetry_binary))

        mask = diff==0
        plot_image(ma.masked_array(diff, mask=mask),cmax=np.max(diff), cmin=0, cmap='Greys',
                    text = str(asymmetry))
        plot_image(image_data)
        return asymmetry, asymmetry_binary
    except:
        print('nan')

def determine_Asymmetry_90(image_data):
    rotate_data_90 = np.rot90(image_data)

    try:
        diff = np.abs(image_data-rotate_data_90)
        asymmetry = np.round(np.sum(diff)/(2*np.sum(image_data)),2)

        image_data_binary = np.where(image_data!=0,1,0)
        rotate_data_90_binary = np.where(rotate_data_90!=0,1,0)
        diff_binary = np.abs(image_data_binary-rotate_data_90_binary)
        asymmetry_binary = np.round(np.sum(diff_binary)/(2*np.sum(rotate_data_90_binary)),2)

#         diff_binary = ma.masked_array(diff_binary, diff_binary==0)
#         plot_image(diff_binary, cmax=1, cmap='Greys', text = str(asymmetry_binary))

#         mask = diff==0
#         plot_image(ma.masked_array(diff, mask=mask),cmax=np.max(diff), cmin=0, cmap='Greys',
#                     text = str(asymmetry))
#         plot_image(image_data)
#         plot_image(rotate_data_90)

        return asymmetry, asymmetry_binary
    except:
        print('nan')

def imageAnalysis(image):
    galaxy, galaxy_name = GalaxyIsolation(image)
    no_of_maxima = find_local_maximum(galaxy)
    asymmetry_flux, asymmetry_binary  = determine_Asymmetry_180(galaxy)
    print(no_of_maxima, asymmetry_binary, asymmetry_flux, galaxy_name)

    return np.array([asymmetry_flux, asymmetry_binary, no_of_maxima, galaxy_name])

    plt.show()

imageAnalysis(imgs[138])
# 138, 773 interesting cases

