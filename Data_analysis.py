'''
Created on 10 Jul 2017

@author: Sahl
'''

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import glob
#import skimage
from numpy.core import multiarray
#import cv2

def plot_image(image_data, cmin=0, cmax=100, cmap='hot'):
    plt.figure()
    plt.imshow(image_data, clim=[cmin,cmax], cmap=cmap)
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.tight_layout()

def smooth_image(image):

    image_data = fits.open(image)[0].data
    moving_avg_img = np.zeros_like(image_data)

    for y in range(image_data.shape[0]):
        for x in range(image_data.shape[1]):
            if y<=2 or x<=2 or y>=253 or x>=253:
                moving_avg_img[x,y]=image_data[x,y]
            else:
                count = 0
                for i in range(3):
                    for j in range(3):
                        moving_avg_img[x,y]+=image_data[x+i,y+j]
                        count+=1
                moving_avg_img[x,y] = moving_avg_img[x,y]/count


    print(np.mean(image_data))
    print(np.median(image_data))


    difference = image_data-moving_avg_img
    input_plus_difference = image_data+difference

#     plot_image(input_plus_difference)
#     plot_image(image_data)
    plot_image(moving_avg_img)

    return moving_avg_img
#     plt.show()

imgs = glob.glob('*.fits')

def light_profile(image):

    image_data = smooth_image(image)
    x_data, y_data = np.arange(0, image_data.shape[1]), np.arange(0, image_data.shape[0])

    x_cent, y_cent = 128, 128
    for r in range(10,150,5):
        print(r)

        data_x, data_y = np.meshgrid(x_data, y_data)
        mask = [(np.sqrt((data_x-x_cent)**2 + (data_y-y_cent)**2)>r)]

        plot_image(ma.MaskedArray(image_data, mask=mask))
        plt.show()


# edges = cv2.Canny(img,100,200)

# light_profile(imgs[2])

# smooth_image(imgs[2])
