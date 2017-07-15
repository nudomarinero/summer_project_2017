'''
Created on 10 Jul 2017

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
# import cv2

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([128,128]))
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)
    return new_image

def plot_image(image_data, cmin=0, cmax=100, cmap='hot',axis=None, text=""):
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
#     f.subplots_adjust(top=0.95, left=0., right=1.)

def smooth_image(image, do_sigma_clipping=True, threshold=None):

    image_data = fits.open(image)[0].data[:,:]
    moving_avg_img = np.zeros_like(image_data)

#     print(image_data.shape)

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
#         print(mean+std)
        threshold = mean+2*std

    return moving_avg_img, threshold
#     plt.show()

# def light_profile(image):
#
#     image_data = smooth_image(image)
#     x_data, y_data = np.arange(0, image_data.shape[1]), np.arange(0, image_data.shape[0])
#
#     x_cent, y_cent = 128, 128
#     for r in range(10,150,5):
#         print(r)
#
#         data_x, data_y = np.meshgrid(x_data, y_data)
#         mask = [(np.sqrt((data_x-x_cent)**2 + (data_y-y_cent)**2)>r)]
#
#         plot_image(ma.MaskedArray(image_data, mask=mask))
#         plt.show()

def find_radius(image_data):
    ymin = np.min(np.nonzero(image_data)[0][:])
    ymax = np.max(np.nonzero(image_data)[0][:])
    xmin = np.min(np.nonzero(image_data)[:][1])
    xmax = np.max(np.nonzero(image_data)[:][1])
    return np.max(np.abs(np.array([ymin, ymax, xmin, xmax])-128))
#     return np.max(np.abs(np.array([xmin,xmax])-128)), np.max(np.abs(np.array([ymin,ymax])-128))


# find_radius(isolated_galaxies[3])

def determine_Asymmetry(image):
    image_data = fits.open(image)[0].data
    flipped_data = image_data[::-1,::-1]

    diff = np.abs(image_data-flipped_data)
    asymmetry = np.round(np.sum(diff)/(2*np.sum(image_data)),2)
#     rad = find_radius(diff)+5
#     print(rad)

    image_data_binary = np.where(image_data!=0,1,0)
    flipped_data_binary = np.where(flipped_data!=0,1,0)

    diff_binary = np.abs(image_data_binary-flipped_data_binary)
#     diff_binary = ma.masked_array(diff_binary, diff_binary==0)
    plot_image(diff_binary, cmax=1, cmap='Greys', text = str(asymmetry))

    mask = diff==0
    plot_image(ma.masked_array(diff, mask=mask),cmax=np.max(diff), cmin=0, cmap='Greys',
                text = str(asymmetry))
#     plot_image(image_data)
    print(np.sum(diff)/(2*np.sum(image_data)), image.split('_')[1])


if __name__ == "__main__":

    imgs = glob.glob('5*.fits')
    print(imgs)
    isolated_galaxies = glob.glob('Isolated_*.fits')
    print(isolated_galaxies)

#     plot_image(smooth_image(imgs[0])[0])
    for ig in isolated_galaxies:
        determine_Asymmetry(ig)

#     for img in imgs:
#         plot_image(smooth_image(img)[0])
    plt.show()
# determine_Asymmetry(isolated_galaxies[4])


# plt.show()

