'''
Created on 15 Jul 2017

@author: Sahl
'''
from Data_analysis import smooth_image, plot_image, determine_Asymmetry
from astropy.io import fits
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import copy
import os
from functools import reduce
import time
import cv2
from skimage import morphology
from skimage import measure
from skimage import filters

imgs = glob.glob('5*.fits')
pic = np.array([[1,1,0,1,1,1,0,1],
                [1,1,0,1,0,1,0,1],
                [1,1,1,1,0,0,0,1],
                [0,0,0,0,0,0,0,1],
                [1,1,1,1,0,1,0,1],
                [0,0,0,1,0,1,0,1],
                [1,1,1,1,0,0,0,1],
                [1,1,1,1,0,1,1,1]])

def get_neighbours(data, row, column):

    max_row, max_column = data.shape[0], data.shape[1]
    if row < max_row and column < max_column:
        neighbours = []

        if row != 0 and column != 0 and row < data.shape[0] and column < data.shape[1]:
            neighbours = data[row-1:row+2, column-1:column+2]
            indices = [row-1,row+2, column-1,column+2]
        else:
            if row == 0 and column == 0:
                neighbours = data[row:row+2,column:column+2]
                indices = [row,row+2, column,column+2]
            elif row == 0 and column !=0:
                neighbours = data[row:row+2,column-1:column+2]
                indices = [row,row+2, column-1,column+2]
            elif row != 0 and column == 0:
                neighbours = data[row-1:row+2,column:column+2]
                indices = [row-1,row+2, column,column+2]

        return neighbours, indices
    else:
        return 'Row or column invalid'

def get_pixel_value(data, row, column):
    return data[row,column]

def label_neighbours(data, threshold):
    labels = np.zeros_like(data)
    label_count = 0

    for row in range(0,data.shape[1]):
        for column in range(0,data.shape[0]):
            if get_pixel_value(data, row, column) < threshold:
                pass
            else:
#                 print(row, column, get_neighbours(data, row, column)[0])
                neighbours, indices = get_neighbours(data, row, column)

                if label_count == 0:
                    labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,1,0)
                    label_count+=1

                else:
                    nearest_neighbours_labels = (labels[indices[0]:indices[1],indices[2]:indices[3]])
                    if np.max(nearest_neighbours_labels)==0:
                        label_count += 1
                        labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,label_count,0)

                    else:
                        min_label = np.min(nearest_neighbours_labels[nearest_neighbours_labels>0])
                        labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,min_label,0)
    return labels

def connect_Image_labels(labels):
#     all_labels_connected = False

    labels_copy = copy.deepcopy(labels)
    l = labels[128,128]

    masks = [labels==0, labels!=l]
    # masks all pixels that do not have the label value of 'l'. This is done so
    # that only nearest neighbours of pixels with the label 'l' are used in the
    # calculations.
    total_mask = reduce(np.logical_or, masks)
    label_mask = ma.masked_array(labels, mask=total_mask)
    label_mask = ma.filled(label_mask, 0)

    nonzero_locations = np.transpose(np.nonzero(label_mask))
    nn = []
    # Finds nearest neighbours
    for coord in nonzero_locations:
        i = coord[0]
        j = coord[1]
        for a in range(-1,2,1):
            for b in range(-1,2,1):
                if i<=0 or j<=0 or i>=254 or j>=254:
                    pass
                else:
                    nn.append(labels_copy[i+a,j+b])

    nn = np.array(nn)
    connected_labels = np.unique(nn)
    connected_labels = connected_labels[connected_labels!=l]
    connected_labels = connected_labels[connected_labels!=0]
#     print(connected_labels)
    # Loop to skip any child labels found.
    for c in connected_labels:
        labels_copy[labels_copy == c] = l

    return labels_copy

def galaxy_isolation(data, output_name):
    labels = label_neighbours(data, threshold_value)

#     plot_image(image, cmax = 100)
#     plot_image(labels, cmax=np.max(labels))
    labels = connect_Image_labels(labels)
    labels = connect_Image_labels(labels)
    labels = np.where(ma.filled(morphology.erosion(labels),0)!=0,1,0)
#     plot_image(labels, cmax=np.max(labels))
    pic_plot = ma.masked_array(data, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])

    test = np.where(ma.filled(morphology.erosion(pic_plot),0)!=0,1,0)
    for i in range(2):
        test = np.where(ma.filled(morphology.erosion(test),0)!=0,1,0)

    pic_plot = ma.masked_array(pic_plot, test==0)
    pic_plot = ma.filled(pic_plot,0)

    labels = label_neighbours(pic_plot, threshold_value)
    labels = connect_Image_labels(labels)
    pic_plot = ma.masked_array(pic_plot, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])
    nonzero_coords = np.nonzero(pic_plot)

#     pic_copy = copy.deepcopy(pic_plot)
#     center_flux = pic_plot[int(labels.shape[1]/2),int(labels.shape[0]/2)]

    ymin = np.min(nonzero_coords[0][:])
    ymax = np.max(nonzero_coords[0][:])
    xmin = np.min(nonzero_coords[:][1])
    xmax = np.max(nonzero_coords[:][1])

    pic_plot = pic_plot[ymin-5:ymax+5, xmin-5:xmax+5]
    labels = labels[ymin-5:ymax+5, xmin-5:xmax+5]

    hdu = fits.PrimaryHDU(ma.filled(pic_plot,0))
    hdulist = fits.HDUList([hdu])
    if len(glob.glob(output_name)) != 0:
        os.remove(output_name)
    hdulist.writeto(output_name)

    return pic_plot

t1 = time.clock()
isolated_galaxies = glob.glob('test_Isolated_*.fits')
for i,img in enumerate(imgs):
    image, threshold_value = smooth_image(img)

    output_name = 'test_Isolated_'+img
    pic = galaxy_isolation(image, output_name=output_name)
    print('Image '+str(i+1)+' processed.')
#     determine_Asymmetry(isolated_galaxies[i])
#     plot_image(pic)

print(time.clock()-t1)

plt.show()



# print(get_neighbours(pic, 7,7))
