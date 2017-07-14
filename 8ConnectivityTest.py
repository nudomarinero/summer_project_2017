'''
Created on 10 Jul 2017

@author: Sahl
'''

from Data_analysis import smooth_image, plot_image
from astropy.io import fits
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import copy
import os
from functools import reduce
import time
imgs = glob.glob('5*.fits')


def isolateGalaxy(image):

    pic = np.array([[1,1,0,1,1,1,0,1],
                   [1,1,0,1,0,1,0,1],
                   [1,1,1,1,0,0,0,1],
                   [0,0,0,0,0,0,0,1],
                   [1,1,1,1,0,1,0,1],
                   [0,0,0,1,0,1,0,1],
                   [1,1,1,1,0,0,0,1],
                   [1,1,1,1,0,1,1,1]])
    # plt.imshow(pic, cmap='hot')

    pic = smooth_image(image)
    # pic = fits.open(imgs[0])[0].data

    labels = np.zeros_like(pic)
    label_count = 0

    # print pic.shape

    # for i in range(-1,2,1):
    #     print 3+i

    for i in range(pic.shape[1]):
        for j in range(pic.shape[0]):
            if pic[i,j]<=35:
                pass
            else:
                if label_count==0:
                    labels[i,j]=1
                    label_count+=1
                else:
                    if i==0:
                        if labels[i,j-1]!=0:
                            labels[i,j]=labels[i,j-1]
                        else:
                            label_count+=1
                            labels[i,j]=label_count
                    else:
                        nearest_neighbours = []
                        for k in range(-1,2,1):
                            if j+k<0 or j+k>7:
                                pass
                            else:
                                nearest_neighbours.append(labels[i-1, j+k])
                        if j-1>=0:
                            nearest_neighbours.append(labels[i,j-1])
                        nearest_neighbours = np.array(nearest_neighbours)
                        nearest_neighbours = ma.masked_array(nearest_neighbours, nearest_neighbours==0, fill_value=99)

                        if nearest_neighbours.all() is np.ma.masked:
                            label_count+=1
                            labels[i,j] = label_count
                        else:
                            labels[i,j] = np.min(nearest_neighbours)
    #                     print nearest_neighbours, labels[i,j], (i,j)

    size = labels.shape[0]

    labels_copy = copy.deepcopy(labels)
    # x=6

    for x in range(2,label_count+1):

        masks = [labels==0, labels>x]
        total_mask = reduce(np.logical_or, masks)
        labels_copy = ma.masked_array(labels, mask=total_mask)

        # print np.min(labels_copy)

        current_label = x
        min_label_connected = current_label
        mins = []

        # print labels

        for i in range(labels.shape[1]):
            for j in range(labels.shape[0]):
                if labels[i,j]!=0:
                    nn = []

                    # current pixel
            #             print(i,j)

                    # vertical and horizontal neighbours
                    if i!=size-1:
            #             print i+1,j
                        nn.append(labels_copy[i+1,j])
                    if i>0:
            #             print i-1,j
                        nn.append(labels_copy[i-1,j])
                    if j!=size-1:
            #             print i,j+1
                        nn.append(labels_copy[i,j+1])
                    if j>0:
            #             print i,j-1
                        nn.append(labels_copy[i,j-1])
                    # diagonal neighbours
                    if i!=size-1:
                        if j!=size-1:
            #                 print i+1,j+1
                            nn.append(labels_copy[i+1,j+1])
                        if j>0:
            #                 print i+1,j-1
                            nn.append(labels_copy[i+1,j-1])
                    if j!=size-1:
                        if i>0:
            #                 print i-1,j+1
                            nn.append(labels_copy[i-1,j+1])
                    if i>0 and j>0:
            #             print i-1,j-1
                        nn.append(labels_copy[i-1,j-1])

            #             print 'labels of nearest neighbours: ', nn
                    nn = np.array(nn)
                    nn = ma.masked_invalid(nn)
            #             print nn.all is np.ma.masked
                    if current_label in nn:
                        if nn.all() is np.ma.masked:
                            pass
                        else:
                            mins.append(np.min(nn))

        mins = np.array(mins)
#         print(mins)
        if len(mins)==0:
            pass
    #         print current_label
        else:
    #         print np.min(np.array(mins))
            min_label_connected = np.min(np.array(mins))

        for i in range(labels.shape[1]):
            for j in range(labels.shape[0]):
                if labels[i,j]==current_label:
                    labels[i,j] = min_label_connected



#     print labels[128,128]

    mask = [labels!=labels[128,128]]
    labels = ma.masked_array(labels, mask=mask)

#     Data_analysis.plot_image(labels)

    pic = ma.masked_array(pic, mask=mask)
    pic = ma.filled(pic, 0)
    plot_image(pic)

#     plt.show()

#     print 'Plots'

    hdu = fits.PrimaryHDU(pic)
    hdulist = fits.HDUList([hdu])
    output_name = 'Isolated_'+image
    if len(glob.glob(output_name)) != 0:
        os.remove(output_name)
    hdulist.writeto('Isolated_'+image)

#     print 'Saved'

#     Data_analysis.plot_image(pic)
#     plt.figure()
#     plt.imshow(pic, cmap='hot')


def GalaxyIsolation(image):
    """
    For the given image, gives each pixel a label by starting with the top left pixel.
    Then goes to the nearest pixel in the next column and labels (if it is above a
    threshold) it by comparing it with it nearest labelled neighbours. Then repeats for
    each row.

    Once everything is labeled, it determines which labels are connected to each other.
    Starts at the 1st label (1) [which is a parent label]
    is and finds all the nearest neighbours of every pixel with the label (1). Any pixels
    with a label other than (1) is clearly connected to (1) [which are child labels].
    Then the child labels are all re-labeled (1). Then it moves onto the parent label.
    And so on until only parent labels remain.
    """
    pic = np.array([[1,1,0,1,1,1,0,1],
                   [1,1,0,1,0,1,0,1],
                   [1,1,1,1,0,0,0,1],
                   [0,0,0,0,0,0,0,1],
                   [1,1,1,1,0,1,0,1],
                   [0,0,0,1,0,1,0,1],
                   [1,1,1,1,0,0,0,1],
                   [1,1,1,1,0,1,1,1]])

    pic = smooth_image(image)

    labels = np.zeros_like(pic)
    label_count = 0
    threshold_value = 37

    for i in range(pic.shape[1]):
        for j in range(pic.shape[0]):
            # Threshold value.
            if pic[i,j] <= threshold_value:
                pass
            else:
                # Setting the 1st label.
                if label_count==0:
                    labels[i,j]=1
                    label_count+=1
                else:
                    # Condition to distinguish labeling the 1st row vs other rows.
                    # If: for the 1st row; else: for the other rows
                    if i==0:
                        if labels[i,j-1]!=0:
                            # If pixel to the left is labeled, gives current pixel the
                            # label of its neighbour.
                            labels[i,j]=labels[i,j-1]
                        else:
                            # Otherwise, a new label is given
                            label_count+=1
                            labels[i,j]=label_count
                    else:
                        # Finds nearest labeled neighbours. If none are labeled, current
                        # pixel given new label, otherwise it is given the minimum value of
                        # the other labels.
                        nearest_neighbours = []
                        for k in range(-1,2,1):
                            if j+k<0 or j+k>7:
                                pass
                            else:
                                nearest_neighbours.append(labels[i-1, j+k])
                        if j-1>=0:
                            nearest_neighbours.append(labels[i,j-1])
                        nearest_neighbours = np.array(nearest_neighbours)
                        nearest_neighbours = ma.masked_array(nearest_neighbours, nearest_neighbours==0, fill_value=99)

                        if nearest_neighbours.all() is np.ma.masked:
                            label_count+=1
                            labels[i,j] = label_count
                        else:
                            labels[i,j] = np.min(nearest_neighbours)

#     plot_image(labels, cmin=0, cmax=np.max(labels))

    size = labels.shape[0]
    skip = []

    labels_copy = copy.deepcopy(labels)
    for l in range(1,size):
        if l not in skip:
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
                # vertical and horizontal neighbours
                if i!=size-1:
                    nn.append(labels_copy[i+1,j])
#                     print(labels_copy[i+1,j])
                if i>0:
                    nn.append(labels_copy[i-1,j])
#                     print(labels_copy[i-1,j])
                if j!=size-1:
                    nn.append(labels_copy[i,j+1])
#                     print(labels_copy[i,j+1])
                if j>0:
                    nn.append(labels_copy[i,j-1])
#                     print(labels_copy[i,j-1])

                # diagonal neighbours
                if i!=size-1:
                    if j!=size-1:
                        nn.append(labels_copy[i+1,j+1])
#                         print(labels_copy[i+1,j+1])
                    if j>0:
                        nn.append(labels_copy[i+1,j-1])
#                         print(labels_copy[i+1,j-1])
                if j!=size-1:
                    if i>0:
                        nn.append(labels_copy[i-1,j+1])
#                         print(labels_copy[i-1,j+1])
                if i>0 and j>0:
                    nn.append(labels_copy[i-1,j-1])
#                     print(labels_copy[i-1,j-1])
            nn = np.array(nn)
            connected_labels = np.unique(nn)
            connected_labels = connected_labels[connected_labels!=0]

            # Loop to skip any child labels found.
            for c in connected_labels:
                if c not in skip:
                    skip.append(c)
                labels_copy[labels_copy == c] = l

#     plot_image(labels_copy, cmin=0, cmax=np.max(labels_copy))

    labels = labels_copy

    mask = [labels!=labels[128,128]]
    labels = ma.masked_array(labels, mask=mask)

    pic = ma.masked_array(pic, mask=mask)
    pic = ma.filled(pic, 0)
    plot_image(pic)

    hdu = fits.PrimaryHDU(pic)
    hdulist = fits.HDUList([hdu])
    output_name = 'Isolated_'+image
    if len(glob.glob(output_name)) != 0:
        os.remove(output_name)
    hdulist.writeto('Isolated_'+image)

#     plt.show()

# t1 = time.clock()
# GalaxyIsolation(imgs[6])
# print(time.clock()-t1)

t1 = time.clock()

for i,img in enumerate(imgs):
    GalaxyIsolation(img)
    print('Image '+str(i+1)+' processed.')

t2 = time.clock()
print(t2-t1)

# t1 = time.clock()
#
# for i,img in enumerate(imgs):
#     isolateGalaxy(img)
#     print('Image '+str(i+1)+' processed.')
#
# t2 = time.clock()
# print(t2-t1)

plt.show()
# isolateGalaxy(imgs[4])


# plt.show()