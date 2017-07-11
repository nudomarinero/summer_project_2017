'''
Created on 10 Jul 2017

@author: Sahl
'''

import Data_analysis
from astropy.io import fits
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import copy
imgs = glob.glob('*.fits')

pic = np.array([[1,1,0,1,1,1,0,1],
               [1,1,0,1,0,1,0,1],
               [1,1,1,1,0,0,0,1],
               [0,0,0,0,0,0,0,1],
               [1,1,1,1,0,1,0,1],
               [0,0,0,1,0,1,0,1],
               [1,1,1,1,0,0,0,1],
               [1,1,1,1,0,1,1,1]])
# plt.imshow(pic, cmap='hot')

pic = Data_analysis.smooth_image(imgs[0])
# pic = fits.open(imgs[0])[0].data

labels = np.zeros_like(pic)
label_count = 0

# print pic.shape

# for i in range(-1,2,1):
#     print 3+i

for i in range(pic.shape[1]):
    for j in range(pic.shape[0]):
        if pic[i,j]<=30:
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
    print mins
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



print
print labels[128,128]

mask = [labels!=labels[128,128]]
labels = ma.masked_array(labels, mask=mask)

# print labels
plt.figure()
plt.imshow(labels, cmap='hot')
plt.figure()
plt.imshow(pic, cmap='hot')
plt.show()