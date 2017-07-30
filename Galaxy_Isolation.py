'''
Created on 15 Jul 2017

@author: Sahl
'''
from Image_Analysis import smooth_image, plot_image
from astropy.io import fits
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import os
from functools import reduce
import time
from skimage import morphology
from skimage import measure
from skimage import filters

class count():
    def __init__(self):
        self.count = 0

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/3.3.3/bin/ffmpeg'

imgs = glob.glob('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/5*.fits')
print(imgs[0].split('/')[8])

count = count()

fig = plt.figure()
ims = []
# print(len(imgs))
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

def label_neighbours(data, threshold, cmax=800, save_animations=False):

    labels = np.zeros_like(data)
    if save_animations:
        im = plt.imshow(labels, clim=[0, np.max(labels)], cmap='hot')
        plt.savefig('Output_images/test'+str(count.count)+'.png')
        count.count += 1
        ims.append([im])
        plt.cla()

    label_count = 0
    for row in range(0, data.shape[1]):
        for column in range(0, data.shape[0]):
            if get_pixel_value(data, row, column) < threshold:
                pass
            else:
                neighbours, indices = get_neighbours(data, row, column)

                if label_count == 0:
                    labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,1,0)
                    label_count+=1

                else:
                    if labels[row,column] == 0:
                        nearest_neighbours_labels = (labels[indices[0]:indices[1],indices[2]:indices[3]])
                        if np.max(nearest_neighbours_labels)==0:
                            label_count += 1
                            labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,label_count,0)
                            im = plt.imshow(labels, clim=[0, cmax], cmap='hot')
                            print(count.count)
                            plt.savefig('Output_images/test'+str(count.count)+'.png')
                            count.count+=1                            
                            plt.cla()
                            ims.append([im])
                        else:
                            min_label = np.min(nearest_neighbours_labels[nearest_neighbours_labels>0])
                            labels[indices[0]:indices[1],indices[2]:indices[3]] = np.where(neighbours>=threshold,min_label,0)
                            # if save_animations:
                            #     im = plt.imshow(labels, clim=[0, cmax], cmap='hot')
                            #     print(count.count)
                            #     plt.savefig('Output_images/test'+str(count.count)+'.png')
                            #     count.count+=1
                            #     ims.append([im]) 
                            #     plt.cla()

    # print(np.max(labels))
    # time.sleep(10)
    if save_animations:
        repeat_frame(labels)
    return labels

def connect_Image_labels(labels, cmax=20, save_animations=False):
    all_labels_connected = False
    count_l = 0
    while not all_labels_connected:
        labels_copy = copy.deepcopy(labels)
        l = labels[128,128]
#         print(l)
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
            nn.append(get_neighbours(labels_copy, i, j)[0].tolist())

        nn = np.array(nn)
        connected_labels = np.unique(nn)

        try:
            if len(connected_labels)==2 or count >= 10:
                all_labels_connected = True
                break
            connected_labels = connected_labels[connected_labels!=l]
            connected_labels = connected_labels[connected_labels!=0]

            for c in connected_labels:
                labels_copy[labels_copy == c] = l
        except:
            connected_labels = []
            for n in np.unique(nn):
                connected_labels.extend(np.unique(n))
            connected_labels = np.unique(np.array(connected_labels))

            if len(connected_labels)==2 or count_l >= 10:
                all_labels_connected = True

            connected_labels = connected_labels[connected_labels!=l]
            connected_labels = connected_labels[connected_labels!=0]
#             print(connected_labels)
            for c in connected_labels:
                labels_copy[labels_copy == c] = l

        labels = labels_copy
        if save_animations:
            repeat_frame(labels)
        count_l += 1
    # plot_image(labels)
    # print(len(ims))
    return labels_copy

def galaxy_isolation(data, output_name, threshold_value):
    labels = label_neighbours(data, threshold_value)
    labels = connect_Image_labels(labels)
    labels = np.where(ma.filled(morphology.erosion(labels),0)!=0,1,0)
    pic_plot = ma.masked_array(data, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])

    pic_erode = np.where(ma.filled(morphology.erosion(pic_plot),0)!=0,1,0)
    for i in range(2):
        pic_erode = np.where(ma.filled(morphology.erosion(pic_erode),0)!=0,1,0)

    pic_plot = ma.masked_array(pic_plot, pic_erode==0)
    pic_plot = ma.filled(pic_plot,0)

    labels = label_neighbours(pic_plot, threshold, cmax=10)
    labels = connect_Image_labels(labels)
    pic_plot[labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)]] = 0

    return pic_plot

def repeat_frame(image_data, no_of_repetitions=50):
    for i in range(no_of_repetitions):
        im = plt.imshow(image_data, cmap='hot')
        print(count.count)
        plt.savefig('Output_images/test'+str(count.count)+'.png')
        count.count += 1
        ims.append([im]) 
        plt.cla()


def labelling_animation(image):
    image_data, threshold = smooth_image(image)
    repeat_frame(image_data, 200)

    labels = label_neighbours(image_data, threshold, cmax=800, save_animations=True)
    labels = connect_Image_labels(labels, save_animations=True)
    labels = np.where(ma.filled(morphology.erosion(labels),0)!=0,1,0)
    repeat_frame(labels)

    label_plot = ma.masked_array(labels, labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)])
    labels_erode = np.where(ma.filled(morphology.erosion(label_plot),0)!=0,1,0)
    repeat_frame(labels_erode)

    for i in range(2):
        labels_erode = np.where(ma.filled(morphology.erosion(labels_erode),0)!=0,1,0)
        repeat_frame(labels_erode)

    label_plot = ma.masked_array(label_plot, labels_erode==0)
    label_plot = ma.filled(label_plot,0)
    repeat_frame(label_plot)

    labels = label_neighbours(label_plot, 1, cmax=20, save_animations=True)
    labels = connect_Image_labels(labels, save_animations=True)

    image_data[labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)]] = 0
    labels[labels!=labels[int(labels.shape[1]/2),int(labels.shape[0]/2)]] = 0

    repeat_frame(labels)
    repeat_frame(image_data)


    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
    repeat_delay=3000, repeat=True)

    # print(len(ims))
    # print('saving')
    # ani.save('labelling_image_2.mp4')
    # print('saved')

    plt.show()





# label_neighbours(pic, 1)
# plt.show()




k = 773

labelling_animation(imgs[k])
isolated_galaxies = glob.glob('test_Isolated_*.fits')



# print(imgs[k])

# plt.show()
# t1 = time.clock()
# pic = galaxy_isolation(image, 'testing_'+imgs[k].split('/')[8], threshold_value=threshold)
# print(time.clock()-t1)
# plot_image(pic, cmax = None)
# plt.show()


# k = 34, 547, 918, 1706 fail;  k = 812 infinite loop (not an image); k = 1210 fails, but should work
# infinite loops between 812 and 1210 not checked

# for ig in isolated_galaxies:
#     determine_Asymmetry(ig)

# plt.show()



# print(get_neighbours(pic, 7,7))
