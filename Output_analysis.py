import glob
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
from Image_Analysis import plot_image

def asymetry_analysis():
    data = pd.read_csv('complete_asymetry2.csv',na_values="nan")
    data = np.array(data)
    print(data.shape)
    
    names = 0
    a180_flux_old = 1
    a180_binary_old = 2
    a90_flux = 3
    a90_binary = 4
    a180_flux_min = 5
    a180_binary_min = 6

    # print(1)

    # key_data = np.array(pd.read_csv('data/key_sel2000.csv'))
    # key_names = key_data[:,1]
    # key_pmg = key_data[:,2]

    # key_pmg_plot = []
    # a180_flux_plot = []
    # a180_flux_plot_old = []

    # for ind,n in enumerate(names):
    #     try:
    #         index = np.where(n == key_names)[0][0]
    #         key_pmg_plot.append(key_pmg[index])
    #         a180_flux_plot.append(a180_flux[ind])
    #         a180_flux_plot_old.append(a180_flux_old[ind])
    #     except:
    #         pass

    # key_pmg_plot = np.array(key_pmg_plot)
    # a180_flux_plot = np.array(a180_flux_plot)
    # a180_flux_plot_old = np.array(a180_flux_plot_old)

#     print(names[a180_flux_plot>0.97])
#     for a,k in zip(a180_flux_plot, key_pmg_plot):
#         print(a,k)

#     print((key_pmg_plot[key_pmg_plot == 0.]))
    # plt.figure()
    # plt.plot(key_pmg_plot, a180_flux_plot, '.', markersize=1)
    # plt.xlabel('P_MG')
    # plt.ylabel('min_a180')

    # plt.figure()
    # plt.plot(key_pmg_plot, a180_flux_plot_old, '.', markersize=1)
    # plt.xlabel('P_MG')
    # plt.ylabel('center_a180')


    # plt.figure()
    # plt.plot(a180_flux_plot, a180_flux_plot_old, '.', markersize=1)
    # plt.xlabel('min_a180')
    # plt.ylabel('center_a180')

    plt.figure()
    plt.plot(data[:, a180_flux_min], data[:, a90_flux], '.', markersize=1)
    plt.xlabel('a180_flux_min')
    plt.ylabel('a90_flux')

    plt.figure()
    plt.plot(data[:, a180_flux_min], data[:, a180_binary_min], '.', markersize=1)
    plt.xlabel('a180_flux_min')
    plt.ylabel('a180_binary_min')

    plt.figure()
    plt.plot(data[:, a180_flux_old], data[:, a180_binary_old], '.', markersize=1)
    plt.xlabel('a180_flux_center')
    plt.ylabel('a180_binary_center')

    plt.figure()
    plt.plot(data[:, a180_flux_old], data[:, a180_binary_min], '.', markersize=1)
    plt.xlabel('a180_flux_center')
    plt.ylabel('a180_binary_min')

    plt.figure()
    x = np.array(data[:, a180_flux_min], dtype='float64')
    y = np.array(data[:, a180_binary_min], dtype='float64')
    xedges, yedges = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    hist, xedges, yedges = np.histogram2d(x, y, bins=1000)
    # hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    # xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
    # yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
    # c = hist[xidx, yidx]
    # plt.scatter(x, y, c=c, marker='.', s=1)

    im = plt.imshow(ma.masked_array(hist, hist == 0), interpolation='nearest', origin='low',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.xlabel('a180_flux_min')
    plt.ylabel('a180_binary_min')

    plt.figure()
    x = np.array(data[:, a180_flux_old], dtype='float64')
    y = np.array(data[:, a180_binary_old], dtype='float64')
    xedges, yedges = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    hist, xedges, yedges = np.histogram2d(x, y, bins=1000)

    im = plt.imshow(ma.masked_array(hist, hist == 0), interpolation='nearest', origin='low',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.xlabel('a180_flux_center')
    plt.ylabel('a180_binary_center')

    plt.show()

def star_analysis(filename, check_results=False):
    # data_4 = np.array(pd.read_csv('detection_test_4.csv'))
    # data_3 = np.array(pd.read_csv('detection_test_3.csv'))
    # data_2 = np.array(pd.read_csv('detection_test_2.csv'))
    # # data_1 = np.array(pd.read_csv('detection_test.csv'))
    file_dir = ('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/')

    data_actual_LOW = np.array(pd.read_csv('Detections_actual_asym_low.csv'))
    data_actual_HIGH = np.array(pd.read_csv('Detections_actual_asym_high.csv'))
    data_parameter_CHECK = np.array(pd.read_csv(filename))

    # print(data_actual_LOW[data_actual_LOW[:, 0]=='587725590381789417.fits'])
    # print(data_parameter_CHECK[data_parameter_CHECK[:, 0]=='587725590381789417.fits'])
    # print('\n')
    # print(data_parameter_CHECK.shape, len(data_actual_LOW), len(data_actual_HIGH))
    diff = np.zeros(len(data_parameter_CHECK[:,2]))
    count_LOW = 0
    count_HIGH = 0
    count_total = 0
    count_0pt3 = 1
    for i in range(len(data_parameter_CHECK)):
        if data_parameter_CHECK[i, 1] > 0.4:
            diff[i] = np.abs(data_actual_HIGH[count_HIGH, 2]-data_parameter_CHECK[i, 2])
            if check_results:
                if diff[i] == 1:
                    print(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2], 'HIGH',
                          data_actual_HIGH[count_HIGH, 2])
            count_HIGH += 1
        else:
            diff[i] = np.abs(data_actual_LOW[count_LOW, 2]-data_parameter_CHECK[i, 2])
            if check_results:
                if diff[i] == 1:
                    print(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2], 'LOW',
                        data_actual_LOW[count_LOW, 2])
            count_LOW += 1

        # if 0.2 < data_parameter_CHECK[i, 1] < 0.25:
        #     print(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2],
        #           data_actual_LOW[count_LOW-1, 2], data_actual_LOW[count_LOW-1, 0])
        #     # print(file_dir+data_parameter_CHECK[i, 0])
        #     image_data = fits.open(file_dir+data_parameter_CHECK[i, 0])[0].data
        #     plot_image(image_data)
        #     plt.title('{}: {}'.format(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2]))
            # print(data_parameter_CHECK[i, 0], count_0pt3)
            # count_0pt3 += 1
    # plt.show()
    # print('\n')
    # print(diff)
    print(filename.split('/')[1], np.sum(diff))
            # print(data_1[i,0], data_1[i, 1], data_1[i, 2],
            #       data_2[i, 2], data_3[i, 2], data_4[i, 2])
            # plot_image(fits.open('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/'+data_1[i, 0])[0].data)
    # plt.show()
    # data = np.genfromtxt('detection_test.txt', delimiter=',')
    # print(data[3:,])
    # diff = data_2[:,]
# asymetry_analysis()

"""
All images between 0.25 and 0.3 checked. All but 1 is detected correctly. Another has star, but is sucessfully isolated
All images between 0.3 and 0.4 checked. All but 1 is detected correctly
All images between 0.4 and 0.5 checked. All detected correctly
All images between 0.5 and 0.7 checked. All correct, but 2 are borderline
All images above 0.7 checked. All correct
"""
# star_analysis('Detections/52_10_1.56.csv', check_results=True)
files = glob.glob('Detections/*.csv')
for file in files:
    star_analysis(file)