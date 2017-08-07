import glob
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
    file_dir = ('/Users/Sahl/Desktop/University/Year_Summer_4/Summer_Project/Data/')

    data_actual_LOW = np.array(pd.read_csv('Detections_actual_asym_low.csv'))
    data_actual_HIGH = np.array(pd.read_csv('Detections_actual_asym_high.csv'))
    data_parameter_CHECK = np.array(pd.read_csv(filename))

    # print(data_actual_LOW[data_actual_LOW[:, 0]=='587725590381789417.fits'])
    # print(data_parameter_CHECK[data_parameter_CHECK[:, 0]=='587725590381789417.fits'])
    # print('\n')
    print(data_parameter_CHECK.shape, len(data_actual_LOW), len(data_actual_HIGH))

    diff = np.zeros(len(data_parameter_CHECK[:, 2]))
    diff_HIGH = np.zeros(len(data_actual_HIGH[:, 2]))
    diff_LOW = np.zeros(len(data_actual_LOW[:, 2]))
    count_LOW = 0
    count_HIGH = 0
    count_total = 0
    count_0pt3 = 1

    for i in range(len(data_parameter_CHECK)):
        if data_parameter_CHECK[i, 1] > 0.4:
            # print(np.where(data_parameter_CHECK[i, 0] == data_actual_LOW[:, 0])[0], data_parameter_CHECK[i, 0])
            index = np.where(data_parameter_CHECK[i, 0] == data_actual_HIGH[:, 0])[0][0]
            diff[i] = np.abs(data_actual_HIGH[index, 2]-data_parameter_CHECK[i, 2])
            diff_HIGH[count_HIGH] = diff[i]
            if check_results:
                if diff[i] == 1:
                    print(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2], 'HIGH',
                          data_actual_HIGH[index, 2], data_actual_HIGH[index, 0])
                    # plot_image(fits.open(file_dir+data_parameter_CHECK[i, 0])[0].data)
                    # plt.title('{}: {}'.format(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2]))
            count_HIGH += 1
        else:
            # print(np.where(data_parameter_CHECK[i, 0] == data_actual_LOW[:, 0])[0], data_parameter_CHECK[i, 0])
            index = np.where(data_parameter_CHECK[i, 0] == data_actual_LOW[:, 0])[0][0]
            diff[i] = np.abs(data_actual_LOW[index, 2]-data_parameter_CHECK[i, 2])
            diff_LOW[count_LOW] = diff[i]
            if check_results:
                if diff[i] == 1:
                    print(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2], 'LOW',
                          data_actual_LOW[index, 2], data_actual_LOW[index, 0])
                    # plot_image(fits.open(file_dir+data_parameter_CHECK[i, 0])[0].data)
                    # plt.title('{}: {}'.format(data_parameter_CHECK[i, 0], data_parameter_CHECK[i, 2]))
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
    plt.show()
    # bins = int(filename.split('/')[-1].split('.csv')[0].split('_')[0])
    # bins_avg = int(filename.split('/')[-1].split('.csv')[0].split('_')[1])
    # t_factor = float(filename.split('/')[-1].split('.csv')[0].split('_')[2])
    print(filename.split('/')[-1], np.sum(diff), np.sum(diff_HIGH), np.sum(diff_LOW))

    # return [bins, bins_avg, t_factor, np.sum(diff_HIGH)]
def read_maxima_from_file(filename):
    with open(filename, encoding="utf-8") as file:
        my_list = file.readlines()

    maxima = []
    galaxy_names = []
    no_of_maxima = []
    galaxy_count = 0
    for num, x in enumerate(my_list):
        if not x.startswith('#'):
            x_split = x.strip().split(',')
            if len(x_split[0]) > 0:
                galaxy_names.append(x_split[0])
                maxima.append([x_split[1], x_split[2], x_split[3]])
                no_of_maxima.append(1)
                galaxy_count += 1
            else:
                no_of_maxima[galaxy_count-1] += 1
                maxima.append([x_split[1], x_split[2], x_split[3]])

    print(galaxy_count)
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

    # print(len(maxima_img), len(galaxy_names))

    for i, j in zip(galaxy_names, no_of_maxima):
        print(i, j)

    return galaxy_names, maxima_img, no_of_maxima

def asymmetry_vs_maxima():
    data_asym = np.array(pd.read_csv('Detections_2k/asymmetry_2k.csv'))
    __, __, data_maxima = read_maxima_from_file('maxima2.csv')

    plt.figure()
    plt.plot(data_asym[:, 5], data_maxima[1:], '.')
    plt.show()

# asymmetry_vs_maxima()


def asymmetry_difference():
    data_s3 = np.array(pd.read_csv('Detections_2k/asymmetry_2k.csv'))
    data_s7 = np.array(pd.read_csv('Detections_2k_size_7/asymmetry_2k_s7.csv'))

    diff = data_s3[:, 5]-data_s7[:, 5] 
    for d, dif in enumerate(diff):
        if dif != 0:
            if data_s3[d, 5] > 0.2 and data_s7[d, 5] < 0.2:
                print(data_s3[d, 0], data_s3[d, 5], data_s7[d, 5], data_s3[d, 6], data_s7[d, 6])
                # print("'{}'".format(data_s3[d, 0]), end=', ')

def roc_classification(filename):
    data_actual_LOW = np.array(pd.read_csv('Detections_actual_asym_low.csv'))
    data_actual_HIGH = np.array(pd.read_csv('Detections_actual_asym_high.csv'))
    data_parameter_CHECK = np.array(pd.read_csv(filename))

    t_positives_high, f_positives_high, tot_positives_high, tot_negatives_high = [], [], [], []
    t_positives_low, f_positives_low, tot_positives_low, tot_negatives_low = [], [], [], []
    t_positives_all, f_positives_all, tot_positives_all, tot_negatives_all = [], [], [], []

    for i in range(len(data_parameter_CHECK)):
        if data_parameter_CHECK[i, 1] > 0.4:
            index = np.where(data_parameter_CHECK[i, 0] == data_actual_HIGH[:, 0])[0][0]
            if data_actual_HIGH[index, 2] and data_parameter_CHECK[i, 2]:
                t_positives_high.append(1)
                t_positives_all.append(1)
                # print(data_actual_HIGH[index, 2], data_parameter_CHECK[i, 2], '| True positive')
            if not data_actual_HIGH[index, 2] and data_parameter_CHECK[i, 2]:
                f_positives_high.append(1)
                f_positives_all.append(1)
                # print(data_actual_HIGH[index, 2], data_parameter_CHECK[i, 2], '| False positive')
            if data_actual_HIGH[index, 2]:
                tot_positives_high.append(1)
                tot_positives_all.append(1)
            if not data_actual_HIGH[index, 2]:
                tot_negatives_high.append(1)
                tot_negatives_all.append(1)
        else:
            index = np.where(data_parameter_CHECK[i, 0] == data_actual_LOW[:, 0])[0][0]
            if data_actual_LOW[index, 2] and data_parameter_CHECK[i, 2]:
                t_positives_low.append(1)
                t_positives_all.append(1)
                # print(data_actual_HIGH[index, 2], data_parameter_CHECK[i, 2], '| True positive')
            if not data_actual_LOW[index, 2] and data_parameter_CHECK[i, 2]:
                f_positives_low.append(1)
                f_positives_all.append(1)
                # print(data_actual_HIGH[index, 2], data_parameter_CHECK[i, 2], '| False positive')
            if data_actual_LOW[index, 2]:
                tot_positives_low.append(1)
                tot_positives_all.append(1)
            if not data_actual_LOW[index, 2]:
                tot_negatives_low.append(1)
                tot_negatives_all.append(1)

    tpr_high = np.sum(t_positives_high)/np.sum(tot_positives_high)
    fpr_high = np.sum(f_positives_high)/np.sum(tot_negatives_high)

    tpr_low = np.sum(t_positives_low)/np.sum(tot_positives_low)
    fpr_low = np.sum(f_positives_low)/np.sum(tot_negatives_low)

    tpr_all = np.sum(t_positives_all)/np.sum(tot_positives_all)
    fpr_all = np.sum(f_positives_all)/np.sum(tot_negatives_all)

    # print(tpr_high, fpr_high, tpr_low, fpr_low)
    return tpr_high, fpr_high, tpr_low, fpr_low, tpr_all, fpr_all
    # print(len(t_positives_high), len(f_positives_high), len(tot_positives_high))

def plot_roc():
    files = glob.glob('Detections_best/*_*_*.csv')
    t_pr_high, f_pr_high = np.zeros_like(files), np.zeros_like(files)
    t_pr_low, f_pr_low = np.zeros_like(files), np.zeros_like(files)
    t_pr_all, f_pr_all = np.zeros_like(files), np.zeros_like(files)
    for f, file in enumerate(files):
        t_pr_high[f], f_pr_high[f], t_pr_low[f], f_pr_low[f], t_pr_all[f], f_pr_all[f] = roc_classification(file)
        print('{}: {:.4f}, {:.4f} | {:.4f}, {:.4f} | {:.4f}, {:.4f}'.format(file.split('/')[1], float(t_pr_high[f]), float(f_pr_high[f]),
                                                float(t_pr_low[f]), float(f_pr_low[f]),
                                                float(t_pr_all[f]), float(f_pr_all[f])))
        # star_analysis(file)
    x = np.linspace(0, 1, 100)
    y = x
    fig = plt.figure()
    ax = fig.gca()
    ax.tick_params(axis='x', colors='white')
    ax.set_xlabel('False Positive rate')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='y', colors='white')
    ax.set_ylabel('True Positive rate')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    # plt.plot(f_pr_high, t_pr_high, '.')
    # plt.plot(f_pr_low, t_pr_low, 'g.')
    ax.plot(f_pr_all, t_pr_all, 'k.')
    ax.plot(x, y, 'r--', label='Random guess')
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=1)
    plt.legend()
    # fig.savefig('Presentation/roc_classification_best.png', facecolor='none', bbox_inches='tight')
    # plt.show()

def read_datah5():
    data = pd.read_hdf('data/data.h5')
    p_mg = np.array(data['P_MG'])
    names = np.array(data['OBJID'], dtype=str)
    
    my_data = pd.read_csv('Detections_best/asymmetry_50k.csv')
    detections = pd.read_csv('Detections_best/50k_52_8_1.72.csv')

    det_merged = my_data.merge(detections, on='Galaxy_name')
    det_merged['OBJID'] = det_merged.apply(lambda x: int(x['Galaxy_name'].split('.')[0]), axis=1)
    # det_merged = det_merged.join(objid, how='outer')
    # print(det_merged)
    data_merge = data.join(det_merged.set_index('OBJID'), on='OBJID', how='inner')
    # print(data_merge)
    # when merging with big data file, use inner/right join

    plt.figure()
    plt.plot(data_merge['P_MG'], data_merge['Min_A_flux_180_y'], '.')
    plt.show()
    
    # n = my_names[0].split('.')[0]
    # print(np.where(n == names))

    # asym_plot = np.zeros_like(my_asym)
    # asym_plot = []
    # p_mg_plot = []
    
    # for count, name in enumerate(my_names):
    #     if not detections[count, 2]:
    #         index = np.where(name.split('.')[0] == names)[0][0]
    #         asym_plot.append(my_asym[count])
    #         p_mg_plot.append(p_mg[index])
    #         if my_asym[count] < 0.2 and p_mg[index] != 0:
    #             print(name, my_asym[count], p_mg[index])
    #     # if count%100 == 0:
    #     #     print(count)
    #     # print(name, names[index])

    # plt.figure()
    # plt.plot(p_mg_plot, asym_plot, '.')
    # plt.xlabel('p_mg')
    # plt.ylabel('Asymmetry')
    # plt.show()
    

    
    # print(names[0:100])

read_datah5()
# plot_roc()

# roc_classification('Detections/52_8_1.68.csv')
# asymmetry_difference()
# asymetry_analysis()

# """
# All images between 0.25 and 0.3 checked. All but 1 is detected correctly. Another has star, but is sucessfully isolated
# All images between 0.3 and 0.4 checked. All but 1 is detected correctly
# All images between 0.4 and 0.5 checked. All detected correctly
# All images between 0.5 and 0.7 checked. All correct, but 2 are borderline
# All images above 0.7 checked. All correct
# """
# star_analysis('Detections_best/53_8_1.73.csv', check_results=True)

# files = glob.glob('Detections_best_home/*_*_*.csv')
# # print(files)
# for f, file in enumerate(files):
#     star_analysis(file)
