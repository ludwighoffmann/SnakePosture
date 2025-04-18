"""Plots of curvature and std of curvature for python and brown tree snake from experimental data."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import pandas as pd
from scipy.stats import binned_statistic
warnings.filterwarnings('ignore')

def get_closest_to_val(interpolation_points, val):
    
    """
    Select the list of interpolation points where the total length is
    closest to whatever value I desire.
    """
    l_over_time = np.array([i[-1] for i in interpolation_points])
    diff_from_val = l_over_time - val
    closest_arg = np.argmin(np.abs(diff_from_val))
    closest_array = interpolation_points[closest_arg]
    return int(closest_arg), closest_array, min(np.abs(diff_from_val))
    

def get_dist_curve_array(val, dirs, snake_kind):
    
    """Get the actual values for curvature etc. for each trial and a given val"""
    
    distance_arrays = []
    curvature_arrays = []
    dir_order = []
    closest_args = []
    diff_from_val_array = []
    # First loop through the videos and get the interpolation points and curvatures
    for d in dirs:
        dir_order.append(d)
        data_path = d + "/full_processed/extracted_data_signed/"
        #if snake_kind == 3:
        #    data_path = d + "/full_processed_w_head/extracted_data_signed/"
        #else:
        #    data_path = d + "/full_processed_w_head/extracted_data_signed/"#"/full_processed/extracted_data_signed/"
        folders = os.listdir(data_path)
        folders = [f for f in folders if "." not in f]  # ignore .DS_Store
        folders = [f for f in folders if "pl" not in f]  # ignore .DS_Store
        folders.sort(key=int)

        interpolation_points = []

        # populate the arrays
        for f in folders:
            arr = np.load(data_path + "/" + f + "/arr.npz")
            interpolation_points.append(arr["arr_0"])

        closest_arg, closest_array, diff = get_closest_to_val(interpolation_points, val)
        closest_args.append(closest_arg)
        distance_arrays.append(closest_array)
        diff_from_val_array.append(diff)
        # now i want to get the interpolator for that specific interpolation point

        with open(data_path + "/" + f"{closest_arg}" + "/interpolator.pkl", "rb") as f:
            spl_loaded = pickle.load(f)
            curve_pred = spl_loaded(closest_array)
            curvature_arrays.append(curve_pred)

    return val, distance_arrays, curvature_arrays, closest_args, diff_from_val_array
    
def get_average(val, width, dirs, snake_kind):
    
    """Computes the average curvature etc. over all trials for a given val."""
    
    lengthsnake, distance_arrays, curvature_arrays, closest_args, diff_from_val_array = get_dist_curve_array(val, dirs, snake_kind)
    length_snake_std = np.mean(diff_from_val_array)
    lenght_snake_max_diff = np.max(diff_from_val_array)
    average = []
    std = []
    for k in range(np.size(distance_arrays[1][::-1])):
        tmp = []
        for j in range(np.size(dirs)):
            tmp.append(width*curvature_arrays[j][k]) #Use width to create dimensionless curvature
        average.append(np.mean(tmp))
        std.append(np.std(tmp))
    norm_dist_array = distance_arrays[1][::-1]/distance_arrays[1][-1]

    return average, std, norm_dist_array, lengthsnake, length_snake_std, lenght_snake_max_diff, diff_from_val_array

def run(snake_kind):
    if snake_kind == 0:
        name_snake = "bi20"
        width = 3.4 #cm
        height = 50 #cm
        val_array = [48, 60, 66, 73, 79]
    elif snake_kind == 3:
        name_snake = "python"
        width = 1.9 #cm
        height = 70 #cm
        val_array = [29, 39, 49, 59, 69]
    else:
        print("Error: Snake kind does not exist.")
        exit()
    
    #select all folders of chosen snake and height
    path = name_snake + str("/")     
    dirs = os.listdir(path)
    dirs = [path + x for x in dirs if os.path.isdir(path + x) and str(height) in x]
    
    #Create empty lists we will append values to in the follwing
    length_array = []
    length_std_array = []
    length_max_diff_array_abs = []
    length_max_diff_array_rel = []
    averages_curvature_array = []
    std_curvature_array = []

    #Now compute the average curvature where the average is over all the trials we have for that height and snake. This is done in the get_average function. Append the average curvature and the corresponding standard deviation to the respective arrays. Do the same for the length of the snake. Laslty, note the absolute and relative difference between the chosen value in val_array and the actual value we choose.
    for val in val_array:
        average, std, norm_dist_array, lengthsnake, length_snake_std, lenght_snake_max_diff, diff_from_val_array = get_average(val, width, dirs, snake_kind)
        averages_curvature_array.append(average)
        std_curvature_array.append(std)
        length_array.append(lengthsnake)
        length_std_array.append(length_snake_std)
        length_max_diff_array_rel.append(length_snake_std/lengthsnake)
        length_max_diff_array_abs.append(lenght_snake_max_diff) #In Tab. S2
        
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc("font", family="Roboto Condensed")
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    params = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}', 'text.latex.preamble': r'\usepackage{bm}'}
    plt.rcParams.update(params)    
    axis_label_size = 10
    tick_size = 8
    legend_size = 7
    line_thickness = 1.0
    
    
    #Plot for avergae curvature (Fig. 1 and Fig. S2)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (9/2.54,4./2.54))
    
    #Create legend to be used in plot
    legend_array = []    
    for i in range(np.size(val_array)):
        legend_array.append(str(length_array[i]) + r' cm')
    
    for i in range(np.size(val_array)):
        ax.plot(norm_dist_array, averages_curvature_array[i], linewidth=1.0, alpha = (i+1)/(np.size(val_array)+1),color = "blue")    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    if snake_kind == 3:
        ax.set_xlabel(r'$s/l$', fontsize=axis_label_size, usetex=True)
    else:
        ax.set_xlabel(r'$s/l$', fontsize=axis_label_size, usetex=True,color='white')
    ax.set_ylabel(r'$\frac{\kappa}{\kappa_0}$', rotation=0, fontsize=axis_label_size+5)
    if snake_kind == 3:
        ax.legend(legend_array, loc='best', bbox_to_anchor=(0.35, 0., 0.5, 0.7), fontsize=legend_size, ncol=3, handlelength=1, columnspacing = 0.7, handletextpad = 0.6)
    else:
        ax.legend(legend_array, loc='best', bbox_to_anchor=(0.45, 0., 0.5, 0.7), fontsize=legend_size, ncol=2, handlelength=1, columnspacing = 0.7, handletextpad = 0.6)
    ax.yaxis.set_label_coords(-0.12, 0.4)
    ax.set_yticks([0.2,0.0,-0.2,-0.4])
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig("figures/" + str(name_snake) + "_h_" + str(height) + "_average_curvature_fin.svg",transparent=True)
    #plt.show()
    plt.close()
    
    
    #Plot for average and std of curvature (Fig. S4)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (9/2.54,4./2.54))
    
    color_array = ["blue", "green", "orange"]
    zorder_array = [1,2,3,4,5,6]

    for i in range(np.size(val_array[::2])):
        av_plus_std = np.array(averages_curvature_array[i*2]) + np.array(std_curvature_array[i*2])
        av_minus_std = np.array(averages_curvature_array[i*2]) - np.array(std_curvature_array[i*2])
        plt.plot(norm_dist_array, averages_curvature_array[i*2], color = color_array[i], zorder=zorder_array[i*2], label = str(length_array[i*2]) + " cm")
        plt.fill_between(norm_dist_array, av_plus_std, av_minus_std, alpha = 0.7, color = color_array[i],zorder=zorder_array[i])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.set_xlabel(r'$s/l$', fontsize=axis_label_size)
    ax.set_ylabel(r"${\kappa}/{\kappa_0}$", rotation=0, fontsize=axis_label_size)
    ax.legend(fontsize=legend_size, ncol=1, handlelength=1, columnspacing = 0.7, handletextpad = 0.6)
    ax.yaxis.set_label_coords(-0.08, 0.4)
    ax.tick_params(axis='both', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig("figures/" + str(name_snake) + "_h_" + str(height) + "_std_shaded_curvature.svg",transparent=True)
    
    
run(0)
run(3) 




