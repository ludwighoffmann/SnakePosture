"""
Code to create some of the plots associated with the stability analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
import matplotlib.legend as mlegend
import csv
from pylab import *
    
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc("font", family="Roboto Condensed")
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")
plt.rcParams.update({"text.usetex": True})
axis_label_size = 10
tick_size = 8
legend_size = 7
line_thickness = 1.0

colors = plt.cm.gist_heat(np.linspace(0,0.8,6))

def Fig4a_bifurcation_diagram():
    
    """Plot the bifurcation diagram Theta_max vs alpha l^3 for different values of lambda using the data in Bifurcation_Diagram_Data from the Julia simulation."""
    
    data_bifurcation_diagram = []

    for j in range(6):
        with open("Bifurcation_Diagram_Data/stabs" + str(j)+ ".csv") as fp:
            reader = csv.reader(fp, delimiter=",")
            data_read = [row for row in reader]
        tmp = np.array(data_read)
        tmp = tmp.flatten()
        tmp = tmp[1:].astype(float)
        data_bifurcation_diagram.append(tmp)

    fig = plt.figure(figsize=(9.6/2.54,4./2.54))
    ax2 = plt.gca()

    a_val = np.linspace(6,20,500)

    for j in range(6):
        ax2.plot(a_val,data_bifurcation_diagram[j], color = colors[j], linewidth = line_thickness)
        
    #The legend is \lambda/l_p. In the simulations we solve (in the passive case) the equation c²∂_x^2 θ - a(1-s) cos(θ) = 0. Thus, c² = B_p and a = \rho g and therefore alpha = a/c^2. We set a = c^2 = 1 in the simulations; thus, alpha = 1. From l_p^3 \alpha \approx 7.837 we find l_p \approx 2
    
    legend_array = [r"$\lambda = 0.0$", r"$0.05$", r"$0.125$", r"$0.25$", r"$0.5$", r'$\infty$']
    ax2.legend(legend_array, fontsize = legend_size, ncol = 2, handlelength=1, columnspacing = 0.7)
    ax2.set_xlabel(r'$\alpha l^3$', usetex=True, fontsize=axis_label_size)
    ax2.set_ylabel(r"$\theta_{\rm max}$", usetex=True, rotation=0, fontsize=axis_label_size)
    ax2.tick_params(axis='both',labelsize=tick_size)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_position(("data", 6.0))
    plt.savefig("Bifurcation_Diagram_Data/BifurcationDiagram.svg",transparent=True)     
    plt.close()

def Fig4b_crit_curves():
    fig = plt.figure(figsize = (3.8/2.54,3.4/2.54))
    ax2 = plt.gca()
    

    legend_array = [3,2,1,0.9,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.1,0.05]
    points_bifurc = [1, 3, 6, 12]
    bval_array = ["b_100","b_50","b_25"]
    bvals = [1,0.5,0.25]
    lp = 2 #see comment above

    linestyle_array = ["dotted", "solid", "dashed"]
    colors_lines = plt.cm.viridis(np.linspace(0.3,0.7,3))
    colors_dots = plt.cm.gist_heat(np.linspace(0,0.8,6))

    count = 2
    for bval in bval_array:
        data_bifurcation_diagram = []
        for j in range(15):
            with open("Crit_Curves_Data/" + str(bval) + "/stabs_" + str(j)+ ".csv") as fp:
                reader = csv.reader(fp, delimiter=",")
                data_read = [row for row in reader]
            tmp = np.array(data_read)
            tmp = tmp.flatten()
            tmp = tmp[1:].astype(float)
            data_bifurcation_diagram.append(tmp)

        data_reorg = np.array(data_bifurcation_diagram[::-1])
        lambda_array = np.array(legend_array[::-1])
        a_val = np.linspace(0.1,16,1000)

        zero_array = [] #find now the smallest lambda value for which teh bifurcation curve is non-zero
        yval = []

        for i in range(len(lambda_array)):
            for j in range(len(a_val)):
                if data_reorg[i][j] > 0.00001:
                    zero_array.append(j)
                    break

        for i in range(len(data_reorg)):
            yval.append(a_val[zero_array[i]])


        ax2.plot(np.concatenate((np.array([0]),lambda_array))/lp, np.concatenate((np.array([(bvals[2 - count] + 1) * 7.837 ]),yval)), linewidth = line_thickness, color = colors_lines[count], linestyle = linestyle_array[count])


        if str(bval) == "b_50":
            x_scatt = np.zeros(len(points_bifurc)+1)
            y_scatt = np.zeros(len(points_bifurc)+1)
            x_scatt[0] = 0
            y_scatt[0] = (bvals[2 - count] + 1) * 7.837
            for jj in range(len(points_bifurc)):
                x_scatt[jj+1] = lambda_array[points_bifurc[jj]]/lp
                y_scatt[jj+1] = yval[points_bifurc[jj]]

        count -= 1

    ax2.plot(np.concatenate((np.array([0]),lambda_array))/lp, 7.837 * np.ones(len(lambda_array)+1), color=colors_dots[-1], linewidth=line_thickness, zorder = 199, linestyle = "solid")

    for jj in range(len(points_bifurc)+1):
        ax2.scatter(x_scatt[jj], y_scatt[jj], color = colors_dots[jj], zorder = 200, s = 15, marker = '*')


    legend_array = [r"$B_{\rm a} = 1.0$", r"$0.5$", r"$0.25$", r"$0.0$"]
    ax2.legend(legend_array, fontsize = legend_size, ncol = 2, handlelength=1, columnspacing = 0.7)
    ax2.set_xlabel(r'$\lambda/l_p$', usetex=True, fontsize=axis_label_size)
    ax2.set_ylabel(r"$\alpha l^3_{\rm c}$", usetex=True, rotation=0, fontsize=axis_label_size)
    ax2.tick_params(axis='both',labelsize=tick_size)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlim([-0.01,0.76])
    ax2.set_ylim([7,16.1])
    #ax2.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax2.set_xticks([0.0, 0.25, 0.5, 0.75])
    ax2.set_yticks([8, 10, 12, 14, 16])
    ax2.spines["left"].set_position(("data", 0.0))
    ax2.spines["bottom"].set_position(("data", 7.))
    plt.savefig("Crit_Curves_Data/CritLength_Lambda.svg",transparent=True)  
    #plt.show()
    plt.close()
def FigS9a_stability_diagram():
    
    """Plot the first 2D stability diagram with the data obtained from the Julia simulation."""
    
    fig = plt.figure(figsize = (3.8/2.54,3.8/2.54))
    ax1 = plt.gca()
    a_val_sim = [0.8708302563726082, 1.7416453923821766, 2.612460528391745, 3.4832756644013134, 4.354090800410882, 5.22490593642045, 6.095721072430019, 6.966536208439587,7.837351344449155, 7.837351344449155, 8.708166480458724, 9.578981616469248, 10.449796752493947, 11.320611888481618, 12.19142702448677, 13.062242160500812, 13.933057296519111, 14.803872432527674]
    b_val_sim = np.append(np.linspace(-1, 0, 10)[1:], np.linspace(0, 1, 10)[:-1])
    b_val_anal = np.linspace(-1, 1, 1000)
    a_val_anal = 7.8373 * (1 + b_val_anal)
    ax1.plot(a_val_anal, b_val_anal, color = "black", alpha = 1, linewidth = line_thickness)
    ax1.scatter(a_val_sim, b_val_sim, color = "black", alpha = 1, linewidth = line_thickness, s = 1.5, marker = ',')
    ax1.set_xticks([0, 5, 10, 15])
    ax1.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax1.set_xlabel(r'$\alpha l^3$', usetex=True, fontsize=axis_label_size)
    ax1.set_ylabel(r'$B_{\rm a}$', usetex=True, rotation=0, fontsize=axis_label_size)
    ax1.xaxis.set_label_coords(0.5, -0.15)
    ax1.yaxis.set_label_coords(-0.25, 0.44)
    ax1.tick_params(axis='both',labelsize=tick_size)
    ax1.spines["right"].set_position(("data", 15.67))
    ax1.spines["top"].set_position(("data", 1.0))
    ax1.spines["left"].set_position(("data", 0.0))
    ax1.spines["bottom"].set_position(("data", -1.0))
    plt.tight_layout()
    plt.savefig("Stability_Diagrams_Plots/S9a_stability_diag.svg",transparent=True)   
    plt.close()

def FigS9b_stability_diagram():
    
    """Plot the second 2D stability diagram with the data obtained from the Julia simulation."""
    
    fig = plt.figure(figsize = (3.8/2.54,3.8/2.54))
    ax1 = plt.gca()
    a_val_sim = [3.551003530593178,3.895596831336392,4.2504179111840745, 4.615007332650709, 4.988937572262826, 5.371809696815527, 5.763250434615441,6.162909545621834,6.570457515409148,6.985583694356498,7.407994423297005,7.837351344449155 ,7.837351344449155, 8.273570782880345, 8.716220888901471, 9.165122187564295, 9.620045862577227, 10.08077300049573, 10.547093797747818, 11.018806808213883, 11.495718294334258, 11.977641631466554,12.464396743512694,12.955809659838689]
    b_val_sim = np.append(np.append(np.append([-0.97777778,-0.88888889],np.linspace(-0.8, 0, 10)),np.linspace(0, 0.8, 10)),[0.88888889, 0.97777778])
    ax1.plot(a_val_sim, b_val_sim, color = colors[3], alpha = 1, linewidth = line_thickness)
    ax1.scatter(a_val_sim, b_val_sim, color = colors[3], s = 1.5, marker = ',')
    ax1.set_xticks([0, 5, 10, 15])
    ax1.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax1.set_xlabel(r'$\alpha l^3$', usetex=True, fontsize=axis_label_size)
    ax1.set_ylabel(r'$B_{\rm a}$', usetex=True, rotation=0, fontsize=axis_label_size)
    ax1.xaxis.set_label_coords(0.5, -0.15)
    ax1.yaxis.set_label_coords(-0.29, 0.44)
    ax1.tick_params(axis='both',labelsize=tick_size)
    ax1.spines["right"].set_position(("data", 15.67))
    ax1.spines["top"].set_position(("data", 1.0))
    ax1.spines["left"].set_position(("data", 0.0))
    ax1.spines["bottom"].set_position(("data", -1.0))
    plt.tight_layout()
    plt.savefig("Stability_Diagrams_Plots/S9b_stability_diag.svg",transparent=True)
    plt.close()

Fig4a_bifurcation_diagram()
Fig4b_crit_curves()
FigS9a_stability_diagram()
FigS9b_stability_diagram()