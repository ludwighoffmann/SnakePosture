"""
Code to solve the optimization problem and plot the results.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
import matplotlib.legend as mlegend
import casadi as cas
import csv
from pylab import *
from scipy import *
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects

maxiter = 2000
ipopts_run = {"ipopt.sb":'yes', "ipopt.max_iter":maxiter, "ipopt.print_level":0, "print_time":False} # run
ipopts_debug = {"ipopt.sb":'yes', "ipopt.max_iter":maxiter, "ipopt.print_level":3, "print_time":False} # debug

def collocation_solver(f, x0, x_lb, x_ub, N, T=1.0, *,
                       xf=None, xf_eq=None,
                       u_lb=-1.0, u_ub=1.0, p0=None, p_lb=None, p_ub=None,
                       opt_guess=None,
                       d=3, ipopt_options=None):
    """
    Collocation solver that we will use to solve the optimization problem. See the documentation of CasAdi for further details.
    """

    if ipopt_options is None:
        ipopt_options = {"ipopt.sb":'yes', "ipopt.print_level":0, "print_time": False,"ipopt.max_iter":2000}


    # Time step
    h = T/N

    # Dimensionality of the state, control and parameter vector
    n = f.size1_in(0)
    m = f.size1_in(1)
    n_p = f.size1_in(2)

    if p_lb is None:
        p_lb = [-np.inf]*n_p
    if p_ub is None:
        p_ub = [np.inf]*n_p
    if p0 is None:
        p0 = [0.0]*n_p

    # Get collocation points
    tau_root = np.append(0, cas.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)


    # ----- Construct the NLP -----

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # NLP variables for the parameters to optimize
    P = cas.MX.sym('P', n_p)
    if n_p > 0:
        w.append(P)
        lbw.append(p_lb)
        ubw.append(p_ub)
        w0.append(p0)

    # "Lift" initial conditions
    Xk = cas.MX.sym('X0', n)
    w.append(Xk)
    lbw.append(x0)
    ubw.append(x0)
    w0.append(x0)
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cas.MX.sym('U_' + str(k), m)
        w.append(Uk)
        lbw.append([u_lb])
        ubw.append([u_ub])
        w0.append([0.0])
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = cas.MX.sym('X_'+str(k)+'_'+str(j), n)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append(x_lb)
            ubw.append(x_ub)
            w0.append(x0)

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1],Uk,P)
            g.append(h*fj - xp)
            lbg.append([0]*n)
            ubg.append([0]*n)

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = cas.MX.sym('X_' + str(k+1), n)
        w.append(Xk)
        lbw.append(x_lb)
        ubw.append(x_ub)
        w0.append(x0)
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0]*n)
        ubg.append([0]*n)

    # - Terminal condition -
    if xf is not None:
        g.append(Xk) # Xk is the final state
        lbg.append(xf)
        ubg.append(xf)
    elif xf_eq is not None:
        g.append(xf_eq(Xk)) # Xk is the final state
        n_eq = xf_eq.size1_out(0)
        lbg.append([0.0]*n_eq)
        ubg.append([0.0]*n_eq)

    # Concatenate vectors
    w = cas.vertcat(*w)
    g = cas.vertcat(*g)
    x_plot = cas.horzcat(*x_plot)
    u_plot = cas.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    solver = cas.nlpsol('solver', 'ipopt', prob, ipopt_options)

    # Function to get x and u trajectories from w
    trajectories = cas.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

    # Solve the NLP
    if opt_guess is not None:
        sol = solver(x0=opt_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    else:
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
    solution_status = solver.stats()['success']

    x_opt, u_opt = trajectories(sol['x'])
    x_opt = x_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return x_opt, u_opt, solution_status , sol['x'], sol

# VdP example from CasADi Docsp
def snake_scan_boundoptim(init_y_val,total_length,print_flag,counter,folder,MuscBound,ControlBound,GravPara,y_disc,MinimiChoiceFlag,**kwargs):
    
    """
    This function defines the problem to be solved and then calls collocation_solver to solve the problem. The meaning of the parameters:
    
    init_y_val: y-position of head
    total_length: total length of the curve
    print_flag: flag to set if solution should be saved in csv file
    MuscBound, ControlBound, and GravPara: bound on muscular force and control variable as well as the gravitational parameter alpha
    y_disc: array of all y-variable for the head positions we consider
    MinimiChoiceFlag: Choice of function to be minimized
    """

    # Problem parameters
    x_0 = -0.1 #x-position of head
    y_0 = init_y_val #y-position of head
    theta_0 = -(np.pi/2-0.2) #angle of head
    kappa_0 = 0.0 #curvature of head
    a_0 = -kappa_0 #active moment at head
    s_0 = 0.0 #value of parameter s along curve at head 
    kappa_max = np.inf #Bound on curvature 
    a_max = MuscBound #Bound on active moment
    u_max = ControlBound #Bound on control
    alpha = GravPara #Gravitational parameter alpha
    
    foldername = folder

    # Degree of interpolating polynomial
    d = 3
    # Time horizon
    T = 1.0

    # Declare model variables
    x_x = cas.SX.sym('x')
    x_y = cas.SX.sym('y')
    x_theta = cas.SX.sym('θ')
    x_kappa = cas.SX.sym('κ')
    x_a = cas.SX.sym('a')
    x_s = cas.SX.sym('s') #appended arc-length to the state for simplicity
    x = cas.vertcat(x_x, x_y, x_theta, x_kappa, x_a, x_s)
    u = cas.SX.sym('u')

    # Model equations
    dx_s = total_length
    dx_x = cas.cos(x_theta) * dx_s
    dx_y = cas.sin(x_theta) * dx_s
    dx_theta = x_kappa * dx_s
    dx_kappa = - u * dx_s + alpha * x_s * cas.cos(x_theta) * dx_s * dx_s
    dx_a = u * dx_s
    xdot = cas.vertcat(dx_x, dx_y, dx_theta, dx_kappa, dx_a, dx_s)

    # Objective term, which one is chosen is set by MinimiChoiceFlag
    if MinimiChoiceFlag == 0:
        L = u**2
    elif MinimiChoiceFlag == 1:
        L = - x_kappa * x_a + 0.001 * u**2
    elif MinimiChoiceFlag == 2:
        L = x_a**2 + 0.001 * u**2
        
    # Continuous time dynamics
    p_dummy = cas.SX.sym('p_dummy', 0)
    f = cas.Function('f', [x, u, p_dummy], [xdot, L], ['x', 'u', 'p_dummy'], ['xdot', 'L'])
    # Initial state (at head)
    x0 = [x_0, y_0, theta_0, kappa_0, a_0, s_0]

    # Final state (at lower perch)
    x_fin = 0.0
    y_fin = 0.0
    theta_fin = 0.0
    
    eq1 = x_x - x_fin
    eq2 = x_y - y_fin
    eq3 = x_theta - theta_fin
    
    eq = cas.vertcat(eq1, eq2, eq3)
    xf_eq = cas.Function('xf_eq', [x], [eq], ['x'], ['eq']) 

    # State constraints
    x_lb = [-np.inf, -np.inf, -np.inf, -kappa_max, -a_max, -np.inf]
    x_ub = [np.inf, np.inf, np.inf, kappa_max, a_max,np.inf]

    # Control bounds
    u_lb = -u_max
    u_ub = u_max

    # Control discretization
    N = 100 # number of control intervals
    
    #Solve the control problem with these boundary conditions and equations, store solution in x_opt and u_opt
    x_opt, u_opt, solution_status, _, _ = collocation_solver(f, x0, x_lb, x_ub, N, T,
                                            xf_eq=xf_eq,
                                            u_lb=u_lb, u_ub=u_ub,
                                            d=d, **kwargs)
    
    
    #Save optimal solution if print_flag == 1
    if print_flag == 1:
    
        np.savetxt(str(foldername) + "/y0_" + str(y_disc[counter]) + ".csv", (x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], np.append(np.nan, u_opt[0])), delimiter=',')
    
    
    #Integrated absolute value of active moment. Used to determine which total length has smallest value of this integral.
    if MinimiChoiceFlag == 0:
        a_val_int = np.sum(np.abs(x_opt[4]))
    elif MinimiChoiceFlag == 1:
        a_val_int = np.sum(np.abs(x_opt[4][10:]))
    elif MinimiChoiceFlag == 2:
        a_val_int = np.sum(np.abs(x_opt[4]))
    
    #Print solution status, is True if an optimal solution was found and False otherwise. If a solution was found we return either the value of the integrated active moment or the optimal solution in case print_flag == 1. See below in run_boundopt for explanation. If no solution is found we return the (random) number 10000 which is much greater than any value for a_val_int.
    print(solution_status)
    if solution_status == True:
        if print_flag == 1:
            return [x_opt[0],x_opt[1]]
        else:
            return a_val_int
    else:
        return 10000
def run_boundopt(MuscBound,ControlBound,GravPara,MinimiChoiceFlag):
    
    """
    This function solves the optimal control problem for a given set of parameters:
    
    MuscBound, ControlBound, and GravPara: bound on muscular force and control variable as well as the gravitational parameter alpha
    MinimiChoiceFlag: Choice of function to be minimized
    
    We find a solution as follows: 
        - Set a y-coordiante for the head such that the coordiante of the head is fully determined
        - The total length of the curve is unknown. Thus, we scan over a range of possible values for the total length of the curve, namely the values np.linspace(1.01*j,1.2*j,20) with j = y-coordinate of the head
        - For each length we attempt to solve the optimal control problem. If a solution is possible we compute the intgrated absolute value of the active moment and append it to the array array_max_a_val. If not solution is found we append in this case the value 10000 (the value is arbitrary but >> any realistic value we have in case a solution is found)
        - After running the optimization problem for all values of the total length, we then select the one which had the smallest intgrated absolute value of the active moment
        - For this total length we run the optimizaiton problem again, now setting the flag = 1 such that the solution is saved. 
        - This is repeated for all y-coordiantes of the head
    """
    
    #Positions of y-coordinates we consider for the head.
    y_disc = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    if not os.path.exists("Scan/"):
        os.mkdir("Scan/")
    
    if MinimiChoiceFlag == 0:
        if not os.path.exists("Scan/uSquaredMini_boundopt/"):
            os.mkdir("Scan/uSquaredMini_boundopt/")
        folder = "Scan/uSquaredMini_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)
    elif MinimiChoiceFlag == 1:
        if not os.path.exists("Scan/Minak_boundopt/"):
            os.mkdir("Scan/Minak_boundopt/")
        folder = "Scan/Minak_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)
    elif MinimiChoiceFlag == 2:
        if not os.path.exists("Scan/aSquaredMiniReg_boundopt/"):
            os.mkdir("Scan/aSquaredMiniReg_boundopt/")
        folder = "Scan/aSquaredMiniReg_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)
    else:
        return print("Invalid ControlChoiceFlag")
    os.mkdir(folder)
    
    #Actual loop to solve the optimization problem for different values for the y-coord. of the head. See above for explanation of procedure.
    curve_plot_array = []
    counter = 0
    for j in y_disc:
        array_max_a_val = []
        flag = 0
        for k in np.linspace(1.01*j,1.2*j,20):
            print("y0 = " + str(j) + "; length = " + str(k))
            array_max_a_val.append(snake_scan_boundoptim(j,k,flag,counter,folder,MuscBound,ControlBound,GravPara,y_disc,MinimiChoiceFlag,ipopt_options=ipopts_run))
        index_min = array_max_a_val.index(min(array_max_a_val))
        flag = 1
        xycoord = snake_scan_boundoptim(j,np.linspace(1.01*j,1.2*j,20)[index_min],flag,counter,folder,MuscBound,ControlBound,GravPara,y_disc,MinimiChoiceFlag,ipopt_options=ipopts_run)
        curve_plot_array.append(xycoord)
        counter += 1    
    
def all_curve_plots_boundopt(MuscBound,ControlBound,GravPara):
    
    """
    Plotting the curves (body in real space, active moment, curvature and control) for the parameters MuscBound, ControlBound, and GravPara. For this, use the data produced by run_boundopt.
    """
    
    folder = "Scan/Minak_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)
    y_disc = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc("font", family="Roboto Condensed")
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rcParams.update({"text.usetex": True})
    
    fig = plt.figure(figsize = (9.6/2.54,7/2.54))
    gs = GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    
    axis_label_size = 10
    tick_size = 8
    legend_size = 7
    line_thickness = 1.0
    
    ax1.set_aspect(0.75)
    for j in range(len(y_disc)):
        geom_quant = np.loadtxt(str(folder) + "/y0_" + str(y_disc[j]) + ".csv", delimiter=",", dtype=float)
        ax1.plot(geom_quant[0],geom_quant[1], color = "brown", alpha = (j+1)/(len(y_disc)+1), linewidth = line_thickness)
    ax1.set_xlim([-0.2,0])
    ax1.set_ylim([0,1.025])
    ax1.set_xlabel(r'$x(s)$', usetex=True, fontsize=axis_label_size)
    ax1.set_ylabel(r"$y(s)$", usetex=True, rotation=0, fontsize=axis_label_size)
    ax1.yaxis.set_label_coords(-0.6, 0.44)
    ax1.tick_params(axis='both',labelsize=tick_size)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.text(0.5, 0.5, r'(a)', transform=ax1.transAxes,fontsize=axis_label_size)

    tgrid = np.linspace(0, 1, 100+1)
    y_disc = [0.4,0.7,1.0]
    invkappa0 = 0.029
    plot_lines = []
    cont = 0
    for j in y_disc:
        geom_quant = np.loadtxt(str(folder) + "/y0_" + str(j) + ".csv", delimiter=",", dtype=float)
        ax2.plot(tgrid,-geom_quant[4][::-1]*invkappa0, color = "red", alpha = (cont+1)/(len(y_disc)), linewidth = line_thickness, linestyle = 'dashed')
        if cont == 1:
            ax2.plot([],[],alpha = 0)
        cont += 1
    leg_1 = ax2.legend([r'$\frac{\tilde{m}_a}{\kappa_0}(y_0 = 0.4)$', r'$\frac{\tilde{m}_a}{\kappa_0}(0.7)$', r'', r'$\frac{\tilde{m}_a}{\kappa_0}(1.0)$'], fontsize=legend_size, ncol=2, handlelength=1,loc='upper right', columnspacing = -2.5, handletextpad = 0.6)    
        
    cont = 0
    for j in y_disc:
        geom_quant = np.loadtxt(str(folder) + "/y0_" + str(j) + ".csv", delimiter=",", dtype=float)
        ax2.plot(tgrid,-geom_quant[3][::-1]*invkappa0, color = "blue", alpha = (cont+1)/(len(y_disc)), linewidth = line_thickness)
        if cont == 1:
            ax2.plot([],[],alpha = 0)
        cont += 1
    leg_2 = ax2.legend([r'$\frac{\kappa}{\kappa_0}(y_0 = 0.4)$', r'$\frac{\kappa}{\kappa_0}(0.7)$', r'', r'$\frac{\kappa}{\kappa_0}(1.0)$'], fontsize=legend_size, ncol=2, handlelength=1, loc='lower right',  columnspacing = -2.5, handletextpad = 0.6)
    leg_2.get_lines()[0].set_color("blue")
    leg_2.get_lines()[1].set_color("blue")
    leg_2.get_lines()[3].set_color("blue")
    leg_2.get_lines()[0].set_linestyle("solid")
    leg_2.get_lines()[1].set_linestyle("solid")
    leg_2.get_lines()[3].set_linestyle("solid")
    ax2.set_xlim([0,1.02])
    ax2.set_xticks([0,0.25,0.5,0.75,1])
    ax2.set_yticks([-0.5,0.0,0.5])
    ax2.set_ylabel(r'', usetex=True, rotation=0, fontsize=axis_label_size)
    ax2.tick_params(axis='both',labelsize=tick_size)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.add_artist(leg_1)
    ax2.text(0.5, 0.5, r'(b)', transform=ax2.transAxes,fontsize=axis_label_size)
    ax2.axhline(0, color="0.5", linewidth=0.5)
    l_array = []
    cont = 0
    for j in y_disc:
        geom_quant = np.loadtxt(str(folder) + "/y0_" + str(j) + ".csv", delimiter=",", dtype=float)
        l1, = ax3.plot(tgrid,geom_quant[6][::-1]*invkappa0, color = "purple", alpha = (cont+1)/(len(y_disc)+1), linewidth = line_thickness)
        l_array.append(l1)
        
        cont += 1
    ax3.set_xlim([0,1.02])
    ax3.set_xticks([0,0.25,0.5,0.75,1])
    ax3.set_xlabel(r'$s/l$', usetex=True, fontsize=axis_label_size)
    ax3.tick_params(axis='both',labelsize=tick_size)
    
    leg3_1 = ax3.legend([l_array[0]], [r'$\frac{u}{\kappa_0}(y_0 = 0.4)$'], ncol=1, fontsize=legend_size, handlelength=1)
    leg3_2 = mlegend.Legend(ax3, [l_array[1], l_array[2]], [r'$\frac{u}{\kappa_0}(0.7)$', r'$\frac{u}{\kappa_0}(1.0)$'], ncol=2, fontsize=legend_size, handlelength=1, columnspacing = 0.7, handletextpad = 0.6)
    leg3_1._legend_box._children.append(leg3_2._legend_box._children[1])
    leg3_1._legend_box.align="left"
    
    
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.text(0.5, 0.5, r'(c)', transform=ax3.transAxes,fontsize=axis_label_size)
    ax3.axhline(0, color="0.5", linewidth=0.5)
    
    
    plt.subplots_adjust(wspace=-0.5, hspace=-0.8)
    plt.tight_layout()
    plt.savefig(str(folder) + "/control_fig.svg",transparent=True)
    plt.close()

def Plots_Paper_SI(MuscBound,ControlBound,GravPara,y0val): 
    
    "Plot to compare the result of the optimal control problem for different cost functionals. For this, use the data produced by run_boundopt."
    
    y_disc = [0.3,1.0]
    
    folder_array = ["Scan/uSquaredMini_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara),"Scan/aSquaredMiniReg_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara),"Scan/Minak_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)]
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc("font", family="Roboto Condensed")
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rcParams.update({"text.usetex": True})
    
    fig = plt.figure(figsize = (9.6/2.54,7/2.54))
    gs = GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    
    axis_label_size = 10
    tick_size = 8
    legend_size = 7
    line_thickness = 1.0
    
    color_array = ["red", "orange", "blue"]
    
    ax1.set_aspect(0.75)
    for j in range(len(folder_array)):
        geom_quant = np.loadtxt(str(folder_array[j]) + "/y0_" + str(y_disc[y0val]) + ".csv", delimiter=",", dtype=float)
        ax1.plot(geom_quant[0],geom_quant[1], color = color_array[j], linewidth = line_thickness)
    ax1.set_xlim([-0.2,0])
    ax1.set_ylim([0,1.025])
    ax1.set_xlabel(r'$x(s)$', usetex=True, fontsize=axis_label_size)
    ax1.set_ylabel(r"$y(s)$", usetex=True, rotation=0, fontsize=axis_label_size)
    ax1.yaxis.set_label_coords(-0.6, 0.44)
    ax1.tick_params(axis='both',labelsize=tick_size)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.legend([r'$j = u^2$',r'$j = m_{\rm a}^2$',r'$j = -m_{\rm a} \kappa$'], fontsize=legend_size,handlelength=1)

    tgrid = np.linspace(0, 1, 100+1)
    invkappa0 = 0.029
    plot_lines = []
    cont = 0
    for j in range(len(folder_array)):
        geom_quant = np.loadtxt(str(folder_array[j]) + "/y0_" + str(y_disc[y0val]) + ".csv", delimiter=",", dtype=float)
        ax2.plot(tgrid,-geom_quant[4][::-1]*invkappa0, color = color_array[j], linewidth = line_thickness, linestyle = 'dashed')
        if cont == 1:
            ax2.plot([],[],alpha = 0)
        cont += 1
    leg_1 = ax2.legend([r'$\frac{\tilde{m}_a}{\kappa_0}(y_0 = 0.4)$', r'$\frac{\tilde{m}_a}{\kappa_0}(0.7)$', r'', r'$\frac{\tilde{m}_a}{\kappa_0}(1.0)$'], fontsize=legend_size, ncol=2, handlelength=1,loc='upper right', columnspacing = -2.5, handletextpad = 0.6)    
        
    cont = 0
    for j in range(len(folder_array)):
        geom_quant = np.loadtxt(str(folder_array[j]) + "/y0_" + str(y_disc[y0val]) + ".csv", delimiter=",", dtype=float)
        ax2.plot(tgrid,-geom_quant[3][::-1]*invkappa0, color = color_array[j], linestyle = 'dotted', linewidth = line_thickness)
        if cont == 1:
            ax2.plot([],[],alpha = 0)
        cont += 1
    leg_2 = ax2.legend([r'$\frac{\kappa}{\kappa_0}(y_0 = 0.4)$', r'$\frac{\kappa}{\kappa_0}(0.7)$', r'', r'$\frac{\kappa}{\kappa_0}(1.0)$'], fontsize=legend_size, ncol=2, handlelength=1, loc='lower right',  columnspacing = -2.5, handletextpad = 0.6)
    leg_2.get_lines()[0].set_linestyle("dotted")
    leg_2.get_lines()[1].set_linestyle("dotted")
    leg_2.get_lines()[3].set_linestyle("dotted")
    ax2.set_xlim([0,1.02])
    ax2.set_xticks([0,0.25,0.5,0.75,1])
    ax2.set_yticks([-0.5,0.0,0.5])
    ax2.set_ylabel(r'', usetex=True, rotation=0, fontsize=axis_label_size)
    ax2.tick_params(axis='both',labelsize=tick_size)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.add_artist(leg_1)
    ax2.axhline(0, color="0.5", linewidth=0.5)
    l_array = []
    cont = 0
    for j in range(len(folder_array)):
        geom_quant = np.loadtxt(str(folder_array[j]) + "/y0_" + str(y_disc[y0val]) + ".csv", delimiter=",", dtype=float)
        l1, = ax3.plot(tgrid,geom_quant[6][::-1]*invkappa0, color = color_array[j], linewidth = line_thickness)
        l_array.append(l1)
        
        cont += 1
    ax3.set_xlim([0,1.02])
    ax3.set_xticks([0,0.25,0.5,0.75,1])
    ax3.set_xlabel(r'$s/l$', usetex=True, fontsize=axis_label_size)
    ax3.tick_params(axis='both',labelsize=tick_size)
    ax3.legend([r'$u$ $(y_0 = 0.4)$', r'$u$ $(y_0 = 0.7)$', r'$u$ $(y_0 = 1.0)$'], fontsize=legend_size, handlelength=1)
    ax3.legend([r'$u(y_0 = 0.4)$', r'$0.7$',r'', r'$1.0$'], fontsize=legend_size, handlelength=1, ncol = 2, columnspacing = -2.5)
    
    leg3_1 = ax3.legend([l_array[0]], [r'$\frac{u}{\kappa_0}(y_0 = 0.4)$'], ncol=1, fontsize=legend_size, handlelength=1)
    leg3_2 = mlegend.Legend(ax3, [l_array[1], l_array[2]], [r'$\frac{u}{\kappa_0}(0.7)$', r'$\frac{u}{\kappa_0}(1.0)$'], ncol=2, fontsize=legend_size, handlelength=1, columnspacing = 0.7, handletextpad = 0.6)
    leg3_1._legend_box._children.append(leg3_2._legend_box._children[1])
    leg3_1._legend_box.align="left"
    
    
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.axhline(0, color="0.5", linewidth=0.5)
    
    
    plt.subplots_adjust(wspace=-0.5, hspace=-0.8)
    plt.tight_layout()
    plt.savefig("Paper_SI_" + str(MuscBound) + "_" + str(ControlBound) + "_" + str(GravPara) + "_DifferentControls_" + str(y0val) + ".svg",transparent=True)
def plot_colored_snake():
    
    "Plot of the colored snake where coloring is given by the active moment along the length of the curve."
    
    MuscBound = 50
    ControlBound = 100
    GravPara = 200
    invkappa0 = 0.029
    y_disc = [1.0]
    folder = "Scan/Minak_boundopt/MuscBound_" + str(MuscBound) + "_ControlBound_" + str(ControlBound) + "_GravPara_" + str(GravPara)
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc("font", family="Roboto Condensed")
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rcParams.update({"text.usetex": True})
    
    fig = plt.figure(figsize = (9.6/2.54,7/2.54))
    gs = GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    
    axis_label_size = 10
    tick_size = 8
    legend_size = 7
    line_thickness = 1.0
    
    def truncated_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap = plt.get_cmap('spring_r')
    n_Reds = truncated_colormap(cmap,0.2,1)

    def plot_colourline(x,y,c):
        col = n_Reds((c-np.min(c))/(np.max(c)-np.min(c)))
        for i in np.arange(len(x)-1):
            ax1.plot([x[i],x[i+1]], [y[i],y[i+1]], c=col[i],linewidth=line_thickness)
        im = ax1.scatter(x, y, c=c, s=0, cmap=n_Reds)
        return im
    
    def colorline(x, y, z=None, cmap=plt.get_cmap('copper')):
        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))
            
        z = np.asarray(z)

        segments = make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=line_thickness, path_effects=[path_effects.Stroke(capstyle="round")])
        ax1.add_collection(lc)

        return lc


    def make_segments(x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments
    
    geom_quant = np.loadtxt(str(folder) + "/y0_1.0.csv", delimiter=",", dtype=float)
    x = geom_quant[0]
    y = geom_quant[1]
    ax1.set_xlim([-0.2,0])
    ax1.set_ylim([0,1.025])
    
    xnew = []
    ynew = []

    for i in range(len(x)-1):
        xnew.append(x[i])
        ynew.append(y[i])
        
        xnew.append((x[i]+x[i+1])/2)
        ynew.append((y[i]+y[i+1])/2)
    
    xnew.append(x[-1])
    ynew.append(y[-1])
    
    xnew2 = []
    ynew2 = []

    for i in range(len(xnew)-1):
        xnew2.append(xnew[i])
        ynew2.append(ynew[i])
        
        xnew2.append((xnew[i]+xnew[i+1])/2)
        ynew2.append((ynew[i]+ynew[i+1])/2)
    
    xnew2.append(xnew[-1])
    ynew2.append(ynew[-1])
    x = xnew2[::-1]
    y = ynew2[::-1]

    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=1).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.abs(-geom_quant[4][::-1]*invkappa0)
    
    znew = []

    for i in range(len(z)-1):
        znew.append(z[i])
        znew.append((z[i]+z[i+1])/2)
    znew.append(z[-1])
    znew2 = []
    for i in range(len(znew)-1):
        znew2.append(znew[i])
        znew2.append((znew[i]+znew[i+1])/2)
    znew2.append(znew[-1])
    z = znew2

    colorline(x, y, z, n_Reds)

    ax1.set_aspect(0.75)
    ax1.set_xlim([-0.2,0])
    ax1.set_ylim([0,1.025])
    ax1.set_xlabel(r'$x(s)$', usetex=True, fontsize=axis_label_size)
    ax1.set_ylabel(r"$y(s)$", usetex=True, rotation=0, fontsize=axis_label_size)
    ax1.yaxis.set_label_coords(-0.6, 0.44)
    ax1.tick_params(axis='both',labelsize=tick_size)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.text(0.5, 0.5, r'(a)', transform=ax1.transAxes,fontsize=axis_label_size)
    
    plt.tight_layout()
    plt.savefig(str(folder) + "/colored_snake.svg",transparent=True)
    plt.close()


"""
First run the optimazation problem for different parameters and cost functions. Then plot the results, with the results corresponding to the results shown in the respective figures indicated here:
"""    
run_boundopt(50,100,200,2)
run_boundopt(50,100,200,0)
run_boundopt(50,100,0,1)
run_boundopt(50,100,200,1)
run_boundopt(50,200,200,1)
all_curve_plots_boundopt(50,100,200) #Fig3
plot_colored_snake() #Fig3
all_curve_plots_boundopt(50,100,0) #FigS7
all_curve_plots_boundopt(50,200,200) #FigS7
Plots_Paper_SI(50,100,200,0) #FigS8
Plots_Paper_SI(50,100,200,1) #FigS8
