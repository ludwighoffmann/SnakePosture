The stability problem is solved in Julia with the relevant modules in the folder here. 

In Convolution.jl some functions are defined that are used to efficiently compute convolutions in Beam.jl. In Beam.jl the equations describing the beam with local and non-local feedback are defined. In Bifurcation.jl the functions to create the data for the figures in the data are defined 

After starting Julia and including these modules, the four functions in Bifurcation.jl can be run. They find the bifurcation diagram for different values of parameters and output either the data for the full bifurcation diagram or just the coordinate of initial instability.

stability_plots.py is used to create some of the figures of the paper.

The two mathematica notebook files are used to create the plots of the example curves and the 3D stability diagram, respectively. 