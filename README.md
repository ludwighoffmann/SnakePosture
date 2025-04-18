The code for the analysis of the experimental videos as well as the simulation code used in the paper "Postural control in an upright snake" 
by Ludwig A. Hoffmann, Petur Bryde, Ian C. Davenport, S Ganga Prasath, Bruce C. Jayne, and L Mahadevan.

1_ImageAnalysis: Contains the code for the image analysis. 

2_ProportionalFeedback: Contains the auto-07p code used to solve the equation for a proportional feedback. Running the auto files performs a parameter scan and outputs 
                        the bifurcation diagram and real-space solution. See readme.txt in this folder for more details.
                        
3_OptimalPosture: Contains the python code using CasAdi to solve the optimal control problem. Running the file snake_posture.py first solves the optimal control problem 
                  for the different sets of parameters used in the paper and then creates the plots. See readme.txt in this folder for more details.
                  
4_Stability: Julia code used to solve the stability problem with the four functions in the Bifurcations.jl module corresponding to different cases used in the paper. 
              The mathematica notebooks and python file are used to create the plots. See readme.txt in this folder for more details.
