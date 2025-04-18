The three cases we consider here and solve with auto-07p have the same structure:

- The program [see documentation of auto for more details]:
	- Equations, initial, boundary conditions are defined in snakeNL.f90
	- Parameters for the numeric solver are set in c.snakeNL
	- Program is run in snakeNL.auto

After installing auto-07p this can be run with the command "auto snakeNL.auto"

- Result:
	- bifurcation diagram data in b.mu
	- solution in real space data in s.mu

- Plots:
	- We create the plots used in the paper using the Jupyter notebook file plots.ipynb



- Code and analysis by LAH and SGP
- Fig. 2 in main text, and corresponding part of SI