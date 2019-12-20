Code supporting the paper:
Entanglement Detection Beyond Measuring Fidelities
M. Weilenmann, B. Dive, D. Trillo, E. A. Aguilar and M. Navascués 
20th December 2019


########################## LICENSE ############################

The code here is made available under the Creative Commons Attribution license (CC BY 4.0)
(https://creativecommons.org/licenses/by/4.0/


####################### INSTALLATION ##########################

The contents of the package do not need any installation beyond the packages required below.
The package can simply be downloaded and run directly. Not that the utils.py module must be in the
same directory as the scripts when they are run.


Packages required: 
python, version > 3.6.8 
numpy, version > 1.17.3
cvxpy, version > 1.024
mosek, version > 9.0.91


cvxpy:
	Installation instructions available at (https://www.cvxpy.org/install/)

mosek:
	Installation instructions available at (https://docs.mosek.com/9.0/install/installation.html)
	An academic license can be obtained for free at https://www.mosek.com/products/academic-licenses/


######################### USAGE #############################

The package is organised as follows. utils.py contains the functions that allow the manipulation of
the quantum objects and the basic SDPs. Each of the other files have the scripts that correspond to
a particular example, figure, or table in the manuscript, and can be run independently.
