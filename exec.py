#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:43:39 2025

@author: jschlieffen
"""

import FEM 
import Visz
import os 
import platform as pt
import numpy as np
import time

# =============================================================================
# TODOs: 1. implement error estimates 
#        2. improve the calculation of the matrix A in terms of runtime 
#           partially done, check for bugs improved running time:
#           ... min. previously, now: 30 seconds
#        3. use multithreading for the calculation of A
#        4. implement monotoring
#        5. Get rid of the dependence of the concrete surface.
#        6. implement UI for the execution.
# =============================================================================


# =============================================================================
# This file is the main executable file for executing the different functions 
# this source code provides. It does five of refinements and FEM algorithms
# If you wish to only refine the surface replace line 54 with
# FEM_cls.only_surface_refinement(). This may be helpfull since the calculation
# of the coefficient matrix A may take some time. 
# (For 1538 vertices it took between 8-10 hours)
# =============================================================================

def start_plots_surface():
    dir = os.path.dirname(__file__)
    if pt.system() == 'Windows':
        path = os.path.join(dir,'plots\\surface_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '\\'
    else:
        path = os.path.join(dir,'plots/surface_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/'
    Visz.Plot_surface(FEM_cls.surface.level_set_function,0.05,500,path + 'surface_plot.html', path +'surface_plot_with_function.html', FEM_cls.ana_sol)
    
def start_plots_discrete_surface():
    dir = os.path.dirname(__file__)
    if pt.system() == 'Windows':
        path = os.path.join(dir,'plots\\discrete_surface_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '\\'
    else:
        path = os.path.join(dir,'plots/discrete_surface_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/'
    Visz.Plot_Discrete_surface(FEM_cls.surface.vert_dict, path + 'discrete_FEM_surface_' + str(FEM_cls.surface.num_vertices) + '.html',
                               FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs),path +'discrete_FEM_function_surface_' + str(FEM_cls.surface.num_vertices) +'.html' )

def start_FEM_algorithm():
    for i in range(1,4):
        print('refinement Number: ' + str(i))
        FEM_cls.surface_refinement()
        step_FEM_algorithm()

def step_FEM_algorithm():
    print('number of nodes :', FEM_cls.surface.num_vertices)
    print('\n The coefficient matrix: \n', FEM_cls.A , '\n')
    print('bear in mind, that the matrix is not properly sorted, thus the terminal message may look like the matrix being 0, even tho this is not the case \n')
    print('rhs: \n', FEM_cls.rhs)
    print('\n numerical solution: \n', FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    ana_sol = np.zeros((FEM_cls.n,1))
    i = 0
    for v_id,v in FEM_cls.surface.vert_dict.items():
        x,y,z = v.get_coordinates()
        ana_sol[i] = x*y
        i += 1
    print('\n ana sol \n')
    print(ana_sol)
    start_plots_discrete_surface()


def main():
    global FEM_cls
    FEM_cls = FEM.FEM()
    start_time = time.time()
    start_plots_surface()
    start_FEM_algorithm()
    print('exec time: ', time.time() - start_time)
if __name__ == '__main__':
    main()