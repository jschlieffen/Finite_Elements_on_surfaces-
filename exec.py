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
import subprocess
import signal

# =============================================================================
# TODOs: 1. implement error estimates done, make larger sim. to get good results
#        2. improve the calculation of the matrix A in terms of runtime 
#           partially done, check for bugs improved running time:
#           12.3 min. previously, now: 30 seconds
#        3. use multithreading for the calculation of A
#           partially done, check for race conditions and optimize task management
#        4. implement monotoring done extended in furture
#        5. Get rid of the dependence of the concrete surface.
#        6. implement UI for the execution.
#        7. Implmente signal handler
# =============================================================================


# =============================================================================
# This file is the main executable file for executing the different functions 
# this source code provides. It does five of refinements and FEM algorithms
# If you wish to only refine the surface replace line 54 with
# FEM_cls.only_surface_refinement(). This may be helpfull since the calculation
# of the coefficient matrix A may take some time. e
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

    for i in range(1,5):
        print('refinement Number: ' + str(i))
        FEM_cls.surface_refinement()
        if i == 1:
            h,l2_error, h1_error = step_FEM_algorithm(0,0,0,True)
        else:
            h,l2_error, h1_error = step_FEM_algorithm(h, l2_error, h1_error)
    
def step_FEM_algorithm(prev_h,prev_l2_error,prev_h1_error, is_first=False):
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
    Error_estimates = FEM.error_calc(FEM_cls.surface, FEM_cls.triangles)
    l2_error = Error_estimates.l2_error(FEM_cls.ana_sol,FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    h1_error= Error_estimates.h1_error(FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs), l2_error)
    #diam = FEM_cls.h
    h = FEM_cls.h

    if not is_first:
        OOC_l2 = Error_estimates.calc_OOC(l2_error, prev_l2_error, h, prev_h)
        OOC_h1 = Error_estimates.calc_OOC(h1_error, prev_h1_error, h, prev_h)
        print(f'\n l2 error: {l2_error}')
        print(f' Order of convergence for the l2 error: {OOC_l2}')
        print(f'\n h1 error: {h1_error}')
        print(f' Order of convergence for the h1 error: {OOC_h1}')
        print(f'Errors calculated for mesh size: ', h)
    start_plots_discrete_surface()
    return h,l2_error, h1_error


#TODO: check for the implementation of Windows/Apple
def start_monotoring():
    system = pt.system()
    #print(system)
    if system == 'Windows':
        subprocess.Popen(["start", "cmd", "\k", "python monotoring/monotoring_launcher.py"], shell=True)
    elif system == 'Darwin':
        subprocess.Popen(["osascript", "-e",
                          'tell app "Terminal" to do script \"python3 monotoring/monotoring_launcher.py\"'])
    elif system == "Linux":
        #print('test')
        subprocess.Popen(["gnome-terminal", "--full-screen", "--", "python3", "monotoring/monotoring_launcher.py"])
    else:
        raise OSError("unsupported os")
        
def start_monotorin_graphs():
    system = pt.system()
    #print(system)
    if system == 'Windows':
        subprocess.Popen(["start", "cmd", "\k", "python monotoring/monotoring_graphs_launcher.py"], shell=True)
    elif system == 'Darwin':
        subprocess.Popen(["osascript", "-e",
                          'tell app "Terminal" to do script \"python3 monotoring/monotoring_graphs_launcher.py\"'])
    elif system == "Linux":
        #print('test')
        subprocess.Popen(["gnome-terminal", "--full-screen", "--", "python3", "monotoring/monotoring_graphs_launcher.py"])
    else:
        raise OSError("unsupported os")
        
    
def kill_mon():
    try:
        with open("tmp/pids_prog_bar.txt", "r") as f:
            pid = int(f.read().strip())
        
        print(f"killing process with Pid: {pid}")
        os.kill(pid, signal.SIGKILL)
        print("Process killed")
        
        os.remove("tmp/pids_prog_bar.txt")
    except Exception as e:
        print(f"Could not kill process: {e}")
        
def kill_mon_graphs():
    try:
        with open("tmp/pids_graph.txt", "r") as f:
            pid = int(f.read().strip())
        
        print(f"killing process with Pid: {pid}")
        os.kill(pid, signal.SIGKILL)
        print("Process killed")
        
        os.remove("tmp/pids_graph.txt")
    except Exception as e:
        print(f"Could not kill process: {e}")
        

def main():
    global FEM_cls
    start_monotoring()
    start_monotorin_graphs()
    FEM_cls = FEM.FEM()
    start_time = time.time()
    start_plots_surface()
    start_FEM_algorithm()
    kill_mon()
    kill_mon_graphs()
    print('exec time: ', time.time() - start_time)
if __name__ == '__main__':
    main()