#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:22:00 2025

@author: jschlieffen
"""

import FEM 
import Visz
import os 
import sys
import platform as pt
import numpy as np
import time
import subprocess
import signal
sys.path.append(os.path.abspath('logscripts/'))
sys.path.append(os.path.abspath('Params/'))
from log_msg import *
import logfile_gen 
import params as par


class exec_:
    
    def __init__(self):
        self.Par_ = par.Params('Params/config.cfg')
        self.Par_.validation_params()
        logger.success('Params valid and successfully set')
        self.FEM_cls = None
        self.start_time = time.time()
        self.running = True
        self.logfile = logfile_gen.Logfile()
        
    
    def start_algo(self):
        #signal.signal(signal.SIGINT, self.signal_handler)
        
        #signal.signal(signal.SIGTERM, self.signal_handler)
        
        if self.Par_.show_progress_bar:
            self.start_monotoring()
        if self.Par_.show_ressource_usage:
            self.start_monotoring_graphs()
        logger.info(f"Number of total refinements: {self.Par_.refinement_numbers}")
        self.FEM_cls = FEM.FEM()
        self.start_plots_surface()
        self.start_FEM_algorithm()
        self.logfile.write()
        self.cleanup()
    
    #TODO: check for the implementation of Windows/Apple
    def start_monotoring(self):
        system = pt.system()
        with open('tmp/general.txt','w') as file:
            #print('test')
            if self.Par_.calculation_of_error_estimates:
                file.write(f'set_error_calc=True \n')
            else:
                file.write(f'set_error_calc=False \n')
            file.flush()
        time.sleep(0.01)
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
            logger.critical('unsopported OS. Terminating.')
            #self.cleanup()
            sys.exit(1)
            
    def reset_monotoring(self, refinement_number = 0):
        time.sleep(5)
        with open('tmp/general.txt','a') as file:
            file.write(f'refinement_number={refinement_number} \n')
            file.flush()
            
        with open('tmp/calc_rhs.txt', 'w') as file:
            file.write(f'h1calc: 0 max = 1 \n')
            file.flush()
        with open('tmp/calc_matrix.txt', 'w') as file:
            file.write(f'h1calc: 0 max = 1 \n')
            file.flush()
        
        with open('tmp/calc_l2.txt', 'w') as file:
            file.write(f'l2calc: 0 max = 1 \n')
            file.flush()
            
        with open('tmp/calc_h1.txt', 'w') as file:
            file.write(f'h1calc: 0 max = 1 \n')
            file.flush()
        
    def start_monotoring_graphs(self):
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
            logger.critical('unsopported OS. Terminating.')
            #self.cleanup()
            sys.exit(1)
            
    def start_plots_surface(self):
        self.FEM_cls = FEM.FEM()
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
        Plot_cls = Visz.Plot_surface(self.FEM_cls.surface.level_set_function,0.05,500,path + 'surface_plot.html',
                                     path +'surface_plot_with_function.html', self.FEM_cls.ana_sol)
        Plot_cls.create_plot()
        Plot_cls.create_plot_function()
        Plot_cls.func = self.FEM_cls.surface.mean_curvature
        Plot_cls.title_func_plot = path +'surface_plot_with_mean_curvature.html'
        Plot_cls.create_plot_function()
        
    def start_plots_discrete_surface(self):
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
        Plot_cls = Visz.Plot_Discrete_surface(self.FEM_cls.surface.vert_dict, path + 'discrete_FEM_surface_' + str(self.FEM_cls.surface.num_vertices) + '.html',
                                              self.FEM_cls.solve_sytem(self.FEM_cls.A, self.FEM_cls.rhs),
                                              path +'discrete_FEM_function_surface_' + str(self.FEM_cls.surface.num_vertices) +'.html' )
        Plot_cls.create_plot()
        Plot_cls.create_plot_func_values()
        mean_curvature = np.zeros((self.FEM_cls.surface.num_vertices,1))
        i = 0
        for v_id,v in self.FEM_cls.surface.vert_dict.items():
            x,y,z = v.get_coordinates()
            mean_curvature[i] = self.FEM_cls.surface.mean_curvature(x, y, z)
            i += 1
            
        Plot_cls.func_vals = mean_curvature
        Plot_cls.title_func_plot =  path +'mean_curvature_on_discrete_surface_' + str(self.FEM_cls.surface.num_vertices) +'.html' 
        Plot_cls.create_plot_func_values()
        
        
    def start_FEM_algorithm(self):

        for i in range(1,self.Par_.refinement_numbers+1):
            self.reset_monotoring(i)
            logger.info('refinement Number: ' + str(i))
            start_time = time.time()
            self.FEM_cls.surface_refinement()
            if self.Par_.calculation_of_error_estimates:
                logger.success('Refinement process finsished. Start with Calculation of the error and the order of convergence')
                if i == 1:
                    h,l2_error, h1_error,OOC_l2, OOC_h1 = self.calculation_error_estimates(0,0,0,True)
                else:
                    h,l2_error, h1_error,OOC_l2, OOC_h1  = self.calculation_error_estimates(h, l2_error, h1_error)
                logger.info(f'Errors calculated for mesh size: {h}')
                logger.success('Calculation of error estimates finsished start with plots')
            else:
                logger.success('Refinement process finsished. Start Plots')
            self.start_plots_discrete_surface()
            exec_time_ref = time.time() - start_time
            if self.Par_.calculation_of_error_estimates:
                ana_sol = np.zeros((self.FEM_cls.n,1))
                i = 0
                for v_id,v in self.FEM_cls.surface.vert_dict.items():
                    x,y,z = v.get_coordinates()
                    ana_sol[i] = self.FEM_cls.ana_sol(x, y, z)
                    i += 1
                #print(self.FEM_cls.A)
                self.logfile.append_refinement(
                    self.FEM_cls.A, self.FEM_cls.rhs, 
                    self.FEM_cls.solve_sytem(self.FEM_cls.A, self.FEM_cls.rhs),
                    ana_sol, self.FEM_cls.surface.vert_dict, exec_time_ref,
                    [l2_error,OOC_l2, h1_error, OOC_h1]
                    )
                
    
    def calculation_error_estimates(self,prev_h,prev_l2_error,prev_h1_error, is_first=False):
        Error_estimates = FEM.error_calc(self.FEM_cls.surface, self.FEM_cls.triangles)
        l2_error = Error_estimates.l2_error(self.FEM_cls.ana_sol,
                                            self.FEM_cls.solve_sytem(self.FEM_cls.A, self.FEM_cls.rhs))
        
        abs_error = Error_estimates.absolute_error(self.FEM_cls.ana_sol,
                                                   self.FEM_cls.solve_sytem(self.FEM_cls.A, self.FEM_cls.rhs))
        logger.info(f'absolute error: {abs_error}')
        logger.info(f'l2 error: {l2_error}')
        h1_error= Error_estimates.h1_error(self.FEM_cls.solve_sytem(self.FEM_cls.A, self.FEM_cls.rhs), l2_error)
        #h1_error = 0
        #OOC_h1 = 0
        logger.info(f'h1 error: {h1_error}')
        h = self.FEM_cls.h

        if not is_first:
            OOC_l2 = Error_estimates.calc_OOC(l2_error, prev_l2_error, h, prev_h)
            logger.info(f'Order of convergence for the l2 error: {OOC_l2}')
            OOC_h1 = Error_estimates.calc_OOC(h1_error, prev_h1_error, h, prev_h)
            logger.info(f'Order of convergence for the h1 error: {OOC_h1}')
            
            if OOC_l2 < 0:
                logger.warning(f'Order of convergence of the l2 error is negative. Should not be the case')
            if OOC_h1 < 0:
                logger.warning(f'Order of convergence of the h1 error is negative. Should not be the case')
        else:
            OOC_l2 = 0
            OOC_h1 = 0
        return h,l2_error, h1_error, OOC_l2, OOC_h1
    
    #Currently not used. If the source code is frocefully terminated. One has to clean up the source code manually
    def signal_handler(self,signum, frame):
        if signum == signal.SIGINT:
            logger.error(f'received signal {signum}. Handling termination')
            self.cleanup()
        elif signum == signal.SIGTERM:
            logger.error(f'received signal {signum}. Handling termination')
            self.cleanup()
    
    
    def cleanup(self):
        if self.Par_.show_progress_bar:
            self.kill_mon()
        if self.Par_.show_ressource_usage:
            self.kill_mon_graphs()
        logger.info(f'Program terminating. Total execution time: {time.time() - self.start_time}')
            
    def kill_mon(self):
        try:
            with open("tmp/pids_prog_bar.txt", "r") as f:
                pid = int(f.read().strip())
            
            logger.info(f"killing process with Pid: {pid}")
            os.kill(pid, signal.SIGKILL)
            logger.success("Process killed")
            
            os.remove("tmp/pids_prog_bar.txt")
        except Exception as e:
            logger.error(f"Could not kill process: {e}")
            
    def kill_mon_graphs(self):
        try:
            with open("tmp/pids_graph.txt", "r") as f:
                pid = int(f.read().strip())
            
            logger.info(f"killing process with Pid: {pid}")
            os.kill(pid, signal.SIGKILL)
            logger.success("Process killed")
            
            os.remove("tmp/pids_graph.txt")
        except Exception as e:
            logger.error(f"Could not kill process: {e}")
            
            
def main():
    exe = exec_()
    exe.start_algo()
    #exe.start_plots_surface()
    
if __name__ == '__main__':
    main()
    