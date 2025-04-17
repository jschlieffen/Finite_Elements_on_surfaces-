#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:49:47 2025

@author: jschlieffen
"""


from datetime import datetime
import configparser

# =============================================================================
# This file creates the flesctrl logfile that can be found in 
# logs/general/
# =============================================================================

class Logfile:
    def __init__(self):
        self.refinement_dict = {}
            
    def write(self):
        config = configparser.ConfigParser()
        config.read('Params/config.cfg')
        run_id = config.getint('general', 'run_id')
        config['general']['run_id'] = str(run_id+1)
        with open('Params/config.cfg', 'w') as configfile:
            config.write(configfile, space_around_delimiters=False)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logfile_name = f'logs/Run_{str(run_id+1)}_{timestamp}_overview_verbose.log'
        with open(logfile_name, 'w') as file:
            file.write(f'Total number of refinements: {len(self.refinement_dict)} \n \n')
            for i, (idx, elem) in enumerate(self.refinement_dict.items()):
                file.write(f"REFINEMENT: {i} \n")
                file.write(f"Matrix A: \n {elem['A']} \n \n")
                file.write(f"Numerical Solution: \n {elem['Numerical solution']} \n \n")
                file.write(f"Analytical Solution: \n {elem['analytical solution']} \n \n")
                file.write(f"Vertex list: \n {elem['vertex list']} \n \n")
                if elem['Error estimates list'] == []:
                    file.write("Error estimates not calculated \n")
                else:
                    error_estimates = elem['Error estimates list']
                    file.write(f"L2 Error {error_estimates[0]} and Order of Convergence {error_estimates[1]} \n")
                    file.write(f"H1 Error {error_estimates[2]} and Order of Convergence {error_estimates[3]} \n")
                file.write(f"total execution time: {elem['time']} \n \n \n")
        logfile_name = f'logs/Run_{str(run_id+1)}_{timestamp}_overview_general.log'
        with open(logfile_name, 'w') as file:
            file.write(f'Total number of refinements: {len(self.refinement_dict)} \n \n')
            for i, (idx, elem) in enumerate(self.refinement_dict.items()):
                file.write(f"REFINEMENT: {i} \n")
                #file.write(f"Matrix A: \n {elem['A']} \n \n")
                #file.write(f"Numerical Solution: \n {elem['Numerical solution']} \n \n")
                #file.write(f"Analytical Solution: \n {elem['analytical solution']} \n \n")
                #file.write(f"Vertex list: \n {elem['vertex list']} \n \n")
                if elem['Error estimates list'] == []:
                    file.write("Error estimates not calculated \n")
                else:
                    error_estimates = elem['Error estimates list']
                    file.write(f"L2 Error {error_estimates[0]} and Order of Convergence {error_estimates[1]} \n")
                    file.write(f"H1 Error {error_estimates[2]} and Order of Convergence {error_estimates[3]} \n")
                file.write(f"total execution time: {elem['time']} \n \n \n")
        
    def append_refinement(self, A, rhs, numerical_solution, ana_sol, vert_dict, time, error_estimates=[]):
        vert_list = []
        for v_id,v in vert_dict.items():
            vert_list.append((v_id,v.get_coordinates()))
        self.refinement_dict[len(vert_dict)] = {
                'A' : A,
                'rhs' : rhs,
                'Numerical solution' : numerical_solution,
                'analytical solution' : ana_sol,
                'vertex list' : vert_list,
                'Error estimates list' : error_estimates,
                'time' : time
                
            }
        
        
        
        

logfile = Logfile()