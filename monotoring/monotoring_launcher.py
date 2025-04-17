#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:37:17 2025

@author: jschlieffen
"""

import os
import monotoring
import curses

with open("tmp/pids_prog_bar.txt", "w") as f:
    f.write(str(os.getpid()))
    
    
def get_params(file_path):
    calc_error_estimates = True
    with open(file_path,'r') as file:
        for line in file:
            if line.startswith('set_error_calc'):
                value_str = line.strip().split('=')[1]
                calc_error_estimates = value_str.lower() == 'true'
                break
    return calc_error_estimates

with open("prog_bar.log", "w") as log_file:
    try:
        calc_error_estimates = get_params('tmp/general.txt')
        if calc_error_estimates:
            file_names = [
                ('tmp/calc_rhs.txt','calculation of the rhs:      '),
                ('tmp/calc_matrix.txt', 'calculation of the matrix:   ') ,
                ('tmp/calc_l2.txt', 'calculation of the L2 error: '),
                ('tmp/calc_h1.txt', 'calculation of the h1 error: ')
                  ]
        else:
            file_names = [
                ('tmp/calc_rhs.txt','calculation of the rhs:    '),
                ('tmp/calc_matrix.txt', 'calculation of the matrix: ')    
                  ]
        log_file.write(str(calc_error_estimates))
        curses.wrapper(monotoring.main,file_names)
    except Exception as e:
        log_file.write(f"Error: {e}\n")
        print(f"Error: {e}")
