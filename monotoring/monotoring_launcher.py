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
    
with open("prog_bar.log", "w") as log_file:
    try:
        file_names = [
            ('tmp/calc_rhs.txt','calculation of the rhs:    '),
            ('tmp/calc_matrix.txt', 'calculation of the matrix: ')    
              ]
        curses.wrapper(monotoring.main,file_names)
    except Exception as e:
        log_file.write(f"Error: {e}\n")
        print(f"Error: {e}")
