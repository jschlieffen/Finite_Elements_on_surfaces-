#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 20:00:10 2025

@author: jschlieffen
"""

import os
import monotoring_graphs
import curses

with open("tmp/pids_graph.txt", "w") as f:
    f.write(str(os.getpid()))
    

with open("graph.log", "w") as log_file:
    try:
        curses.wrapper(monotoring_graphs.main)
    except Exception as e:
        log_file.write(f"Error: {e}\n")
        print(f"Error: {e}")
