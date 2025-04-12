#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:42:13 2025

@author: jschlieffen
"""

import os
import psutil

def get_cpu_num():
    return os.cpu_count()

def get_process_usage(pid):
    try:
        process = psutil.Process(pid)
        

        memory_info = process.memory_info()
        memory_usage = memory_info.rss  
        
        cpu_usage = process.cpu_percent(interval=1)  
        
        return memory_usage, cpu_usage
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} does not exist.")
        return None, None

def find_python_process(keyword):
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if keyword in cmdline:
                    print(f"Found Python process: {proc.info}")
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None
