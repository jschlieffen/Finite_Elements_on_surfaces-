#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:38:11 2025

@author: jschlieffen
"""

import psutil
import os

def get_process_usage(pid):
    try:
        # Get process by PID
        process = psutil.Process(pid)
        
        # Get memory usage in bytes
        memory_info = process.memory_info()
        memory_usage = memory_info.rss  # Resident Set Size (physical memory usage)
        
        # Get CPU usage as a percentage
        cpu_usage = process.cpu_percent(interval=1)  # You can specify the interval (in seconds) for CPU usage
        
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

# Example usage:
pid = find_python_process('exec.py')  # You can directly set the PID if you know it
print('test')
if pid:
    memory, cpu = get_process_usage(pid)
    if memory is not None and cpu is not None:
        print(f"Process {pid} - Memory Usage: {memory / 1024 / 1024:.2f} MB, CPU Usage: {cpu}%")
