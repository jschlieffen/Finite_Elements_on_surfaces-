#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:43:19 2025

@author: jschlieffen
"""


import re
import curses
import plotext as plt
import io
from contextlib import redirect_stdout
import Resource_usage as Reu
import time

# =============================================================================
# Translate the plotext output, which is written in ANSI to something ncurses 
# understands (ACS, colour pairs, etc.)
# =============================================================================
def strip_and_translate_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[m]')
    color_codes = []
    result_text = []
    last_pos = 0
    
    def replace_with_curses(match):
        code_str = match.group(0)
        color_code = None
        nonlocal last_pos
        if ';' not in code_str:  
            try:
                color_code = int(code_str[2][:-1]) 
            except ValueError:
                return ''  
        elif '38;5;' in code_str:
            try:
                color_code = int(code_str.split(';')[2][:-1]) 
            except ValueError:
                return '' 
            
        if color_code is not None:
            text_segment = text[last_pos:match.start()]
            clean_segment = ansi_escape.sub('', text_segment)
            if text_segment:
                result_text.append(('text', clean_segment))  
            result_text.append(('color', color_code))  
            last_pos = match.end()  
            return '' 

        return ''
    
    processed_txt = ansi_escape.sub(replace_with_curses, text)

    if last_pos < len(text):
        text_segment = text[last_pos:]
        clean_segment = ansi_escape.sub('', text_segment) 
        if clean_segment:
            result_text.append(('text', clean_segment))


    return result_text



# =============================================================================
# This function draws the graph in the terminal. For this it uses plotext.
# Since plotext and ncurses does not work well together, the output 
# of plotext is redirected into a buffer. The output is then
# line by line translated and given to ncurses for the displaying
# =============================================================================
def draw_Graph_mem_usage(stdscr, mem_usage):
    curses.start_color()

    plt.clf()
    while True:
        if len(mem_usage) > 20:
            mem_usage.pop(0)
            #tmpstmp.pop(0)
        else:
            break
        #lbl = calc_outout_str(key)
    #plt.plot(tmpstmp,mem_usage)
    plt.plot(mem_usage)   
    plt.theme("dark")
    plt.title("Memory usage in MB")
    plt.plot_size(20,15)
    buf = io.StringIO()
    with redirect_stdout(buf):
        plt.show()
    plot_str = buf.getvalue()
    max_y, max_x = stdscr.getmaxyx()  
    lines = plot_str.splitlines()
    for i, line in enumerate(lines):
        if i < max_y - 1:  
            result_arr = strip_and_translate_ansi_escape_sequences(line)         
            color_pair = 0
            char = ''
            y, x = 0, 0
            was_prev_color = False
            count = 0
            for j,tup in enumerate(result_arr):
                if tup[0] == 'color':
                    color_pair = tup[1]
                    was_prev_color = True
                elif tup[0] == 'text':
                    char = tup[1]
                    stdscr.addstr(i+15,x,char, curses.color_pair(int(color_pair)))
                    was_prev_color = False
                    count += 1
                    
                    x += len(char)
            x = 0
            
def draw_Graph_cpu_usage(stdscr, cpu_usage):
    curses.start_color()

    plt.clf()
    while True:
        if len(cpu_usage) > 20:
            cpu_usage.pop(0)
            #tmpstmp.pop(0)
        else:
            break
        #lbl = calc_outout_str(key)
    #plt.plot(tmpstmp,cpu_usage)
    plt.plot(cpu_usage)   
    plt.theme("dark")
    plt.title("CPU usage in %")
    plt.plot_size(20,15)
    buf = io.StringIO()
    with redirect_stdout(buf):
        plt.show()
    plot_str = buf.getvalue()
    max_y, max_x = stdscr.getmaxyx()  
    lines = plot_str.splitlines()
    for i, line in enumerate(lines):
        if i < max_y - 1:  
            result_arr = strip_and_translate_ansi_escape_sequences(line)         
            color_pair = 0
            char = ''
            y, x = 0, 0
            was_prev_color = False
            count = 0
            for j,tup in enumerate(result_arr):
                if tup[0] == 'color':
                    color_pair = tup[1]
                    was_prev_color = True
                elif tup[0] == 'text':
                    char = tup[1]
                    stdscr.addstr(i+15,x+45,char, curses.color_pair(int(color_pair)))
                    was_prev_color = False
                    count += 1
                    
                    x += len(char)
            x = 0
            
    
def init_color_pairs_v2():
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i, i, -1)
            
def main(stdscr):
    stdscr.clear()
    init_color_pairs_v2()
    pid = Reu.find_python_process('execution_V2.py')
    mem_usage, cpu_usage, tmpstmp = [],[],[]
    start_time = time.time()
    cpu_count = Reu.get_cpu_num()
    while True:
        stdscr.addstr(0,0,'Monotoring of the usage of the resources',curses.color_pair(6))
        time_stamp = time.time()
        if (time_stamp - start_time) >= 1:
            memory_usage_float, cpu_usage_float = Reu.get_process_usage(pid)
            mem_usage.append(memory_usage_float / 1024 / 1024)
            cpu_usage.append(cpu_usage_float/cpu_count)
            #tmpstmp.append(datetime.fromtimestamp(time_stamp).strftime('%d/%m/%Y'))
            start_time = time_stamp
            
        draw_Graph_mem_usage(stdscr, mem_usage)
        draw_Graph_cpu_usage(stdscr, cpu_usage)
        stdscr.refresh()
        

if __name__ == '__main__':
    curses.wrapper(main)         