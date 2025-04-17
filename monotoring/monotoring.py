#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:53:15 2025

@author: jschlieffen
"""
import subprocess
import re
import curses
import select
import time



def process_line(line):
    match = re.search(r"\s*(\d+),\s*max\s*=\s*(\d+)", line)
    if match:
        matrix_val = int(match.group(1))
        max_val = int(match.group(2))
        result = matrix_val / max_val if max_val != 0 else 0
        return result
    else:
        return 0


def tail_file(file_path,interval):
    proc = subprocess.Popen(
        ['tail', '-n', '0', '-F', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=0
    )
    while True:
        rlist, _, _ = select.select([proc.stdout], [], [],interval)
        if rlist:
            line = proc.stdout.readline()
            if not line:
                continue
            yield line
        else:
            yield None

def file_follower(file_path):
    with open(file_path, 'r') as f:
        f.seek(0, 2)  # Go to end of file
        while True:
            line = f.readline()
            if line:
                yield line
            #else:
                #time.sleep(0.01)


def draw_progress_bar(stdscr, data_dict):
    bar_width = 50
    i = 3
    for key,val in data_dict.items():
        
        stdscr.addstr(i,0,val['typ'], curses.color_pair(5))
        line = val['last_line']
        progress = process_line(line)
        #stdscr.addstr(i+10,0,line)
        green = u'\u2500' * int(progress * bar_width)
        red = u'\u2500' * (bar_width - len(green))
        stdscr.addstr(green, curses.color_pair(2))
        stdscr.addstr(red, curses.color_pair(1))
        i += 1
    
def init_color_pairs_v2():
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i, i, -1)

def main(stdscr,file_names):
    stdscr.clear()
    
    init_color_pairs_v2()
    data_dict = {}
    for file_name in file_names:
        data_dict[file_name[0]] = {
            'typ' : file_name[1],
            'tail' : tail_file(file_name[0],0.01),
            #'tail' : file_follower(file_name[0]),
            'last_line' : '', 
            'current_calc' : False
        }
    other_calc = False
    ref_num_tail = tail_file('../tmp/general.txt',1)
    refinement_number = 0
    while True:
        #line_ref_num = next(ref_num_tail)
        #if line_ref_num:
            #refinement_number = int(line_ref_num.split('=')[1])
        stdscr.addstr(0,0,f'Monotoring of the current progress, Refinement',curses.color_pair(6))

        for key,val in data_dict.items():
            #try:
                line = next(val['tail'])
                if line is not None:
                    data_dict[key]['last_line'] = line
                    other_calc = True
                '''
                elif line is None:
                    if other_calc:
                        #TODO: Find better sol. to that.
                        if 'rhs' in val['typ']:
                            data_dict[key]['last_line'] = '1, max = 1'
                        elif 'matrix' in val['typ']:
                            data_dict[key]['last_line'] = '1, max = 1'
                        elif 'l2' in val['typ']:
                            data_dict[key]['last_line'] = '1, max = 1'
                        elif 'h1' in val['typ']:
                            data_dict[key]['last_line'] = '0, max = 1'
                        other_calc = False
                '''
                #else:
                    #data_dict[key]['last_line'] = '1, max = 1'
                #gen_list += [tail_file(file_name)]
                #res = process_line(line)
                draw_progress_bar(stdscr, data_dict)

        stdscr.refresh()
                #print(line)
                #print(res)
                #time.sleep(0.01)
            #except StopIteration:
                #continue



if __name__ == '__main__':
    #'''
    #print(os.path.exists('tmp/calc_matrix.txt'))
    file_names = [
        ('tmp/calc_matrix.txt', 'calculation of the matrix: '),
        ('tmp/calc_rhs.txt','calculation of the rhs:    ')        
          ]
    curses.wrapper(main,file_names)
    #'''
    '''
    gen1 = tail_file('tmp/calc_matrix.txt')
    gen2 = tail_file('tmp/calc_rhs.txt')
    while True:
        val1 = next(gen1)
        val2= next(gen2)
        if val1 is not None:
            print(val1)
        if val2 is not None:
            print(val2)
    '''