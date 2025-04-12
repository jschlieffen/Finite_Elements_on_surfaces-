#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:19:17 2024

@author: jschlieffen
"""

import numpy as np
import math
import Triangulation as Tr
import Visz
import concurrent.futures
import threading
import time
from multiprocessing import Pool, Manager, Array, Lock

# =============================================================================
# This class computes the FEM method on surfaces. It will use the triangulation
# computed in the file Triangulation.py and then proceed to calculate the matrix
# A. Currently it iterates through all triangles, but this will be changed due 
# to the high amount of calculation time. After computing the matrix, it will 
# compute the rhs. The formulas for the normal vector and the mean curvature
# of the surface are calculated by hand and can be found in the file Triangulation.py
# To solve the linear system we use the numpy libary.
# =============================================================================
class FEM:
    
    def __init__(self):
        self.surface = Tr.Surface(0)
        self.triangles = {}
        self.get_triangles()
        self.h = -1
        self.calc_h()
        self.n = 0 
        self.calc_n()
        self.rhs = np.zeros((self.n,1))
        self.calc_rhs()
        self.A = np.zeros((self.n, self.n))
        self.calc_A()
        self.A_w_threads = np.zeros((self.n, self.n))
        #self.locks = np.array([threading.Lock() for _ in range(self.n)])
        self.calc_A_with_threads()
        self.test_time_wo_threads = 0
        self.test_time_w_threads = 0


    def get_triangles(self):
        index = ''
        for t1_id,t1 in self.surface.vert_dict.items():
            for t2_id,t2 in t1.get_neighbors():
                if t2_id != t1_id:
                    common_neigbours = self.surface.check_common_neighbours(t1_id, t2_id)
                    for t3_id, t3 in common_neigbours:
                        if t1_id != t3_id and t2_id != t3_id:
                            index = '_'.join(str(idx) for idx in sorted([t1_id,t2_id,t3_id]))
                            if index not in self.triangles.keys():
                                self.triangles[index] = Triangle(t1.get_coordinates(),
                                                                 t2.get_coordinates(), 
                                                                 t3.get_coordinates())
                    
                    
    def calc_h(self):
        for t_id,triangle in self.triangles.items():
            if triangle.diameter > self.h:
                self.h = triangle.diameter
                
    def calc_n(self):
        self.n = len(self.surface.vert_dict)
    
    def calc_A(self):
        i = 0
        max_vert = self.surface.num_vertices

        for v_id,v in self.surface.vert_dict.items():
            v_i = v.get_coordinates()
            j = 0
            #print(i)
            if i % 10 == 0 and max_vert > 100:
                with open('tmp/calc_matrix.txt','a') as file:
                    file.write(f'matrixcalc: {i}, max = {max_vert - 1} \n')
                    file.flush()
            for w_id,w in self.surface.vert_dict.items():
                if v_id == w_id or v.check_if_adjacent(w_id):
                    v_j = w.get_coordinates()
                    #count = 0
                    for triangle_index, triangle in self.triangles.items():
                        if triangle.chi_v(v_i,triangle.v1) or triangle.chi_v(v_i,triangle.v2) or triangle.chi_v(v_i,triangle.v3):
                            if triangle.chi_v(v_j,triangle.v1) or triangle.chi_v(v_j,triangle.v2) or triangle.chi_v(v_j,triangle.v3):
                                #count += 1
                                #print(count)
                                self.A[i][j] += (np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                j += 1
            i += 1

        with open('tmp/calc_matrix.txt','w') as file:
            file.write(f'matrixcalc: 1, max = 1 \n')
            file.flush()
            
            
    def calculate_matrix_entries(self,i,v_id,v):
        #print('test')
        try:
            with self.locks[i]:
                v_i = v.get_coordinates()
                j = 0
                #print(i)
                for w_id,w in self.surface.vert_dict.items():
                    if v_id == w_id or v.check_if_adjacent(w_id):
                        
                            v_j = w.get_coordinates()
                            for triangle_index, triangle in self.triangles.items():
                                if triangle.chi_v(v_i,triangle.v1) or triangle.chi_v(v_i,triangle.v2) or triangle.chi_v(v_i,triangle.v3):
                                    if triangle.chi_v(v_j,triangle.v1) or triangle.chi_v(v_j,triangle.v2) or triangle.chi_v(v_j,triangle.v3):
            
                                        self.A_w_threads[i][j] += (np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                    j += 1
        except Exception as e:
            print(f"Error in thread for vertex {v_id}: {e}")
    
            
    def calc_A_with_threads_v2(self):
        
        i = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for v_id,v in self.surface.vert_dict.items():
                futures.append(executor.submit(self.calculate_matrix_entries,i,v_id,v))
                i += 1
                #print(i)
    
            #wait until all threads are finished        
            for future in futures:
                future.result() 
        print('calculation succeeded')
        
    
    def calculate_matrix_entries_parallel(self,i, v_id, v,n , A_w_threads):
        
        #i, v_id, v, n, triangles, surface_vertices, A_w_threads, lock = args
        print(i)
        v_i = v.get_coordinates()
        # Your logic for matrix calculation goes here, using the shared triangles and surface
        for j, (w_id, w) in enumerate(self.surface.vert_dict.items()):
            #print(j, 'test')
            #print(w_id)
            #print(w)
            if v_id == w_id or v.check_if_adjacent(w_id):
                
                v_j = w.get_coordinates()
                for triangle_index, triangle in self.triangles.items():
                    
                    if triangle.chi_v(v_i, triangle.v1) or triangle.chi_v(v_i, triangle.v2) or triangle.chi_v(v_i, triangle.v3):
                        #print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
                        if triangle.chi_v(v_j, triangle.v1) or triangle.chi_v(v_j, triangle.v2) or triangle.chi_v(v_j, triangle.v3):
                            # Update shared data
                            #print(i)
                            
                            #print(index)
                            index = i * n + j
                            #A_w_threads[index] += (np.dot(triangle.Grad_chi_v(v), triangle.Grad_chi_v(v_j)) * triangle.area)
                            #print(A_w_threads[i][j])
                            #if index >= 0 and index < n * n:
                                #A_w_threads[index] += (np.dot(triangle.Grad_chi_v(v), triangle.Grad_chi_v(v_j)) * triangle.area)
                            #else:
                                #print(f"Index {index} is out of bounds for A_w_threads.")
                            #+if 0 <= index < len(A_w_threads):
                                # Use the lock to ensure thread-safe updates to shared memory
                            val = (np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                            #with lock:
                            A_w_threads[index] += val
                                #print('test')
                            #else:
                                #print(f"Warning: Index {index} out of bounds!")
                                #return  # Exit early to avoid any further issues

    #TODO:optimize source code, such that this will be faster than the sequentiell implementation
    # make that each thread is created exaclty once, remove lock if possible and create and give each thread a min. number of matrix rows 
    # maybe I need pathos.multithreading or joblib, when I try to execute the threads
    def calc_A_with_threads(self):
        with Manager() as manager:
            A_w_threads = manager.list([0]*(self.n*self.n))
            A_w_threads = manager.list(A_w_threads)  
            chunks = [
                (i, v_id, v, self.n, A_w_threads) 
                for i, (v_id, v) in enumerate(self.surface.vert_dict.items())
            ]
            with Pool(processes=6) as pool:
                pool.starmap(self.calculate_matrix_entries_parallel,chunks)
            A_w_threads_np = np.array(list(A_w_threads)).reshape((self.n, self.n))
            
            
        self.A_w_threads = A_w_threads_np
        print('calculation succeeded')

        
    #edit before FEM
    def f(self, A):
        x,y,z = A
        normal_x,normal_y,normal_z = self.surface.normal_vector(x,y,z)
        mean_curvature = self.surface.mean_curvature(x, y, z)
        return 2*normal_x*normal_y + mean_curvature*(y*normal_x + x*normal_y)
    
    

    #TODO: improve
    def calc_rhs(self):
        i = 0
        max_vert = self.surface.num_vertices
        for v_index, v in self.surface.vert_dict.items():
            if i % 10 == 0 and max_vert > 100:
                with open('tmp/calc_rhs.txt', 'a') as file:
                    file.write(f'rhscalc: {i}, max = {max_vert -1} \n')
                    file.flush()
            res = 0
            #print('rhs', i)
            for triangle_index, triangle in self.triangles.items():
                A, B ,C = triangle.v1,triangle.v2, triangle.v3
                v_i = v.get_coordinates()
                res_prev = res
                sq_det_G = (triangle.det_G)**(1/2)
                if triangle.chi_v(v_i,A):
                    res += sq_det_G*(((self.f(A)*triangle.chi_v(v_i,A))/6))
                elif triangle.chi_v(v_i,B):
                    res += sq_det_G*((self.f(B)*triangle.chi_v(v_i,B))/6)
                elif triangle.chi_v(v_i,C):
                    res += sq_det_G*((self.f(C)*triangle.chi_v(v_i,C))/6) 
            self.rhs[i] = res
            i += 1
        with open('tmp/calc_rhs.txt', 'w') as file:
            file.write(f'rhscalc: 1, max = 1 \n')
            file.flush()
            
    def solve_sytem(self,A,F):
        return np.linalg.solve(A,F)
    
    def only_surface_refinement(self):
        self.surface.refine()
        
    def surface_refinement(self):
        self.surface.refine()
        self.triangles = {}
        self.get_triangles()
        self.h = -1
        self.calc_h()
        self.n = 0 
        self.calc_n()
        self.rhs = np.zeros((self.n,1))
        self.calc_rhs()
        self.A = np.zeros((self.n, self.n))
        start_time_wo_threads = time.time()
        self.calc_A()
        end_time_wo_threads = time.time()
        self.A_w_threads = np.zeros((self.n, self.n))
        #self.locks = np.array([threading.Lock() for _ in range(self.n)])
        #self.calc_A_with_threads()
        end_time_w_threads = time.time()
        self.test_time_wo_threads = end_time_wo_threads - start_time_wo_threads
        self.test_time_w_threads = end_time_w_threads - end_time_wo_threads
        
    def ana_sol(self,x,y,z):
        return x*y
        
# =============================================================================
# This class is used to describe a triangle. Its attributes consists of important
# functions for the FEM method such as the Gradient of a given hat-function 
# over the triangle
# =============================================================================
class Triangle:
    
    def __init__(self,v1,v2,v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.diameter = 0
        self.det_G = 0
        self.area = 0
        self.calc_area()
        self.calc_diameter()
        self.calc_det_G()
        
    def __str__(self):
        return 'vertices :' + str(self.v1) + ', ' + str(self.v2) + ', ' + str(self.v3) + ' and diam = ' + str(self.diameter)
        
    def calc_diameter(self):
        self.diameter = max([np.linalg.norm(self.v1 - self.v2),
                             np.linalg.norm(self.v1 - self.v3),
                             np.linalg.norm(self.v2 - self.v3)])
        
    def calc_det_G(self):
        first_sum = np.dot(self.v2 - self.v1, self.v2 - self.v1) * np.dot(self.v3 - self.v1, self.v3 - self.v1)
        second_sum = (np.dot((self.v2 - self.v1),(self.v3 - self.v1)))**2
        self.det_G = first_sum - second_sum
        
        
    def calc_area(self):
        self.area = np.linalg.norm(np.cross(self.v2 - self.v1, self.v3 - self.v1)) / 2 

    def chi_v(self, v, x):
        if np.array_equal(v,x):
            return 1
        return 0
    
    def Grad_chi_v(self,v):
        first_sum = np.dot(self.v3 - self.v1, self.v3 - self.v1)*(self.chi_v(v, self.v2) - self.chi_v(v, self.v1))*(self.v2 - self.v1)
        second_sum = np.dot(self.v2 - self.v1, self.v3 - self.v1)*(self.chi_v(v,self.v3) - self.chi_v(v, self.v1))*(self.v2 - self.v1) 
        third_sum = np.dot(self.v2 - self.v1, self.v3 - self.v1)*(self.chi_v(v,self.v2) - self.chi_v(v, self.v1))*(self.v3 - self.v1) 
        fourth_sum = np.dot(self.v2 - self.v1, self.v2 - self.v1)*(self.chi_v(v, self.v3) - self.chi_v(v, self.v1))*(self.v3 - self.v1)
        return (1/self.det_G)* (first_sum - second_sum - third_sum + fourth_sum)
    

# =============================================================================
# TODO
# =============================================================================
class error_calc:
    
    def __init__(self, surface, triangles):
        self.surface = surface
        self.triangles = triangles
    
    def refine_of_surface(self, surface, triangles):
        self.surface = surface
        self.triangles = triangles
    

    #TODO: debug
    def calc_dS(self, v):
        dS = 0
        count = 0
        for triangle_index, triangle in self.triangles.items():
            if np.array_equal(v, triangle.v1) or  np.array_equal(v, triangle.v2) or  np.array_equal(v, triangle.v3):
                dS += triangle.area
        return dS
    
    def l2_error(self,u,u_h):
        l2_error = 0
        i = 0
        for v_id, v in self.surface.vert_dict.items():
            v_i = v.get_coordinates()
            dS = self.calc_dS(v_i)
            x_i,y_i,z_i = v_i
            dist = np.linalg.norm(u(x_i,y_i,z_i) - u_h[i])
            l2_error += (dist**2)*dS  
            i += 1
        return math.sqrt(l2_error)
    
    def grad_ana_sol_proj(self,x,y,z):
        P = np.zeros((3,3))
        normal_vector = self.surface.normal_vector(x, y, z)
        for i in range(0,3):
            for j in range(0,3):
                if i == j:
                    P[i][j] = 1 - normal_vector[i]*normal_vector[j]
                else:
                    P[i][j] = - normal_vector[i]*normal_vector[j]
        grad_ana_sol = np.dot(P,np.array([y,x,0]))
        return grad_ana_sol
    
    def grad_ana_sol(self,x,y,z):
        return np.array([y,x,0])
    
    def grad_disc_sol_proj(self,u):
        grad_disc_sol = 0
        for v_id,v_i in self.surface.vert_dict.items():
            v = v_i.get_coordinates()
            x_v, y_v, z_v = v
            for traingle_id,triangle in self.triangles.items():
                if np.array_equal(v, triangle.v1) or  np.array_equal(v, triangle.v2) or  np.array_equal(v, triangle.v3):
                    grad_disc_sol += u(x_v, y_v, z_v )* triangle.Grad_chi_v(v)
        return grad_disc_sol     
    
    
    
    def grad_disc_sol_v2(self,u_h,x_node,x_id):
        i = 0
        x = x_node.get_coordinates()
        grad_disc_sol = np.zeros(3)
        for v_id,v in self.surface.vert_dict.items():
            grad_chi_sum = np.zeros(3)
            if v_id == x_id or x_node.check_if_adjacent(v_id):
                v_i = v.get_coordinates()
                count = 0
                triangle_meas = 0
                for triangle_index, triangle in self.triangles.items():
                    if np.array_equal(x, triangle.v1) or  np.array_equal(x, triangle.v2) or  np.array_equal(x, triangle.v3):
                        triangle_meas += triangle.area
                        if np.array_equal(v_i, triangle.v1) or  np.array_equal(v_i, triangle.v2) or  np.array_equal(v_i, triangle.v3):
                            
                            grad_chi_sum +=triangle.Grad_chi_v(v_i)*triangle.area
                            count += 1
                grad_chi_sum /= triangle_meas
                grad_disc_sol += u_h[i]*grad_chi_sum
            i += 1
        return grad_disc_sol
    
    #TODO: check if calculation is correct
    def grad_disc_sol_v3(self,u_h,x):
        grad_disc_sol = np.zeros(3)
        i = 0
        for triangle_index, triangle in self.triangles.items():
            if np.array_equal(x, triangle.v1) or  np.array_equal(x, triangle.v2) or  np.array_equal(x, triangle.v3):
                x1,y1,z1 = triangle.v1
                x2,y2,z2 = triangle.v2
                x3,y3,z3 = triangle.v3
                area = triangle.area
                delta_x = (y2 - y3)/(2*area)
                delta_y = (z2 - z3)/(2*area)
                delta_z = (x2 - x3)/(2*area)
                grad_disc_sol += u_h[i]*np.array([delta_x,delta_y,delta_z])
                i+=1
        return grad_disc_sol
        
    def project_grad2surf(self,grad_disc_sol,grad_ana_sol,x,y,z):
        P = np.zeros((3,3))
        normal_vector = self.surface.normal_vector(x, y, z)
        for i in range(0,3):
            for j in range(0,3):
                if i == j:
                    P[i][j] = 1 - normal_vector[i]*normal_vector[j]
                else:
                    P[i][j] = - normal_vector[i]*normal_vector[j]
        grad_sum = grad_ana_sol - grad_disc_sol
        grad_proj = np.dot(P,grad_sum)            
        return grad_proj
        
    #TODO: check calculation of dS, it is zero at some points, what should not be the case
    def h1_error(self,u_h,l2_error):
        h1_semi_error = 0
        i = 0
        for v_id, v in self.surface.vert_dict.items():
            v_i = v.get_coordinates()
            dS = self.calc_dS(v_i)
            x_i,y_i,z_i = v_i
            dist = np.linalg.norm(self.grad_ana_sol_proj(x_i,y_i,z_i) - self.grad_disc_sol_v2(u_h,v,v_id))
            h1_semi_error += (dist**2)*dS  
            i += 1
        h1_semi_error = math.sqrt(h1_semi_error)
        return h1_semi_error
        
    def calc_OOC(self,error_prev,error_now,diam_prev,diam_now):
        return (math.log2(error_prev/error_now))/(math.log2(diam_prev/diam_now))
    
    
    
# =============================================================================
# Function to execute the main functions of the file and print it.
# Used for debugging purposes.
# =============================================================================
def main():
    FEM_cls = FEM()
    
    print('first refinement')
    FEM_cls.only_surface_refinement()
    print('second refinement')
    FEM_cls.only_surface_refinement()
    print('third refinement')
    #FEM_cls.only_surface_refinement()
    print('fourth refinement')
    FEM_cls.surface_refinement()
    '''
    for triangle_index, triangle in FEM_cls.triangles.items():
        print(triangle)
        print('det:  ', triangle.det_G)
        print('area: ', triangle.area, '\n')
        A = triangle.v1
        print('grad: ', triangle.Grad_chi_v(A))
        print('\n')
    
    '''
    print(FEM_cls.h)
    print('\n coeff matrix: \n')
    print(FEM_cls.A)
    print('\n rhs: \n')
    #print(FEM_cls.rhs)
    print('numerical solution: \n')
    #print('is symmetric: ', np.allclose(FEM_cls.A, FEM_cls.A.T))
    print('detereminant: ', np.linalg.det(FEM_cls.A))
    #print(FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    ana_sol = np.zeros((FEM_cls.n,1))
    i = 0
    for v_id,v in FEM_cls.surface.vert_dict.items():
        x,y,z = v.get_coordinates()
        ana_sol[i] = x*y
        i += 1
    print('\n ana sol \n')
    #print(ana_sol)
    print('\n')
    Error_estimates = error_calc(FEM_cls.surface, FEM_cls.triangles)
    l2_error = Error_estimates.l2_error(FEM_cls.ana_sol,FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    h1_error= Error_estimates.h1_error(FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs), l2_error)
    diam = FEM_cls.h
    #h1_error= Error_estimates.h1_error(FEM_cls.ana_sol, l2_error)
    #print(l2_error)
    print(h1_error)

    print('test')
    FEM_cls.surface_refinement()
    print('calc error estimates')
    Error_estimates.refine_of_surface(FEM_cls.surface, FEM_cls.triangles)
    l2_error_new = Error_estimates.l2_error(FEM_cls.ana_sol,FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    h1_error_new = Error_estimates.h1_error(FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs), l2_error_new)
    diam_new = FEM_cls.h
    OOC_l2 = Error_estimates.calc_OOC(l2_error, l2_error_new, diam, diam_new)
    OOC_h1 = Error_estimates.calc_OOC(h1_error, h1_error_new, diam, diam_new)
    print(OOC_l2)
    print(OOC_h1)
    print(h1_error)
    print(h1_error_new)
    #print(FEM_cls.surface.num_vertices)
    #Visz.Plot_Discrete_surface(FEM_cls.surface.vert_dict, 'discrete_FEM_surface.html', FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs),'discrete_FEM_function_surface_refinement.html' )
    print(Error_estimates.surface.num_vertices)
    
def main_V2():
    FEM_cls = FEM()
    equals_arr = []
    time_arr = []
    for i in range(0,4):
        FEM_cls.surface_refinement()
        #time_start_wo_threads = time.time()
        A_wo_threads = FEM_cls.A
        #time_end_wo_threads = time.time()
        A_w_threads = FEM_cls.A_w_threads
        #time_end_w_threads = time.time()
        is_equal = np.array_equal(A_wo_threads, A_w_threads)
        #print(A_w_threads)
        print('is equal: ', is_equal )
        #time_wo_threads = time_end_wo_threads - time_start_wo_threads
        #time_w_threads = time_end_w_threads - time_end_wo_threads
        time_wo_threads = FEM_cls.test_time_wo_threads
        time_w_threads = FEM_cls.test_time_w_threads
        equals_arr.append(is_equal)
        time_arr.append((time_wo_threads, time_w_threads))
    
    print('\n')
    
    for i in range(0,4):
        print('First refinement')
        print('matrices are equal: ', equals_arr[0])
        print('Time without threads: ', time_arr[i][0])
        print('Time with threads: ', time_arr[i][1])

if __name__ == '__main__':
    main()
    #main_V2()