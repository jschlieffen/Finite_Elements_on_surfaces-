#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:19:17 2024

@author: jschlieffen
"""

import numpy as np
import Triangulation as Tr
import Visz

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


    def get_triangles(self):
        index = ''
        for t1_id,t1 in self.surface.vert_dict.items():
            for t2_id,t2 in t1.get_neighbors():
                if t2_id != t1_id:
                    common_neigbours = self.surface.check_common_neighbours(t1_id, t2_id)
                    #print(common_neigbours)
                    for t3_id, t3 in common_neigbours:
                        if t1_id != t3_id and t2_id != t3_id:
                            #index = [str(idx) + '_' for idx in sorted([t1_id,t2_id,t3_id])][0]
                            index = '_'.join(str(idx) for idx in sorted([t1_id,t2_id,t3_id]))
                
                            #print(index)
                            if index not in self.triangles.keys():
                                #print('test')
                                self.triangles[index] = Triangle(t1.get_coordinates(),
                                                                 t2.get_coordinates(), 
                                                                 t3.get_coordinates())
                    
                    
    def calc_h(self):
        for t_id,triangle in self.triangles.items():
            if triangle.diameter > self.h:
                self.h = triangle.diameter
        print(self.h)
                
    def calc_n(self):
        self.n = len(self.surface.vert_dict)
    
    def calc_A(self):
        i = 0
        #print(self.A)
        test_it_i = 0
        for v_id,v in self.surface.vert_dict.items():
            v_i = v.get_coordinates()
            j = 0
           # print(v_i)
            for w_id,w in self.surface.vert_dict.items():
                
                test_it_j = 0
                v_j = w.get_coordinates()
               # print(v_j)
                for triangle_index, triangle in self.triangles.items():
                    #print('for i,j = ',i,j)
                    #print('\n')
                    #print(np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                    #print('\n')
                    #print((triangle.Grad_chi_v(v_i)))
                    #print(triangle)
                    #print('A: ', triangle.chi_v(v_i, triangle.v1))
                    #print('B: ', triangle.chi_v(v_i, triangle.v2))
                    #print('B: ', triangle.chi_v(v_i, triangle.v3))
                    if triangle.chi_v(v_i,triangle.v1) or triangle.chi_v(v_i,triangle.v2) or triangle.chi_v(v_i,triangle.v3):
                        if triangle.chi_v(v_j,triangle.v1) or triangle.chi_v(v_j,triangle.v2) or triangle.chi_v(v_j,triangle.v3):
                            self.A[i][j] += (np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                    #if v.check_if_adjacent(w_id):
                        #print('test')
                        #self.A[i][j] += (np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                    #print('i,j = ',i,j)
                    #print('res: ', np.dot(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j)) * triangle.area)
                    #print('i,j = ', i,j)
                    #print(triangle_index)
                    #print(triangle.Grad_chi_v(v_i), triangle.Grad_chi_v(v_j))
                    #print('\n')
                j += 1
            i += 1
            print(i)

    #edit before FEM
    def f(self, A):
        x,y,z = A
        normal_x,normal_y,normal_z = self.surface.normal_vector(x,y,z)
        #level_set = self.level_set(x,y,z)
        #level_set_x,level_set_y,level_set_z = level_set
        #print(normal_x, normal_y, normal_z)
        mean_curvature = self.surface.mean_curvature(x, y, z)
        #print(mean_curvature)
        #print('mean curvature: ',mean_curvature)
        #print('normal x: ', normal_x)
        #print('normal y: ', normal_y)
        #print('normal z: ', normal_z)
        return 2*normal_x*normal_y + mean_curvature*(y*normal_x + x*normal_y)
    
    

    
    def calc_rhs(self):
        i = 0
        #print('calc_rhs')
        for v_index, v in self.surface.vert_dict.items():
            res = 0
            for triangle_index, triangle in self.triangles.items():
                A, B ,C = triangle.v1,triangle.v2, triangle.v3
                v_i = v.get_coordinates()
                #print(v_i)
                #print(((self.f(A)*triangle.chi_v(v_i,A))/6))
                #print((self.f(B)*triangle.chi_v(v_i,B))/6)
                #print((self.f(C)*triangle.chi_v(v_i,C))/6)
                #print('\n')
                res_prev = res
                sq_det_G = (triangle.det_G)**(1/2)
                if triangle.chi_v(v_i,A):
                    res += sq_det_G*(((self.f(A)*triangle.chi_v(v_i,A))/6))
                elif triangle.chi_v(v_i,B):
                    res += sq_det_G*((self.f(B)*triangle.chi_v(v_i,B))/6)
                elif triangle.chi_v(v_i,C):
                    res += sq_det_G*((self.f(C)*triangle.chi_v(v_i,C))/6) 
                #res += sq_det_G*(((self.f(A)*triangle.chi_v(v_i,A))/6) + ((self.f(B)*triangle.chi_v(v_i,B))/6) + ((self.f(C)*triangle.chi_v(v_i,C))/6)) 
                #print(res)
                '''
                if i == 14:
                    if res != res_prev:

                        print('änderung')
                        print(res)
                        print(triangle.chi_v(v_i,A))
                        print(sq_det_G)
                        x,y,z = A
                        der_x, der_y, der_z = self.surface.gradient_level_set_fct(x, y, z)
                        der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z = self.surface.Hessian_lvl_set_fct(x, y, z)
                        print('der x :', der_x)
                        print('der y :', der_y)
                        print('der z: ', der_z)
                        print('der x x :', der_x_x)
                        print('der x z :', der_x_z)
                        print('der y y :', der_y_y)
                        print('der z x :', der_z_x)
                        print('der z z :' , der_z_z)
                        print(A)
                        print('normed gradient: ', np.linalg.norm(self.surface.gradient_level_set_fct(x,y,z)))
                        print(self.surface.normal_vector(x,y,z))
                        print('mean curvature: ',self.surface.mean_curvature(x,y,z))
                '''
                #print('v_i: ', v_i)
                #print('A: ', A)
                #print('f: ', self.f(A))
                #print('\n')
            self.rhs[i] = res
            if self.rhs[i] > 5:
                print(self.rhs[i])
                print(i)
            i += 1
            print(i)
            
    def solve_sytem(self,A,F):
        #print(A)
        print('test')
        #print(A == np.zeros((self.n,self.n)))
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
        self.calc_A()
        
    def ana_sol(self,x,y,z):
        return x*y
        

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
        else:
            return 0
    
    def Grad_chi_v(self,v):
        first_sum = np.dot(self.v3 - self.v1, self.v3 - self.v1)*(self.chi_v(v, self.v2) - self.chi_v(v, self.v1))*(self.v2 - self.v1)
        second_sum = np.dot(self.v2 - self.v1, self.v3 - self.v1)*(self.chi_v(v,self.v3) - self.chi_v(v, self.v1))*(self.v2 - self.v1) 
        third_sum = np.dot(self.v2 - self.v1, self.v3 - self.v1)*(self.chi_v(v,self.v2) - self.chi_v(v, self.v1))*(self.v3 - self.v1) 
        fourth_sum = np.dot(self.v2 - self.v1, self.v2 - self.v1)*(self.chi_v(v, self.v3) - self.chi_v(v, self.v1))*(self.v3 - self.v1)
        #print('first:  ',first_sum)
        #print('second: ',second_sum)
        #print('third:  ',third_sum)
        #print('fourth: ',fourth_sum)
        #print('res:    ',(first_sum - second_sum - third_sum + fourth_sum))
        return (1/self.det_G)* (first_sum - second_sum - third_sum + fourth_sum)
    

class error_calc:
    
    def __init__(self):
        self.surface = Tr.Surface(0)
        
    

    
    def grad_ana_sol(self,x,y,z):
        return None
    
    def l2_error(self,u,u_h):
        return None
    
    def h1_error(self,u,u_h):
        return None
    
def main():
    FEM_cls = FEM()
    
    print('first refinement')
    FEM_cls.only_surface_refinement()
    print('second refinement')
    FEM_cls.only_surface_refinement()
    print('third refinement')
    FEM_cls.only_surface_refinement()
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
    print(FEM_cls.rhs)
    print('numerical solution: \n')
    print('is symmetric: ', np.allclose(FEM_cls.A, FEM_cls.A.T))
    print('detereminant: ', np.linalg.det(FEM_cls.A))
    print(FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs))
    ana_sol = np.zeros((FEM_cls.n,1))
    i = 0
    for v_id,v in FEM_cls.surface.vert_dict.items():
        x,y,z = v.get_coordinates()
        ana_sol[i] = x*y
        i += 1
    print('\n ana sol \n')
    print(ana_sol)
    Visz.Plot_Discrete_surface(FEM_cls.surface.vert_dict, 'discrete_FEM_surface.html', FEM_cls.solve_sytem(FEM_cls.A, FEM_cls.rhs),'discrete_FEM_function_surface_refinement.html' )

if __name__ == '__main__':
    main()