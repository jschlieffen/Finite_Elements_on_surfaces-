#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:56:53 2024

@author: jschlieffen
"""

import numpy as np
import plotly.graph_objects as go
import Visz
from termcolor import colored
import math
import sympy as sp

class Node:
    def __init__(self, x,y,z):
        self.coordinates = np.array([x,y,z])
        self.adjacent = {}
                
    def __str__(self):
        return str(self.coordinates) + ' adjacent: ' + str([x.get_coordinates() for x_id,x in self.adjacent.items()]) #+ 'type: ' + str(type(self.coordinates))

    def add_neighbor(self, neighbor_id,neighbor):
        self.adjacent[neighbor_id] = neighbor
        
    def remove_neighbor(self,neighbor):
        #print('test1')
        #print('neighbor: ',neighbor)
        if neighbor in self.adjacent.keys():
            #print('test')
            self.adjacent.pop(neighbor)

    def get_connections(self):
        return self.adjacent.keys()
    
    def get_neighbors(self):
        res = []
        for w_id,w in self.adjacent.items():
            res += [(w_id,w)]
        return res

    def get_coordinates(self):
        return self.coordinates
    
    def redefine_coordinates(self,x,y,z):
        self.coordinates = np.array([x,y,z])
    
    def check_if_adjacent(self,v_id):
        for w_id,w in self.get_neighbors():
            if w_id == v_id:
                return True
        return False


class Triangulation:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
        self.current_index = 0

    def __iter__(self):
        return iter(self.vert_dict.values())
    
    def __copy__(self):
        return Triangulation(self.vert_dict,self.num_vertices)
    
    def print_Triangulation(self):
        for vertex_key, vertex in self.vert_dict.items():
            print(vertex)
            print('\n')

    
    def delete_Graph(self):
        v_ids = []
        for v_id,v in self.vert_dict.items():
            v_ids += [v_id]
        #print(v_ids)
        for v_id in v_ids:
            #print(self.vert_dict[v_id])
            self.remove_vertex(v_id)
            #print(self.vert_dict[v_id])
    
    def check_vertex_exists_V2(self,coordinates):
        for v_id, v in self.vert_dict.items():
            if np.array_equal(coordinates, v.get_coordinates()):
                return True
        return False
    
    def check_vertex_id(self,vertex_id):
        for v_id,v in self.vert_dict.items():
            if vertex_id == v_id:
                return False
        return True
    
    def add_vertex(self, coordinates,vertex_id = -1):
        if not self.check_vertex_exists_V2(coordinates):
            self.num_vertices = self.num_vertices + 1
            #print(self.current_index)
            new_vertex = Node(*coordinates)
            self.vert_dict[self.current_index] = new_vertex
            self.current_index += 1
        return self.current_index - 1

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None
        
    def get_vertex_id(self, vertex_coord):
        for v_id,v in self.vert_dict.items():
            if np.array_equal(vertex_coord, v.get_coordinates()):
                return v_id
        print('failed')
    
    def change_vertex_id(self, vertex_id,new_vertex_id):
        x,y,z = new_vertex_id
        edge_list = []
        for w_id in self.vert_dict[vertex_id].get_neighbors():
            edge_list  += [w_id]
        self.remove_vertex(vertex_id)
        self.add_vertex((x,y,z))
        for w_id in edge_list:
            self.add_edge(new_vertex_id,tuple(w_id.get_coordinates()),new_vertex_id,tuple(w_id.get_coordinates()))
        
    def get_edges(self):
        edge_list = []
        for vertex_id,vertex in self.vert_dict.items():
            for adjacent_id,adjacent in vertex.get_neighbors():
                #if not any(set(edge) == set(edge_l) for edge_l in edge_list):
                edge_list += [(vertex_id,adjacent_id)]
        return edge_list
            
    def check_vertex_exists(self,vertex_id,vertex_coordinates):
        if vertex_id not in self.vert_dict:
            x,y,z = vertex_coordinates
            vertex_coord_array = np.array([x,y,z])
            for v_id, v in self.vert_dict.items():
                if np.array_equal(vertex_coord_array, v.get_coordinates()):
                    return v_id
        return vertex_id
            
    def add_edge(self, frm, to, frm_coordinates = [],to_coordinates = [] ):   
        if frm_coordinates != []:
            self.add_vertex(frm_coordinates)
        if to_coordinates != []:
            self.add_vertex(to_coordinates)
        self.vert_dict[frm].add_neighbor(to,self.vert_dict[to])
        self.vert_dict[to].add_neighbor(frm,self.vert_dict[frm])
    
    def remove_edge(self,frm,to):
        #frm = self.check_vertex_exists(frm, frm)
        #to = self.check_vertex_exists(to, to)
        #print('frm :', frm)
        #print('to: ', to)
        self.vert_dict[frm].remove_neighbor(to)
        self.vert_dict[to].remove_neighbor(frm)
        #print(self.vert_dict[frm])
        #print(self.vert_dict[to])
        
    def remove_vertex(self,vertex_id):
        for adjacent_id,adjacent in self.vert_dict[vertex_id].get_neighbors():
            #vert_tuple = tuple(vert.get_coordinates())
            #print('adjacent: ', adjacent_id)
            self.remove_edge(vertex_id, adjacent_id)
        del self.vert_dict[vertex_id]
        self.num_vertices -= 1

    def get_vertices(self):
        return self.vert_dict.keys()
    
    def check_common_neighbours(self,v,w):
        if v != w:
            #print(v)
            #print(w)
            v_neighbours = self.vert_dict[v].get_neighbors()
            w_neighbours = self.vert_dict[w].get_neighbors()
            #print('test12')
            #print(v_neighbours)
            common_neighbours = [vertex for vertex in v_neighbours if vertex in w_neighbours]
            return common_neighbours
        return []
    
    def Project(self,projection):
        old_vertices = [(vertex_id,vertex) for vertex_id,vertex in self.vert_dict.items()]
        for vertex_id,vertex in old_vertices:
            x1,x2,x3 = projection(vertex_id)
            self.add_vertex((x1,x2,x3))
            for adjacent in vertex.get_neighbors():
                self.add_edge((x1,x2,x3), tuple(adjacent.get_coordinates()),(x1,x2,x3),tuple(adjacent.get_coordinates()))
            self.remove_vertex(vertex_id)


    
class Icosahedron(Triangulation):
    
    def __init__(self,create_surface_plot):
        super().__init__()
        #self.create_octahedron()
        self.create_square()
        if (create_surface_plot):
            self.plot_surface()
        
    def copycat(self):
        return Icosahedron()
    
    def test_ten_points(self):
        Nodes = [(0,0,1), (0,0,-1)]
        
    
    def create_square(self):
        #Nodes_unmapped = [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]
        Nodes_unmapped = [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0)]
        for Node in Nodes_unmapped:
            vertex = Node/self.norm(Node)
            self.add_vertex(Node)
        #min_dist = self.calc_min_dist()
        #self.add_vertex((0,0,1))
        #self.add_vertex((0,0,-1))
        #v1 = self.vert_dict[0].get_coordinates()
        #v2 = self.vert_dict[1].get_coordinates()
        #test_vec = ((v1-v2)/3)/self.norm((v1-v2)/3)
        #test_vec2 = ((v2-v1)/3)/self.norm((v2-v1)/3)
        #print(test_vec)
        #print(test_vec2)
        #self.add_vertex(test_vec)
        #self.add_vertex(test_vec2)
        edges = [(0, 1) ,(0,2), (0,4),(0,6),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,5)
                 ,(3,7),(4,5),(6,7)]
        for v_id,w_id in edges:
            self.add_edge(v_id, w_id)
        '''
        for v_id,v in self.vert_dict.items():
            for w_id,w in self.vert_dict.items():
                euclidean_distance = self.norm(v.get_coordinates() - w.get_coordinates())
                if euclidean_distance == math.sqrt(2) or euclidean_distance == 2:
                    self.add_edge(v_id, w_id)
        '''
        self.project_initial_coords()
    
    
    def project_initial_coords(self):
        v_ids = []
        for v_id,v in self.vert_dict.items():
            #x,y,z = v.get_coordinates()
            #x_new,y_new,z_new = self.mapp(x,y,z)
            
            #v.redefine_coordinates(x_new,y_new,z_new)
            eucl_norm = self.norm(v.get_coordinates())
            x_new, y_new, z_new = (v.get_coordinates())/eucl_norm
            v.redefine_coordinates(x_new,y_new,z_new)
    '''
    def calc_min_dist(self):
        min_dist = 0
        for v_id,v in self.vert_dict.items()
            for w_id,w in self.vert_dict
    '''
    
    def create_octahedron(self):
        Nodes = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
        for vertex in Nodes:
            self.add_vertex(vertex)
        for v_id,v in self.vert_dict.items():
            for w_id,w in self.vert_dict.items():
                check_vec = v.get_coordinates() + w.get_coordinates()
                if np.array_equal(check_vec, np.array([0,0,0]), equal_nan=False) == False:
                    self.add_edge(v_id, w_id)
    
    def geodesic_distance(self,v,w):
        euclidean_distance = np.linalg.norm(v - w)
        phi = np.arcsin((euclidean_distance/2))
        gc_dist = 2*phi
        return gc_dist
    
    def norm(self,vertex):
        return np.linalg.norm(vertex)
    
    def geodesic_midpoint(self,v,w):
        midpoint = (v+w)/2
        return midpoint/self.norm(midpoint)
        
    def get_old_common_neighbours(self,v,w,old_edges):
        old_common_neighbours = []
        if v != w:
            for (v1,v2) in old_edges:
                if v == v1:
                    if (w,v2) in old_edges:
                        if v2 != w and v != v2:
                            old_common_neighbours += [v2]
        return old_common_neighbours
    
    def refine_edges(self,v_id,w_id,v,w, new_vertex,old_vertices, old_edges):
        common_neigbours = []
        #print('old vertices: ',old_vertices)
        old_common_neighbours = self.get_old_common_neighbours(v_id, w_id, old_edges)
        #print(old_common_neighbours)
        #print(v_id)
        #print(w_id)
        for vertex in old_common_neighbours:
            #print('test123')
            common_neigbours = self.check_common_neighbours(vertex, v_id) + self.check_common_neighbours(vertex, w_id)
            #print('test1234')
            #print(common_neigbours)
            for vert_id,vert in common_neigbours:
                #vert_tuple = tuple(vert.get_coordinates())
                #if (not any(vert_tuple == old_vertex[0] for old_vertex in old_vertices)):
                if not any(vert_id == old_vertex[0] for old_vertex in old_vertices):
                    self.add_edge(new_vertex,vert_id)
                    self.add_edge(vert_id,new_vertex)
        
        self.add_edge(new_vertex, v_id)
        self.add_edge(new_vertex, w_id)   
               
    
    
    def refinement(self):
        #print('test')
        old_vertices = [(v_id,vertex) for v_id,vertex in self.vert_dict.items()]
        old_edges = self.get_edges()
        #print(old_edges)
        seen = []
        for v_id,v in old_vertices:
            #print('test2')
            for w_id,w in v.get_neighbors():
                #print('test3')
                if any(w_id == old_vertex[0] for old_vertex in old_vertices):
                    #print('test1')
                    self.remove_edge(v_id, w_id)
                    #self.remove_edge( , v_id)
                    x,y,z = self.geodesic_midpoint(v.get_coordinates(), w.get_coordinates())
                    new_vertex = self.add_vertex((x,y,z))
                    self.refine_edges(v_id, w_id ,v,w, new_vertex,old_vertices,old_edges)
    
    def level_set_fct_Unit_ball(self,x,y,z):
        return x**2 + y**2 + z**2 -1


class Surface(Triangulation):
    
    def __init__(self,plot_surface):
        super().__init__()
        if plot_surface:
            self.plot_surface()
        self.Unit_ball = Icosahedron(0)
        self.create_discrete_surface()
        self.Hessian_lvl_set = self.compute_Hessian()
        self.grad_lvl_set_fct = self.compute_grad()
        
    def create_discrete_surface(self):
        for v_id,vertex in self.Unit_ball.vert_dict.items():
            self.add_vertex(vertex.get_coordinates())
        for v_id,vertex in self.Unit_ball.vert_dict.items():
            for w_id,w in vertex.get_neighbors():
                #print('w_id: ', w_id)
                #print('w: ',w)
                self.add_edge(v_id, w_id)
        self.mapp_vertices()
    
    def refine(self):
        self.delete_Graph()
        self.Unit_ball.refinement()
        map_vertex_id = []
        for v_id,vertex in self.Unit_ball.vert_dict.items():
            vertex_id = self.add_vertex(vertex.get_coordinates())
            map_vertex_id += [(v_id,vertex_id)]
        #print(map_vertex_id)
        for v_id,vertex in self.Unit_ball.vert_dict.items():
            for w_id,w in vertex.get_neighbors():
                frm_id = [new_vid for vid,new_vid in map_vertex_id if vid == v_id][0]
                #print(frm_id)
                to_id = [new_vid for vid,new_vid in map_vertex_id if vid == w_id][0]
                #neighbor_id = self.check_vertex_exists(w_id,w.get_coordinates())
                self.add_edge(frm_id,to_id)
        self.mapp_vertices()
    
    def mapp(self,x,y,z):
        x_new = 2*x
        y_new = y
        z_new = (1/2)*z*(1+(1/2)*np.sin(2*np.pi*x))
        return x_new,y_new,z_new
                
    def mapp_vertices(self):
        v_ids = []
        for v_id,v in self.vert_dict.items():
            x,y,z = v.get_coordinates()
            x_new,y_new,z_new = self.mapp(x,y,z)
            v.redefine_coordinates(x_new,y_new,z_new)
            v_ids += [(v_id,(x_new,y_new,z_new))]
        #for v_id, new_v_id in v_ids:
            #self.change_vertex_id(v_id,new_v_id)
            

    def level_set_function(self, x, y, z):
        return (1/4)*x**2 + y**2 + (4*z**2)/((1+(1/2)*np.sin(np.pi*x))**2) - 1
    
    def gradient_level_set_fct(self,x,y,z):
        grad_x = x/2 - (32*(z**2)*np.pi*np.cos(np.pi*x))/((2+np.sin(np.pi*x))**3)
        grad_y = 2*y
        grad_z = (32*z)/((2+np.sin(np.pi*x))**2)
        return np.array([grad_x,grad_y,grad_z])
    
    def normal_vector(self,x,y,z):
        gradient_p = self.gradient_level_set_fct(x,y,z)
        #print('test')
        #print(gradient_p)
        gradient_p = self.compute_grad_at_point(x, y, z)
        #print(gradient_p)
        return gradient_p/(self.norm(gradient_p))
    
    def Hessian_lvl_set_fct(self,x,y,z):
        
        der_x_x = 4*(z**2) *(((np.pi**2)*np.sin(np.pi*x))/((1/2)*np.sin(np.pi*x)+1)**3 +(3*(np.pi**2)*(np.cos(np.pi*x))**2)/(2*((1/2)*np.sin(np.pi*x)+1)**4)) +(1/2)
        
        der_x_y = 0
        
        der_x_z = -(64*z*np.pi*np.cos(np.pi*x))/((2+np.sin(np.pi*x))**3)
        
        der_y_x = 0
        
        der_y_y = 2
        
        der_y_z = 0
        
        der_z_x = -(64*np.pi*z*np.cos(np.pi*x))/((np.sin(np.pi*x)+2)**3)
        
        der_z_y = 0
        
        der_z_z = 32/((np.sin(np.pi*x)+2)**2)
        
        return der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z
        
    def mean_curvature(self,x,y,z):
        gradient_p = self.gradient_level_set_fct(x, y, z)
        der_x, der_y, der_z = gradient_p 
        normed_gradient_p = self.norm(gradient_p)
        der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z = self.Hessian_lvl_set_fct(x, y, z)
        #der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z = self.compute_sec_derv(x,y,z)
        first_sum = (1 - (der_x**2)/(normed_gradient_p**2))*der_x_x
        #second_sum = ((der_x*der_y)/(normed_gradient_p**2))*der_x_y
        third_sum = ((der_x*der_z)/(normed_gradient_p**2))*der_x_z
        #fourth_sum = ((der_y*der_x)/(normed_gradient_p**2))*der_y_z
        fifth_sum = (1 - (der_y**2)/(normed_gradient_p**2))*der_y_y
        #sixt_sum = ((der_y*der_z)/(normed_gradient_p**2))*der_y_z
        sevent_sum = ((der_z*der_x)/(normed_gradient_p**2))*der_z_x
        #eighth_sum = ((der_z*der_y)/(normed_gradient_p**2))*der_z_y
        nineth_sum = (1 - (der_z**2)/(normed_gradient_p**2))*der_z_z
        res = (1/normed_gradient_p)*(first_sum - third_sum + fifth_sum - sevent_sum + nineth_sum)
        return res
        
    def compute_sec_derv(self,x,y,z):
        x1,x2,x3 = sp.symbols('x1 x2 x3')
        point = {x1: x, x2: y, x3: z}
        der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z = self.Hessian_lvl_set.subs(point)
        return der_x_x, der_x_y, der_x_z, der_y_x, der_y_y, der_y_z, der_z_x, der_z_y, der_z_z
    
    def compute_Hessian(self):
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        lvl_set_fct = (1/4) * x1**2 + x2**2 + (4 * x3**2) / (1 + (1/2) * sp.sin(sp.pi * x1))**2 - 1
        hessian = sp.hessian(lvl_set_fct, (x1,x2,x3))
        return hessian
    
    def compute_grad(self):
        x1,x2,x3 = sp.symbols('x1 x2 x3')
        lvl_set_fct = (1/4) * x1**2 + x2**2 + (4 * x3**2) / (1 + (1/2) * sp.sin(sp.pi * x1))**2 - 1
        grad = [sp.diff(lvl_set_fct, var) for var in [x1, x2 ,x3]]
        return grad
    
    def compute_grad_at_point(self,x,y,z):
        #print(x,y,z)
        x1,x2,x3 = sp.symbols('x1 x2 x3')
        point = {x1: x, x2: y, x3: z}
        grad_x = self.grad_lvl_set_fct[0].subs(point).evalf()
        grad_y = self.grad_lvl_set_fct[1].subs(point).evalf()
        grad_z = self.grad_lvl_set_fct[2].subs(point).evalf()
        return np.array([grad_x,grad_y,grad_z], dtype=np.float64)
    
    def create_macrotriangulation(self):
        Nodes = [(2,0,0),(0,1,0),(0,0,1/2),(-2,0,0),(0,-1,0),(0,0,-1/2)]
        for vertex in Nodes:
            self.add_vertex(vertex)
        for v_id,v in self.vert_dict.items():
            for w_id,w in self.vert_dict.items():
                check_vec = v.get_coordinates() + w.get_coordinates()
                if np.array_equal(check_vec, np.array([0,0,0]), equal_nan=False) == False:
                    self.add_edge(v_id, w_id) 

    def norm(self,v):
        return np.linalg.norm(v)
    
    def midpoint(self,v,w):
        return (v+w)/2
    

    def check_common_neighbours(self,v,w):
        v_neighbours = self.vert_dict[v].get_neighbors()
        w_neighbours = self.vert_dict[w].get_neighbors()
        common_neighbours = [vertex for vertex in v_neighbours if vertex in w_neighbours]
        return common_neighbours
    
    def get_old_common_neighbours(self,v,w,old_edges):
        old_common_neighbours = []
        if v != w:
            for (v1,v2) in old_edges:
                if v == v1:
                    if (w,v2) in old_edges:
                        if v2 != w and v != v2:
                            old_common_neighbours += [v2]
        return old_common_neighbours
    
    def refine_edges(self,v_id,w_id,v,w, new_vertex,old_vertices, old_edges):
        common_neigbours = []
        old_common_neighbours = self.get_old_common_neighbours(v_id, w_id, old_edges)
        for vertex in old_common_neighbours:
            common_neigbours = self.check_common_neighbours(vertex, v_id) + self.check_common_neighbours(vertex, w_id)
            for vert in common_neigbours:
                vert_tuple = tuple(vert.get_coordinates())
                if (not any(vert_tuple == old_vertex[0] for old_vertex in old_vertices)):
                    self.add_edge(new_vertex,vert_tuple)
                    self.add_edge(vert_tuple,new_vertex)
        
        self.add_edge(new_vertex, v_id)
        self.add_edge(new_vertex, w_id)   
        
    
    def project_vertex(self,new_vertex):
        x,y,z = new_vertex
        normal = self.normal_vector(x, y, z)
        t = 0
        while np.abs(self.level_set_function(x, y, z)) >= 10**(-2):
            x,y,z = new_vertex - normal*t
            t += 0.0001

        return x,y,z
        
    
    def nodal_wal(self):
        old_vertices = [(v_id,vertex) for v_id,vertex in self.vert_dict.items()]
        old_edges = self.get_edges()
        seen = []
        for v_id,v in old_vertices:
            for w in v.get_neighbors():
                if any(w == old_vertex[1] for old_vertex in old_vertices):
                    self.remove_edge(v_id, tuple(w.get_coordinates()))
                    self.remove_edge(tuple(w.get_coordinates()) , v_id)
                    x,y,z = self.midpoint(v.get_coordinates(), w.get_coordinates())
                    eucl_midpoint = np.array([x,y,z])
                    x_proj,y_proj,z_proj = self.project_vertex(eucl_midpoint)
                    new_vertex = self.add_vertex((x_proj,y_proj,z_proj))
                    self.refine_edges(v_id, tuple(w.get_coordinates()) ,v,w, new_vertex,old_vertices,old_edges)
                    if np.isnan(x):
                        print(v.get_coordinates())
                        print(w.get_coordinates())
    
def main():
    print('start plots and define classes')
    surface = Surface(0)
    Visz.Plot_surface(surface.level_set_function, 0.01, 500, 'integration_test.html')
    surface.refine()
    #surface.refine()
    #surface.refine()
    #print(np.linalg.norm(surface.gradient_level_set_fct(0, 1, 0)))
    #print('Unit ball')
    #surface.Unit_ball.print_Triangulation()
    #print('surface')
    #surface.print_Triangulation()
    Visz.Plot_Discrete_surface(surface.vert_dict, 'Integration_test_discrete_surface.html')
    #surface.refine()
    #surface.refine()
    #surface.plot_discrete_surface('discrete_surface_.html')
    #surface.plot_surface_png()
    #surface.plot_discrete_surface_png()

def main_v2():
    print('Start main V2', "red")
    square = Icosahedron(0)
    #square.refinement()
    Visz.Plot_Discrete_surface(square.vert_dict, 'Test_square.html')    
        
if __name__ == '__main__':
    #main()
    main_v2()   
        
        
        
