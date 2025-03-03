#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:35:09 2025

@author: jschlieffen
"""

import plotly.graph_objects as go
import numpy as np

class Plot_surface:
    
    def __init__(self,lvl_set_fct,treshold, num_points,title_surface,title_func_plot,function):
        self.lvl_set_fct = lvl_set_fct
        self.treshold = treshold
        self.num_points = num_points
        self.title = title_surface
        self.title_func_plot = title_func_plot
        self.func = function
        self.create_plot()
        self.create_plot_function()
        
    def define_mesh(self):
        x = np.linspace(-5, 5, self.num_points)
        y = np.linspace(-5, 5, self.num_points)
        z = np.linspace(-5, 5, self.num_points)
        return np.meshgrid(x, y, z)
        
    def define_layout(self,fig):
        fig.update_layout(
            title='Interactive 3D Level Set (Sphere)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(
                    range=[-2,2],
                    tickvals=[-2,-1,0,1,2],
                    ticktext=['-2','-1','0','1','2'],
                ),
                yaxis=dict(
                    range=[-2,2],
                    tickvals=[-2,-1,0,1,2],
                    ticktext=['-2','-1','0','1','2'],
                ),
                zaxis=dict(
                    range=[-1,1],
                    tickvals=[-2,-1,0,1,2],
                    ticktext=['-2','-1','0','1','2'],
                ),
            ),
            showlegend=False
        )
    
    def level_set_fct(self,x,y,z):
        return self.lvl_set_fct(x, y, z)
    
    def create_plot(self):
        X,Y,Z = self.define_mesh()
        F = self.level_set_fct(X, Y, Z)
        threshold = 0.05
        points = np.abs(F) < threshold 
        x_surface = X[points]
        y_surface = Y[points]
        z_surface = Z[points]
        
        fig = go.Figure(data=[go.Scatter3d(x=x_surface, y=y_surface, z=z_surface,)])
        self.define_layout(fig)
        fig.write_html(self.title)
        
    def create_plot_function(self):
        X,Y,Z = self.define_mesh()
        F = self.level_set_fct(X, Y, Z)
        #points_func = self.calc_func_vals(F)
        
        threshold = 0.05
        points = np.abs(F) < threshold 
        x_surface = X[points]
        y_surface = Y[points]
        z_surface = Z[points]
        points_func = self.func(x_surface,y_surface,z_surface)
        #points_func = np.abs
        fig = go.Figure(data=[go.Scatter3d(
            x=x_surface, 
            y=y_surface, 
            z=z_surface,
            mode='markers',
            marker=dict(
                size=5,
                color=points_func,
                colorscale='Viridis',
                colorbar=dict(title='x * y'))
        )])
        self.define_layout(fig)
        fig.write_html(self.title_func_plot)
        #fig.show()
        
class Plot_Discrete_surface:
    
    def __init__(self,vert_dict,title,func_vals, title_func_plot):
        self.vert_dict = vert_dict
        self.title = title
        self.title_func_plot = title_func_plot
        self.func_vals = func_vals
        self.create_plot()
        self.create_plot_func_values()
    
    def create_node_list(self):
        x,y,z, Node_ids = [],[],[],[]
        for elem_id,elem in self.vert_dict.items() :
            coord_x, coord_y, coord_z = elem.get_coordinates()
            if False:
                if coord_x >= 0 and coord_y >= 0 and coord_z >= 0:
                    x.append(coord_x)
                    y.append(coord_y)
                    z.append(coord_z)
            else:
                x.append(coord_x)
                y.append(coord_y)
                z.append(coord_z)
                Node_ids.append(elem_id)
        return x,y,z, Node_ids
        
    def create_func_val_list(self):
        x,y,z, Node_ids = [],[],[],[]
        for elem_id,elem in self.vert_dict.items() :
            coord_x, coord_y, coord_z = elem.get_coordinates()
            if False:
                if coord_x >= 0 and coord_y >= 0 and coord_z >= 0:
                    x.append(coord_x)
                    y.append(coord_y)
                    z.append(coord_z)
            else:
                x.append(coord_x)
                y.append(coord_y)
                z.append(coord_z)
                Node_ids.append(elem_id)
        return x,y,z, Node_ids

    def create_edge_list(self):
        edge_lines = []
        for v_id,v in self.vert_dict.items():
            for w_id, w in v.get_neighbors():
                x1, y1, z1 = v.get_coordinates()
                x2, y2, z2 = w.get_coordinates()
                if False:
                    if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0 and z1 >= 0 and z2 >= 0:
                        edge_lines.append(go.Scatter3d(
                            x=[x1,x2],
                            y=[y1,y2],
                            z=[z1,z2],
                            mode='lines',
                            line=dict(color='red', width=2)
                        ))
                else: 
                    edge_lines.append(go.Scatter3d(
                        x=[x1,x2],
                        y=[y1,y2],
                        z=[z1,z2],
                        mode='lines',
                        line=dict(color='red', width=2)
                    ))
        return edge_lines
    
    def define_layout(self,fig):
        fig.update_layout(
            title='Oktahedron',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
                ),
            showlegend=False
            )
        
    
    def create_plot(self):
        x,y,z,Node_ids = self.create_node_list()
        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5, color='blue'),
            text = Node_ids,
            hoverinfo = 'text'
        )
        edge_lines = self.create_edge_list()
        fig = go.Figure(data=[scatter]+edge_lines)
        self.define_layout(fig)
        fig.write_html(self.title)
    
    def create_plot_func_values(self):
        x,y,z,Node_ids = self.create_node_list()
        #func_vals = self.create_func_val_list()
        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5, 
                        color=self.func_vals.flatten(),
                        colorscale='Viridis',
                        colorbar=dict(title="Function value")),
            text = Node_ids,
            hoverinfo = 'text'
        )
        edge_lines = self.create_edge_list()
        fig = go.Figure(data=[scatter]+edge_lines)
        self.define_layout(fig)
        fig.write_html(self.title_func_plot)
        
        
        
        
def main():
    def level_set_fct_Unit_ball(x,y,z):
        return x**2 + y**2 + z**2 -1
    
    test_plot = Plot_surface(level_set_fct_Unit_ball, 0.01, 500, 'test.html')

if __name__ =='__main__':
    main()
        
        