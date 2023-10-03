import numpy as np
import random
import copy
from expert.BFS import BFSPlanner
from simulator.maze_env import MazeEnv
import torch
from configs.str2config import str2config, add_default_argument_and_parse
import argparse
from configs.str2config import str2config, add_default_argument_and_parse
from utils.config import *
from expert.CBS import cbs, CBSPlanner
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
def get_neighborhood_cellindex(node_x_index,node_y_index,cell_width_num,cells_list):
    neighborhood_cellindex_list=[]
    step_list=[-1,0,1]
    for i in step_list:
        for j in step_list:
            new_node_x_index=node_x_index+i
            new_node_y_index=node_y_index+j
            if new_node_x_index<0 or new_node_y_index<0 or new_node_x_index>=cell_width_num or new_node_y_index>=cell_width_num:
                continue
            cell_index=cell_width_num*new_node_x_index+new_node_y_index
            cell_nodes=cells_list[cell_index]
            if len(cell_nodes)!=0:
                for k in range(len(cell_nodes)):
                    neighborhood_cellindex_list.append(cell_nodes[k])
    return neighborhood_cellindex_list
def fixed_radius_near_neighborhoods_nodes(env):
    minimum_w=-1
    maximum_w=1
    robot_R=0.01
    minimum_interval=2*robot_R
    cell_width_num=int((maximum_w-minimum_w)/minimum_interval)
    graph = env.graphs
    points = graph['points']
    node_num = np.shape(points)[0]
    total_cell_num=cell_width_num*cell_width_num
    cells_list=[]
    for i in range(total_cell_num):
        cells_list.append([])
    for i in range(node_num):
        node_pos=points[i]
        node_x=node_pos[0]
        node_y=node_pos[1]
        node_x_index=int((node_x-minimum_w)/minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        cell_index=cell_width_num*node_x_index+node_y_index
        cells_list[cell_index].append(i)
    for i in range(node_num):
        node_pos = points[i]
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        neighborhood_cellindex_list=get_neighborhood_cellindex(node_x_index,node_y_index,cell_width_num,cells_list)
        print(neighborhood_cellindex_list)
def merge_list(list1,list2):
    for i in range(len(list2)):
        if list2[i] not in list1:
            list1.append(list2[i])
    return list1
def updates_cellis_list_nodes(node_pos1,node_pos2,cell_width_num,minimum_w,minimum_interval):
    all_nodes_pos=[]
    all_insert_index=[]
    all_nodes_pos.append(node_pos1)
    all_nodes_pos.append(node_pos2)
    factor=100
    R = 0.02/factor
    segment_len=int(np.linalg.norm(node_pos2-node_pos1)/0.02*factor)
    for i in range(segment_len-1):
        all_nodes_pos.append((i+1)*R/np.linalg.norm(node_pos2-node_pos1)*(node_pos2-node_pos1)+node_pos1)
    for node_pos in all_nodes_pos:
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        cell_index = cell_width_num * node_x_index + node_y_index
        if cell_index not in all_insert_index:
            all_insert_index.append(cell_index)
    return all_insert_index
def get_all_neighborhoods_edges_for_intersections(node_pos1,node_pos2,cell_width_num,minimum_w,minimum_interval,cells_list_nodes):
    all_nodes_pos = []
    all_insert_index = []
    all_nodes_pos.append(node_pos1)
    all_nodes_pos.append(node_pos2)
    factor=100
    R=0.02/factor
    segment_len = int(np.linalg.norm(node_pos2 - node_pos1) / 0.02*factor)
    for i in range(segment_len - 1):
        all_nodes_pos.append(
            (i + 1) * R / np.linalg.norm(node_pos2 - node_pos1) * (node_pos2 - node_pos1) + node_pos1)
    for node_pos in all_nodes_pos:
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        cell_index = cell_width_num * node_x_index + node_y_index
        for index in cells_list_nodes[cell_index]:
            if index not in all_insert_index:
                all_insert_index.append(index)
    return all_insert_index
def fixed_radius_near_neighborhoods_edges(env):
    minimum_w = -1
    maximum_w = 1
    robot_R = 0.01
    minimum_interval = 2 * robot_R
    cell_width_num = int((maximum_w - minimum_w) / minimum_interval)
    graph = env.graphs
    points = graph['points']
    node_num = np.shape(points)[0]
    total_cell_num = cell_width_num * cell_width_num
    cells_list = []
    cells_list_nodes=[]
    intersection_list=[]
    edges_to_edges_dict={}
    for i in range(total_cell_num):
        cells_list.append([])
        cells_list_nodes.append([])
    edge_index = graph['edge_index']
    edges = []
    edge_index_dict = {}
    edge_count = 0
    start_time=time.time()
    for i in range(np.shape(edge_index)[0]):
        check_edge = edge_index[i, :]
        if check_edge[0] != check_edge[1] and [check_edge[1], check_edge[0]] not in edges:
            edges.append([check_edge[0], check_edge[1]])
            edge_index_dict[(check_edge[0], check_edge[1])] = edge_count
            edge_count = edge_count + 1
    end_time=time.time()
    print("phase1")
    print(end_time-start_time)
    start_time = time.time()
    for i in range(edge_count):
        endpoint_1=edges[i][0]
        endpoint_2=edges[i][1]
        node_pos1 = points[endpoint_1]
        node_x = node_pos1[0]
        node_y = node_pos1[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        cell_index1 = cell_width_num * node_x_index + node_y_index
        cells_list[cell_index1].append(i)
        node_pos2 = points[endpoint_2]
        node_x = node_pos2[0]
        node_y = node_pos2[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        cell_index2 = cell_width_num * node_x_index + node_y_index
        cells_list[cell_index2].append(i)
        all_insert_index=updates_cellis_list_nodes(node_pos1,node_pos2,cell_width_num,minimum_w,minimum_interval)
        for index in all_insert_index:
            cells_list_nodes[index].append(i)
    print("phase2")
    end_time = time.time()
    print(end_time - start_time)
    start_time = time.time()
    select_edges_pairs=[]
    nodes_edges_pairs=[]
    for i in range(len(points)):
        node_pos = points[i]
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        neighborhood_cellindex_list = get_neighborhood_cellindex(node_x_index, node_y_index, cell_width_num,
                                                                      cells_list_nodes)
        for edge in neighborhood_cellindex_list:
            nodes_edges_pairs.append((i,edge))
    print("phase3")
    end_time = time.time()
    print(end_time - start_time)
    start_time = time.time()
    for i in range(edge_count):
        endpoint_1=edges[i][0]
        endpoint_2=edges[i][1]
        node_pos = points[endpoint_1]
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        neighborhood_cellindex_list_end1 = get_neighborhood_cellindex(node_x_index, node_y_index, cell_width_num, cells_list)
        node_pos = points[endpoint_2]
        node_x = node_pos[0]
        node_y = node_pos[1]
        node_x_index = int((node_x - minimum_w) / minimum_interval)
        node_y_index = int((node_y - minimum_w) / minimum_interval)
        neighborhood_cellindex_list_end2 = get_neighborhood_cellindex(node_x_index, node_y_index, cell_width_num,
                                                                      cells_list)
        neighborhood_cellindex_list=merge_list(neighborhood_cellindex_list_end1,neighborhood_cellindex_list_end2)
        for j in range(len(neighborhood_cellindex_list)):
            if (neighborhood_cellindex_list[j],i) not in edges_to_edges_dict and (i,neighborhood_cellindex_list[j]) not in edges_to_edges_dict:
                if i==neighborhood_cellindex_list[j]:
                    continue
                edges_to_edges_dict[(i,neighborhood_cellindex_list[j])]=True
                edges_to_edges_dict[(neighborhood_cellindex_list[j],i)]=True
                select_edges_pairs.append((i,neighborhood_cellindex_list[j]))
        #print(neighborhood_cellindex_list)
    #print(select_edges_pairs)
    #print(nodes_edges_pairs)
    #print(nodes_edges_pairs)
    print("phase4")
    end_time = time.time()
    print(end_time - start_time)
    start_time = time.time()
    count = 0
    for i in range(edge_count):
        endpoint_1 = edges[i][0]
        endpoint_2 = edges[i][1]
        node_pos1 = points[endpoint_1]
        node_pos2=points[endpoint_2]
        all_insert_index=get_all_neighborhoods_edges_for_intersections(node_pos1,node_pos2,cell_width_num,minimum_w,minimum_interval,cells_list_nodes)

        for index in all_insert_index:
            if index!=i:
                if (i,index) not in edges_to_edges_dict and (index,i) not in edges_to_edges_dict:
                    select_edges_pairs.append((i,index))
                    edges_to_edges_dict[(i,index)]=True
                    edges_to_edges_dict[(index,i)]=True
                flag=False
                if edges[index][0]==endpoint_1 or edges[index][0]==endpoint_2 or edges[index][1]==endpoint_1 or edges[index][1]==endpoint_2:
                    flag=True
                if flag==False:
                    #print(points[edges[index][0]])
                    #print(points[edges[index][1]])
                    #print(points[endpoint_1])
                    #print(points[endpoint_2])
                    count=count+1
    print(count)
    print("phase5")
    end_time = time.time()
    print(end_time - start_time)
    print(len(select_edges_pairs))
    return select_edges_pairs,nodes_edges_pairs,edges_to_edges_dict

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Start the experiment agent")
    config_setup = add_default_argument_and_parse(arg_parser, 'experiment')
    config_setup = process_config(config_setup)
    config_setup.env_name = "MazeEnv"
    env = MazeEnv(config_setup)
    env.init_new_problem_graph(index=0)
    env.init_new_problem_instance(index=0)
    fixed_radius_near_neighborhoods_edges(env)