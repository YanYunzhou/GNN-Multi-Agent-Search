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
from line_segment_intersection import *

minimum_checking_time=0
def update_edge_interval_from_edges(points, current_node,parent_node, current_time, next_time, edge_dict,edges_conflicts_dict):
    current_start_pos = points[parent_node]
    current_end_pos = points[current_node]
    #V1=(current_end_pos-current_start_pos)/np.linalg.norm((current_end_pos-current_start_pos))
    #dst_threshold = 0.0212
    eps = 0
    #R=0.01
    edges_conflicts_set=edges_conflicts_dict[(parent_node,current_node)]

    if current_time>minimum_checking_time:
        for conflicts in edges_conflicts_set:
            edge=conflicts[0]
            collision_time=conflicts[1]
            delay=conflicts[2]
            if edge[0]==current_node:
                edge1=current_end_pos-current_start_pos
                edge1=edge1/np.linalg.norm(edge1)
                edge2=points[edge[1]]-points[edge[0]]
                edge2 = edge2 / np.linalg.norm(edge2)
                #print(np.dot(edge1,edge2))
                #print(delay)
            new_delay=copy.deepcopy(delay)
            new_delay[0]=min(delay[0],delay[1])
            new_delay[1] = max(delay[0], delay[1])
            delay=new_delay
            minimum_valid_time=np.linalg.norm(points[edge[0]]-points[edge[1]])
            valid_time = [-minimum_valid_time-0.02, next_time - current_time+0.02]
            if (collision_time[0]>=-0.02 and collision_time[0]<=valid_time[1]) or (collision_time[1]>=-0.02 and collision_time[1]<=valid_time[1]) or  (min(collision_time)<=0 and max(collision_time)>=valid_time[1]-0.02):
                collision_start=max(valid_time[0],delay[0])
                collision_end=min(valid_time[1],delay[1])
                #if current_node not in edge and parent_node not in edge:
                    #print(delay)
                    #print(edge)
                    #print(parent_node)
                    #print(current_node)
                    #print(valid_time)
                    #print(collision_start)
                    #print(collision_end)
                collision_start=current_time+collision_start
                collision_end=current_time+collision_end
                if collision_start<0:
                    collision_start=0

                if collision_end<=0 or collision_start>collision_end:
                    return edge_dict
                if type(edge_dict[edge])==type(False):
                    edge_dict[edge] = np.array(
                        [[collision_start, collision_end]])
                else:
                    # edge_dict[(best_node.state, best_node.parent.state)]=np.append(edge_dict[(best_node.state, best_node.parent.state)],np.array([[leave_time,best_node.time]]),axis=0)
                    edge_dict[edge] = insert_collision_interval(
                        edge_dict[edge],
                        np.array([[collision_start - eps, collision_end + eps]]))

    return edge_dict
    # print(points_on_segment1)


def select_edges(points,edge_end, edge_start, neighbors,selected_neighborhood_num=2):
    edges_set = []
    new_points = []
    all_points=[edge_start,edge_end]
    ignored_set=[]
    edges_set.append([edge_end,edge_start])
    for neigh in neighbors[edge_start]:
        relative_pos=points[edge_end]-points[edge_start]
        if neigh != edge_start and neigh != edge_end:
            check_rel_pos=points[neigh]-points[edge_start]
            dot_prod=np.dot(relative_pos,check_rel_pos)/(np.linalg.norm(relative_pos)*np.linalg.norm(check_rel_pos))
            if dot_prod>=0:
                edges_set.append([edge_start, neigh])
            else:
                ignored_set.append([edge_start,neigh])
            if neigh not in new_points:
                new_points.append(neigh)
                all_points.append(neigh)
    for neigh in neighbors[edge_end]:
        relative_pos = points[edge_start] - points[edge_end]
        if neigh != edge_end and neigh != edge_start:
            check_rel_pos = points[neigh] - points[edge_end]
            dot_prod = np.dot(relative_pos, check_rel_pos)
            if dot_prod>=0:
                edges_set.append([edge_end, neigh])
            else:
                ignored_set.append([edge_end,neigh])
            if neigh not in new_points:
                new_points.append(neigh)
                all_points.append(neigh)
    for k in range(selected_neighborhood_num-1):
        new_points_copy = new_points.copy()
        new_points = []
        for point in new_points_copy:
            point_neighbor = neighbors[point]
            for neigh in point_neighbor:
                if neigh != point and [point, neigh] not in edges_set and [neigh, point] not in edges_set and [point, neigh] not in ignored_set and [neigh, point] not in ignored_set:
                    edges_set.append([point, neigh])
                    if neigh not in all_points:
                        new_points.append(neigh)
                        all_points.append(neigh)

    return edges_set
def insert_nodes_record(nodes_record,agent,next_node,start_time,leave_time):
    interval_num=len(nodes_record)
    if interval_num==0:
        nodes_record.append([agent,next_node,start_time,leave_time])
    elif interval_num==1:
        if leave_time<=nodes_record[0][2]:
            nodes_record.insert(0,[agent,next_node,start_time,leave_time])
        elif start_time>=nodes_record[0][3]:
            nodes_record.insert(1, [agent, next_node, start_time, leave_time])
        else:
            print("something error")
    elif leave_time<=nodes_record[0][2]:
        nodes_record.insert(0, [agent, next_node, start_time, leave_time])
    elif start_time>=nodes_record[interval_num-1][3]:
        nodes_record.append([agent, next_node, start_time, leave_time])
    else:
        #print(nodes_record)
        #print(start_time)
        #print(leave_time)
        low_index=0
        high_index=interval_num-1
        check_index=int((low_index+high_index+1)/2)
        terminate_flag=False
        insert_index=-1
        #print(nodes_record)
        #print(start_time)
        #print(leave_time)
        while terminate_flag==False:
            if check_index==interval_num-1:
                high_index=check_index-1
                check_index=int((low_index+high_index+1)/2)
                continue
            elif start_time>=nodes_record[check_index][3] and leave_time<=nodes_record[check_index+1][2]:
                terminate_flag=True
                insert_index=check_index+1
                continue
            if high_index==low_index+1 and leave_time<=nodes_record[check_index][2] and check_index==high_index:
                high_index=low_index
                check_index=low_index
            elif leave_time<=nodes_record[check_index][2]:
                high_index=check_index
                check_index = int((low_index + high_index + 1) / 2)
            elif start_time>=nodes_record[check_index][3]:
                low_index=check_index
                check_index = int((low_index + high_index + 1) / 2)
        nodes_record.insert(insert_index, [agent, next_node, start_time, leave_time])

def select_points(edge_start, edge_end, neighbors,selected_neighborhood_num=2):
    new_points = []
    all_points=[edge_start,edge_end]
    for neigh in neighbors[edge_start]:
        if neigh != edge_start and neigh != edge_end:
            if neigh not in new_points:
                new_points.append(neigh)
                all_points.append(neigh)
    for neigh in neighbors[edge_end]:
        if neigh != edge_end and neigh != edge_start:
            if neigh not in new_points:
                new_points.append(neigh)
                all_points.append(neigh)

    for i in range(selected_neighborhood_num):
        new_points_copy = new_points.copy()
        new_points = []
        for point in new_points_copy:
            point_neighbor = neighbors[point]
            for neigh in point_neighbor:
                if neigh != point:
                    if neigh not in all_points:
                        new_points.append(neigh)
                        all_points.append(neigh)
    return all_points
def select_edges_points(current_point, neighbors,selected_neighborhood_num):
    edges_set = []
    new_points = []
    all_points=[current_point]
    for neigh in neighbors[current_point]:
        if neigh != current_point:
            edges_set.append([current_point, neigh])
            if neigh not in new_points:
                new_points.append(neigh)
                all_points.append(neigh)
    for i in range(selected_neighborhood_num):
        new_points_copy = new_points.copy()
        new_points = []
        for point in new_points_copy:
            point_neighbor = neighbors[point]
            for neigh in point_neighbor:
                if neigh != point and [point, neigh] not in edges_set and [neigh, point] not in edges_set:
                    edges_set.append([point, neigh])
                    if neigh not in all_points:
                        new_points.append(neigh)
                        all_points.append(neigh)
    return edges_set

def update_edge_intervals_from_points( points, current_node, current_time, next_time, edge_dict,nodes_edges_conflicts_dict):
    current_node = current_node.state
    current_pos = points[current_node]
    dst_threshold = 0.022
    eps = 0
    edges_conflicts_set=nodes_edges_conflicts_dict[current_node]
    if len(edges_conflicts_set)==0:
        return edge_dict
    #print(edges_conflicts_set)
    if current_time>minimum_checking_time:
        for select_edge in edges_conflicts_set:
            collision_start = current_time
            collision_end = next_time
            current_edge=select_edge[0]
            found_points=select_edge[1]
            check_start_pos = points[current_edge[0]]
            check_end_pos = points[current_edge[1]]
            line_seg=np.linalg.norm(check_start_pos-check_end_pos)
            check_current_node = current_edge[0]
            check_parent_node = current_edge[1]
            if found_points[0][0] - check_start_pos[0] == 0:
                index1=0
            else:
                index1=(found_points[0][0] - check_start_pos[0]) / (check_end_pos[0] - check_start_pos[0])
            if (found_points[1][0] - check_start_pos[0]) == 0:
                index2=0
            else:
                index2= (found_points[1][0] - check_start_pos[0]) / (check_end_pos[0] - check_start_pos[0])
            min_index=min(index1,index2)
            max_index=max(index1,index2)
            collision_start=current_time-max_index*line_seg
            if collision_start<0:
                collision_start=0
            collision_end=next_time-min_index*line_seg
            if collision_end<0:
                collision_end=0
            if type(edge_dict[(check_current_node, check_parent_node)]) == type(False):
                edge_dict[(check_current_node, check_parent_node)] = np.array(
                    [[collision_start - eps, collision_end + eps]])
            else:
                edge_dict[(check_current_node, check_parent_node)] = insert_collision_interval(
                    edge_dict[(check_current_node, check_parent_node)],
                    np.array([[collision_start - eps, collision_end + eps]]))

            if found_points[0][0] - check_end_pos[0] == 0:
                index1 = 0
            else:
                index1=(found_points[0][0] - check_end_pos[0]) / (check_start_pos[0] - check_end_pos[0])
            if found_points[1][0] - check_end_pos[0] == 0:
                index2=0
            else:
                index2= (found_points[1][0] - check_end_pos[0]) / (check_start_pos[0] - check_end_pos[0])
            min_index = min(index1, index2)
            max_index = max(index1, index2)
            collision_start = current_time - max_index * line_seg
            if collision_start < 0:
                collision_start = 0
            collision_end = next_time - min_index * line_seg
            if collision_end < 0:
                collision_end = 0
            if type(edge_dict[(check_parent_node, check_current_node)]) == type(False):
                edge_dict[(check_parent_node, check_current_node)] = np.array(
                    [[collision_start - eps, collision_end + eps]])
            else:
                edge_dict[(check_parent_node, check_current_node)] = insert_collision_interval(
                    edge_dict[(check_parent_node, check_current_node)],
                    np.array([[collision_start - eps, collision_end + eps]]))
    return edge_dict
def search_interval_index(node_interval,interval_num,time):
    low_index=0
    high_index=interval_num-1
    terminate_flag=False
    check_index=int((low_index+high_index+1)/2)
    selected_index=-1
    while terminate_flag==False:
        if time>=node_interval[check_index][0] and time<=node_interval[check_index][1]:
            selected_index=check_index
            terminate_flag=True
            continue
        else:
            if time<node_interval[check_index][0]:
                high_index=check_index
                if high_index==low_index+1:
                    check_index=low_index
                else:
                    check_index = int((low_index + high_index + 1) / 2)
            else:
                low_index=check_index
                check_index = int((low_index + high_index + 1) / 2)
    return selected_index


def find_points_at_distance_r(p, p1, p2, r):
    # Convert points to numpy arrays for easier calculations
    p, p1, p2 = np.array(p), np.array(p1), np.array(p2)

    # Calculate the squared length of the line segment
    len_squared = np.sum((p2 - p1) ** 2)

    # Check if the line segment is a single point (length is zero)
    if len_squared == 0:
        raise ValueError("The line segment is a single point, no solution exists.")

    # Calculate the projection of point p onto the line segment
    t = np.dot(p - p1, p2 - p1) / len_squared
    #t = np.clip(t, 0, 1)  # Clamp the value to lie between [0, 1] to ensure it lies within the segment
    projection = p1 + t * (p2 - p1)

    # Calculate the distance from the point to the projection
    distance_to_proj = np.linalg.norm(p - projection)
    #print(distance_to_proj)

    # Check if the given distance r is larger than the distance to the projection
    if r > distance_to_proj:
        # Calculate the distance from the projection to the points on the line segment at distance r
        t_offset = np.sqrt(r ** 2 - distance_to_proj ** 2) / np.linalg.norm(p2 - p1)
        points = [projection + t_offset * (p2 - p1), projection - t_offset * (p2 - p1)]
        t1 = np.dot(points[0] - p1, p2 - p1) / len_squared
        t2= np.dot(points[1] - p1, p2 - p1) / len_squared
        if t1<0:
            t1=0
        elif t1>1:
            t1=1
        if t2<0:
            t2=0
        elif t2>1:
            t2=1
        points[0]=p1 + t1 * (p2 - p1)
        points[1] = p1 + t2 * (p2 - p1)
        return points
    elif r==distance_to_proj:
        # Calculate the distance from the projection to the points on the line segment at distance r
        t_offset = np.sqrt(r ** 2 - distance_to_proj ** 2) / np.linalg.norm(p2 - p1)
        points = [projection + t_offset * (p2 - p1), projection - t_offset * (p2 - p1)]
        t1 = np.dot(points[0] - p1, p2 - p1) / len_squared
        t2 = np.dot(points[1] - p1, p2 - p1) / len_squared
        if t1 < 0:
            t1 = 0
        elif t1 > 1:
            t1 = 1
        if t2 < 0:
            t2 = 0
        elif t2 > 1:
            t2 = 1
        points[0] = p1 + t1 * (p2 - p1)
        points[1] = p1 + t2 * (p2 - p1)
        return points[0]
    else:
        raise ValueError("There are no points on the line segment at distance r from the given point.")
def update_nodes_intervals_from_edges(points, current_node, parent_node, current_time, next_time, node_interval,edges_nodes_conflicts_dict):
    #node_interval = copy.deepcopy(node_interval)
    current_start_pos = points[parent_node]
    current_end_pos = points[current_node]
    dst_threshold = 0.0211
    eps = 0
    index = -1
    nodes_conflicts_list=edges_nodes_conflicts_dict[(parent_node,current_node)]
    if len(nodes_conflicts_list)==0:
        return node_interval,index
    # for i in node_interval:
    # if len(node_interval[i])%2==0:
    # print("Something has wrong 1")
    if current_time>minimum_checking_time:
        for node in nodes_conflicts_list:
            node_num=node[0]
            collision_pos1=node[1][0]
            collision_pos2=node[1][1]
            t1=abs(collision_pos1[0]-current_start_pos[0])/abs(current_end_pos[0]-current_start_pos[0])
            t2 = abs(collision_pos2[0] - current_start_pos[0]) / abs(current_end_pos[0] - current_start_pos[0])
            time1=t1*(next_time-current_time)+current_time
            time2=t2*(next_time-current_time)+current_time
            insert_current_time = min(time1, time2)
            insert_next_time = max(time1, time2)
            i=node_num
            interval_num = len(node_interval[i])
            index = i
            first_index = search_interval_index(node_interval[i], interval_num, insert_current_time)
            first_index_copy = first_index
            for p in range(1, first_index_copy + 1):
                test_index = first_index_copy - p
                if node_interval[i][test_index][0] <= insert_current_time and node_interval[i][test_index][
                    1] >= insert_current_time:
                    first_index = test_index
                else:
                    break
            second_index = search_interval_index(node_interval[i], interval_num, insert_next_time)
            second_index_copy = second_index
            for p in range(1, interval_num - 1 - second_index_copy + 1):
                test_index = second_index_copy + p
                if node_interval[i][test_index][0] <= insert_next_time and node_interval[i][test_index][
                    1] >= insert_next_time:
                    second_index = test_index
                else:
                    break
            if first_index % 2 == 1 and second_index == first_index + 1:
                node_interval[i][first_index][1] = insert_next_time + eps
                node_interval[i][first_index + 1][0] = insert_next_time + eps
            elif first_index % 2 == 0 and second_index == first_index:
                original_time = node_interval[i][first_index][1]
                node_interval[i][first_index][1] = insert_current_time - eps
                node_interval[i].insert(first_index + 1, [insert_current_time - eps, insert_next_time + eps])
                node_interval[i].insert(first_index + 2, [insert_next_time + eps, original_time])

            elif first_index % 2 == 0 and second_index == first_index + 2:
                node_interval[i][first_index][1] = insert_current_time - eps
                node_interval[i][first_index + 1][0] = insert_current_time - eps
                node_interval[i][first_index + 1][1] = insert_next_time + eps
                node_interval[i][first_index + 2][0] = insert_next_time + eps
            elif first_index % 2 == 0 and second_index == first_index + 1:
                node_interval[i][first_index][1] = insert_current_time
                node_interval[i][second_index][0] = insert_current_time
            elif first_index % 2 == 1 and second_index == first_index + 2:
                node_interval[i][first_index][1] = node_interval[i][second_index][1]
                node_interval[i].pop(first_index + 1)
                node_interval[i].pop(first_index + 1)
            elif first_index % 2 == 1 and first_index == second_index:
                continue
            elif first_index % 2 == 0 and second_index == first_index + 3:
                node_interval[i][first_index][1] = insert_current_time
                node_interval[i][first_index + 1][0] = insert_current_time
                node_interval[i][first_index + 1][1] = node_interval[i][second_index][1]
                node_interval[i].pop(first_index + 2)
                node_interval[i].pop(first_index + 2)
            elif first_index % 2 == 1 and second_index == first_index + 3:
                node_interval[i][first_index][1] = insert_next_time
                node_interval[i][second_index][0] = insert_next_time
                node_interval[i].pop(first_index + 1)
                node_interval[i].pop(first_index + 1)
            elif first_index % 2 == 0 and second_index == first_index + 4:
                node_interval[i][first_index][1] = insert_current_time
                node_interval[i][first_index + 1][0] = insert_current_time
                node_interval[i][first_index + 1][1] = insert_next_time
                node_interval[i][second_index][0] = insert_next_time
                node_interval[i].pop(first_index + 2)
                node_interval[i].pop(first_index + 2)
            elif first_index%2==0 and second_index==first_index+5:
                node_interval[i][first_index][1] = insert_current_time
                node_interval[i][first_index+1][0] = insert_current_time
                node_interval[i][first_index+1][1]=node_interval[i][second_index][1]
                node_interval[i].pop(first_index + 2)
                node_interval[i].pop(first_index + 2)
                node_interval[i].pop(first_index + 2)
                node_interval[i].pop(first_index + 2)
            else:
                print(first_index)
                print(second_index)

    # if flag==True:
    # print(node_interval[index])
    # for i in node_interval:
    # if len(node_interval[i])%2==0:
    # print("Something has wrong 2")
    return node_interval, index


def distance_between_points(p1, p2):
    # Function to compute the Euclidean distance between two points.
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def point_on_segment_with_distance(segment_start, segment_end, point, desired_distance):
    # Function to find a point on the line segment that has a distance
    # equal to the given desired distance from the specified point.

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def vector_subtraction(v1, v2):
        return v1[0] - v2[0], v1[1] - v2[1]

    def vector_magnitude(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    def distance_point_to_segment(p, v, w):
        vw = vector_subtraction(w, v)
        pv = vector_subtraction(p, v)
        pw = vector_subtraction(p, w)

        dot_product_pv = dot_product(pv, vw)
        dot_product_pw = dot_product(pw, vw)

        if dot_product_pv <= 0:
            return vector_magnitude(pv)
        elif dot_product_pw >= 0:
            return vector_magnitude(pw)
        else:
            # The point is between the two endpoints of the line segment,
            # so the distance is the perpendicular distance to the line.
            return abs(vw[0] * pv[1] - vw[1] * pv[0]) / vector_magnitude(vw)

    segment_vector = vector_subtraction(segment_end, segment_start)
    segment_length = vector_magnitude(segment_vector)
    point_vector = vector_subtraction(point, segment_start)

    # Calculate the projection of the point vector onto the segment vector
    t = dot_product(point_vector, segment_vector) / (segment_length ** 2)

    # If the projection is outside the range [0, 1], the closest point is one of the segment endpoints
    if t < 0:
        closest_point = segment_start
    elif t > 1:
        closest_point = segment_end
    else:
        # Calculate the closest point on the segment
        closest_point = (segment_start[0] + t * segment_vector[0], segment_start[1] + t * segment_vector[1])

    # Calculate the distance between the closest point and the given point
    distance_to_point = distance_between_points(closest_point, point)

    # Calculate the scaling factor to adjust the distance to the desired_distance
    scaling_factor = desired_distance / distance_to_point

    # Calculate the point on the line segment with the desired distance from the given point
    desired_point = (point[0] + scaling_factor * (closest_point[0] - point[0]),
                     point[1] + scaling_factor * (closest_point[1] - point[1]))

    return desired_point


def points_with_min_distance(seg1_start, seg1_end, seg2_start, seg2_end, desired_distance):
    # Function to find the points on the first line segment that have a minimum distance
    # equal to the given desired distance from the second line segment.

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def vector_subtraction(v1, v2):
        return v1[0] - v2[0], v1[1] - v2[1]

    def vector_magnitude(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    def distance_point_to_segment(p, v, w):
        vw = vector_subtraction(w, v)
        pv = vector_subtraction(p, v)
        pw = vector_subtraction(p, w)

        dot_product_pv = dot_product(pv, vw)
        dot_product_pw = dot_product(pw, vw)

        if dot_product_pv <= 0:
            return vector_magnitude(pv)
        elif dot_product_pw >= 0:
            return vector_magnitude(pw)
        else:
            # The point is between the two endpoints of the line segment,
            # so the distance is the perpendicular distance to the line.
            return abs(vw[0] * pv[1] - vw[1] * pv[0]) / vector_magnitude(vw)

    num_points = 300  # Number of points to sample on the first line segment
    points_on_seg1 = []
    seg1_vector = vector_subtraction(seg1_end, seg1_start)

    for i in range(num_points + 1):
        t = i / num_points
        point = (seg1_start[0] + t * seg1_vector[0], seg1_start[1] + t * seg1_vector[1])
        distance_to_seg2 = distance_point_to_segment(point, seg2_start, seg2_end)
        if abs(distance_to_seg2 - desired_distance) < 1e-5:
            points_on_seg1.append(point)

    return points_on_seg1


def distance_between_segments(seg1_start, seg1_end, seg2_start, seg2_end):
    # Function to compute the minimum distance between two line segments defined by their endpoints.

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def vector_subtraction(v1, v2):
        return v1[0] - v2[0], v1[1] - v2[1]

    def vector_magnitude(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    def distance_point_to_segment(p, v, w):
        vw = vector_subtraction(w, v)
        pv = vector_subtraction(p, v)
        pw = vector_subtraction(p, w)

        dot_product_pv = dot_product(pv, vw)
        dot_product_pw = dot_product(pw, vw)

        if dot_product_pv <= 0:
            return vector_magnitude(pv)
        elif dot_product_pw >= 0:
            return vector_magnitude(pw)
        else:
            # The point is between the two endpoints of the line segment,
            # so the distance is the perpendicular distance to the line.
            return abs(vw[0] * pv[1] - vw[1] * pv[0]) / vector_magnitude(vw)

    # Calculate distances between each point of one segment and the other segment
    distances = [
        distance_point_to_segment(seg1_start, seg2_start, seg2_end),
        distance_point_to_segment(seg1_end, seg2_start, seg2_end),
        distance_point_to_segment(seg2_start, seg1_start, seg1_end),
        distance_point_to_segment(seg2_end, seg1_start, seg1_end)
    ]

    return min(distances)

def search_agents_record(nodes_record,time):
    flag=False
    select_neighborhood=-1
    record_num=len(nodes_record)
    eps=0.0000001
    if record_num==0:
        return select_neighborhood,flag
    if time<=nodes_record[0][2]:
        return select_neighborhood,flag
    elif time>=nodes_record[record_num-1][3]-eps:
        select_neighborhood=nodes_record[record_num-1][1]
        flag=True
        return select_neighborhood,flag
    else:
        flag=True
        low_index=0
        high_index=record_num-1
        check_index=int((low_index+high_index+1)/2)
        terminate_flag=False
        while terminate_flag==False:
            if check_index==record_num-1:
                if time>=nodes_record[check_index][3]-eps:
                    select_neighborhood=nodes_record[check_index][1]
                    terminate_flag=True
                elif high_index==low_index+1 and time<=nodes_record[check_index][2]+eps and check_index==high_index:
                    high_index=low_index
                    check_index=low_index
                else:
                    high_index=check_index
                    check_index=int((low_index+high_index+1)/2)
                continue
            elif time>=nodes_record[check_index][3]-eps and time<=nodes_record[check_index+1][2]+eps:
                terminate_flag=True
                select_neighborhood=nodes_record[check_index][1]
                continue
            if high_index==low_index+1 and time<=nodes_record[check_index][2]+eps and check_index==high_index:
                high_index=low_index
                check_index=low_index
            elif time<=nodes_record[check_index][2]+eps:
                high_index=check_index
                check_index = int((low_index + high_index + 1) / 2)
            elif time>=nodes_record[check_index][3]-eps:
                low_index=check_index
                check_index = int((low_index + high_index + 1) / 2)
        return select_neighborhood,flag

def distance_point_to_line_segment(point_x, point_y, start_x, start_y, end_x, end_y):
    # Calculate the squared length of the line segment
    segment_length_squared = (end_x - start_x) ** 2 + (end_y - start_y) ** 2

    # If the line segment has zero length, return the distance to the start point
    if segment_length_squared == 0:
        return math.sqrt((point_x - start_x) ** 2 + (point_y - start_y) ** 2)

    # Calculate the parametric value (t) where the perpendicular from the point intersects the line segment
    t = max(0, min(1, ((point_x - start_x) * (end_x - start_x) + (point_y - start_y) * (
            end_y - start_y)) / segment_length_squared))

    # Calculate the nearest point on the line segment to the given point
    nearest_x = start_x + t * (end_x - start_x)
    nearest_y = start_y + t * (end_y - start_y)

    # Calculate the distance between the given point and the nearest point on the line segment
    distance = math.sqrt((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2)
    return distance


def distance_point_to_segment(p, v, w):
    # Function to compute the distance between a point (p) and a line segment defined by two endpoints (v, w).
    # The distance is calculated as the perpendicular distance from the point to the line segment.

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def vector_subtraction(v1, v2):
        return v1[0] - v2[0], v1[1] - v2[1]

    def vector_magnitude(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    vw = vector_subtraction(w, v)
    pv = vector_subtraction(p, v)
    pw = vector_subtraction(p, w)

    dot_product_pv = dot_product(pv, vw)
    dot_product_pw = dot_product(pw, vw)

    if dot_product_pv <= 0:
        return vector_magnitude(pv)
    elif dot_product_pw >= 0:
        return vector_magnitude(pw)
    else:
        # The point is between the two endpoints of the line segment,
        # so the distance is the perpendicular distance to the line.
        return abs(vw[0] * pv[1] - vw[1] * pv[0]) / vector_magnitude(vw)


def compute_minimum_dist(dist, Big_R):
    return np.sqrt(dist + Big_R)


def visualization(env, agents_path, agents_time):
    image_path = "C:/MS/Research/SLAM_Optimization/GNN_MAPF_yyz/images/"
    theta = np.linspace(0, 2 * np.pi, 150)
    radius = 0.01
    map = env.map
    map = map.astype(int)
    resolution = 2 / 15
    minimum_interval = 0.005
    largest_time = 0
    graph = env.graphs
    points = graph['points']
    agent_num = 0
    step = 0.05  # 0.002 #0.001
    agent_current_index={}
    map_obstalces = [
        [env._inverse_transform([i, j])[0] + (1 / env.width), env._inverse_transform([i, j])[1] + (1 / env.width),
         (1 / env.width) * 1.3] for j in range(env.map.shape[1]) for i in range(env.map.shape[0]) if env.map[i, j] == 1]
    for agent in agents_path.keys():
        time = agents_time[agent]
        agent_num = agent_num + 1
        agent_current_index[int(agent)]=0
        if np.max(time) > largest_time:
            largest_time = np.max(time)
    interval_num = int(largest_time / minimum_interval) + 1
    print(interval_num)
    for i in range(interval_num):
        print(i)
        figure, axes = plt.subplots(1)
        file_path = image_path + str(i) + ".png"
        current_time = minimum_interval * i
        for agent in agents_path.keys():
            path = agents_path[agent]
            time = agents_time[agent]
            if agent_current_index[int(agent)] == len(path) - 1:
                current_location = points[path[agent_current_index[int(agent)]]]
            else:
                while current_time < time[agent_current_index[int(agent)]]:
                    agent_current_index[int(agent)] = agent_current_index[int(agent)] - 1
                if agent_current_index[int(agent)] == len(path) - 1:
                    current_location = points[path[agent_current_index[int(agent)]]]
                else:
                    while current_time > time[agent_current_index[int(agent)] + 1]:
                        agent_current_index[int(agent)] = agent_current_index[int(agent)] + 1
                        if agent_current_index[int(agent)] == len(path) - 1:
                            break
                    if agent_current_index[int(agent)] == len(path) - 1:
                        current_location = points[path[agent_current_index[int(agent)]]]
                    else:
                        prev_location = points[path[agent_current_index[int(agent)]]]
                        next_location = points[path[agent_current_index[int(agent)] + 1]]
                        prev_time = time[agent_current_index[int(agent)]]
                        next_time = time[agent_current_index[int(agent)] + 1]
                        current_location = prev_location + (next_location - prev_location) * (
                                current_time - prev_time) / (next_time - prev_time)
            x = radius * np.cos(theta) + current_location[0]
            y = radius * np.sin(theta) + current_location[1]
            axes.plot(x, y)
        env.plot_polygon(env.occupied_area, ax=axes, alpha=1.0, fc='#253494', ec='#253494')
        # for j in range(len(map_obstalces)):
        # current_obstacle=map_obstalces[j]
        # current_radius=current_obstacle[2]
        # current_x=current_obstacle[1]+current_radius*np.cos(theta)
        # current_y=current_obstacle[0]+current_radius*np.sin(theta)
        # axes.plot(current_x,current_y,"b")
        # for j in range(15):
        # for k in range(15):
        # if map[j,k]==1:
        # center_x=-1+k*resolution
        # center_y=1-(j+1)*resolution
        # axes.add_patch(Rectangle((center_x,center_y),resolution,resolution))
        axes.set_aspect(1)
        plt.title('Visualization of the MAPF')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.savefig(file_path)
        plt.close()


class SIPP_Node:
    def __init__(self, state, time, safe_interval, fScore, gScore, interval_order,edge_order):
        self.state = state
        self.time = time
        self.parent = None
        self.safe_interval = safe_interval
        self.fScore = fScore
        self.gScore = gScore
        self.interval_order = interval_order
        self.edge_order=edge_order
        self.hScore=self.fScore-self.gScore
    def __eq__(self, other):
        if isinstance(other, SIPP_Node):
            return (self.state, self.safe_interval,self.edge_order) == (other.state, other.safe_interval,other.edge_order)
        return NotImplemented

    def __hash__(self):
        return hash((self.state, self.safe_interval,self.edge_order))



class node_interval:
    def __init__(self, interval, agent=None):
        self.interval = interval
        self.agent = agent


class collision_interval:
    def __init__(self, interval, agent=None):
        self.interval = interval
        self.agent = agent


def insert_collision_interval(collision_interval, interval):
    # print(collision_interval)
    # print(interval)
    #collision_interval_copy=copy.deepcopy(collision_interval)
    interval = np.reshape(interval, (2,))
    if interval[1]<interval[0] or interval[1]==interval[0]:
        return collision_interval
    interval_num = np.shape(collision_interval)[0]
    if interval_num < 1:
        return collision_interval
    if interval[1] <= collision_interval[0, 0]:
        collision_interval = np.insert(collision_interval, 0, interval, axis=0)
    elif interval[0] >= collision_interval[interval_num - 1, 1]:
        collision_interval = np.insert(collision_interval, interval_num, interval, axis=0)
    else:
        if interval[0]>=collision_interval[interval_num-1,0]:
            collision_interval[interval_num - 1, 1]=max(collision_interval[interval_num - 1, 1],interval[1])
            return collision_interval
        elif interval[0]<collision_interval[0,0]:
            if interval[1]<=collision_interval[0,1]:
                collision_interval[0,0]=interval[0]
            elif interval_num==1:
                collision_interval[0, 0] = interval[0]
                collision_interval[0, 1] = interval[1]
            elif interval[1]>=collision_interval[0,1] and interval[1]<=collision_interval[1,0]:
                collision_interval[0, 0] = interval[0]
                collision_interval[0, 1] = interval[1]
            else:
                collision_interval[0, 0] = interval[0]
                collision_interval[0, 1] = interval[1]
                count=0
                for r in range(1,interval_num):
                    if interval[1]>=collision_interval[r, 0]:
                        collision_interval[0,1]=max(collision_interval[r,1],interval[1])
                        count=count+1
                    else:
                        break
                for r in range(count):
                    collision_interval = np.delete(collision_interval, 1, 0)
            return collision_interval
        elif interval_num==1:
            return collision_interval
        elif interval_num==2:
            if interval[0] >= collision_interval[0, 0] and interval[0] <= collision_interval[1, 0]:
                selected_index = 0
            elif interval[1]>=collision_interval[1,1]:
                collision_interval[1,1]=interval[1]
                return collision_interval
            elif interval[1]<collision_interval[1,1]:
                return collision_interval
        else:
            low_index = 0
            high_index = interval_num - 2
            check_index=int((low_index+high_index+1)/2)
            termination_flag=False
            while termination_flag==False:
                if interval[0]>=collision_interval[check_index,0] and interval[0]<=collision_interval[check_index+1,0]:
                    selected_index=check_index
                    termination_flag=True
                    continue
                elif interval[0]<collision_interval[check_index,0]:
                    high_index=check_index
                    if high_index==low_index+1:
                        check_index=low_index
                    else:
                        check_index = int((low_index + high_index + 1) / 2)
                elif interval[0]>collision_interval[check_index+1,0]:
                    low_index=check_index
                    check_index = int((low_index + high_index + 1) / 2)
        i = selected_index
        if interval[0] >= collision_interval[i, 1] and interval[1] <= collision_interval[i + 1, 0]:

            collision_interval = np.insert(collision_interval, i + 1, interval, axis=0)
        elif interval[0] <= collision_interval[i, 1] and interval[1] <= collision_interval[i + 1, 0] and interval[1]>=collision_interval[i, 1]:
            collision_interval[i, 0] = min(collision_interval[i, 0], interval[0])
            collision_interval[i, 1] = interval[1]
        elif interval[0] >= collision_interval[i, 1] and interval[1] <= collision_interval[i + 1, 1]:
            collision_interval[i + 1, 0] = min(collision_interval[i + 1, 0],interval[0])
            collision_interval[i + 1, 1] = max(collision_interval[i + 1, 1], interval[1])
        elif interval[0] <= collision_interval[i, 1] and interval[1] > collision_interval[i + 1, 0]:
            collision_interval[i, 1] = max(collision_interval[i + 1, 1], interval[1])
            collision_interval[i, 0] = min(collision_interval[i, 0], interval[0])
            count = 1
            for j in range(i + 2, interval_num):
                if interval[1] >= collision_interval[j, 0]:
                    count = count + 1
                    collision_interval[i,1]=max(collision_interval[j,1],interval[1])
                else:
                    break

            for j in range(count):
                collision_interval = np.delete(collision_interval, i + 1, 0)
        elif interval[0] <= collision_interval[i+1, 0] and interval[1] > collision_interval[i + 1, 0] and interval[0] >= collision_interval[i, 1]:
            collision_interval[i+1, 1] = max(collision_interval[i + 1, 1], interval[1])
            collision_interval[i+1, 0] = interval[0]
            count = 0
            for j in range(i + 2, interval_num):
                if interval[1] >= collision_interval[j, 0]:
                    count = count + 1
                    collision_interval[i+1,1]=max(collision_interval[j,1],interval[1])
                else:
                    break
            #print("The count number")
            #print(count)
            for j in range(count):
                collision_interval = np.delete(collision_interval, i + 2, 0)
        elif interval[1]<=collision_interval[i,1]:
            return collision_interval
        else:
            print("error!")
            print(collision_interval)
            print(interval)
        #else:
            #print("error!")
            #print( collision_interval[i, :])
            #print( collision_interval[i+1, :])
            #print(interval)
    #error_flag = False
    #for i in range(0, np.shape(collision_interval)[0] - 1):
        #if collision_interval[i, 1] > collision_interval[i + 1, 0]:
            #error_flag=True
            #print(collision_interval[i, 1])
            #print(collision_interval[i + 1, 0])
            #break
    #if error_flag==True:
       # print("Here is a error!")
        #print(collision_interval_copy)
        #print(collision_interval)
        #print(interval)
    return collision_interval


def search_collision_interval(collision_interval, time,end_time,edge_time):
    edge_intervals_lists=[]
    interval_num = np.shape(collision_interval)[0]
    eps=0.0205
    end_time=end_time-eps
    if end_time<time:
        return edge_intervals_lists
    if time < collision_interval[0, 0]:
        if end_time<collision_interval[0, 0]:
            edge_interval = np.array([time, end_time])
            edge_intervals_lists.append(edge_interval)
        if end_time>=collision_interval[0, 0]:
            edge_interval = np.array([time, collision_interval[0, 0]])
            edge_intervals_lists.append(edge_interval)
            for i in range(interval_num-1):
                if collision_interval[i,1]<=end_time and collision_interval[i+1,0]>=end_time:
                    edge_intervals_lists.append(np.array([collision_interval[i,1],end_time]))
                if collision_interval[i,0]<=end_time<=collision_interval[i,1]:
                    break
                if collision_interval[i+1,1]<=end_time:
                    edge_intervals_lists.append(np.array([collision_interval[i, 1], collision_interval[i+1, 0]]))

    elif time >= collision_interval[interval_num - 1, 0] and collision_interval[interval_num - 1, 1] == np.inf:
        edge_interval = np.array([-np.inf, -np.inf])
        edge_intervals_lists.append(edge_interval)
    elif time > collision_interval[interval_num - 1, 1]:
        edge_interval = np.array([time,end_time])
        edge_intervals_lists.append(edge_interval)
    else:
        low_index = 0
        high_index = interval_num - 1
        check_index = int((low_index + high_index + 1) / 2)
        terminiate_flag = False
        while terminiate_flag == False:
            # print(low_index)
            # print(high_index)
            if check_index == interval_num - 1:
                if time >= collision_interval[check_index, 1]:
                    edge_interval = np.array([time, end_time])
                    terminiate_flag = True
                    continue
                elif collision_interval[check_index, 0] <= time <= collision_interval[check_index, 1]:
                    if end_time>collision_interval[check_index, 1]:
                        edge_interval = np.array([collision_interval[check_index, 1], end_time])
                    else:
                        edge_interval = np.array([-np.inf, -np.inf])
                    terminiate_flag = True
                    continue
                else:
                    high_index = check_index
                    check_index = int((low_index + check_index) / 2)
            else:
                if collision_interval[check_index, 0] <= time <= collision_interval[check_index, 1]:
                    if end_time>collision_interval[check_index, 1]:
                        if end_time<=collision_interval[check_index+1,0]:
                            edge_interval = np.array(
                                [collision_interval[check_index, 1], end_time])
                        else:
                            edge_interval = np.array(
                                [collision_interval[check_index, 1], collision_interval[check_index+1,0]])
                    else:
                        edge_interval = np.array([-np.inf, -np.inf])
                    terminiate_flag = True
                    continue
                elif collision_interval[check_index, 1] <= time <= collision_interval[check_index + 1, 0]:
                    if end_time>=collision_interval[check_index + 1, 0]:
                        edge_interval = np.array(
                            [time, collision_interval[check_index + 1, 0]])
                    else:
                        edge_interval = np.array(
                            [time, end_time])
                    terminiate_flag = True
                    continue
                else:
                    if time >= collision_interval[check_index + 1, 0]:
                        low_index = check_index
                        check_index = int((check_index + high_index + 1) / 2)
                    else:
                        high_index = check_index
                        check_index = int((low_index + check_index) / 2)
        if edge_interval[1]!=-np.inf:
            edge_intervals_lists.append(edge_interval)
            for i in range(check_index + 1, interval_num - 1):
                if collision_interval[i, 0] <= end_time <= collision_interval[i, 1]:
                    break
                if collision_interval[i, 1] <= end_time <= collision_interval[i + 1, 0]:
                    edge_intervals_lists.append(np.array([collision_interval[i, 1], end_time]))
                    break
                if collision_interval[
                    i + 1, 0] <= end_time:
                    edge_intervals_lists.append(np.array([collision_interval[i, 1], collision_interval[i + 1, 0]]))
    return edge_intervals_lists

def SIPP_PP(env, starts, goals, edges_conflicts_dict,nodes_edges_conflicts_dict, edges_nodes_conflicts_dict,edge_dict,backplanner=None):

    start_time = time.time()
    penalty_param = 1
    R = 0.02 * 0.02
    eps = 0.022
    bigger_eps=0.022
    agent_R = 0.01
    minimum_wait_time=0.022
    selected_neighborhood_num=4
    if backplanner is None:
        backplanner = env
    start_time = time.time()
    open_set = list()
    closed_set = list()
    initial_safe_interval = []
    infeasible_agents_list=[]
    initial_safe_interval.append([-np.inf, np.inf])
    agent_num = env.num_agents
    graph = env.graphs
    points = graph['points']
    node_num = np.shape(points)[0]
    nodes_safe_intervals = {}
    for i in range(node_num):
        nodes_safe_intervals[i] = copy.deepcopy(initial_safe_interval)
    agent_id = 0
    edge_cost = graph['edge_cost']
    edge_index = graph['edge_index']
    print(np.shape(graph['edge_index']))
    print(starts)
    print(goals)
    print(graph.keys())
    print(np.shape(graph['points']))
    print(agent_num)
    neighbors = graph['neighbors']
    priority_order = random.sample((np.arange(agent_num)).tolist(), agent_num)
    print(priority_order)
    high_priority_index = np.argmax(priority_order)
    current_start = starts[high_priority_index]
    current_end = goals[high_priority_index]
    heur = backplanner.distance(agent_id, current_start, current_end)
    heur = heur * penalty_param
    fScore = heur
    gScore = 0
    Node_start = SIPP_Node(state=current_start, time=0, fScore=fScore, gScore=gScore,
                           safe_interval=nodes_safe_intervals[current_start][0], interval_order=0,edge_order=0)
    open_set.append(Node_start)
    agents_time_list = {}
    path_dict = {}
    closed_dict = {}
    all_node_list = list()
    all_node_dict = {}
    node_expand_num = 0
    find_flag=False
    while len(open_set):
        node_expand_num = node_expand_num + 1
        best_node = min(open_set, key=lambda SIPP_Node: (SIPP_Node.fScore, -SIPP_Node.gScore))
        if best_node.state == current_end and best_node.safe_interval[1] == np.inf:
            find_flag=True
            break
        neighbor_num = len(neighbors[best_node.state])
        successors = list()
        # print(neighbor_num)
        for i in range(neighbor_num):
            neighbor_state = neighbors[best_node.state][i]
            if neighbor_state != best_node.state:
                m_time = edge_cost[best_node.state][i]
                start_t = best_node.time + m_time
                end_t = best_node.safe_interval[1] + m_time-bigger_eps
                heur = backplanner.distance(agent_id, neighbor_state, current_end)
                heur = heur * penalty_param
                interval_num = int((len(nodes_safe_intervals[neighbor_state]) + 1) / 2)
                for j in range(interval_num):
                    if nodes_safe_intervals[neighbor_state][2 * j][0] == np.inf:
                        continue
                    if nodes_safe_intervals[neighbor_state][2 * j][0]+eps >= end_t or \
                            nodes_safe_intervals[neighbor_state][2 * j][
                                1] <= start_t + eps:
                        continue
                    if start_t <= nodes_safe_intervals[neighbor_state][2 * j][0] + eps:
                        t = nodes_safe_intervals[neighbor_state][2 * j][0] + eps
                    else:
                        t = start_t
                    if t>=best_node.safe_interval[1] + m_time-bigger_eps:
                        continue
                    fScore = t + heur
                    New_Node = SIPP_Node(state=neighbor_state, time=t, fScore=fScore, gScore=t,
                                         safe_interval=nodes_safe_intervals[neighbor_state][2 * j], interval_order=j,edge_order=0)
                    New_Node.parent = best_node
                    successors.append(New_Node)
        # print(len(successors))
        # print(len(open_set))
        # print(len(closed_set))
        for i in range(len(successors)):
            node_key = str(successors[i].state) + "_" + str(successors[i].interval_order) + "_" + str(
                successors[i].edge_order)
            flag1 = False
            flag2 = False
            if node_key in closed_dict:
                flag2 = True
            # flag2=any(obj.state == successors[i].state and obj.safe_interval == successors[i].safe_interval for obj in closed_set)
            # for j in range(len(closed_set)):
            # if closed_set[j].state == successors[i].state and closed_set[j].safe_interval == successors[i].safe_interval:
            # flag2 = True
            # break
            if flag2 == False:
                if node_key in all_node_dict:
                    flag1 = True
                    existing_node = all_node_list[all_node_dict[node_key]]
                else:
                    all_node_dict[node_key] = len(all_node_list)
                    all_node_list.append(successors[i])
                # for j in range(len(open_set)):
                # if open_set[j].state == successors[i].state and open_set[j].safe_interval == successors[i].safe_interval:
                # existing_node=open_set[j]
                # flag1 = True
                # break
                if flag1 == True:
                    if existing_node.gScore > successors[i].gScore:
                        existing_node.gScore = successors[i].gScore
                        existing_node.fScore = successors[i].fScore
                        existing_node.hScore = successors[i].hScore
                        existing_node.time = successors[i].time
                        existing_node.parent = successors[i].parent
                        existing_node.safe_interval = successors[i].safe_interval
                        existing_node.interval_order = successors[i].interval_order
                        existing_node.edge_order = successors[i].edge_order
                else:
                    open_set.append(successors[i])
        node_key = str(best_node.state) + "_" + str(best_node.interval_order) + "_" + str(best_node.edge_order)
        closed_dict[node_key] = len(closed_set)
        closed_set.append(best_node)
        open_set.remove(best_node)
    last_time = best_node.time
    path = np.array([], dtype=int)
    path_time_list = np.array([])
    path = np.append(path, best_node.state)
    path_time_list = np.append(path_time_list, best_node.time)
    insert_index = nodes_safe_intervals[best_node.state].index(best_node.safe_interval)
    nodes_safe_intervals[best_node.state].remove(best_node.safe_interval)
    nodes_safe_intervals[best_node.state].insert(insert_index, [best_node.safe_interval[0], best_node.time])
    nodes_safe_intervals[best_node.state].insert(insert_index + 1, [best_node.time, np.inf])
    nodes_safe_intervals[best_node.state].insert(insert_index + 2, [np.inf, np.inf])
    #nodes_agent_record[best_node.state].append([high_priority_index,-1,best_node.time,np.inf])
    edge_dict = update_edge_intervals_from_points(points, best_node, best_node.time, np.inf, edge_dict,nodes_edges_conflicts_dict)
    # edge_dict = compute_minimum_distance(edges, points, best_node.state, best_node.state, best_node.time,
    # np.inf, edge_dict)
    best_node_copy = copy.deepcopy(best_node)
    while best_node.parent != None:
        insert_index = nodes_safe_intervals[best_node.parent.state].index(best_node.parent.safe_interval)
        index = neighbors[best_node.parent.state].index(best_node.state)
        leave_time = last_time - edge_cost[best_node.parent.state][index]
        travel_time=best_node.time-leave_time
        stay_time = leave_time - best_node.parent.time
        nodes_safe_intervals[best_node.parent.state].remove(best_node.parent.safe_interval)
        nodes_safe_intervals[best_node.parent.state].insert(insert_index,
                                                            [best_node.parent.safe_interval[0], best_node.parent.time])
        index = neighbors[best_node.parent.state].index(best_node.state)
        leave_time = last_time - edge_cost[best_node.parent.state][index]
        if  type(edge_dict[(best_node.state, best_node.parent.state)])==type(False):
            edge_dict[(best_node.state, best_node.parent.state)] = np.array([[max(0,leave_time-travel_time-0.02), best_node.time+0.02]])
        else:
            edge_dict[(best_node.state, best_node.parent.state)] = insert_collision_interval(
                edge_dict[(best_node.state, best_node.parent.state)], np.array([[max(0,leave_time-travel_time-0.02), best_node.time+0.02]]))
        if type(edge_dict[(best_node.parent.state, best_node.state)])==type(False):
            edge_dict[(best_node.parent.state, best_node.state)] = np.array([[leave_time, min(leave_time+minimum_wait_time,best_node.time)]])
        else:
            edge_dict[(best_node.parent.state, best_node.state)] = insert_collision_interval(
                edge_dict[(best_node.parent.state, best_node.state)], np.array([[leave_time, min(leave_time+minimum_wait_time,best_node.time)]]))

            # edge_dict[(best_node.parent.state, best_node.state)] = np.append(
            # edge_dict[(best_node.parent.state, best_node.state)], np.array([[leave_time, best_node.time]]),axis=0)
        edge_dict = update_edge_interval_from_edges(points, best_node.state, best_node.parent.state, leave_time,
                                             best_node.time, edge_dict,edges_conflicts_dict)
        #nodes_agent_record[best_node.parent.state].append([high_priority_index, best_node.state,best_node.parent.time,best_node.parent.time+stay_time])
        best_node = best_node.parent
        if stay_time < 0.01 * eps:
            path = np.append(path, best_node.state)
            path_time_list = np.append(path_time_list, best_node.time)
        else:
            path = np.append(path, best_node.state)
            path = np.append(path, best_node.state)
            path_time_list = np.append(path_time_list, best_node.time + stay_time)
            path_time_list = np.append(path_time_list, best_node.time)
            edge_dict = update_edge_intervals_from_points( points, best_node, best_node.time,
                                                            best_node.time + stay_time, edge_dict,nodes_edges_conflicts_dict)
            # edge_dict = compute_minimum_distance(edges, points, best_node.state, best_node.state, best_node.time,
            # best_node.time + stay_time, edge_dict)
        current_time = best_node.time
        last_time = current_time
        nodes_safe_intervals[best_node.state].insert(insert_index + 1, [best_node.time, leave_time])
        nodes_safe_intervals[best_node.state].insert(insert_index + 2, [leave_time, best_node.safe_interval[1]])
    last_time = best_node_copy.time
    # print(nodes_safe_intervals)
    while best_node_copy.parent != None:
        index = neighbors[best_node_copy.parent.state].index(best_node_copy.state)
        leave_time = last_time - edge_cost[best_node_copy.parent.state][index]
        selected_points = select_points(best_node_copy.state, best_node_copy.parent.state, neighbors,selected_neighborhood_num)
        nodes_safe_intervals, index = update_nodes_intervals_from_edges(points, best_node_copy.state,
                                                                        best_node_copy.parent.state,
                                                                        leave_time, best_node_copy.time,
                                                                        nodes_safe_intervals,edges_nodes_conflicts_dict)
        best_node_copy = best_node_copy.parent
        current_time = best_node_copy.time
        last_time = current_time
    path = np.flip(path)
    path_time_list = np.flip(path_time_list)
    agents_time_list[high_priority_index] = path_time_list
    path_dict[high_priority_index] = path
    priority_index = np.flip(np.argsort(priority_order))
    wait_count = 0
    #for node in nodes_agent_record:
        #if len(nodes_agent_record[node])>0:
            #print(nodes_agent_record[node])
    # for i in nodes_safe_intervals:
    # if len(nodes_safe_intervals[i])%2==0:
    # print("Something has wrong 3")
    # end_time = time.time()
    # print(path)
    # print(node_expand_num)
    # print(end_time - start_time)
    for x in range(1, np.shape(priority_index)[0]):
        print(x)
        find_dst_flag=False
        iteration_start = time.time()
        find_flag=False
        node_expand_num = 0
        # start_time = time.time()
        open_set = list()
        closed_set = list()
        closed_dict = {}
        current_start = starts[priority_index[x]]
        current_end = goals[priority_index[x]]
        initial_interval = nodes_safe_intervals[current_start][0]
        heur = backplanner.distance(agent_id, current_start, current_end)
        heur = heur * penalty_param
        fScore = heur
        gScore = 0
        Node_start = SIPP_Node(state=current_start, time=0, fScore=fScore, gScore=gScore,
                               safe_interval=initial_interval, interval_order=0,edge_order=0)
        # node_dict={}
        # node_key = str(Node_start.state) + "_" + str(Node_start.interval_order)
        # node_dict[node_key]=0
        open_set.append(Node_start)
        all_node_list = list()
        all_node_dict = {}
        while len(open_set):
            node_expand_num = node_expand_num + 1
            if find_dst_flag==False:
                best_node = min(open_set, key=lambda SIPP_Node: (SIPP_Node.fScore, -SIPP_Node.gScore))
            else:
                best_node = min(open_set, key=lambda SIPP_Node: (SIPP_Node.hScore,SIPP_Node.fScore, -SIPP_Node.gScore))
            if best_node.state == current_end and best_node.safe_interval[1] == np.inf and best_node.safe_interval[
                0] != np.inf:
                find_flag=True
                break
            neighbor_num = len(neighbors[best_node.state])
            successors = list()
            current_pos = points[best_node.state]
            for i in range(neighbor_num):
                neighbor_state = neighbors[best_node.state][i]
                if neighbor_state != best_node.state:
                    neighbor_pos = points[neighbor_state]
                    current_rel_pos = current_pos - neighbor_pos
                    best_node_eps=eps
                    m_time = edge_cost[best_node.state][i]
                    start_t = best_node.time + m_time
                    end_t = best_node.safe_interval[1] + m_time-bigger_eps
                    if end_t<start_t:
                        continue
                    heur = backplanner.distance(agent_id, neighbor_state, current_end)
                    heur = heur * penalty_param
                    interval_num = int((len(nodes_safe_intervals[neighbor_state]) + 1) / 2)
                    edge_flag = False
                    if  type(edge_dict[(best_node.state, neighbor_state)])!=type(False):
                        # collision_interval_num=np.shape(edge_dict[(best_node.state, neighbor_state)])[0]
                        edge_flag = True
                        edge_interval_lists = search_collision_interval(edge_dict[(best_node.state, neighbor_state)],
                                                                  best_node.time,best_node.safe_interval[1],m_time)
                        # edge_interval[0] = edge_interval[0] + eps
                        # print(edge_dict[(best_node.state, neighbor_state)])
                        # print(edge_interval)
                        # print(best_node.time)
                        # print(edge_dict[(best_node.state,neighbor_state)])
                        # print("edge")
                        # if edge_interval[0] > best_node.safe_interval[1] - eps:
                        # continue
                        # for j in range(collision_interval_num):
                        # if edge_dict[(best_node.state, neighbor_state)][j, 1] >= best_node.time > \
                        # edge_dict[(best_node.state, neighbor_state)][j, 0] :
                        # edge_interval=np.array([best_node.time,edge_dict[(best_node.state, neighbor_state)][j,0]])
                        # print(edge_interval)
                        # break
                    for j in range(interval_num):
                        if nodes_safe_intervals[neighbor_state][2 * j][0] == np.inf:
                            continue
                        current_eps=eps
                        if edge_flag == False:
                            if nodes_safe_intervals[neighbor_state][2 * j][0] >= end_t -current_eps or \
                                    nodes_safe_intervals[neighbor_state][2 * j][
                                        1] <= start_t +current_eps:
                                continue
                            if start_t < nodes_safe_intervals[neighbor_state][2 * j][0] + current_eps:
                                t = nodes_safe_intervals[neighbor_state][2 * j][0] + current_eps
                            else:
                                t = start_t
                            if t>nodes_safe_intervals[neighbor_state][2 * j][
                                        1]-current_eps or t>best_node.safe_interval[1] + m_time-current_eps:
                                continue
                            fScore = t + heur
                            New_Node = SIPP_Node(state=neighbor_state, time=t, fScore=fScore, gScore=t,
                                                 safe_interval=nodes_safe_intervals[neighbor_state][2 * j],
                                                 interval_order=j,edge_order=0)
                            New_Node.parent = best_node
                            successors.append(New_Node)
                        else:
                            edge_count=0
                            for edge_interval in edge_interval_lists:
                                if edge_interval[1] == -np.inf:
                                    continue
                                if edge_interval[0] > best_node.safe_interval[1]-bigger_eps:
                                    continue
                                if edge_interval[1] < best_node.time:
                                    continue
                                elif edge_interval[1]-edge_interval[0]<0.01:
                                    continue
                                else:
                                    start_t = np.max([start_t, edge_interval[0] + m_time])
                                    end_t = np.min([end_t, edge_interval[1]+m_time])
                                    if start_t>end_t:
                                        continue
                                    # print([start_t,end_t])
                                    # print("newline")
                                    if nodes_safe_intervals[neighbor_state][2 * j][0] >= end_t - current_eps or \
                                            nodes_safe_intervals[neighbor_state][2 * j][
                                                1] <= start_t + current_eps:
                                        continue
                                    if start_t <= nodes_safe_intervals[neighbor_state][2 * j][0] + current_eps:
                                        t = nodes_safe_intervals[neighbor_state][2 * j][0] + current_eps
                                    else:
                                        t = start_t
                                    if t > nodes_safe_intervals[neighbor_state][2 * j][
                                        1] - current_eps or t>best_node.safe_interval[1] + m_time-bigger_eps or t<best_node.safe_interval[0] + m_time+bigger_eps:
                                        continue
                                    fScore = t + heur
                                    # if t-m_time>best_node.time+eps:
                                    # print(start_t)
                                    # print(end_t)
                                    # print(best_node.time)
                                    # print([t-m_time,t])
                                    New_Node = SIPP_Node(state=neighbor_state, time=t, fScore=fScore, gScore=t,
                                                         safe_interval=nodes_safe_intervals[neighbor_state][2 * j],
                                                         interval_order=j,edge_order=edge_count)

                                    New_Node.parent = best_node
                                    successors.append(New_Node)
                                    edge_count=edge_count+1
            for i in range(len(successors)):
                node_key = str(successors[i].state) + "_" + str(successors[i].interval_order) + "_" + str(
                    successors[i].edge_order)
                flag1 = False
                flag2 = False
                if node_key in closed_dict:
                    flag2 = True
                # flag2 = any(
                # obj.state == successors[i].state and obj.safe_interval == successors[i].safe_interval for obj in
                # closed_set)
                # for j in range(len(closed_set)):
                # if closed_set[j].state == successors[i].state and closed_set[j].safe_interval == successors[
                # i].safe_interval:
                # flag2 = True
                # break
                if flag2 == False:
                    if node_key in all_node_dict:
                        flag1 = True
                        existing_node = all_node_list[all_node_dict[node_key]]
                    else:
                        all_node_dict[node_key] = len(all_node_list)
                        all_node_list.append(successors[i])
                    # for j in range(len(open_set)):
                    # if open_set[j].state == successors[i].state and open_set[j].safe_interval == successors[
                    # i].safe_interval:
                    # existing_node = open_set[j]
                    # flag1 = True
                    # break
                    if flag1 == True:
                        if existing_node.gScore > successors[i].gScore:
                            existing_node.gScore = successors[i].gScore
                            existing_node.fScore = successors[i].fScore
                            existing_node.hScore=successors[i].hScore
                            existing_node.time = successors[i].time
                            existing_node.parent = successors[i].parent
                            existing_node.safe_interval = successors[i].safe_interval
                            existing_node.interval_order = successors[i].interval_order
                            existing_node.edge_order=successors[i].edge_order
                    else:
                        if successors[i].state==current_end:
                            find_dst_flag=True
                        open_set.append(successors[i])
            node_key = str(best_node.state) + "_" + str(best_node.interval_order) + "_" + str(
                best_node.edge_order)
            closed_dict[node_key] = len(closed_set)
            closed_set.append(best_node)
            open_set.remove(best_node)
        if find_flag==True:
            #print(best_node.safe_interval)
            #iteration_start=time.time()
            # print(node_expand_num)
            last_time = best_node.time
            path = np.array([], dtype=int)
            path_time_list = np.array([])
            path = np.append(path, best_node.state)
            path_time_list = np.append(path_time_list, best_node.time)
            insert_index = nodes_safe_intervals[best_node.state].index(best_node.safe_interval)
            nodes_safe_intervals[best_node.state].remove(best_node.safe_interval)
            nodes_safe_intervals[best_node.state].insert(insert_index, [best_node.safe_interval[0], best_node.time])
            nodes_safe_intervals[best_node.state].insert(insert_index + 1, [best_node.time, np.inf])
            nodes_safe_intervals[best_node.state].insert(insert_index + 2, [np.inf, np.inf])

            #insert_nodes_record(nodes_agent_record[best_node.state],priority_index[x],-1,best_node.time,np.inf)
            best_node_copy = copy.deepcopy(best_node)
            # edge_dict = compute_minimum_distance(edges, points, best_node.state, best_node.state, best_node.time,
            # np.inf, edge_dict)
            edge_dict = update_edge_intervals_from_points(points, best_node, best_node.time, np.inf,
                                                            edge_dict,nodes_edges_conflicts_dict)
            while best_node.parent != None:
                index = neighbors[best_node.parent.state].index(best_node.state)
                leave_time = last_time - edge_cost[best_node.parent.state][index]
                travel_time=best_node.time-leave_time
                stay_time = leave_time - best_node.parent.time
                insert_mode=0
                #for node in nodes_safe_intervals:
                    #interval = nodes_safe_intervals[node]
                    #for t in range(len(interval) - 1):
                        #if interval[t][1] != interval[t + 1][0]:
                            #print("something wrong 1!")
                if best_node.parent.safe_interval not in nodes_safe_intervals[best_node.parent.state]:
                    #print("This node has already been checked")
                    insert_mode=1
                    #print(best_node.parent.safe_interval)
                    #print(nodes_safe_intervals[best_node.parent.state])
                    #print(safe_interval_copy[best_node.parent.state])
                    #print(best_node.time)
                    interval_num=int((len(nodes_safe_intervals[best_node.parent.state])+1)/2)
                    #print(interval_num)
                    for r in range(interval_num):
                        check_interval=nodes_safe_intervals[best_node.parent.state][2*r]
                        #print(check_interval)
                        if check_interval[0]<=best_node.time<=check_interval[1]:
                            nodes_safe_intervals[best_node.parent.state].remove(check_interval)
                            insert_index = 2 * r
                            break

                else:
                    insert_index = nodes_safe_intervals[best_node.parent.state].index(best_node.parent.safe_interval)
                    nodes_safe_intervals[best_node.parent.state].remove(best_node.parent.safe_interval)
                if insert_mode==0:
                    nodes_safe_intervals[best_node.parent.state].insert(insert_index,
                                                                        [best_node.parent.safe_interval[0],
                                                                         best_node.parent.time])
                else:
                    nodes_safe_intervals[best_node.parent.state].insert(insert_index,
                                                                        [check_interval[0],
                                                                         best_node.parent.time])
                if  type(edge_dict[(best_node.state, best_node.parent.state)])==type(False):
                    edge_dict[(best_node.state, best_node.parent.state)] = np.array([[max(0,leave_time-travel_time-0.02), best_node.time+0.02]])
                else:
                    edge_dict[(best_node.state, best_node.parent.state)] = insert_collision_interval(
                        edge_dict[(best_node.state, best_node.parent.state)], np.array([[max(0,leave_time-travel_time-0.02), best_node.time+0.02]]))
                if  type(edge_dict[(best_node.parent.state, best_node.state)])==type(False):
                    edge_dict[(best_node.parent.state, best_node.state)] = np.array(
                        [[leave_time, min(leave_time + minimum_wait_time, best_node.time)]])
                else:
                    edge_dict[(best_node.parent.state, best_node.state)] = insert_collision_interval(
                        edge_dict[(best_node.parent.state, best_node.state)],
                        np.array([[leave_time, min(leave_time + minimum_wait_time, best_node.time)]]))
                edge_dict= update_edge_interval_from_edges(points, best_node.state, best_node.parent.state,
                                                     leave_time,
                                                     best_node.time, edge_dict,edges_conflicts_dict)
                #print(nodes_safe_intervals[best_node.parent.state])
                #print(nodes_safe_intervals[best_node.parent.state])
                #print(best_node.parent.safe_interval)
                #print(best_node.parent.time)
                #print(best_node.time)
                #insert_nodes_record(nodes_agent_record[best_node.parent.state],priority_index[x],best_node.state,best_node.parent.time,best_node.parent.time+stay_time)
                best_node = best_node.parent
                if stay_time > eps + 0.00001:
                    wait_count = wait_count + 1
                if stay_time < 0.001 * eps:
                    path = np.append(path, best_node.state)
                    path_time_list = np.append(path_time_list, best_node.time)
                else:
                    edge_dict = update_edge_intervals_from_points( points, best_node, best_node.time,
                                                                    best_node.time + stay_time, edge_dict,nodes_edges_conflicts_dict)
                    # edge_dict = compute_minimum_distance(edges, points, best_node.state, best_node.state, best_node.time,
                    # best_node.time+stay_time, edge_dict)
                    path = np.append(path, best_node.state)
                    path = np.append(path, best_node.state)
                    path_time_list = np.append(path_time_list, best_node.time + stay_time)
                    path_time_list = np.append(path_time_list, best_node.time)
                current_time = best_node.time
                last_time = current_time
                if insert_mode==0:
                    nodes_safe_intervals[best_node.state].insert(insert_index + 1, [best_node.time, leave_time])
                    nodes_safe_intervals[best_node.state].insert(insert_index + 2,
                                                                 [leave_time, best_node.safe_interval[1]])
                else:
                    nodes_safe_intervals[best_node.state].insert(insert_index + 1, [best_node.time, leave_time])
                    nodes_safe_intervals[best_node.state].insert(insert_index + 2,
                                                                 [leave_time, check_interval[1]])

                #for node in nodes_safe_intervals:
                    #interval = nodes_safe_intervals[node]
                    #for t in range(len(interval) - 1):
                        #if interval[t][1] != interval[t + 1][0]:
                            #print("something wrong 2!")
            path = np.flip(path)
            #print(np.shape(path))
            path_time_list = np.flip(path_time_list)
            agents_time_list[priority_index[x]] = path_time_list
            path_dict[priority_index[x]] = path
            # print(path)
            # print(node_expand_num)
            # print(end_time - start_time)
            # for i in nodes_safe_intervals:
            # if len(nodes_safe_intervals[i]) % 2 == 0:
            # print("Something has wrong 3")
            last_time = best_node_copy.time
            while best_node_copy.parent != None:
                index = neighbors[best_node_copy.parent.state].index(best_node_copy.state)
                leave_time = last_time - edge_cost[best_node_copy.parent.state][index]
                selected_points = select_points(best_node_copy.state, best_node_copy.parent.state, neighbors,selected_neighborhood_num)
                nodes_safe_intervals, index = update_nodes_intervals_from_edges(points,
                                                                                best_node_copy.state,
                                                                                best_node_copy.parent.state,
                                                                                leave_time, best_node_copy.time,
                                                                                nodes_safe_intervals,edges_nodes_conflicts_dict)
                best_node_copy = best_node_copy.parent
                current_time = best_node_copy.time
                last_time = current_time
        else:
            print("cannot find a valid path!")
            infeasible_agents_list.append(priority_index[x])
            #print(nodes_safe_intervals[current_end])
            #for edge in edge_dict:
                #if edge[0]==current_end or edge[1]==current_end:
                    ##print(edge_dict[edge])

        #print(node_expand_num)
        #print(iteration_end - iteration_start)
    path_keys = list(path_dict)
    collision_count = 0
    # for node in nodes_safe_intervals:
    # if nodes_safe_intervals[node][-1] == [np.inf, np.inf]:
    # print(nodes_safe_intervals[node])
    # start_time=time.time()
    inter_collision_agents = []
    possible_collision_agents = []
    collision_dict = {}
    collision_edges=[]
    for i in range(1, len(path_keys)):
        current_agent_path = path_dict[path_keys[i]]
        current_agent_time = agents_time_list[path_keys[i]]
        current_agent_lasttime = current_agent_time[-1]
        current_agent_timenum = np.shape(current_agent_time)[0]
        for j in range(0, i):
            current_collision_num = 0
            another_agent_path = path_dict[path_keys[j]]
            another_agent_time = agents_time_list[path_keys[j]]
            another_agent_lastime = another_agent_time[-1]
            another_agent_timenum = np.shape(another_agent_time)[0]
            current_agent_timeindex = 0
            another_agent_timeindex = 0
            if current_agent_lasttime < another_agent_lastime:
                while (current_agent_timeindex < current_agent_timenum - 1):
                    current_node_start_pos = points[int(current_agent_path[current_agent_timeindex])]
                    current_node_end_pos = points[int(current_agent_path[current_agent_timeindex + 1])]
                    current_edge=(int(current_agent_path[current_agent_timeindex]),int(current_agent_path[current_agent_timeindex + 1]))
                    another_node_start_pos = points[int(another_agent_path[another_agent_timeindex])]
                    another_node_end_pos = points[int(another_agent_path[another_agent_timeindex + 1])]
                    another_edge=(int(another_agent_path[another_agent_timeindex]),int(another_agent_path[another_agent_timeindex + 1]))
                    if current_agent_time[current_agent_timeindex] <= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex] and current_agent_time[current_agent_timeindex + 1] <= \
                            another_agent_time[another_agent_timeindex + 1]:
                        overlap_interval = [another_agent_time[another_agent_timeindex],
                                            current_agent_time[current_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[0] - current_agent_time[current_agent_timeindex]) / (
                                                    current_agent_time[current_agent_timeindex + 1] -
                                                    current_agent_time[current_agent_timeindex])
                        current_end_pos = current_node_end_pos
                        another_start_pos = another_node_start_pos
                        another_end_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[1] - another_agent_time[another_agent_timeindex]) / (
                                                  another_agent_time[another_agent_timeindex + 1] -
                                                  another_agent_time[another_agent_timeindex])
                        current_agent_timeindex = current_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] <= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex + 1]:
                        overlap_interval = [another_agent_time[another_agent_timeindex],
                                            another_agent_time[another_agent_timeindex + 1]]
                        another_start_pos = another_node_start_pos
                        another_end_pos = another_node_end_pos
                        current_start_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[0] - current_agent_time[current_agent_timeindex]) / (
                                                    current_agent_time[current_agent_timeindex + 1] -
                                                    current_agent_time[current_agent_timeindex])
                        current_end_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[1] - current_agent_time[current_agent_timeindex]) / (
                                                  current_agent_time[current_agent_timeindex + 1] -
                                                  current_agent_time[current_agent_timeindex])
                        another_agent_timeindex = another_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] >= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] <= another_agent_time[
                        another_agent_timeindex + 1]:

                        overlap_interval = [current_agent_time[current_agent_timeindex],
                                            current_agent_time[current_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos
                        current_end_pos = current_node_end_pos
                        another_start_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[0] - another_agent_time[another_agent_timeindex]) / (
                                                    another_agent_time[another_agent_timeindex + 1] -
                                                    another_agent_time[another_agent_timeindex])
                        another_end_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[1] - another_agent_time[another_agent_timeindex]) / (
                                                  another_agent_time[another_agent_timeindex + 1] -
                                                  another_agent_time[another_agent_timeindex])
                        current_agent_timeindex = current_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] >= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex + 1]:
                        overlap_interval = [current_agent_time[current_agent_timeindex],
                                            another_agent_time[another_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos
                        another_end_pos = another_node_end_pos
                        current_end_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[1] - current_agent_time[current_agent_timeindex]) / (
                                                  current_agent_time[current_agent_timeindex + 1] -
                                                  current_agent_time[current_agent_timeindex])
                        another_start_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[0] - another_agent_time[another_agent_timeindex]) / (
                                                    another_agent_time[another_agent_timeindex + 1] -
                                                    another_agent_time[another_agent_timeindex])
                        another_agent_timeindex = another_agent_timeindex + 1

                    # A=current_start_pos[0]-another_start_pos[0]
                    # B=current_end_pos[0]-another_end_pos[0]
                    # C=current_start_pos[1]-another_start_pos[1]
                    # D=current_end_pos[1]-another_end_pos[1]
                    # a=A*A+B*B-2*A*B+C*C+D*D-2*C*D
                    # b=2*A*B-2*B*B+2*C*D-2*D*D
                    # c=B*B+D*D-R
                    para_a = another_start_pos[0] - current_start_pos[0]
                    para_c = another_start_pos[1] - current_start_pos[1]
                    para_b = another_end_pos[0] - another_start_pos[0] - current_end_pos[0] + current_start_pos[0]
                    para_d = another_end_pos[1] - another_start_pos[1] - current_end_pos[1] + current_start_pos[1]
                    a = para_b * para_b + para_d * para_d
                    b = -2 * para_a * para_b - 2 * para_c * para_d - 2 * para_b * para_b - 2 * para_d * para_d
                    c = para_a * para_a + para_c * para_c + para_b * para_b + para_d * para_d - R + 2 * para_a * para_b + 2 * para_c * para_d
                    possible_min1 = c
                    possible_min2 = a + b + c
                    collision_flag = False
                    if current_end_pos[0]==another_start_pos[0] and current_end_pos[1]==another_start_pos[1]:
                        continue
                    elif current_start_pos[0]==another_end_pos[0] and current_start_pos[1]==another_end_pos[1]:
                        continue
                    if a > 0:
                        if -b / 2 / a < 1 and -b / 2 / a > 0:
                            if b * b - 4 * a * c > 0:
                                if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                    current_collision_num = current_collision_num + 1
                                    collision_count = collision_count + 1
                                    inter_collision_agents.append([path_keys[i], path_keys[j]])
                                    collision_flag=True
                                    print(current_node_start_pos)
                                    print(current_node_end_pos)
                                    print(another_node_start_pos)
                                    print(another_node_end_pos)
                                    collision_edges.append([current_edge,another_edge])
                                    if current_node_end_pos[0] == another_start_pos[0] and current_node_end_pos[1] == \
                                            another_start_pos[1]:
                                        print(np.dot(current_node_start_pos - current_node_end_pos,
                                                     another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                            current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                            another_node_end_pos - current_node_end_pos)))
                        elif possible_min1 < 0:
                            if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                collision_count = collision_count + 1
                                inter_collision_agents.append([path_keys[i], path_keys[j]])
                                current_collision_num = current_collision_num + 1
                                collision_flag = True
                                print(current_node_start_pos)
                                print(current_node_end_pos)
                                print(another_node_start_pos)
                                print(another_node_end_pos)
                                collision_edges.append([current_edge, another_edge])
                                if current_node_end_pos[0] == another_start_pos[0] and current_node_end_pos[1] == \
                                        another_start_pos[1]:
                                    print(np.dot(current_node_start_pos - current_node_end_pos,
                                                 another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                        current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                        another_node_end_pos - current_node_end_pos)))
                        elif possible_min2 < 0:
                            if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                collision_count = collision_count + 1
                                inter_collision_agents.append([path_keys[i], path_keys[j]])
                                current_collision_num = current_collision_num + 1
                                collision_flag = True
                                print(current_node_start_pos)
                                print(current_node_end_pos)
                                print(another_node_start_pos)
                                print(another_node_end_pos)
                                collision_edges.append([current_edge, another_edge])
                                if current_node_end_pos[0] == another_start_pos[0] and current_node_end_pos[1] == \
                                        another_start_pos[1]:
                                    print(np.dot(current_node_start_pos - current_node_end_pos,
                                                 another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                        current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                        another_node_end_pos - current_node_end_pos)))
                    elif possible_min1 < 0:
                        if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                            collision_count = collision_count + 1
                            inter_collision_agents.append([path_keys[i], path_keys[j]])
                            current_collision_num = current_collision_num + 1
                            collision_flag = True
                            print(current_node_start_pos)
                            print(current_node_end_pos)
                            print(another_node_start_pos)
                            print(another_node_end_pos)
                            collision_edges.append([current_edge, another_edge])
                            if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                print(np.dot(current_node_start_pos - current_node_end_pos,
                                             another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                    current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                    another_node_end_pos - current_node_end_pos)))
                    elif possible_min2 < 0:
                        if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                            collision_count = collision_count + 1
                            inter_collision_agents.append([path_keys[i], path_keys[j]])
                            current_collision_num = current_collision_num + 1
                            collision_flag = True
                            print(current_node_start_pos)
                            print(current_node_end_pos)
                            print(another_node_start_pos)
                            print(another_node_end_pos)
                            collision_edges.append([current_edge, another_edge])
                            if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                print(np.dot(current_node_start_pos - current_node_end_pos,
                                             another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                    current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                    another_node_end_pos - current_node_end_pos)))
                    if collision_flag==True:
                        #print(current_node_start_pos)
                        #print(current_node_end_pos)
                        #print(another_node_start_pos)
                        #print(another_node_end_pos)
                        #print(current_start_pos)
                        #print(current_end_pos)
                        #print(another_start_pos)
                        #print(another_end_pos)
                        continue

            else:
                while (another_agent_timeindex < another_agent_timenum - 1):
                    current_node_start_pos = points[int(current_agent_path[current_agent_timeindex])]
                    current_node_end_pos = points[int(current_agent_path[current_agent_timeindex + 1])]
                    current_edge=(int(current_agent_path[current_agent_timeindex]),int(current_agent_path[current_agent_timeindex + 1]))
                    another_node_start_pos = points[int(another_agent_path[another_agent_timeindex])]
                    another_node_end_pos = points[int(another_agent_path[another_agent_timeindex + 1])]
                    another_edge=(int(another_agent_path[another_agent_timeindex]),int(another_agent_path[another_agent_timeindex + 1]))
                    if current_agent_time[current_agent_timeindex] <= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex] and current_agent_time[current_agent_timeindex + 1] <= \
                            another_agent_time[another_agent_timeindex + 1]:
                        overlap_interval = [another_agent_time[another_agent_timeindex],
                                            current_agent_time[current_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[0] - current_agent_time[current_agent_timeindex]) / (
                                                    current_agent_time[current_agent_timeindex + 1] -
                                                    current_agent_time[current_agent_timeindex])
                        current_end_pos = current_node_end_pos
                        another_start_pos = another_node_start_pos
                        another_end_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[1] - another_agent_time[another_agent_timeindex]) / (
                                                  another_agent_time[another_agent_timeindex + 1] -
                                                  another_agent_time[another_agent_timeindex])
                        current_agent_timeindex = current_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] <= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex + 1]:
                        overlap_interval = [another_agent_time[another_agent_timeindex],
                                            another_agent_time[another_agent_timeindex + 1]]
                        another_start_pos = another_node_start_pos
                        another_end_pos = another_node_end_pos
                        current_start_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[0] - current_agent_time[current_agent_timeindex]) / (
                                                    current_agent_time[current_agent_timeindex + 1] -
                                                    current_agent_time[current_agent_timeindex])
                        current_end_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[1] - current_agent_time[current_agent_timeindex]) / (
                                                  current_agent_time[current_agent_timeindex + 1] -
                                                  current_agent_time[current_agent_timeindex])
                        another_agent_timeindex = another_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] >= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] <= another_agent_time[
                        another_agent_timeindex + 1]:

                        overlap_interval = [current_agent_time[current_agent_timeindex],
                                            current_agent_time[current_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos
                        current_end_pos = current_node_end_pos
                        another_start_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[0] - another_agent_time[another_agent_timeindex]) / (
                                                    another_agent_time[another_agent_timeindex + 1] -
                                                    another_agent_time[another_agent_timeindex])
                        another_end_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[1] - another_agent_time[another_agent_timeindex]) / (
                                                  another_agent_time[another_agent_timeindex + 1] -
                                                  another_agent_time[another_agent_timeindex])
                        current_agent_timeindex = current_agent_timeindex + 1
                    elif current_agent_time[current_agent_timeindex] >= another_agent_time[another_agent_timeindex] and \
                            current_agent_time[current_agent_timeindex + 1] >= another_agent_time[
                        another_agent_timeindex + 1]:
                        overlap_interval = [current_agent_time[current_agent_timeindex],
                                            another_agent_time[another_agent_timeindex + 1]]
                        current_start_pos = current_node_start_pos
                        another_end_pos = another_node_end_pos
                        current_end_pos = current_node_start_pos + (current_node_end_pos - current_node_start_pos) * (
                                overlap_interval[1] - current_agent_time[current_agent_timeindex]) / (
                                                  current_agent_time[current_agent_timeindex + 1] -
                                                  current_agent_time[current_agent_timeindex])
                        another_start_pos = another_node_start_pos + (another_node_end_pos - another_node_start_pos) * (
                                overlap_interval[0] - another_agent_time[another_agent_timeindex]) / (
                                                    another_agent_time[another_agent_timeindex + 1] -
                                                    another_agent_time[another_agent_timeindex])
                        another_agent_timeindex = another_agent_timeindex + 1
                    para_a = another_start_pos[0] - current_start_pos[0]
                    para_c = another_start_pos[1] - current_start_pos[1]
                    para_b = another_end_pos[0] - another_start_pos[0] - current_end_pos[0] + current_start_pos[0]
                    para_d = another_end_pos[1] - another_start_pos[1] - current_end_pos[1] + current_start_pos[1]
                    a = para_b * para_b + para_d * para_d
                    b = -2 * para_a * para_b - 2 * para_c * para_d - 2 * para_b * para_b - 2 * para_d * para_d
                    c = para_a * para_a + para_c * para_c + para_b * para_b + para_d * para_d - R + 2 * para_a * para_b + 2 * para_c * para_d
                    possible_min1 = c
                    possible_min2 = a + b + c
                    collision_flag = False
                    if current_end_pos[0]==another_start_pos[0] and current_end_pos[1]==another_start_pos[1]:
                        continue
                    elif current_start_pos[0]==another_end_pos[0] and current_start_pos[1]==another_end_pos[1]:
                        continue
                    if a > 0:
                        if -b / 2 / a < 1 and -b / 2 / a > 0:
                            if b * b - 4 * a * c > 0:
                                if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                    collision_count = collision_count + 1
                                    inter_collision_agents.append([path_keys[i], path_keys[j]])
                                    current_collision_num = current_collision_num + 1
                                    collision_flag=True
                                    print(current_node_start_pos)
                                    print(current_node_end_pos)
                                    print(another_node_start_pos)
                                    print(another_node_end_pos)
                                    collision_edges.append([current_edge, another_edge])
                                    if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                        print(np.dot(current_node_start_pos-current_node_end_pos,another_node_end_pos-current_node_end_pos)/(np.linalg.norm(current_node_start_pos-current_node_end_pos)*np.linalg.norm(another_node_end_pos-current_node_end_pos)))
                        elif possible_min1 < 0:
                            if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                collision_count = collision_count + 1
                                inter_collision_agents.append([path_keys[i], path_keys[j]])
                                current_collision_num = current_collision_num + 1
                                collision_flag=True
                                print(current_node_start_pos)
                                print(current_node_end_pos)
                                print(another_node_start_pos)
                                print(another_node_end_pos)
                                collision_edges.append([current_edge, another_edge])
                                if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                    print(np.dot(current_node_start_pos - current_node_end_pos,
                                                 another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                        current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                        another_node_end_pos - current_node_end_pos)))
                        elif possible_min2 < 0:
                            if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                                collision_count = collision_count + 1
                                inter_collision_agents.append([path_keys[i], path_keys[j]])
                                current_collision_num = current_collision_num + 1
                                collision_flag=True
                                print(current_node_start_pos)
                                print(current_node_end_pos)
                                print(another_node_start_pos)
                                print(another_node_end_pos)
                                collision_edges.append([current_edge, another_edge])
                                if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                    print(np.dot(current_node_start_pos - current_node_end_pos,
                                                 another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                        current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                        another_node_end_pos - current_node_end_pos)))
                    elif possible_min1 < 0:
                        if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                            collision_count = collision_count + 1
                            inter_collision_agents.append([path_keys[i], path_keys[j]])
                            current_collision_num = current_collision_num + 1
                            collision_flag = True
                            print(current_node_start_pos)
                            print(current_node_end_pos)
                            print(another_node_start_pos)
                            print(another_node_end_pos)
                            collision_edges.append([current_edge, another_edge])
                            if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                print(np.dot(current_node_start_pos - current_node_end_pos,
                                             another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                    current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                    another_node_end_pos - current_node_end_pos)))
                    elif possible_min2 < 0:
                        if [path_keys[i], path_keys[j]] not in inter_collision_agents:
                            collision_count = collision_count + 1
                            inter_collision_agents.append([path_keys[i], path_keys[j]])
                            current_collision_num = current_collision_num + 1
                            collision_flag = True
                            print(current_node_start_pos)
                            print(current_node_end_pos)
                            print(another_node_start_pos)
                            print(another_node_end_pos)
                            collision_edges.append([current_edge, another_edge])
                            if current_node_end_pos[0]==another_start_pos[0] and current_node_end_pos[1]==another_start_pos[1]:
                                print(np.dot(current_node_start_pos - current_node_end_pos,
                                             another_node_end_pos - current_node_end_pos) / (np.linalg.norm(
                                    current_node_start_pos - current_node_end_pos) * np.linalg.norm(
                                    another_node_end_pos - current_node_end_pos)))
                    if collision_flag==True:
                        #print(current_node_start_pos)
                        #print(current_node_end_pos)
                        #print(another_node_start_pos)
                        #print(another_node_end_pos)
                        #print(current_start_pos)
                        #print(current_end_pos)
                        #print(another_start_pos)
                        #print(another_end_pos)
                        continue
            collision_dict[(path_keys[i], path_keys[j])] = current_collision_num
    end_time = time.time()
    print(len(path_dict.keys()))
    print(end_time - start_time)
    print(len(inter_collision_agents))
    print(infeasible_agents_list)
    time_cost = 0
    for i in agents_time_list.keys():
        time_cost = time_cost + agents_time_list[i][-1]
    print(time_cost)
    new_path_dict={}
    new_agents_time_list={}
    for i in range(len(inter_collision_agents)):
        #print(path_dict[inter_collision_agents[i][0]])
        #print(agents_time_list[inter_collision_agents[i][0]])
        #print(path_dict[inter_collision_agents[i][1]])
        #print(agents_time_list[inter_collision_agents[i][1]])
        if inter_collision_agents[i][0] not in new_path_dict:
            new_path_dict[inter_collision_agents[i][0]]=path_dict[inter_collision_agents[i][0]]
            new_agents_time_list[inter_collision_agents[i][0]]=agents_time_list[inter_collision_agents[i][0]]
        if inter_collision_agents[i][1] not in new_path_dict:
            new_path_dict[inter_collision_agents[i][1]]=path_dict[inter_collision_agents[i][1]]
            new_agents_time_list[inter_collision_agents[i][1]]=agents_time_list[inter_collision_agents[i][1]]
    for node in nodes_safe_intervals:
        safe_inv_length=len(nodes_safe_intervals[node])
        for i in range(safe_inv_length-1):
            if abs(nodes_safe_intervals[node][i][1]-nodes_safe_intervals[node][i+1][0])>=0.00001:
                print("something is wrong! 1")
    for edge_pair in edge_dict.keys():
        current_edge_interval=edge_dict[edge_pair]
        if type(current_edge_interval)==type(False):
            continue
        interval_num=np.shape(current_edge_interval)[0]
        if interval_num==1:
            continue
        else:
            for i in range(interval_num-1):
                if current_edge_interval[i,1]>current_edge_interval[i+1,0]:
                    print("something is wrong! 2")
    for edge_pair in collision_edges:
        #print(edges_conflicts_dict[edge_pair[0]])
        print(edge_pair[1])
        for conflict in edges_conflicts_dict[edge_pair[0]]:
            another_edge=conflict[0]
            if edge_pair[1][0]==another_edge[0] and edge_pair[1][1]==another_edge[1]:
                print(conflict)
    return path_dict,agents_time_list

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Start the experiment agent")
    config_setup = add_default_argument_and_parse(arg_parser, 'experiment')
    config_setup = process_config(config_setup)
    config_setup.env_name = "MazeEnv"
    env = MazeEnv(config_setup)
    env.init_new_problem_graph(index=0)
    env.init_new_problem_instance(index=0)
    edges_conflicts_dict, nodes_edges_conflicts_dict, edges_nodes_conflicts_dict,edges_dict = precompute_conflicts(env)
    path_dict, agents_time_list = SIPP_PP(env, env.agent_starts_vidxs, env.agent_goals_vidxs,edges_conflicts_dict,nodes_edges_conflicts_dict, edges_nodes_conflicts_dict,edges_dict)
    visualization(env,path_dict,agents_time_list)
    # SIPP_time=time.time()
    # result = CBSPlanner().result = CBSPlanner().plan(env, env.agent_starts_vidxs, env.agent_goals_vidxs)
    # print(result.solution)