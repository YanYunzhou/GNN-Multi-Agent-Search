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
from SIPPPlanner_Naive_Precompute import *
from poly_point_isect import *
from fixed_radius_near_neighbors import *
minimum_checking_time=0

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False
def compute_edgesToedges_conflicts(edges, points,check_edges_pairs,edges_collision_dict):
    dst_threshold = 0.0212
    eps = 0
    R = 0.01
    for pair in check_edges_pairs:
        first_edge=edges[pair[0]]
        second_edge=edges[pair[1]]
        if first_edge==second_edge:
            continue
        parent_node=first_edge[0]
        current_node=first_edge[1]
        check_parent_node=second_edge[0]
        check_current_node=second_edge[1]
        current_start_pos = points[parent_node]
        current_end_pos = points[current_node]
        V1 = (current_end_pos - current_start_pos) / np.linalg.norm((current_end_pos - current_start_pos))
        V1_len=np.linalg.norm((current_end_pos - current_start_pos))
        check_start_pos = points[check_parent_node]
        check_end_pos = points[check_current_node]
        minimum_dist = distance_between_segments(current_start_pos, current_end_pos, check_start_pos,
                                                 check_end_pos)
        flag1 = False
        flag2 = False
        flag3 = False
        flag4 = False
        flag5 = False
        flag6 = False
        flag7 = False
        flag8 = False
        if minimum_dist<=dst_threshold or (do_intersect(current_start_pos,current_end_pos,check_start_pos,check_end_pos)==True):
            # case 1
            V2 = check_end_pos - check_start_pos
            V2_len=np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_start_pos
            P_delta = current_start_pos - check_start_pos
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold

            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag1 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(parent_node,current_node)].append([(check_parent_node,check_current_node),(min_time,max_time),delay_range,[A,B,D]])
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if current_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif current_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif parent_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
                else:
                    print(first_edge)
                    print(second_edge)
            # case 2
            V2 = check_start_pos - check_end_pos
            V2_len=np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_end_pos
            P_delta = P1 - P2
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag2 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(parent_node, current_node)].append(
                        [(check_current_node, check_parent_node), (min_time, max_time),delay_range,[A,B,D]])
                    #print(min_time)
                    #print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if current_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif current_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
            current_start_pos = points[current_node]
            current_end_pos = points[parent_node]
            V1 = (current_end_pos - current_start_pos) / np.linalg.norm((current_end_pos - current_start_pos))
            V1_len=np.linalg.norm((current_end_pos - current_start_pos))
            check_start_pos = points[check_parent_node]
            check_end_pos = points[check_current_node]
            # case 3
            V2 = check_end_pos - check_start_pos
            V2_len=np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_start_pos
            P_delta = current_start_pos - check_start_pos
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag3 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(current_node, parent_node)].append(
                        [(check_parent_node, check_current_node), (min_time, max_time),delay_range,[A,B,D]])
                    #print(min_time)
                    #print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if parent_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif current_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
            # case 4
            V2 = check_start_pos - check_end_pos
            V2_len=np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_end_pos
            P_delta = P1 - P2
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag4 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)

                    edges_collision_dict[(current_node, parent_node)].append(
                        [(check_current_node, check_parent_node), (min_time, max_time),delay_range,[A,B,D]])


                    #print(min_time)
                    #print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif parent_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif current_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
        first_edge=edges[pair[1]]
        second_edge=edges[pair[0]]
        parent_node = first_edge[0]
        current_node = first_edge[1]
        check_parent_node = second_edge[0]
        check_current_node = second_edge[1]
        current_start_pos = points[parent_node]
        current_end_pos = points[current_node]
        V1 = (current_end_pos - current_start_pos) / np.linalg.norm((current_end_pos - current_start_pos))
        V1_len=np.linalg.norm((current_end_pos - current_start_pos))
        check_start_pos = points[check_parent_node]
        check_end_pos = points[check_current_node]
        minimum_dist = distance_between_segments(current_start_pos, current_end_pos, check_start_pos,
                                                 check_end_pos)
        if minimum_dist <= dst_threshold or (do_intersect(current_start_pos,current_end_pos,check_start_pos,check_end_pos)==True ):
            # case 1
            V2 = check_end_pos - check_start_pos
            V2_len= np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_start_pos
            P_delta = current_start_pos - check_start_pos
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag5 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(parent_node, current_node)].append(
                        [(check_parent_node, check_current_node), (min_time, max_time),delay_range,[A,B,D]])
            else:
                if parent_node == check_parent_node or parent_node == check_current_node or current_node == check_parent_node or current_node == check_current_node:
                    if current_node == check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len - 2 * R, V1_len + 2 * R),
                             [V1_len - 2 * R, V1_len + 2 * R], [A, B, D]])
                    elif current_node == check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len + V1_len + 2 * R], [A, B, D]])
                    elif parent_node == check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
            # case 2
            V2 = check_start_pos - check_end_pos
            V2_len = np.linalg.norm(V2)
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_end_pos
            P_delta = P1 - P2
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag6 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(parent_node, current_node)].append(
                        [(check_current_node, check_parent_node), (min_time, max_time),delay_range,[A,B,D]])
                    # print(min_time)
                    # print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if current_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif current_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
            current_start_pos = points[current_node]
            current_end_pos = points[parent_node]
            V1 = (current_end_pos - current_start_pos) / np.linalg.norm((current_end_pos - current_start_pos))
            check_start_pos = points[check_parent_node]
            check_end_pos = points[check_current_node]
            # case 3
            V2 = check_end_pos - check_start_pos
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_start_pos
            P_delta = current_start_pos - check_start_pos
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag7 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    edges_collision_dict[(current_node, parent_node)].append(
                        [(check_parent_node, check_current_node), (min_time, max_time),delay_range,[A,B,D]])
                    # print(min_time)
                    # print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if parent_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif current_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
            # case 4
            V1 = (current_end_pos - current_start_pos) / np.linalg.norm((current_end_pos - current_start_pos))
            V2 = check_start_pos - check_end_pos
            V2 = V2 / np.linalg.norm(V2)
            V_delta = V1 - V2
            P1 = current_start_pos
            P2 = check_end_pos
            P_delta = P1 - P2
            A = np.dot(V_delta, V_delta)
            B = 2 * (np.dot(V1, V1) - np.dot(V1, V2))
            C = np.dot(V1, V1)
            D = 2 * np.dot(V1 - V2, P1 - P2)
            E = 2 * np.dot(V1, P_delta)
            F = np.dot(P_delta, P_delta) - dst_threshold * dst_threshold
            if abs(np.dot(V1, V2)) <= 1 - 0.000000001:
                if (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                        D * D - 4 * A * F) < 0:
                    flag8 = True
                else:
                    center = (B * D - 2 * A * E) / (4 * A * C - B * B)
                    delay_delta = np.sqrt(
                        (2 * B * D - 4 * A * E) * (2 * B * D - 4 * A * E) + 4 * (4 * A * C - B * B) * (
                                D * D - 4 * A * F)) / (2 * (4 * A * C - B * B))
                    delay_range = [center - delay_delta, center + delay_delta]
                    collision_time_1 = (-B * delay_range[0] - D) / (2 * A)
                    collision_time_2 = (-B * delay_range[1] - D) / (2 * A)
                    min_time = min(collision_time_1, collision_time_2)
                    max_time = max(collision_time_1, collision_time_2)
                    #if parent_node==check_current_node:
                        #current_rel_pos = points[parent_node] - points[current_node]
                        #current_rel_pos = current_rel_pos / np.linalg.norm(current_rel_pos)
                        #check_rel_pos = points[check_parent_node] - points[check_current_node]
                        #check_rel_pos = check_rel_pos / np.linalg.norm(check_rel_pos)
                        #score = np.dot(current_rel_pos, check_rel_pos)
                    edges_collision_dict[(current_node, parent_node)].append(
                        [(check_current_node, check_parent_node), (min_time, max_time),delay_range,[A,B,D]])
                    # print(min_time)
                    # print(max_time)
            else:
                if parent_node==check_parent_node or parent_node==check_current_node or current_node==check_parent_node or current_node==check_current_node:
                    if parent_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (V1_len-2*R,V1_len+2*R), [V1_len-2*R,V1_len+2*R], [A, B, D]])
                    elif parent_node==check_parent_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-V2_len - 2 * R, -V2_len+V1_len+2*R], [A, B, D]])
                    elif current_node==check_current_node:
                        edges_collision_dict[(parent_node, current_node)].append(
                            [(check_parent_node, check_current_node), (- 2 * R, V1_len + 2 * R),
                             [-2 * R, 2 * R], [A, B, D]])
    return edges_collision_dict

def select_edges_zero_level(points,edge_end, edge_start, neighbors,edge_index_dict):
    edges_set = []
    new_points = []
    all_points=[edge_start,edge_end]
    edges_set.append((edge_end,edge_start))
    select_index_list=[]
    for neigh in neighbors[edge_start]:
        #relative_pos=points[edge_end]-points[edge_start]
        if neigh != edge_start and neigh != edge_end:
            #check_rel_pos=points[neigh]-points[edge_start]
            edges_set.append((edge_start, neigh))
    for neigh in neighbors[edge_end]:
        #relative_pos = points[edge_start] - points[edge_end]
        if neigh != edge_end and neigh != edge_start:
            #check_rel_pos = points[neigh] - points[edge_end]
            edges_set.append((edge_end, neigh))
    for i in range(len(edges_set)):
        select_index_list.append(edge_index_dict[edges_set[i]])
    return select_index_list

def precompute_conflicts(env):
    #data = (
        #((0.000000, 1.000000), (1.000000, 0.000000)),
        #((0.000000, 0.000000), (1.000000, 1.000000)),
    #)
    #intersection=isect_segments_include_segments(data, validate=True)
    #intersection=intersection[0]
    #print(intersection[0])
    #print(intersection[1])

    minimum_w = -1
    maximum_w = 1
    robot_R = 0.01
    dst_threshold = 0.0212
    minimum_interval = 2 * robot_R
    cell_width_num = int((maximum_w - minimum_w) / minimum_interval)
    graph = env.graphs
    points = graph['points']
    node_num = np.shape(points)[0]
    neighbors = graph['neighbors']
    total_cell_num = cell_width_num * cell_width_num
    cells_list = []
    for i in range(total_cell_num):
        cells_list.append([])
    edge_index = graph['edge_index']
    edges = []
    edges_dict={}
    edge_index_dict = {}
    edges_pos_dict={}
    edges_data=[]
    edge_count = 0
    edges_collision_dict={}
    nodes_edges_conflicts_dict={}
    for i in range(node_num):
        nodes_edges_conflicts_dict[i]=[]
    edges_nodes_conflicts_dict={}
    for i in range(np.shape(edge_index)[0]):
        check_edge = edge_index[i, :]
        if check_edge[0] != check_edge[1] and (check_edge[1], check_edge[0]) not in edges and (check_edge[0], check_edge[1]) not in edges:
            edges.append((check_edge[0], check_edge[1]))
            edge_index_dict[(check_edge[0], check_edge[1])] = edge_count
            edge_index_dict[(check_edge[1], check_edge[0])] = edge_count
            edges_dict[(check_edge[0], check_edge[1])] = False
            edges_dict[(check_edge[1], check_edge[0])] = False
            edges_collision_dict[(check_edge[0], check_edge[1])]=[]
            edges_collision_dict[(check_edge[1], check_edge[0])] = []
            point1_list=points[check_edge[0]].tolist()
            point2_list=points[check_edge[1]].tolist()
            point1_list[0]=round(point1_list[0],ndigits=3)
            point1_list[1] = round(point1_list[1], ndigits=3)
            point2_list[0] = round(point2_list[0], ndigits=3)
            point2_list[1] = round(point2_list[1], ndigits=3)
            edges_data.append((tuple(point1_list),tuple(point2_list)))
            edges_pos_dict[(tuple(point1_list),tuple(point2_list))]=edge_count
            edges_pos_dict[(tuple(point2_list), tuple(point1_list))] = edge_count
            edge_count = edge_count + 1
    edges_data=tuple(edges_data)
    #intersection=[]
    #intersection = isect_segments_include_segments(edges_data, validate=True)
    #print(len(intersection))
    check_edges_pair=[]
    #print(edges_pos_dict)
    for i in range(edge_count):
        current_edge=edges[i]
        edges_nodes_conflicts_dict[current_edge]=[]
        edges_nodes_conflicts_dict[(current_edge[1],current_edge[0])]=[]
    #for i in range(len(intersection)):
        #current_intersection=intersection[i]
        #check_edges_pair.append((edges_pos_dict[current_intersection[1][0]],edges_pos_dict[current_intersection[1][1]]))
    check_edges_pair_copy=copy.deepcopy(check_edges_pair)
    select_edges_pair_part1,nodes_edges_pairs,edges_to_edges_dict=fixed_radius_near_neighborhoods_edges(env)
    start_time=time.time()
    for pair in nodes_edges_pairs:
        node=pair[0]
        edge_num=pair[1]
        edge=edges[edge_num]
        edge_endpoint1=points[edge[0]]
        edge_endpoint2=points[edge[1]]
        node_pos=points[node]
        dist=distance_point_to_line_segment(node_pos[0],node_pos[1],edge_endpoint1[0],edge_endpoint1[1],edge_endpoint2[0],edge_endpoint2[1])
        if dist<dst_threshold:
            found_points = find_points_at_distance_r(node_pos,edge_endpoint1,edge_endpoint2,dst_threshold)
            nodes_edges_conflicts_dict[node].append((edge,found_points))
            nodes_edges_conflicts_dict[node].append(((edge[1],edge[0]), found_points))
            edges_nodes_conflicts_dict[(edge[0],edge[1])].append((node,found_points))
            edges_nodes_conflicts_dict[(edge[1], edge[0])].append((node, found_points))
    for i in range(len(select_edges_pair_part1)):
        if select_edges_pair_part1[i] not in check_edges_pair_copy and (select_edges_pair_part1[i][1],select_edges_pair_part1[i][0]) not in check_edges_pair_copy:
            check_edges_pair.append(select_edges_pair_part1[i])
    end_time=time.time()
    print("phase 6")
    print(end_time-start_time)
    start_time = time.time()
    check_edges_list_part2=[]
    for i in range(edge_count):
        edge_start=edges[i][0]
        edge_end=edges[i][1]
        edges_set=select_edges_zero_level(points,edge_start,edge_end,neighbors,edge_index_dict)
        edges_set.remove(i)
        for j in range(len(edges_set)):
            if (edges_set[j],i) not in check_edges_list_part2 and (edges_set[j],i) not in edges_to_edges_dict and (i,edges_set[j]) not in edges_to_edges_dict:
                check_edges_list_part2.append((i,edges_set[j]))
    for i in range(len(check_edges_list_part2)):
        check_edges_pair.append(check_edges_list_part2[i])
    end_time = time.time()
    print("phase 7")
    print(end_time - start_time)
    #start_time=time.time()
    #for edge_pair in check_edges_pair:
        #edge1=edge_pair[0]
        #edge2=edge_pair[1]
       # print("point1")
        #print(points[edges[edge1][0]])
        #print(points[edges[edge1][1]])
        #print("point2")
        #print(points[edges[edge2][0]])
        #print(points[edges[edge2][1]])
    start_time = time.time()
    edges_conflicts_dict=compute_edgesToedges_conflicts(edges,points,check_edges_pair,edges_collision_dict)
    end_time = time.time()
    print("phase 8")
    print(end_time - start_time)
    #end_time=time.time()
    return edges_conflicts_dict,nodes_edges_conflicts_dict,edges_nodes_conflicts_dict,edges_dict

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Start the experiment agent")
    config_setup = add_default_argument_and_parse(arg_parser, 'experiment')
    config_setup = process_config(config_setup)
    config_setup.env_name = "MazeEnv"
    env = MazeEnv(config_setup)
    env.init_new_problem_graph(index=0)
    env.init_new_problem_instance(index=0)
    edges_conflicts_dict,nodes_edges_conflicts_dict,edges_nodes_conflicts_dict,edges_dict=precompute_conflicts(env)