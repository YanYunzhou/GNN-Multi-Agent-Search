import numpy as np
import random

class PriorityPlanner:
    def __init__(self):
        self.explored_nodes = []
    def plan(self, env, starts, goals):
        agent_num=env.num_agents
        graph=env.graphs
        map=env.map
        points=graph['points']
        edge_cost=graph['edge_cost']
        neighbors=graph['neighbors']
        edge_index=graph['edge_index']
        priority_order=random.sample((np.arange(agent_num)).tolist(), agent_num)
        high_priority_index=np.argmax(priority_order)
        current_start=starts[high_priority_index]
        current_end=goals[high_priority_index]
        print(current_start)
        print(points[current_start])
        print(edge_cost[current_start])
        print(neighbors[current_start])
        print(priority_order)
        print(np.shape(edge_index))
        print(starts)
        print(np.shape(starts))
        print(goals)
        print(np.shape(goals))