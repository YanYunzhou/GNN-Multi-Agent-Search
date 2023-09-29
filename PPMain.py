from simulator.maze_env import MazeEnv
import torch
from configs.str2config import str2config, add_default_argument_and_parse
import argparse
from configs.str2config import str2config, add_default_argument_and_parse
from utils.config import *

def main():
    arg_parser = argparse.ArgumentParser(description="Start the experiment agent")
    config_setup = add_default_argument_and_parse(arg_parser, 'experiment')
    config_setup = process_config(config_setup)
    config_setup.env_name = "MazeEnv"
    env = MazeEnv(config_setup)
    env.init_new_problem_graph(index=0)
    env.init_new_problem_instance(index=0)
