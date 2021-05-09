import os
import argparse
import yaml
from pprint import pprint
from deca_solver import deca_solver

def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(
        description='PyTorch DECA solver.')
    new_parser.add_argument('--config', default='configs/config.yaml')
    new_parser.add_argument('--save_dict_path', default='snapshot/')
    new_parser.add_argument('--save_mesh_path', default='results/')
    new_parser.add_argument('--resume', type=str, default=None)
    new_parser.add_argument('--visualize', action='store_true')
    new_parser.add_argument('--image_path', type=str, default=None)
    new_parser.add_argument('--output_path', type=str, default=None)

    new_parser.add_argument('--gpus', type=str, default='0,1')
    # exclusive arguments
    group = new_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train_coarse', action='store_true')
    group.add_argument('--train_detail', action='store_true')
    
    group.add_argument('--evaluate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--count_op', action='store_true')
    group.add_argument('--convert', action='store_true')
    group.add_argument('--convert_nart', action='store_true')

    return new_parser.parse_args()

def main():
    # parse args and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
        
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    agent = deca_solver(config)

    if args.train_coarse:
        agent = deca_solver(config, mode='train_coarse')
        agent.trainval()
    elif args.train_detail:
        agent = deca_solver(config, mode='train_detail')
        agent.trainval()
    elif args.test:
        agent = deca_solver(config, mode='test')
        agent.test()
    else:
        raise Warning('Invalid mode')

if __name__ == '__main__':
    main()