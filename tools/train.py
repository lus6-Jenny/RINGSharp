# Zhejiang University

import argparse
import torch

from trainer import do_train
from glnet.utils.params import TrainingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--val', dest='val', action='store_true')    
    parser.add_argument('--weight', type=str, required=False, help='Path to the model file')
    
    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Experiment name: {}'.format(args.exp_name))
    print('Debug mode: {}'.format(args.debug))
    print('Resume: {}'.format(args.resume))
    print('Validation: {}'.format(args.val))    
    if args.resume:
        print('Resume from: {}'.format(args.weight))
    else:
        args.weight = None
    
    params = TrainingParams(args.config, args.model_config)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    do_train(params, resume=args.resume, exp_name=args.exp_name, debug=args.debug, weight=args.weight, do_val=args.val, device=device)
