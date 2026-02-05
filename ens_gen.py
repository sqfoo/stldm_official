"""
Sample Command
"""
import os, sys, logging, argparse
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from stldm import *
from stldm.config import *
import utilspp as utpp
from data.config import SEVIR_13_12, HKO7_5_20, METEONET_5_20
from data.loader import GET_TestLoader
from data.dutils import resize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset related
    parser.add_argument('-d', '--dataset', type=str, default='', help='the dataset definition to be set')
    # model related
    parser.add_argument('-f', dest='checkpt', type=str, default='', help='model checkpoint to be loaded from (Empty = not loading)')
    parser.add_argument('-m', '--model', type=str, default='', help='the model definition to be created')
    parser.add_argument('--type', type=str, default='3D', help='Determine which kind of model to use, 2D or 3D')
    parser.add_argument('--c_str', type=float, default=0.0, help='CFG strength')
    parser.add_argument('--e_id', type=int, default=0, help='Ensemble ID')
    # hyperparams
    parser.add_argument('-s', '--step', type=int, default=-1, help='The number of steps to run. -1: the entire dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    parser.add_argument('-o', '--output', default=None, help='The path to save the log files')
    args = parser.parse_args()

    # prepare logger
    if args.output is None:
        path_list = args.checkpt.split("/")
        logfile_name = os.path.join(*path_list[:-1], 'logs', f'{path_list[-1][:-3]}.log')
    else:
        logfile_name = f'{args.output}.log'
    logging.basicConfig(level=logging.NOTSET, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    logging.info(f'Model checkpoint: {args.checkpt}')
    logging.info(f'Steps: {args.step}')

    sampler_dir = os.path.join(*logfile_name.split("/")[:-2], f'CFG={args.c_str}_samples')
    os.makedirs(sampler_dir, exist_ok=True)

    # Prepare Dataloader
    dataset_config = globals()[args.dataset]
    dataset_param, dataset_meta = dataset_config['param'], dataset_config['meta']
    loader = GET_TestLoader(dataset_meta, dataset_param, args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare Model
    assert args.type in ['2D', '3D'], 'Please specify either 2D or 3D'
    model_config = globals()[args.model]
    model = n2n_setup[args.type](model_config, print_info=True, cfg_str=args.c_str if args.c_str != 0.0 else None).to(device)
    logging.info(f'CFG Scheduler: Const-{args.c_str}')
    
    data = torch.load(args.checkpt, map_location=device)
    if 'model' in data.keys():
        model.load_state_dict(data['model'])
    else:
        model.load_state_dict(data)
    
    
    in_len, out_len = model_config['vp_param']['shape_in'][0], model_config['vp_param']['shape_out'][0]
    img_size = model_config['vp_param']['shape_in'][-1]
    
    step = 0
    out = []
    while args.step < 0 or step <=args.step:
        model.eval()

        if dataset_meta['dataset'] == 'HKO-7':  
            setattr(args, 'seq_len', in_len)
            try:
                data = loader.sample(batch_size=args.batch_size)
            except Exception as e:
                logging.error(e)
                break
            x_seq, x_mask, dt_clip, _ = data                  
            x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args)
        elif dataset_meta['dataset'] == 'SEVIR':
            data = loader.sample(batch_size=args.batch_size)
            if data is None:
                break
            x, y = data['vil'][:, :in_len], data['vil'][:, in_len:]
        elif dataset_meta['dataset'].startswith('meteo'):
            try:
                x, y = next(loader)
            except Exception as e:
                logging.error(e)
                break
                
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if x.shape[-1] != img_size:
                x = resize(x, img_size)
                y = resize(y, img_size) # TO compare with DiffCast paper
            if model_config['pre'] is not None:
                x = model_config['pre'](x)

            y_pred = model(x)
                
            if model_config['post'] is not None:
                x = model_config['post'](x)
                y_pred = model_config['post'](y_pred)
            y_pred = y_pred.clamp(0,1)
            
            out.append(y_pred.detach().cpu())

        step += 1
        # log/print every
        if step == 1 or step % args.print_every == 0:
            logging.info(f'{step} Steps Generated, {len(out)} in out array')  
        
    logging.info(f'{step} Steps Generated, {len(out)} in out array')  
    out = torch.cat(out, dim=0)
    out = out.numpy()
    save_path = os.path.join(sampler_dir, f'BTCHW_total-no:{len(out)}_e={args.e_id}.npy')
    np.save(save_path, out)
    print('Output saved in', save_path)