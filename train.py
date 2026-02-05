'''

'''

import os
import sys
import torch
import logging
import argparse
import numpy as np

from torch import nn
from torch.utils import tensorboard

from stldm import *
from stldm.config import *
# Library Issue
from data import dutils
import utilspp as utpp
from utilspp import SequentialLR, warmup_lambda
from data.config import SEVIR_13_12, HKO7_5_20, METEONET_5_20
from data.loader import GET_TrainLoader
from data.dutils import resize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset related
    parser.add_argument('-d', '--dataset', type=str, default='', help='Dataset config to be trained')
    parser.add_argument('--seq_len', type=int, default=10, help='The input sequence length')
    parser.add_argument('--out_len', type=int, default=10, help='The output (prediction) sequence length') 
    # model related
    parser.add_argument('-f', dest='checkpt', type=str, default='', help='model checkpoint to be loaded from (Empty = not loading)')
    parser.add_argument('-o', '--output', type=str, default='ckpts', help='The output directory')
    parser.add_argument('-m', '--model', type=str, default='', help='The global configuration to be used (The var name in config.py)')
    parser.add_argument('--type', type=str, default='3D', help='Determine which kind of model to use, 2D or 3D')
    # Training Components Related
    parser.add_argument('--ae_ckpt', type=str, default=None, help='Pre-trained AE checkpoint, freeze it during training')
    parser.add_argument('--ae_eval', action='store_false', help='Set AE to be trainable')
    parser.add_argument('--back_ckpt', type=str, default=None, help='Pre-trained backbone checkpoint, freeze it during traing')
    parser.add_argument('--back_eval', action='store_false', help='Set Backbone to be trainable')
    parser.add_argument('--set_mu_to_0', action='store_false', help='Set the constraint loss to 0')
    # hyperparams
    parser.add_argument('--lr', type=float, default=0.0001, help='The initial learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('-s', "--training_steps", type=int, default=200000, help="number of training steps")
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='The batch size')
    parser.add_argument('--micro_batch', type=int, default=1, help='Micro Batch size')
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    parser.add_argument('--validate_every', type=int, default=5, help='The number of steps to perform validation once')
    parser.add_argument('--v_steps', type=int, default=50, help='Validation steps')    
    parser.add_argument('--remarks', type=str, default='', help='This section will affect the model name to be saved')
    parser.add_argument('--save_every_epoch', action='store_true', help='Save ckpt for every validation epochs, otherwise save the best')
    args = parser.parse_args()

    # args validation
    assert args.model != '', 'You must specify the model config using -m/--model!'

    # read the model config
    dataset_config = globals()[args.dataset]
    dataset_type = dataset_config['savedir']
    dataset_param, dataset_meta = dataset_config['param'], dataset_config['meta']

    model_config = globals()[args.model]
    model_type =  model_config['model']
    save_path = utpp.build_model_path(args.output, dataset_type, model_type, timestamp=True) + args.remarks
    os.makedirs(save_path, exist_ok=True)
    img_size = model_config['vp_param']['shape_in'][-1]
    # prepare dataloader
    total_seq_len = args.seq_len + args.out_len
    
    if dataset_type.startswith('meteo'):
        train_iter, validate_iter = GET_TrainLoader(dataset_meta, dataset_param, args.batch_size, args.seq_len, args.out_len)
        train_loader, valid_loader = iter(train_iter), iter(validate_iter)
    else:
        train_loader, valid_loader = GET_TrainLoader(dataset_meta, dataset_param, args.batch_size, args.seq_len, args.out_len)

    if dataset_type.startswith('sevir'):
        steps_per_epoch = len(train_loader)
        epochs = args.epoch
    elif dataset_type.startswith('hko'):
        steps_per_epoch = 2500
        epochs = args.training_steps // steps_per_epoch
    elif dataset_type.startswith('meteo'):
        steps_per_epoch = len(train_loader)
        epochs = args.training_steps // steps_per_epoch
    else:
        raise Exception(f'Undefined dataset config name: {dataset_type}')
    total_training_steps = epochs * steps_per_epoch

    # define the model
    model_param = model_config['param']
    model_pathname = utpp.build_model_name(model_type, model_param)
    setattr(args, 'step', total_training_steps)

    # prepare logger
    logfile_name = os.path.join(save_path, f'_log.log')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    logging.info(f'args: {args}')
    logging.info('The resulting model will be saved as: {}'.format(os.path.join(save_path, model_pathname)))
    logging.info(f'Training Epochs: {epochs} and Total Training Steps: {total_training_steps}')
    # Writing logs for tensorboard
    log_dir = os.path.join(save_path, 'logs')
    writer = tensorboard.SummaryWriter(log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setattr(args, 'device', device)
    assert args.type in ['2D', '3D'], 'Please specify either 2D or 3D'
    model = n2n_setup[args.type](model_config).to(device)

    assert args.ae_ckpt!=args.back_ckpt or (args.ae_ckpt is None and args.back_ckpt is None), 'Please specify from End to End (set both to None), LDM only (set args.back_ckpt), or LDM + Meta (set args.ae_ckpt)'
    # Load Pre-trained AutoEncoder
    if args.ae_ckpt is not None:
        try:
            data = torch.load(args.ae_ckpt)
            model.backbone.vae.load_state_dict(data)
            model.backbone.vae.requires_grad_(args.ae_eval)
            logging.info(f'Load pre-trained AE well, Set require grads to be {args.ae_eval}')
        except:
            logging.info('Failed to load pre-trained AE')
    
    if args.back_ckpt is not None:
        try:
            model.backbone.load_state_dict(torch.load(args.back_ckpt, map_location=torch.device(device)))
            model.backbone.requires_grad_(args.back_eval)
            logging.info(f'Load pre-trained backbone well, Set require grads to be {args.back_eval}')
        except:
            logging.info('Failed to load pre-trained backbone')

    if args.checkpt != '':
        try:
            model.load_state_dict(torch.load(args.checkpt, map_location=torch.device(device)))
        except:
            logging.error("Loading weights failed")

    logging.info(f'Set require grads of VAE to be {args.ae_eval}')
    logging.info(f'Set require grads of backbone to be {args.back_eval}')

    # The original methods in the NeurIPS 2015 paper
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    warmup_iter = 2000
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=0.1))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_training_steps - warmup_iter)//args.micro_batch, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])

    best_val_loss = 1e10
    total_step = 0
    for epoch in range(1, epochs+1):
        if dataset_type.startswith('sevir'):
            train_loader.reset()
        elif dataset_type.startswith('meteo'):
            train_loader = iter(train_iter)

        for step in range(steps_per_epoch):
            total_step += 1
            model.train()
            optimizer.zero_grad()

            if args.ae_eval == False:
                model.backbone.vae.eval()
            
            if args.back_eval == False:
                model.backbone.eval()

            if dataset_type == 'sevir':
                data = train_loader.sample(batch_size=args.batch_size)
                x, y = data['vil'][:, :args.seq_len], data['vil'][:, args.seq_len:]
            elif dataset_type.startswith('meteo'):
                data = next(train_loader)
                x, y = data
            elif dataset_type.startswith('hko'):
                x_seq, x_mask, dt_clip, _ = train_loader.sample(batch_size=args.batch_size)
                x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args)

            x, y = x.to(device), y.to(device)
            if x.shape[-1] != img_size:
                x, y = resize(x, img_size), resize(y, img_size)
            if model_config['pre'] is not None:
                x = model_config['pre'](x)
                y = model_config['pre'](y)
            
            recon_loss, mu_loss, diff_loss, prior_loss = model.compute_loss(x, y)
            loss = (recon_loss + mu_loss + diff_loss + prior_loss)
            loss.backward()

            if total_step% args.micro_batch == 0:
                if args.back_ckpt is None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            # -----------------------------------------------------
            # On Step End
            # -----------------------------------------------------
            # terminal log every {print_every} steps.
            if total_step == 1 or total_step % args.print_every == 0:            
                logging.info(f'[Epoch {epoch}][Step {step}] recon_loss: {float(recon_loss):.4}, mu_loss: {float(mu_loss):.4}, diff_loss: {float(diff_loss):.4}')
            writer.add_scalar('Training recon_loss', float(recon_loss), global_step=total_step)
            writer.add_scalar('Training mu_loss', float(mu_loss), global_step=total_step)
            writer.add_scalar('Training diff_loss', float(diff_loss), global_step=total_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=total_step)

        # validate every {validate_every} epochs
        if epoch == 1 or epoch % args.validate_every == 0:
            rand_batch = np.random.randint(min(args.batch_size, 8))
            if dataset_type == 'sevir' or dataset_type.startswith('hko'):
                valid_loader.reset()
            elif dataset_type.startswith('meteo'):
                valid_loader = iter(validate_iter)

            acc_ae, acc_diff, acc_mu = 0, 0, 0
            for v_step in range(args.v_steps):
                model.eval()

                if dataset_type == 'sevir':
                    data = valid_loader.sample(batch_size=args.batch_size)
                    x, y = data['vil'][:, :args.seq_len], data['vil'][:, args.seq_len:]
                elif dataset_type.startswith('meteo'):
                    data = next(valid_loader)
                    x, y = data
                elif dataset_type.startswith('hko'):
                    x_seq, x_mask, dt_clip, _ = valid_loader.sample(batch_size=args.batch_size)
                    x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args)
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    if x.shape[-1] != img_size:
                        x, y = resize(x, img_size), resize(y, img_size)
                    if model_config['pre'] is not None:
                        x = model_config['pre'](x)
                        y = model_config['pre'](y)
                    ae_loss, mu_loss, diff_loss, _ = model.compute_loss(x, y, validate=True)
                    acc_ae += ae_loss
                    acc_diff += diff_loss
                    acc_mu += mu_loss
                    if model_config['post'] is not None:
                        x = model_config['post'](x)
                        y = model_config['post'](y)
                
            
            acc_ae, acc_mu, acc_diff = acc_ae/args.v_steps, acc_mu/args.v_steps, acc_diff/args.v_steps
            writer.add_scalar('Val AE loss', float(acc_ae), global_step=total_step)
            writer.add_scalar('Val VP loss', float(acc_mu), global_step=total_step)
            writer.add_scalar('Val Diff loss', float(acc_diff), global_step=total_step)
            logging.info(f'[Epoch {epoch}][Validation] AE_loss:{float(acc_ae):.4}, VP_loss:{float(acc_mu):.4}, Diff_loss:{float(acc_diff):.4}')
            val_loss = (acc_mu+acc_diff)/2
            
            with torch.no_grad():
                if model_config['pre'] is not None:
                    x = model_config['pre'](x)
                y_pred, mu = model(x, include_mu=True)
                if model_config['post'] is not None:
                    y_pred = model_config['post'](y_pred)
                    mu = model_config['post'](mu)
                    x = model_config['post'](x)

            out_x, out_y, mu_pred, out_y_pred = x[rand_batch].unsqueeze(0), y[rand_batch].unsqueeze(0), mu[rand_batch].unsqueeze(0), y_pred[rand_batch].unsqueeze(0)
            utpp.torch_visualize({'input': out_x, 'ground truth': out_y, 'mu_pred': mu_pred, 'predicted': out_y_pred}, savedir=os.path.join(save_path, f'temp-{total_step}.png'), vmin=0, vmax=1)

            if args.save_every_epoch:
                torch.save(model.state_dict(), os.path.join(save_path, f'{model_pathname}_epoch={epoch}.pt'))
            else:
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), os.path.join(save_path, f'{model_pathname}_best.pt'))
                    best_val_loss = val_loss

    torch.save(model.state_dict(), os.path.join(save_path, f'{model_pathname}_final.pt'))
