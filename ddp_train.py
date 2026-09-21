import os
import sys
import torch
import logging
import argparse
import numpy as np

from torch.utils import tensorboard

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from stldm import *
from stldm.config import *

from data import dutils
import utilspp as utpp
from utilspp import SequentialLR, warmup_lambda
from data.config import SEVIR_13_12, HKO7_5_20, METEONET_5_20
from data.loader import GET_TrainLoader
from data.dutils import resize

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def ddp_train(rank, world_size, arguments):
    # create default process group
    setup(rank, world_size)

    ## Setup Dataloader
    dataset_config = globals()[arguments.dataset]
    dataset_type = dataset_config['savedir']
    dataset_param, dataset_meta = dataset_config['param'], dataset_config['meta']
    dataset_param['num_shard'] = world_size
    dataset_param['rank'] = rank

    seq_len, out_len = dataset_meta['seq_len'], dataset_meta['out_len']

    if dataset_type.startswith('meteo'):
        torch.manual_seed(42 + rank)
        train_iter, validate_iter = GET_TrainLoader(dataset_meta, dataset_param, arguments.local_batch_size, seq_len, out_len)
        train_loader, valid_loader = iter(train_iter), iter(validate_iter)
    else:
        train_loader, valid_loader = GET_TrainLoader(dataset_meta, dataset_param, arguments.local_batch_size, seq_len, out_len)

    if dataset_type.startswith('sevir'):
        steps_per_epoch = len(train_loader)
        epochs = arguments.epoch
    elif dataset_type.startswith('hko'):
        steps_per_epoch = 2500
        epochs = arguments.training_steps // steps_per_epoch
    elif dataset_type.startswith('meteo'):
        steps_per_epoch = len(train_loader)
        epochs = arguments.training_steps // steps_per_epoch
    else:
        raise Exception(f'Undefined dataset config name: {dataset_type}')
    total_training_steps = epochs * steps_per_epoch
    setattr(arguments, 'total_training_steps', total_training_steps)

    ## Setup Logger
    model_config = globals()[arguments.model]
    model_param = model_config['param']
   
    model_type =  model_config['model']
    save_path = utpp.build_model_path(arguments.output, dataset_config['savedir'], model_type, timestamp=True) + arguments.remarks
    os.makedirs(save_path, exist_ok=True)
    model_pathname = utpp.build_model_name(model_type, model_param)
    logfile_name = os.path.join(save_path, f'_log.log')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    if rank == 0: 
        logging.info(f'args: {arguments}')
        logging.info('The resulting model will be saved as: {}'.format(os.path.join(save_path, model_pathname)))
        logging.info(f'Total Training Epochs: {epochs} Total Training Steps: {arguments.total_training_steps}')
    # Writing logs for tensorboard
    log_dir = os.path.join(save_path, 'logs')
    writer = tensorboard.SummaryWriter(log_dir)

    ## Setup Model
    img_size = model_config['vp_param']['shape_in'][-1]
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    model = n2n_setup[arguments.type](model_config).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Load Pre-trained Checkpoint
    if arguments.ae_ckpt is not None:
        try:
            data = torch.load(arguments.ae_ckpt)
            model.backbone.vae.load_state_dict(data)
            model.backbone.vae.requires_grad_(arguments.ae_eval)
            if rank == 0:
                logging.info(f'Load pre-trained AE well, Set require grads to be {arguments.ae_eval}')
        except:
            if rank == 0:
                logging.info('Failed to load pre-trained AE')
    
    if arguments.back_ckpt is not None:
        try:
            model.backbone.load_state_dict(torch.load(arguments.back_ckpt, map_location=torch.device(device)))
            model.backbone.requires_grad_(arguments.back_eval)
            if rank == 0:
                logging.info(f'Load pre-trained backbone well, Set require grads to be {arguments.back_eval}')
        except:
            if rank == 0:
                logging.info('Failed to load pre-trained backbone')

    if arguments.checkpt != '':
        try:
            model.load_state_dict(torch.load(arguments.checkpt, map_location=torch.device(device)))
        except:
            if rank == 0:
                logging.error("Loading weights failed")
    
    if rank == 0:
        logging.info(f'Set require grads of VAE to be {arguments.ae_eval}')
        logging.info(f'Set require grads of backbone to be {arguments.back_eval}')

    ## Optimizer and Scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=arguments.lr)
    warmup_iter = 2000
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=0.1))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_training_steps - warmup_iter), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])

    total_step = 0
    best_val_loss = 1e10
    for epoch in range(1, epochs + 1):
        if dataset_type.startswith('sevir'):
            train_loader.reset()
        elif dataset_type.startswith('meteo'):
            train_loader = iter(train_iter)

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            try:
                if dataset_type == 'sevir':
                    data = train_loader.sample(batch_size=arguments.local_batch_size)
                    x, y = data['vil'][:, :seq_len], data['vil'][:, seq_len:]
                elif dataset_type.startswith('meteo'):
                    data = next(train_loader)
                    x, y = data
                elif dataset_type.startswith('hko'):
                    x_seq, x_mask, dt_clip, _ = train_loader.sample(batch_size=arguments.local_batch_size)
                    x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, arguments)
            except:
                continue
            
            x, y = x.to(rank), y.to(rank)
            if x.shape[-1] != img_size:
                x, y = resize(x, img_size), resize(y, img_size)
            recon_loss, mu_loss, diff_loss, prior_loss = model.compute_loss(x, y) # AE_loss, vp_loss, diff_loss
            loss = (recon_loss + mu_loss + diff_loss + prior_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            if rank == 0 and (total_step == 1 or total_step % arguments.print_every == 0):
                logging.info(f'[Epoch {epoch}][Step {step}] recon_loss: {float(recon_loss):.4}, mu_loss: {float(mu_loss):.4}, diff_loss: {float(diff_loss):.4}')
            
            if rank == 0:
                writer.add_scalar('Training recon_loss', float(recon_loss), global_step=total_step)
                writer.add_scalar('Training mu_loss', float(mu_loss), global_step=total_step)
                writer.add_scalar('Training diff_loss', float(diff_loss), global_step=total_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=total_step)
            
        dist.barrier() # Synchronize
        # Validation    
        if rank==0 and (epoch==1 or epoch%arguments.validate_every==0):
            rand_batch = np.random.randint(min(arguments.local_batch_size, 8))

            if dataset_type == 'sevir' or dataset_type.startswith('hko'):
                valid_loader.reset()
            elif dataset_type.startswith('meteo'):
                valid_loader = iter(validate_iter)

            acc_ae, acc_diff, acc_mu = 0, 0, 0
            for v_step in range(arguments.v_steps):
                ddp_model.eval()

                if dataset_type == 'sevir':
                    data = valid_loader.sample(batch_size=arguments.local_batch_size)
                    x, y = data['vil'][:, :seq_len], data['vil'][:, seq_len:]
                elif dataset_type.startswith('meteo'):
                    data = next(valid_loader)
                    x, y = data
                elif dataset_type.startswith('hko'):
                    x_seq, x_mask, dt_clip, _ = valid_loader.sample(batch_size=arguments.local_batch_size)
                    x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, arguments)
                x, y = x.to(rank), y.to(rank)
                
                with torch.no_grad():
                    if x.shape[-1] != img_size:
                        x, y = resize(x, img_size), resize(y, img_size)
                    ae_loss, mu_loss, diff_loss, prior_loss = model.compute_loss(x, y, validate=True)
                
                acc_ae += ae_loss
                acc_mu += mu_loss
                acc_diff += diff_loss
            
            acc_ae, acc_mu, acc_diff = acc_ae/arguments.v_steps, acc_mu/arguments.v_steps, acc_diff/arguments.v_steps
            writer.add_scalar('Val AE loss', float(acc_ae), global_step=total_step)
            writer.add_scalar('Val VP loss', float(acc_mu), global_step=total_step)
            writer.add_scalar('Val Diff loss', float(acc_diff), global_step=total_step)
            logging.info(f'[Epoch {epoch}][Validation] AE_loss:{float(acc_ae):.4}, VP_loss:{float(acc_mu):.4}, Diff_loss:{float(acc_diff):.4}')
            val_loss = (acc_mu+acc_diff)/2

            with torch.no_grad():
                y_pred, mu = ddp_model(x)
            out_x, out_y, mu_pred, out_y_pred = x[rand_batch].unsqueeze(0), y[rand_batch].unsqueeze(0), mu[rand_batch].unsqueeze(0), y_pred[rand_batch].unsqueeze(0)
            utpp.torch_visualize({'input': out_x, 'ground truth': out_y, 'mu_pred': mu_pred, 'predicted': out_y_pred}, savedir=os.path.join(save_path, f'temp-{total_step}.png'), vmin=0, vmax=1)

            if arguments.save_every_epoch:
                torch.save(ddp_model.module.state_dict(), os.path.join(save_path, f'{model_pathname}_epoch={epoch}.pt'))
            else:
                if val_loss < best_val_loss:
                    torch.save(ddp_model.module.state_dict(), os.path.join(save_path, f'{model_pathname}_best.pt'))
                    best_val_loss = val_loss
        

    
    if rank == 0:
        logging.info("Saving final model checkpoint...")
        torch.save(ddp_model.module.state_dict(), os.path.join(save_path, f'{model_pathname}_final.pt'))
        logging.info("Final model saved successfully.")

    cleanup()

def main(arguments):
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_train,
        args=(world_size, arguments, ),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # model related
    parser.add_argument('-d', '--dataset', type=str, default='', help='Dataset config to be trained')
    parser.add_argument('-m', '--model', type=str, default='', help='The global configuration to be used (The var name in config.py)')
    parser.add_argument('--type', type=str, default='3D', help='Determine which kind of model to use, 2D or 3D')
    parser.add_argument('-o', '--output', type=str, default='m_ckpts', help='The output directory')
    parser.add_argument('-f', dest='checkpt', type=str, default='', help='model checkpoint to be loaded from (Empty = not loading)')
    # Training Components Related
    parser.add_argument('--ae_ckpt', type=str, default=None, help='Pre-trained AE checkpoint, freeze it during training')
    parser.add_argument('--ae_eval', action='store_false', help='Set AE to be trainable')
    parser.add_argument('--back_ckpt', type=str, default=None, help='Pre-trained backbone checkpoint, freeze it during traing')
    parser.add_argument('--back_eval', action='store_false', help='Set Backbone to be trainable')
    parser.add_argument('--set_mu_to_0', action='store_false', help='Set the constraint loss to 0')
    # hyperparams
    parser.add_argument('--lr', type=float, default=0.0001, help='The initial learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('-s', "--training_steps", type=int, default=0, help="number of training steps")
    parser.add_argument('-b', '--local_batch_size', type=int, default=16, help='The batch size')  # Change from 128 to 16
    parser.add_argument('--scheduler', type=str, default='cosine', help='which lr scheduler to use (cyclic, reduce)')   # Change from cyclic to cosine, # reduce
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    parser.add_argument('--validate_every', type=int, default=5, help='The number of steps to perform validation once')
    parser.add_argument('--v_steps', type=int, default=50, help='Validation steps')    
    parser.add_argument('--remarks', type=str, default='', help='This section will affect the model name to be saved')
    parser.add_argument('--save_every_epoch', action='store_true', help='Save ckpt for every validation epochs, otherwise save the best')
    args = parser.parse_args()

    main(args)