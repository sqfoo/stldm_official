import os, sys, logging, argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import utilspp as utpp
from utilspp import mae, mse, ssim, psnr, lpips64, csi, hss
from data.config import SEVIR_13_12, HKO7_5_20, METEONET_5_20
from data.loader import GET_TestLoader
from data.dutils import resize

class MetricListEvaluator():    
    '''
    To evaluate a list of metrics. Supported metrics:
    - CSI, HSS (Eg. `csi-84, hss-84`)
    - CSI-pooled (Eg. `csi_4-84`)
    - MAE
    - MSE
    - SSIM
    - PSNR  
    '''
    def __init__(self, metric_list):        
        self.metric_holder = {}
        self.batch_count = 0
        for metric_name in metric_list:
            threshold = ''
            radius = ''
            if '-' in metric_name:
                metric, threshold = metric_name.split('-')
                if '_' in metric:
                    metric, radius = metric.split('_')
                    radius = int(radius)
            # initialize metrics
            threshold = float(threshold) / 255 if threshold.isdigit() else threshold
            self.metric_holder[metric_name] = self.init_metric(metric_name, threshold=threshold, radius=radius)

    def init_metric(self, metric_name, **kwarg):
        '''
        return a tuple of three items in order:
        - the function to call during eval
        - the value(s) to keep track of
        - a dict of any additional item to pass into the function
        '''
        if metric_name.split('-')[0] in ['csi', 'hss']:
            # use tfpn instead
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float32), {'threshold': kwarg['threshold']}] # tp, 
        elif '_' in metric_name.split('-')[0]: # Indicate Pooling
            return [utpp.tfpn_pool, np.array([0, 0, 0, 0], dtype=np.float32), {'threshold': kwarg['threshold'], 'radius': kwarg['radius']}]
        else:
            # directly convert the string name into function call
            return [eval(metric_name), 0, {}]

    def eval(self, y_pred, y):
        self.batch_count += 1
        for _, metric in self.metric_holder.items():
            temp = metric[0](y_pred, y, **metric[-1])      
            if temp is list:
                temp = np.array(temp)
            elif type(temp) == torch.Tensor:
                temp = temp.detach().cpu().numpy()
            metric[1] += temp
            
    def get_results(self):
        output_holder = {}
        for key, metric in self.metric_holder.items():
            val = metric[1]
            # special handle of tfpn => compute the final score now
            if metric[0] is utpp.tfpn:
                metric_name, threshold = key.split('-')
                val = eval(metric_name)(*list(metric[1]))
            elif metric[0] is utpp.tfpn_pool:
                metric_name, info = key.split('_')
                val = eval(metric_name)(*list(metric[1]))
            else:
                val /= self.batch_count
            output_holder[key] = val
        return output_holder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset related
    parser.add_argument('-d', '--dataset', type=str, default='', help='the dataset definition to be set')
    parser.add_argument('--out_len',type=int, required=True, help='The actual prediction length')
    # ensemble npy filename with {}
    parser.add_argument('--e_file', default='', type=str, help='Ensemble npy filename with included \{ \}')
    parser.add_argument('--ens_no', default=1, type=int, help='Total ensemble number')
    # hyperparams
    parser.add_argument('-s', '--step', type=int, default=-1, help='The number of steps to run. -1: the entire dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
    # config override
    parser.add_argument('--metrics', type=str, default=None, help='A list of metrics to be evaluated, separated by character /')    
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    args = parser.parse_args()

    # Prepare logger
    path_list = args.e_file.split("/")
    logfile_name = os.path.join(*path_list[:-1], 'ensemble_eval.log')
    logging.basicConfig(level=logging.NOTSET, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    logging.info(f'Steps: {args.step}')

    dataset_config = globals()[args.dataset]
    dataset_param, dataset_meta = dataset_config['param'], dataset_config['meta']
    loader = GET_TestLoader(dataset_meta, dataset_param, args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # prepare metrics
    metric_list = dataset_meta['metrics']
    if args.metrics is not None: 
        metric_list = args.metrics.lower().split('/')
        logging.info(f'Overwriting metrics list with: {metric_list}')
    evaluator = MetricListEvaluator(metric_list)

    for e in range(args.ens_no):
        prediction = np.load(args.e_file.format(str(e)))
        prediction = torch.tensor(prediction, device=device)
        
        step = 1
        if dataset_meta['dataset'] in ['SEVIR', 'HKO-7']:
            loader.reset() # Reset it, otherwise alignment error
        else:
            pass
            
        while args.step < 0 or step <= args.step:
            if dataset_meta['dataset'] == 'SEVIR':
                data = loader.sample(batch_size=args.batch_size)
                if data is None:
                    break
                y = data['vil'][:, -args.out_len:] # Expected to be same as prediction
            elif dataset_meta['dataset'] == 'HKO-7':
                setattr(args, 'seq_len', dataset_meta['seq_len'])
                try:
                    data = loader.sample(batch_size=args.batch_size)
                except Exception as e:
                    logging.error(e)
                    break
                x_seq, x_mask, dt_clip, _ = data
                x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args)
            elif dataset_meta['dataset'].startswith('meteo'):
                try:
                    x, y = next(loader)
                except Exception as e:
                    logging.error(e)
                    break

            with torch.no_grad():
                y = y.to(device)
                y_pred = prediction[(step-1)*args.batch_size:step*args.batch_size]

                if y.shape[-1] != y_pred.shape[-1]:
                    y = resize(y, y_pred.shape[-1])

                y, y_pred = y.clamp(0,1), y_pred.clamp(0,1) # B T C H W
                evaluator.eval(y_pred, y)
            # log/print every
            if step == 1 or step % args.print_every == 0:
                logging.info(f'E_ID:{e} -> {step} Steps evaluated')
            step += 1
    # log the final scores
    final_results = evaluator.get_results()
    for k, v in final_results.items():
        logging.info(f'{k}: {v}')