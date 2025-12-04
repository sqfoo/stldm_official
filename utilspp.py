import os
import torch
import numpy as np
import lpips as lp
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
from bisect import bisect_right
import torchvision.transforms as T
from torch import nn

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from data import dutils

# =======================================================================
# Scheduler Helper Function
# =======================================================================

class SequentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self.optimizer = optimizer
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1

    def step(self, ref=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            # Check HERE
            if isinstance(self._schedulers[idx], torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._schedulers[idx].step(ref)
            else:
                self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda

# =======================================================================
# Utils in utils :)
# =======================================================================
def to_cpu_tensor(*args):
    '''
    Input arbitrary number of array/tensors, each will be converted to CPU torch.Tensor
    '''
    out = []
    for tensor in args:
        if type(tensor) is np.ndarray:
            tensor = torch.Tensor(tensor)    
        if type(tensor) is torch.Tensor:
            tensor = tensor.cpu()
        out.append(tensor)
    # single value input: return single value output
    if len(out) == 1:
        return out[0]
    return out

def merge_leading_dims(tensor, n=2):
    '''
    Merge the first N dimension of a tensor
    '''
    return tensor.reshape((-1, *tensor.shape[n:]))

# =======================================================================
# Model Preparation, saving & loading (copied from utils.py)
# =======================================================================
def build_model_name(model_type, model_config):
    '''
    Build the model name (without extension)
    '''
    model_name = model_type + '_'
    for k, v in model_config.items():
        model_name += k
        if type(v) is list or type(v) is tuple:
            model_name += '-'
            for i, item in enumerate(v):
                model_name += (str(item) if type(item) is not bool else '') + ('-' if i < len(v)-1 else '')                
        else:
            model_name += (('-' + str(v)) if type(v) is not bool else '')
        model_name += '_'
    return model_name[:-1]

def build_model_path(base_dir, dataset_type, model_type, timestamp=None):
    if timestamp is None:
        return os.path.join(base_dir, dataset_type, model_type)
    elif timestamp == True:
        return os.path.join(base_dir, dataset_type, model_type, pd.Timestamp.now().strftime('%Y%m%d%H%M%S'))
    return os.path.join(base_dir, dataset_type, model_type, timestamp)

# =======================================================================
# Preprocess Function for Loading HKO-7 dataset
# =======================================================================

def hko7_preprocess(x_seq, x_mask, dt_clip, args):
    resize = args.resize if 'resize' in args else x_seq.shape[-1]
    seq_len = args.seq_len if 'seq_len' in args else 5

    # post-process on HKO-10
    x_seq = x_seq.transpose((1, 0, 2, 3, 4)) / 255. # => (batch_size, seq_length, 1, 480, 480)
    if 'scale' in args and args.scale == 'non-linear':
        x_seq = dutils.linear_to_nonlinear_batched(x_seq, dt_clip)
    else:
        x_seq = dutils.nonlinear_to_linear_batched(x_seq, dt_clip)

    b, t, c, h, w = x_seq.shape
    assert c == 1, f'# channels ({c}) != 1'

    # resize (downsample) the images if necessary
    x_seq = torch.Tensor(x_seq).float().reshape((b*t, c, h, w))
    if resize != h:
        tform = T.Compose([
            T.ToPILImage(), 
            T.Resize(resize),
            T.ToTensor(),
        ])
    else:
        tform = T.Compose([])

    x_seq = torch.stack([tform(x_frame) for x_frame in x_seq], dim=0)
    x_seq = x_seq.reshape((b, t, c, resize, resize))

    x, y = x_seq[:, :seq_len], x_seq[:, seq_len:]
    return x, y

# =======================================================================
# Evaluation Metrics-Related 
# =======================================================================

mae = lambda *args: torch.nn.functional.l1_loss(*args).cpu().detach().numpy()
mse = lambda *args: torch.nn.functional.mse_loss(*args).cpu().detach().numpy()

def ssim(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    b, t, c, h, w = y.shape
    y = y.reshape((b*t, c, h, w))
    y_pred = y_pred.reshape((b*t, c, h, w))
    # to further ensure any of the input is not negative
    y = torch.clamp(y, 0, 1)
    y_pred = torch.clamp(y_pred, 0, 1)
    return torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)(y_pred, y)

def psnr(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    b, t, c, h, w = y.shape
    y = y.reshape((b*t, c, h, w))
    y_pred = y_pred.reshape((b*t, c, h, w))
    acc_score = 0
    for i in range(b*t):
        acc_score += torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)(y_pred[i], y[i]) / (b*t)
    return acc_score

GLOBAL_LPIPS_OBJ = None # a static variable
def lpips64(y_pred, y, net='vgg'):
    # convert the image range into [-1, 1], assuming the input range to be [0, 1]
    y = merge_leading_dims(y)
    y_pred = merge_leading_dims(y_pred)

    y = torch.nn.functional.interpolate(y, (64, 64), mode='bicubic').clamp(0,1)
    y_pred = torch.nn.functional.interpolate(y_pred, (64, 64), mode='bicubic').clamp(0,1)
    
    y = (2 * y - 1)
    y_pred = (2 * y_pred - 1)
    global GLOBAL_LPIPS_OBJ
    if GLOBAL_LPIPS_OBJ is None:
        GLOBAL_LPIPS_OBJ = lp.LPIPS(net=net).to(y.device)
    return GLOBAL_LPIPS_OBJ(y_pred, y).mean()

def tfpn(y_pred, y, threshold, radius=1):
    '''
    convert to cpu, and merge the first two dimensions
    '''
    y = merge_leading_dims(y)
    y_pred = merge_leading_dims(y_pred)
    with torch.no_grad():
        if radius > 1:
            pool = nn.MaxPool2d(radius)
            y = pool(y)
            y_pred = pool(y_pred) 
        y = torch.where(y >= threshold, 1, 0)
        y_pred = torch.where(y_pred >= threshold, 1, 0)
        mat = torchmetrics.functional.confusion_matrix(y_pred, y, task='binary', threshold=threshold)
        (tn, fp), (fn, tp) = to_cpu_tensor(mat)
    return tp, tn, fp, fn

def tfpn_pool(y_pred, y, threshold, radius):
    y_pred = merge_leading_dims(y_pred)
    y = merge_leading_dims(y)
    pool = nn.MaxPool2d(radius, stride=radius//4 if radius//4 > 0 else radius) 
    with torch.no_grad():
        y = torch.where(y>=threshold, 1, 0).float()
        y_pred = torch.where(y_pred>=threshold, 1, 0).float()
        y = pool(y)
        y_pred = pool(y_pred)
        mat = torchmetrics.functional.confusion_matrix(y_pred, y, task='binary', threshold=threshold)
        (tn, fp), (fn, tp) = to_cpu_tensor(mat)
    return tp, tn, fp, fn

def csi(tp, tn, fp, fn):
    '''Critical Success Index. The larger the better.'''
    if (tp + fn + fp) < 1e-7:
        return 0.
    return tp / (tp + fn + fp)

def hss(tp, tn, fp, fn):
    '''Heidke Skill Score. (-inf, 1]. Larger better.'''
    if (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) == 0:
        return 0.
    return 2 * (tp*tn - fp*fn) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))

# =======================================================================
# Data Visualization
# =======================================================================

def torch_visualize(sequences, savedir=None, horizontal=10, vmin=0, vmax=1):
    '''
    input: sequences, a list/dict of numpy/torch arrays with shape (B, T, C, H, W) 
    C is assumed to be 1 and squeezed 
    If batch > 1, only the first sequence will be printed 
    '''        
    # First pass: compute the vertical height and convert to proper format
    vertical = 0
    display_texts = []
    if (type(sequences) is dict):
        temp = []
        for k, v in sequences.items():
            vertical += int(np.ceil(v.shape[1] / horizontal)) 
            temp.append(v)
            display_texts.append(k)            
        sequences = temp
    else:
        for i, sequence in enumerate(sequences):
            vertical += int(np.ceil(sequence.shape[1] / horizontal))
            display_texts.append(f'Item {i+1}')
    sequences = to_cpu_tensor(*sequences)
    # Plot the sequences   
    j = 0
    fig, axes = plt.subplots(vertical, horizontal, figsize=(2*horizontal, 2*vertical), tight_layout=True)
    plt.setp(axes, xticks=[], yticks=[])
    for k, sequence in enumerate(sequences):
        # only take the first batch, now seq[0] is the temporal dim
        sequence = sequence[0].squeeze() # (T, H, W)
        axes[j, 0].set_ylabel(display_texts[k])
        for i, frame in enumerate(sequence):
            j_shift = j + i // horizontal 
            i_shift = i % horizontal
            axes[j_shift, i_shift].imshow(frame, vmin=vmin, vmax=vmax, cmap='gray')
        j += int(np.ceil(sequence.shape[0] / horizontal))    
    if savedir:
        plt.savefig(savedir + '' if savedir.endswith('.png') else '.png')
        plt.close()
    else:
        plt.show()

""" Visualize function with colorbar and a line seprate input and output """
def color_visualize(sequences, savedir='', horizontal=5, skip=1, ypos=0):
    '''
    input: sequences, a list/dict of numpy/torch arrays with shape (B, T, C, H, W) 
    C is assumed to be 1 and squeezed 
    If batch > 1, only the first sequence will be printed 
    '''        
    plt.style.use(['science', 'no-latex'])
    VIL_COLORS = [[0, 0, 0],
                [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
                [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
                [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
                [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
                [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
                [0.9607843137254902, 0.9607843137254902, 0.0],
                [0.9294117647058824, 0.6745098039215687, 0.0],
                [0.9411764705882353, 0.43137254901960786, 0.0],
                [0.6274509803921569, 0.0, 0.0],
                [0.9058823529411765, 0.0, 1.0]]

    VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]

    # First pass: compute the vertical height and convert to proper format
    vertical = 0
    display_texts = []
    if (type(sequences) is dict):
        temp = []
        for k, v in sequences.items():
            vertical += int(np.ceil(v.shape[1] / horizontal)) 
            temp.append(v)
            display_texts.append(k)            
        sequences = temp
    else:
        for i, sequence in enumerate(sequences):
            vertical += int(np.ceil(sequence.shape[1] / horizontal))
            display_texts.append(f'Item {i+1}')
    sequences = to_cpu_tensor(*sequences)
    # Plot the sequences   
    j = 0
    fig, axes = plt.subplots(vertical, horizontal, figsize=(2*horizontal, 2*vertical), tight_layout=True)
    plt.subplots_adjust(hspace=0.0, wspace=0.0) # tight layout
    plt.setp(axes, xticks=[], yticks=[])    
    for k, sequence in enumerate(sequences):
        # only take the first batch, now seq[0] is the temporal dim
        sequence = sequence[0].squeeze() # (T, H, W)
        
        ## =================
        # = labels of time =
        if k == 0:
            for i in range(len(sequence)):
                axes[j, i].set_xlabel(f'$t-{(len(sequence)-i)-1}$', fontsize=16)
                axes[j, i].xaxis.set_label_position('top') 
        elif k == len(sequences)-1:
            for i in range(len(sequence)):
                axes[j, i].set_xlabel(f'$t+{skip*i+1}$', fontsize=16)
                axes[j, i].xaxis.set_label_position('bottom')            
        ## =================        
        axes[j, 0].set_ylabel(display_texts[k], fontsize=16)
        for i, frame in enumerate(sequence):
            j_shift = j + i // horizontal 
            i_shift = i % horizontal
            im = axes[j_shift, i_shift].imshow(frame*255, cmap=ListedColormap(VIL_COLORS), \
                                          norm=BoundaryNorm(VIL_LEVELS, ListedColormap(VIL_COLORS).N))
        j += int(np.ceil(sequence.shape[0] / horizontal))    
    
    ## = plot splittin line =
    if ypos == 0:
        ypos = 1 - 1 / len(sequences) - 0.017
    fig.lines.append(Line2D((0, 1), (ypos, ypos), transform=fig.transFigure, ls='--', linewidth=2, color='#444'))
    # color bar
    cax = fig.add_axes([1, 0.05, 0.02, 0.5])
    fig.colorbar(im, cax=cax)
    ## =================
    if savedir:
        plt.savefig(savedir + '' if len(savedir)>0 else 'out.png')
        plt.close()
    else:
        plt.show()
