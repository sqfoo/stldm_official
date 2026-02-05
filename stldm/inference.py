import torch
from typing import Tuple

from stldm.stldm import model_setup, guidance_scheduler
from stldm.stldm_spatial import model_setup as spatial_setup
from stldm.stldm_hf import GaussianDiffusion as hf_setup

n2n_setup = {'2D': spatial_setup, '3D': model_setup, 'HF': hf_setup}

class InferenceHub:
    """
    Unified inference interface for STLDM models

    Support local checkpoints and the checkpoint uploaded to Hugging Face.
    
    Params:
    - model_config: dict, the model configuration found in "stldm/model_config.py"
    - model_ckpt: str, the path to the model checkpoint. For 'HF' model_type, this can be None.
    - cfg_str: float, the classifier-free guidance strength. If None, no CFG is applied.
    - model_type: str, the type of the model. Options are '2D', '3D', and 'HF'.
    """
    
    def __init__(self, model_config, model_ckpt:str=None, cfg_str:float=None, model_type:str='3D', gpu='auto'):
        self.input_size = model_config['vp_param']['shape_in']
        self.sampling_steps = model_config['param']['timesteps']
        self.model_config = self.setup_config(model_config, model_type)

        self.model = self.setup_model(model_type, self.model_config, model_ckpt)
        self.setup_cfg(cfg_str)

        if gpu is not None:
            if gpu == 'auto':
                if torch.cuda.device_count() > 0:
                    self.model.to(device="cuda")
            else:
                self.model.to(device=f"cuda:{gpu}")

    def setup_config(self, model_config, model_type):
        if model_type == 'HF':
            HF_config = {
                'vp_param': model_config['vp_param'],
                'stldm_param': model_config['stldm_param'],
                **model_config['param'],
            }
            return HF_config
        else:
            return model_config

    def setup_model(self, model_type, model_config, model_ckpt):
        if model_type not in n2n_setup:
            raise ValueError(f"model_type should be one of {str(list(n2n_setup.keys()))}")
        
        if model_type == 'HF':
            model = n2n_setup[model_type](**model_config).from_pretrained("sqfoo/STLDM_official")
        else:
            model = n2n_setup[model_type](model_config)
            model.load_state_dict(torch.load(model_ckpt))
        model.eval()
        return model

    def setup_cfg(self, cfg_str):
        guidance = guidance_scheduler(sampling_step=self.sampling_steps, const=cfg_str) if cfg_str is not None else None
        self.model.setup_guidance(guidance)

    """
    This method performs inference on the input tensor.

    Params:
    - input_x: torch.tensor, the input tensor with shape (B T C H W) or (T C H W)
    - include_mu: bool, whether to return the intermediate representation 'mu' along with the final prediction
    """
    @torch.no_grad()
    def __call__(self, input_x: torch.tensor, include_mu: bool = False):
        ndim = input_x.ndim
        if ndim not in (5, 4):
            raise ValueError("Please specify the input has the either format of (B T C H W) or (T C H W)")
        input_device = input_x.device

        if ndim == 4:
            input_x = input_x.unsqueeze(0)
        
        if input_x.shape[1:] != self.input_size:
            raise ValueError(f"Ensure that the input has the shape of {str(self.input_size)}")

        input_x = input_x.to(self.model.device)
        if include_mu:
            y_pred, mu = self.model(input_x, includ_mu=include_mu)
        else:
            y_pred = self.model(input_x, includ_mu=include_mu)
            mu = None

        if mu is not None:
            mu = mu.to(input_device)
        y_pred = y_pred.to(input_device)

        if ndim == 4:
            y_pred = y_pred[0]
            mu = mu if mu is None else mu[0]
        return (y_pred, mu) if include_mu else y_pred