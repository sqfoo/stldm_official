import torch, random
from torch import nn
from einops import rearrange

from stldm.submodules import *

class Down_Block(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, time_dim, is_last, patch_size=None, num_groups=8, heads=4, dim_head=32):
        super(Down_Block, self).__init__()
        self.block1 = ResnetBlock(dim=in_ch, dim_out=hid_ch, time_emb_dim=time_dim, groups=num_groups)
        self.attn_spatial = Residual(PreNorm(hid_ch, Quadratic_SpatialAttention(dim=hid_ch, heads=heads, dim_head=dim_head))) if patch_size is None else Residual(PreNorm(hid_ch, Linear_SpatialAttention(dim=hid_ch, patch_size=patch_size, heads=heads, dim_head=dim_head)))
        self.block2 = ResnetBlock(dim=hid_ch, dim_out=hid_ch, groups=num_groups)
        # self.attn_temporal = Residual(PreNorm(hid_ch, TemporalAttention_Pos(dim=hid_ch, heads=heads, dim_head=dim_head)))
        self.attn_temporal = Residual(PreNorm(hid_ch, TemporalAttention(dim=hid_ch, heads=heads, dim_head=dim_head)))
        self.last = Downsample2D(dim_in=hid_ch, dim_out=out_ch) if not is_last else ChannelConversion(hid_ch, out_ch)
    
    def forward(self, x, time_emb, cond=None, relative_pos=None):
        assert x.ndim==5
        B, T, C, H, W = x.shape

        x = x.reshape(B*T, C, H, W)
        if cond is None:
            cond = torch.zeros_like(x) # -> Unconditioning

        time_emb = time_emb.unsqueeze(1) # From (B C) to (B 1 C)
        time_emb = time_emb.repeat(1, T, 1)
        time_emb = time_emb.reshape(B*T, -1)
        
        out = torch.cat((x, cond), dim=1) # BT, 2C, H, W
        out = self.block1(out, time_emb)
        
        spatial_attn = self.attn_spatial(out)
        out = self.block2(spatial_attn, time_emb)
        *_, c, h, w = out.shape
        out = out.reshape(B,T,c,h,w)
        
        # temporal_attn = self.attn_temporal(out, relative_pos)
        temporal_attn = self.attn_temporal(out)
        temporal_attn = temporal_attn.reshape(B*T,c,h,w)

        out = self.last(temporal_attn)
        *_, c, h, w = out.shape

        return out.reshape(B, T, c, h, w), spatial_attn, temporal_attn

class MidBlock(nn.Module):
    def __init__(self, in_ch, time_dim, num_groups=8, heads=4, dim_head=32):
        super(MidBlock, self).__init__()
        self.block1 = ResnetBlock(dim=in_ch, dim_out=in_ch, time_emb_dim=time_dim, groups=num_groups)
        self.qattn_spatial = Residual(PreNorm(in_ch, Quadratic_SpatialAttention(dim=in_ch, heads=heads, dim_head=dim_head)))
        self.block2 = ResnetBlock(dim=in_ch, dim_out=in_ch, time_emb_dim=time_dim, groups=num_groups)
        # self.qattn_time = Residual(PreNorm(in_ch, TemporalAttention_Pos(dim=in_ch, heads=heads, dim_head=dim_head)))
        self.qattn_time = Residual(PreNorm(in_ch, TemporalAttention(dim=in_ch, heads=heads, dim_head=dim_head)))
        self.block3 = ResnetBlock(dim=in_ch, dim_out=in_ch, time_emb_dim=time_dim, groups=num_groups)

    def forward(self, x, time_emb, relative_pos=None):
        assert x.ndim==5
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)

        time_emb = time_emb.unsqueeze(1) # From (B C) to (B 1 C)
        time_emb = time_emb.repeat(1, T, 1)
        time_emb = time_emb.reshape(B*T, -1)

        out = self.block1(x, time_emb)
        out = self.qattn_spatial(out)
        out = self.block2(out, time_emb) # a little bit difference here

        out = out.reshape((B, T, C, H, W))
        # out = self.qattn_time(out, relative_pos).reshape(B*T, C, H, W)
        out = self.qattn_time(out).reshape(B*T, C, H, W)
        out = self.block3(out, time_emb)

        *_, c, h, w = out.shape
        return out.reshape(B, T, c, h, w)

class Up_Block(nn.Module):
    def __init__(self, in_chs, hid_ch, out_ch, is_last, time_dim, patch_size=None, num_groups=8, heads=4, dim_head=32):
        super(Up_Block, self).__init__()
        in_ch, skip_ch = in_chs
        self.up = Upsample2D(dim_in=in_ch, dim_out=hid_ch) if not is_last else ChannelConversion(in_ch, hid_ch)
        self.attn_spatial = Residual(PreNorm(hid_ch, Quadratic_SpatialAttention(dim=hid_ch, heads=heads, dim_head=dim_head) if patch_size is None else Linear_SpatialAttention(dim=hid_ch, patch_size=patch_size, heads=heads, dim_head=dim_head)))
        self.block1 = ResnetBlock(dim=hid_ch+skip_ch, dim_out=hid_ch, time_emb_dim=time_dim, groups=num_groups)
        # self.attn_temporal =  Residual(PreNorm(hid_ch, TemporalAttention_Pos(dim=hid_ch, heads=heads, dim_head=dim_head)))
        self.attn_temporal =  Residual(PreNorm(hid_ch, TemporalAttention(dim=hid_ch, heads=heads, dim_head=dim_head)))
        self.block2 = ResnetBlock(dim=hid_ch+skip_ch, dim_out=out_ch, time_emb_dim=time_dim, groups=num_groups)
    
    def forward(self, x, time_emb, spatialattn_skip, tempattn_skip, relative_pos=None):
        assert x.ndim==5
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)

        time_emb = time_emb.unsqueeze(1) # From (B C) to (B 1 C)
        time_emb = time_emb.repeat(1, T, 1)
        time_emb = time_emb.reshape(B*T, -1)

        out = self.up(x)
        *_, c, h, w = out.shape
        out = out.reshape(-1, T, c, h, w)
        
        # out = self.attn_temporal(out, relative_pos).reshape(B*T, c, h, w)
        out = self.attn_temporal(out).reshape(B*T, c, h, w)
        
        out = torch.cat((out, tempattn_skip), dim=1)
        out = self.block1(out, time_emb)
        
        out = self.attn_spatial(out)
        
        out = torch.cat((out, spatialattn_skip), dim=1)
        out = self.block2(out, time_emb)
        *_, c, h, w = out.shape
        return out.reshape(B, T, c, h, w)

class LDM(nn.Module):
    def __init__(self, in_ch, chs_mult:tuple, patch_size=None, num_groups=8, heads=4, dim_head=32, base_ch=64):
        super(LDM, self).__init__()
        # Time Embedding MLP
        time_dim = 4*base_ch
        fourier_dim = base_ch
        self.time_mlp = Time_MLP(dim=base_ch, time_dim=time_dim, fourier_dim=fourier_dim)

        ups, downs = [], []
        conditions = []

        layer_no = len(chs_mult)
        chs = [in_ch, *map(lambda m: base_ch*m, chs_mult)]
        ch_in, ch_out = chs[:-1], chs[1:]
        up_in, up_out = list(reversed(ch_out)), list(reversed(ch_in))
        
        patches = None if patch_size is None else [patch_size//(2**n) for n in range(layer_no)] # Patch Size should be 2^N
        for n in range(layer_no):
            downs.append(
                Down_Block(in_ch=2*ch_in[n], hid_ch=ch_in[n], out_ch=ch_out[n], time_dim=time_dim, patch_size=None if patch_size is None else patches[n], is_last=(n==layer_no-1), num_groups=num_groups, heads=heads, dim_head=dim_head)
            )
            ups.append(
                Up_Block(in_chs=(up_in[n], ch_in[-n-1]), hid_ch=up_in[n], out_ch=up_out[n], time_dim=time_dim, patch_size=None if patch_size is None else patches[layer_no-n-1], is_last=(n==0), num_groups=num_groups, heads=heads, dim_head=dim_head)
            )
            if n != -1:
                conditions.append(
                    Downsample2D(dim_in=ch_in[n], dim_out=ch_out[n])
                )

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.conditions = nn.ModuleList(conditions)
        self.mid = MidBlock(in_ch=ch_out[-1], time_dim=time_dim, num_groups=num_groups, heads=heads, dim_head=dim_head)
        # self.relative_pos = RelativePositionBias(heads=heads)
    
    def forward(self, x, time, conds=None):
        t = self.time_mlp(time)

        hid_spatial = []
        hid_temporal = []

        # relative_position = self.relative_pos(x.shape[1], x.device) # Calculate The Relative Position

        for n, down_block in enumerate(self.downs):
            # print(x.shape)
            # x, spatial_attn, time_attn = down_block(x, t, conds, relative_position)
            x, spatial_attn, time_attn = down_block(x, t, conds)
            hid_spatial.append(spatial_attn)
            hid_temporal.append(time_attn)
            if conds is not None:
                conds = self.conditions[n](conds)
        
        # out = self.mid(x, t, relative_position)
        out = self.mid(x, t)

        for up_block in self.ups:
            # out = up_block(out, t, hid_spatial.pop(), hid_temporal.pop(), relative_position)
            out = up_block(out, t, hid_spatial.pop(), hid_temporal.pop())
        
        return out

# constants
from collections import namedtuple
from torch.cuda.amp import autocast
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None

def guidance_scheduler(sampling_step: int, const: float):
    return const*torch.ones(sampling_step)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        vp_model,
        diffusion,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super(GaussianDiffusion, self).__init__()

        self.backbone = vp_model
        self.diff_unet = diffusion

        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    @property
    def device(self):
        return self.betas.device

    # CFG schdeuler => by taking pre-setting scheduler
    def setup_guidance(self, scheduler):
        if exists(scheduler):
            self.CFG_sch = scheduler.to(self.device)
        else:
            self.CFG_sch = scheduler

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond, clip_x_start = False, rederive_pred_noise = False):
        # print(t.device)
        if exists(self.CFG_sch):
            uncond = self.diff_unet(x, t, conds=None) #conds=torch.zeros_like(cond))
            model_output = self.diff_unet(x, t, conds=cond)
            time = int(t[0])
            model_output = model_output - self.CFG_sch[time] * (uncond - model_output)
        else:
            model_output = self.diff_unet(x, t, conds=cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t, cond=None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond=cond, clip_x_start=False,)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond=cond, clip_denoised = False)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, return_all_timesteps = False):
        batch, device = shape[0], self.device

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable=True):
            frames_pred, _ = self.p_sample(frames_pred, t, cond=cond)
            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, return_all_timesteps = False):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        device = self.device
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=True):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)            
            pred_noise, x_start, *_ = self.model_predictions(
                            frames_pred, 
                            time_cond,
                            cond = cond, #cond.copy(),
                            clip_x_start = False, 
                            rederive_pred_noise = True
                        )

            if time_next < 0:
                frames_pred = x_start
                imgs.append(frames_pred)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(frames_pred)

            frames_pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def sample(self, frames_in, return_all_timesteps = False):
        assert frames_in.ndim == 5
        B, T_in, C, H, W = frames_in.shape
        device = self.device

        backbone_output, conds, *_ = self.backbone(frames_in)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        *_, c, h, w = conds.shape
        tgt_shape = conds.reshape(B, -1, c, h, w).shape
        ldm_pred = sample_fn(
            tgt_shape,
            cond=conds,
            return_all_timesteps = return_all_timesteps
        )
        
        ldm_pred = rearrange(ldm_pred, 'b t c h w -> (b t) c h w')
        frames_pred = self.backbone.vae.decode(ldm_pred)
        frames_pred = rearrange(frames_pred, '(b t) c h w -> b t c h w', b=B)
        return frames_pred, backbone_output

    def predict(self, frames_in, compute_loss=False, **kwargs):
        pred, mu = self.sample(frames_in=frames_in)
        return pred, mu

    def compute_loss(self, frames_in, frames_gt, validate=False):
        compute_loss = True and (not validate)
        B, T_in, C, H, W = frames_in.shape
        T_out = frames_gt.shape[1]
        device = frames_in.device

        """
        Diffusion Loss
        """
        backbone_output, conds = self.backbone(frames_in)
        hid_gt, _ = self.backbone.vae.encode(
            rearrange(frames_gt, 'b t c h w -> (b t) c h w')
        )
        hid_gt = rearrange(hid_gt, '(b t) c h w -> b t c h w', b=B)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        if random.random() > 0.85: # Unconditional
            conds = None
        diff_loss = self.p_losses(hid_gt.detach(), t, cond=conds)

        """
        Backbone Loss
        """
        mu_loss = self.backbone._losses_(frames_in, frames_gt)

        """
        VAE Loss
        """
        ae_loss, kl_loss = self.backbone.vae._losses_(
            rearrange(torch.cat((frames_in, frames_gt), dim=1), 'b t c h w -> (b t) c h w'),
            rearrange(torch.cat((frames_in, frames_gt), dim=1), 'b t c h w -> (b t) c h w')
        )
        kl_weight = 1E-6
        recon_loss = ae_loss + kl_weight*kl_loss

        """
        Prior Loss at t=T [Noisy]
        """
        hid_gt, _ = self.backbone.vae.encode(
            rearrange(frames_gt, 'b t c h w -> (b t) c h w')
        )
        hid_gt = rearrange(hid_gt, '(b t) c h w -> b t c h w', b=B)
        T = torch.ones((B,), device=self.device).long() * (self.num_timesteps - 1)
        mu_noisy = extract(self.sqrt_alphas_cumprod, T, hid_gt.shape) * hid_gt
        sigma_noisy = extract(self.sqrt_one_minus_alphas_cumprod, T, hid_gt.shape)
        log_var_noisy = 2*torch.log(sigma_noisy)
        prior_loss = self.kl_from_standard_normal(mu_noisy, log_var_noisy)
        
        return recon_loss, mu_loss, diff_loss, prior_loss
    

    def kl_from_standard_normal(self, mean, log_var):
        kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
        return kl.mean()

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None, cond=None):
        b, T, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise) # Use q_sample here for updating: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L763

        model_out = self.diff_unet(x, t, conds=cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none') # (B, T, C, H, W)
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    @torch.no_grad()
    def forward(self, input_x, include_mu=False, **kwargs):
        pred, mu = self.predict(input_x, compute_loss=False)
        if include_mu:
            return pred, mu
        else:
            return pred

from stldm.modules import SimVPV2_Model, VAE
def model_setup(model_config, print_info=False, cfg_str=None):
    if print_info:
        print('Setup the model with considering temporal attention be (BHW, T, C) ... ...')
        print('Train it from end to end')
    vp_config = model_config['vp_param']
    ldm_config = model_config['stldm_param']

    vpm = SimVPV2_Model(**vp_config)
    ldm = LDM(**ldm_config)
    model = GaussianDiffusion(vp_model=vpm, diffusion=ldm, **model_config['param'])
    
    scheduler = guidance_scheduler(sampling_step=model_config['param']['timesteps'], const=cfg_str) if cfg_str is not None else None
    model.setup_guidance(scheduler)

    return model

def ae_setup(model_config):
    vp_config = model_config['vp_param']
    vpm = SimVPV2_Model(**vp_config)
    ae = vpm.vae
    return ae

def backbone_setup(model_config):
    vp_config = model_config['vp_param']
    vpm = SimVPV2_Model(**vp_config)
    return vpm