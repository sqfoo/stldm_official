import torch, math
from torch import nn
from einops import rearrange

# building block modules
def exists(x):
    return x is not None

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        if dim_out%groups != 0:
            groups = 1
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

"""
Input Tensor and Output Tensor should be in the format of (BT, C, H, W) with # dims = 4
"""
class Linear_SpatialAttention(nn.Module):
    def __init__(self, dim, patch_size, heads=4, dim_head=32):
        super(Linear_SpatialAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.patch_size = patch_size
        self.heads = heads
        hidden_dim = dim_head*heads # No of Channel for (Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            LayerNorm(dim)
        )

    def forward(self, x):
        assert x.ndim == 4
        BT, C, H, W = x.shape
        nh, nw = H//self.patch_size, W//self.patch_size
        qkv = self.to_qkv(x).chunk(3, dim=1) # qkv tuple in (q, k , v)
        # [B, Head × C, X × P, Y × P] -> [B, Head × X × Y, C, P × P]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) (nh ph) (nw pw) -> b (h nh nw) c (ph pw)', h=self.heads, ph=self.patch_size, pw=self.patch_size, nh=nh, nw=nw), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q*self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b (h nh nw) c (ph pw) -> b (h c) (nh ph) (nw pw)', h=self.heads, ph=self.patch_size, pw=self.patch_size, nh=nh, nw=nw)
        out = self.to_out(out)
        return out

"""
Input Tensor and Output Tensor should be in the format of (B, T, C, H, W) with # dims = 5
"""
class Linear_TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Linear_TemporalAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head*heads # No of Channel for (Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            LayerNorm(dim)
        )

    def forward(self, x):
        assert x.ndim == 5
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        qkv = self.to_qkv(x).chunk(3, dim=1) # qkv tuple in (q, k , v)
        # [B, Head × C, X × P, Y × P] -> [B, Head × X × Y, C, P × P]
        q, k, v = map(lambda t: rearrange(t, '(b t) (h c) x y -> b (h x y) c t', h=self.heads, x=H, y=W, t=T), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q*self.scale
        v /= (H*W)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b (h x y) c t -> (b t) (h c) x y', h=self.heads, x=H, y=W, t=T)
        out = self.to_out(out)
        return out.reshape(B, T, C, H, W)

# Does not Follow what suggested by the paper as could not ensure the spatial factor of 2
def Downsample2D(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=(4, 4), stride=(2, 2), padding=(1,1))

def Upsample2D(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, kernel_size=(4, 4), stride=(2, 2), padding=(1,1))

def ChannelConversion(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=(3,3), padding=(1,1))

"""
Input Tensor and Output Tensor should be in the format of (BT, C, H, W) with # dims = 4
"""
class Quadratic_SpatialAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Quadratic_SpatialAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head*heads # No of Channel for (Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        assert x.ndim == 4
        BT, C, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # qkv tuple in (q, k , v)
        # [B, Head × C, X × P, Y × P] -> [B, Head × X × Y, C, P × P]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q*self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = H, y = W)

        out = self.to_out(out)
        return out

"""
Input Tensor and Output Tensor should be in the format of (B, T, C, H, W) with # dims = 5
"""
class Quadratic_TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Quadratic_TemporalAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head*heads # No of Channel for (Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, padding=0, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        assert x.ndim == 5
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        qkv = self.to_qkv(x).chunk(3, dim=1) # qkv tuple in (q, k , v)
        # [B, Head × C, X × P, Y × P] -> [B, Head × X × Y, C, P × P]
        q, k, v = map(lambda t: rearrange(t, '(b t) (h c) x y -> b h (c x y) t', h=self.heads, x=H, y=W, t=T), qkv)
        q = q*self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h t (c x y) -> (b t) (h c) x y', h=self.heads, x=H, y=W, t=T)
        out = self.to_out(out)
        return out.reshape(B, T, C, H, W)

"""
A series of functions required for Diffusion Model copied from DiffCast code
"""
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Time_MLP(nn.Module):
    def __init__(self, dim, time_dim, fourier_dim=32):
        super(Time_MLP, self).__init__()
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

"""
Input Tensor and Output Tensor should be in the format of (B, T, C, H, W) with # dims = 5
"""
class TemporalAttention_Pos(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(TemporalAttention_Pos, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head*heads # No of Channel for (Q, K, V)
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, padding=0)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x, rel_pos=None):
        assert x.ndim == 5
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        qkv = self.to_qkv(x).chunk(3, dim=1) # qkv tuple in (q, k , v)
        # [B, Head × C, X × P, Y × P] -> [B, Head × X × Y, C, P × P]
        q, k, v = map(lambda t: rearrange(t, '(b t) (h c) x y -> (b x y) h c t', h=self.heads, x=H, y=W, t=T), qkv)
        q = q*self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        if rel_pos is not None:
            sim += rel_pos
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, '(b x y) h t c -> (b t) (h c) x y', h=self.heads, x=H, y=W, t=T)
        out = self.to_out(out)
        return out.reshape(B, T, C, H, W)


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(TemporalAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head*heads
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        assert x.ndim == 5
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b (h w) t c')

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = rearrange(q, '... n (h d) -> ... h n d', h=self.heads) # B (H W) Head T Dim
        k = rearrange(k, '... n (h d) -> ... h n d', h=self.heads)
        v = rearrange(v, '... n (h d) -> ... h n d', h=self.heads)
        q = q*self.scale

        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h i d -> ... i (h d)', h=self.heads)
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) t c -> b t c h w', h=H, w=W)
        return out