import torch
from torch import nn

from stldm.submodules import ChannelConversion
from stldm.simvpv2 import stride_generator, ConvSC, MidMetaNet

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
            ChannelConversion(C_hid, 2*C_hid)
        )

    def forward(self, x):
        for encoder in self.enc:
            x = encoder(x)
        (mean, log_var) = torch.chunk(x, 2, dim=1)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, last_activation='sigmoid'):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            ChannelConversion(C_hid, C_hid),
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(C_hid, C_hid, stride=strides[-1], transpose=True)# Modify HERE
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        if last_activation=='sigmoid':
            self.last = nn.Sigmoid()
        else:
            self.last = nn.Identity()
    
    def forward(self, x):
        for decoder in self.dec:
            x = decoder(x)
        Y = self.readout(x)
        return self.last(Y)


class VAE(nn.Module):
    def __init__(self, C_in, hid_S, N_S, last_activation='none'):
        super(VAE, self).__init__()
        self.encoder = Encoder(C_in, hid_S, N_S)
        self.decoder = Decoder(hid_S, C_in, N_S, last_activation)

    def sample_from_standard_normal(self, mean, log_var):
        std = (0.5 * log_var).exp()
        return mean + std * torch.randn_like(mean)
    
    def encode(self, x):
        assert x.ndim==4
        mean, log_var = self.encoder(x)
        return mean, log_var

    def decode(self, z):
        assert z.ndim==4
        dec = self.decoder(z)
        return dec
    
    def kl_from_standard_normal(self, mean, log_var):
        kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
        return kl.mean()

    def _losses_(self, x, y):
        mean, log_var = self.encode(x)
        kl_loss = self.kl_from_standard_normal(mean, log_var)

        y_pred = self.forward(x)
        recon_loss = nn.MSELoss()(y_pred, y)
        return recon_loss, kl_loss

    def forward(self, x):
        mu_z, log_var = self.encode(x)

        z = self.sample_from_standard_normal(mu_z, log_var)
        recon = self.decode(z)
        return recon

class SimVPV2_Model(nn.Module):
    def __init__(self, shape_in, shape_out, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, last_activation='none', act_inplace=True, **kwargs):
        super(SimVPV2_Model, self).__init__()
        T, C, H, W = shape_in  # T is pre_seq_length
        T2, C2, H2, W2 = shape_out # T2 is output length
        assert C==C2 and H==H2 and W==W2, 'Need to be the same image shape for input and output'
        self.T2 = T2
        self.T = T
        
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)

        self.vae = VAE(C_in=C, hid_S=hid_S, N_S=N_S, last_activation=last_activation)
        self.hid = MidMetaNet(T*hid_S, T2*hid_S*2, hid_T, N_T,
                    input_resolution=(H, W), model_type='gsta',
                    mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)            

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, log_var = self.vae.encode(x)
        embed = self.vae.sample_from_standard_normal(embed, log_var)
        *_, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)

        hid, *_ = self.hid(z)
        hid_mu, log_var_hid = torch.chunk(hid, 2, dim=1)
        hid = self.vae.sample_from_standard_normal(hid_mu, log_var_hid)
        
        hid = hid.reshape(B*self.T2, C_, H_, W_)
        # conds_ = hid
        conds_ = hid_mu.reshape(B*self.T2, C_, H_, W_)

        Y = self.vae.decode(hid)
        Y = Y.reshape(B, self.T2, C, H, W)
        return Y, conds_

    def _losses_(self, x, y):
        y_pred, *_ = self.forward(x)
        recon_loss = nn.MSELoss()(y_pred, y)
        return recon_loss