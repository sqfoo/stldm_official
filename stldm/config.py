STLDM_SEVIR = {
    'model': "stldm",
    'pre': None,
    'post': None,
    'vp_param': {      
        'shape_in': (13, 1, 128, 128),
        'shape_out': (12, 1, 128, 128),
        'hid_S': 32,
        'hid_T': 512,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    },
    'stldm_param': {
        'in_ch': 32, 
        'chs_mult': [1,2,4,8],
        'num_groups': 8,
        'heads': 4,
        'dim_head': 32,
        'base_ch': 64,
        'patch_size': 16
    },
    'param': {
        'timesteps': 50,
        'sampling_timesteps': 20,
        'objective': 'pred_v'
    }
}

STLDM_HKO = {
    'model': "stldm",
    'pre': None,
    'post': None,
    'vp_param': {      
        'shape_in': (5, 1, 128, 128),
        'shape_out': (20, 1, 128, 128),
        'hid_S': 32,
        'hid_T': 512,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    },
    'stldm_param': {
        'in_ch': 32, 
        'chs_mult': [1,2,4,8],
        'num_groups': 8,
        'heads': 4,
        'dim_head': 32,
        'base_ch': 64,
        'patch_size': 16
    },
    'param': {
        'timesteps': 50,
        'sampling_timesteps': 20,
        'objective': 'pred_v'
    }
}

STLDM_METEO = {
    'model': "stldm",
    'pre': None,
    'post': None,
    'vp_param': {      
        'shape_in': (5, 1, 128, 128),
        'shape_out': (20, 1, 128, 128),
        'hid_S': 32,
        'hid_T': 512,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    },
    'stldm_param': {
        'in_ch': 32, 
        'chs_mult': [1,2,4,8],
        'num_groups': 8,
        'heads': 4,
        'dim_head': 32,
        'base_ch': 64,
        'patch_size': 16
    },
    'param': {
        'timesteps': 50,
        'sampling_timesteps': 20,
        'objective': 'pred_v'
    }
}


STLDM_HKO_HF = {
    'vp_param': {      
        'shape_in': (5, 1, 128, 128),
        'shape_out': (20, 1, 128, 128),
        'hid_S': 32,
        'hid_T': 512,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    },
    'stldm_param': {
        'in_ch': 32, 
        'chs_mult': [1,2,4,8],
        'num_groups': 8,
        'heads': 4,
        'dim_head': 32,
        'base_ch': 64,
        'patch_size': 16
    },
    'timesteps': 50,
    'sampling_timesteps': 20,
    'objective': 'pred_v'
}