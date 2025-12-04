from data import dutils 

SEVIR_13_12 = {
    'meta': {
        'dataset': 'SEVIR',    
        'seq_len': 13,
        'out_len': 12,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips64', 
                    'csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219',
                    'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219',
                    'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219',
                    'hss-16', 'hss-74', 'hss-133', 'hss-160', 'hss-181', 'hss-219' ],
    },
    'param': {        
        'seq_len': 25,
        'data_types': ['vil'], 
        'sample_mode': 'sequent',
        'layout': 'NTCHW', 
        'raw_seq_len': 25, 
        'start_date': dutils.SEVIR_TRAIN_TEST_SPLIT_DATE,
        'end_date': None,
    },
    'savedir': 'sevir'
}

HKO7_5_20 = {
    'meta': {
        'dataset': 'HKO-7',
        'seq_len': 5,
        'out_len': 20,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips64', 
                    'csi-84', 'csi-117', 'csi-140', 'csi-158', 'csi-185',
                    'csi_4-84', 'csi_4-117', 'csi_4-140', 'csi_4-158', 'csi_4-185',
                    'csi_16-84', 'csi_16-117', 'csi_16-140', 'csi_16-158', 'csi_16-185',
                    'hss-84', 'hss-117', 'hss-140', 'hss-158', 'hss-185'],
    },
    'param': {            
        'pd_path': 'data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl',
        'sample_mode': 'sequent',
        'seq_len': 25,
        'stride': 13,
    },
    'savedir': 'hko-7'
}

METEONET_5_20 = {
    'meta': {
        'dataset': 'meteonet',
        'seq_len': 5,
        'out_len': 20,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips64', 
                    'csi-44', 'csi-64', 'csi-87', 'csi-117',
                    'csi_4-44', 'csi_4-64', 'csi_4-87', 'csi_4-117', 
                    'csi_16-44', 'csi_16-64', 'csi_16-87', 'csi_16-117',
                    'hss-44', 'hss-64', 'hss-87', 'hss-117']
    },
    'param': {        
        'img_size': 128,
        'in_len': 5, 
    },
    'savedir': 'meteonet'
}