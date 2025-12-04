import torch.nn.functional as F

from data import dutils
from nowcasting.hko_iterator import HKOIterator

def GET_TrainLoader(meta, param, batch_size, in_len, out_len):
    if meta['dataset'] == 'SEVIR':
        total_seq_len = in_len + out_len
        train_config = {
            'data_types': ['vil'],
            'layout': 'NTCHW',
            'seq_len': total_seq_len,
            'raw_seq_len': total_seq_len,
            'end_date': dutils.SEVIR_TRAIN_TEST_SPLIT_DATE,
            'start_date': None
        }
        test_config = {
            'data_types': ['vil'],
            'layout': 'NTCHW',
            'seq_len': total_seq_len,
            'raw_seq_len': total_seq_len,
            'end_date': None,
            'start_date': dutils.SEVIR_TRAIN_TEST_SPLIT_DATE
        }
        train_loader = dutils.SEVIRDataIterator(**train_config, batch_size=batch_size)
        test_loader = dutils.SEVIRDataIterator(**test_config, batch_size=8 if batch_size > 8 else batch_size)
        return train_loader, test_loader
    elif meta['dataset'].startswith('HKO'):
        total_seq_len = in_len + out_len
        pkl_path = param['pd_path']
        train_loader = HKOIterator(pd_path=pkl_path.replace('test', 'train'), sample_mode="random", seq_len=total_seq_len, stride=1)
        test_loader = HKOIterator(pd_path=pkl_path, sample_mode="sequent", seq_len=total_seq_len, stride=in_len)
        return train_loader, test_loader
    elif meta['dataset'] == 'meteonet':
        train_loader, test_loader = dutils.load_meteonet(batch_size=batch_size, val_batch_size=8 if batch_size > 8 else batch_size, train=True, **param)
        return train_loader, test_loader
    else:
        raise Exception(f'Undefined dataset config name: {dataset_config["dataset"]}')

def GET_TestLoader(meta, param, batch_size):
    if meta['dataset'] == 'SEVIR':
        return dutils.SEVIRDataIterator(**param, batch_size=batch_size)
    elif meta['dataset'].startswith('HKO'):
        return HKOIterator(**param)
    elif meta['dataset'] == 'meteonet':
        _, test_iter = dutils.load_meteonet(batch_size=batch_size, val_batch_size=8, train=False, **param)
        return iter(test_iter)
    else:
        raise Exception(f'Undefined dataset config name: {dataset_config["dataset"]}')