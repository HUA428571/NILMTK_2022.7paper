# TensorFlow 2
import warnings

warnings.filterwarnings("ignore")
from nilmtk.api import API
from nilmtk.disaggregate import Mean, CO, Hart85
from nilmtk_contrib.disaggregate import DAE, Seq2Point, Seq2Seq, RNN, WindowGRU, AFHMM, AFHMM_SAC, DSC

redd = {
    'power': {
        'mains': ['apparent', 'active'],
        'appliance': ['apparent', 'active']
    },
    'sample_rate': 120,

    'appliances': [
        'fridge',
        'light',
        'dish washer',
        'sockets',
        'microwave'
    ],
    'methods': {
        'Mean': Mean({}),  # 此代码可以正常运行
        # 'CO': CO({}),  # 此代码可以正常运行
        # 'Hart85': Hart85({}),  # 此代码可以正常运行

        # 'AFHMM': AFHMM({}),
        # "AFHMM_SAC": AFHMM_SAC({}),
        # 'DAE': DAE({'n_epochs': 50, 'batch_size': 32}),

        # "DSC": DSC({'learning_rate': 1e-11, 'iterations': 300}),
        # "FHMM_EXACT": {},
        'RNN': RNN({'n_epochs': 50, 'batch_size': 32}),
        'Seq2Point': Seq2Point({'n_epochs': 50, 'batch_size': 32}),
        'Seq2Seq': Seq2Seq({'n_epochs': 50, 'batch_size': 512}),
        # 'WindowGRU': WindowGRU({'n_epochs': 50, 'batch_size': 32}),
    },
    'train': {
        'datasets': {
            'REDD': {
                'path': r'..\data\low_freq\redd_low_new.h5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-04-30'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'REDD': {
                'path': r'..\data\low_freq\redd_low_new.h5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-30',
                        'end_time': '2011-05-24'
                    },
                }
            }
        },
        'metrics': [
            'mae',
            'f1score',
            'accuracy',
            'precision',
            'recall'
        ]
    }
}

api_res = API(redd)
