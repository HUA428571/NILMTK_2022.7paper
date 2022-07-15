import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    'sample_rate': 1,
    'display_predictions': True,
    # 'artificial_aggregate': True,
    'appliances': [
        'fridge',
        # 'light',
        # 'dish washer',
        # 'sockets',
        # 'microwave'
    ],
    'methods': {
        # 'Mean': Mean({}),  # 此代码可以正常运行
        # 'CO': CO({}),  # 此代码可以正常运行
        # 'Hart85': Hart85({}),  # 此代码可以正常运行

        # 'AFHMM': AFHMM({}),
        # "AFHMM_SAC": AFHMM_SAC({}),
        # "DAE":{'n_epochs':50,'batch_size':1024},
        #
        # "DSC": DSC({'learning_rate': 1e-11, 'iterations': 300}),
        # "FHMM_EXACT": {},
        # 'RNN': RNN({'n_epochs': 50, 'batch_size': 1024}),
        # 'Seq2Point': Seq2Point({'n_epochs': 50, 'batch_size': 1024}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 50, 'batch_size': 1024}),
        'Seq2Seq': Seq2Seq({'n_epochs': 30, 'batch_size': 1024}),
        # 'WindowGRU': WindowGRU({'n_epochs': 30, 'batch_size': 1024}),
    },
    'train': {
        'datasets': {
            'REDD': {
                'path': r'..\data\low_freq\redd_low_new.h5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-05-19'
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
                        'start_time': '2011-05-20',
                        'end_time': '2011-05-24'
                    },
                }
            }
        },
        'metrics': ['mae',  # 平均绝对误差/相对误差
                    'rmse',
                    'f1score',
                    # 'accuracy',
                    # 'precision',
                    # 'recall'
                    ]
    }
}

redd_convert = {
    'power': {
        'mains': ['apparent', 'active'],
        'appliance': ['apparent', 'active']
    },
    'sample_rate': 1,
    'display_predictions': True,

    'appliances': [
        # 'fridge',
        # 'light',
        # 'dish washer',
        'sockets',
        # 'microwave'
    ],
    'methods': {
        # 'Mean': Mean({}),  # 此代码可以正常运行
        # 'CO': CO({}),  # 此代码可以正常运行
        # 'Hart85': Hart85({}),  # 此代码可以正常运行

        # 'AFHMM': AFHMM({}),
        # "AFHMM_SAC": AFHMM_SAC({}),
        # "DAE":{'n_epochs':50,'batch_size':1024},
        #
        # "DSC": DSC({'learning_rate': 1e-11, 'iterations': 300}),
        # "FHMM_EXACT": {},
        # 'RNN': RNN({'n_epochs': 50, 'batch_size': 1024}),
        # 'Seq2Point': Seq2Point({'n_epochs': 10, 'batch_size': 1024}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 50, 'batch_size': 1024}),
        'Seq2Seq': Seq2Seq({'n_epochs': 10, 'batch_size': 1024}),
        # 'WindowGRU': WindowGRU({'n_epochs': 30, 'batch_size': 1024}),
    },
    'train': {
        'datasets': {
            'REDD': {
                'path': r'..\data\low_freq\redd_low_convert_20220505.h5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-05-19'
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
                        'start_time': '2011-05-20',
                        'end_time': '2011-05-24'
                    },
                }
            }
        },
        'metrics': ['mae',  # 平均绝对误差/相对误差
                    'rmse',
                    'f1score',
                    # 'accuracy',
                    # 'precision',
                    # 'recall'
                    ]
    }
}

api_result = API(redd)
# api_result = API(redd_convert)
# plt.show()

# api_res = API(ukdale)

# errors_keys = api_result.errors_keys
# errors = api_result.errors
# for i in range(len(errors)):
#     print(errors_keys[i])
#     print(errors[i])
#     print("\n\n")
