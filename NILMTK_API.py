# python3.7
# tensorflow 1.14

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    # redd数据集
    'sample_rate': 60,

    'appliances': ['fridge',
                   # 'light',
                   # 'sockets',
                   # 'microwave',
                   # 'dish washer'
                   ],

    'methods': {
        'Mean': Mean({}),  # 此代码可以正常运行
        'CO': CO({}),  # 此代码可以正常运行
        'Hart85': Hart85({}),  # 此代码可以正常运行

        'AFHMM': AFHMM({}),
        # "AFHMM_SAC": AFHMM_SAC({}),
        # 'DAE': DAE({'n_epochs': 50, 'batch_size': 32}),

        # "DSC": DSC({'learning_rate': 1e-11, 'iterations': 300}),
        # "FHMM_EXACT": {},
        # 'RNN': RNN({'n_epochs': 50, 'batch_size': 32}),
        # 'Seq2Point': Seq2Point({'n_epochs': 50, 'batch_size': 32}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 50, 'batch_size': 512}),
        # 'WindowGRU': WindowGRU({'n_epochs': 50, 'batch_size': 32}),
    },
    'train': {
        'datasets': {
            'REDD': {
                'path': r'..\data\low_freq\redd_low_new.h5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-24'
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
                        'start_time': '2011-04-25',
                        'end_time': '2011-04-27'
                    },
                }
            }
        },
        # 'metrics': ['mae'],
        'metrics': ['mae', 'f1score', 'accuracy', 'precision', 'recall']
    }
}

ukdare = {
    'power': {
        'mains': ['apparent', 'active'],
        'appliance': ['apparent', 'active']
    },

    # redd数据集
    # 'sample_rate': 60,
    #
    # 'appliances': ['fridge',
    #                'light',
    #                'sockets',
    #                'microwave',
    #                'dish washer'
    #                ],

    # SynD数据集
    # 'sample_rate': 10,
    #
    # 'appliances': ['fridge',
    #                # 'Electric space heater',
    #                # 'Clothes iron',
    #                # 'Dish washer',
    #                # 'Washing machine'
    #                ],

    # ukdare数据集
    'sample_rate': 6,
    'appliances': ['fridge freezer', 'dish washer'],

    'methods': {
        'Mean': Mean({}),  # 此代码可以正常运行
        # 'CO': CO({}),  # 此代码可以正常运行
        # 'Hart85': Hart85({}),  # 此代码可以正常运行

        # 'AFHMM': AFHMM({}),
        # "AFHMM_SAC": AFHMM_SAC({}),
        # 'DAE': DAE({'n_epochs': 50, 'batch_size': 32}),

        # "DSC": DSC({'learning_rate': 1e-11, 'iterations': 300}),
        # "FHMM_EXACT": {},
        # 'RNN': RNN({'n_epochs': 50, 'batch_size': 32}),
        # 'Seq2Point': Seq2Point({'n_epochs': 50, 'batch_size': 32}),
        'Seq2Seq': Seq2Seq({'n_epochs': 50, 'batch_size': 512}),
        # 'WindowGRU': WindowGRU({'n_epochs': 50, 'batch_size': 32}),
    },
    'train': {
        'datasets': {
            # 'Dataport': {
            # 'path': r'..\data\low_freq\redd_low_new.h5',
            # 'buildings': {
            #     1: {
            #         # 'start_time': '2011-04-17',
            #         # 'end_time': '2011-04-24'
            # 'start_time': '2020-02-01',
            # 'end_time': '2020-02-10'
            #     },
            # }

            'UK-DALE': {
                'path': r'..\data\ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2013-04-13',
                        'end_time': '2014-01-01'
                    },
                }
            }
        }
    },
    'test': {
        'datasets': {
            # 'Datport': {
            #     # 'path': r'..\data\low_freq\redd_low_new.h5',
            #     'path': r'../data/synd.h5',
            #     'buildings': {
            #         1: {
            #             # 'start_time': '2011-04-25',
            #             # 'end_time': '2011-04-27'
            #             'start_time': '2020-02-11',
            #             'end_time': '2020-02-12'
            #         },
            #     }
            # }

            'UK-DALE': {
                'path': r'..\data\ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2014-01-01',
                        'end_time': '2014-03-30'
                    },
                }
            }
        },
        # 'metrics': ['mae'],
        'metrics': ['mae', 'f1score', 'accuracy', 'precision', 'recall']
    }
}

api_res = API(redd)
api_res.errors
api_res.errors_keys
