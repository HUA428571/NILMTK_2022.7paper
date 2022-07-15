# 2022 HuaCL
# NILMTK SynD数据集示例代码
# 代码分析&注释

# Do imports!

# 防止代码不兼容问题存在
# 详见 https://blog.csdn.net/xiaotao_1/article/details/79460365
from __future__ import print_function, division
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt

# pandas是一个数据分析的包
import pandas as pd
# numpy是python的数学计算的包
import numpy as np
import warnings
from six import iteritems

from sklearn.metrics import mean_squared_error

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE
from nilmtk.utils import compute_rmse

from nilmtk.elecmeter import ElecMeterID

# Step 2: Let's define performance metrics and the prediction procedure!
def compute_RMSE(gt, pred):
    rms_error = {}
    for appliance in gt.columns:
        rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
    return pd.Series(rms_error)


def compute_MNE(gt, pred):
    mne = {}
    for appliance in gt.columns:
        mne[appliance] = np.sum(abs(gt[appliance] - pred[appliance])**2) / np.sum(gt[appliance]**2)
    return pd.Series(mne)

def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt = {}

    for i, chunk in enumerate(test_elec.mains().load(sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        try:
            pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        except RuntimeError:
            continue
        gt[i] = {}

        for meter in test_elec.submeters().meters:
            # Only use the meters that we trained on (this saves time!)
            gt[i][meter] = next(meter.load(sample_period=sample_period))
        gt[i] = pd.DataFrame({k: v.squeeze() for k, v in iteritems(gt[i]) if len(v)},
                             index=next(iter(gt[i].values())).index).dropna()

    # If everything can fit in memory
    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()

    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]

    # Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)

    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.ix[common_index_local]
    pred_overall = pred_overall.ix[common_index_local]
    appliance_labels = [m for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall


# Step3: Define settings and create variables!
################## SETTINGS ##################

sample_period = 10

d_dir = '../data/'

################## VARS ##################

train = DataSet(d_dir+'SynD.h5')
test = DataSet(d_dir+'SynD.h5')

train.set_window(end="2020-02-07")
test.set_window(start="2020-02-07")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

top_5_train_elec = train_elec.submeters().select_top_k(k=5)

# Step 4: Train and predict!
################## DISAGGREGATE ##################
predictions = {}

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}

for clf_name, clf in classifiers.items():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    clf.train(top_5_train_elec, sample_period=sample_period)
    gt, predictions[clf_name] = predict(clf, test_elec, sample_period, train.metadata['timezone'])


# Finally: Check performance of FHMM and CO
rmse = {}
mne = {}

for clf_name in classifiers.keys():
    rmse[clf_name] = compute_RMSE(gt, predictions[clf_name])
    mne[clf_name] = compute_MNE(gt, predictions[clf_name])

print('\n\n+++++ RESULTS +++++')

print('\n++ RMSE ++')
print(pd.DataFrame(rmse).round(1))
res_1 = pd.DataFrame(rmse).round(1)
print('\n++ MNE ++')
print(pd.DataFrame(mne).round(2))

rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = compute_rmse(gt, predictions[clf_name], pretty=True)

rmse = pd.DataFrame(rmse)
print(rmse)