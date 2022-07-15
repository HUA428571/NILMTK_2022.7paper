# from nilmtk.dataset_converters import convert_redd
#
# convert_redd(r'..\data\low_freq',
#              r'..\data\low_freq\redd_low_new.h5')

from __future__ import print_function, division
import pandas as pd
import numpy as np
from nilmtk.dataset import DataSet
# from nilmtk.metergroup import MeterGroup
# from nilmtk.datastore import HDFDataStore
# from nilmtk.timeframe import TimeFrame
# 一下两行属于瞎写，官方的写的才能用
# from nilmtk.disaggregate.combinatorial_optimisation import CombinatorialOptimisation
# from nilmtk.legacy.disaggregate.fhmm_exact import FHMM
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM, MLE


# 获取数据
train = DataSet(r'..\data\low_freq\redd_low_new.h5')  # 读取数据集
test = DataSet(r'..\data\low_freq\redd_low_new.h5')  # 读取数据集
building = 1  ## 选择家庭house
train.set_window(end="30-4-2011")  ## 划分数据集，2011年4月20号之前的作为训练集
test.set_window(start="30-4-2011")  ## 四月40号之后的作为测试集

## elec包含了这个家庭中的所有的电器信息和总功率信息,building=1-6个家庭
train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec

top_5_train_elec = train_elec.submeters().select_top_k(k=5)  ## 选择用电量排在前5的来进行训练和测试

# 计算
def predict(clf, test_elec, sample_period, timezone):  ## 定义预测的方法
    pred = {}
    gt = {}
    # 获取总的负荷数据
    for i, chunk in enumerate(test_elec.mains().load(sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()  ### 丢到缺省值
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)  #### 分解，disaggregate_chunk #通过调用这个方法实现分解，这部分代码在下面可以见到
        gt[i] = {}  ## 这是groudtruth，即真实的单个电器的消耗功率

        for meter in test_elec.submeters().meters:
            # Only use the meters that we trained on (this saves time!)
            gt[i][meter] = next(meter.load(sample_period=sample_period))
        gt[i] = pd.DataFrame({k: v.squeeze() for k, v in gt[i].items()},
                             index=next(iter(gt[i].values())).index).dropna()  #### 上面这一块主要是为了得到pandas格式的gt数据

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
    appliance_labels = [m.label() for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels

    return gt_overall, pred_overall


predictions = {}

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}

sample_period = 120  ## 采样周期是两分钟
for clf_name, clf in classifiers.items():
    print("*" * 20)
    print(clf_name)
    print("*" * 20)
    clf.train(top_5_train_elec, sample_period=sample_period)  ### 训练部分
    gt, predictions[clf_name] = predict(clf, test_elec, 120, train.metadata['timezone'])

# 评估
def compute_rmse(gt, pred):   ### 评估指标 rmse
    from sklearn.metrics import mean_squared_error
    rms_error = {}
    for appliance in gt.columns:
        rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance])) ## 评价指标的定义很简单，就是均方根误差
    return pd.Series(rms_error)
# rmse = {}
# for clf_name in classifiers.keys():
#     rmse[clf_name] = compute_rmse(gt, predictions[clf_name])
# rmse = pd.DataFrame(rmse)

# Finally: Check performance of FHMM and CO
rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = compute_rmse(gt, predictions[clf_name])

print('\n\n+++++ RESULTS +++++')
print('\n++ RMSE ++')
print(pd.DataFrame(rmse).round(1))
res_1 = pd.DataFrame(rmse).round(1)
rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = compute_rmse(gt, predictions[clf_name], pretty=True)

rmse = pd.DataFrame(rmse)
print(rmse)