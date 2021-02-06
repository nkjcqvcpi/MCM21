import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

WORKSPACE = '/Users/houtonglei/OneDrive - stu.hqu.edu.cn/数学建模/2021美赛/'
DATASET_DESCRIPTION = WORKSPACE + '2021_ICM_ProblemC/2021MCM_ProblemC_DataSet.xlsx'

dataset_description = pd.read_excel(DATASET_DESCRIPTION)

dataset_description = dataset_description[['GlobalID', 'Lab Status', 'Latitude', 'Longitude']]

dataset_description = dataset_description.replace('Positive ID', 1)
dataset_description = dataset_description.replace('Negative ID', 0)
temp = dataset_description[(dataset_description['Lab Status'] == 1) | (dataset_description['Lab Status'] == 0)][['Lab Status', 'Latitude', 'Longitude']]

x_train, x_test, y_train, y_test = train_test_split(temp[['Latitude', 'Longitude']], temp['Lab Status'], test_size=0.25)

# 进行标准化处理   因为目标结果经过sigmoid函数转换成了[0,1]之间的概率，所以目标值不需要进行标准化。
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
y_test = y_test.astype(np.int8)
y_train = y_train.astype(np.int8)

# 逻辑回归预测
lg = LogisticRegression(C=1.0)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
lg.fit(x_train, y_train)

# 回归系数
print(lg.coef_)

# 进行预测
y_predict = lg.predict(x_test)

print("准确率：", lg.score(x_test, y_test))

print("召回率：\n", classification_report(y_test, y_predict, labels=[0, 1], target_names=["0", "1"]))
