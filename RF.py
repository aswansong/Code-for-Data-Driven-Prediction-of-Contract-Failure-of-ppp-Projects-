import time

import numpy as np
import pandas as pd
import warnings
from sklearn.tree import export_graphviz
import pydot as pydot
from imblearn.over_sampling import SMOTE
from collections import Counter
import pylab as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 10
pd.options.display.max_columns = None


df = pd.read_csv('final2.csv')
df = df.drop(columns = 'Unnamed: 0')


skl_df = df[['ContractPeriod','IncomeGroup','RegulatoryQuality',
        'IDA Status','Type of PPI','Project status',
        'Financial closure year','Capacity','RuleofLaw',
        'TotalInvestment','PhysicalAssets','Sponsors',
        'Sponsors Country','BiLateralSupport','Primary sector',
        'UnsolicitedProposal','PublicDisclosure','GovernmentEffectiveness',
        'PercentPrivate','VoiceandAccountability','Political StabilityNoViolence',
   'ControlofCorruption','Access to electricity (% of population)',
   'Domestic credit to private sector (% of GDP)','Exports of goods and services (% of GDP)',
   'GDP deflator (base year varies by country)','GDP per capita (constant 2015 US$)','Imports of goods and services (% of GDP)',
   'Life expectancy at birth, total (years)','Official exchange rate (LCU per US$, period average)','Tax revenue (% of GDP)']]
skl_df = skl_df.fillna(0)
for i in range(len(skl_df['BiLateralSupport'])):
    if skl_df['BiLateralSupport'][i]!=1:
        skl_df['BiLateralSupport'][i]=0
# print(skl_df)
data_by = skl_df.groupby('Primary sector')
count = 0
for project,sorted_df in data_by:
    count+=1
    print('Primary sector:',project)
    y = sorted_df['Project status'].values
    x_name = ['ContractPeriod','IncomeGroup','RegulatoryQuality',
        'IDA Status','Type of PPI','Financial closure year','Capacity','RuleofLaw',
        'TotalInvestment','PhysicalAssets','Sponsors','Sponsors Country','BiLateralSupport',
        'UnsolicitedProposal','PublicDisclosure','GovernmentEffectiveness','PercentPrivate','VoiceandAccountability','Political StabilityNoViolence',
   'ControlofCorruption','Access to electricity (% of population)',
   'Domestic credit to private sector (% of GDP)','Exports of goods and services (% of GDP)',
   'GDP deflator (base year varies by country)','GDP per capita (constant 2015 US$)','Imports of goods and services (% of GDP)',
   'Life expectancy at birth, total (years)','Official exchange rate (LCU per US$, period average)','Tax revenue (% of GDP)']
    x = sorted_df[x_name].values
    # print(Counter(y))

    smo = SMOTE()#平衡数据集
    s1  = StandardScaler()#数据归一化处理
    x = s1.fit_transform(x)#对标签数据进行归一化StandardScaler
    pac  = PCA()#pac降维处理
    pac.fit(x)
    x = pac.transform(x)

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)#按照0.3的比例拆分数据集为训练集和验证机
    X_train, y_train = smo.fit_resample(X_train, y_train)#somte训练集，使其均衡
    # X_test,y_test = smo.fit_resample(X_test,y_test)、
    model = RandomForestClassifier(n_estimators=120,min_samples_leaf=3,criterion='gini')
    model.fit(X_train,y_train)
    y_pred =model.predict(X_test)#，模型预测
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
    print(train_score, test_score)
    print('',model.feature_importances_)

    #绘制混淆矩阵
    matrix = confusion_matrix(y_test,y_pred)
    plt.matshow(matrix, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # 更改横纵坐标的刻度
    plt.xticks([0, 1], ['Success', 'Failure'])
    plt.yticks([0, 1], ['Success', 'Failure'])

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.annotate(matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 绘制pr曲线
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.figure(1)
    plt.title('Pre Rec')
    plt.plot(precision, recall)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

    #绘制决策树：
    estimator = model.estimators_[5]

    # 导出为dot 文件
    export_graphviz(estimator, out_file=f'tree{count}.dot',
                    feature_names=x_name,
                    rounded=True, proportion=False,
                    precision=2, filled=True)


    # 绘制AUC曲线
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()