## Train
## when you have data, and splite them into training dataset and test dataset.
## when you train the model using XX algorithm by training dataset,
## and get the input varibles X_train and y_train.

# 装备库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline #使用jupyter时需要加上
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import model_selection
from scipy import interp
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)

# 定义分类器，比如随机森林
clf = ensemble.RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                          class_weight='balanced',
                                          criterion='gini', max_depth=5,
                                          max_features='auto',
                                          max_leaf_nodes=None, max_samples=None,
                                          min_impurity_decrease=0.0,
                                          min_impurity_split=None,
                                          min_samples_leaf=1,
                                          min_samples_split=2,
                                          min_weight_fraction_leaf=0.0,
                                          n_estimators=460, n_jobs=-1,
                                          oob_score=False,
                                          random_state=1,
                                          verbose=0, warm_start=False)

# 考虑是否交叉验证
cv = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 1)
#classifier = clf1
#cv = model_selection.StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 1.1, step=0.1))

# 训练模型并交叉验证
# 需要修改的是输入变量，X_train and y_train.
for train, test in cv.split(X_train, y_train):
    probas_ = clf.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# 画ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'ROC AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 相关的设置被注释掉，可以恢复并查看画图效果。
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
#plt.xlim([-0, 1])
#plt.ylim([-0, 1])
plt.xlabel('1-Specificity', fontsize = 'x-large')
plt.ylabel('Sensitivity', fontsize = 'x-large')
#plt.title('Receiver operating characteristic example', fontsize = 'x-large')
plt.legend(loc="lower right" , fontsize = 'large')
#plt.savefig('Train-ROC.tiff',dpi=200)
plt.show()