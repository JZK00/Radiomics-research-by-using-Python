## Test
## when you have data, and splite them into training dataset and test dataset.
## when you get the trained model clf, and want to test it by test dataset,
## and get the test varibles X_test and y_test.

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

# 准备测试并画ROC曲线
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
lw=2
i = 0
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 1.1, step=0.1))
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
probas_ = clf.predict_proba(X_test)  #需要修改的是clf，即训练得到的model；以及测试集的X_test和y_test.
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
fpr=fpr
tpr=tpr
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
#plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
plt.plot(fpr, tpr, color='b', alpha=.8, lw=lw, linestyle='-',label='ROC AUC = %0.2f' % roc_auc) 
#plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',alpha=.6)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)

#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
#plt.xlim([-0, 1])
#plt.ylim([-0, 1])
plt.xlabel('1-Specificity', fontsize = 'x-large')
plt.ylabel('Sensitivity', fontsize = 'x-large')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right" , fontsize = 'large')
#plt.savefig('Test-ROC.tiff',dpi=200)
plt.show()