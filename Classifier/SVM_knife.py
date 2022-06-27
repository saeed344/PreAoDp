import numpy as np
import pandas as pd
from time import time
import scipy.io as sio
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
#from L1_Matine import elasticNet, lassodimension,lassolarsdimension
#from L1_Matine import selectFromLinearSVC,selectFromExtraTrees,logistic_dimension
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.model_selection import StratifiedKFold
#from LR_test import logistic_LR
import utils.tools as utils

####################input data##################
train_data = pd.read_csv(r'AAC_Bigram_PSSM_feature.csv', header=None) #, index_col=None)
X = np.array(train_data)
Y1=np.ones((254,1))#Value can be changed
Y2=np.zeros((1522,1))
Y=np.append(Y1,Y2)
y=Y
#####################################Feature selection ##############################
#data_1,mask1=elasticNet(shu, label)#弹性网络
#data_2,mask2=lassodimension(shu,label)#lasso
#data_3,mask3=lassolarsdimension(shu,label)
#data_4=selectFromLinearSVC(shu,label,1.5)#从线性支持向量机中选择
#data_5,importance=selectFromExtraTrees(shu,label)
#data_5,mask_5=logistic_dimension(shu,label,1.5)
##data_6,mask6=logistic_LR(shu,label,0.5,0.001)
#X=shu
#label[label==-1]=0

num_class=2
loo = LeaveOneOut()
sepscores = []
y_score=np.ones((1,2))*0.5
y_class=np.ones((1,1))*0.5      

#########################optimize peratmeter################
'''
C_range = 2. ** np.arange(-15, 15)
gamma_range = 2. ** np.arange(-15, -5)
param_grid = dict(gamma=gamma_range, C=C_range)
clf = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=10)
clf.fit(X,Y)
C=clf.best_params_['C']
gamma=clf.best_params_['gamma']'''
C=3.0314
gamma=0.0039063
##################################################
for train, test in loo.split(X): 
    cv_clf = SVC(C=C,gamma=gamma,kernel='rbf',probability=True)
    X_train=X[train]
    y_train=y[train] 
    X_test=X[test]
    y_test=y[test]
    y_sparse=utils.to_categorical(y)
    y_train_sparse=utils.to_categorical(y_train)
    y_test_sparse=utils.to_categorical(y_test)
    hist=cv_clf.fit(X_train, y_train)
    y_predict_score=cv_clf.predict_proba(X_test) 
    y_predict_class= utils.categorical_probas_to_classes(y_predict_score)
    y_score=np.vstack((y_score,y_predict_score))
    y_class=np.vstack((y_class,y_predict_class))
    cv_clf=[]
y_class=y_class[1:]
y_score=y_score[1:]
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
roc_auc = auc(fpr, tpr)
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y)
result=[acc,precision,npv,sensitivity,specificity,mcc,roc_auc]
row=y_score.shape[0]
y_sparse=utils.to_categorical(y)
yscore_sum = pd.DataFrame(data=y_score)
yscore_sum.to_csv('yscore_SVM_420_knife.csv')
ytest_sum = pd.DataFrame(data=y_sparse)
ytest_sum.to_csv('ytest_SVM_420_knife.csv')
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
auc_score=result[6]
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('SVM_420_knife.csv')
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='SVM ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
lr_precision, lr_recall, _ = precision_recall_curve(y_sparse[:,0], y_score[:,0])
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
#no_skill = len(testy[testy==1]) / len(testy)
#plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
