from featureGenerator import *
from readToMatrix import *
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, KFold
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix



def getMatrix(dirname):
    pssmList = os.listdir(dirname)
    pssmList.sort(key=lambda x: eval(x[:]))
    m = len(pssmList)
    reMatrix = np.zeros((m,20*20))
    for i in range(m):
        matrix= readToMatrix(dirname + '/' + pssmList[i], 'pssm')
        reMatrix[i, :] = bigram_pssm(matrix)
    print(reMatrix.shape)
    return reMatrix

def main():
    x1 = getMatrix("pssm")
    y = [-1 for i in range(x1.shape[0])]
    y.extend([1 for i in range(x2.shape[0])])
    y = np.array(y)

    test_x1 = getMatrix("pssm")
    test_y = [-1 for i in range(test_x1.shape[0])]
    test_y.extend([1 for i in range(test_x2.shape[0])])

    #
    CC = []
    gammas = []
    for i in range(-5, 16, 2):
        CC.append(2 ** i)
    for i in range(3, -16, -2):
        gammas.append(2 ** i)
    param_grid = {"C": CC, "gamma": gammas}
    gs = GridSearchCV(SVC(probability=True), param_grid, cv=10)
    gs.fit(x, y)
    print(gs.best_estimator_)
    print(gs.best_score_)

    # LOO
    clf = gs.best_estimator_
    loo = LeaveOneOut()
    score = cross_val_score(clf, x, y, cv=loo).mean()
    print("Loo-ACC{}".format(score))
    #
    loo_probas_y = []  #
    loo_test_y = []  #
    loo_predict_y = []  #
    for train, test in loo.split(x):
        clf.fit(x[train], y[train])
        loo_predict_y.extend(clf.predict(x[test]))  #
        loo_probas_y.extend(clf.predict_proba(x[test]))  #
        loo_test_y.extend(y[test])  #
    loo_probas_y = np.array(loo_probas_y)
    loo_test_y = np.array(loo_test_y)
    print(loo_probas_y.shape)


    #
    confusion = confusion_matrix(loo_test_y, loo_predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC:{}".format(roc_auc_score(loo_test_y, loo_probas_y[:, 1])))
    print("SP:{}".format(TN / (TN + FP)))
    print("SN:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE:{}".format(TP / (TP + FP)))
    print("MCC:{}".format(n))
    print("F-score:{}".format((2 * TP) / (2 * TP + FP + FN)))
    print("ACC:{}".format((TP + TN) / (TP + FP + TN + FN)))

    # Ind
    clf = gs.best_estimator_
    clf.fit(x, y)
    predict_y = clf.predict(test_x)
    probas_y = clf.predict_proba(test_x)
    print("Ind-ACC：{}".format(accuracy_score(test_y, predict_y)))


    # Ind
    confusion = confusion_matrix(test_y, predict_y)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print("ROC:{}".format(roc_auc_score(test_y, probas_y[:, 1])))
    print("SP:{}".format(TN / (TN + FP)))
    print("SN:{}".format(TP / (TP + FN)))
    n = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    print("PRE:{}".format(TP / (TP + FP)))
    print("MCC:{}".format(n))
    print("F-score:{}".format((2 * TP) / (2 * TP + FP + FN)))
    print("ACC:{}".format((TP + TN) / (TP + FP + TN + FN)))

main()
