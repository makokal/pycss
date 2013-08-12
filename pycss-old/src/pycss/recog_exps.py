
# simple recognition experiments

import numpy as np
import pylab as plt
from scipy import interp
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from scipy.ndimage import filters


def rebin(a, shape):
    # resize a numpy 2d array to given size
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    # return a.reshape(sh).mean(-1).mean(1)
    return clean_data(a.reshape(sh).mean(-1).mean(1))


def clean_data(data):
    df = filters.gaussian_filter(data, 5)
    return df



def load_and_pack_data(data, lenx,leny):
    # get the data and pack it as [Training Labels]

    Xdata = np.zeros(shape=(len(data), (lenx*leny)))
    Ydata = np.zeros(len(data))

    idx = 0
    for f, v in data.items():
        # print f, v, idx
        d = np.load(f)
        Xdata[idx, :] = rebin(d, [lenx,leny]).reshape(lenx*leny)
        Ydata[idx] = v
        idx += 1
    # print Ydata
    return Xdata, Ydata


def reduce_pca(X, components, retall=False):
    """ reduce_pca(X, components, retall)
    Reduce dimension by pca and returned the projected data and other config
    """

    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X)

    if not retall:
        return X_pca
    else:
        return X_pca, pca


def reduce_kpca(X, kern, retall=False):
    """ reduce_kpca(X, components, kern, retall=False)
    Reduce dim by Kernel PCA
    """

    kpca = KernelPCA(kernel=kern, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)

    if not retall:
        return X_kpca, X_back
    else:
        return X_kpca, X_back, kpca


def reduce_nmf(X, components, retall=False):
    """ reduce_nmf(X, components, retall)
    Reduce dim by non-negative matrix factorization
    """

    nmf = NMF(n_components=components, init='nndsvda', beta=5.0, tol=5e-3, sparseness='components')
    X_nmf = nmf.fit_transform(X)
    components_ = nmf.components_

    print 'NMF reconstruction error = ', nmf.reconstruction_err_

    if not retall:
        return X_nmf, components_
    else:
        return X_nmf, components_, nmf


def classify_svm(X, Y, ctitle, poslabels):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

    C = 1.0
    svc = svm.SVC(kernel='poly', C=C, degree=3, gamma=0.7, probability=True, verbose=False)
    svc.fit(X_train, y_train)
    print svc
    

    # cv = StratifiedKFold(Y, n_folds=4)
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)

    # print 'compuring ROC curves'
    # for i, (train, test) in enumerate(cv):
    #     probas_ = svc.fit(X[train], Y[train]).predict_proba(X[test])
    #     # Compute ROC curve and area the curve
    #     fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1], pos_label=poslabels)
    #     mean_tpr += interp(mean_fpr, fpr, tpr)
    #     mean_tpr[0] = 0.0
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, lw=1.5, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    # plt.legend(loc="lower right")
    # plt.title(ctitle)
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(-0.1, 1.1)
    # plt.show(block=True)


    # print svc.score(X_test, y_test)
    # print svc.score(X, Y)
    # print 'checking results on cross validation'
    # scores = cross_validation.cross_val_score(svc, X, Y, cv=5)
    # print scores

    pd = OneVsRestClassifier(svm.SVC(kernel='linear', C=C, degree=3, probability=True, verbose=False)).fit(X, Y).predict(X)
    acc = pd == Y
    print 'Accuracy: ', (np.count_nonzero(acc) / 25.0)
    print pd == Y


def classify_trees(X, Y, choice):
    """ Decision trees """
    
    if choice == 'tree':
        # clf = tree.DecisionTreeClassifier()
        clf = OneVsRestClassifier(tree.DecisionTreeClassifier())
    else:
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=5))
        # clf = RandomForestClassifier(n_estimators=5)

    clf.fit(X, Y)
    pd = clf.predict(X)
    acc = pd == Y
    print 'Accuracy: ', (np.count_nonzero(acc) / 25.0)
    print pd == Y

    # scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
    # print scores




def classify_knn(X, Y, Xt, metric):
    """ classify_knn
    """
    # clf = NearestCentroid(metric=metric, shrink_threshold=None)
    clf = neighbors.KNeighborsClassifier(3, weights='uniform')
    clf.fit(X, Y)
    # print clf.predict(Xt)
    # scores = cross_validation.cross_val_score(clf, X, Y, cv=3)
    # print scores
    pd = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(X)
    print pd == Y



def plot_projections(Xp, Y, name):
    """ Plot simple proejctions of the data """
    reds = Y == 1
    blues = Y == 2
    greens = Y == 3
    magentas = Y == 4
    yellows = Y == 5

    print 'In data shape: ', Xp.shape

    plt.figure()
    plt.plot(Xp[reds, 0], Xp[reds, 1], "ro")
    plt.plot(Xp[blues, 0], Xp[blues, 1], "bo")
    plt.plot(Xp[greens, 0], Xp[greens, 1], "go")
    plt.plot(Xp[magentas, 0], Xp[magentas, 1], "mo")
    plt.plot(Xp[yellows, 0], Xp[yellows, 1], "yo")
    plt.title(name)
    plt.xlabel('1st Component')
    plt.ylabel('2nd Component')
    plt.legend(['Coffee cup', 'Football', 'Beer bottle', 'Cereal box', 'Banana'], loc=3)


def balance_classes(X1, Y1, X2, Y2):
    # super ugly hack
    c1x = X1
    c1y = Y1
    c1x = np.append(c1x, X1, axis=0)
    c1x = np.append(c1x, X1, axis=0)
    c1x = np.append(c1x, X1, axis=0)

    c1y = np.append(c1y, Y1, axis=0)
    c1y = np.append(c1y, Y1, axis=0)
    c1y = np.append(c1y, Y1, axis=0)

    # add the other class
    c1x = np.append(c1x, X2, axis=0)
    c1y = np.append(c1y, Y2, axis=0)

    return c1x, c1y




if __name__ == '__main__':
    # load data from current dir
    dnames = {'cup1.npy' : 1, 'cup2.npy' : 1, 'cup3.npy' : 1, 'cup4.npy' : 1, 'cup5.npy' : 1, 'ball1.npy' : 2, 'ball2.npy' : 2, 'ball3.npy' : 2, 'ball4.npy' : 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

    dmaxs = {'cup1_maxs.npy' : 1, 'cup2_maxs.npy' : 1, 'cup3_maxs.npy' : 1, 'cup4_maxs.npy' : 1, 'cup5_maxs.npy' : 1, 'ball1_maxs.npy' : 2, 'ball2_maxs.npy' : 2, 'ball3_maxs.npy' : 2, 'ball4_maxs.npy' : 2, 'ball5_maxs.npy' : 2, 'banana1_maxs.npy' : 5, 'banana2_maxs.npy' : 5, 'banana3_maxs.npy' : 5, 'banana4_maxs.npy' : 5, 'banana5_maxs.npy' : 5, 'box1_maxs.npy' : 4, 'box2_maxs.npy' : 4, 'box3_maxs.npy' : 4, 'box4_maxs.npy' : 4, 'box5_maxs.npy' : 4, 'bottle1_maxs.npy' : 3, 'bottle2_maxs.npy' : 3, 'bottle3_maxs.npy' : 3, 'bottle4_maxs.npy' : 3, 'bottle5_maxs.npy' : 3}

    # X, Y = load_and_pack_data(dnames, 100, 100)
    X, Y = load_and_pack_data(dnames, 300, 300)
    # plot_projections(X, Y, 'Original Data')
    
    X_pca, pca = reduce_pca(X, 100, True)    
    # plot_projections(X_pca, Y, 'Projection by PCA')

    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_))
    # plt.title('Explained Variance')
    # plt.xlabel('Component')
    # plt.ylabel('Variance')
    # plt.grid

    X_kpca, X_back = reduce_kpca(X, 'rbf')    
    # plot_projections(X_kpca, Y, 'Projection by Kernel PCA')
    # plot_projections(X_back, Y, 'Projection by KPCA Backprojected')

    X_nmf, nmf_cmps = reduce_nmf(X, 20)    
    # plot_projections(X_nmf, Y, 'NMF Projection ')
    # print nmf_cmps


    # testing 2 class svm
    c1 = Y == 1
    c2 = Y != 1

    # cby = np.append(Y[c1], Y[c2], axis=0)

    # print X[c1].shape, Y[c1].shape, X[c2].shape, Y[c2].shape
    
    # cbx, cby = balance_classes(X[c1], Y[c1], X[c2], Y[c2])

    # print cbx.shape, cby.shape
    # classify_svm(cbx, cby, 'Original Data SVM', 1)
    
    
    # cbx, cby = balance_classes(X_kpca[c1], Y[c1], X_kpca[c2], Y[c2])
    # classify_svm(X_kpca, Y, 'KPCA SVM', 1)

    # cbx = np.append( X_pca[c1], X_pca[c2], axis=0)
    # classify_svm(cbx, cby, 'PCA SVM', 1)

    # cbx = np.append( X_nmf[c1], X_nmf[c2], axis=0)
    # classify_svm(cbx, cby, 'NMF SVM', 1)

    # classify_knn(X, Y, X, 'euclidean')

    classify_trees(X, Y, 'forrest')
    # classify_svm(X, Y, 'Original', 1)

    # clean_data(X)
    # plt.show(block=True)

