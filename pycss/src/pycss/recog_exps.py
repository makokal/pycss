
# simple recognition experiments

import numpy as np
import pylab as plt
from scipy import interp
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA, KernelPCA, NMF


def rebin(a, shape):
    # resize a numpy 2d array to given size
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


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


def classify_svm(X, Y):

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

	C = 1.0
	svc = svm.SVC(kernel='rbf', C=C, degree=5, gamma=0.7, probability=True, verbose=False)
	svc.fit(X_train, y_train)
	print svc
	

	cv = StratifiedKFold(Y, n_folds=4)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	for i, (train, test) in enumerate(cv):
		probas_ = svc.fit(X[train], Y[train]).predict_proba(X[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1], pos_label=5)
    	mean_tpr += interp(mean_fpr, fpr, tpr)
    	mean_tpr[0] = 0.0
    	roc_auc = auc(fpr, tpr)
    	plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.legend(loc="lower right")
	plt.show(block=True)


	# print svc.score(X_test, y_test)
	# print svc.score(X, Y)

	# scores = cross_validation.cross_val_score(svc, X, Y, cv=3)
	# print scores


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
	plt.legend(['Coffee cup', 'Football', 'Beer bottle', 'Cereal box', 'Banana'])





if __name__ == '__main__':
    # load data from current dir
    dnames = {'cup1.npy' : 1, 'cup2.npy' : 1, 'cup3.npy' : 1, 'cup4.npy' : 1, 'cup5.npy' : 1, 'ball1.npy' : 2, 'ball2.npy' : 2, 'ball3.npy' : 2, 'ball4.npy' : 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

    d_train = {'cup1.npy' : 1, 'cup2.npy' : 1, 'cup3.npy' : 1, 'cup4.npy' : 1, 'cup5.npy' : 1, 'ball1.npy' : 2, 'ball2.npy' : 2, 'ball3.npy' : 2, 'ball4.npy' : 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

    X, Y = load_and_pack_data(dnames, 20, 20)
    
    # X_pca = reduce_pca(X, 30)    
    # plot_projections(X_pca, Y, 'Projection by PCA')

    # X_kpca, X_back = reduce_kpca(X, 'rbf')    
    # plot_projections(X_kpca, Y, 'Projection by KPCA in Feature space')
    # plot_projections(X_back, Y, 'Projection by KPCA Backprojected')

    # X_nmf, nmf_cmps = reduce_nmf(X, 20)    
    # plot_projections(X_nmf, Y, 'NMF Projection')
    # print nmf_cmps


    # plt.show(block=True)

    # testing svm
    print Y[0:9]
    classify_svm(X[0:9], Y[0:9])

