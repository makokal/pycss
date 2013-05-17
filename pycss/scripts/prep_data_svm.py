# Prep data dn run svm

import numpy as np
import pylab as plt
from scikits.learn import svm

def rebin(a, shape):
	# resize a numpy 2d array
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_and_pack_data(data):
	# get the data and pack it

	Xdata = np.zeros(shape=(len(data), 10000))
	Ydata = np.zeros(len(data))

	idx = 0
	for f, label in data.items():
		d = np.load(f)
		Xdata[idx, :] = rebin(d, [100,100]).reshape(10000)
		Ydata[idx] = label
		idx += 1

	return Xdata, Ydata



if __name__ == '__main__':
	# file names and labels
    dnames = {'cup2.npy' : 1, 'cup3.npy' : 1, 'cup3.npy' : 1, 'ball1.npy' : 2, 'ball1.npy' : 2, 'ball4.npy' : 2, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana5.npy' : 5}


    ## basic svm classification
    X, Y = load_and_pack_data(dnames)
    asave = np.zeros(shape=(len(dnames), 10001))
    for i in range(6):
    	asave[i, :] = np.append(Y[i], X[i, :])

    np.savetxt('training.dat', asave, delimiter=' ')
    np.savetxt('test.dat', X, delimiter=' ')

    print X.shape, Y.shape
    C = 1.0
    svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    print svc

    # create a mesh to plot in
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    print xx.ravel().shape, yy.ravel().shape
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)



    # # load some test data for checking resizing
    # c = np.load('cup3.npy')
    # small_c = rebin(c, [100,100])
    # plt.matshow(small_c, cmap=plt.cm.spectral)

    plt.show(block=True)