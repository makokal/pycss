
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm


def rebin(a, shape):
    # resize a numpy 2d array to given size
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_and_pack_data(data, lenx, leny):
    # get the data and pack it as [Training Labels]
    Xdata = np.zeros(shape=(len(data), (lenx*leny)))
    Ydata = np.zeros(len(data))

    idx = 0
    for f, v in data.items():
        # print f, v, idx
        d = np.load(f)
        Xdata[idx, :] = rebin(d, [lenx, leny]).reshape(lenx*leny)
        Ydata[idx] = v
        idx += 1
    # print Ydata
    return Xdata, Ydata


dnames = {'cup1.npy': 1, 'cup2.npy': 1, 'cup3.npy': 1, 'cup4.npy': 1, 'cup5.npy': 1, 'ball1.npy': 2, 'ball2.npy': 2, 'ball3.npy': 2, 'ball4.npy': 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

X, Y = load_and_pack_data(dnames, 50, 50)

# we create 40 separable points
# np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()