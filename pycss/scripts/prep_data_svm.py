# Prep data dn run svm

import numpy as np
import pylab as plt
from scikits.learn import svm
from scikits.learn.decomposition import PCA, KernelPCA

def rebin(a, shape):
    # resize a numpy 2d array
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_and_pack_data(data, lenx,leny):
    # get the data and pack it

    Xdata = np.zeros(shape=(len(data), (lenx*leny)))
    Ydata = np.zeros(len(data))

    idx = 0
    for f, v in data.items():
        print f, v, idx
        d = np.load(f)
        Xdata[idx, :] = rebin(d, [lenx,leny]).reshape(lenx*leny)
        Ydata[idx] = v
        idx += 1
    # print Ydata
    return Xdata, Ydata



if __name__ == '__main__':
    # file names and labels
    dnames = {'cup1.npy' : 1, 'cup2.npy' : 1, 'cup3.npy' : 1, 'cup4.npy' : 1, 'cup5.npy' : 1, 'ball1.npy' : 2, 'ball2.npy' : 2, 'ball3.npy' : 2, 'ball4.npy' : 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

    X, Y = load_and_pack_data(dnames, 20, 20)
    print Y
    XX = X[0:5, :]  # take cups and balls only

    """ basic svm classification """
    # asave = np.zeros(shape=(len(dnames), 10001))
    # for i in range(6):
    #   asave[i, :] = np.append(Y[i], X[i, :])

    # np.savetxt('training.dat', asave, delimiter=' ')
    np.savetxt('training.dat', X.T, delimiter=' ')
    np.savetxt('test.dat', X.T, delimiter=' ')

    # print X.shape, Y.shape
    C = 1.0
    svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    print svc

    # # create a mesh to plot in
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    print xx.ravel().shape, yy.ravel().shape
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # plt.axis('off')

    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)


    """ testing rescale by averaging """
    # load some test data for checking resizing
    # c = np.load('cup3.npy')
    # small_c = rebin(c, [100,100])
    # plt.matshow(small_c, cmap=plt.cm.spectral)
    # plt.show(block=True)
