import numpy as np
from pylab import *
from scikits.learn.decomposition import PCA, KernelPCA


def rebin(a, shape):
    # resize a numpy 2d array
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def load_and_pack_data(data, lenx, leny):
    # get the data and pack it

    Xdata = np.zeros(shape=(len(data), (lenx * leny)))
    Ydata = np.zeros(len(data))

    idx = 0
    for f, v in data.items():
        print f, v, idx
        d = np.load(f)
        Xdata[idx, :] = rebin(d, [lenx, leny]).reshape(lenx * leny)
        Ydata[idx] = v
        idx += 1
    print Ydata
    return Xdata, Ydata



dnames = {'cup1.npy': 1, 'cup2.npy': 1, 'cup3.npy': 1, 'cup4.npy': 1, 'cup5.npy': 1, 'ball1.npy': 2, 'ball2.npy': 2, 'ball3.npy' : 2, 'ball4.npy' : 2, 'ball5.npy' : 2, 'banana1.npy' : 5, 'banana2.npy' : 5, 'banana3.npy' : 5, 'banana4.npy' : 5, 'banana5.npy' : 5, 'box1.npy' : 4, 'box2.npy' : 4, 'box3.npy' : 4, 'box4.npy' : 4, 'box5.npy' : 4, 'bottle1.npy' : 3, 'bottle2.npy' : 3, 'bottle3.npy' : 3, 'bottle4.npy' : 3, 'bottle5.npy' : 3}

X, Y = load_and_pack_data(dnames, 20, 20)

# PCA and Kernel PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
print 'done simple pca'

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
print 'fitted kernel pca'
X_back = kpca.inverse_transform(X_kpca)
print 'done back transforming with kpca'


# plots
reds = Y == 1
blues = Y == 2
greens = Y == 3
magentas = Y == 4
yellows = Y == 5

print 'plotting the results'
plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
plot(X_kpca[greens, 0], X_kpca[greens, 1], "go")
plot(X_kpca[magentas, 0], X_kpca[magentas, 1], "mo")
plot(X_kpca[yellows, 0], X_kpca[yellows, 1], "yo")
title('Projection by KPCA')

figure()
plot(X_pca[reds, 0], X_pca[reds, 1], "ro")
plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
plot(X_pca[greens, 0], X_pca[greens, 1], "go")
plot(X_pca[magentas, 0], X_pca[magentas, 1], "mo")
plot(X_pca[yellows, 0], X_pca[yellows, 1], "yo")
title('Projection by PCA')

show(block=True)