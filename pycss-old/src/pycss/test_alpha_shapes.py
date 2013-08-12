
import numpy as np
from pylab import *
import subprocess


def get_alpha_shape(infile,radius):
    command = "%s -A -aa %s -r -m10000000 -oN -oFpoints < %s" % (hull_path, str(radius), infile)
    print >> sys.stderr, "Running command: %s" % command
    retcode = subprocess.call(command, shell=True)
    results_file = open("hout-alf")
    results_file.next() 
    results_indices = [[int(i) for i in line.rstrip().split()] for line in results_file]
    results_file.close()
    return results_indices


# t = np.linspace(0.6, 5.7, 500)
# C = 2 * np.vstack([np.cos(t), np.sin(t)])
# C = C + np.random.rand(2, 500)
C = np.loadtxt('../../../../../mThesis/datasets/experiments/balls/ex5/slice150.txt', unpack=False).T


hull_path = "/usr/local/bin/hull.exe"
infile = '../../../../../mThesis/datasets/experiments/balls/ex5/slice150.txt'
# with open(infile, 'w') as f:
   # np.savetxt(f, C.T, delimiter=' ', fmt="%0.7f %0.7f\n")

radius = .2
indices = get_alpha_shape(infile, radius)
# print 'Indices:', indices
print 'C shape', C.shape

# cv = np.zeros(shape=(2, len(indices)))
cv = []
for ind in (indices):
    cv.append( [ C[0, ind[0]], C[1, ind[1]] ] )
    # cv[0, i] = C[0, ind[0]]
    # cv[0, i] = C[1, ind[1]]

tl = len(cv)
cvv = np.array(cv).T
print cvv
# alpha = C.T[indices]
plot(C[0, 0:tl], 'r'), plot(C[1, 0:tl],'b'), plot(cvv[0, :],'g'), plot(cvv[1, :],'m' )

figure()
plot(C[0], C[1],  '.-')
plot(cvv[0, :], cvv[1, :], lw=1, color='r')
show(block=True)

# plot(alpha[1:].T[0],alpha[1:].T[1],'r')