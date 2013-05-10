# CSS relared features computation

import numpy as np

def find_zero_crossings(kappa):
	""" find_zero_crossings(kappa)
	Locate the zero crossing points of the curvature kappa
	"""
	
	crossings = []

	for i in range(0, kappa.size-2):
		if (kappa[i] < 0.0 and kappa[i+1] > 0.0 ) or \
			(kappa[i] > 0.0 and kappa[i+1] < 0.0 ):
			crossings.append(i)

	return crossings