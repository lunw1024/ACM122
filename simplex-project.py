import numpy as np

def simplex_project(y):
	'''
	project vector y onto the standard simplex
	reference: https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf 

	Argument:
		y: a 1-D list

	Output:
		x: a 1-D list, same length as y
	'''
	assert isinstance(y, list) and len(y)>0
	n = len(y)
	u = np.sort(y)[::-1]
	partial_sum = np.cumsum(u)
	max_j = ([j for j in range(n) if u[j] + 1.0 / (j + 1) * (1 - partial_sum[j]) > 0])[-1]
	x = [max(0, e + 1.0 / (max_j + 1) * (1 - partial_sum[max_j])) for e in y]
	return x