import numpy as np

A = np.array([[-0.13793103448275856, 4.206896551724137], [1.0, 0.0], [0.0, 1.0]])

b = np.array([896.5517241379312, 6000.0, 500.0])
b = b.reshape((-1, 1))

Ab = np.concatenate((A, b), axis=1)
np.savetxt("rc_test_data.csv", Ab)
