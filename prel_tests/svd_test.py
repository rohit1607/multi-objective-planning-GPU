import numpy as np
m, n = 9, 6
a = np.random.randn(m,n)
u, s, vh = np.linalg.svd(a, full_matrices=True)
n_modes = 3
u1 = u[:, 0:n_modes]
s1_mat = np.diag(s[0:n_modes])
vh1 = vh[0:n_modes, :]
a1 = np.matmul(u1, np.matmul(s1_mat, vh1))

a1_mean = np.mean(a1, axis = 1)

print(s)
print(u.shape, s.shape, vh.shape)
print(u1.shape, s1_mat.shape, vh1.shape)

print(a.shape)
print(a1.shape)
print(a1_mean.shape)
