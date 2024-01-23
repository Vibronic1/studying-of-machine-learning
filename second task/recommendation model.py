import scipy as sc
import numpy as np

# Define the user-movie matrix
def svd_only_n_str(X, n):
    U, s, V = sc.linalg.svd(X)
    Unstr = only_n_str(U, n)
    snstr = only_n_str(s, n)
    Vnstr = only_n_str(V, n)
    return Unstr, snstr, Vnstr
def only_n_str(X, n):
    t = []
    for i in range(n):
        t.append(X[i])
    return np.array(t)
def find_max_rate_index_no_watch_film(user_vector, inv_trans):
    non_watched_max_value = -13131
    index = -100
    for i in range(len(inv_trans)):
        if user_vector[i] == 0 and inv_trans[i] > non_watched_max_value:
            non_watched_max_value = inv_trans[i]
            index = i
    return inv_trans[index], index
user_movie_matrix = np.array(((1, 5, 0, 5, 4), (5, 4, 4, 3, 2), (0, 4, 0, 0, 5), (4, 4, 1, 4, 0), (0, 4, 3, 5, 0), (2, 4, 3, 5, 3)))

# Perform SVD with n=2 and calculate recommendations for user
n = 2
Unstr, snstr, Vnstr = svd_only_n_str(user_movie_matrix, n)
new_user = np.array((0, 0, 3, 4, 0))
print('n: ', n)
VT = Vnstr.T
print('VT: \n', VT)
lowdim = np.matmul(user_movie_matrix[2], VT)
inv_trans_2 = np.matmul(lowdim, Vnstr)
rate, max_index = find_max_rate_index_no_watch_film(user_movie_matrix[2], inv_trans_2)
print('\nrate: ', inv_trans_2, '\nmax_rate: ', rate, '\nfilm_index: ', max_index)
print('\n')

# Perform SVD with n=2 and calculate recommendations for a new user
n = 2
Unstr, snstr, Vnstr = svd_only_n_str(user_movie_matrix, n)
new_user = np.array((0, 0, 3, 4, 0))
VT = Vnstr.T
print('n: ', n)
print('VT: \n', VT)
lowdim_new_user = np.matmul(new_user, VT)
inv_trans_new_user = np.matmul(lowdim_new_user, Vnstr)
rate_new, max_index_new = find_max_rate_index_no_watch_film(new_user, inv_trans_new_user)
print('\nrate_new: ', inv_trans_new_user, '\nmax_rate_new: ', rate_new, '\nfilm_index_new: ', max_index_new)
print('\n')

# Perform SVD with n=3 and calculate recommendations for user
n = 3
Unstr, snstr, Vnstr = svd_only_n_str(user_movie_matrix, n)
print('n: ', n)
VT = Vnstr.T
print('VT: \n', VT)
lowdim = np.matmul(user_movie_matrix[2], VT)
inv_trans_2 = np.matmul(lowdim, Vnstr)
rate, max_index = find_max_rate_index_no_watch_film(user_movie_matrix[2], inv_trans_2)
print('\nrate: ', inv_trans_2, '\nmax_rate: ', rate, '\nfilm_index: ', max_index)
