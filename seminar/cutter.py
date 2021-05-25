import os
import random

ds_name = 'ml-1m'
ds_train_size = 994169
want_size = 150000

# ds_name = 'pinterest-20'
# ds_train_size = 1445622
# want_size = 150000

ds_dir = 'ds'
ds_pref = ds_dir + os.sep + ds_name
ds_result_pref = ds_dir + '_cut' + os.sep + ds_name

ds_train = ds_pref + '.train.rating'
ds_test = ds_pref + '.test.rating'
ds_neg = ds_pref + '.test.negative'

ds_result_train = ds_result_pref + '.train.rating'
ds_result_test = ds_result_pref + '.test.rating'
ds_result_neg = ds_result_pref + '.test.negative'

mx_u = -1
mx_i = -1
mx_u_all = []
mx_i_all = []

with open(ds_train) as f:
    with open(ds_result_train, 'w') as res_f:
        line = f.readline().strip()

        while line != '':
            u, i, r, t = line.split('\t')
            u, i = int(u), int(i)

            if u > mx_u:
                mx_u = u
                mx_u_all = [u, i, r, t]

            if i > mx_i:
                mx_i = i
                mx_i_all = [u, i, r, t]

            if random.uniform(0, 1) < want_size / ds_train_size:
                res_f.write('\t'.join(map(str, [u, i, r, t])) + '\n')

            line = f.readline().strip()

with open(ds_result_train, 'a') as res_f:
    res_f.write('\t'.join(map(str, mx_u_all)) + '\n')
    res_f.write('\t'.join(map(str, mx_i_all)) + '\n')

print(mx_u, mx_i)
