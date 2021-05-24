import os
import random

# ds_name = 'ml-1m'
# ds_train_size = 994169
# want_size = 150000

ds_name = 'pinterest-20'
ds_train_size = 1445622
want_size = 150000

ds_dir = 'ds'
ds_pref = ds_dir + os.sep + ds_name
ds_result_pref = ds_dir + '_cut' + os.sep + ds_name

ds_train = ds_pref + '.train.rating'
ds_test = ds_pref + '.test.rating'

ds_neg = ds_pref + '.test.negative'

ds_result_train = ds_result_pref + '.train.rating'

with open(ds_train) as f:
    with open(ds_result_train, 'w') as res_file:
        line = f.readline().strip()
        while line != '':
            if random.randint(1, ds_train_size) < want_size:
                res_file.write(line + '\n')
            line = f.readline().strip()
