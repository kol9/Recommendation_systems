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
ds_result_test = ds_result_pref + '.test.rating'
ds_result_neg = ds_result_pref + '.test.negative'

items_set = set()
to_write = []
mx = -1
user_old_id_to_new_id = {}

with open(ds_train) as f:
    line = f.readline().strip()

    last_real = 0
    last_id = 0
    first = True

    while line != '':
        first_tab = line.find('\t')
        user_id = line[:first_tab]

        if random.uniform(0, 1) < want_size / ds_train_size:
            if user_id != last_real:
                last_id += 1

            if first:
                last_id = 0
                first = False

            final_id = last_id
            user_old_id_to_new_id[user_id] = final_id
            mx = max(mx, final_id)

            second_tab = line[first_tab + 1:].index('\t')
            item_id = int(line[first_tab + 1:][:second_tab])
            items_set.add(item_id)

            to_write.append([final_id, item_id, line[first_tab + 1:][second_tab:]])

            last_real = user_id
        line = f.readline().strip()

cur_id = 0
item_old_id_to_new_id = {}
for item in sorted(items_set):
    item_old_id_to_new_id[item] = cur_id
    cur_id += 1

print(len(item_old_id_to_new_id), mx + 1)

with open(ds_result_train, 'w') as res_file:
    for line in to_write:
        final_id, item_id, rest = line
        res_file.write(str(final_id) + '\t' + str(item_old_id_to_new_id[item_id]) + rest + '\n')

with open(ds_test) as f:
    with open(ds_result_test, 'w') as res_file:
        line = f.readline().strip()

        while line != '':
            user_id, item_id, rating, timestamp = line.split('\t')
            user_id, item_id = int(user_id), int(item_id)

            if user_id in user_old_id_to_new_id and item_id in item_old_id_to_new_id:
                res_file.write('\t'.join(map(str, [user_old_id_to_new_id[user_id], item_old_id_to_new_id[item_id],
                                                   rating, timestamp])) + '\n')

            line = f.readline().strip()
