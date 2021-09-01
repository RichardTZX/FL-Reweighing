'''
splits data into train and test sets
'''
import json
import os

from collections import OrderedDict

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dir = os.path.join(parent_path, 'compas', 'data')
subdir = os.path.join(dir, 'train')
gandir = os.path.join(dir, 'all_data')

files, ganfile = [], []
if os.path.exists(subdir):
    files = os.listdir(subdir)
if os.path.exists(gandir):
    ganfile = os.listdir(gandir)
if len(files) == 0:
    print("Error data_train.json must be generated before using gan_data.py")
files = [f for f in files if f.endswith('.json')]

ganfile = [f for f in ganfile if f.endswith('gan.json')]

for f in files:
    file_dir = os.path.join(subdir, f)
    gan_file_dir = os.path.join(gandir,ganfile[0])
    with open(file_dir, 'r') as inf:
        # Load data into an OrderedDict, to prevent ordering changes
        # and enable reproducibility
        data = json.load(inf, object_pairs_hook=OrderedDict)

    with open(gan_file_dir, 'r') as infg:
        data_gan = json.load(infg)

    num_samples_train = []
    fictive_user = data_gan["users"]

    for u in data['users']: 
        data['user_data'][u]['x'] = data['user_data'][u]['x'] + data_gan['user_data']['x']
        data['user_data'][u]['y'] = data['user_data'][u]['y'] + data_gan['user_data']['y']

        num_train_samples = len(data['user_data'][u]['y'])
        num_samples_train.append(num_train_samples)


    data['num_samples'] = num_samples_train

    file_name_train = 'gan_%s_train_%s.json' % ((f[:-5]), 9)
    ouf_dir_train = os.path.join(dir, 'train', file_name_train)
    print('writing %s' % file_name_train)
    print("added {} GAN samples to every clients".format(len(data_gan['user_data']['y'])))
    with open(ouf_dir_train, 'w') as outfile:
        json.dump(data, outfile)

