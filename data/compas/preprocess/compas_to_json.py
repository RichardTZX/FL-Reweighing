import json
import numpy as np
import os
import argparse

import pandas as pd

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main():
    args=parse_args()
    np.random.seed(args.seed)
    print("Preprocessing COMPAS dataset ...")
    compas_preprocessed_dir = os.path.join(parent_path, 'data', "compas_preprocessed.csv")
    if os.path.exists(compas_preprocessed_dir):
        print("Readind compas_preprocessed.csv ...")
        df_compas = pd.read_csv(compas_preprocessed_dir)
        labels = df_compas['class']
        df_compas = df_compas.drop("class", axis=1)
        print(df_compas.head())
    else:
        df_compas, labels = get_compas_clear()
    num_workers = args.num_workers
    heterog = args.heterog
    gan, niid, sensattr  = args.gan, args.niid, args.sens_attr

    dir_path = os.path.join(parent_path, 'data', 'all_data')

    if gan:
        if niid:
            users, num_samples, user_data = to_leaf_format_gan_100_0(df_compas, labels, num_workers, sensattr)

            gan_data = user_data[str(num_workers)]
            del user_data[str(num_workers)]

            save_json(dir_path, 'data.json', users[:-1], num_samples[:-1], user_data)
            save_json(dir_path, 'gan.json', users[-1], num_samples[-1], gan_data)
            print("data.json ready with the compas-scores-two-years.csv data distributed to {} workers and gan.json with the little subset of data that every worker will share".format(num_workers))
            print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")
        else:
            users, num_samples, user_data = to_leaf_format_gan(df_compas, labels, num_workers)

            gan_data = user_data[str(num_workers)]
            del user_data[str(num_workers)]

            save_json(dir_path, 'data.json', users[:-1], num_samples[:-1], user_data)
            save_json(dir_path, 'gan.json', users[-1], num_samples[-1], gan_data)
            print("data.json ready with the compas-scores-two-years.csv data distributed to {} workers and gan.json with the little subset of data that every worker will share".format(num_workers))
            print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")
    else:
        if niid:
            users, num_samples, user_data = to_leaf_format_100_0(df_compas, labels, num_workers, sensattr)
        elif heterog:
            users, num_samples, user_data = to_leaf_format_het(df_compas, labels, num_workers)
        else:
            users, num_samples, user_data = to_leaf_format(df_compas, labels, num_workers)
            print('Standard distribution among the workers.')

        save_json(dir_path, 'data.json', users, num_samples, user_data)
        print("data.json ready with the compas-scores-two-years.csv data distributed to {} workers".format(num_workers))
        print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")


def get_compas_clear(): #Load dataset / Preprocessing from IBM github 
    data_dir = os.path.join(parent_path, 'data', 'compas-scores-two-years.csv')

    # COMPAS PREPROCESSING DESCRIBED IN 
    # Abey et al. - Mitigating Bias in Federated Learning

    df = pd.read_csv(data_dir)
    df['class'] = df['two_year_recid']
    df = df.drop('two_year_recid', axis=1)
    
    # map 'sex' feature values based on sensitive attribute privileged/unprivileged groups
    df['sex'] = df['sex'].map({'Female': 1, 'Male': 0})
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(float)

    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix
    ix = (df['is_recid'] != -1) & ix
    ix = (df['c_charge_degree'] != "O") & ix
    ix = (df['score_text'] != 'N/A') & ix
    df = df.loc[ix, :]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) -
                                            pd.to_datetime(df['c_jail_in'])).apply(
                                            lambda x: x.days)

    # filter out columns unused in training, and reorder columns
    df = df.loc[~df['race'].isin(
        ['Native American', 'Hispanic', 'Asian', 'Other']), :]
    df = df[['sex', 'race', 'age_cat', 'c_charge_degree',
                                            'score_text', 'priors_count', 'is_recid',
                                            'length_of_stay', 'class']]
    df['priors_count'] = df['priors_count'].astype(int)

    # Quantize priors count between 0, 1-3, and >3
    def quantizePrior(x):
        col = []
        for i in x:
            if i <= 0:
                col.append('0')
            elif 1 <= i <= 3:
                col.append('1 to 3')
            else:
                col.append('More than 3')
        return col

    # Quantize length of stay
    def quantizeLOS(x):
        col = []
        for i in x:
            if i <= 7:
                col.append('<week')
            elif 8 <= i <= 93:
                col.append('<3months')
            else:
                col.append('>3 months')
        return col

    # Quantize length of stay
    def adjustAge(x):
        col = []
        for i in x:
            if i == '25 - 45':
                col.append('25 to 45')
            else:
                col.append(i)
        return col

    # Quantize score_text to MediumHigh
    def quantizeScore(x):
        col = []
        for i in x:
            if (i == 'High') | (i == 'Medium'):
                col.append('MediumHigh')
            else:
                col.append(i)
        return col

    # Map race to 0/1 based on unprivileged/privileged groups
    def group_race(x):
        col = []
        for i in x:
            if i == "Caucasian":
                col.append(1)
            else:
                col.append(0)
        return col

    def flipclass(x):
        col = []
        for i in x:
            if i == 1:
                col.append(0)
            if i == 0:
                col.append(1)
        return col

    df['priors_count'] = quantizePrior(df['priors_count'])
    df['length_of_stay'] = quantizeLOS(df['length_of_stay'])
    df['score_text'] = quantizeScore(df['score_text'])
    df['age_cat'] = adjustAge(df['age_cat'])
    df['race'] = group_race(df['race'])
    df['class'] = flipclass(df['class'])

    new_cols = ['age_cat = 25 to 45', 'age_cat = Greater than 45', 'age_cat = Less than 25', 'priors_count = 0',
                'priors_count = 1 to 3', 'priors_count = More than 3', 'c_charge_degree = F', 'c_charge_degree = M']

    for i in new_cols:
        df[i] = 0

    for index, row in df.iterrows():
        if row['age_cat'] == '25 to 45':
            df.loc[index, 'age_cat = 25 to 45'] = 1
        elif row['age_cat'] == 'More than 45':
            df.loc[index, 'age_cat = Greater than 45'] = 1
        elif row['age_cat'] == 'Less than 45':
            df.loc[index, 'age_cat = Less than 25'] = 1

    for index, row in df.iterrows():
        if row['priors_count'] == '0':
            df.loc[index, 'priors_count = 0'] = 1
        elif row['priors_count'] == '1 to 3':
            df.loc[index, 'priors_count = 1 to 3'] = 1
        elif row['priors_count'] == 'More than 3':
            df.loc[index, 'priors_count = More than 3'] = 1

    for index, row in df.iterrows():
        if row['c_charge_degree'] == "F":
            df.loc[index, 'c_charge_degree = F'] = 1
        elif row['c_charge_degree'] == "M":
            df.loc[index, 'c_charge_degree = M'] = 1

    df = df.drop(
        ['age_cat', 'priors_count', 'c_charge_degree', 'is_recid', 'score_text', 'length_of_stay'], axis=1)

    label = df['class']
    df.drop(['class'], axis=1, inplace=True)
    df['class'] = label


    df.to_csv("compas_preprocessed.csv", index_label= False) # Save compas preprocessing in a .csv in order to load it faster than preprocess the dataset every time
    
    print(df.head())
    return(df,label)

def get_indices_workers(df, num_workers): #Generate the samples indices for each worker
    mean = int(len(df) / num_workers)
    indices = np.array(df.index) #Indices of the samples
    np.random.shuffle(indices)
    indices_workers = [indices[i*mean:(i+1)*mean] for i in range(num_workers)] #For each worker, we have a list of the samples he possesses
    return indices_workers

def get_indices_workers_het(df, num_workers): #Generate the samples indices for each worker
    num_samples = len(df)
    mean1 = int(num_samples / (2*0.8*num_workers)) #80% of users will hold 50% of data
    mean2 = int(num_samples / (2*0.2*num_workers)) #20% of the rest will hold the 50% remaining data

    indices = np.array(df.index) # Indices of the samples
    np.random.shuffle(indices)

    data_remaining = num_samples # Number of samples undistributed remaining in the dataset
    num_samples_workers =  [0]
    for i in range(num_workers-1):
        if i < 0.8*num_workers + 1: # 80% users' data
            num_samples_i = np.random.poisson(mean1)
            if data_remaining - num_samples_i < 0:
                num_samples_i = data_remaining
                break
            num_samples_workers.append(num_samples_i)
            data_remaining -= num_samples_i
        else: # 20% remaining users' data
            num_samples_i = np.random.poisson(mean2)
            if data_remaining - num_samples_i < 0:
                num_samples_i = data_remaining
                break

            num_samples_workers.append(num_samples_i)
            data_remaining -= num_samples_i

    num_samples_workers.append(data_remaining)
    make_indices = np.cumsum(num_samples_workers)
    indices_workers = [indices[make_indices[i]:make_indices[i+1]] for i in range(len(make_indices)-1)] #For each worker, we have a list of the samples he possesses
    return indices_workers

def gan_indices(df,indices,size = 2):

    '''Choose samples that will be in the little subset simulating the GAN dataset.
    We take 'size' samples of each value of sensitive attributes in order to have diversity in the little subset of data. '''

    gan_ind = []
    cnt1, cnt2, cnt3, cnt4 = 0,0,0,0
    i = 0
    while len(gan_ind) < 4 * size:
        if df['sex'].loc[indices[i]] == 0 and df['race'].loc[indices[i]] == 0 and cnt1 < size:
            gan_ind.append(indices[i])
            cnt1 += 1
        elif df['sex'].loc[indices[i]] == 0 and df['race'].loc[indices[i]] == 1 and cnt2 < size:
            gan_ind.append(indices[i])
            cnt2 += 1
        elif df['sex'].loc[indices[i]] == 1 and df['race'].loc[indices[i]] == 0 and cnt3 < size:
            gan_ind.append(indices[i])
            cnt3 += 1
        elif df['sex'].loc[indices[i]] == 1 and df['race'].loc[indices[i]] == 1 and cnt4 < size:
            gan_ind.append(indices[i])
            cnt4 += 1
        i+=1
    return gan_ind

def get_indices_workers_gan(df, num_workers):
    num_samples = len(df)
    mean1 = int(num_samples / (2*0.8*num_workers)) #80% of users will hold 50% of data
    mean2 = int(num_samples / (2*0.2*num_workers)) #20% of the rest will hold the 50% remaining data

    indices = np.array(df.index) # Indices of the samples
    np.random.shuffle(indices)

    # Take a litte subset of data from the original dataset that will be shared by every workers (simulate data generated by a GAN)
    gan_ind = gan_indices(df, indices)
    delete_indices = []
    for i in range(len(gan_ind)):
        delete_indices.append(np.where(indices == gan_ind[i])[0][0])
    indices = np.delete(indices, delete_indices) # Remove the little subset from the 'to be distributed' data

    data_remaining = num_samples - len(gan_ind) # Number of samples undistributed remaining in the dataset
    num_samples_workers =  [0]
    for i in range(num_workers-1):
        if i < 0.8*num_workers + 1: # 80% users' data
            num_samples_i = np.random.poisson(mean1)
            if data_remaining - num_samples_i < 0:
                num_samples_i = data_remaining
                break
            num_samples_workers.append(num_samples_i)
            data_remaining -= num_samples_i
        else: # 20% remaining users' data
            num_samples_i = np.random.poisson(mean2)
            if data_remaining - num_samples_i < 0:
                num_samples_i = data_remaining
                break

            num_samples_workers.append(num_samples_i)
            data_remaining -= num_samples_i

    num_samples_workers.append(data_remaining)
    make_indices = np.cumsum(num_samples_workers)
    indices_workers = [indices[make_indices[i]:make_indices[i+1]] for i in range(len(make_indices)-1)] #For each worker, we have a list of the samples he possesses

    # At the end of the workers' dataset we add a fictive worker that owns the little subset of data that everyone will share
    indices_workers.append(gan_ind)

    return indices_workers

def get_indices_workers_100_0(df, num_workers, sensattr): # N_samples race attribute ['0' : 6796, '1' : 41189]
    if sensattr == "race": 
        n0, n1 = 6796, 41189
    else: #sensattr == "gender"
        n0, n1 = 15944, 32041

    proportion_0 = n0 / (n0 + n1)
    nb_user_0 = int(proportion_0 * num_workers)
    nb_user_1 = num_workers - nb_user_0
    
    indices = np.array(df.index) # Indices of the samples
    
    samples0, samples1 = [], []

    for i in indices: # Detect/split samples with sensitive attribute value "0" and "1"
        if df[sensattr].loc[i] == 0: #TBD : sensitive attribute 
            samples0.append(i)
        else:
            samples1.append(i)

    mean0 = int(len(samples0) / nb_user_0) 
    mean1 = int(len(samples1) / nb_user_1) 

    np.random.shuffle(samples0)
    np.random.shuffle(samples1)

    data0_remaining = len(samples0) # Number of samples yet to be distributed to workers
    data1_remaining = len(samples1)

    num_samples_workers0, num_samples_workers1 =  [0], [0]
    for i in range(nb_user_0-1): # We determine the number of samples for each of the worker that will hold samples with sensitive attribute value "0"
        # We sample the number of samples for each of the worker except the last one that will have the remaining number of data
        num_samples_i = np.random.poisson(mean0) # Sample from a poisson distribution
        if data0_remaining - num_samples_i < 0:
            num_samples_i = data0_remaining
            num_samples_workers0.append(num_samples_i)
            break
        num_samples_workers0.append(num_samples_i) # Worker i will have num_samples_i samples in his local dataset
        data0_remaining -= num_samples_i
    if data0_remaining > 0: # If there are still data remaning then the last worker will have the remaining data otherwise we don't add a worker
        num_samples_workers0.append(data0_remaining)

    for i in range(nb_user_1-1): # And we determine the number of samples for each of the worker that will hold samples with sensitive attribute value "1"
        num_samples_i = np.random.poisson(mean1)
        if data1_remaining - num_samples_i < 0:
            num_samples_i = data1_remaining
            num_samples_workers1.append(num_samples_i)
            break
        num_samples_workers1.append(num_samples_i)
        data1_remaining -= num_samples_i
    if data1_remaining > 0:
        num_samples_workers1.append(data1_remaining)

    make_indices0 = np.cumsum(num_samples_workers0)
    make_indices1 = np.cumsum(num_samples_workers1)
    indices_workers0 = [samples0[make_indices0[i]:make_indices0[i+1]] for i in range(len(make_indices0)-1)] 
    indices_workers1 = [samples1[make_indices1[i]:make_indices1[i+1]] for i in range(len(make_indices1)-1)]
    #For each worker, we now have a list of the samples he possesses

    indices_workers = indices_workers0 + indices_workers1 # Concatenate '0' and '1' workers' indices 
    np.random.shuffle(indices_workers)

    return indices_workers

def get_indices_workers_gan_100_0(df, num_workers, sensattr): # N_samples race attribute ['0' : 6796, '1' : 41189] // N_samples gender/sex attribute ['0' : 15944, '1' : 32041]
    if sensattr == "race": 
        n0, n1 = 6796, 41189
    else: #sensattr == "gender"
        n0, n1 = 15944, 32041

    #For workers values of sensitive attribute we conserve the proportion from values of sensitive attribute in data
    proportion_0 = n0 / (n0 + n1)
    nb_user_0 = int(proportion_0 * num_workers)
    nb_user_1 = num_workers - nb_user_0
    
    indices = np.array(df.index) # Indices of the samples
    
    # Take a litte subset of data from the original dataset that will be shared by every workers (simulate data generated by a GAN)
    gan_ind = gan_indices(df, indices)
    delete_indices = []
    for i in range(len(gan_ind)):
        delete_indices.append(np.where(indices == gan_ind[i])[0][0])
    indices = np.delete(indices, delete_indices) # Remove the little subset from the 'to be distributed' data

    samples0, samples1 = [], []

    for i in indices: # Detect/split samples with sensitive attribute value "0" and "1"
        if df[sensattr].loc[i] == 0: 
            samples0.append(i)
        else:
            samples1.append(i)

    mean0 = int(len(samples0) / nb_user_0) 
    mean1 = int(len(samples1) / nb_user_1) 

    np.random.shuffle(samples0)
    np.random.shuffle(samples1)

    data0_remaining = len(samples0) # Number of samples yet to be distributed to workers
    data1_remaining = len(samples1)

    num_samples_workers0, num_samples_workers1 =  [0], [0]
    for i in range(nb_user_0-1): # We determine the number of samples for each of the worker that will hold samples with sensitive attribute value "0"
        # We sample the number of samples for each of the worker except the last one that will have the remaining number of data
        num_samples_i = np.random.poisson(mean0) # Sample from a poisson distribution
        if data0_remaining - num_samples_i < 0:
            num_samples_i = data0_remaining
            num_samples_workers0.append(num_samples_i)
            break
        num_samples_workers0.append(num_samples_i) # Worker i will have num_samples_i samples in his local dataset
        data0_remaining -= num_samples_i
    if data0_remaining > 0: # If there are still data remaning then the last worker will have the remaining data otherwise we don't add a worker
        num_samples_workers0.append(data0_remaining)

    for i in range(nb_user_1-1): # And we determine the number of samples for each of the worker that will hold samples with sensitive attribute value "1"
        num_samples_i = np.random.poisson(mean1)
        if data1_remaining - num_samples_i < 0:
            num_samples_i = data1_remaining
            num_samples_workers1.append(num_samples_i)
            break
        num_samples_workers1.append(num_samples_i)
        data1_remaining -= num_samples_i
    if data1_remaining > 0:
        num_samples_workers1.append(data1_remaining)

    make_indices0 = np.cumsum(num_samples_workers0)
    make_indices1 = np.cumsum(num_samples_workers1)
    indices_workers0 = [samples0[make_indices0[i]:make_indices0[i+1]] for i in range(len(make_indices0)-1)] 
    indices_workers1 = [samples1[make_indices1[i]:make_indices1[i+1]] for i in range(len(make_indices1)-1)]
    #For each worker, we now have a list of the samples he possesses

    indices_workers = indices_workers0 + indices_workers1 # Concatenate '0' and '1' workers' indices 
    np.random.shuffle(indices_workers)
    # At the end of the workers' dataset we add a fictive worker that owns the little subset of data that everyone will share
    indices_workers.append(gan_ind)
    return indices_workers


def to_leaf_format(df, labels, num_workers):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers(df,num_workers)

    for i, indices in enumerate(indices_workers):
        x, y = df.loc[indices].values.tolist(), labels[indices].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data

def to_leaf_format_het(df, labels, num_workers):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers_het(df,num_workers)

    for i, indices in enumerate(indices_workers):
        x, y = df.loc[indices].values.tolist(), labels[indices].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data

def to_leaf_format_gan(df, labels, num_workers):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers_gan(df,num_workers)

    for i, indices in enumerate(indices_workers):
        x, y = df.loc[indices].values.tolist(), labels[indices].tolist()
        print(x)
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data

def to_leaf_format_100_0(df, labels, num_workers, sensattr = "race"):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers_100_0(df,num_workers, sensattr)

    for i, indices in enumerate(indices_workers):
        x, y = df.loc[indices].values.tolist(), labels[indices].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data
    
def to_leaf_format_gan_100_0(df, labels, num_workers, sensattr = "race"):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers_gan_100_0(df,num_workers, sensattr)

    for i, indices in enumerate(indices_workers):
        x, y = df.loc[indices].values.tolist(), labels[indices].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data

def save_json(json_dir, json_name, users, num_samples, user_data):
	if not os.path.exists(json_dir):
		os.makedirs(json_dir)
	
	json_file = {
		'users': users,
		'num_samples': num_samples,
		'user_data': user_data,
	}
	
	with open(os.path.join(json_dir, json_name), 'w') as outfile:
		json.dump(json_file, outfile)


def parse_args():
	parser = argparse.ArgumentParser()
    
	parser.add_argument(
		'-num-workers',
		help='number of devices;',
		type=int,
		required=True)
	parser.add_argument(
		'-seed',
		help='seed for the random processes;',
		type=int,
		default=931231,
		required=False)
	parser.add_argument(
        '-heterog',
        help='heterogeneity setup (skewed data);',
        type=bool,
        default=False,
        required=False)
	parser.add_argument(
		'-gan',
		help='GAN setup (add a subset of data to all workers);',
		type=bool,
		default=False,
		required=False)
	parser.add_argument(
		'-niid',
		help='Non-IID 100-0 setup (add a subset of data to all workers);',
		type=bool,
		default=False,
		required=False)
	parser.add_argument(
		'-sens-attr',
		help='The sensitive attribute to be considered in the dataset (race or sex);',
		type=str,
		default=None,
        choices = ['race', 'sex'],
		required=False)
        

	return parser.parse_args()

if __name__ == '__main__':
	main()