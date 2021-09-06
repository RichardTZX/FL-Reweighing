import json
import numpy as np
import os
import argparse

import pandas as pd

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main():
    args=parse_args()
    np.random.seed(args.seed)
    print("Preprocessing adult.csv ...")
    adult_preprocessed_dir = os.path.join(parent_path, 'data', 'adult_preprocessed.csv')
    if os.path.exists(adult_preprocessed_dir):
        print("Reading adult_preprocessed.csv ...")
        df_adult = pd.read_csv(adult_preprocessed_dir)
        labels = df_adult['income']
        df_adult = df_adult.drop("income", axis=1)
        print(df_adult.head())
    else:
        df_adult, labels = get_adult_clear()
    num_workers = args.num_workers
    heterog = args.heterog
    gan, niid, sensattr  = args.gan, args.niid, args.sens_attr

    dir_path = os.path.join(parent_path, 'data', 'all_data')

    if gan:
        if niid:
            users, num_samples, user_data = to_leaf_format_gan_100_0(df_adult, labels, num_workers, sensattr)

            gan_data = user_data[str(num_workers)]
            del user_data[str(num_workers)]

            save_json(dir_path, 'data.json', users[:-1], num_samples[:-1], user_data)
            save_json(dir_path, 'gan.json', users[-1], num_samples[-1], gan_data)
            print("data.json ready with the adult.csv data distributed to {} workers and gan.json with the little subset of data that every worker will share".format(num_workers))
            print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")
        else:
            users, num_samples, user_data = to_leaf_format_gan(df_adult, labels, num_workers)

            gan_data = user_data[str(num_workers)]
            del user_data[str(num_workers)]

            save_json(dir_path, 'data.json', users[:-1], num_samples[:-1], user_data)
            save_json(dir_path, 'gan.json', users[-1], num_samples[-1], gan_data)
            print("data.json ready with the adult.csv data distributed to {} workers and gan.json with the little subset of data that every worker will share".format(num_workers))
            print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")
    else:
        if niid:
            users, num_samples, user_data = to_leaf_format_100_0(df_adult, labels, num_workers, sensattr)
        elif heterog:
            users, num_samples, user_data = to_leaf_format_het(df_adult, labels, num_workers)
        else:
            users, num_samples, user_data = to_leaf_format(df_adult, labels, num_workers)
            print('Standard distribution among the workers.')

        save_json(dir_path, 'data.json', users, num_samples, user_data)
        print("data.json ready with the adult.csv data distributed to {} workers".format(num_workers))
        print("You can run \" cd .. \" and \"./preprocess.sh --sf 1.0 -k 0 -t sample\" to end the preprocessing for the full sized data set")


def get_adult_clear(): #Load dataset 
    data_dir = os.path.join(parent_path, 'data', 'adult.csv')

    # ADULT PREPROCESSING DESCRIBED IN 
    # Abey et al. - Mitigating Bias in Federated Learning

    df = pd.read_csv(data_dir, usecols = ['age','educational-num','race','gender','income'])
    df['income'] = df['income'].replace('<=50K', 0).replace('>50K', 1)
    df['gender'] = df['gender'].replace("Male",1).replace("Female",0)
    
    # RACE to [|0,1|] 
    # 1 : White 
    # 2 : Amer-Indian-Eskimo,Asian-Pac-Islander,Black,Other

    df_race = pd.get_dummies(df['race'])
    df_clear = pd.concat((df_race, df), axis=1)
    df_clear = df_clear.drop(["race","Amer-Indian-Eskimo","Asian-Pac-Islander","Black","Other"], axis=1)
    df_clear = df_clear.rename(columns={"White" : "race"})

    # AGE to 7 features in a one hot encoded way

    df_clear['age'] = df_clear['age'].astype(int)
    df_clear['educational-num'] = df_clear['educational-num'].astype(int)

    for i in range(8):
            if i != 0:
                df_clear['age' + str(i)] = 0

    for index, row in df_clear.iterrows():
        if row['age'] < 20:
            df_clear.loc[index, 'age1'] = 1
        elif ((row['age'] < 30) & (row['age'] >= 20)):
            df_clear.loc[index, 'age2'] = 1
        elif ((row['age'] < 40) & (row['age'] >= 30)):
            df_clear.loc[index, 'age3'] = 1
        elif ((row['age'] < 50) & (row['age'] >= 40)):
            df_clear.loc[index, 'age4'] = 1
        elif ((row['age'] < 60) & (row['age'] >= 50)):
            df_clear.loc[index, 'age5'] = 1
        elif ((row['age'] < 70) & (row['age'] >= 60)):
            df_clear.loc[index, 'age6'] = 1
        elif row['age'] >= 70:
            df_clear.loc[index, 'age7'] = 1

    df_clear['ed6less'] = 0
    for i in range(13):
        if i >= 6:
            df_clear['ed' + str(i)] = 0
    df_clear['ed12more'] = 0

    for index, row in df_clear.iterrows():
        if row['educational-num'] < 6:
            df_clear.loc[index, 'ed6less'] = 1
        elif row['educational-num'] == 6:
            df_clear.loc[index, 'ed6'] = 1
        elif row['educational-num'] == 7:
            df_clear.loc[index, 'ed7'] = 1
        elif row['educational-num'] == 8:
            df_clear.loc[index, 'ed8'] = 1
        elif row['educational-num'] == 9:
            df_clear.loc[index, 'ed9'] = 1
        elif row['educational-num'] == 10:
            df_clear.loc[index, 'ed10'] = 1
        elif row['educational-num'] == 11:
            df_clear.loc[index, 'ed11'] = 1
        elif row['educational-num'] == 12:
            df_clear.loc[index, 'ed12'] = 1
        elif row['educational-num'] > 12:
            df_clear.loc[index, 'ed12more'] = 1

    df_clear.drop(['age', 'educational-num'], axis=1, inplace=True)

    # Remove NaNs
    object_col = df_clear.select_dtypes(include=object).columns.tolist()
    for col in object_col: #Remove NaNs from Adult dataset
        df_clear.loc[df_clear[col] == '?', col] = np.nan
    df_clear = df_clear.dropna(axis=0, how = 'any')

    # Make indexes coherent after NaNs removal
    df_clear.index = np.arange(np.array(df_clear.index).shape[0])

    df_clear.to_csv("adult_preprocessed.csv", index_label= False) # Save adult preprocessing in a .csv in order to load it faster than preprocess the dataset every time

    labels = df_clear['income']
    df_clear = df_clear.drop("income", axis=1)
    
    print(df_clear.head())
    return(df_clear,labels)

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
        if df['gender'].loc[indices[i]] == 0 and df['race'].loc[indices[i]] == 0 and cnt1 < size:
            gan_ind.append(indices[i])
            cnt1 += 1
        elif df['gender'].loc[indices[i]] == 0 and df['race'].loc[indices[i]] == 1 and cnt2 < size:
            gan_ind.append(indices[i])
            cnt2 += 1
        elif df['gender'].loc[indices[i]] == 1 and df['race'].loc[indices[i]] == 0 and cnt3 < size:
            gan_ind.append(indices[i])
            cnt3 += 1
        elif df['gender'].loc[indices[i]] == 1 and df['race'].loc[indices[i]] == 1 and cnt4 < size:
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
    indices_workers = get_indices_workers_100_0(df,num_workers,sensattr)

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
		help='The sensitive attribute to be considered in the dataset (race or gender);',
		type=str,
		default=None,
        choices = ['race', 'gender'],
		required=False)
        

	return parser.parse_args()

if __name__ == '__main__':
	main()