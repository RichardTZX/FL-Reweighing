import json
import numpy as np
import os
import argparse

import pandas as pd

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main():
    args=parse_args()
    np.random.seed(args.seed)
    df_adult, labels = get_adult_clear()
    num_workers = args.num_workers
    heterog = args.heterog

    if heterog:
        users, num_samples, user_data = to_leaf_format_het(df_adult, labels, num_workers)
    else:
        users, num_samples, user_data = to_leaf_format(df_adult, labels, num_workers)

    dir_path = os.path.join(parent_path, 'data', 'all_data')

    save_json(dir_path, 'data.json', users, num_samples, user_data)
    print("data.json ready with the adult.csv data distributed to {} workers".format(num_workers))
    print("You can run \" cd .. \" and \"./preprocess.sh -s niid --sf 1.0 -k 0 -t sample \" for the full sized data set")

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

def to_leaf_format(df, labels, num_workers):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers(df,num_workers)

    for i, indices in enumerate(indices_workers):
        x, y = df.iloc[indices].values.tolist(), labels[indices].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x' : x, 'y' : y}

    return users, num_samples, user_data

def to_leaf_format_het(df, labels, num_workers):
    users, num_samples, user_data = [], [], {}
    indices_workers = get_indices_workers_het(df,num_workers)

    for i, indices in enumerate(indices_workers):
        x, y = df.iloc[indices].values.tolist(), labels[indices].tolist()
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

	return parser.parse_args()

if __name__ == '__main__':
	main()