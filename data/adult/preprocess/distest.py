import json
import numpy as np
import os
import argparse

import pandas as pd

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))



def main():
    df_adult, labels = get_adult_clear()

    df_counts = df_adult.drop(columns=['educational-num','native-country'])
    # print(np.unique(df_counts, axis=0 ,return_counts = True))
    print(np.unique(labels, axis=0 ,return_counts = True))

def get_adult_clear(): #Load dataset 
    data_dir = os.path.join(parent_path, 'data', 'adult.csv')

    # ADULT PREPROCESSING DESCRIBED IN 
    # Abey et al. - Mitigating Bias in Federated Learning

    df = pd.read_csv(data_dir, usecols = ['age','educational-num','race','gender','native-country','income'])
    df['income'] = df['income'].replace('<=50K', 0).replace('>50K', 1)
    df['gender'] = df['gender'].replace("Male",1).replace("Female",0)
    
    # RACE to [|0,1|] 
    # 1 : White 
    # 2 : Amer-Indian-Eskimo,Asian-Pac-Islander,Black,Other

    df_race = pd.get_dummies(df['race'])
    df_clear = pd.concat((df, df_race), axis=1)
    df_clear = df_clear.drop(["race","Amer-Indian-Eskimo","Asian-Pac-Islander","Black","Other"], axis=1)
    df_clear = df_clear.rename(columns={"White" : "race"})

    # AGE to [|0,1,2,3,4,5,6,7,8,9|] 
    # 0 : 10yo-19yo 
    # 1 : 20yo-29yo ...

    df_clear['age'] = (df_clear['age'] // 10) - 1 #Map age by decade
    df_clear['educational-num'] = (df_clear['educational-num'] // 10)

    # Remove NaNs
    object_col = df_clear.select_dtypes(include=object).columns.tolist()
    for col in object_col: #Remove NaNs from Adult dataset
        df_clear.loc[df_clear[col] == '?', col] = np.nan
    df_clear = df_clear.dropna(axis=0, how = 'any')

    # NATIVE-COUNTRY to continent 
    # 0 : North America 
    # 1 : Europe 
    # 2 : Asia 
    # 3 : Latin America

    df_clear['native-country'] = df_clear['native-country'].replace(["Canada","Mexico","United-States","Outlying-US(Guam-USVI-etc)"],0)
    df_clear['native-country'] = df_clear['native-country'].replace(["England","France","Germany","Greece","Holand-Netherlands","Hungary","Ireland","Italy","Poland","Portugal","Scotland","Yugoslavia"],1)
    df_clear['native-country'] = df_clear['native-country'].replace(["Cambodia","China","Hong","India","Iran","Japan","Laos","Philippines","South","Taiwan","Thailand","Vietnam"],2)
    df_clear['native-country'] = df_clear['native-country'].replace(["Columbia","Cuba","Dominican-Republic","Ecuador","El-Salvador","Guatemala","Haiti","Honduras","Jamaica","Nicaragua","Peru","Puerto-Rico","Trinadad&Tobago"],3)

    # Make indexes coherent after NaNs removal
    df_clear.index = np.arange(np.array(df_clear.index).shape[0])

    labels = df_clear['income']
    df_clear = df_clear.drop("income", axis=1)
    df_clear = df_clear[['age','gender','race','educational-num','native-country']]
    
    return(df_clear,labels)

def get_indices_workers(df, num_workers): #Generate the samples indices for each worker : Each worker has the same amount of data
    mean = int(len(df) / num_workers)
    indices = np.array(df.index) #Indices of the samples
    np.random.shuffle(indices)
    indices_workers = [indices[i*mean:(i+1)*mean] for i in range(num_workers)] #For each worker, we have a list of the samples he possesses
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



if __name__ == '__main__':
	main()