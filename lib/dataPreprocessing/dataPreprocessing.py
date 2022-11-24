import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler,MaxAbsScaler,KBinsDiscretizer
from sklearn.feature_extraction import DictVectorizer


## Deal with missing data #############################################################
def imputeMissingData(dataset,maxPct=0.5,replaceByZero=False,replaceByMedian=True):
    string_columns = list(dataset.select_dtypes(include=['object']).columns) 
    list_numeric_data = list(dataset._get_numeric_data()) 
    for column in list_numeric_data:
        if (dataset[column].isnull().sum()/len(dataset) > maxPct):
            dataset.drop([column],axis=1,inplace=True)
    list_numeric_data = list(dataset._get_numeric_data())
    if len(string_columns)>0:
        dataset[string_columns] = dataset[string_columns].fillna('Missing values')
    if len(list_numeric_data)>0:
        if replaceByZero:
            dataset[list_numeric_data] = dataset[list_numeric_data].fillna(0)
        elif replaceByMedian:
            dataset[list_numeric_data] = dataset[list_numeric_data].fillna(dataset[list_numeric_data].median())
        else:
            dataset[list_numeric_data] = dataset[list_numeric_data].fillna(dataset[list_numeric_data].mean())
    else:
        print('Incorrect format.')
    return dataset


## Reduce nulber of modalities for categorical variables ##############################
def reduceNbModalities(dataset,pctFilter=True,topFilter=False,minPct=0.05,topValues=10):
    string_columns = list(dataset.select_dtypes(include=['object']).columns)
    dict_of_values = {}
    if topFilter:
        for column in string_columns:
            temp = list(dataset[column].value_counts(normalize=False,sort=True,ascending=False).head(topValues).index)
            dataset[column][~dataset[column].isin(temp)]='other'
            dict_of_values.update({column : temp})
    elif pctFilter:
        for column in string_columns:
            tmp = pd.DataFrame(dataset[column].value_counts(normalize=True)<minPct)
            dataset[column][dataset[column].isin(list(tmp[tmp[column]==True].index))]='other'
            dict_of_values.update({column : temp})
    else:
        print('Only for categorical data.')
    return dataset, dict_of_values

def transform_reduceNbModalities(dataset,dict_of_values):
    for column in list(dict_of_values.keys()):
        dataset[column] = [value if value in dict_of_values[column] else 'other' for value in dataset[column]]
    return dataset


## Deal with outliers ################################################################
def process_outliers(dataset,flooring_pct=0.025,capping_pct=0.975):
    list_numeric_data = list(dataset._get_numeric_data()) 
    for var in list_numeric_data:
        floor_value = dataset[var].quantile(flooring_pct)
        cappe_value = dataset[var].quantile(capping_pct)
        dataset[var] = np.where(dataset[var]<floor_value,floor_value,dataset[var])
        dataset[var] = np.where(dataset[var]>cappe_value,cappe_value,dataset[var])
    return dataset

def transform_process_outliers(dataset,maxValues):
    list_numeric_data = list(dataset._get_numeric_data()) 
    for var in list_numeric_data:
        cappe_value = float(maxValues.loc[var,'maxValue'])
        dataset[var] = np.where(dataset[var]>cappe_value,cappe_value,dataset[var])
    return dataset


## OHE + scaling ####################################################################
def preprocessing(dataset,filename_dict_model,filename_scale_model,scaling='RobustScaler'):
    binariser = DictVectorizer(sparse=True)
    dataset = binariser.fit_transform(dataset.to_dict(orient='record'))
    pickle.dump(binariser, open("./MODEL/{}".format(filename_dict_model), 'wb')) #Saving the binariser model
    list_features = binariser.get_feature_names()
    pd.DataFrame(list_features).to_csv('./MODEL/list_features.csv',index=False) #export list of features
    if scaling=='RobustScaler':
        scaler = RobustScaler(with_centering=False)
        dataset = scaler.fit(dataset.toarray()).transform(dataset)
        pickle.dump(scaler, open("./MODEL/{}".format(filename_scale_model), 'wb')) #Saving the scaler model
    elif scaling=='MaxAbsScaler':
        scaler = MaxAbsScaler()
        dataset = scaler.fit_transform(dataset)
        pickle.dump(scaler, open("./MODEL/{}".format(filename_scale_model), 'wb')) #Saving the scaler model
    elif scaling=='MinMaxScaler':
        scaler = MinMaxScaler()
        dataset = scaler.fit_transform(pd.DataFrame(dataset.toarray()))
        pickle.dump(scaler, open("./MODEL/{}".format(filename_scale_model), 'wb')) #Saving the scaler model   
    return dataset, list_features


## Encoding cyclical continuous features #############################################
def cyclical_data(dataframe, list_cyclical_var):
    embeddings = {}
    for cyclical_var in list_cyclical_var:
        nb_of_unique_cat = float(dataframe[cyclical_var].nunique())
        embeddings[cyclical_var] = [[np.sin(2.*np.pi*dataframe[cyclical_var]/nb_of_unique_cat)[i],
                                     np.cos(2.*np.pi*dataframe[cyclical_var]/nb_of_unique_cat)[i]] for i in range(len(dataframe))]
    
    return embeddings