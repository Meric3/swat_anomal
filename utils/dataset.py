import pandas as pd
import sklearn.preprocessing 
from sklearn.decomposition import PCA
import torch
import numpy as np

def swat_preprocessing(train, test, validation = False):
    train = train.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    test = test.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    
    x_colname = list(train.columns)[:-1]
    y_colname = list(train.columns)[-1]

    train_x = train[x_colname]
    train_y = train[y_colname]
    test_x = test[x_colname]
    test_y= test[y_colname]

    test_y[test_y == 'Attack'] = 1
    test_y[test_y == 'Normal'] = 0
    return train_x, test_x, test_y

def pca_return(train_path, test_path, validation = False):
    #load data
    normal_data=pd.read_csv(train_path,index_col = 0)
    attack_data=pd.read_csv(test_path,index_col = 0)

    train_x, test_x, test_y = swat_preprocessing(normal_data, attack_data)

    # label index
    train_index = normal_data.index
    test_index = attack_data.index
 
    scaler = sklearn.preprocessing.StandardScaler(with_std=False)
    scaler.fit(train_x)

    # train zero center about train data
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    pca = PCA(n_components = train_x.shape[1])

    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)
    """ 
    train_row = X_train.size(0)
    train_col = X_train.size(1)
    test_row = X_test.size(0) 
    test_col = X_test.size(1)
    """
    return train_x, test_x, test_y 


# SWaT DATA LOAD Func
def dataset(train_path, test_path):
    
    #load data
    normal_data = pd.read_csv(train_path, index_col = 0)
    attack_data = pd.read_csv(test_path, index_col = 0)
        
#     normal_data = normal_data.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
#     attack_data = attack_data.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    
    # columns setting 
    x_col = list(normal_data.columns)[:-1]
    y_col = list(normal_data.columns)[-1]
    
    # X, Y setting
    X_normal = normal_data[x_col]
    Y_normal = normal_data[y_col]
    
    X_attack = attack_data[x_col]
    Y_attack = attack_data[y_col]
    
    # label setting ( attack = 1, normal = 0)
    Y_train = Y_normal.copy()
    Y_test = Y_attack.copy()
    
    Y_train[Y_train == 'Attack'] = 1
    Y_train[Y_train == 'Normal'] = 0

    Y_test[Y_test == 'Attack'] = 1
    Y_test[Y_test == 'Normal'] = 0  
    
    Y_train = torch.FloatTensor(Y_train).cuda()
    Y_test = torch.FloatTensor(Y_test).cuda()
    
    unique, counts = np.unique(Y_normal, return_counts=True)
    print('In train :', dict(zip(unique, counts)))
    unique, counts = np.unique(Y_attack, return_counts=True)
    print('In test :', dict(zip(unique, counts)))     
    print("data length is {}".format(len(x_col)))
    
    # label index
    train_index = normal_data.index
    test_index = attack_data.index
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_normal)
    # train zero center about train data
    X_train = scaler.transform(X_normal)

    # test zero center about train data
    X_test = scaler.transform(X_attack)

    pca = PCA(n_components = len(x_col))

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = torch.FloatTensor(X_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()
    
    train_row = X_train.size(0)
    train_col = X_train.size(1)
    test_row = X_test.size(0) 
    test_col = X_test.size(1)
        
    
    return  X_train, X_test,  Y_test
