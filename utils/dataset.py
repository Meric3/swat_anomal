import pandas as pd
import sklearn.preprocessing 
from sklearn.decomposition import PCA


def swat_perprocessing(train, test, validation = False):
	train = train.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
	test = test.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
	
	x_colname = list(train.columns)[:-1]
	y_colname = list(train.columns)[-1]

	train_x = train[x_colname]
	train_y = train[y_colname]
	test_x	= test[x_colname]
	test_y 	= test[y_colname]

	test_y[test_y == 'Attack'] = 1
	test_y[test_y == 'Normal'] = 0
	return train_x, test_x, test_y



def pca_return(train_path, test_path, component_num, validation = False):
    
    #load data
    normal_data = pd.read_csv(train_path, index_col = 0)
    attack_data = pd.read_csv(test_path, index_col = 0)
    
	train_x, test_x, test_y = swat_preprocessing(normal_data, attack_data) 
    
    # label index
    train_index = normal_data.index
    test_index = attack_data.index
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_normal)
    # train zero center about train data
    X_train = scaler.transform(X_normal)

    # test zero center about train data
    X_test = scaler.transform(X_attack)

    pca = PCA(n_components = 51)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = torch.FloatTensor(X_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()
    
    train_row = X_train.size(0)
    train_col = X_train.size(1)
    test_row = X_test.size(0) 
    test_col = X_test.size(1)
        
    
    return  X_train, X_test, Y_train, Y_test, train_index, test_index, train_row, train_col, test_row, test_col
