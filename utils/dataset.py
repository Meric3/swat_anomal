import pandas as pd
import sklearn.preprocessing 
from sklearn.decomposition import PCA

def swat_preprocessing(train, test, validation = False):
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

