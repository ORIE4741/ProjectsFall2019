import datetime
import pandas as pd
import lightgbm as lgb
#import features
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import roc_auc_score
import numpy as np

LOCAL_TEST = True # only use train data

train_df = pd.read_csv('_train_df.csv')
test_df = pd.read_csv('_test_df.csv')
#print(train_df.head())
#print(test_df.head())
#load data
def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=4):
	
	folds = GroupKFold(n_splits=NFOLDS) #cross validation

	X,y = tr_df[features_columns], tr_df[target] #train    
	P,P_y = tt_df[features_columns], tt_df[target]   #test
	split_groups = tr_df['DT_M']

	tt_df = tt_df[['TransactionID',target]]    
	predictions = np.zeros(len(tt_df))
	oof = np.zeros(len(tr_df))
	
	for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
		print('Fold:',fold_)
		tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
		vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
			
		print(len(tr_x),len(vl_x))
		tr_data = lgb.Dataset(tr_x, label=tr_y)
		vl_data = lgb.Dataset(vl_x, label=vl_y)  

		estimator = lgb.train(
			lgb_params,
			tr_data,
			valid_sets = [tr_data, vl_data],
			verbose_eval = 200,
		)   
		
		pp_p = estimator.predict(P)
		predictions += pp_p/NFOLDS
		
		oof_preds = estimator.predict(vl_x)
		oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

		if LOCAL_TEST:
			feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
			print(feature_imp)
		
		del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
		gc.collect()
		
	tt_df['prediction'] = predictions
	print('OOF AUC:', metrics.roc_auc_score(y, oof))
	if LOCAL_TEST:
		print('Holdout AUC:', metrics.roc_auc_score(tt_df[TARGET], tt_df['prediction']))
	
	return tt_df

#parameters

lgb_params = {
		'objective':'binary',
		'boosting_type':'gbdt',
		'metric':'auc',
		'n_jobs':-1,
		'learning_rate':0.01,
		'num_leaves': 2**8,
		'max_depth':-1,
		'tree_learner':'serial',
		'colsample_bytree': 0.85,
		'subsample_freq':1,
		'subsample':0.85,
		'n_estimators':2**9,
		'max_bin':255,
		'verbose':-1,
		'seed': 42,
		'early_stopping_rounds':100,
		'reg_alpha':0.3,
		'reg_lamdba':0.243
	    } 

TARGET = 'isFraud'

features_columns = ['uid_DT_D', 'D5_fq_enc', 'D7_fq_enc', 'M_sum', 'D4_fq_enc', 'D3_fq_enc', 'D6_fq_enc', 'id_31_device', 'id_30_version', 'C12_fq_enc', 'M_na']

	
'''
# to find features_columns

from scipy.stats import ks_2samp
features_check = []
columns_to_check = set(list(train_df)).difference(base_columns+rm_cols)
for i in columns_to_check:
	features_check.append(ks_2samp(test_df[i], train_df[i])[1])

features_check = pd.Series(features_check, index=columns_to_check).sort_values() 
features_discard = list(features_check[features_check==0].index)
print(features_discard)

features_discard = [] 

# Final features list
features_columns = [col for col in list(train_df) if col not in rm_cols + features_discard]

'''


if LOCAL_TEST:
	lgb_params['learning_rate'] = 0.01
	lgb_params['n_estimators'] = 20000
	lgb_params['early_stopping_rounds'] = 100
	print('lightgbm')
	test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
	print('model finished!')
	print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
	print('done')
else:
	lgb_params['learning_rate'] = 0.007
	lgb_params['n_estimators'] = 1800
	lgb_params['early_stopping_rounds'] = 100    
	test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=6)


