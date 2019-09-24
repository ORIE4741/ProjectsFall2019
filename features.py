"""
Created on Mon April 9 10:12:09 2018
@author: Keh Yao
"""
import os, sys, gc, warnings, random, datetime
import pandas as pd
import datetime
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import roc_auc_score

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
LOCAL_TEST = True
print('Load Data')
#train_df = pd.read_csv('train_transaction.csv')
train_df = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')
#df_train = pd.merge(train_df, train_identity,on='TransactionID',how='left')

'''
test_df = pd.read_csv('test_transaction100.csv')
test_identity = pd.read_csv('test_identity100.csv')
df_test = pd.merge(test_df, test_identity,on='TransactionID',how='left')
'''

if LOCAL_TEST:
   
	train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
	train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
	print(train_df['DT_M'])
	test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
	train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
	
	
	test_identity  = train_identity[train_identity['TransactionID'].isin(
									test_df['TransactionID'])].reset_index(drop=True)
	train_identity = train_identity[train_identity['TransactionID'].isin(
									train_df['TransactionID'])].reset_index(drop=True)
	del train_df['DT_M'], test_df['DT_M']
	
else:
	test_df = pd.read_pickle('../input/ieee-data-minification/test_transaction.pkl')
	train_identity = pd.read_pickle('../input/ieee-data-minification/train_identity.pkl')
	test_identity = pd.read_pickle('../input/ieee-data-minification/test_identity.pkl')
	
base_columns = list(train_df) + list(train_identity)
print('Shape control:', train_df.shape, test_df.shape)




'''	
else:
	test_df = pd.read_csv('test_transaction.csv')
	train_identity = pd.read_csv('train_identity.csv')
	test_identity = pd.read_csv('test_identity.csv')
'''		
base_columns = list(train_df) + list(train_identity)
print('Shape control:', train_df.shape, test_df.shape)


for df in [train_df, test_df]:
	df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
	df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
	df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
	df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear
	
	df['DT_hour'] = df['DT'].dt.hour
	df['DT_day_week'] = df['DT'].dt.dayofweek
	df['DT_day'] = df['DT'].dt.day
	
	df['D9'] = np.where(df['D9'].isna(),0,1)
	
i_cols = ['card1']

for col in i_cols: 
	valid_card = pd.concat([train_df[[col]], test_df[[col]]])
	valid_card = valid_card[col].value_counts()
	valid_card = valid_card[valid_card>2]
	valid_card = list(valid_card.index)

	train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
	test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

	train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
	test_df[col]  = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

for df in [train_df, test_df]:
	df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
	df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
	
train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)
test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)

train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']

for col in i_cols:
	for agg_type in ['mean','std']:
		new_col_name = col+'_TransactionAmt_'+agg_type
		temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col,'TransactionAmt']]])
		temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
												columns={agg_type: new_col_name})
		
		temp_df.index = list(temp_df[col])
		temp_df = temp_df[new_col_name].to_dict()   
	
		train_df[new_col_name] = train_df[col].map(temp_df)
		test_df[new_col_name]  = test_df[col].map(temp_df)
		   
train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])  

p = 'P_emaildomain'
r = 'R_emaildomain'
uknown = 'email_not_provided'

for df in [train_df, test_df]:
	df[p] = df[p].fillna(uknown)
	df[r] = df[r].fillna(uknown)
	
	df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

	df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
	df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])
	

for df in [train_identity, test_identity]:

	df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
	df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
	df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
	
	df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
	df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
	df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
	
	df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
	df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
	
temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)


i_cols = ['card1','card2','card3','card5',
					'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
					'D1','D2','D3','D4','D5','D6','D7','D8',
					'addr1','addr2',
					'dist1','dist2',
					'P_emaildomain', 'R_emaildomain',
					'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
					'id_30','id_30_device','id_30_version',
					'id_31_device',
					'id_33',
					'uid','uid2','uid3',
				 ]

for col in i_cols:
		temp_df = pd.concat([train_df[[col]], test_df[[col]]])
		fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
		train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
		test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


for col in ['DT_M','DT_W','DT_D']:
		temp_df = pd.concat([train_df[[col]], test_df[[col]]])
		fq_encode = temp_df[col].value_counts().to_dict()
						
		train_df[col+'_total'] = train_df[col].map(fq_encode)
		test_df[col+'_total']  = test_df[col].map(fq_encode)
				

periods = ['DT_M','DT_W','DT_D']
i_cols = ['uid']
for period in periods:
		for col in i_cols:
				new_column = col + '_' + period
						
				temp_df = pd.concat([train_df[[col,period]], test_df[[col,period]]])
				temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
				fq_encode = temp_df[new_column].value_counts().to_dict()
						
				train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
				test_df[new_column]  = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
				
				train_df[new_column] /= train_df[period+'_total']
				test_df[new_column]  /= test_df[period+'_total']
				

for col in list(train_df):
	train_df[col] = train_df[col].fillna('unseen_before_label')
	
	test_df[col]  = test_df[col].fillna('unseen_before_label')
	
	train_df[col] = train_df[col].astype(str)
	test_df[col] = test_df[col].astype(str)
	
	le = LabelEncoder()
	le.fit(list(train_df[col])+list(test_df[col]))
	train_df[col] = le.transform(train_df[col])
	test_df[col]  = le.transform(test_df[col])
	
	train_df[col] = train_df[col].astype('category')
	test_df[col] = test_df[col].astype('category')
	
	'''
rm_cols = [
	'TransactionID','TransactionDT', # These columns are pure noise right now
	TARGET,                          # Not target in features))
	'uid','uid2','uid3',             # Our new client uID -> very noisy data
	'bank_type',                     # Victims bank could differ by time
	'DT','DT_M','DT_W','DT_D',       # Temporary Variables
	'DT_hour','DT_day_week','DT_day',
	'DT_D_total','DT_W_total','DT_M_total',
	'id_30','id_31','id_33',
]


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

train_df.to_csv('_train_df.csv',index=None)
test_df.to_csv('_test_df.csv',index=None)