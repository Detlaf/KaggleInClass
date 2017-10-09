# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:24:14 2017

@author: Kategl
"""
#Соревнование Kaggle: Catch me if you can

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
import seaborn as sns

# загрузим обучающую и тестовую выборки
train_df = pd.read_csv('C:/Users/Kategl/Desktop/mlcourse_open-master/homework/Alice/data/train_sessions.csv',index_col='session_id')
test_df = pd.read_csv('C:/Users/Kategl/Desktop/mlcourse_open-master/homework/Alice/data/test_sessions.csv',index_col='session_id')

# приведем колонки time1, ..., time10 к временному формату
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# отсортируем данные по времени
train_df = train_df.sort_values(by='time1')

# посмотрим на заголовок обучающей выборки
print(train_df.head())


#%%
# приведем колонки site1, ..., site10 к целочисленному формату и заменим пропуски нулями
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# загрузим словарик сайтов
with open(r"C:/Users/Kategl/Desktop/mlcourse_open-master/homework/Alice/data/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# датафрейм словарика сайтов
sites_dict_df = pd.DataFrame(list(site_dict.keys()),
                          index=list(site_dict.values()),
                          columns=['site'])
print(u'всего сайтов:', sites_dict_df.shape[0])


#%%
# наша целевая переменная
y_train = train_df['target']

# объединенная таблица исходных данных
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# индекс, по которому будем отделять обучающую выборку от тестовой
idx_split = train_df.shape[0]

#%%
# табличка с индексами посещенных сайтов в сессии
full_sites = full_df[sites]

# последовательность с индексами
sites_flatten = full_sites.values.flatten()

# искомая матрица
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

X_train_sparse = full_sites_sparse[:idx_split]
X_test_sparse = full_sites_sparse[idx_split:]
#%%


def get_auc_lr_valid(X, y, C=1.0, ratio=0.9, seed=17):
    '''
    X, y – выборка
    ratio – в каком отношении поделить выборку
    C, seed – коэф-т регуляризации и random_state
              логистической регрессии
    '''
    train_len = int(ratio * X.shape[0])
    X_train = X[:train_len, :]
    X_valid = X[train_len:, :]
    y_train = y[:train_len]
    y_valid = y[train_len:]

    logit = LogisticRegression(C=C, n_jobs=-1, random_state=seed)

    logit.fit(X_train, y_train)

    valid_pred = logit.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y_valid, valid_pred)
#%%


# функция для записи прогнозов в файл
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


#%%
# Обучение на всей выборке

logit = LogisticRegression(n_jobs=-1, random_state=17)
logit.fit(X_train_sparse, y_train)

#%%

test_pred = logit.predict_proba(X_test_sparse)[:, 1]
pd.Series(test_pred, index=range(1, test_pred.shape[0] + 1),
         name='target').to_csv('benchmark1.csv', header=True, index_label='session_id')

#%%


new_feat_train = pd.DataFrame(index=train_df.index)
new_feat_test = pd.DataFrame(index=test_df.index)

new_feat_train['year_month'] = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
new_feat_test['year_month'] = test_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)

#%%


scaler = StandardScaler()
scaler.fit(new_feat_train['year_month'].values.reshape(-1, 1))

new_feat_train['year_month_scaled'] = scaler.transform(new_feat_train['year_month'].values.reshape(-1, 1))
new_feat_test['year_month_scaled'] = scaler.transform(new_feat_test['year_month'].values.reshape(-1, 1))

#%%
#Вычисление ROC_AUC на отложенной выборке

X_train_sparse_new = csr_matrix(hstack([X_train_sparse,
                             new_feat_train['year_month_scaled'].values.reshape(-1, 1)]))
get_auc_lr_valid(X_train_sparse_new, y_train)

#%%
# Месяц начала сессии
new_feat_train['start_month'] = train_df['time1'].apply(lambda x: x.month)
new_feat_test['start_month'] = test_df['time1'].apply(lambda x: x.month)

# Час начала сессии
new_feat_train['start_hour'] = train_df['time1'].apply(lambda x: x.hour)
new_feat_test['start_hour'] = test_df['time1'].apply(lambda x: x.hour)

# Утро - час меньше 11 или нет
new_feat_train['morning'] = new_feat_train['start_hour'] <= 11
new_feat_test['morning'] = new_feat_test['start_hour'] <= 11

#%%
# Сайты + месяц + час
X_train_sparse_new1 = csr_matrix(hstack([X_train_sparse_new,
                             new_feat_train['start_month'].values.reshape(-1, 1)]))
X_train_sparse_new_month_hour = csr_matrix(hstack([X_train_sparse_new1,
                             new_feat_train['start_hour'].values.reshape(-1, 1)]))

AUC_month_hour = get_auc_lr_valid(X_train_sparse_new_month_hour, y_train)

# Сайты + месяц + утро
X_train_sparse_new_month_morn = csr_matrix(hstack([X_train_sparse_new1,
                             new_feat_train['morning'].values.reshape(-1, 1)]))

AUC_month_morn = get_auc_lr_valid(X_train_sparse_new_month_morn, y_train)

# Сайты + месяц + час + утро
X_train_sparse_new_month_hour_morn = csr_matrix(hstack([X_train_sparse_new_month_hour,
                             new_feat_train['morning'].values.reshape(-1, 1)]))

AUC_month_hour_morn = get_auc_lr_valid(X_train_sparse_new_month_hour_morn, y_train)

#%%
print('Месяц + час', AUC_month_hour)
print('Месяц + утро', AUC_month_morn)
print('Месяц + час + утро', AUC_month_hour_morn)

#%%
c_param = np.logspace(-3, 1, 10)
AUC_dict ={}
AUC_list = []
c_list = []

for c in c_param:
    AUC_month_hour_morn = get_auc_lr_valid(X_train_sparse_new_month_hour_morn, y_train, c)
    AUC_list.append(AUC_month_hour_morn)
    c_list.append(c)
AUC_dict = dict(zip(AUC_list, c_list))
AUC_max = np.max(AUC_list)
c_max = AUC_dict[AUC_max]
#%%
logit = LogisticRegression(n_jobs=-1, random_state=17, C=c_max)
logit.fit(X_train_sparse, y_train)
test_pred = logit.predict_proba(X_test_sparse)[:, 1]
pd.Series(test_pred, index=range(1, test_pred.shape[0] + 1),
         name='target').to_csv('benchmark2.csv', header=True, index_label='session_id')