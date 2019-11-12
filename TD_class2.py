import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import random as rnd
import time
import gc


# Classifiers
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
# ========

path=''

VALIDATE = False

cols =    [ 'app',
            'channel',
            'click_id',
            'device',
            #'from',
            'ip',
            'is_attributed',
            'os',
            #'day',
            'hour',
            'click_AppBefore',
            'click_AppAfter',
            'click_speed',
            #'click_byIp',
            'ip_byChannel',
            'click_speed_median',
            'click_speed_distance',
            #'click_byDayByIp',
            'click_byDayByIp_log',
            'ratio_click_onAppBefore_byDayByIp',
            'ratio_click_onAppAfter_byDayByIp',
            'app_byIp',
            'app_byChannel',
            ##'click_byChannel',
            ##'channel_byIP',
            'click_byIp_log',
            'channel_byIP_log',
            'ratio_byChannel',
            'click_byChannel_log'
            ]

print('loading train data...')
start_time = time.time()

p = 0.005
train_df = pd.read_csv(path+"train_new.csv", 
                        header=0,
                        usecols=cols,
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )

print('[{}] Finished to load data'.format(time.time() - start_time))


# Sampling
#~ print('Sampling train data...')
#~ start_time = time.time()

#~ tr = train_df[train_df['is_attributed']==1]
#~ smpl= len(tr)
#~ train_df = pd.concat([tr,train_df[train_df['is_attributed']==0].sample(smpl)])

#~ print('[{}] Finished sampling'.format(time.time() - start_time))
#~ print(len(train_df), len(tr))




# scatter plot matrix
#~ scatter_matrix(train_df)
#~ plt.show()

#~ X = train_df.values
#~ X =  StandardScaler().fit_transform(train_df)
#~ Y = target.values



if VALIDATE:
    # Split-out validation dataset
    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                    test_size=validation_size, 
                                                    random_state=seed)

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))

    #~ models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []

    print('Evaluate models...')
    start_time = time.time()

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


    #~ # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()




metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 15,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
        'metric':metrics
}

target = 'is_attributed'

predictors =[ 'app',
            'channel',
            'click_id',
            'device',
            'ip',
            'os',
            'hour',
            'click_AppBefore',
            'click_AppAfter',
            'click_speed',
            'ip_byChannel',
            'click_speed_median',
            'click_speed_distance',
            'click_byDayByIp_log',
            'ratio_click_onAppBefore_byDayByIp',
            'ratio_click_onAppAfter_byDayByIp',
            'app_byIp',
            'app_byChannel',
            'click_byIp_log',
            'channel_byIP_log',
            'ratio_byChannel',
            'click_byChannel_log'
            ]


categorical = ['app', 'device', 'os', 'channel']



print('Start fitting and prediction data...')
start_time = time.time()


xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

del train_df
gc.collect()

OPT_ROUNDS = 680
num_boost_round=OPT_ROUNDS

bst = lgb.train(lgb_params, 
                 xgtrain, 
                 num_boost_round=num_boost_round,
                 verbose_eval=10, 
                 feval=None)

del xgtrain
gc.collect()




print('loading test data...')
start_time = time.time()




# Make predictions on validation dataset
test_df = pd.read_csv(path+"test_new.csv", 
                        header=0,
                        usecols=cols,
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )



sub = pd.DataFrame()
sub['click_id'] =test_df['click_id']


print('[{}] Finished loading test data'.format(time.time() - start_time))





print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])


# Sorting
sub = sub.sort_values(by=['click_id'], ascending=[True])

print("writing...")
sub.to_csv("prediction3.csv", index=False, float_format='%.9f')



print(sub.info())
print(sub.head())



print('[{}] Done !!!'.format(time.time() - start_time))





