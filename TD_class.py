import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import random as rnd
import time
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
# ========

path=''

VALIDATE = False

cols =    [ #'app',
            #'channel',
            'click_id',
            #'device',
            #'from',
            #'ip',
            'is_attributed',
            #'os',
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
print('Sampling train data...')
start_time = time.time()

tr = train_df[train_df['is_attributed']==1]
smpl= len(tr)
train_df = pd.concat([tr,train_df[train_df['is_attributed']==0].sample(smpl)])

print('[{}] Finished sampling'.format(time.time() - start_time))
print(len(train_df), len(tr))

target =  train_df['is_attributed']

train_df = train_df.drop(['is_attributed', 'click_id'], axis=1)


# scatter plot matrix
#~ scatter_matrix(train_df)
#~ plt.show()

#~ X = train_df.values
X =  StandardScaler().fit_transform(train_df)
Y = target.values



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



print('loading test data...')
start_time = time.time()

# Make predictions on validation dataset
test_df = pd.read_csv(path+"test_new.csv", 
                        header=0,
                        usecols=cols,
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )

clickID = test_df['click_id']

test_df = test_df.drop(['is_attributed', 'click_id'], axis=1)

print('[{}] Finished loading test data'.format(time.time() - start_time))





print('Start fitting and prediction data...')
start_time = time.time()

# Make fitting and prediction
X_test =  StandardScaler().fit_transform(test_df)

rf = RandomForestClassifier()
rf.fit(X, Y)

#Y_pred = rf.predict(X_test)
Y_prob = rf.predict_proba(X_test)

print('[{}] Done !!!'.format(time.time() - start_time))

sub = pd.DataFrame()
sub['click_id'] = clickID
sub['is_attributed'] = pd.DataFrame(Y_prob[:,1])

# Sorting
sub = sub.sort_values(by=['click_id'], ascending=[True])

print("writing...")
sub.to_csv("prediction.csv", index=False, float_format='%.9f')



print(sub.info())
print(sub.head())


#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
