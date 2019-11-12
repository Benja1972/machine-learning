import pandas as pd
import time
import numpy as np
#~ from sklearn.cross_validation import train_test_split
#~ import xgboost as xgb
#~ from xgboost import plot_tree
#~ from xgboost import plot_importance
#import matplotlib.pyplot as plt
import gc
import random as rnd

path = '../BIP/kaggle/'
pathOut = ''

dtypes = {'ip'            : 'uint32',
          'app'           : 'uint16',
          'device'        : 'uint16',
          'os'            : 'uint16',
          'channel'       : 'uint16',
          'is_attributed' : 'uint8',
          'click_id'      : 'uint32'
          }

print('loading train data...')

p = .0001
start_time = time.time()

# take the train dataset
train = pd.read_csv(path+"train.csv", dtype=dtypes, 
                        usecols=['ip','app','device','os', 'channel', 'click_time','is_attributed'],
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )

train['click_id'] = int(0) 
train['from'] = 'train'

# take the test   cols = [click_id,ip,app,device,os,channel,click_time]
test = pd.read_csv(path+"test.csv", dtype=dtypes, 
                        usecols=['click_id','ip','app','device','os', 'channel', 'click_time'],
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )
test['is_attributed'] = float('nan')
test['from'] = 'test'

# take the test_sup_without 
test_supp = pd.read_csv(path+"test_sup_without_fromGauss3.csv", dtype=dtypes, 
                        usecols=['click_id','ip','app','device','os', 'channel', 'click_time'],
                        #~ skiprows=lambda i: i>0 and rnd.random() > p
                        )
#~ test_supplement_df = test_supplement_df.drop(['Unnamed: 0'], axis=1)
test_supp['is_attributed'] = float('nan')
test_supp['from'] = 'testSup'


#~ train_df = pd.concat([train, test, test_supp], keys=['train', 'test','testSup'])
train_df = pd.concat([train, test, test_supp])

#del test_df
#del test_supplement_df


print('[{}] Finished to load data'.format(time.time() - start_time))
print('DF_shape = '+ str(train_df.shape))



print('Extracting day and hour...')
click_time = pd.to_datetime(train_df.click_time)
train_df['day']  = click_time.dt.day.astype('uint8')

train_df['hour'] = click_time.dt.hour + click_time.dt.minute/60 + click_time.dt.second/3600 



train_df = train_df.drop(['click_time'], axis=1)
gc.collect()


# sorting
start_time = time.time()
print('Sorting...  ')
train_df = train_df.sort_values(by=['ip','device','os','day','hour'], ascending=[True,True,True,True,True])
print('Sorting done  '+ str((time.time() - start_time)))


def dfill(a):
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))


def argunsort(s):
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u

def cumcount(a):
    n = a.size
    s = a.argsort(kind='mergesort')
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]

### === cumcount example
#~ aaa = np.array(list('29731g236b29gfs7812334dba'))

#~ array(['2', '9', '7', '3', '1', 'g', '2', '3', '6', 'b', '2', '9', 'g',
       #~ 'f', 's', '7', '8', '1', '2', '3', '3', '4', 'd', 'b', 'a'], 
      #~ dtype='<U1')

#~ cumcount(aaa) 
#~ array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 0, 1, 3, 2, 3, 0, 0,
       #~ 1, 0])

# -----------------


start_time = time.time()
print('Grouping...')
train_df['click_AppBefore'] = train_df.groupby(['ip','device','os','day'])['app'].transform(lambda x: cumcount(x.values))

print('click_AppBefore done...'+ str((time.time() - start_time)))

train_df['click_AppAfter'] = train_df.groupby(['ip','device','os','day'])['app'].transform(lambda x: cumcount(x.values[::-1])[::-1] )

print('click_AppAfter done...'+ str((time.time() - start_time)))

gc.collect()


train_df = train_df.sort_values(by=['ip','day','hour'], ascending=[True,True,True])
train_df['click_speed'] = train_df.groupby(['ip','day'])['hour'].transform(lambda x: np.gradient(x.values) if len(x) > 1 else 0)

print('Grouping done'+ str((time.time() - start_time)))



start_time = time.time()
print('Calculating statistics...')

# Count the number of clicks by ip
ip_count = train_df.groupby('ip')['channel'].count().reset_index()
ip_count.columns = ['ip',  'click_byIp']
train_df = pd.merge(train_df, ip_count, on='ip', how='left', sort=False)
train_df['click_byIp'] = train_df['click_byIp'].astype('uint16')

del ip_count

# count the number of ip by channel
ip_by_channel = train_df.groupby(['channel'])['ip'].nunique().reset_index()
ip_by_channel.columns = ['channel','ip_byChannel']
train_df = pd.merge(train_df, ip_by_channel, on=['channel'], how='left', sort=False)

del ip_by_channel

# calcul the median and mad of click_interval_after

ip_count = train_df.groupby('ip')['click_speed'].median().reset_index()
ip_count.columns = ['ip',  'click_speed_median']
train_df = pd.merge(train_df, ip_count, on='ip', how='left', sort=False)

del ip_count

train_df['click_speed_distance'] = np.arcsinh( train_df['click_speed']-train_df['click_speed_median'] )



# Count the number of clicks by day by ip
ip_count = train_df.groupby(['ip','day'])['channel'].count().reset_index()
ip_count.columns = ['ip','day',  'click_byDayByIp']
train_df = pd.merge(train_df, ip_count, on=['ip','day'], how='left', sort=False)
train_df['click_byDayByIp'] = train_df['click_byDayByIp'].astype('uint16')

train_df['click_byDayByIp_log'] = np.log(train_df['click_byDayByIp'])

del ip_count


train_df['ratio_click_onAppBefore_byDayByIp'] = (train_df['click_AppBefore'])/(train_df['click_byDayByIp'])
train_df['ratio_click_onAppAfter_byDayByIp'] = (train_df['click_AppAfter'])/(train_df['click_byDayByIp'])



# Count the number of app_byChannel
channel_uniqueApp = train_df.groupby(['ip'])['app'].nunique().reset_index()
channel_uniqueApp.columns = ['ip','app_byIp']
train_df = pd.merge(train_df, channel_uniqueApp, on=['ip'], how='left', sort=False)


del channel_uniqueApp


# Count the number of app_byChannel
channel_uniqueApp = train_df.groupby(['channel'])['app'].nunique().reset_index()
channel_uniqueApp.columns = ['channel','app_byChannel']
train_df = pd.merge(train_df, channel_uniqueApp, on=['channel'], how='left', sort=False)

del channel_uniqueApp

# count the number of click by channel
channel_count = train_df.groupby(['channel'])['app'].count().reset_index()
channel_count.columns = ['channel','click_byChannel']
train_df = pd.merge(train_df, channel_count, on=['channel'], how='left', sort=False)

del channel_count


# count the number of channel by ip
channel_by_ip = train_df.groupby(['ip'])['channel'].nunique().reset_index()
channel_by_ip.columns = ['ip','channel_byIP']
train_df = pd.merge(train_df, channel_by_ip, on=['ip'], how='left', sort=False)

del channel_by_ip
train_df['click_byIp_log'] = np.log(train_df['click_byIp'])
train_df['channel_byIP_log'] = np.log(train_df['channel_byIP'])

train_df['ratio_byChannel'] = (train_df['click_byChannel']/train_df['app_byChannel'])


train_df["click_byChannel_log"] = np.log(train_df["click_byChannel"])

print('Staistics done'+ str((time.time() - start_time)))




#~ ['hour',
 #~ 'click_interval_before_distance',
 #~ 'click_interval_after_distance',
 #~ 'click_onAppAfter_byDayByIp',
 #~ 'ratio_click_onAppBefore_byDayByIp',
 #~ 'ratio_click_onAppAfter_byDayByIp',
 #~ 'click_byDayByIp_log',
 #~ 'app_byIp',
 #~ 'app_byChannel',
 #~ 'click_byIp_log',
 #~ 'channel_byIP_log',
 #~ 'click_byChannel_log',
 #~ 'ip_byChannel']








start_time = time.time()
print('Saving...')
train_df[train_df['from']=='train'].to_csv(pathOut+"train_new.csv") 
train_df[train_df['from']=='test'].to_csv(pathOut+"test_new.csv")
print('Done!...'+ str((time.time() - start_time)))


#~ ['ip',
 #~ 'app',
 #~ 'device',
 #~ 'os',
 #~ 'channel',
 #~ 'is_attributed',
 #~ 'day',
 #~ 'hour',
 #~ 'click_AppBefore',
 #~ 'click_AppAfter',
 #~ 'click_speed',
 #~ 'click_byIp',
 #~ 'ip_byChannel',
 #~ 'click_speed_median',
 #~ 'click_speed_distance',
 #~ 'click_byDayByIp',
 #~ 'click_byDayByIp_log',
 #~ 'ratio_click_onAppBefore_byDayByIp',
 #~ 'ratio_click_onAppAfter_byDayByIp',
 #~ 'app_byIp',
 #~ 'app_byChannel',
 #~ 'click_byChannel',
 #~ 'channel_byIP',
 #~ 'click_byIp_log',
 #~ 'channel_byIP_log',
 #~ 'ratio_byChannel',
 #~ 'click_byChannel_log']




