import statsmodels.tsa.stattools as ts
from keras.layers import Bidirectional,LSTM,RNN
import warnings
import math
from scipy.spatial.distance import squareform, pdist, cdist
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import kde
from scipy.integrate import tplquad,dblquad,quad
warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib.pyplot as plt  
from keras.models import Sequential
from sklearn import preprocessing
import copy
# tf.enable_eager_execution()
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
#######################################################################读取数据，获取批次时间
data = pd.read_csv('100_Batches_IndPenSim_V3.csv')
data1 = np.array(data)
index = data.columns
batch_time = [0]#######################################################批次时间
for i in range(1,len(data)):
    if data1[i-1,13] > 1 and  data1[i,13] < 0.1:
        batch_time.append(i)
batch_time.append(113934)    ##################################共100个batch，长度不一

train_batch_time1 = batch_time[:11]
train_batch_time2 = batch_time[30:41]
train_batch_time3 = batch_time[60:71]       #####训练批次
######################################################################互信息筛选变量
def MI(x,y,l):   #####x,y变量向量
    d=(4/(2+2))**(1/(4+2))*(l**(-1/(2+4)))   ######silverman 方法
    px=kde.gaussian_kde(x,bw_method=d)
    pxy=kde.gaussian_kde((x,y),bw_method=d)
    py=kde.gaussian_kde(y,bw_method=d)
    a=px(x)
    b=py(y)
    c=pxy((x,y))
    mi=0
    for i in range(l):
        mi=mi+math.log(c[i]/a[i]/b[i])/math.log(2)
    mi=mi/l
    return(mi)
######################################################################蒙特卡洛法计算互信息阈值
mi_r = []
for i in range(1000):
    print(i)
    a = np.random.normal(0,1,1150)
    b = np.random.normal(0,1,1150)
    mi_r.append(MI(a,b,1150))
# mi_s = np.mean(mi_r) + 6*np.std(mi_r) 
np.save('mi_r.npy',mi_r)
px=kde.gaussian_kde(mi_r,bw_method='silverman')###############得到概率密度估计函数
for mi_s in np.arange(0,1,0.001):
    if quad(lambda  x:px(x),-mi_s,mi_s)[0] > 0.99:
        mi_s = mi_s
        break######################对函数积分得到阈值

###################################################################分布可视化
# plt.figure(figsize=(16,9))
plt.style.use('seaborn')
c={"KDE" : mi_r}
d={"Histogram" : mi_r}
weights = np.ones_like(mi_r)/float(len(mi_r))
ax = pd.DataFrame(d).plot(kind = 'hist', bins = 25, color = 'lightgreen',figsize=(16,9),density=True)
pd.DataFrame(c).plot(kind ='kde', color='b',figsize=(16,9),bw_method='silverman',ax=ax)
# plt.hist(mi_r,color='lightgreen',bins=25,label='Histogram',stacked=True,weights=weights)
plt.ylabel('Proportation',fontsize=14)
plt.xlabel('Mutual Information',fontsize=14)
# plt.xlim(0.02,0.07)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
#############################################################互信息筛选变量
drop = [0,2,21,23,26,27,29]
data_use = np.vstack((data1[:,1],data1[:,3:21].T))
data_use = data_use.T
data_use = np.vstack((data_use.T,data1[:,22]))#################################################用到的变量
data_use = data_use.T
data_use = np.hstack((data_use,data1[:,24:26]))
data_use = np.vstack((data_use.T,data1[:,28]))
data_use = data_use.T

mi_ab = []
for j in range(len(data_use.T)):
    print(j)
    mi_1 = 0
    if j!= 11:
        for i in range(10):
             mi_1 =  mi_1 + MI(data_use[train_batch_time1[i]:train_batch_time1
                [i+1],j],data_use[train_batch_time1[i]:train_batch_time1[i+1],11],len(data_use[train_batch_time1[i]:train_batch_time1
                   [i+1],j]))
        for i in range(10):
             mi_1 =  mi_1 + MI(data_use[train_batch_time2[i]:train_batch_time2
                [i+1],j],data_use[train_batch_time2[i]:train_batch_time2[i+1],11],len(data_use[train_batch_time2[i]:train_batch_time2
                   [i+1],j]))
        for i in range(10):
             mi_1 =  mi_1 + MI(data_use[train_batch_time3[i]:train_batch_time3
                [i+1],j],data_use[train_batch_time3[i]:train_batch_time3[i+1],11],len(data_use[train_batch_time3[i]:train_batch_time3
                   [i+1],j]))                                                                                               
        mi_ab.append(mi_1/30)                                                                                       

drop1 = []
for i in range(len(mi_ab)):
    if mi_ab[i] < mi_s:
        drop1.append(i)

plt.figure(figsize=(16,9))
plt.bar(range(1,23),mi_ab)
plt.ylabel('Mutual Information',fontsize=14)
plt.xlabel('Variable No.',fontsize=14)
plt.hlines(mi_s,0,len(mi_ab),linewidth=3,color='r',label='Threshold')
# plt.xlim(0.02,0.07)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

data_use1 = np.hstack((data_use[:,:2],data_use[:,3:8]))
data_use1 = np.hstack((data_use1,data_use[:,9:]))
#########################################################################局部近邻标准化

for i in range(len(train_batch_time1)-1):#################################提取训练数据normal
    if i == 0:
        normal = data_use1[train_batch_time1[i]:train_batch_time1[i+1],:]
    else:
        normal = np.vstack((normal,data_use1[train_batch_time1[i]:train_batch_time1[i+1],:]))
for i in range(len(train_batch_time2)-1):
    normal = np.vstack((normal,data_use1[train_batch_time2[i]:train_batch_time2[i+1],:]))
for i in range(len(train_batch_time3)-1):
    normal = np.vstack((normal,data_use1[train_batch_time3[i]:train_batch_time3[i+1],:]))
#######################################################标准化   
# train_normal = np.empty(shape=normal.shape) 
# for i in range(len(normal)):
#     print(i)
#     dev=[]
#     for j in range(len(normal)):
#         dev.append(np.sqrt(np.sum([(a-b)*(a-b) for a, b in zip(normal[i,:], normal[j,:])])))
#     m = dev
#     min_index = pd.Series(m).sort_values().index[:500].tolist()
#     for h in range(500):
#         if h == 0:
#             center = normal[min_index[h],:]
#         else:
#             center = np.vstack((center,normal[min_index[h],:]))       
#     train_normal[i,:]=(normal[i,:]-np.mean(center,0))/np.std(center,0)

# # np.save('train_normal.npy',train_normal)
# train_normal = np.load('train_normal.npy')
# train_normal=np.nan_to_num(train_normal)
mean = np.mean(normal,axis=0)
std = np.std(normal,axis=0)
train_normal = (normal-mean)/std
#############################################################定义模型
class MIM_AE(tf.keras.layers.Layer):
    def __init__(self,output_size1,output_size_N, output_size_S, output_size2, return_sequences,**kwargs):
        super(MIM_AE,self).__init__()
        self.output_size1 = output_size1#############################ht
        self.output_size_N = output_size_N#############################dt
        self.output_size_S = output_size_S#############################tt
        self.output_size2 = output_size2#############################Ht        
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        super(MIM_AE,self).build(input_shape)
        input_size = int(input_shape[-1])
        
        self.wf = self.add_weight('wf', shape=(input_size,output_size1))
        self.wi = self.add_weight('wi', shape=(input_size,output_size1))
        self.wo = self.add_weight('wo', shape=(input_size,output_size1))
        self.wc = self.add_weight('wc', shape=(input_size,output_size1))
        
        self.uf = self.add_weight('uf', shape=(output_size1,output_size1))
        self.ui = self.add_weight('ui', shape=(output_size1,output_size1))
        self.uo = self.add_weight('uo', shape=(output_size1,output_size1))
        self.uc = self.add_weight('uc', shape=(output_size1,output_size1))
        
        self.bf = self.add_weight('bf', shape=(1,output_size1))
        self.bi = self.add_weight('bi', shape=(1,output_size1))
        self.bo = self.add_weight('bo', shape=(1,output_size1))
        self.bc = self.add_weight('bc', shape=(1,output_size1))
        
        self.wf1 = self.add_weight('wf1', shape=(output_size1,output_size_N))
        self.wi1 = self.add_weight('wi1', shape=(output_size1,output_size_N))
        self.wo1 = self.add_weight('wo1', shape=(output_size1,output_size_N))
        self.wc1 = self.add_weight('wc1', shape=(output_size1,output_size_N))
        
        self.uf1 = self.add_weight('uf1', shape=(output_size_N,output_size_N))
        self.ui1 = self.add_weight('ui1', shape=(output_size_N,output_size_N))
        self.uo1 = self.add_weight('uo1', shape=(output_size_N,output_size_N))
        self.uc1 = self.add_weight('uc1', shape=(output_size_N,output_size_N))
        
        self.bf1 = self.add_weight('bf1', shape=(1,output_size_N))
        self.bi1 = self.add_weight('bi1', shape=(1,output_size_N))
        self.bo1 = self.add_weight('bo1', shape=(1,output_size_N))
        self.bc1 = self.add_weight('bc1', shape=(1,output_size_N))
        
        self.wf2 = self.add_weight('wf2', shape=(output_size_N,output_size_S))
        self.wi2 = self.add_weight('wi2', shape=(output_size_N,output_size_S))
        self.wo2 = self.add_weight('wo2', shape=(output_size_N,output_size_S))
        self.wc2 = self.add_weight('wc2', shape=(output_size_N,output_size_S))
        
        self.uf2 = self.add_weight('uf2', shape=(output_size2,output_size_S))
        self.ui2 = self.add_weight('ui2', shape=(output_size2,output_size_S))
        self.uo2 = self.add_weight('uo2', shape=(output_size2,output_size_S))
        self.uc2 = self.add_weight('uc2', shape=(output_size2,output_size_S))
        
        self.bf2 = self.add_weight('bf2', shape=(1,output_size_S))
        self.bi2 = self.add_weight('bi2', shape=(1,output_size_S))
        self.bo2 = self.add_weight('bo2', shape=(1,output_size_S))
        self.bc2 = self.add_weight('bc2', shape=(1,output_size_S))
        
        self.vo = self.add_weight('vo', shape=(output_size_S,output_size_S))
        
        self.wi3 = self.add_weight('wi3', shape=(output_size1,output_size2))
        self.wo3 = self.add_weight('wo3', shape=(output_size1,output_size2))
        self.wc3 = self.add_weight('wc3', shape=(output_size1,output_size2))
        
        self.ui3 = self.add_weight('ui3', shape=(output_size2,output_size2))
        self.uo3 = self.add_weight('uo3', shape=(output_size2,output_size2))
        self.uc3 = self.add_weight('uc3', shape=(output_size2,output_size2))
        
        self.bi3 = self.add_weight('bi3', shape=(1,output_size2))
        self.bo3 = self.add_weight('bo3', shape=(1,output_size2))
        self.bc3 = self.add_weight('bc3', shape=(1,output_size2))
        
    
    def call(self, x):
        sequence_outputs1 = []
        sequence_outputs_N = []
        sequence_outputs_S = []
        sequence_outputs2 = []
        for i in range(sequence_length):
            if i == 0:
                xt = x[:, 0, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + self.bo)
                gt = tf.tanh(tf.matmul(xt, self.wc) + self.bc)
                ct = it * gt
                ht = ot* tf.tanh(ct)
                
                ft1 = tf.sigmoid(tf.matmul(ht, self.wf1) + self.bf1)
                it1 = tf.sigmoid(tf.matmul(ht, self.wi1) + self.bi1)
                ot1 = tf.sigmoid(tf.matmul(ht, self.wo1) + self.bo1)
                gt1 = tf.tanh(tf.matmul(ht, self.wc1) + self.bc1)
                nt =  it1 * gt1
                dt = ot1* tf.tanh(nt)
                
                ft2 = tf.sigmoid(tf.matmul(dt, self.wf2) + self.bf2)
                it2 = tf.sigmoid(tf.matmul(dt, self.wi2) + self.bi2)
                gt2 = tf.tanh(tf.matmul(dt, self.wc2) + self.bc2)
                st = it2 * gt2
                ot2 = tf.sigmoid(tf.matmul(dt, self.wo2) + tf.matmul(st, self.vo) + self.bo2)
                tt = ot2* tf.tanh(st)
                
                it3 = tf.sigmoid(tf.matmul(ht, self.wi3) + self.bi3)
                gt3 = tf.tanh(tf.matmul(ht, self.wc3) + self.bc3)
                Ct = tt + it3 * gt3
                ot3 = tf.sigmoid(tf.matmul(ht, self.wo3) + self.bo3)
                Ht = ot3* tf.tanh(Ct)
                
            else:
                xt = x[:, i, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + tf.matmul(ht, self.uf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + tf.matmul(ht, self.ui) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + tf.matmul(ht, self.uo) + self.bo)
                gt = tf.tanh(tf.matmul(xt, self.wc) + tf.matmul(ht, self.uc) + self.bc)
                ct = ft * ct + it * gt
                ht1 = ot* tf.tanh(ct)############################################# ht1为第t个时刻的ht，计算后再将ht1赋值给ht
                hht = tf.subtract(ht1,ht)#########################################hht定义为差分项
                
                ft1 = tf.sigmoid(tf.matmul(hht, self.wf1) + tf.matmul(nt, self.uf1) + self.bf1)
                it1 = tf.sigmoid(tf.matmul(hht, self.wi1) + tf.matmul(nt, self.ui1) + self.bi1)
                gt1 = tf.tanh(tf.matmul(hht, self.wc1) + tf.matmul(nt, self.uc1) + self.bc1)
                nt =  ft1 * nt + it1 * gt1
                ot1 = tf.sigmoid(tf.matmul(hht, self.wo1) + tf.matmul(nt, self.uo1)+ self.bo1)
                dt = ot1* tf.tanh(nt)
                
                ft2 = tf.sigmoid(tf.matmul(dt, self.wf2) + tf.matmul(Ct, self.uf2) + self.bf2)
                it2 = tf.sigmoid(tf.matmul(dt, self.wi2) + tf.matmul(Ct, self.ui2) + self.bi2)
                gt2 = tf.tanh(tf.matmul(dt, self.wc2) +  tf.matmul(Ct, self.uc2) + self.bc2)
                st = ft2 * st + it2 * gt2
                ot2 = tf.sigmoid(tf.matmul(dt, self.wo2) + tf.matmul(Ct, self.uo2) + tf.matmul(st, self.vo) + self.bo2)
                tt = ot2* tf.tanh(st)
                
                it3 = tf.sigmoid(tf.matmul(ht, self.wi3) + tf.matmul(Ht, self.ui3) + self.bi3)
                gt3 = tf.tanh(tf.matmul(ht, self.wc3) + tf.matmul(Ht, self.uc3) + self.bc3)
                Ct = tt + it3 * gt3
                ot3 = tf.sigmoid(tf.matmul(ht, self.wo3) + tf.matmul(Ht, self.uo3) + self.bo3)
                Ht = ot3* tf.tanh(Ct)
                ht = ht1
            sequence_outputs1.append(ht)
            sequence_outputs_N.append(dt)
            sequence_outputs_S.append(tt)
            sequence_outputs2.append(Ht)
        sequence_outputs1 = tf.stack(sequence_outputs1)
        sequence_outputs1 = tf.transpose(sequence_outputs1, (1, 0, 2))
#        
        sequence_outputs_N = tf.stack(sequence_outputs_N)
        sequence_outputs_N = tf.transpose(sequence_outputs_N, (1, 0, 2))
        
        sequence_outputs_S = tf.stack(sequence_outputs_S)
        sequence_outputs_S = tf.transpose(sequence_outputs_S, (1, 0, 2))
        
        sequence_outputs2 = tf.stack(sequence_outputs2)
        sequence_outputs2 = tf.transpose(sequence_outputs2, (1, 0, 2))
        if self.return_sequences:
            return sequence_outputs2
        return sequence_outputs2[:, -1, :]
    def get_config(self):
        
        
        config = {'output_size1':self.output_size1,
                       'output_size_N':self.output_size_N,
                       'output_size_S':self.output_size_S,
                       'output_size2':self.output_size2,
                       'return_sequences':self.return_sequences}
        base_config = super(MIM_AE,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



##############################################################划分训练数据输入模型格式


sequence_length = 6
input_size = len(train_normal.T)-1
output_size1 = 32
output_size_N = 64
output_size_S = 128
output_size2 = 128

seg = []
for i in range(10):
    seg.append(train_batch_time1[i+1]-train_batch_time1[i])
for i in range(10):
    seg.append(train_batch_time2[i+1]-train_batch_time2[i])
for i in range(10):
    seg.append(train_batch_time3[i+1]-train_batch_time3[i])
data_batch = []
for i in range(int(len(seg)*0.7)):
    if i == 0:
        s = 0
    else:
        s = 0
        for u in range(i):
            s = s + seg[u]   
    print(s)
    for j in range(seg[i]-sequence_length+1):
        data_batch.append(train_normal[j+s:j+s+sequence_length,:])
data_batch=np.array(data_batch)

valid_batch = []
for i in range(int(len(seg)*0.7),len(seg)):
    s = 0
    for u in range(i):
        s = s + seg[u]   
    print(s)
    for j in range(seg[i]-sequence_length+1):
        valid_batch.append(train_normal[j+s:j+s+sequence_length,:])
valid_batch=np.array(valid_batch)

x_train = tf.convert_to_tensor(np.concatenate((data_batch[:,:,:9],data_batch[:,:,10:]),axis=2))
y_train = tf.convert_to_tensor(data_batch[:,-1,9].reshape(len(data_batch),1,1))
x_valid = tf.convert_to_tensor(np.concatenate((valid_batch[:,:,:9],valid_batch[:,:,10:]),axis=2))
y_valid = tf.convert_to_tensor(valid_batch[:,-1,9].reshape(len(valid_batch),1,1))

#################################################################################搭建模型开始训练
#new_model = keras.models.load_model('sl=3.h5', custom_objects={'MIM_AE':MIM_AE})
#model = new_model

model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(64,input_shape=(1,33)))
#model.add(tf.keras.layers.Dense(64))
model.add(MIM_AE(output_size1,output_size_N, output_size_S, output_size2,return_sequences = True))
#model.add(MIM_DE(output_size1_d,output_size_N_d, output_size_S_d, output_size2_d,return_sequences = False))
#model.add(tf.keras.layers.LSTM(128))
# model.add(MIM_AE(output_size1,output_size_N, output_size_S, output_size2,return_sequences = True))
# model.add(MIM_AE(output_size1,output_size_N, output_size_S, output_size2,return_sequences = True))
#model.add(tf.keras.layers.Dense(128,activation='sigmoid'))
#model.add(tf.keras.layers.Dense(64,activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss=tf.keras.losses.MSE,optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1,patience=20,verbose=1,mode='auto',baseline=None,restore_best_weights=False)
history = model.fit(x_train, y_train, batch_size=100, epochs = 1000, steps_per_epoch=5, validation_data=(x_valid,y_valid),verbose=1,callbacks=[early_stopping])
# history = model.fit(x_train, y_train, batch_size=50, epochs =100, steps_per_epoch=5, validation_data=(x_valid,y_valid),verbose=1)
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='valid')
plt.xlabel('epoch', fontsize='14')
plt.ylabel('loss', fontsize='14')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.show()
