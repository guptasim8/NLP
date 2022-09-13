#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install plotly


# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
df = pd.read_csv('dataset/adult.data',names=columns, na_values='?')
df.head(5)


# In[3]:


print(df.shape)


# In[4]:


df.dtypes


# In[5]:


df.isnull().sum()


# In[6]:


df.nunique()


# In[7]:


df.describe().T


# In[8]:


df['workClass'].value_counts()


# In[9]:


df['workClass'] = df['workClass'].replace(' ?', ' Private')


# In[10]:


df['occupation'].value_counts()


# In[11]:


df['occupation'] = df['occupation'].replace(' ?', ' Prof-specialty')


# In[12]:


df['native-country'].value_counts()


# In[13]:


df['native-country'] = df['native-country'].replace(' ?', ' United-States')


# In[14]:


df['education'].value_counts()


# In[15]:


df['marital-status'].value_counts()


# In[16]:


sns.countplot(x=df['income'], data=df, palette='coolwarm', hue='marital-status');


# In[17]:


# income
df.income = df.income.replace(' <=50K', 0)
df.income = df.income.replace(' >50K', 1)
print(df.income.value_counts())
df.head(5)


# In[18]:


df.corr()


# In[19]:


del df['education-num']
del df['fnlwgt']


# In[20]:


sns.heatmap(df.corr(), annot=True);


# In[21]:


df.hist(figsize=(12,12), layout=(3,3), sharex=False);


# In[22]:


df['education'].value_counts()


# In[23]:


sns.countplot(x=df['income'], palette='coolwarm', hue='education', data=df);


# In[24]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
df1= df.copy()
le = LabelEncoder()
df1= df1.apply(le.fit_transform)
df1.head()


# In[25]:


df1.head(10)


# In[26]:


df1['income'].value_counts()


# In[27]:


24720/7841


# In[28]:


# Class count
count_class_0, count_class_1 = df1['income'].value_counts()

# Divide by class
df_class_0 = df1[df1['income'] == 0]
df_class_1 = df1[df1['income'] == 1]


# In[29]:


# Undersample 0-class and concat the DataFrames of both class
df_class_0_under = df_class_0.sample(count_class_1)
df2 = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df2.income.value_counts())
df2.head()


# In[30]:


df2.reset_index(inplace=True)
df2.head()


# In[31]:


del df2["index"]
df2.head()


# In[32]:


# df2.head()


# In[47]:


# from sklearn import preprocessing
X = df2.iloc[:, 0:12].values
y = df2.iloc[:, [12]].values
print(len(X))

from sklearn.model_selection import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(X, y, test_size=0.20,random_state = 42,stratify=y)
print ("Train:%d +  Test:%d = Total:%d"  % (len(feature_train),len(feature_test),len(feature_train)+len(feature_test)))


# In[48]:


print((labels_train == 0).sum())
print((labels_train == 1).sum())


# In[50]:


feature_test, feature_val, labels_test, labels_val = train_test_split(feature_test, labels_test,test_size=0.50, random_state = 42,stratify=labels_test)
print ("Test:%d +  Val:%d = Total:%d"  % (len(feature_test),len(feature_val),len(feature_test)+len(feature_val)))


# In[51]:


print(feature_train[0])
print(feature_train.shape)


# In[52]:


print(feature_train[0])


# In[53]:


output_df=pd.DataFrame(feature_train,columns = ['age','workClass','education','marital-status','occupation','relationship','race',
                                        'sex','capital-gain','capital-loss','hours-per-week','native-country'])
out_label=pd.DataFrame(labels_train,columns=['class'])

# output_df=output_df.join(out_label)
output_df.head()


# In[54]:


class Dataset(torch.utils.data.Dataset):

    def __init__(self, inputs,labels):
        self.n_samples=len(inputs)
        self.inputs = torch.tensor(inputs,dtype=torch.float32, requires_grad=True)
        self.labels = torch.tensor(labels,dtype=torch.float32)
        
    def __len__(self):
        return self.n_samples

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __getitem__(self, idx):
        return self.inputs[idx],self.labels[idx]


# In[55]:


train_data=Dataset(feature_train,labels_train)
test_data=Dataset(feature_test,labels_test)
val_data=Dataset(feature_val,labels_val)


# In[56]:


class Logistic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Logistic,self).__init__()
        self.linear1 = nn.Linear(input_size,input_size)
        self.linear2 = nn.Linear(input_size,output_size)
        
    def forward(self,input):
        out = torch.relu(self.linear1(input))
        yp = self.linear2(out)
        return yp


# In[57]:


from math import fabs
def l_norm(V,n):
    sum = 0
    for x in V:
        sum += pow(fabs(x),n)
    return pow(sum,1/n)


# In[58]:


def soft_max(x):
    with torch.no_grad():
        y=x.numpy()
        res=np.exp(y)/np.sum(np.exp(y),axis=0)
    return res


# In[59]:


from torch.optim import Adam
from tqdm import tqdm



def train(model, train_data, val_data, learning_rate, epochs,batch_size):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    
    class_pred=[]
    prob_epoch=[]
    Xgrad_epoch=[]
    ydiff_epoch=[]
    Xgrad_aggr_epoch=[]
    prev_val=0
    
    for epoch in range(50):
        total_acc_train = 0
        total_loss_train = 0
        total=0
        
        model.train()
        prob=[]
        diff=[]
        cls=[]
        for train_input, train_label in tqdm(train_dataloader):
            output =model(train_input)
            batch_loss = criterion(output, train_label.squeeze(1).long())
            total_loss_train += batch_loss.item()
            i=0
            for row in output:
                #p=max(soft_max(row))
                p=soft_max(row)
                d=abs(p[0]-p[1]).item()
                prob.append(p)
                diff.append(d)
                i+=1
            cls.extend(list(output.argmax(dim=1).numpy()))
            acc = (output.argmax(dim=1)== train_label.squeeze(1)).sum().item()
            total_acc_train += acc
            total+=train_input.shape[0]
            
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()#
        class_pred.append(cls)
        prob_epoch.append(prob)
        ydiff_epoch.append(diff)
        
        total_acc_val = 0
        total_loss_val = 0
        total_val=0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):
                output = model(val_input)
                batch_loss = criterion(output, val_label.squeeze(1).long())
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1)== val_label.squeeze(1)).sum().item()
                total_acc_val += acc
                total_val+=val_input.shape[0]
        
        agg_grad = []
        with torch.no_grad():
            #Input dimension
            #print(len(train_data.inputs.grad))
            #print(len(train_data.inputs.grad[0]))
            #Gradient wrt input
            for xg in train_data.inputs.grad:
                agg_grad.append(l_norm(xg,2))
            agg_grad = torch.tensor(agg_grad,dtype=torch.float32)
            
            Xgrad_epoch.append(train_data.inputs.grad)
            Xgrad_aggr_epoch.append(agg_grad)
        
            val_acc=total_acc_val / total_val
            print(f'Epochs: {epoch+1}| Train Loss: {total_loss_train / total: .3f}| Train Accuracy: {total_acc_train / total: .3f} | Val Loss: {total_loss_val / total_val: .3f}| Val Accuracy: {val_acc: .3f}')
        
        train_data.inputs.grad.data.zero_()
        if f'{val_acc:.4f}'>f'{prev_val:.4f}' or epoch<15:
            prev_val=val_acc
        else: break
        
    return Xgrad_epoch,Xgrad_aggr_epoch,ydiff_epoch,class_pred,prob_epoch


# In[60]:


model = Logistic(12,2)
learning_rate = 0.001
batch_size=32
iter = 11761 /32
epochs=50


# In[61]:


Xgrad_epoch,Xgrad_aggr_epoch,ydiff_epoch,class_pred,prob_epoch=train(model, train_data, val_data, learning_rate, epochs,batch_size)


# In[62]:


def evaluate(model, test_data,batch_size):
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False)
    total_acc_test = 0
    total=0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):
            output = model(test_input)

            acc = (output.argmax(dim=1) == test_label.squeeze(1)).sum().item()
            total_acc_test += acc
            total+=test_input.shape[0]
    
    print(f'Test Accuracy: {total_acc_test / total: .3f}')


# In[63]:


evaluate(model,test_data,batch_size)


# In[ ]:


# print(class_pred[4:6])
# print(class_pred[4][:5])


# In[64]:


ep=len(prob_epoch)-1
# print(len(prob_epoch))
# print(len(prob_epoch[ep]))
# print(len(ydiff_epoch))
# print(len(ydiff_epoch[ep]))
prob_0=[]
prob_1=[]
diff=ydiff_epoch[ep]
less_than_10=[]
less_than_20=[]
less_than_30=[]
less_than_40=[]
i=0
for d in diff:
    if d<=.40:
        less_than_40.append(i)
    if d<=.30:
        less_than_30.append(i)
    if d<=.20:
        less_than_20.append(i)
    if d<=.10:
        less_than_10.append(i)
    i+=1

for row in prob_epoch[ep]:
    prob_0.append(row[0])
    prob_1.append(row[1])


# In[65]:


print(len(less_than_10))
print(len(less_than_20))
print(len(less_than_30))
print(len(less_than_40))
print(len(diff))


# In[66]:


import math
batch=[]
for i in output_df.index.tolist():
    i=i/batch_size
    i=math.floor(i)+1
    batch.append(i)
train_output_df=output_df.copy()
train_output_df['batch']=batch
for ep in range(len(class_pred)):
    train_output_df['class_pred'+str(ep)]=class_pred[ep]
train_output_df['prob_0']=prob_0
train_output_df['prob_1']=prob_1
train_output_df['abs_diff']=diff
#actual class label
output_df['prob_0']=prob_0
output_df['prob_1']=prob_1
output_df['abs_diff']=diff
output_df=output_df.join(out_label)
train_output_df=train_output_df.join(out_label)
#last epoch prediction
output_df['class_pred']=class_pred[len(class_pred)-1]
train_output_df['class_pred']=class_pred[len(class_pred)-1]
train_output_df.head()


# In[67]:


output_df.head(10)


# In[68]:


train_output_df.to_csv (r'train_output.csv', index = False, header=True)


# In[69]:


#ANALYSIS


# In[70]:


print(len(Xgrad_epoch))
print(Xgrad_epoch[0].shape)
print(len(Xgrad_aggr_epoch))
print(Xgrad_aggr_epoch[0].shape)
print(Xgrad_aggr_epoch[0])


# In[71]:


def findMean(a, n):
    sum = 0
    for i in range(0, n):
        sum += a[i]
    return float(sum/n)

def findMedian(a, n):
    sorted(a)
    if n % 2 != 0:
        return float(a[int(n/2)])
 
    return float((a[int((n-1)/2)] +
                  a[int(n/2)])/2.0)

def findVariance(a,mean, n):
    var = (sum(abs(a-mean)) / n)
    return var


# In[72]:


a=torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,16.0],dtype=float)
print(findMean(a,10))
print(findVariance(a,findMean(a,10),10))


# In[73]:


def findVariance2(a,mean, n):
    anew = []
    for x in a:
        if x>mean:
            anew.append(x-mean)
        else:
            anew.append(mean-x)
    var = (sum(anew) / n)
    return var


# In[74]:


def compute_measures(x,indexes):
    x=[x[i].item() for i in indexes]
    
    minimum=min(x)
    print(f'Min grad  : {minimum}')
    maximum=max(x)
    print(f'Max grad  : {maximum}')
    mean=findMean(x,len(x))
    print(f'Mean grad  : {mean}')
    variance=findVariance2(x,mean,len(x))
    print(f'Variance grad  : {variance}')
    med=findMedian(x,len(x))
    print(f'Median grad : {med}')

    return minimum,maximum,mean,variance,med
minimum,maximum,mean,variance,med=compute_measures(Xgrad_aggr_epoch[0],[1,2,3])
print(minimum,maximum,mean,variance,med)


# In[75]:


index_class_0=out_label.index[out_label['class'] == 0].tolist()
index_class_1=out_label.index[out_label['class'] == 1].tolist()


# In[76]:


def distribution(inputxgrad,indexes):
    ep= []
    mini=[]
    maxi=[]
    variance=[]
    mean=[]
    median=[]
    for epoch in range(len(inputxgrad)):
        ep.append(epoch)
        print(f'--Epoch {epoch+1}--')
        minimum,maximum,m,var,med=compute_measures(inputxgrad[epoch],indexes)
        mini.append(minimum)
        maxi.append(maximum)
        mean.append(m)
        variance.append(var)
        median.append(med)
    fig, axes =plt.subplots(2,3)
    axes[0, 0].plot(ep,variance)
    axes[0, 0].set_title("Variance/Epoch")
    axes[0, 1].plot(ep,mean)
    axes[0, 1].set_title("Mean/Epoch")
    axes[0, 2].plot(ep,maxi)
    axes[0, 2].set_title("Maximum/Epoch")
    axes[1, 0].plot(ep,mini)
    axes[1, 0].set_title("Minimum/Epoch")
    axes[1, 1].plot(ep,median)
    axes[1, 1].set_title("Median/Epoch")
    plt.show()


# In[77]:


distribution(Xgrad_aggr_epoch,index_class_0)


# In[78]:


distribution(Xgrad_aggr_epoch,index_class_1)


# In[ ]:





# In[79]:


ep= []
rank_grad=[]
for epoch in range(len(Xgrad_aggr_epoch)):
    ep.append(epoch)
#     print(f'--Epoch {epoch+1}--')
    rank_g = pd.DataFrame(Xgrad_aggr_epoch[epoch],columns=['Grad']).rank()
#     print(f'Rank shape : {rank_g.shape}')
    rank_grad.append(rank_g)
print(f'Rank shape of grad : {len(rank_grad)}*{len(rank_grad[0])}')   


# In[80]:


data = Xgrad_aggr_epoch[0]
count, bins_count = np.histogram(data, bins=20)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()


# In[81]:


# No of Data points
N2 = len(ydiff_epoch[0])
data2 = ydiff_epoch[0]
count2, bins_count2 = np.histogram(data2, bins=20)
pdf2 = count2 / sum(count2)
cdf2 = np.cumsum(pdf2)
plt.plot(bins_count2[1:], pdf2, color="red", label="PDF")
plt.plot(bins_count2[1:], cdf2, label="CDF")
plt.legend()


# In[82]:


N = len(Xgrad_aggr_epoch[0])
last_ep=len(Xgrad_aggr_epoch)-1
data = Xgrad_aggr_epoch[last_ep]
  
# getting data of the histogram
count, bins_count = np.histogram(data, bins=20)
  
pdf = count / sum(count)
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()


# In[83]:


N2 = len(ydiff_epoch[0])
last_ep=len(ydiff_epoch)-1
data2 = ydiff_epoch[last_ep]
count2, bins_count2 = np.histogram(data2, bins=20)
pdf2 = count2 / sum(count2)
cdf2 = np.cumsum(pdf2)
plt.plot(bins_count2[1:], pdf2, color="red", label="PDF")
plt.plot(bins_count2[1:], cdf2, label="CDF")
plt.legend()


# In[84]:


def compute_measures_diff(x,indexes):
    x=[x[i] for i in indexes]
    
    minimum=min(x)
    print(f'Min diff  : {minimum}')
    maximum=max(x)
    print(f'Max diff  : {maximum}')
    mean=findMean(x,len(x))
    print(f'Mean diff  : {mean}')
    variance=findVariance2(x,mean,len(x))
    print(f'Variance of diff  : {variance}')
    med=findMedian(x,len(x))
    print(f'Median diff : {med}')

    return minimum,maximum,mean,variance,med


# In[85]:


def distributionDiff(inputxdiff,indexes):
    ep= []
    mini=[]
    maxi=[]
    variance=[]
    mean=[]
    median=[]
    for epoch in range(len(inputxdiff)):
        ep.append(epoch)
        print(f'--Epoch {epoch+1}--')
        minimum,maximum,m,var,med=compute_measures_diff(inputxdiff[epoch],indexes)
        mini.append(minimum)
        maxi.append(maximum)
        mean.append(m)
        variance.append(var)
        median.append(med)
    fig, axes =plt.subplots(2,3)
    axes[0, 0].plot(ep,variance)
    axes[0, 0].set_title("Variance/Epoch")
    axes[0, 1].plot(ep,mean)
    axes[0, 1].set_title("Mean/Epoch")
    axes[0, 2].plot(ep,maxi)
    axes[0, 2].set_title("Maximum/Epoch")
    axes[1, 0].plot(ep,mini)
    axes[1, 0].set_title("Minimum/Epoch")
    axes[1, 1].plot(ep,median)
    axes[1, 1].set_title("Median/Epoch")
    plt.show()


# In[86]:


distributionDiff(ydiff_epoch,index_class_0)


# In[87]:


distributionDiff(ydiff_epoch,index_class_1)


# In[88]:


distributionDiff(ydiff_epoch,less_than_10)


# In[89]:


distributionDiff(ydiff_epoch,less_than_20)


# In[90]:


distributionDiff(ydiff_epoch,less_than_30)


# In[91]:


distributionDiff(ydiff_epoch,less_than_40)


# In[92]:


ep2= []
rank_grad2=[]
for epoch in range(len(ydiff_epoch)):
    ep2.append(epoch)
#     print(f'--Epoch {epoch+1}--')
    rank_g = pd.DataFrame(ydiff_epoch[epoch],columns=['Grad']).rank()
#     print(f'Rank shape : {rank_g.shape}')
    rank_grad2.append(rank_g)
print(f'Rank shape of diff : {len(rank_grad)}*{len(rank_grad[0])}') 


# In[ ]:





# In[93]:


from scipy.stats import spearmanr
coeff=[]
pvalue=[]
e=[]
for ep in range(len(rank_grad2)-1):
    e.append(ep)
    coef, p = spearmanr(rank_grad[ep], rank_grad2[ep])
    print(coef,p)
    coeff.append(coef)
    pvalue.append(pvalue)
# print(coeff)


# In[94]:


print(rank_grad[5])


# In[95]:


fig, axes =plt.subplots(1,1)
axes.plot(e,coeff)
axes.set_title("Correlation between Rankings/Epoch")
plt.show()


# In[102]:


# find top k elements
i=0
values=[] 
indexes=[]

for xfind in Xgrad_aggr_epoch:
    value, index = torch.topk(xfind, 40)
    values.append(value.numpy())
    indexes.append(index.numpy())
    print("Epoch:",i+1)
    # print top 26 elements
    print(f'Top {value.shape[0]} gradient values:', value)
    # print index of top 26 elements
    print(f'Top {value.shape[0]} gradient indices:', index.numpy())
    bat=[]
    yout_pred=[]
    yout=[]
    p=[]
    for x in index:
        bat.append(batch[x])
        yout_pred.append(class_pred[i-1][x])
        yout.append(labels_train[x][0])
        p.append(max(prob_epoch[i][x][0],prob_epoch[i][x][1]))
        
    print(f'Batch number:', bat)
    print(f'Actual class   :', yout)
    print(f'Class predected:', yout_pred)
    print("probability: ",p)
    print('\n')
    i+=1


# In[103]:



for point in indexes[len(indexes)-1]:
    point_grad=[]
    point_diff=[]
    print("point:",point)
    for epoch in range(len(Xgrad_aggr_epoch)):
    #     
    #     print(Xgrad_aggr_epoch[epoch][2])
    #     print(len(Xgrad_aggr_epoch[epoch][2]))
        point_grad.append(Xgrad_aggr_epoch[epoch][point])
        point_diff.append(ydiff_epoch[epoch][point])
    plt.plot(range(len(Xgrad_aggr_epoch)),point_grad, label = "grad")
    plt.plot(range(len(Xgrad_aggr_epoch)),point_diff, label = "diff")
    plt.legend()
    plt.show()


# In[104]:


# output_df["batch"].hist(figsize=(12,12));


# In[105]:


from itertools import product, starmap
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return intersection / union

print("Jaccard Similarity between top k gradient indexes in different epochs=",len(indexes))
# result=(jaccard_similarity(list(indexes),list(indexes)))
inputs = product(indexes, indexes)
result = torch.tensor(list(starmap(jaccard_similarity, inputs)),dtype=float).view(len(indexes),len(indexes))
result


# In[108]:


i=0
values2=[]
indexes2=[]
for xfind in ydiff_epoch:
    value, index = torch.topk(torch.tensor(xfind,dtype=float), 40, largest=False)
    values2.append(value.numpy())
    indexes2.append(index.numpy())
    print("Epoch:",i+1)
    # print top k elements
#     print(f'Top {value.shape[0]} probability difference values:', value.numpy())
    # print index of top k elements
    print(f'Top {value.shape[0]} probability difference indices:', index.numpy())
    bat=[]
    yout_pred=[]
    yout=[]
    p=[]
    for x in index:
        bat.append(batch[x])
        yout_pred.append(class_pred[i-1][x])
        yout.append(labels_train[x][0])
        p.append(max(prob_epoch[i][x][0],prob_epoch[i][x][1]))
        
    print(f'Batch number:', bat)
    print(f'Actual class   :', yout)
    print(f'Class predected:', yout_pred)
    print(f'Probability:', p)
    print('\n')
    i+=1


# In[109]:


from itertools import product, starmap
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return intersection / union

print("Jaccard Similarity between top k probability difference indexes in different epochs=",len(indexes))
# result=(jaccard_similarity(list(indexes),list(indexes)))
inputs = product(indexes2, indexes2)
result = torch.tensor(list(starmap(jaccard_similarity, inputs)),dtype=float).view(len(indexes2),len(indexes2))
result


# In[110]:


# print(output_df["batch"].shape)
# output_df["batch"].hist(figsize=(12,12));


# In[111]:


output_df.head()
# for i in indexes:
#     print(output_df["class"].iloc[i])
#     print(output_df["class_pred4"].iloc[i])
#     print(output_df.iloc[[i],-5].values,output_df.iloc[[i],-4].values)


# In[112]:


# trans_out=le.inverse_transform(output_df[:][:12])
# trans_out.head()


# In[113]:


print(output_df.shape)
len(X)


# In[114]:


############### POISONING #############


# In[115]:


from torch.optim import Adam
from tqdm import tqdm

def train2(model, train_data, val_data, learning_rate, epochs,batch_size):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    prev_val=0
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total=0
        model.train()
        
        for train_input, train_label in tqdm(train_dataloader):
            output =model(train_input)
            batch_loss = criterion(output, train_label.squeeze(1).long())
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1)== train_label.squeeze(1)).sum().item()
            total_acc_train += acc
            total+=train_input.shape[0]
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        total_acc_val = 0
        total_loss_val = 0
        total_val=0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):
                output = model(val_input)

                batch_loss = criterion(output, val_label.squeeze(1).long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1)== val_label.squeeze(1)).sum().item()
                total_acc_val += acc
                total_val+=val_input.shape[0]
    
        print(
            f'Epochs: {epoch+1}|Train Loss: {total_loss_train / total: .4f}|Train Accuracy: {total_acc_train / total: .4f} \
            |Val Loss: {total_loss_val / total_val: .4f}|Val Accuracy: {total_acc_val / total_val: .4f}')
        val_acc=total_acc_val / total_val
        if f'{val_acc:.4f}'>f'{prev_val:.4f}' or epoch<15:
            prev_val=val_acc
        else: break


# In[116]:


def replicate(x,y,indexes):
    f=x.copy()
    l=y.copy()
    ind=[i for i in range(len(f))]
    # print (ind)
    ind=np.append(ind,indexes)
    # print (ind)
    # print(len(f))
    f=[f[x][:] for x in ind]
    l=[l[x][:] for x in ind]
    for i in range(len(f)):
        f[i]=f[i].tolist()
        l[i]=l[i].tolist()
    return f,l


# In[117]:


print(len(indexes))


# In[118]:


feature_train_poisoned,labels_train_poisoned=replicate(feature_train,labels_train,indexes[len(indexes)-1])
print("Training data points: ",len(feature_train),"*",len(feature_train[0]))
print("Training data points after poisoning: ",len(feature_train_poisoned),"*",len(feature_train_poisoned[0]))
df_poisoned = pd.DataFrame(feature_train_poisoned)
train_data_poisoned=Dataset(feature_train_poisoned,labels_train_poisoned)


# In[119]:


# print(feature_train[:5])
# print(df_poisoned[:5])


# In[120]:


print("Type of poisoned data: ",type(feature_train_poisoned))


# In[121]:


model2 = Logistic(12,2)


# In[122]:


train2(model2, train_data_poisoned, val_data, learning_rate, epochs,batch_size)


# In[123]:


evaluate(model2,test_data,batch_size)


# In[124]:


#deleting k rows with max gradient
feature_train_poisoned2 = np.delete(feature_train,indexes[len(indexes)-1], axis=0)
labels_train_poisoned2 = np.delete(labels_train,indexes[len(indexes)-1], axis=0)


# In[126]:


train_data_poisoned2=Dataset(feature_train_poisoned2,labels_train_poisoned2)
print(len(train_data_poisoned2))


# In[ ]:


# print(feature_train[:5])
# print(feature_train_poisoned2[:5])


# In[127]:


model3 = Logistic(12,2)


# In[128]:


train2(model3, train_data_poisoned2, val_data, learning_rate, epochs,batch_size)


# In[129]:


evaluate(model3,test_data,batch_size)


# In[ ]:


# MIN DIFF based poisoning


# In[130]:


#Replicating 
feature_train_poisoned3,labels_train_poisoned3=replicate(feature_train,labels_train,indexes2[len(indexes2)-1])
print("Training data points: ",len(feature_train),"*",len(feature_train[0]))
print("Training data points after poisoning: ",len(feature_train_poisoned3),"*",len(feature_train_poisoned3[0]))

df_poisoned3 = pd.DataFrame(feature_train_poisoned3)
train_data_poisoned3=Dataset(feature_train_poisoned3,labels_train_poisoned3)


# In[131]:


model4 = Logistic(12,2)


# In[132]:


train2(model4, train_data_poisoned3, val_data, learning_rate, epochs,batch_size)


# In[133]:


evaluate(model4,test_data,batch_size)


# In[134]:


#deleting k rows with min diff 
feature_train_poisoned4 = np.delete(feature_train,indexes2[len(indexes2)-1], axis=0)
labels_train_poisoned4 = np.delete(labels_train,indexes2[len(indexes2)-1], axis=0)


# In[135]:


train_data_poisoned4=Dataset(feature_train_poisoned4,labels_train_poisoned4)


# In[136]:


print(len(train_data_poisoned4))


# In[137]:


model5 = Logistic(12,2)


# In[138]:


train2(model5, train_data_poisoned4, val_data, learning_rate, epochs,batch_size)


# In[139]:


evaluate(model5,test_data,batch_size)


# In[ ]:




