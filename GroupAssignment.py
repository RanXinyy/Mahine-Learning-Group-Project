import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import scipy as sp
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import seaborn as sns
from datetime import datetime
import matplotlib.ticker as ticker

import math
#%%
#compute RMSE
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

#load data for neural network
def load_data():
    df=pd.read_excel('data_final.xlsx')
    df.set_index(["quarter, year"], inplace=True)
    df=df.dropna(axis=0)
    return df

def Z_score(dataset):
    scaler=StandardScaler()
    scaler.fit(dataset)
    dataset_scaler=pd.DataFrame(scaler.transform(dataset))
    return dataset_scaler

def Split_dataset(dataset):
    df_X, df_y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(df_X,df_y,test_size = 0.05 )
    return df_X, df_y,X_train, X_test, y_train, y_test

#log(y) for NN
def y_log(dataset):
    dataset.iloc[:,-1]=np.log10(dataset.iloc[:,-1])
    dataset=dataset.dropna(axis=0)
    return dataset

#split dataset to trainset and testset for NN
def split_data(dataset):
    x_sample=dataset.iloc[:,:-1]    # feature array
    y_sample=dataset.iloc[:,-1]    # target value array
    #5% of total as testset
    X_train,X_test,y_train,y_test=train_test_split(x_sample,y_sample,test_size=0.05)
    return X_train,X_test,y_train,y_test

#build  NN model
def Model_cv(X_train,X_test,y_train,y_test,layer,solver,activition):
    
    X_all = pd.concat([X_train,X_test])  #merge x
    y_all = pd.concat([y_train,y_test])  #merge y
    
    cv = KFold(n_splits=10)    # 10 folds cross validation

    lst_rmse_train=[]   #those list to record 
    lst_rmse_test=[]
    lst_score=[]
    
    #model build
    model=MLPRegressor(hidden_layer_sizes=(layer, ),solver=solver,alpha=1e-5,random_state=1,activation=activition) 
    # different training set to build
    for k, (train, test) in enumerate(cv.split(X_all, y_all)):
        X_train_cv, X_test_cv = X_all.iloc[list(train),:], X_all.iloc[list(test),:]
        y_train_cv, y_test_cv = y_all.iloc[list(train)], y_all.iloc[list(test)]
        model.fit(np.array(X_train_cv), np.array(y_train_cv))
        
        pred_y=model.predict(np.array(X_test_cv))   #predict y on x_test
        pred_y_train=model.predict(np.array(X_train_cv))   #predict y on x_train
        score=model.score(np.array(X_train_cv),np.array(y_train_cv))  #score accuracy
        
        # calculate rmse
        rmse_train=rmse(pred_y_train,np.array(y_train_cv).ravel())
        rmse_test=rmse(pred_y,np.array(y_test_cv).ravel())
        
        lst_rmse_train.append(rmse_train)
        lst_rmse_test.append(rmse_test)
        lst_score.append(score)
        
    arr_rmse_train=np.array(lst_rmse_train)
    arr_rmse_test=np.array(lst_rmse_test)
    arr_score=np.array(lst_score)

    print("layer in NN: %d.  solver in NN:  %s   activition in NN:  %s" % (layer,solver,activition))
    print("rmse_train_std in NN: %.4f" %arr_rmse_train.std())
    print("rmse_train_mean in NN: %.4f" %arr_rmse_train.mean())
    print("rmse_test_std in NN: %.4f" %arr_rmse_test.std())
    print("rmse_test_mean in NN: %.4f" %arr_rmse_test.mean())
    print("score_std in NN: %.4f" %arr_score.std())
    print("score_mean in NN: %.4f" %arr_score.mean())
    return arr_rmse_train,arr_rmse_test,arr_score

#use PCA
def PCA_re(X_train,X_test,y_train,n):
    pca = PCA(n_components=n)
    pca.fit(X_train,y_train)
    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)
    return X_train_pca,X_test_pca
#%%
#############################Neural Network Start##############################   

df=load_data()
df=y_log(df)
X_train,X_test,y_train,y_test = split_data(df)
X_train,X_test=Z_score(X_train),Z_score(X_test)
#X_train_pca,X_test_pca=PCA_re(X_train,X_test,y_train,0.8)

####--------------------parameter selection--------------------------#####
#solver_range=['sgd','lbfgs','adam']
#activition_range=['relu','identity','logistic','tanh']
#for i in range(1,11):
#    for j in solver_range:
#        for k in activition_range:
#            df=load_data()
#            arr_rmse_train,arr_rmse_test,arr_score=Model_cv(X_train,X_test,y_train,y_test,i,j,k) 
#            print('----------------------------------------')
###--------------------parameter selection END--------------------------#####

#arr_rmse_train,arr_rmse_test,arr_score=Model_cv(X_train,X_test,y_train,y_test,4,'lbfgs','identity')

##---------------------------fluctuation 10 folds--------------------------------#####
#x=np.array(range(10))
#y_1,y_2,y_3=arr_rmse_train,arr_rmse_test,arr_score
#plt.style.use('ggplot')
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(x,y_1,'b--')
#ax.plot(x,y_2,'r--')
##ax.plot(x,y_3,'g--')
#plt.legend(['training rmse','testing rmse'],ncol=1,loc=0)
#ax.set_title('Fluctuation of 10 Folds Cross Validation')
#ax.set_xlabel('Times')
#ax.set_ylabel('RMSE')
#plt.show()
#fig.savefig('fluctuation')
##---------------------------fluctuation 10 folds  END--------------------------------#####

###--------------------Thermal map--------------------——————######   
    
#df=load_data()
#dfData=df.corr()
#plt.subplots(figsize=(20, 10)) # set the size of figure
## cmap = sns.cubehelix_palette(start = 0.8, rot = 3, gamma=1, as_cmap = True)
#cmap = sns.cubehelix_palette(start = 0.8, rot = 3, gamma=0.2, as_cmap = True)
#sns.heatmap(dfData, annot=False, vmax=0.5, square=True, cmap='Blues')
##plt.savefig('BluesStateRelation.png')
#plt.show()

###--------------------Thermal map END-------------------——————######       
    
###--------------------Actual VS Predict--------------------——————######   

#df=load_data()
#df=y_log(df)
#X_train=df.iloc[:,:-1]
#y_train=df.iloc[:,-1]
#X_train=Z_score(X_train)
#model=MLPRegressor(hidden_layer_sizes=(4, ),solver='lbfgs',alpha=1e-5,random_state=1,activation='identity') 
#model.fit(X_train,y_train)
#pred_y=model.predict(X_train) 
#
#X_time = [datetime.strptime(i[-4:]+'-'+str(int(i[7:8])*3),'%Y-%m') for i in df.index]
#x=np.array(range(0,df.shape[0]))
#y_1=np.array(y_train)
#y_2=pred_y
#plt.style.use('ggplot')
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(X_time,y_1,'b--')
#ax.scatter(X_time,y_2,c='red')
#plt.legend(['actural Y','predict Y'],ncol=1,loc=0)
#ax.set_title('Actual VS Predict ')
#plt.xlabel('Date')
#plt.ylabel('log(y) (log( HIV CASE))')
#plt.show()
#fig.savefig('actualvspredict')
    
###--------------------Actual VS Predict  END--------------------——————######     
    
    
###--------------------HIV TRENDS--------------------——————######
#df=load_data()
#x=[datetime.strptime(i[-4:]+'-'+str(int(i[7:8])*3),'%Y-%m') for i in df.index]
#y=df.iloc[:,-1]
#plt.style.use('ggplot')
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(x,y,"b-")
#ax.locator_params('x',nbins=20)
#plt.xlabel('Date')
#plt.ylabel('HIV Incidence')
#plt.show()
#fig.savefig("HIV Incidence")     
###--------------------HIV TRENDS END--------------------——————######     

###########################Neural Network End##################################    
#%%
#########################Support Vector Regression Start#######################
df_original = load_data()
df_original['Y'] = np.log10(df_original['Y'])
    
X_all = df_original.iloc[:,:-1]
y_all = df_original.iloc[:,-1]
    
#use GridSearchCV to find best parameters

our_model = SVR()
parameters = [
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 100, 1000, 100000],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
    },
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 100, 1000, 100000],
        'kernel': ['linear']
    },
    {
         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 100, 1000, 100000],
         'kernel': ['poly'],
         'degree': [1, 2, 3]
     }
]
scaler_std = StandardScaler()
scaler_std.fit(X_all)#normalize X_all through z-score normalization
X_all_std = scaler_std.transform(X_all)
my_rmse_score = make_scorer(rmse,greater_is_better=False)
lst_best_params = []
for i in parameters:
    grid_search = GridSearchCV(our_model, param_grid=i,scoring=my_rmse_score,cv = 10)
    grid_search.fit(X_all_std, y_all)
    print(grid_search.best_params_)
    lst_best_params.append(grid_search.best_params_)
   
#use selected best combines of parameters to construct SVR model
svr_rbf = SVR(kernel = lst_best_params[0]['kernel'], C = lst_best_params[0]['C'] , gamma = lst_best_params[0]['gamma'])
svr_linear = SVR(kernel = lst_best_params[1]['kernel'], C = lst_best_params[1]['C'])
svr_poly = SVR(kernel = lst_best_params[2]['kernel'], C = lst_best_params[2]['C'],degree= lst_best_params[2]['degree'])

#10-fold validation to compute rmse   
lst_rmse_rbf = []
lst_rmse_linear = []
lst_rmse_poly = []
test_lst = []
kfold = KFold(n_splits=10,shuffle=True)
for k, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
    X_train_kfold = X_all.iloc[train_idx,:]
    X_test_kfold = X_all.iloc[test_idx,:]
    test_lst.append(test_idx)
    #standardise X
    std_scaler = StandardScaler()
    std_scaler.fit(X_train_kfold)
    X_train_kfold_std = std_scaler.transform(X_train_kfold)
    X_test_kfold_std = std_scaler.transform(X_test_kfold)
    y_train_kfold = y_all.iloc[train_idx]
    y_test_kfold = y_all.iloc[test_idx]
    #train dada and predict
    
    svr_rbf.fit(X_train_kfold_std,y_train_kfold)
    y_predict_rbf = svr_rbf.predict(X_test_kfold_std)
    
    svr_linear.fit(X_train_kfold_std,y_train_kfold)
    y_predict_linear = svr_linear.predict(X_test_kfold_std)
    
    svr_poly.fit(X_train_kfold_std,y_train_kfold)
    y_predict_poly = svr_poly.predict(X_test_kfold_std)
    
    lst_rmse_rbf.append(rmse(y_test_kfold,y_predict_rbf))
    lst_rmse_linear.append(rmse(y_test_kfold,y_predict_linear))
    lst_rmse_poly.append(rmse(y_test_kfold,y_predict_poly))
        
print("rbf avg rmse in SVR:",sum(lst_rmse_rbf)/len(lst_rmse_rbf))
print("rbf std rmse in SVR:",math.sqrt(((np.array(lst_rmse_rbf) - (sum(lst_rmse_rbf)/len(lst_rmse_rbf))) ** 2).sum()))
print("linear avg rmse in SVR:",sum(lst_rmse_linear)/len(lst_rmse_linear))
print("linear std rmse in SVR:",math.sqrt(((np.array(lst_rmse_linear) - (sum(lst_rmse_linear)/len(lst_rmse_linear))) ** 2).sum()))
print("poly avg rmse in SVR:",sum(lst_rmse_poly)/len(lst_rmse_poly))
print("poly std rmse SVR:",math.sqrt(((np.array(lst_rmse_poly) - (sum(lst_rmse_poly)/len(lst_rmse_poly))) ** 2).sum()))
    

#plot fitting curve for the above three models
svr_rbf = SVR(kernel = lst_best_params[0]['kernel'], C = lst_best_params[0]['C'] , gamma = lst_best_params[0]['gamma'])
svr_linear = SVR(kernel = lst_best_params[1]['kernel'], C = lst_best_params[1]['C'])
svr_poly = SVR(kernel = lst_best_params[2]['kernel'], C = lst_best_params[2]['C'],degree= lst_best_params[2]['degree'])

svr_rbf.fit(X_all_std,y_all)
svr_linear.fit(X_all_std,y_all)
svr_poly.fit(X_all_std,y_all)

y_svr_rbf = svr_rbf.predict(X_all_std)
y_svr_linear = svr_linear.predict(X_all_std)
y_svr_poly = svr_poly.predict(X_all_std)

for_graph = pd.DataFrame(index=X_all.index,columns=['Data','rbf','linear','poly'])
for_graph['data'] = y_all
for_graph['rbf'] = y_svr_rbf
for_graph['linear'] = y_svr_linear
for_graph['poly'] = y_svr_poly

plt.style.use('ggplot')
fig = plt.figure(figsize=[9,6])
X_time = [datetime.strptime(i[-4:]+'-'+str(int(i[7:8])*3),'%Y-%m') for i in for_graph.index]
x = np.array(range(0,for_graph.shape[0]))
y_1 = np.array(for_graph['data'])
y_2 = np.array(for_graph['rbf'])
y_3 = np.array(for_graph['linear'])
y_4 = np.array(for_graph['poly'])
plt.plot(X_time, y_1,'ro',label='data')
plt.plot(X_time, y_3,'g-',label='rbf')
plt.plot(X_time, y_2,'c-.',label='linear')
plt.plot(X_time, y_4,'y--',label='poly')

plt.title('Support Vector Regression')
plt.xlabel("Time")
plt.ylabel('log(HIV case no.)')
plt.legend(loc='upper left',fontsize=12)
plt.show()
######################Support Vector Regression End############################
#%%
for i in range(24):
    plt.subplot(5,5,i+1)
    sns.regplot(df.columns[i],"Y",data=df,color='steelblue',marker='+').yaxis.set_major_locator(ticker.NullLocator())
    sns.regplot(df.columns[i],"Y",data=df,color='steelblue',marker='+').xaxis.set_major_locator(ticker.NullLocator())
    plt.axis("off")
