# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:27:56 2018

@author: ahmed
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:39:37 2018
#my first trial
@author: nb
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import inspect
from sklearn.metrics import classification_report
from sklearn import cross_validation
#from keras.models import Sequential
#from keras.layers import Dense
#from keras import layers
import logging
from sklearn.utils import class_weight
import gc
from sklearn.neural_network import MLPClassifier 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier





class PredictClass(object):
 
    trainFrame=pd.DataFrame()
    
    testFrame=pd.DataFrame()
    

###################################################################################
    ########## Loading Service
    ##########  3 services
##################################################################################
    
    def LoadReshapeTrain(cls,fileName): 
      # load traing file and reshape the features column
        cls.trainFrame=pd.read_csv(fileName)
        num_cols=cls.trainFrame.shape[1]
        print (num_cols)  # print the dimensions
        cls.trainFrame['FEATURES'] = cls.trainFrame['FEATURES'].str.replace(',','.') #some versions have problems converting string to flost with ,
#        cls.trainFrame['ANSWER'] = cls.trainFrame['ANSWER'].str.replace(',','.') #some versions have problems converting string to flost with ,
        
        tempFrame=pd.DataFrame(cls.trainFrame['FEATURES'].str.split(' ').tolist()).astype(float) # reshape step into 299 columns
        
        cls.trainFrame.drop(['FEATURES'], axis=1, inplace=True)
        num_cols=tempFrame.shape[1] # print dimensions 
        print (num_cols)   
        cls.trainFrame=pd.concat([tempFrame,cls.trainFrame],axis=1)
        cls.trainFrame = cls.trainFrame.drop(['ROWID'],1)
         ######################################################################################   
       
    def LoadReshapeTest(cls,testfileName):
      # load traing file and reshape the features column
        cls.testFrame=pd.read_csv(testfileName)
        num_cols=cls.testFrame.shape[1]
        print (num_cols)  # print the dimensions
        cls.testFrame['FEATURES'] = cls.testFrame['FEATURES'].str.replace(',','.') #some versions have problems converting string to flost with ,
        
        tempFrame=pd.DataFrame(cls.testFrame['FEATURES'].str.split(' ').tolist()).astype(float) # reshape step into 299 columns
        
        cls.testFrame.drop(['FEATURES'], axis=1, inplace=True)
        num_cols=tempFrame.shape[1] # print dimensions 
        
        print (num_cols)   
        cls.testFrame=pd.concat([tempFrame,cls.testFrame],axis=1)
        cls.testFrame = cls.testFrame.drop(['ROWID', 'ANSWER'],1)
        ############################################################
    def LoadandReshapeData(cls,trainFile,testFile): #should call this 
        
        cls.LoadReshapeTrain(cls,trainFile)
        cls.LoadReshapeTest(cls,testFile)
        
    ########################################################################
    def LoadTrain(cls,fileName): 
      # load traing file 
        cls.trainFrame=pd.read_csv(fileName)
        num_cols=cls.trainFrame.shape[1]
        print (num_cols)  # print the dimensions
        #cls.featureFrame = cls.trainFrame.drop(['ROWID','ANSWER'],1)
        #cls.targetVector = np.array(cls.trainFrame['ANSWER'])
    ###############################################################################
    ########## Preprocessing and features Engineering
    ########################################################################################
    
    def FixNanValues(cls,strategy):
        # Explore how much the data is missing by replacing the mssing values with median
        cls.trainFrame = cls.trainFrame.fillna(0)
        
        
        ##############################
            
    def PercentageMissing(cls, threshold):
   # """this function will return the percentage of missing values in a dataset """
        adict={} #a dictionary conatin keys columns names and values percentage of missin value in the columns
        for col in cls.trainFrame.columns:
            adict[col]=(np.count_nonzero(cls.trainFrame[col].isnull())*100)/len(cls.trainFrame[col])
            print (col, adict[col])
            if adict[col]> threshold: #if the percentage exceeds the given threshold, remove it from both traing and testing
                cls.trainFrame.drop([col], axis=1, inplace=True)
                cls.testFrame.drop([col], axis=1, inplace=True)
         
        num_cols=cls.trainFrame.shape[1]
        print (num_cols)       

    ############################################
    def StandardizeFeatures(cls, option):  
        
        if option=='standard':
            scaler=StandardScaler()
            print(cls.trainFrame.head(1))
            x=cls.trainFrame.loc[:,cls.trainFrame.columns!='ANSWER']
            print(len(x.columns))
            scaler.fit(x)
            print(len(x.columns))
            x=pd.DataFrame(scaler.transform(x))
            print(len(x.columns))
            
#            x.insert(loc = len(x.columns), column = 'ANSWER',value= cls.trainFrame['ANSWER'])
            x['ANSWER'] = list(cls.trainFrame['ANSWER'])
            cls.trainFrame = x
            
            Y = cls.trainFrame.loc[:,cls.trainFrame.columns!='ANSWER']
            Y = pd.DataFrame(scaler.transform(Y))
#            Y['ANSWER'] = list(cls.testFrame['ANSWER'])
            cls.testFrame = Y
            
#            cls.testFrame.loc[:,cls.testFrame.columns!='ANSWER']=pd.DataFrame(scaler.transform(x))
         
            
        else:
            scaler=MinMaxScaler()
            print(cls.trainFrame.head(1))
            x=cls.trainFrame.loc[:,cls.trainFrame.columns!='ANSWER']
            scaler.fit(x)
            cls.trainFrame.loc[:,cls.trainFrame.columns!='ANSWER']=pd.DataFrame(scaler.transform(x))
            cls.testFrame.loc[:,cls.testFrame.columns!='ANSWER']=pd.DataFrame(scaler.transform(x))
            
     ###############################        
    def EncodeCategorical(cls): # this function is additional if the task was updated to contain categerorical variables
        ## there are other ways to do so using sklearn.preprocessing: LabelEncoder and OneHotEncoder
        
        ## convert the neighbourhood column by mapping via dict
        catFeaturesList=[]
        for col_name in cls.featureFrame.columns: #retreive the categorical variables and unique values
            if cls.houseFrame[col_name].dtypes=='object':
                unique_cat=len (cls.featureFrame[col_name].unique())
                catFeaturesList.append(col_name)
                print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name,unique_cat=unique_cat))
                
        print (cls.featureFrame.head())
        
        ## convert the building column by Hot Encoding
        cls.featureFrame=pd.get_dummies(cls.featureFrame[catFeaturesList])
        print (cls.featureFrame.head())
        
     ###########################################################

    
    def HighFrequencyFilterclasses(cls,threshold): #threshold in percentage
    
        Classes = cls.trainFrame['ANSWER'].value_counts().index
        Values = cls.trainFrame['ANSWER'].value_counts().values
       

        Values = Values / len(cls.trainFrame)
        TempDF = pd.DataFrame(data = Classes,columns = ['Columns'] )
        TempDF.insert(1,'perecentage',Values)
        
        median = (np.median(Values))
        meanvalue = (np.mean(Values))
        meanvalue = (meanvalue)*len(cls.trainFrame)
        meanvalue = int(meanvalue)
        BelowMedianandthreshold = median - (median * threshold/100)
        medianvalue = TempDF[TempDF['perecentage'] == median]['perecentage']*len(cls.trainFrame)
        medianvalue = int(medianvalue)
        print(BelowMedianandthreshold)
        TempDF = TempDF[TempDF['perecentage'] >= BelowMedianandthreshold]
        cls.trainFrame = cls.trainFrame[cls.trainFrame['ANSWER'].isin(TempDF['Columns'])]
        for index, row in TempDF.iterrows():
            Classx = row['Columns']
            Per = row['perecentage']
            print(Per)
       
            if(Per > (median + (median * threshold/100))  ):
                
                Samples = cls.trainFrame[cls.trainFrame['ANSWER'] == Classx]
                cls.trainFrame = cls.trainFrame[cls.trainFrame['ANSWER'] != Classx]

                Samples = Samples.head(meanvalue)

                cls.trainFrame = cls.trainFrame.append(Samples)

            else:
                Samples = cls.trainFrame[cls.trainFrame['ANSWER'] == Classx]
                cls.trainFrame = cls.trainFrame[cls.trainFrame['ANSWER'] != Classx]

                Samples = Samples.append([Samples]*2).head(meanvalue)

                cls.trainFrame = cls.trainFrame.append(Samples) 
     ###################################################
     
    def Feature_Selection(cls,no_of_features):
        
        
        cls.X = np.array(cls.trainFrame.drop(['ANSWER'],1))
        cls.y = np.array(cls.trainFrame['ANSWER'])   
        selector = SelectKBest(f_classif, k=no_of_features) # select the optimal k-features these affect the output the most
        #split into train and test
        X_train, X_validation, cls.y_train, cls.y_validation = cross_validation.train_test_split(cls.X, cls.y, test_size = 0.4)
        selector.fit(X_train,cls.y_train)
        cls.Xtrain_selected=selector.transform(X_train)
        cls.Xtest_selected=selector.transform(X_validation)
        cls.Predict_Selected=selector.transform(cls.testFrame)

#############################################################################
     ##################################### Visualization Services
############################################################
             
    def VisualizeClassSamplesData(cls,valueorpercetage = 1): # 1 for value any other for percentage 
        
        Classes = list(map(str,cls.trainFrame['ANSWER'].value_counts().index ))
        Values = cls.trainFrame['ANSWER'].value_counts().values
        x_pos = np.arange(len(Values))
         
        if valueorpercetage==1:
           
            plt.figure(figsize=(15,5))
            plt.bar(x_pos, Values,align='center' , alpha=0.5)
            plt.xticks(x_pos,Classes,rotation=60)
            plt.ylabel('samples numbers')
            plt.xlabel('Classes')
            plt.title('Classes balancing')
            plt.show()
        
        else:
            
            Values = Values / len(cls.trainFrame)
            plt.figure(figsize=(15,5))
            plt.bar(x_pos, Values,align='center' , alpha=0.5)
            plt.xticks(x_pos,Classes,rotation=60)
            plt.ylabel('samples percentage')
            plt.xlabel('Classes')
            plt.title('Classes balancing')
            plt.show() 
 ###########################################################################

    def VisualizeNumberOffeaturesPerEachSample(cls,valueorpercetage = 1): # 1 for value any other for percentage 
            
            MyCount = cls.trainFrame.count(axis = 1)
            MyCount = MyCount-2
            MyCount = MyCount.value_counts()
            Values = MyCount.values
            Classes = MyCount.index
            x_pos = np.arange(len(Values))
            plt.figure(figsize=(50,5))
            plt.bar(x_pos, Values , alpha=0.5)
            plt.xticks(x_pos,Classes,rotation=70)
            plt.ylabel('samples numbers')
            plt.xlabel('Classes')
            plt.title('Number of sample vs number of feature')
            plt.show()            

#######################################################################
            ##############  Prediction Services
#################################################################################
           
    def ApplyKNN(cls):
           
                 
#           cls.clf = MLPClassifier(solver='adam', alpha=0.0001,activation='tanh',
#                    learning_rate_init=0.001,max_iter=500,
#                    hidden_layer_sizes=(75,75),momentum=0.8)
            cls.clf=KNeighborsClassifier(n_neighbors=3)
            X = cls.trainFrame.loc[:, cls.trainFrame.columns != 'ANSWER']
            y = cls.trainFrame['ANSWER']
            X_train, X_validation, y_train, y_validation = cross_validation.train_test_split(X, y, test_size = 0.4)

            cls.clf.fit(X_train, np.asarray(y_train,dtype="|S8"))
            predictions = cls.clf.predict(X_validation)
            print(classification_report(np.asarray(y_validation,dtype="|S8"),np.asarray(predictions,dtype="|S8" )))
           
           
       
           
    def PredictEmpty(cls,testfileName):
        
        cls.originalTestFrame=pd.read_csv(testfileName)
        cls.originalTestFrame['FEATURES'] = cls.originalTestFrame['FEATURES'].str.replace(',','.') #some versions have problems converting string to flost with ,
        #        cls.trainFrame['ANSWER'] = cls.trainFrame['ANSWER'].str.replace(',','.') #some versions have problems converting string to flost with ,
        
        tempFrame=pd.DataFrame(cls.originalTestFrame['FEATURES'].str.split(' ').tolist()).astype(float) # reshape step into 299 columns
            
        cls.originalTestFrame.drop(['FEATURES'], axis=1, inplace=True)
        num_cols=tempFrame.shape[1] # print dimensions 
        
        cls.originalTestFrame=pd.concat([tempFrame,cls.originalTestFrame],axis=1)
        cls.originalTestFrame = cls.originalTestFrame.drop(['ROWID'],1)
        cls.Predict_Selected = cls.originalTestFrame.loc[:,cls.originalTestFrame.columns != 'ANSWER']
        cls.Predict_Selected = cls.Predict_Selected.fillna(0)
        predictions = cls.clf.predict(cls.Predict_Selected)
        print (type(predictions))
        x = list(predictions)
        x= [y.decode("utf-8") for y in x]
        temp = pd.DataFrame(data = np.array(x),columns =['Results'])
        cls.originalTestFrame= pd.concat([cls.originalTestFrame,temp],axis=1)
        cls.originalTestFrame=cls.originalTestFrame.drop(['ANSWER'],1)
        cls.originalTestFrame.to_csv("predict_Updated.csv")
   

                
################################################################
            ############### Main ######################
            
logFile=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'Operations.log'
logging.info('--------------------- Start - Loading The dataset ---------------------')
PredictClass.LoadandReshapeData(PredictClass,"training_set.csv","predict.csv")

logging.info('--------------------- Start - Visuallization before pre-processing ---------------------')
PredictClass.VisualizeClassSamplesData(PredictClass,1)
PredictClass.VisualizeNumberOffeaturesPerEachSample(PredictClass,1)

logging.info('--------------------- Start - pre-processing ---------------------')

PredictClass.FixNanValues(PredictClass,"median")
      
PredictClass.ApplyKNN(PredictClass)

logging.info('--------------------- Start - writing back to file ---------------------')

PredictClass.PredictEmpty(PredictClass,"predict.csv")
gc.collect()

