#!/usr/bin/env python
# coding: utf-8

# # **VIKRAM SAINI**

# # **Stores sales prediction using machine learning algorithms**

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix,precision_score
import seaborn as sns
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


# ### Import Dataset

# In[2]:


df1_train = pd.read_csv("C:/Users/Amit Kumar Saini/Desktop/iNeuron Internship/Train.csv")
df1_train.head(2)


# In[3]:


df1_test = pd.read_csv("C:/Users/Amit Kumar Saini/Desktop/iNeuron Internship/Test.csv")
df1_test.head(2)


# ###### Shape of dataset

# In[4]:


df1_train.shape


# In[5]:


df1_test.shape


# ###### Combine the train and test dataset

# In[6]:


df = df1_train.append(df1_test)
df.head(3)


# In[7]:


df.shape


# In[8]:


df.info()


# ######  Extract numeric and categorical columns from dataset

# In[9]:


numerics = ['int16', 'int32', 'int64', 'float64']
df.select_dtypes(include=numerics)


# In[10]:


# Extract categorical columns from dataset
#categorical = ['object']
#df_train.select_dtypes(include=categorical)

cat_col = []
for x in df.dtypes.index:
  if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col


# In[11]:


df.nunique() #Check unique values 


# ### Data Pre-processing

# ###### Handling Missing Values

# In[12]:


df.isnull().sum() #Checking missing values for train dataset


# In[13]:


#df.isnull().sum().sort_values(ascending=False)

#Checking missing values using heatmap
#sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

missing_percentages = df.isnull().sum().sort_values(ascending=False)/len(df)
print(missing_percentages)
missing_percentages.plot(kind = 'bar')


# In[14]:


missing_percentages[missing_percentages !=0].plot(kind='bar')


# In[15]:


# mean value of "Item_Weight" column
df['Item_Weight'].mean()
# filling the missing values in "Item_weight column" with mean value
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)


# In[16]:


# mode of "Outlet_Size" column
df['Outlet_Size'].mode()
# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(mode_of_Outlet_size)
miss_values = df['Outlet_Size'].isnull()
df.loc[miss_values, 'Outlet_Size'] = df.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])
# checking for missing values
df.isnull().sum()


# In[17]:


df.Item_Fat_Content.value_counts()


# In[18]:


df.replace({'Item_Fat_Content':{'LF':'Low Fat','reg':'Regular','low fat':'Low Fat',}},inplace=True)
df.Item_Fat_Content.value_counts()


# ###  Exploratory Data Analysis(EDA)

# ##### Exploratory Data Analysis (EDA) is an approach to analyze the data using visual techniques. It is used to discover trends, patterns, or to check assumptions with the help of statistical summary and graphical representations.

# ###### Statistical Summary

# In[19]:


df.describe()


# In[20]:


sns.distplot(df["Item_Weight"])


# Here we see that the item weight maximum lie between 5 to 20.

# In[21]:


sns.distplot(df["Item_Visibility"])


# In[22]:


sum(df['Item_Visibility']==0)
df.loc[:,"Item_Visibility"].replace([0],[df['Item_Visibility'].mean()],inplace=True)


# Graph show that item visibility is negatively skewned. 

# In[23]:


sns.distplot(df["Item_MRP"])


# In[24]:


sns.distplot(df.Item_Outlet_Sales)


# ###### 1. Find top five itme type which buy a customer? [Top 5]

# In[25]:


Item_Type_name = df.Item_Type.value_counts().index
Item_Type_Value = df.Item_Type.value_counts().values
plt.pie(Item_Type_Value[:5],labels =Item_Type_name[:5],autopct='%1.2f%%')


# Observation - 
# The Customers buys fruits and vegetables the most.

# In[26]:


# plt.rcParams['figure.figsize'] = (12,5) ,f = list(df.Item_Type.unique())
# x = sns.countplot(df.Item_Type) ,x.set_xticklabels(labels=f,rotation=90)


# ###### 2. Find maximum outet type

# In[27]:


Outlet_Type_name = df.Outlet_Type.value_counts().index
Outlet_Type_value = df.Outlet_Type.value_counts().values
plt.pie(Outlet_Type_value,labels=Outlet_Type_name,autopct='%1.2f%%')


# Maximum outlet are Supermarket type1 and minimum are supermarket type2.

# In[28]:


sns.countplot(df.Outlet_Size)


# Maximum outlet size is medium.

# In[29]:


sns.countplot(df.Outlet_Establishment_Year)


# In year 1985 maximum outlet establish and in 1998 minimum.

# ### Label Encoding

# #### Label Encoding refers to converting the labels into a numeric form so as to convert them into the machine-readable form.

# In[30]:


df.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1,inplace=True)


# In[31]:


label_encoder = LabelEncoder()
df['Item_Fat_Content'] = label_encoder.fit_transform(df['Item_Fat_Content'])
df['Item_Type'] = label_encoder.fit_transform(df['Item_Type'])
df['Outlet_Size'] = label_encoder.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type'] = label_encoder.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type'] = label_encoder.fit_transform(df['Outlet_Type'])


# In[32]:


#df_train=pd.get_dummies(df_train, columns=['Item_Fat_Content','Item_Type','Outlet_Identifier'])


# In[33]:


df.head(3)


# In[34]:


#sns.distplot(df.Item_Weight)


# In[35]:


df_test=df[df['Item_Outlet_Sales'].isnull()]
df_test.head()
f = df_test.dropna(axis=1)
f.head()


# In[36]:


df_train=df[~df['Item_Outlet_Sales'].isnull()]
df_train.head()
df_train.shape


# In[37]:


x = df_train.iloc[:,:-1]
Y = df_train.iloc[:,-1] 


# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=2)


# In[39]:


#st_x= StandardScaler()    
#X_train= st_x.fit_transform(X_train)    
#X_test= st_x.transform(X_test)    


# ### 1. Multiple Linear Regression Model

# In[40]:


#Training the Multiple Linear Regression on training set
reg = LinearRegression()
reg.fit(X_train,Y_train)


# In[41]:


#Predicting the Test set results
Pre_Item_Outlet_Sales_test = reg.predict(X_test)
Pre_Item_Outlet_Sales_test


# In[42]:


Pre_Item_Outlet_Sales_train = reg.predict(X_train)
Pre_Item_Outlet_Sales_train


# In[43]:


# In order to check the performance of the model we find the R squared Value
r2_sales = metrics.r2_score(Y_test,Pre_Item_Outlet_Sales_test)
print('R^2 value = ', r2_sales)


# In[44]:


r2_sales = metrics.r2_score(Y_train,Pre_Item_Outlet_Sales_train)
print('R^2 value = ', r2_sales)


# ### Multiple Linear Regression
# Training Dataset - R_squared_value = 0.505143550207401
# 
#   Test Dataset - R_squared_value = 0.4870428508645147

# ### 2. Random Forest Regression

# In[45]:


# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
 
# fit the regressor with x and y data
regressor.fit(X_train,Y_train) 


# In[46]:


Pre_Item_Outlet_Sales_train = regressor.predict(X_train)
Pre_Item_Outlet_Sales_train


# In[47]:


r2_sales = metrics.r2_score(Y_train,Pre_Item_Outlet_Sales_train)
print('R_squared_value = ', r2_sales)


# In[48]:


Pre_Item_Outlet_Sales_test = regressor.predict(X_test)
Pre_Item_Outlet_Sales_test


# In[49]:


r2_sales = metrics.r2_score(Y_test,Pre_Item_Outlet_Sales_test)
print('R_squared_value = ', r2_sales)


# ### Random Forest Regression
# Training Dataset - R_squared_value = 0.935819613148664
# 
# Test Dataset     - R_squared_value = 0.5354147889262313

# ### Item_Outlet_Sales for given test dataset

# In[50]:


Item_Outlet_Sales = regressor.predict(f)
f['Item_Outlet_Sales'] = Item_Outlet_Sales
f


# ### 3. XGBoost

# In[51]:


xgb = XGBRegressor()
xgb.fit(X_train,Y_train)


# In[52]:


Pre_Item_Outlet_Sales_train = xgb.predict(X_train)
Pre_Item_Outlet_Sales_train


# In[53]:


r2_sales = metrics.r2_score(Y_train,Pre_Item_Outlet_Sales_train)
print('R_squared_value = ', r2_sales)


# In[54]:


r2_sales = metrics.r2_score(Y_test,Pre_Item_Outlet_Sales_test)
print('R_squared_value = ', r2_sales)


# ### XGBoost
# Training Dataset - R_squared_value = 0.847608874947968
# 
# Test Dataset     - R_squared_value = 0.5354147889262313

# **Random Forest Regression gives better result compare to other ML algorithms.** 
