# 🛒 Case Study - Predictors of mental health illness

<p align="right"> Using Python - Google Colab </p>


## :books: Table of Contents <!-- omit in toc -->

- [🔢 PYTHON - GOOGLE COLAB](#-python---google-colab)
  - [Import Library and dataset](#-import-library-and-dataset)
  - [Explore data ](#1%EF%B8%8F⃣-explore-data-analysis)
  - [Preprocessing - Encoding](#2%EF%B8%8F⃣--preprocessing---encoding)
  - [Covariance Matrix](#3%EF%B8%8F⃣-covariance-matrix)
  - [Relationship Charts](#4%EF%B8%8F⃣-some-charts-to-see-data-relationship)
  - [Scaling & Fitting](#5%EF%B8%8F⃣-scaling-and-fitting)
  - [Tuning](#6%EF%B8%8F⃣tuning)
  - [Evaluate Model](#7%EF%B8%8F⃣-evaluate-models)
  - [Success method plt](#8%EF%B8%8F⃣-success-method-plot)
  - [Creating predictions on test set](#9%EF%B8%8F⃣-creating-predictions-on-test-set)
  - [Saving model](#-saving-model)

---

## 👩🏼‍💻 PYTHON - GOOGLE COLAB

### 🔤 IMPORT LIBRARY AND DATASET 

<details><summary> Click to expand code </summary>
  
```python
#Import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score


#ensemble
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.impute import SimpleImputer

#Library label encoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
```

```python
#import dataset
df = pd.read_csv('/content/ex1.csv')
```
  
</details>

---
### 1️⃣ Explore Data Analysis

- There are 3 things that i would to do in this step:
  - The overall info 
  - Cleaning missing values
  - Checking values of all columns

<details><summary> 1.1 The  Overall Infomation </summary>
  
```python
df.head() 
```
![image](https://user-images.githubusercontent.com/101379141/203503490-5e514c69-a860-473a-8757-cd83a3633716.png)
  
```python
df.tail()
```
![image](https://user-images.githubusercontent.com/101379141/203503535-a3fc7b50-444a-4506-a7c5-8984730d99d2.png)
    
```python
df.info()
```  
![image](https://user-images.githubusercontent.com/101379141/203503625-bfb615ca-a92a-4448-933c-205182de4e92.png)
  
```python
df.describe()
```    
![image](https://user-images.githubusercontent.com/101379141/203503686-fe20ffc2-6892-4341-9040-3fff5d5b5a85.png)

</details>

<details><summary> 1.2. Clean missing values </summary>  
  
<br> We would check and clean the null values of all columns, beside that we also drop some unnecessary columns.
  
<details><summary> 1.2.a Check Null values </summary>

 ```python
df.isnull().sum()
 ```
![image](https://user-images.githubusercontent.com/101379141/203505779-681fc8b1-c367-4e7a-aa67-2773c0e35c14.png)

```python
#% Null values
dict_null = dict()
for i in df.columns:
  dict_null[i] = df[i].isnull().sum()/len(df['Timestamp'])*100
df1 = pd.DataFrame.from_dict(dict_null.items())
print(df1)
```
![image](https://user-images.githubusercontent.com/101379141/203506087-1709522f-ec27-4784-a498-6b36f1365956.png)

   
```python
df.drop(columns = ['Timestamp','state','Country','comments'], inplace = True)
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/101379141/203506299-8d4aef53-5e1f-49fd-8940-03d0c286e987.png)

</details>
 
<details><summary>  1.2.b Clean missing values of self_employed column  </summary>

 ``` python
df['self_employed'].unique() 
```
![image](https://user-images.githubusercontent.com/101379141/203506826-e7248295-e214-4fd2-bd75-c2391eb6f833.png)
  
  
```python
df['self_employed'].value_counts()
```
![image](https://user-images.githubusercontent.com/101379141/203506911-41280ea0-f49e-4196-b4bd-9497361deed7.png)

```python
# Replace Null values by the mode 
df['self_employed'].replace(np.NaN,'No',inplace=True)
df['self_employed'].unique()
```
![image](https://user-images.githubusercontent.com/101379141/203507148-ad53076c-7f10-4801-a248-d94f90f09baa.png)

 </details> 

<details><summary> 1.2.c Clean missing values of work_interfere column </summary>

```python
df['work_interfere'].unique()
```
![image](https://user-images.githubusercontent.com/101379141/203507974-d8980080-f83a-451d-b1bc-ecd729da0aa6.png)

```python
df['work_interfere'].value_counts()
```
![image](https://user-images.githubusercontent.com/101379141/203508032-bac8d92a-268a-4841-8cf6-d24f17911047.png)
  
```python
# Replace Null values
df['work_interfere'].replace(np.NaN, "Don't Know",inplace = True)
df['work_interfere'].value_counts()
```
![image](https://user-images.githubusercontent.com/101379141/203508172-adf418ec-db39-473b-bbe8-8fd0ffc85abf.png)

</details> 

<details><summary> Dataset with 0 Null values </summary>

```python
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/101379141/203508526-5e04e1b0-ae0a-4dfa-9717-c0dc7fa2a644.png)

</details> 
  
</details> 

<details><summary> 1.3. Checking values of all columns </summary>  

<br> After check values of all columns, we can see that there are some outliers in Gender and Age column 

<details><summary> Code here </summary> 
  
```python
my_list = df.columns.values.tolist()

for column in my_list:
  print(column)
  print(df[column].unique())  
```
![image](https://user-images.githubusercontent.com/101379141/203513372-7c48e84f-c537-478a-ab5c-09abb088f4b5.png)
![image](https://user-images.githubusercontent.com/101379141/203513431-d8c289e9-7e02-4aad-b761-bb13d1f93d98.png)

</details> 

<details><summary> 1.3.a Age Column </summary>  

```python
from matplotlib.pyplot import figure

figure(figsize=(10, 10))
df['Age'].value_counts().plot( kind= 'bar')  
```
![image](https://user-images.githubusercontent.com/101379141/203514344-2a02fc03-4f88-46a1-be28-ddd5d1fa556e.png)

```python
outliers =[]
for age in df['Age'].values:
  if age < 0 or age >100 :
    outliers.append(age)
    print(outliers)   
```
![image](https://user-images.githubusercontent.com/101379141/203514466-7edf6a18-6b0a-4bac-887d-33fd9c2908da.png)

```python
#Because There is only 5 outliers comparing total 1259 entries, so we can remove values of outliers

df = df.loc[(df['Age'] > 18) & (df['Age'] <100)]
                                                 
# 0 values means no outliers 
print(df[df["Age"].isin(outliers)] )
                                                
```
![image](https://user-images.githubusercontent.com/101379141/203514808-8a94c840-5fe3-46c7-b6a0-489d50ccaeb3.png)

```python
#Grouping Age
Age_Group = pd.cut(df['Age'],bins=[17,23,30,61,100],labels=['18-22', '23-30 ','31-50', '> 51'])
df.insert(23,'Age_Group',Age_Group)
df['Age_Group'].unique()                                                 
``` 
![image](https://user-images.githubusercontent.com/101379141/203514958-99f8b983-74e6-468b-9add-8bd849857770.png)     

```python
# Drop Age column, because we create Age grouped                                                 
df = df.drop(columns='Age')                                                 
```                                                
</details> 
  
<details><summary> 1.3.b Gender Column </summary>  

```python
df1= df['Gender'].unique()
print(df1)
```
![image](https://user-images.githubusercontent.com/101379141/203515507-eec125bc-adc6-44a8-8255-913128d85441.png)
  
```python
male_string = ["M", "Male", "male", "m", "Male-ish", "maile", "Cis Male", "Mal", "Male (CIS)","Make", "Male ", "Man","msle", "Mail", "cis male","Malr","Cis Man"]
female_string = ["Female", "female", "Cis Female", "F","Woman",  "f", "Femake","woman", "Female ", "cis-female/femme","Female (cis)","femail"]
others_string = ["Trans-female", "something kinda male?", "queer/she/they", "non-binary","Nah", "all", "Enby", "fluid", "Genderqueer", "Androgyne", "Agender", "male leaning androgynous", "Guy (-ish) ^_^", "Trans woman", "Neuter", "Female (trans)", "queer", "ostensibly male, unsure what that really means"]           

for index, row in df.iterrows():

    if str(row.Gender) in male_string:
        df['Gender'].replace(to_replace=row.Gender, value='male', inplace=True)

    if str(row.Gender) in female_string:
        df['Gender'].replace(to_replace=row.Gender, value='female', inplace=True)

    if str(row.Gender) in others_string:
        df['Gender'].replace(to_replace=row.Gender, value='other', inplace=True)


print(df['Gender'].unique())
```
![image](https://user-images.githubusercontent.com/101379141/203515581-7ec6c102-e6e8-413e-95eb-f5cd50487d08.png)
  
</details> 
</details> 
</details> 
</details> 

---
  
### 2️⃣  Preprocessing - Encoding

- There are 2 things I would do in this step:
  - We encode (convert) the feature columns excluding Age column into number for model analyst.
    - We dont encode the Age column because We grouped Age column into Age_group column and we encoded the Age_Group.
  - Create label feature for up-comming steps

<details><summary> Label-Enconding  </summary>
  
```python
label_dict = {}
#Label-Enconding
le = preprocessing.LabelEncoder()
for feature in df.columns:
  if feature != 'Age':
    le.fit(df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df[feature] = le.transform(df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    label_dict[labelKey] =labelValue
  else:
    label_dict['label_Age'] = list(df['Age'])

```
```python
df.info()
df.head() 
```
![image](https://user-images.githubusercontent.com/101379141/203689607-cac4134c-d4c6-4d42-809a-834013789ee5.png)
  
```python
for key, value in label_dict.items():     
    print(key, value)
```
![image](https://user-images.githubusercontent.com/101379141/203689659-b26ccd3c-3538-4125-8af9-d6b62cba9e5e.png)
  
</details>

---
### 3️⃣ Covariance Matrix.

- Variability comparison between categories of variables 

--> The final result showed that The strongest relationship between target variable (treatment) and feature variable is work_interfere columns with the question (If you have a mental health condition, do you feel that it interferes with your work?) and family_history (Do you have a family history of mental illness?)

--> It is clear that if we work or live in an environment nearly people who has mental illness or full of job stress, we could be affected. 

<details><summary> The  Code Here  </summary>



```python
#treatment correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
corrmat = df.corr()
k = 23 #number of variables for heatmap
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```
![image](https://user-images.githubusercontent.com/101379141/204680141-6c193cda-3fb5-452f-a057-a2014c3bcbf0.png)

  
</details>
 
---
### 4️⃣ Some charts to see data relationship


<details><summary> Age Group & Treatment  </summary>

<br>
  
--> The possibility of being mental illness is increasing by age.
 ```python
# Age & Treatment

g = sns.FacetGrid(df, col ='treatment', height=8)
g = g.map(sns.countplot, "Age_Group")

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i == 0): labels[i] = '18-22'
        elif(i ==1.0):labels[i] = '23-30'
        elif(i ==2.0):labels[i] = '31-50'
        elif(i ==3.0):labels[i] = '> 51'  
    ax.set_xticklabels(labels, rotation=30) # set new labels
plt.show()
 ```
![image](https://user-images.githubusercontent.com/101379141/204680210-9444de57-07e6-4fdf-81de-0daeb2af2991.png)
  
</details>

<details><summary> Gender & Treatment  </summary> 
<br>
  --> Male has higher possibility of being mental illness comparing to Female.
    
```python
#Gender & Treatment
df1 = df
df1['Gender'] = df1['Gender'].astype('category')
print(df1['Gender'].unique())
plt.figure(figsize=(12,8))
g = sns.FacetGrid(df1, col='treatment', height=8)
g.map(sns.countplot,'Gender')

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i == 0): labels[i] = 'Female'
        elif(i ==1):labels[i] = 'Male'
        else: labels[i] ='Other'  
    ax.set_xticklabels(labels, rotation=30) # set new labels
plt.show()
  
```
![image](https://user-images.githubusercontent.com/101379141/203714266-11193591-f268-4de4-b503-df74f5d67181.png)
  
</details>
 
<details><summary> Percentage treatment for family_history by Gender  </summary> 
<br>

--> If your family members has experience the mental illness, people has high possibility of being mental illness too
  
```python
#Draw a catplot to show Percentage treatment for family_history by Gender

g = sns.catplot(x="family_history", y="treatment", hue="Gender", data=df, kind="bar",  ci=None, size=5, aspect=2, legend_out = True)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i == 0): labels[i] = 'No'
        else: labels[i] ='Yes'
    ax.set_xticklabels(labels, rotation=30) # set new labels

# title
g._legend.set_title('Gender')
new_labels = ['Female', 'Male', 'Other']
# replace labels
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

plt.title('Probability of health condition by family_history and Gender')
plt.ylabel('Probability x 100')
plt.xlabel('Family History')  
```
![image](https://user-images.githubusercontent.com/101379141/203715984-c3fa3385-2c6d-4b97-b5d5-52e845c71f83.png)
   
</details>

<details><summary> Percentage treatment for Work_interfere by Gender  </summary> 
<br>

--> we can see that , the mental illness has negative effect to the workplace where always create the high intensity of stress.
  
```python
#Draw a catplot to show Percentage treatment for Work_interfere by Gender

g = sns.catplot(x="work_interfere", y="treatment", hue="Gender", data=df, kind="bar",  ci=None, size=5, aspect=2, legend_out = True)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i == 0): labels[i] = "Don't Know" 
        elif(i ==1):labels[i] = 'Never'
        elif(i ==2):labels[i] = 'Often'
        elif(i ==3):labels[i] = 'Rarely'
        else: labels[i] = 'Sometimes'
    ax.set_xticklabels(labels, rotation=30) # set new labels

# title
g._legend.set_title('Gender')
new_labels = ['Female', 'Male', 'Other']
# replace labels
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

g.fig.subplots_adjust(top=1,right=0.8)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('work_interfere')
```
![image](https://user-images.githubusercontent.com/101379141/203717144-5b5fc232-6610-4744-8417-ceea7ee1c333.png)
  
</details>

<details><summary> Percentage treatment for Care Benefit by Gender  </summary> 
<br>

--> We can't see the relationship between Care Option and Treatment clearly. 
  
```python
#Draw a catplot to show Percentage treatment for Care Benefit by Gender

g = sns.catplot(x="benefits", y="treatment", hue="Gender", data=df, kind="bar",  ci=None, size=5, aspect=2, legend_out = True)

for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i == 0): labels[i] = "Don't Know" 
        elif(i ==1):labels[i] = "No"
        else: labels[i] = "Yes"
    ax.set_xticklabels(labels, rotation=30) # set new labels

# title
g._legend.set_title('Gender')
new_labels = ['Female', 'Male', 'Other']
# replace labels
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

g.fig.subplots_adjust(top=1,right=0.8)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Care Options')
```
![image](https://user-images.githubusercontent.com/101379141/203719464-08846bf2-4c5b-4eb5-95bc-64631eb67f5c.png)

</details>

---
### 5️⃣ Splitting Data

<details><summary> Splitting Dataset  </summary> 
<br>
 
```python
y = df['treatment']
X = df.drop(columns='treatment')


# split dataset to test and training set (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
  
```
</details>
  
---  
###  6️⃣Tuning

<br>
Firstly, I would write a function to evaluate the models (Confusion matrix & accuracy_score) and also applied it to Tunning Function too. 
</br>

<br>
<details><summary> Writing Evaluate Model Function  </summary>
  
 ```python
  
 methodDict = {} # This would be used for plotting the model's performance


# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve,classification_report
from sklearn.model_selection import cross_val_score

def EvaluateModel(model, y_test, y_pred, plot=False):
    
    #Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_true =y_test, y_pred = y_pred)
  

    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    #Metrics computed from a confusion matrix
    #Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Classification Accuracy:', accuracy)
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred))
    
    #Classification Report
    print('Classification Accuracy:' ,classification_report(y_test,y_pred))
    
  
    
    methodDict[model.__class__.__name__] = metrics.accuracy_score(y_test, y_pred) * 100
 
 ```

</details>

<details><summary> Tunning Function </summary>
<br>

  - Because dataset is small, I still would like to use Random Search instead of Bayes, or gridsearch because I want to minimize the tuning time and better result,. In this case : I use RandomizedSearchCV .
  
  --> Reference to https://towardsdatascience.com/gridsearch-vs-randomizedsearch-vs-bayesiansearch-cfa76de27c6b 

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 2)

def RandomSearch(model, param_dist):
  reg_bay = RandomizedSearchCV(estimator=model,
                    param_distributions=param_dist,
                    n_iter=20,  # search 20 times 
                    cv=kf,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state =3)
  reg_bay.fit(X_train,y_train)
  y_pred = reg_bay.predict(X_test)
  print('RandomSearch. Best Score: ', reg_bay.best_score_)
  print('RandomSearch. Best Params: ', reg_bay.best_params_)
  accuracy_score = EvaluateModel(model, y_test, y_pred, plot =True)

  ```
                                                                                      
</details>  


---  
### 7️⃣ Evaluate Models
  


<details><summary> Logistic Regression </summary>

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
    
# make class predictions for the testing set
y_pred = logreg.predict(X_test)
    
print('########### Logistic Regression ###############')
    
accuracy_score = EvaluateModel(logreg, y_test, y_pred, plot =True)
      
```
![image](https://user-images.githubusercontent.com/101379141/204680565-8463ef0e-b56a-4482-9376-f42e169ff415.png)
  
</details>  

<details><summary> K-neighbors </summary>

```python
model = KNeighborsClassifier()

param_dist = {'n_neighbors': list(range(1,31)),
              'weights' :['uniform', 'distance']}

RandomSearch(model, param_dist)
  
```
  
![image](https://user-images.githubusercontent.com/101379141/203885509-1278e4e7-2c32-4711-abc6-a5409350dd23.png)
    
</details>  

<details><summary> Decision-Tree </summary>

```python
model_2 = DecisionTreeClassifier()
param_dist = {'max_depth': list(range(3, 9)),
              "max_features": list(range(1, len(X.columns))),
              "min_samples_split": list(range(2, 9)),
              "min_samples_leaf": list(range(1, 9)),
              "criterion": ["gini", "entropy"]} 

RandomSearch(model_2, param_dist)
  
```
![image](https://user-images.githubusercontent.com/101379141/203885667-8f6fa33c-eb11-45e9-ab9e-9af9f4be8bb9.png)

```python
import graphviz
model_2.fit(X_train,y_train)
dot_data  = tree.export_graphviz(model_2,max_depth = 8,filled = True)

graph = graphviz.Source(dot_data, format="png") 
graph.render("decision_tree_graphivz")
  
```
</details>  

<details><summary> RandomForest </summary>

```python
model_3 = RandomForestClassifier()
estimators = [int(x) for x in np.linspace(start = 1, stop = 100, num = 10)]
param_dist = {'n_estimators' : estimators,
             'max_depth': list(range(3, 9)),
              "max_features": list(range(1, len(X.columns))),
              "min_samples_split": list(range(2, 9)),
              "min_samples_leaf": list(range(1, 9)),
              "criterion": ["gini", "entropy"]} 
RandomSearch(model_3, param_dist)
  
```
![image](https://user-images.githubusercontent.com/101379141/203885729-43fcccce-7916-42ef-b942-dccacee8acc4.png)
    
</details>  

<details><summary> AdaBoosting </summary>

```python
tree = DecisionTreeClassifier(max_depth = 1)
model = AdaBoostClassifier(base_estimator= tree, n_estimators= 100,random_state = 4)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

EvaluateModel(model, y_test, y_pred, True)  
```
![image](https://user-images.githubusercontent.com/101379141/203885879-e7862df9-bf75-44aa-bf3c-2777f796e5bf.png)
    
</details>  

<details><summary> Gradient Boosting </summary>

```python
model = GradientBoostingClassifier(n_estimators =100, max_depth =1,random_state = 5 )
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

EvaluateModel(model, y_test, y_pred, True)  
```
![image](https://user-images.githubusercontent.com/101379141/203885959-f7d804e2-590d-4250-b646-0945ed671a8f.png)

    
</details>  

<details><summary> Bagging </summary>

```python
tree = DecisionTreeClassifier()

model_4 = BaggingClassifier(base_estimator = tree, bootstrap_features=False, n_estimators = 100,random_state = 2)
param_dist = {'base_estimator__max_depth' : [1,2,3]}


print('########### Bagging ###############')

RandomSearch(model_4, param_dist)
 
  
```
![image](https://user-images.githubusercontent.com/101379141/204692826-07b9c034-ea0b-495e-b9e1-9e60144918f2.png)
  
</details>  

---

### 8️⃣ Success method plot

<br>
We would like to show the summary of models's performance to compare and select the best one.
</br>
<br>

<details><summary> Code here </summary>

```python
s = pd.Series(methodDict)
s = s.sort_values(ascending=False)
plt.figure(figsize=(12,8))

ax = s.plot(kind='bar') 
for p in ax.patches:
  ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.ylim([70.0, 90.0]) 
plt.xticks(rotation = 45)
plt.xlabel('Method')
plt.ylabel('Percentage')
plt.title('Success of methods')
     
plt.show()
```
![image](https://user-images.githubusercontent.com/101379141/204692652-f5feafd6-d6ac-41d0-b498-bae51a07acd6.png)
  
</details>  

---
### 9️⃣ Creating predictions on test set

Because the result showed that Decision Tree, RandomForest,Bagging have the same result. So we can use the decision Tree with best parameters for saving time but still get the best result

<details><summary> Code here </summary>

```python
#Because the result showed that Decision Tree, RandomForest,Bagging have the same result. So we can use the decision Tree with best parameters for saving time but still get the best result
model = DecisionTreeClassifier(min_samples_split= 7, min_samples_leaf= 7, max_features= 17, max_depth = 2, criterion = 'gini')

model.fit(X_train, y_train)
dfTestPredictions = model.predict(X_test)

# Write predictions to csv file
results = pd.DataFrame({'Index': X_test.index, 'predict_Treatment': dfTestPredictions,'test_treatment': y_test})
# Save to file
# This file will be visible after publishing in the output section
results.to_csv('results.csv', index=False)
print(results)
EvaluateModel(model, y_test, y_pred, True)
```
![image](https://user-images.githubusercontent.com/101379141/204692504-44f67ca0-103e-4cb7-8fd2-4a75d7fcf424.png)
  
</details>  

---
### 🔟 Saving model

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
modeltosave = model

import _pickle as cPickle
import os
with open('/content/drive/MyDrive/Mindx_final/DucDat_De1_model.pkl', 'wb') as f:
    cPickle.dump(modeltosave, f)

print('Saved Model')
```
