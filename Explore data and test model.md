# 🛒 Case Study - Predictors of mental health illness

<p align="right"> Using Python - Google Colab </p>


## :books: Table of Contents <!-- omit in toc -->

- [🔢 PYTHON - GOOGLE COLAB](#-python---google-colab)
  - [Import Library and dataset](#-import-library-and-dataset)
  - [Explore, clean & transform data](#-import-library-and-dataset)

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
```

```python
#import dataset
df = pd.read_csv('/content/ex1.csv')
```
  
</details>

### 🔎 1️⃣ Explore Data Analysis

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
Age_Group = pd.cut(df['Age'],bins=[17,23,30,61,100],labels=['18-22', '23-30 ','30-60', '> 61'])
df.insert(23,'Age_Group',Age_Group)
df['Age_Group'].unique()                                                 
``` 
![image](https://user-images.githubusercontent.com/101379141/203514958-99f8b983-74e6-468b-9add-8bd849857770.png)     
                                                 
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
 #correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
 ```
![image](https://user-images.githubusercontent.com/101379141/203692179-340350ea-3d7f-4973-9d12-7afb062831b9.png)

```python
#treatment correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()  
```
![image](https://user-images.githubusercontent.com/101379141/203692256-78d617f8-6243-4fe8-8a03-3ea149154f60.png)

  
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
        elif(i ==2.0):labels[i] = '31-60'
        elif(i ==3.0):labels[i] = '> 61'  
    ax.set_xticklabels(labels, rotation=30) # set new labels
plt.show()
 ```
![image](https://user-images.githubusercontent.com/101379141/203710998-cf9ac81f-811e-479b-97c3-912937987f7d.png)
 
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
### 5️⃣ Order Reviews Dataset

- There are 2 things that we are doing with this dataset:
  - The Overall
  - Transform data type from object to datetime 

<details><summary> The  Overall  </summary>

 ```python
 order_reviews.head() 
 ```
![image](https://user-images.githubusercontent.com/101379141/202593250-30d0b6e6-fd98-4413-98ac-93772b75b8d7.png)
  
```python
order_reviews.info() 
```
![image](https://user-images.githubusercontent.com/101379141/202593274-eb0ce20e-5c1e-4b96-8936-ed3a2d43536a.png)
  
```python
order_reviews['review_score'].value_counts()
```
![image](https://user-images.githubusercontent.com/101379141/202593298-38b5ceb6-5e8d-4695-93c6-8d624c479258.png)
 
</details>

<details><summary> Transform data type  </summary>

```python
 order_reviews['review_creation_date'] = pd.to_datetime(order_reviews['review_creation_date'])
order_reviews['review_answer_timestamp'] = pd.to_datetime(order_reviews['review_answer_timestamp'])

order_reviews['review_creation_date'] = order_reviews.review_creation_date.dt.strftime('%m/%d/%Y')
order_reviews['review_answer_timestamp'] = order_reviews.review_answer_timestamp.dt.strftime('%m/%d/%Y')
order_reviews.head(5)
 ```
  
![image](https://user-images.githubusercontent.com/101379141/202593442-736774bf-875a-4ff0-a273-bb31b2958a31.png)
 
</details>

---  
###  6️⃣Products Dataset
  
- There are 3 things that we are doing with this dataset:
  - The Overall
  - Checking Null values .
  - Replacing the "0 gram" of product weight to median

<details><summary> The  Overall  </summary>
  
 ```python
 products.head() 
 ```
![image](https://user-images.githubusercontent.com/101379141/202595562-89179cb5-d1b8-4503-ac9b-908cc286c44a.png)
  
```python
products.info() 
```
![image](https://user-images.githubusercontent.com/101379141/202595592-4a82a95a-9136-48ed-bba7-2fa4bc89777c.png)

  
```python
products.describe()
``` 
![image](https://user-images.githubusercontent.com/101379141/202595632-653f740c-1449-4279-a542-d9d506b269bf.png)

```python
# Min of product_weight_g = 0 , so we check this column to make sure there is nothing anomaly
products[products['product_weight_g']== 0]  
```
  
 ![image](https://user-images.githubusercontent.com/101379141/202595685-8a7e6a1c-c51d-4c21-a6b4-779cce86637b.png)

</details>

<details><summary> Check Null Values </summary>

```python
  #Check Null Values
  products.isnull().sum()
  ```
  ![image](https://user-images.githubusercontent.com/101379141/202596089-660af9b9-c2b1-4f9b-b894-945d6c388aba.png)


```python  
#Check Null values of category name column
products[products['product_category_name'].isnull() == True]
```
![image](https://user-images.githubusercontent.com/101379141/202596188-5f0c384f-8126-4b1e-b4b6-fc80c8d0841b.png)

```python
#Check Null values of weight column
products[products['product_weight_g'].isnull() == True]
```
  
![image](https://user-images.githubusercontent.com/101379141/202596235-c4e5dffb-90cf-4c80-97a0-3d14e83ba554.png)  

 ```python
  #Drop all 610 Null value rows , because they are not significant ( 610  rows compare to 32951 total entries )
  products = products.dropna()  
  products.isnull().sum()  
 ```
 ![image](https://user-images.githubusercontent.com/101379141/202596277-466fbd1b-d48b-4621-87a7-de256a357f78.png)
                                                                                       
</details>  

<details><summary> Check product weight column </summary>

  ```python
  #Check product_weight_g distribution
  sns.distplot(products['product_weight_g'])
  ```
  ![image](https://user-images.githubusercontent.com/101379141/202597280-5893fdcf-addb-40af-8b80-13b6561c8070.png)
  
  ```python
  #Replace "0" values of weight to "median"
  products['product_weight_g']= products['product_weight_g'].replace(0, products['product_weight_g'].median())  
  ```
  
  ```python
  products.describe()
  ```
  ![image](https://user-images.githubusercontent.com/101379141/202597233-2e49fc07-7420-4dad-98a2-39934266b62a.png)
  
</details>  

---  
### 7️⃣ Product Name Translation Dataset
  
- There are 3 things that we are doing with this dataset:
  - Checking The Overall  
  - Merge the product name of 2 table  
  - Checking Null values of merged table and replacing Null values by new category. 

<details><summary> The Overall </summary>

```python
product_name_translation.head()
```
![image](https://user-images.githubusercontent.com/101379141/202599864-11041880-bf87-475b-b51e-2fb433797183.png)

```python
product_name_translation.info()
```
  
![image](https://user-images.githubusercontent.com/101379141/202599948-948b1539-f4af-48cd-b166-b622589b4209.png)
  
</details>  

<details><summary> Merge product name of 2 table </summary>

```python
#Compare the product name of 2 table 
print(product_name_translation['product_category_name'].nunique())
print(products['product_category_name'].nunique()) 
```
![image](https://user-images.githubusercontent.com/101379141/202600071-0df0c1bc-816a-48df-8eef-aa62d1f147b6.png)

```python
product_summarize = products.merge(product_name_translation,how ='left', on = 'product_category_name' )  
```
  
</details>  

<details><summary> Check Null values of merged table and Replace Null values </summary>
  
```python
#Check Null values
product_summarize.isnull().sum()  
```
![image](https://user-images.githubusercontent.com/101379141/202600293-a3e49db7-04e0-4845-8eb0-f3ee59b72501.png)

```python
product_summarize[product_summarize['product_category_name_english'].isnull() == True]  
```
![image](https://user-images.githubusercontent.com/101379141/202600383-93313b22-bed2-4d2c-836b-27bf91d69c18.png)

```python
#Replace Null Value by Unspecified

product_summarize['product_category_name_english'] = product_summarize['product_category_name_english'].fillna(value ='Unspecified')  
product_summarize.isnull().sum()  

```
![image](https://user-images.githubusercontent.com/101379141/202600501-2c762e90-fa24-4e68-a958-ca4564de51c6.png)
    
</details>  

---
### ✔ Save File 

<details><summary> Code here  </summary>

```python
#File customers
customers.to_csv('/content/drive/MyDrive/Final/De 1/customers_dataset.csv',index=False)

#File orders dataset
orders.to_csv('/content/drive/MyDrive/Final/De 1/orders_dataset.csv',index=False)

#File orders items
order_items.to_csv('/content/drive/MyDrive/Final/De 1/order_items_dataset.csv',index=False)

#File order payments
order_payments.to_csv('/content/drive/MyDrive/Final/De 1/order_payments_dataset.csv',index=False)

#File order review
order_reviews.to_csv('/content/drive/MyDrive/Final/De 1/order_reviews_dataset.csv',index=False)

#Merged file of product & produc_translation 
product_summarize.to_csv('/content/drive/MyDrive/Final/De 1/product_summarize_dataset.csv',index=False)

```
</details>  

---
## 📊 POWER BI

### 1. Transform Data

After import dataset, we need to promote header of columns and change some data type columns. 

<details><summary> Customers dataset  </summary>

 - Source (first 10 rows)
  
![image](https://user-images.githubusercontent.com/101379141/202607728-04d35ccc-0db2-49b4-97f8-0d6e2cb0c03c.png)
  
 - Transformed 
  
 ![image](https://user-images.githubusercontent.com/101379141/202607690-acfd75d9-4359-4af6-85b8-c98c78fac434.png)

</details>  

<details><summary> Order Items dataset  </summary>
 
- Source (first 10 rows)
  
 ![image](https://user-images.githubusercontent.com/101379141/202607942-2038f7a4-e235-4a46-ac7b-86e2b673b294.png)
  
- Transformed
  
 ![image](https://user-images.githubusercontent.com/101379141/202608029-b7bc5871-cca9-477f-a03b-773566b168aa.png)
  
</details>  


<details><summary> Order Payments dataset  </summary>

- Source (First 10 rows)
  ![image](https://user-images.githubusercontent.com/101379141/202608207-1e51c2b0-5257-458c-8560-acbe82bdc4ec.png)
  
- Transformed
  ![image](https://user-images.githubusercontent.com/101379141/202608270-29d59313-6861-4c00-a2e1-643fc7f92ccd.png)
</details>  

<details><summary> Order Reviews dataset  </summary>

- Source (First 10 rows)
![image](https://user-images.githubusercontent.com/101379141/202608439-6de93b9f-57e5-4dde-8baf-46037492f1d8.png)

- Transformed
![image](https://user-images.githubusercontent.com/101379141/202608488-a2aa5431-19b6-4203-bf35-3515ab38ebdf.png)

</details>  

<details><summary> Orders dataset  </summary>

- Source (First 10 rows)
 ![image](https://user-images.githubusercontent.com/101379141/202608610-952075c6-cc13-4447-af29-f3a0d6ca5d7d.png)
  
- Transformed
  ![image](https://user-images.githubusercontent.com/101379141/202608652-21c233c4-5298-4060-a50b-043992d4cfdd.png)

</details>  

<details><summary> Product Summarize Dataset  </summary>
  
- Source (First 10 rows)
![image](https://user-images.githubusercontent.com/101379141/202608743-b762ec37-e78f-4db7-ba56-fc6e6d2fd238.png)

- Transformed  
![image](https://user-images.githubusercontent.com/101379141/202608775-130d0dd2-b3ec-4063-9eb1-174b5270b585.png)

</details>  

### 2. Dax, Measure

To support for anlysis chart, We need to create following measure and dax :

<details><summary> 1%star - to filter 1 star review  </summary>

```
%1star = divide(calculate(count(order_items_dataset[English_name_product]),order_items_dataset[Average_score] = 1),count(order_items_dataset[English_name_product]))
  
```  
</details>  

  
<details><summary> 5%star - to filter 5 star review  </summary>

```
%5star = divide(calculate(count(order_items_dataset[English_name_product]),order_items_dataset[Average_score] = 5),count(order_items_dataset[English_name_product]))
```
</details>  


<details><summary> %Comment - to calculate % order has comment   </summary>

```
%Comment = Divide(CALCULATE(count(order_reviews_dataset[Comment]), order_reviews_dataset[Comment] = "Comment"),count(order_reviews_dataset[order_id]))
```
</details> 

<details><summary> Average_Score - Average score of orders   </summary>

```
Average_Score = SUM(order_items_dataset[Average_score])/count(order_items_dataset[order_id])
```
</details>

<details><summary> Comment - Count of orders has comment   </summary>

```
Comment = CALCULATE(count(order_reviews_dataset[Comment]),order_reviews_dataset[Comment] = "Comment")
```
</details>

<details><summary> Comment_Star - Calculate review score of orders having comment   </summary>

```
Comment_Star = calculate(count(order_reviews_dataset[review_score]),order_reviews_dataset[Comment] = "Comment")
```
</details>

<details><summary> Total_time_to_delivery average per customer_city   </summary>

```
Total_time_to_delivery average per customer_city = DIVIDE(sum(orders_dataset[Total_time_to_delivery]),count(orders_dataset[order_id]))
```
</details>

<details><summary> Voucher_cat - calculate orders has applied voucher  </summary>

```
Voucher_cat = Divide(CALCULATE(count(order_payments_dataset[payment_type]),order_payments_dataset[payment_type] = "voucher"),count(order_items_dataset[product_id]))
```
</details>

<details><summary> Count Product  </summary>

```
Count_Product = COUNT(order_items_dataset[English_name_product])
```
  
</details>

<details><summary> Rank Product  </summary>

```
Rank_Product = RANKX(all(order_items_dataset[English_name_product]),[Count_Product])
```
  
</details>

### 3. Create New Table

To match the average score of order. I have to create new table 

```
Average = SUMMARIZECOLUMNS(order_reviews_dataset[order_id],"Average_Score",AVERAGE(order_reviews_dataset[review_score]))
```
<details><summary> The First Few Rows  </summary>
 
![image](https://user-images.githubusercontent.com/101379141/202612783-d8974939-f0b0-43e3-a655-f003e98c0758.png)
  
</details>

**Final Model**

<details><summary> Click Here  </summary>

![image](https://user-images.githubusercontent.com/101379141/202614575-3ffb8db6-9e53-42af-8a08-99f5423c4a5e.png)

</details>
