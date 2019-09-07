---
title: Learn With Sarfaraz Kaggle Competition
excerpt: In this competition you’ll predict what types of trees there are in an area based on various geographic features.

The competition datasets comes from a study conducted in four wilderness areas within the beautiful Roosevelt National Forest of northern Colorado. These areas represent forests with very little human disturbances – the existing forest cover types there are more a result of ecological processes rather than forest management practices.

The data is in raw form and contains categorical data such as wilderness areas and soil type.
---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

# Learn With Other Kaggle Users
### Classify forest types based on information about the area
![full](https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52)

This is a beginner friendly Kaggle Competition so we will learn Machine Learning together in this competition using the Forestry Data Set


## Welcome to our first, invite-only, beginner-friendly competition.
Some people associate Kaggle with ultra-competitive machine learning masterminds. But we're also a community of people learning together. In this competition we’re skipping the medals, points, and prizes so people who are just getting started can share ideas, experiment with different techniques and learn from each other.

## The challenge:
In this competition you’ll predict what types of trees there are in an area based on various geographic features.

The competition datasets comes from a study conducted in four wilderness areas within the beautiful Roosevelt National Forest of northern Colorado. These areas represent forests with very little human disturbances – the existing forest cover types there are more a result of ecological processes rather than forest management practices.

The data is in raw form and contains categorical data such as wilderness areas and soil type.

## Evaluation
Submissions are evaluated on categorization accuracy.

That's just what fraction of predictions did you get right. You will want to use Classifier models like RandomForestClassifier rather than Regression models. With classifier models, the predict method will tell you which category is most likely.

Submission File
For each ID in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:
```
ID,TARGET
2,0
5,3
6,2
etc.
```


## Data Description
The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

1. - Spruce/Fir
2. - Lodgepole Pine
3. - Ponderosa Pine
4. - Cottonwood/Willow
5. - Aspen
6. - Douglas-fir
7. - Krummholz

The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

## Data Fields
* Elevation - Elevation in meters
* Aspect - Aspect in degrees azimuth
* Slope - Slope in degrees
* Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
* Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
* Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
* Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
* Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
* Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
* Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
* Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
* Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

The wilderness areas are:

1. - Rawah Wilderness Area
2. - Neota Wilderness Area
3. - Comanche Peak Wilderness Area
4. - Cache la Poudre Wilderness Area

The soil types are:

1. Cathedral family - Rock outcrop complex, extremely stony.
2. Vanet - Ratake families complex, very stony.
3. Haploborolis - Rock outcrop complex, rubbly.
4. Ratake family - Rock outcrop complex, rubbly.
5. Vanet family - Rock outcrop complex complex, rubbly.
6. Vanet - Wetmore families - Rock outcrop complex, stony.
7. Gothic family.
8. Supervisor - Limber families complex.
9. Troutville family, very stony.
10. Bullwark - Catamount families - Rock outcrop complex, rubbly.
11. Bullwark - Catamount families - Rock land complex, rubbly.
12. Legault family - Rock land complex, stony.
13. Catamount family - Rock land - Bullwark family complex, rubbly.
14. Pachic Argiborolis - Aquolis complex.
15. unspecified in the USFS Soil and ELU Survey.
16. Cryaquolis - Cryoborolis complex.
17. Gateview family - Cryaquolis complex.
18. Rogert family, very stony.
19. Typic Cryaquolis - Borohemists complex.
20. Typic Cryaquepts - Typic Cryaquolls complex.
21. Typic Cryaquolls - Leighcan family, till substratum complex.
22. Leighcan family, till substratum, extremely bouldery.
23. Leighcan family, till substratum - Typic Cryaquolls complex.
24. Leighcan family, extremely stony.
25. Leighcan family, warm, extremely stony.
26. Granile - Catamount families complex, very stony.
27. Leighcan family, warm - Rock outcrop complex, extremely stony.
28. Leighcan family - Rock outcrop complex, extremely stony.
29. Como - Legault families complex, extremely stony.
30. Como family - Rock land - Legault family complex, extremely stony.
31. Leighcan - Catamount families complex, extremely stony.
32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33. Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34. Cryorthents - Rock land complex, extremely stony.
35. Cryumbrepts - Rock outcrop - Cryaquepts complex.
36. Bross family - Rock land - Cryumbrepts complex, extremely stony.
37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38. Leighcan - Moran families - Cryaquolls complex, extremely stony.
39. Moran family - Cryorthents - Leighcan family complex, extremely stony.
40. Moran family - Cryorthents - Rock land complex, extremely stony.

## Data Analysis and Visualization
Ok, so before diving into the modeling part we will do some Data Analysis and Data Visualization

There is a great Notebook by [Fatih Bilgin](https://www.kaggle.com/fatihbilgin) :
https://www.kaggle.com/fatihbilgin/quick-visualization-and-eda-for-beginners which performs this well


```python
import pandas as pd

PATH = "../input/learn-together"

train = pd.read_csv(f"{PATH}/train.csv", index_col = "Id")
test = pd.read_csv(f"{PATH}/test.csv", index_col = "Id")
```


```python
train.head()
```


```python
print(f"Training Data Set Shape: {train.shape} \nTest Data Set Shape: {test.shape}")
```


```python
train.info()
```


```python
train.describe().T
```

All Wilderness_Area and Soil_Type columns have values in the range of 0 and 1. Quite likely these columns are categorical and consist of 0 and 1. To validate this i'm checking distinct values of following columns:


```python
print(train.iloc[:, 10:-1].columns)
```


```python
pd.unique(train.iloc[:,10:-1].values.ravel())
```

Yes all wilderness area and soil type columns consist of 0 and 1. In other words they are categorical. So i'm convering these columns to categorical ones.


```python
train.iloc[:,10:-1] = train.iloc[:,10:-1].astype("category")
test.iloc[:,10:] = test.iloc[:,10:].astype("category")
```

I'm trying to find out correlation between columns with heatmap in this step.


```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')

f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(train.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
```

It seems the most important correlations are between "Horizontal Distance To Hydrology" and "Vertical Distance To Hydrology" with 70%; between "Aspect" and "Hillshade 3pm" with 60%; between "Hillshade Noon" and "Hillshade 3pm" with %60; between "Elevation" and "Horizontal Distance To Roadways" with %60. Let's see how they are looking.


```python
train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='yellow', figsize = (12,9))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()
```


```python
train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', alpha=0.5, color='maroon', figsize = (12,9))
plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")
plt.show()
```


```python
train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', alpha=0.5, color='purple', figsize = (12,9))
plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")
plt.show()
```

There are obvious patterns if we ignore to outliers. And with this patterns, our model will learn.

Boxplot can be used to see outliers. For a better visualization i will use plotly this time.

Wow, this is very interesting plot! I need to learn this too!


```python
import plotly.graph_objs as go
from plotly.offline import iplot

trace1 = go.Box(
    y= train["Vertical_Distance_To_Hydrology"],
    name = 'Vertical Distance',
    marker = dict(color = 'rgb(0,145,119)')
)
trace2 = go.Box(
    y= train["Horizontal_Distance_To_Hydrology"],
    name = 'Horizontal Distance',
    marker = dict(color = 'rgb(5, 79, 174)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Distance To Hydrology', paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))
fig = dict(data=data, layout=layout)
iplot(fig)
```


```python
trace1 = go.Box(
    y= train["Hillshade_Noon"],
    name = 'Hillshade Noon',
    marker = dict(color = 'rgb(255,111,145)')
)
trace2 = go.Box(
    y= train["Hillshade_3pm"],
    name = 'Hillshade 3pm',
    marker = dict(color = 'rgb(132,94,194)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Hillshade 3pm and Noon', paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))
fig = dict(data=data, layout=layout)
iplot(fig)
```

This time I'll compare vertical and horizontal distance to hydrology with histogram.


```python
f,ax=plt.subplots(1,2,figsize=(15,7))
train.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='crimson')
ax[0].set_title('Vertical Distance To Hydrology')
x1=list(range(-150,350,50))
ax[0].set_xticks(x1)
train.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='darkmagenta')
ax[1].set_title('Horizontal Distance To Hydrology')
x2=list(range(0,1000,100))
ax[1].set_xticks(x2)
plt.show()
```

Let's take a look our categorical categorical variables soil types and wilderness areas.


```python
soil_types = train.iloc[:,14:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, palette="rocket")
plt.xticks(rotation= 75)
plt.ylabel('Total')
plt.title('Count of Soil Types With Value 1',color = 'darkred',fontsize=12)
```

Type 7, Type 8, Type 15 and Type 25 have either no or too few values. Must examine carefully before create a model.


```python
wilderness_areas = train.iloc[:,10:14].sum(axis=0)

plt.figure(figsize=(7,5))
sns.barplot(x=wilderness_areas.index,y=wilderness_areas.values, palette="Blues_d")
plt.xticks(rotation=90)
plt.title('Wilderness Areas',color = 'darkred',fontsize=12)
plt.ylabel('Total')
plt.show()
```

I wonder how many (y) labels we have in each class. I'll take a look the last column (cover type) for this.


```python
import plotly.express as px

cover_type = train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', height=400, width=650)
fig.show()
```

There are same amount of data for each class exactly...

In terms of horizontal distance to x point, distribution of class charts following...


```python
f,ax=plt.subplots(1,3,figsize=(21,7))
train.plot.scatter(ax=ax[0],x='Cover_Type', y='Horizontal_Distance_To_Fire_Points', alpha=0.5, color='purple')
ax[0].set_title('Horizontal Distance To Fire Points')
x1=list(range(1,8,1))
ax[0].set_ylabel("")
ax[0].set_xlabel("Cover Type")
train.plot.scatter(ax=ax[1],x='Cover_Type', y='Horizontal_Distance_To_Roadways', alpha=0.5, color='purple')
ax[1].set_title('Horizontal Distance To Roadways')
x2=list(range(1,8,1))
ax[1].set_ylabel("")
ax[1].set_xlabel("Cover Type")
train.plot.scatter(ax=ax[2],x='Cover_Type', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='purple')
ax[2].set_title('Horizontal Distance To Hydrology')
x2=list(range(1,8,1))
ax[2].set_ylabel("")
ax[2].set_xlabel("Cover Type")
plt.show()
```

### Pandas Profiling
Actually there is a faster way for exploratory data analysis. Pandas provides you powerful HTML profiling reports with pandas-profiling. It's like a magic! You can click "Overview", "Variables" etc tabs for a quick run.


```python
import pandas_profiling as pp

report = pp.ProfileReport(train)
report.to_file("report.html")

report
```

# Modeling

Let's do it step by step first I'll slip the training data to make up a validation set and then try out different classification algorithms and then compare them


```python
# Separate the target variable
y = train.Cover_Type
train.drop(['Cover_Type'], axis = 1, inplace = True)
```


```python
# Split Data
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(train, y, random_state= 0)

print(f"Train Shape: {train_X.shape}, {train_y.shape}")
print(f"Validation Shape: {val_X.shape}, {val_y.shape}")
```


```python
# Let's fit the Decision Tree Model first

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# Fit the model
model.fit(train_X, train_y)
```


```python
# Get predictions for the validation set

predictions = model.predict(val_X)
```


```python
# Let's check our categorization accuracy

from sklearn.metrics import accuracy_score

accuracy_score(val_y, predictions)
```

That is good for our first model
I know we can improve upon this but let's see how well we stand on Leaderboard from this very basic model let's create a submission file and submit the scores and then move on to try different models


```python
# Predict the test data set
predictions_test = model.predict(test)
```


```python
submission = pd.DataFrame({
    "ID": test.index,
    "Cover_Type": predictions_test
})

submission.to_csv("submission.csv", index = False)
```


```python
submission.head()
```

Okay this worked! My score on leaderboard with this was 0.64322 and I received 435th/504 rank
Now I'll try few other models or variations
Let's try Random Forest but first let's try Decision Tree with some other parameters and let's see if we can improve our Decision Tree's accuracy
By using all default parameters our Validation Set accuracy is 0.76534



```python
# Let's create a helper function

def calc_tree_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state = 0)

    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    acc = accuracy_score(val_y, preds)

    return(acc)
```


```python
# Now let's fit our model with different max leaf nodes
accuracies = {}
for max_leaf_nodes in range(5, 2000, 50) :
    my_acc = calc_tree_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y)
    accuracies[max_leaf_nodes] = my_acc
    print(f"Max leaf nodes: {max_leaf_nodes} - - Accuracy: {my_acc}")

print("Best Max Leaf Nodes Parameter:", max(accuracies.items(), key = lambda k : k[1]))
```

That's amazing! My accuracy with 5 max leaf nodes was 55% and it kept growing until around max leaf nodes 1705
I think the highest accuracy was around 0.776 at 855 max leaf nodes


```python
# Let's create Random Forest model now
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state = 0)
model.fit(train_X, train_y)

predictions = model.predict(val_X)

print("Random Forest Accuracy: ", accuracy_score(val_y, predictions))
```

That is a huge improvement!
Let's see where this leads us on the Leaderboard


```python
predictions_test = model.predict(test)
submission = pd.DataFrame({
    "ID": test.index,
    "Cover_Type": predictions_test
})

submission.to_csv("submission.csv", index = False)
```

Now my score on leaderboard is 0.70138 and ranked 398th out of 505

Let's try different parameters using grid search cv


```python
%time

from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state = 0)

param_grid = {
    'n_estimators': [5, 50, 100, 200, 500, 750, 1000, 1250, 1500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'criterion' :['gini', 'entropy']
}

model_CV = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)

# We do not need validation set that's why we will train it on full training set
model_CV.fit(train, y)

# Let's check the best parameters
model_CV.best_params_
```

Okay so let's train with the best parameters now


```python
model = RandomForestClassifier(random_state=0, max_features='auto', criterion='entropy', max_depth=10, n_estimators=500)

model.fit(train_X, train_y)

predictions = model.predict(val_X)

print("Best Parameters Random Forest Accuracy Is: ", accuracy_score(val_y, predictions))
```

This is almost same or slightly lower than before...
Let's still check on test set


```python
predictions_test = model.predict(test)
submission = pd.DataFrame({
    "ID": test.index,
    "Cover_Type": predictions_test
})

submission.to_csv("submission.csv", index = False)
```

Let's try XGBoost now


```python
from xgboost import XGBClassifier

model = XGBClassifier()

# A parameter grid for XGBoost
grid_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
fit_params={"early_stopping_rounds":20,
            "eval_metric" : "merror",
            "eval_set" : [[val_X, val_y]]}

model_CV = GridSearchCV(estimator = model, grid_params, fit_params = fit_params, verbose = 10)
```
