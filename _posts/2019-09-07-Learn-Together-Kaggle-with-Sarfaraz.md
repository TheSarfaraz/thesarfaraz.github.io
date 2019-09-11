---
title: Learn Together Kaggle Competition with Sarfaraz
excerpt: "In this competition we’ll predict what types of trees there are in an area based on various geographic features.
The data is in raw form and contains categorical data such as wilderness areas and soil type."
header:
  overlay_image: "https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52"
  caption: "Credit: Kaggle.com"
toc_label: {title}
---


# Learn With Other Kaggle Users
## Classify forest types based on information about the area
![](https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52)

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
from google.colab import drive
drive.mount('/gdrive')
```

    Drive already mounted at /gdrive; to attempt to forcibly remount, call drive.mount("/gdrive", force_remount=True).



```python
PATH = "/gdrive/My\ Drive/DataSciLab/competitions/learn-together/"
Path = "/gdrive/My Drive/DataSciLab/competitions/learn-together/"
!ls {PATH}
```

    sample_submission.csv  test.csv  train.csv



```python
import pandas as pd


train = pd.read_csv(f"{Path}train.csv", index_col = "Id")
test = pd.read_csv(f"{Path}test.csv", index_col = "Id")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>Wilderness_Area4</th>
      <th>Soil_Type1</th>
      <th>Soil_Type2</th>
      <th>Soil_Type3</th>
      <th>Soil_Type4</th>
      <th>Soil_Type5</th>
      <th>Soil_Type6</th>
      <th>Soil_Type7</th>
      <th>Soil_Type8</th>
      <th>Soil_Type9</th>
      <th>Soil_Type10</th>
      <th>Soil_Type11</th>
      <th>Soil_Type12</th>
      <th>Soil_Type13</th>
      <th>Soil_Type14</th>
      <th>Soil_Type15</th>
      <th>Soil_Type16</th>
      <th>Soil_Type17</th>
      <th>Soil_Type18</th>
      <th>Soil_Type19</th>
      <th>Soil_Type20</th>
      <th>Soil_Type21</th>
      <th>Soil_Type22</th>
      <th>Soil_Type23</th>
      <th>Soil_Type24</th>
      <th>Soil_Type25</th>
      <th>Soil_Type26</th>
      <th>Soil_Type27</th>
      <th>Soil_Type28</th>
      <th>Soil_Type29</th>
      <th>Soil_Type30</th>
      <th>Soil_Type31</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Training Data Set Shape: {train.shape} \nTest Data Set Shape: {test.shape}")
```

    Training Data Set Shape: (15120, 55)
    Test Data Set Shape: (565892, 54)



```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15120 entries, 1 to 15120
    Data columns (total 55 columns):
    Elevation                             15120 non-null int64
    Aspect                                15120 non-null int64
    Slope                                 15120 non-null int64
    Horizontal_Distance_To_Hydrology      15120 non-null int64
    Vertical_Distance_To_Hydrology        15120 non-null int64
    Horizontal_Distance_To_Roadways       15120 non-null int64
    Hillshade_9am                         15120 non-null int64
    Hillshade_Noon                        15120 non-null int64
    Hillshade_3pm                         15120 non-null int64
    Horizontal_Distance_To_Fire_Points    15120 non-null int64
    Wilderness_Area1                      15120 non-null int64
    Wilderness_Area2                      15120 non-null int64
    Wilderness_Area3                      15120 non-null int64
    Wilderness_Area4                      15120 non-null int64
    Soil_Type1                            15120 non-null int64
    Soil_Type2                            15120 non-null int64
    Soil_Type3                            15120 non-null int64
    Soil_Type4                            15120 non-null int64
    Soil_Type5                            15120 non-null int64
    Soil_Type6                            15120 non-null int64
    Soil_Type7                            15120 non-null int64
    Soil_Type8                            15120 non-null int64
    Soil_Type9                            15120 non-null int64
    Soil_Type10                           15120 non-null int64
    Soil_Type11                           15120 non-null int64
    Soil_Type12                           15120 non-null int64
    Soil_Type13                           15120 non-null int64
    Soil_Type14                           15120 non-null int64
    Soil_Type15                           15120 non-null int64
    Soil_Type16                           15120 non-null int64
    Soil_Type17                           15120 non-null int64
    Soil_Type18                           15120 non-null int64
    Soil_Type19                           15120 non-null int64
    Soil_Type20                           15120 non-null int64
    Soil_Type21                           15120 non-null int64
    Soil_Type22                           15120 non-null int64
    Soil_Type23                           15120 non-null int64
    Soil_Type24                           15120 non-null int64
    Soil_Type25                           15120 non-null int64
    Soil_Type26                           15120 non-null int64
    Soil_Type27                           15120 non-null int64
    Soil_Type28                           15120 non-null int64
    Soil_Type29                           15120 non-null int64
    Soil_Type30                           15120 non-null int64
    Soil_Type31                           15120 non-null int64
    Soil_Type32                           15120 non-null int64
    Soil_Type33                           15120 non-null int64
    Soil_Type34                           15120 non-null int64
    Soil_Type35                           15120 non-null int64
    Soil_Type36                           15120 non-null int64
    Soil_Type37                           15120 non-null int64
    Soil_Type38                           15120 non-null int64
    Soil_Type39                           15120 non-null int64
    Soil_Type40                           15120 non-null int64
    Cover_Type                            15120 non-null int64
    dtypes: int64(55)
    memory usage: 6.5 MB



```python
train.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elevation</th>
      <td>15120.0</td>
      <td>2749.322553</td>
      <td>417.678187</td>
      <td>1863.0</td>
      <td>2376.0</td>
      <td>2752.0</td>
      <td>3104.00</td>
      <td>3849.0</td>
    </tr>
    <tr>
      <th>Aspect</th>
      <td>15120.0</td>
      <td>156.676653</td>
      <td>110.085801</td>
      <td>0.0</td>
      <td>65.0</td>
      <td>126.0</td>
      <td>261.00</td>
      <td>360.0</td>
    </tr>
    <tr>
      <th>Slope</th>
      <td>15120.0</td>
      <td>16.501587</td>
      <td>8.453927</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>22.00</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Hydrology</th>
      <td>15120.0</td>
      <td>227.195701</td>
      <td>210.075296</td>
      <td>0.0</td>
      <td>67.0</td>
      <td>180.0</td>
      <td>330.00</td>
      <td>1343.0</td>
    </tr>
    <tr>
      <th>Vertical_Distance_To_Hydrology</th>
      <td>15120.0</td>
      <td>51.076521</td>
      <td>61.239406</td>
      <td>-146.0</td>
      <td>5.0</td>
      <td>32.0</td>
      <td>79.00</td>
      <td>554.0</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Roadways</th>
      <td>15120.0</td>
      <td>1714.023214</td>
      <td>1325.066358</td>
      <td>0.0</td>
      <td>764.0</td>
      <td>1316.0</td>
      <td>2270.00</td>
      <td>6890.0</td>
    </tr>
    <tr>
      <th>Hillshade_9am</th>
      <td>15120.0</td>
      <td>212.704299</td>
      <td>30.561287</td>
      <td>0.0</td>
      <td>196.0</td>
      <td>220.0</td>
      <td>235.00</td>
      <td>254.0</td>
    </tr>
    <tr>
      <th>Hillshade_Noon</th>
      <td>15120.0</td>
      <td>218.965608</td>
      <td>22.801966</td>
      <td>99.0</td>
      <td>207.0</td>
      <td>223.0</td>
      <td>235.00</td>
      <td>254.0</td>
    </tr>
    <tr>
      <th>Hillshade_3pm</th>
      <td>15120.0</td>
      <td>135.091997</td>
      <td>45.895189</td>
      <td>0.0</td>
      <td>106.0</td>
      <td>138.0</td>
      <td>167.00</td>
      <td>248.0</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <td>15120.0</td>
      <td>1511.147288</td>
      <td>1099.936493</td>
      <td>0.0</td>
      <td>730.0</td>
      <td>1256.0</td>
      <td>1988.25</td>
      <td>6993.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area1</th>
      <td>15120.0</td>
      <td>0.237897</td>
      <td>0.425810</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area2</th>
      <td>15120.0</td>
      <td>0.033003</td>
      <td>0.178649</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area3</th>
      <td>15120.0</td>
      <td>0.419907</td>
      <td>0.493560</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Wilderness_Area4</th>
      <td>15120.0</td>
      <td>0.309193</td>
      <td>0.462176</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type1</th>
      <td>15120.0</td>
      <td>0.023479</td>
      <td>0.151424</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type2</th>
      <td>15120.0</td>
      <td>0.041204</td>
      <td>0.198768</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type3</th>
      <td>15120.0</td>
      <td>0.063624</td>
      <td>0.244091</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type4</th>
      <td>15120.0</td>
      <td>0.055754</td>
      <td>0.229454</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type5</th>
      <td>15120.0</td>
      <td>0.010913</td>
      <td>0.103896</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type6</th>
      <td>15120.0</td>
      <td>0.042989</td>
      <td>0.202840</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type7</th>
      <td>15120.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Soil_Type8</th>
      <td>15120.0</td>
      <td>0.000066</td>
      <td>0.008133</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type9</th>
      <td>15120.0</td>
      <td>0.000661</td>
      <td>0.025710</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type10</th>
      <td>15120.0</td>
      <td>0.141667</td>
      <td>0.348719</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type11</th>
      <td>15120.0</td>
      <td>0.026852</td>
      <td>0.161656</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type12</th>
      <td>15120.0</td>
      <td>0.015013</td>
      <td>0.121609</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type13</th>
      <td>15120.0</td>
      <td>0.031481</td>
      <td>0.174621</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type14</th>
      <td>15120.0</td>
      <td>0.011177</td>
      <td>0.105133</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type15</th>
      <td>15120.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Soil_Type16</th>
      <td>15120.0</td>
      <td>0.007540</td>
      <td>0.086506</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type17</th>
      <td>15120.0</td>
      <td>0.040476</td>
      <td>0.197080</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type18</th>
      <td>15120.0</td>
      <td>0.003968</td>
      <td>0.062871</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type19</th>
      <td>15120.0</td>
      <td>0.003042</td>
      <td>0.055075</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type20</th>
      <td>15120.0</td>
      <td>0.009193</td>
      <td>0.095442</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type21</th>
      <td>15120.0</td>
      <td>0.001058</td>
      <td>0.032514</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type22</th>
      <td>15120.0</td>
      <td>0.022817</td>
      <td>0.149326</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type23</th>
      <td>15120.0</td>
      <td>0.050066</td>
      <td>0.218089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type24</th>
      <td>15120.0</td>
      <td>0.016997</td>
      <td>0.129265</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type25</th>
      <td>15120.0</td>
      <td>0.000066</td>
      <td>0.008133</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type26</th>
      <td>15120.0</td>
      <td>0.003571</td>
      <td>0.059657</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type27</th>
      <td>15120.0</td>
      <td>0.000992</td>
      <td>0.031482</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type28</th>
      <td>15120.0</td>
      <td>0.000595</td>
      <td>0.024391</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type29</th>
      <td>15120.0</td>
      <td>0.085384</td>
      <td>0.279461</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type30</th>
      <td>15120.0</td>
      <td>0.047950</td>
      <td>0.213667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type31</th>
      <td>15120.0</td>
      <td>0.021958</td>
      <td>0.146550</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type32</th>
      <td>15120.0</td>
      <td>0.045635</td>
      <td>0.208699</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type33</th>
      <td>15120.0</td>
      <td>0.040741</td>
      <td>0.197696</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type34</th>
      <td>15120.0</td>
      <td>0.001455</td>
      <td>0.038118</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type35</th>
      <td>15120.0</td>
      <td>0.006746</td>
      <td>0.081859</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type36</th>
      <td>15120.0</td>
      <td>0.000661</td>
      <td>0.025710</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type37</th>
      <td>15120.0</td>
      <td>0.002249</td>
      <td>0.047368</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type38</th>
      <td>15120.0</td>
      <td>0.048148</td>
      <td>0.214086</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type39</th>
      <td>15120.0</td>
      <td>0.043452</td>
      <td>0.203880</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Soil_Type40</th>
      <td>15120.0</td>
      <td>0.030357</td>
      <td>0.171574</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <td>15120.0</td>
      <td>4.000000</td>
      <td>2.000066</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.00</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



All Wilderness_Area and Soil_Type columns have values in the range of 0 and 1. Quite likely these columns are categorical and consist of 0 and 1. To validate this i'm checking distinct values of following columns:


```python
print(train.iloc[:, 10:-1].columns)
```

    Index(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
           'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
           'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
           'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
           'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
           'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
           'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
           'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
           'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
           'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
           'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'],
          dtype='object')



```python
pd.unique(train.iloc[:,10:-1].values.ravel())
```




    array([1, 0])



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


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_15_0.png)


It seems the most important correlations are between "Horizontal Distance To Hydrology" and "Vertical Distance To Hydrology" with 70%; between "Aspect" and "Hillshade 3pm" with 60%; between "Hillshade Noon" and "Hillshade 3pm" with %60; between "Elevation" and "Horizontal Distance To Roadways" with %60. Let's see how they are looking.


```python
train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='yellow', figsize = (12,9))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()
```


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_17_0.png)



```python
train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', alpha=0.5, color='maroon', figsize = (12,9))
plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")
plt.show()
```


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_18_0.png)



```python
train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', alpha=0.5, color='purple', figsize = (12,9))
plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")
plt.show()
```


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_19_0.png)


There are obvious patterns if we ignore to outliers. And with this patterns, our model will learn.

Boxplot can be used to see outliers. For a better visualization i will use plotly this time.

Wow, this is very interesting plot! I need to learn this too!


```python
!pip install plotly
```

    Requirement already satisfied: plotly in /usr/local/lib/python3.6/dist-packages (4.1.1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from plotly) (1.12.0)
    Requirement already satisfied: retrying>=1.3.3 in /usr/local/lib/python3.6/dist-packages (from plotly) (1.3.3)



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

{% raw %}

<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="ef689e60-fe28-4c15-aa89-b5f96cc3cc40" class="plotly-graph-div" style="height:500px; width:700px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("ef689e60-fe28-4c15-aa89-b5f96cc3cc40")) {
                    Plotly.newPlot(
                        'ef689e60-fe28-4c15-aa89-b5f96cc3cc40',
                        [{"marker": {"color": "rgb(0,145,119)"}, "name": "Vertical Distance", "type": "box", "y": [0, -6, 65, 118, -1, -15, 5, 7, 56, 11, 51, 26, 69, 46, 4, 2, -1, 6, 5, 10, 8, 30, 82, 0, 14, 0, -4, 23, 30, 53, 27, 1, 4, 4, 4, 20, 23, 18, 53, 0, 0, 0, -3, 17, 23, -5, -4, 0, 7, 13, 17, 43, 0, 19, 11, 19, 9, -4, 13, 13, 17, 7, 22, 10, 27, 20, 10, 6, 42, -2, 53, 10, 30, 0, 19, 74, 65, 14, 95, 42, 96, 22, 82, 88, 17, 98, 17, 58, 48, 24, -5, 8, 9, -3, 85, 98, 7, 61, 0, 16, 16, 2, -5, 71, 23, 30, -2, 23, 89, 0, 33, 26, 24, 30, 18, 46, 63, 58, 30, 45, 34, -6, 53, 10, 39, 4, 101, 27, 25, 34, 0, 8, 0, 20, 0, 10, 3, 94, 7, 13, 108, 23, 16, 79, 6, 32, 23, 69, 10, 16, 0, 99, 36, 39, 55, 0, 36, 17, 19, -7, 42, 16, 89, 22, 13, 39, -25, 33, 20, 66, 16, 7, 158, 54, 92, 125, 39, 128, 23, 109, 43, -1, 13, 3, 49, 10, 46, 9, 39, -3, 89, 135, -2, 10, -2, 39, 52, 210, 39, 6, 43, 59, 34, 144, -2, 0, 0, 49, 35, 22, 180, 63, 161, 177, 17, 76, 10, 180, 0, 66, 15, 134, 16, 43, 85, 21, 137, 31, 105, 148, 26, 7, 23, 65, 21, 26, 53, 36, 15, 58, 35, 51, 69, 128, 33, 7, 49, 151, 108, 35, 62, 23, 20, 89, 4, 132, 85, 20, 78, 53, 46, 0, 0, 23, 66, 0, 86, 20, 56, 36, 23, 0, 0, 14, 8, -2, 33, 50, 22, 36, 49, 12, 2, 13, 144, 8, 0, -13, 7, 3, 20, 43, -19, 98, 49, 21, 3, 37, 167, 27, 10, 19, 13, 32, -32, 47, 19, 33, 101, 56, 26, 0, 10, 10, 29, 18, 68, 10, 20, -6, 3, -6, 26, 26, 10, 19, 3, 0, 52, 13, 30, 36, 85, 19, -18, 7, -6, 0, 7, 49, 66, 13, 0, 32, 17, 23, 4, 46, 20, 0, 62, 13, 42, 103, 110, 29, 8, 20, 42, 66, 4, 107, 15, -28, 12, 46, -9, -25, -35, 49, 13, 4, 3, 41, 112, 125, 15, 20, 25, 47, 8, -12, 3, 22, 119, 105, 93, 55, 3, 101, 92, 89, 79, 77, 75, 79, 62, 2, 0, 16, 19, 0, 0, 18, 69, 35, 4, 6, 3, 16, 43, 49, 31, 23, 0, -3, 69, 0, 40, 22, 0, 3, 7, 76, 45, 0, 26, 9, 0, 0, 59, 7, 5, 29, 11, 0, 2, 7, 23, 4, 49, 13, 0, 32, 13, 0, 45, 63, 50, 0, 2, 0, 23, 7, -23, 9, 13, 6, -1, 23, 26, 18, 29, 23, 50, 72, 39, 55, 13, 0, 24, 16, 60, 23, 9, 9, 20, 101, 54, 39, 89, 85, 50, 51, 36, 23, 111, 36, 29, 27, 39, 49, 53, 0, 17, 5, 5, -2, 114, 75, 0, 16, 47, 35, 9, 127, 0, 33, 34, 60, 55, 3, 13, 23, 17, 0, -7, 80, -6, 16, 141, 0, 78, -1, 19, 59, 16, 0, 7, 0, 0, 37, 14, 16, 89, 74, 13, 73, 95, 66, -15, 92, 2, 14, 104, 93, 89, 99, 36, 82, 47, 38, 20, 66, 2, 22, 5, 22, 17, 78, 28, 66, 26, 32, 4, 119, 33, 63, 35, 21, 22, 62, 61, 110, 47, 51, 16, 19, -3, 9, 9, 45, 22, 33, -4, -3, 26, 4, 30, 12, 6, 33, 41, 79, 39, 0, 49, 0, 20, 29, 4, -2, 58, 11, 16, 42, 8, -3, 23, 3, 0, 13, 7, -3, 1, 0, 79, 8, 3, 6, 11, 33, 3, -2, -1, 17, 24, 13, 0, 0, 13, -1, 59, 24, 59, 3, 4, 0, 45, -3, 0, 0, 0, 23, 70, 42, 32, 114, 5, 0, 11, -5, 33, 20, 6, 10, 4, 4, 0, 20, 14, 12, 33, 66, 7, 42, 7, 0, 35, 63, 51, 6, 143, 0, 7, 139, 64, -1, 4, 18, 13, 13, 18, 66, -6, 34, -5, 16, 6, 181, 50, 134, 182, 6, 14, 21, 36, 0, 0, -1, 4, 3, 53, 13, 65, 44, -1, 56, 215, 47, -5, -9, 43, 45, 43, 32, 97, 72, 186, 78, 15, 52, 26, 245, 43, 121, 99, 52, 3, 7, 11, 29, 27, 6, 17, 105, -20, 43, 32, 171, 45, 51, 1, -5, 8, 32, -9, 40, -3, 20, 43, 61, 36, 45, 22, 1, -12, 16, 43, 33, 3, -45, 13, 39, 0, 7, 13, 9, 26, 71, 9, 46, -3, 10, 42, 36, 49, 50, 10, 55, 22, 41, 0, 21, 159, -3, -3, 4, 51, 60, 19, 0, -2, 46, 55, 0, 47, 57, 4, 20, 158, 0, -8, 0, 2, 1, 16, 20, 50, 14, 95, 45, 9, -10, 14, 53, 6, 52, 13, 0, 0, 1, 13, 42, -4, 22, 22, 76, 15, 7, 20, 20, 33, 56, 2, -16, 23, 39, 16, -1, 37, -18, 48, 17, 3, 13, 43, 11, 43, 88, 26, 0, 4, 1, 4, 26, 46, 7, 6, 115, 6, 4, 36, 21, 5, 6, 22, 18, 0, 4, 55, 112, 41, 19, 3, -3, 8, 31, 65, 5, 42, 5, 19, 7, 0, 85, 14, 1, 2, 11, 0, -3, 4, 11, 74, 21, 55, 1, 10, 12, 3, 3, 18, -2, 0, 14, 66, 32, 57, 21, 26, 10, 13, 0, 2, 5, 26, 36, 0, 14, 27, 29, 93, 60, 51, -11, 25, 8, 0, 27, 33, 33, 4, 9, 39, 58, 18, 19, 24, -20, -2, 8, 34, 48, 136, 41, 24, 5, 1, 11, -1, 1, 5, 3, 65, 61, 28, 44, 41, -1, 38, 14, 102, 0, 8, 8, 30, 72, 0, 6, 4, 15, 12, 95, -3, 0, 3, 0, 2, 20, 45, -1, 16, 21, 5, 41, 44, 21, 49, 5, 10, 94, 31, 33, 21, 7, 23, 44, 0, 0, 16, 9, 25, 50, 13, 18, 31, 0, 60, 9, 1, 0, -2, 160, 55, 0, 51, 9, 60, 2, 10, 39, 112, 44, 29, 63, 30, 11, 29, 25, 19, 109, -2, 12, 47, 25, 29, 65, 50, 18, 30, 38, 46, 1, 16, 28, 22, 13, 3, 58, 13, 12, 13, 11, 9, 130, 3, 0, 0, 9, 9, 12, -2, 0, 10, 102, 12, 0, -24, 19, 28, 0, 0, 2, 23, 78, 74, 20, 11, 13, 12, 36, 32, 88, 4, 24, 5, 41, 17, 8, 10, 9, -15, 98, 70, 84, 9, 36, 71, 25, 4, 15, 56, 0, 97, 1, 14, 6, 141, -1, 0, 4, 151, 4, 11, 30, 49, 2, 98, 25, 8, 26, 25, 12, 17, 14, -1, 94, 5, 22, 67, 27, 48, 88, 8, 11, 7, 1, 16, 30, 20, 25, 86, 5, 2, 10, 74, 46, 59, 2, 0, 57, 33, 131, 40, 36, 80, 43, 63, 37, 20, 12, 8, 60, 59, 0, 36, 0, 39, 15, 67, 92, 70, 35, 32, 119, 74, 68, 65, 43, 4, 22, 3, 99, 105, 57, 11, 73, 0, 110, 100, 91, 57, 37, 52, 143, 86, 65, 52, 46, 15, 176, 106, 95, 59, 38, 4, 92, 79, 41, 67, 3, 21, 42, 46, 22, 2, 21, 197, 169, 146, 15, 10, 129, 74, 60, 75, 70, 46, 29, 21, 152, 129, 92, 71, 77, 82, -6, 4, 1, 107, 75, 118, 74, 105, 39, 78, 52, 30, 58, -5, 122, 96, 68, 14, 10, 105, 64, 136, 71, 54, 36, 35, 2, 12, 25, -36, 163, 127, 91, 48, 2, 96, 23, 4, 0, 32, 5, 171, 137, 64, -8, 46, 193, 141, 37, -1, 96, 10, 88, 76, 105, 83, 61, 61, 97, -8, 11, 0, -37, 135, 91, 69, 146, 92, 60, 73, 16, 128, 30, 3, 8, 126, 47, 6, 6, 0, 107, 59, 45, 109, 66, 1, 2, 13, -15, 19, -7, -8, -19, 8, 19, 10, 78, 6, 83, 80, 3, -13, 38, 38, 52, 35, 102, 0, -8, 16, -14, 35, 24, 23, -2, -15, 95, 77, 33, 62, -23, 0, 3, -19, 112, 14, 76, 43, 0, 48, -8, 145, 49, 41, 7, 0, 34, 37, 119, 0, 115, 13, 154, 109, 75, 60, 2, 31, 76, 8, 151, 135, 130, 98, 3, 47, 113, 8, 49, 127, 136, 124, 134, 18, 33, 57, 151, 6, 185, 149, 140, 47, -7, 47, 112, 3, 18, 199, 150, 118, 35, 25, 132, 60, -6, 145, 81, 72, 35, 187, 141, 52, -3, 240, 95, 20, 195, 110, -7, 11, -1, -30, 163, 53, 6, 18, 68, 35, 78, 24, 61, 13, 45, 34, 3, 2, 3, 26, 31, 10, 58, 48, -2, 34, 0, -9, 0, 40, 0, 1, 56, 5, 10, 0, -6, -7, 146, -12, 32, -6, 69, 9, 12, 36, 55, 37, 52, 40, 28, -20, 57, -26, 20, 68, 70, 52, 15, 41, 117, 10, -30, -32, 65, 14, 87, 47, 15, -6, 80, 41, 54, 51, 103, 0, 21, 70, 3, 109, 52, 5, 146, 11, 31, 13, -5, 147, 81, 0, 179, 17, 4, -134, 63, 3, -7, -6, 29, 45, 33, 22, 84, -2, 9, 6, 18, 0, -85, 24, 16, 11, 31, 17, 18, -29, 87, -18, -6, 0, -6, -5, 0, -20, -17, 0, 0, 0, -8, -15, 22, 10, 5, 74, 169, -17, 5, -3, -26, -35, 27, 0, -5, 24, -60, -71, 88, 74, 10, 87, 45, 0, 28, 53, 31, 85, 51, -2, 12, 7, -100, -110, 51, -95, -108, 137, 68, 52, 8, 120, 16, 9, 12, 128, 29, 89, 98, 78, 13, 87, 67, 85, 97, 55, 59, -6, 21, -8, 2, -3, 20, -17, 1, 9, 20, 0, -6, -9, 41, -30, 30, -35, 70, 19, 87, 5, 6, -50, 85, 22, 3, -50, -62, 70, 53, -65, 93, 93, 46, 55, 45, 37, -51, 135, 66, 15, 22, 50, 50, 49, 7, 11, -5, 17, -6, 22, -51, 29, 33, 23, 150, 111, 5, 34, 27, 26, 32, 31, 0, -15, 114, 14, 64, 49, 17, 19, 43, 31, 16, 0, 54, 52, 75, 20, -47, -10, 161, 18, -11, 53, -64, 217, 199, 70, 48, 0, 0, 13, 25, 16, 230, 136, 87, 91, 11, -1, 26, 210, 129, 97, 71, 2, 0, 4, 15, 185, 129, 106, 73, 62, 2, -7, 23, 6, 15, 3, -76, 241, 132, 118, 86, 82, 4, 2, -9, 10, 5, 5, -72, 145, 54, 47, 7, -10, -12, -41, 18, 8, -100, 189, 170, 71, 94, 23, 60, 17, 3, 0, 181, 122, 69, 70, 14, 0, 0, 0, 85, -3, 0, 137, 70, 55, 1, 0, 85, -7, 244, 121, 62, 60, 0, 109, 0, 88, 0, 23, 138, 98, 554, 73, -4, 35, -5, 40, 24, 28, -2, 25, 87, 58, 35, 186, 14, 95, 42, 29, 39, 8, 103, 42, 71, 120, 34, 28, 59, 155, 99, 34, -1, 82, 72, 11, 107, 32, 9, 131, 95, 83, 14, -9, 112, 82, 64, 145, 81, 14, 32, -2, -3, 63, 55, 13, 23, 94, 19, 41, 40, 93, 60, 8, 3, 72, 93, 0, 30, 89, 113, 160, 59, 87, 49, 54, 70, 53, 115, 0, 4, 54, 0, 15, 22, 10, 6, 15, 36, 84, 158, 547, 93, 0, 0, 0, 19, 2, 72, 9, 49, 105, 125, 8, 12, 80, 82, 84, 165, 164, 100, 7, 19, 5, 7, 54, 50, 76, 0, 28, 12, 158, 31, 6, 97, 43, 9, 71, 82, 105, 131, 64, 13, -13, 75, 0, 18, 168, 36, 90, 79, 68, 75, 2, 143, 151, 22, 170, 8, 26, 142, 21, 47, 354, 83, 98, 129, 0, 151, 78, 44, -19, 74, 47, 93, 91, 90, 150, 5, 120, 52, 92, 86, 152, 82, 184, 0, 72, 144, 173, 175, 172, 50, 52, 80, 112, 13, 4, 143, 3, 89, 65, 113, 83, 74, 12, 0, 124, 0, 32, 29, 26, 26, 107, 95, 16, 28, 71, 31, 37, 26, 0, 110, 134, 117, 37, 7, 106, 124, 167, 192, 11, -40, 10, 19, 0, 23, 3, 4, 13, 26, 74, 65, 71, 211, 14, 5, 22, 0, 2, 27, 29, 39, 5, 6, 67, 69, 84, 126, 14, 43, 0, 88, 51, 91, 73, 8, 25, 123, 118, 228, 16, 3, 4, 38, 7, 0, 56, 0, 11, 14, 7, 3, 132, 64, 59, 74, 31, 9, 0, 76, 14, 29, 7, 4, 62, 209, 249, 16, 0, 10, 35, 79, 0, 5, 37, 29, 47, 5, 10, 76, 45, 219, 62, 56, 17, 76, 32, 0, 64, 1, 1, 9, 11, 162, 152, 128, 116, -1, 168, 222, 241, 0, 9, 139, 40, 15, 0, 0, 2, 21, 11, 24, 37, 160, 49, 12, 0, 75, 127, 137, -24, 0, 32, 22, 6, 0, 40, 0, 245, 104, 21, 42, 65, 95, 42, 122, 9, 6, 0, 0, -2, -1, 0, 12, 5, 8, 3, 130, 119, 35, 247, 0, 86, 15, 17, 34, 119, 18, 12, 0, 5, 0, 0, 41, 39, 37, 137, 104, 28, 62, 56, 15, 62, 29, -18, 56, 22, 9, 5, 58, 12, -82, 78, 89, 26, 6, 118, 40, -1, 5, 0, 53, 12, 0, 43, 0, 7, 109, 86, 104, 94, 0, 53, 12, -4, 67, 8, 46, 52, 68, 55, 46, 18, 2, 10, -83, 149, 62, 11, 59, 21, 2, 2, 0, -9, 12, 15, 0, 0, 30, 123, 76, 56, 56, 8, 12, 22, 163, 124, 5, 10, 37, 0, -5, 73, 104, 133, 2, 6, 96, 60, 134, 143, 128, 74, 60, 35, 23, 11, -10, 2, 19, 16, 163, 63, 43, 8, 6, 0, 39, 51, 57, 4, 2, 14, 112, 72, 93, 104, 52, 29, 13, 5, 42, 9, 20, 17, -13, 30, 239, 168, 147, 13, 33, 0, 109, 117, 165, 12, 29, -63, 11, 19, 23, -8, 0, 10, -7, 9, 34, 16, 16, 0, 91, 16, 0, 66, 115, 9, 42, 161, 213, 132, 24, 8, -2, 29, 9, -5, 15, 8, 18, 24, 44, 86, 84, 81, 144, 15, 115, 118, 36, 154, 114, 23, 31, 27, 12, -4, 17, 15, 15, 0, 67, 28, 24, 102, 125, 206, 245, 199, 162, 176, 4, 41, 5, 0, -13, 25, -9, 2, 0, 3, 7, -5, 8, 6, 63, 17, 33, 60, 10, 28, 77, 41, -21, 79, 69, 22, -3, -29, 0, 0, 0, 0, 0, 0, 2, 0, 51, 27, 204, 159, 0, 96, 4, 0, 35, -5, 204, 84, 49, 9, 63, 0, 0, 13, 13, 0, 3, 3, 21, 59, 103, 37, 2, 101, 78, 114, 165, 1, 0, -1, 2, 196, 0, 20, 37, 87, 69, 0, 7, 0, 0, 0, 14, 26, 30, -7, 5, -2, 0, 41, 60, 61, 0, -6, 16, 10, 16, 60, 91, 70, 25, 32, 162, 70, 0, -6, -33, 3, 9, 27, -7, 7, 5, 38, 73, 4, 141, 126, 19, 80, 20, -43, 160, 0, -4, -8, 16, 33, 74, 6, 0, 128, 155, 21, 41, 11, -1, -9, -7, -8, -18, 10, 70, 76, 32, 94, 106, 50, 45, 81, -3, 48, 284, -5, 0, 0, 0, -10, 16, 8, -19, 1, 24, 82, 88, 16, 30, -3, 12, 0, 3, 3, 35, 28, 100, 25, 72, 29, 24, 13, 0, -15, 1, 0, 0, 88, 54, 71, 12, 0, 0, 67, 5, -42, 60, 5, 5, 3, 3, -1, -20, -26, -1, -33, 39, 4, 0, 97, 20, 58, 50, 0, 0, 0, 0, 0, 5, 114, 174, 168, 37, -19, 39, 102, 25, 30, 11, 6, 77, 52, 61, 101, 0, 0, 0, 7, 1, 0, 0, 5, 0, 2, 2, 14, 32, 14, 44, -35, 56, 36, 40, 37, 10, 2, 0, 11, 12, 38, 91, 35, 0, 0, 3, 0, 0, 1, 0, 0, 74, -22, 30, 233, 29, -19, -35, 37, 0, 0, 8, 18, 92, 67, 20, 0, 0, 5, 0, 1, 1, 0, 0, 7, 119, 23, 81, 60, 28, -13, 3, 96, 14, 0, 32, 85, -6, 40, 10, 0, 0, 0, 0, 66, 6, 0, 32, 80, 159, 106, -28, -104, 44, 60, 50, 22, -30, 60, 0, 0, 4, 9, 32, -11, 115, 0, 0, 98, 72, 91, 15, 11, 15, -38, -69, -7, 18, 5, -1, 0, 0, 25, 11, 9, 109, 112, 85, 80, 70, 121, 16, 1, 0, 47, 58, -78, -87, 26, 18, 6, -13, -2, 8, 0, 1, 6, 155, 11, 142, 20, -4, 45, 15, 0, 100, 106, 112, -6, -15, -34, 0, 15, 4, 108, 0, 121, 122, 127, 121, 90, 45, 108, 8, 41, 0, 0, 203, 94, 136, 37, 77, 88, -37, 6, 0, 0, 17, 71, 34, 72, -19, 27, -11, 45, 144, 77, 108, -16, -39, 83, 50, 42, 91, 0, 123, 184, 155, 141, 90, -4, -13, 34, 12, 0, 55, 196, -17, -21, 25, 62, 67, 83, -24, -39, 78, 72, 12, 131, 165, -26, 7, 0, 16, 25, 193, 55, 17, 165, 105, 105, 50, 0, -1, 4, 48, 76, 110, 33, 0, 14, 0, 189, 0, 103, 232, 116, 18, 28, 37, 25, 9, 65, 36, 0, -2, 0, 0, 0, 0, 0, 153, 32, 24, 0, 0, 0, 75, 86, 0, 155, 130, 66, 159, 151, -53, 35, 3, 12, 0, 0, 2, 11, 65, 41, 42, 3, 0, 0, 49, 39, 25, 248, 135, 51, 83, 22, 0, 0, -3, 0, 0, 0, 39, 26, 0, 71, 59, 60, 64, 65, 37, 53, 0, 12, 0, 0, 163, -13, -4, 0, 23, 69, 11, 0, 129, 128, 81, 58, 66, 2, 6, 22, 29, 8, 38, 16, 104, 120, 89, 68, 35, 10, 92, 12, 37, 130, 96, 55, 71, 91, 204, 37, 31, 0, 11, 92, 132, 117, 94, 66, 43, 11, 0, 106, 11, 94, 108, 119, 84, 56, 54, 25, 10, 25, 27, 17, 0, 142, 43, 49, 86, 63, 16, 65, 118, 117, 40, 59, 80, 262, -16, 14, 47, 14, 11, 25, 0, 77, 73, -15, 14, 52, 0, 0, 21, 136, 84, -7, 197, 161, 133, 88, 45, 46, -12, 69, 102, 18, 58, 55, 111, 95, 0, 0, 42, 100, 47, 20, 217, 147, 132, 110, 93, 44, 17, 100, 40, 54, 20, 0, 32, 144, 245, 231, 213, 202, 184, 163, 128, 43, 21, 7, 0, 105, 21, 51, 26, 0, -10, 13, -1, 0, 67, 73, 60, 51, 241, 214, 182, 102, 81, 62, 59, 41, 25, 6, 0, 87, 40, 37, 13, 105, 121, 83, 132, 108, 49, 84, 43, 37, 24, 7, 3, 0, 165, 163, 160, 6, 0, 124, 46, 60, 68, 76, 90, 74, 202, 174, 139, 112, 89, 72, 35, 10, 0, 0, 35, 12, 88, 123, 20, 89, 83, 56, 11, 92, 39, 0, 0, 10, 58, 54, 73, 91, 107, 86, 228, 213, 198, 179, 174, 162, 148, 111, 89, 49, 31, 16, 0, 0, 77, 111, 31, 37, 44, 28, 132, 23, 25, 18, 54, 79, 64, 51, 32, 77, 85, 0, 15, 53, 147, 33, 36, 0, 36, 66, 71, 64, 54, 48, 82, 58, 225, 180, 165, 151, 121, 81, 41, 27, 12, 0, 0, 61, 56, 25, 104, -12, 45, 45, 43, 88, 72, 61, 49, 0, 59, 77, 46, 135, 0, 0, 0, 152, 0, 95, 108, 238, 49, 55, 59, 68, 84, 90, 74, 60, 53, 47, 72, 64, 230, 209, 187, 150, 125, 110, 94, 77, 18, 5, 0, -3, -1, 11, 75, 58, 78, 66, 121, 105, 29, 35, 38, 42, 51, 0, 52, 83, 83, 110, 0, 33, 42, 42, 42, 56, 83, 41, 35, 31, 52, 59, 237, 214, 198, 143, 128, 123, 114, 101, 83, 31, 0, 45, 82, 60, 65, 65, 26, 95, 49, 24, 37, 58, 51, 7, 37, -49, 0, -3, 18, 183, 44, 37, 35, 37, 37, 40, 52, 68, 74, 37, 28, 21, 21, 39, 132, 116, 102, 61, 93, 44, 11, 20, 17, 55, 48, 57, 30, 13, 7, 0, 0, 40, 124, 4, 7, 37, 27, 25, 34, 48, 66, 62, 15, 10, 212, 226, 204, 185, 160, 124, 79, 76, 59, 59, 17, 2, 0, 90, 16, 33, 0, -2, 0, 0, 0, 0, 29, 10, 4, 115, 30, 19, 23, 44, 54, 53, 25, 12, 7, 0, 38, 129, 150, 147, 151, 0, 19, 31, 75, 225, 23, 16, 16, 20, 31, 61, 66, 29, 14, -2, 209, 218, 193, 167, 149, 116, 99, 85, 59, 37, 32, 0, 97, 162, 175, -11, 0, 3, 0, 2, 2, 15, 147, 82, 13, 15, 30, 25, 26, 46, 16, 0, 1, 155, 178, 0, 0, -2, -3, 0, 0, 181, 34, 11, 4, 167, 207, 206, 198, 116, 99, 88, 64, 35, 25, 16, 0, 128, 0, 0, 0, 3, 31, 86, 82, 11, 18, -3, 101, 135, 105, 3, 7, 6, 0, 0, 243, 256, 33, 21, -2, 4, 47, 28, 164, 152, 155, 164, 176, 191, 195, 180, 96, 87, 64, 26, 7, 0, 43, 102, 122, 113, 9, 0, 0, 0, 0, 0, 0, 115, 127, 4, 18, 11, 4, 21, 25, 30, 23, 216, 98, 0, 24, 0, 222, 245, 17, 4, -4, -2, 33, 39, 183, 155, 142, 140, 153, 176, 174, 166, 142, 121, 103, 69, 38, 24, 10, 0, 0, 0, 0, 0, 0, 65, 22, 4, -7, 21, 3, 0, 0, 5, 0, 102, -33, 50, 113, 4, 6, 12, 66, 75, 152, 187, 106, 5, -1, 0, 0, 50, 214, 261, 241, 1, 0, 12, 172, 145, 133, 128, 135, 151, 153, 129, 58, 42, 35, 0, 0, 175, 52, 114, 17, 0, 0, 88, 89, 100, 103, 87, 0, 0, 0, 5, 109, 0, 9, 164, 104, 102, 5, 0, 0, 0, 0, 0, 0, 253, 270, 264, 152, 138, 128, 150, 162, 169, 154, 151, 137, 115, 119, 129, 122, 107, 86, 69, 53, 32, 17, 0, 44, 130, 48, 0, 0, 66, 5, 49, 98, 100, 8, 27, 0, 55, 0, 24, 51, 12, -1, 0, 2, 133, 144, 134, 106, 81, 76, 63, 47, 29, 0, 0, 117, 195, 128, -32, 0, 130, 54, 0, -35, 145, 28, 0, 149, 171, 8, 0, 19, 32, 142, 132, 118, 114, 127, 135, 127, 102, 88, 91, 94, 88, 55, 52, 41, 7, 18, 0, 60, -14, 13, 106, 87, 0, 51, 72, 70, 24, 28, 0, 67, 166, 0, 0, 17, 67, 134, 124, 112, 80, 77, 75, 60, 47, 37, 24, 10, 0, 75, 88, 122, 121, 128, 93, 0, 4, 67, 106, 91, 71, 19, 103, 171, 73, 80, 2, -3, 0, 0, -8, 15, 80, 49, 113, 110, 73, 62, 53, 40, 31, 11, 0, 0, 147, 8, 26, 56, 134, 10, 9, 7, 134, 51, 57, 55, 48, 23, 51, 147, 6, 0, 0, 10, 1, -6, 0, 7, 0, 0, -1, 59, 66, 97, 140, 147, 120, 112, 35, 18, 9, 0, 0, 0, 0, 4, 115, 158, 0, 22, 177, 91, 6, 0, 0, -3, 0, 0, 0, 0, 0, -6, 0, 0, 7, 4, 16, 98, 224, 141, 132, 99, 69, 61, 15, 0, 9, 24, 17, 0, 5, 0, 9, 73, 28, 49, 41, 0, 192, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 7, 0, 0, 8, 6, 207, 149, 142, 141, 99, 75, 37, 21, 7, 0, 0, 0, 0, 50, 115, 17, 0, 142, 0, 61, 34, 17, 35, 0, 0, 98, 0, 0, 4, 0, 12, 6, 0, 0, 2, 0, 0, 1, 7, 98, 91, 197, 145, 127, 141, 129, 127, 111, 101, 100, 69, 58, 51, 0, 0, 0, 0, 5, 44, 0, 0, 0, 74, 72, 39, 74, 19, 28, 17, 0, 0, 0, 0, 17, 0, 0, 7, 13, 0, 41, 7, 0, 8, 0, 4, -2, 111, 143, 143, 144, 132, 122, 113, 110, 101, 91, 79, 58, 48, 35, 6, 0, 0, 0, 0, 68, 50, 15, 0, 0, 73, 0, 24, 42, 73, 27, 8, 0, 8, 0, 0, 19, 0, 0, 0, 0, 48, 0, 206, 139, 142, 144, 115, 106, 111, 94, 90, 79, 70, 61, 15, 0, 0, 8, 34, 0, 13, 0, 41, 49, 0, 76, 0, 4, 63, 120, 28, 22, 0, 0, 9, 0, 0, 0, 8, 11, 0, 0, 0, 0, 0, 15, -4, 8, 219, 146, 134, 130, 129, 135, 108, 99, 81, 75, 70, 60, 43, 40, 11, 43, 29, 0, 52, 3, 0, 11, 62, 87, 102, 147, 66, 27, 5, 32, 80, 81, 48, 0, 0, 0, 0, 11, 0, -1, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 10, 19, 95, 0, 2, 14, 159, 145, 132, 121, 123, 121, 105, 93, 74, 62, 75, 66, 48, 41, 32, 23, 24, 0, 14, 8, 32, 70, 72, 13, 0, 0, 0, 0, 212, 0, 21, 0, 0, 0, 79, 37, 216, 226, 160, 149, 139, 114, 109, 108, 112, 107, 86, 70, 31, 25, 17, 7, 10, 0, 0, 54, 63, 45, 23, 12, 0, 0, 0, 21, 80, 15, 0, 6, 54, 6, 8, 13, 34, 68, 32, 26, 0, 0, 0, 10, 52, 0, 6, -1, 0, 0, 111, 114, 0, 5, 233, 162, 160, 153, 148, 137, 113, 102, 96, 100, 108, 102, 23, 26, 14, 8, 0, 0, 0, 35, 42, 27, 10, 0, 6, 0, 0, 83, 92, 27, 9, 14, 95, 102, 0, 6, -1, 4, 7, 3, 5, 58, 31, 23, 18, 0, 0, 0, 0, 3, 8, 12, 29, 17, 8, 0, 111, 121, 0, 0, 170, 165, 161, 155, 134, 121, 110, 88, 92, 97, 75, 28, 11, 12, 0, 0, 37, 24, 37, 33, 7, 0, 40, 0, 21, 0, 11, 0, -18, 0, 0, 5, 2, 1, -1, 12, 4, 0, 25, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 0, 176, 172, 165, 162, 156, 147, 140, 77, 81, 76, 26, 17, 0, 36, 37, 46, 35, 14, 0, 0, 51, 60, 0, 0, 83, 137, 52, 112, 117, 121, 32, 19, 0, 105, 169, 4, 3, 104, 35, 0, 39, 8, 0, 12, 41, 0, 0, 6, 2, 108, 180, 167, 0, 180, 174, 165, 143, 106, 94, 78, 67, 76, 62, 51, 28, 12, -17, 2, 101, 49, 0, 45, 55, 56, 0, 0, 13, 0, 99, 15, 3, 7, 2, 3, -11, 4, 102, 0, 164, 0, 0, 0, 189, 173, 168, 159, 144, 122, 109, 101, 88, 59, 55, 61, 37, 0, -19, -2, 55, 11, 44, 0, 0, 0, 107, 23, 0, 57, 76, 104, 127, 0, 41, -14, 8, 9, 8, 15, 91, 42, 46, 27, 0, 92, 30, 69, 63, 199, 199, 193, 182, 159, 152, 139, 125, 92, 87, 74, 43, 43, 35, 25, 11, 0, -13, 0, 40, 33, 0, 4, 68, 95, 124, 0, 153, 9, 10, 0, 0, 231, 13, 19, 24, -6, 9, 15, 7, 6, 2, 83, 44, 10, 20, 1, 9, 5, 0, 50, 0, 103, 159, 159, 43, 2, 0, 0, 0, 42, 65, 178, 142, 134, 117, 96, 85, 75, 54, 40, 30, 27, 10, 0, 4, 42, 23, 8, 0, 74, 0, 62, 89, 119, 29, 0, 54, 14, 4, 9, 19, 11, -1, -7, 0, 0, 0, 0, 0, 49, 84, 0, 0, 134, 156, 166, 8, 0, 0, 0, 0, 0, 48, 133, 200, 199, 175, 137, 101, 54, 47, 34, 22, 11, 7, 0, 0, 0, 3, 49, 61, 0, 0, 5, 73, 81, 132, 81, 0, 4, 145, 0, 105, 180, 0, 40, 0, 25, 0, 4, 0, 0, 25, 4, 0, 65, 0, 0, -14, 39, 211, 0, 154, 208, 192, 185, 177, 160, 98, 69, 43, 32, 25, 13, 6, 0, 4, 13, 37, 61, 35, 0, 100, 138, 86, 4, 10, 8, 20, 4, 129, 157, 27, 90, 3, 0, 38, 0, 69, 103, 152, 181, -1, 0, 55, 0, 0, 0, 188, 177, 161, 140, 115, 89, 40, 24, 14, 0, 6, 8, 0, 0, 45, 3, 0, 0, 8, 73, -1, 129, 177, 58, 11, 1, 216, 24, 19, 0, 0, 44, 27, 0, 66, 0, -16, 81, 235, 238, 188, 160, 142, 119, 106, 93, 80, 57, 45, 29, 7, 0, 0, 11, 62, 16, 4, 0, 0, 30, 0, 0, 0, 25, 36, 0, 0, 0, 175, 0, 0, 18, 0, 149, 229, 43, 15, 13, 7, 0, 46, 0, 195, 242, 10, 31, 55, 80, 220, 19, 95, 176, 165, 141, 111, 98, 83, 68, 54, 42, 33, 0, 0, -4, 13, 25, 0, 13, 28, 32, 61, 31, 0, 0, 0, 61, 108, 35, 0, 0, 36, 35, 18, 26, 1, 14, 36, 72, 64, 0, 254, 41, 42, 37, 230, 229, 100, 129, 157, 154, 149, 98, 75, 58, 28, 14, 7, 0, 0, 4, 5, 9, 30, 2, 0, 20, 52, 73, 83, 80, 9, 0, 0, 0, 73, 78, 0, 0, 176, 107, 36, 11, 7, 23, 69, 53, 4, 0, 0, 70, 65, 5, 8, 53, 85, 0, 0, 19, 154, 153, 145, 159, 149, 127, 109, 91, 47, 31, 17, 0, 0, 0, 28, 4, 10, 0, 78, 37, 12, -5, -8, 0, 0, 18, 10, 69, 90, 0, 6, 0, 55, 138, 104, 18, 32, 15, 0, 0, 0, 22, 110, 271, 262, 37, 44, 5, 35, 0, 161, 241, 0, 88, 143, 251, 151, 148, 139, 86, 52, 24, 0, 54, 87, 109, 57, 60, 15, 0, 191, 0, 0, 0, 3, 15, 0, 0, 55, 0, 20, 36, 0, 29, 0, 32, 57, 91, 117, 257, 161, 157, 147, 141, 129, 114, 101, 87, 21, 5, 0, 41, 3, 22, 52, 57, 110, 119, 155, 125, 0, 0, 0, 0, 79, 42, 9, 10, 99, -6, 2, 0, -4, 31, 214, 91, 116, 36, 80, 59, 0, 79, 98, 92, 0, 30, 45, 27, 111, 199, 2, 98, 58, 80, 161, 135, 124, 88, 81, 50, 33, 20, 10, 3, 67, 20, 6, 0, 14, 0, 17, -10, 136, 0, 60, 132, -1, -2, 208, 184, 23, 24, 50, 98, 56, 0, 0, 41, 90, 14, 0, 63, 0, 0, 100, 161, 133, 99, 66, 63, 51, 36, 21, 21, 9, 0, 16, 0, 52, 0, 44, 33, 84, 63, 9, 2, 20, 0, 100, 1, 216, 79, 3, 0, 0, 3, 120, 102, 36, 64, 66, 0, 72, 69, 160, 127, 106, 87, 75, 41, 32, 15, 9, 0, 27, 67, 80, 0, 14, 0, 92, 0, 58, 5, -2, 0, 34, 29, 24, 15, 43, 0, 129, 265, 22, 8, 0, 32, 0, 78, 159, 135, 100, 65, 56, 45, 18, 12, 5, 0, 16, 97, 121, 9, 15, 38, 0, 65, 0, 5, 8, 5, 114, 52, 66, 124, 0, 257, 247, 99, 79, 27, 10, 59, 32, 0, 8, 75, 0, 0, 92, 151, 93, 76, 59, 44, 39, 30, 24, 7, 111, -43, 14, 0, 36, 58, 0, 43, 86, 45, 31, 35, 0, -2, 309, 53, 16, 74, 5, 106, 62, 0, 0, 28, 90, 92, 90, 223, 0, 128, 106, 82, 69, 53, 30, 24, 11, 104, 30, 13, 0, 90, 175, 0, 65, 0, -11, 73, 109, 145, 23, 41, 115, 133, 139, 102, 3, -2, -9, 56, 263, 44, 67, 64, 33, 21, 11, 29, 57, 7, 257, 80, 0, 0, 110, 135, 97, 78, 46, 8, 9, 6, 34, 7, 0, 42, 25, 0, 0, 95, 89, 85, 11, 0, 0, 47, 27, 14, 0, 36, 108, 41, -7, 37, 23, 54, 53, 49, 2, 0, 0, 64, 66, 112, 245, 123, 148, 139, 124, 106, 87, 69, 53, 37, 28, 16, 5, 43, 0, 39, 121, 74, 111, 67, 53, 122, 76, 71, 0, 161, 48, 109, 0, 113, 138, 193, 211, 0, 138, 113, 96, 77, 58, 42, 28, 17, 0, 0, -10, 0, 69, 172, 0, -6, 0, 150, 57, 0, 25, 115, 38, 71, 44, 0, 55, 49, 0, 0, 220, 0, 0, 5, 153, 117, 102, 67, 33, 19, 0, 19, 217, 33, 0, -6, 0, 0, 32, 99, 0, 63, 53, 3, 5, 5, 89, 23, 33, 19, 0, 9, 126, 62, 57, 0, 105, 196, 107, 75, 60, 44, 28, 12, 0, 29, 10, 17, 0, 0, 6, 70, 87, 0, 38, 51, 0, 6, 17, 53, 112, 103, 72, 7, 45, 1, 1, 8, 0, 188, 217, 0, 104, 96, 67, 51, 36, 22, 9, 0, 25, 59, 83, 182, 31, 0, 0, 22, 29, 0, 59, 60, 95, 121, 35, 0, 116, 96, 10, 10, 22, 41, 34, 115, 133, 154, 97, 32, 85, 0, 134, 182, 208, 85, 61, 47, 35, 23, 0, 24, 125, 22, 0, 0, 68, 95, 30, 0, 17, 0, 0, 105, 149, 152, 133, 111, 36, 30, 72, 97, 36, 0, 0, 120, 75, 67, 0, 17, 3, 14, 0, 5, 0, 12, 45, 20, 0, 0, 58, 67, 0, 0, 160, 150, 52, 55, 61, 58, 0, 7, 93, 0, -2, 0, 8, 2, 11, 0, 0, 10, 40, 5, 105, 96, 53, 0, 20, 167, 164, 165, 170, 70, 73, 18, 87, 75, 19, 36, 154, 0, 7, 73, 39, 42, 0, 86, 80, 0, 48, -1, 9, 0, 3, 16, 71, 0, 78, 167, 91, 27, 0, 36, 133, 0, 39, 80, 0, 48, 26, 85, 38, 39, 44, 28, 0, 22, 0, 0, 0, 94, 125, 147, 109, 80, 0, 64, 184, 74, 158, 120, 121, 74, 7, 79, 8, 30, 7, 3, 23, 0, 63, 40, 0, 5, 0, 80, 126, 11, 8, 3, 37, 242, 121, 0, 85, 123, 143, 0, 0, 24, 35, 17, 48, 21, 93, 3, 0, 0, 0, 99, 133, 147, 106, 12, 55, 0, 128, 100, 32, 0, 58, 37, 20, 15, 12, 0, 13, 12, 18, 5, 22, 101, 0, 64, 79, 0, 0, 39, 108, 148, 193, 99, 89, 0, 32, 115, 170, 150, 4, 28, 100, 12, 8, 0, 96, 44, 20, 0, 65, 0, 19, 45, 77, 165, 10, 13, 57, 17, 11, 14, 136, 161, 0, 0, 38, 148, 8, 88, 0, 11, 32, 0, 60, 48, 0, 80, 191, 82, 95, 0, 143, 27, 12, 62, 87, 85, 37, 19, 37, 30, -3, 21, 10, 0, 0, 2, 81, 0, 107, 124, 62, 66, 147, 68, 61, 29, 134, 14, 97, 110, 91, 0, 39, 13, 0, 68, 66, 0, 61, 93, 138, 161, 0, 15, 4, 47, 71, 41, 18, 28, 67, 42, 0, 50, 125, 92, 153, 27, 19, 63, 5, 71, 0, 87, 70, 80, 141, 148, 89, 0, 52, 7, 13, 34, 56, 71, 41, 103, 9, 8, 82, 73, 5, 61, 85, 4, 31, 35, 45, 59, 90, 91, 86, -13, 0, 98, 36, 78, 123, 117, 2, 8, 111, 47, 40, 36, 10, 100, 12, 94, 95, 0, 22, 20, 82, 66, 9, 0, 0, 0, 15, 0, 14, 25, 123, 0, 2, 9, 105, -1, 67, 112, 86, 77, 0, 26, 9, -7, 77, 70, 28, 18, 34, 79, 0, 73, 123, 64, 40, 9, 27, 55, 6, 38, 12, 10, 15, 29, 0, 1, 7, 12, 0, 149, 43, 86, 73, 0, 0, 121, 58, 1, 41, 0, 3, 5, 31, 31, 26, 0, 105, 29, 21, 0, 11, 23, 66, 0, 0, 115, 110, 95, 91, 91, 69, 28, 0, 22, 4, 47, 0, 56, 5, -2, 20, 21, 0, 158, 25, 14, 0, 72, 65, -4, 63, 0, 32, 27, 7, 48, 0, 37, 3, 13, 0, 22, 17, 5, 0, 34, 42, 39, 30, 16, 0, 54, 71, 94, 106, 61, 65, 49, 14, 95, 41, 29, 30, 6, -1, 9, 17, 23, 0, 53, 0, -4, 95, 0, 0, 36, 0, 81, 79, 67, -1, 17, 4, 42, 10, 23, 22, 6, 0, 51, 48, 43, 5, 0, 0, 2, 39, 31, 0, 9, 18, 52, 80, 95, 82, 8, 110, 0, 71, 28, 9, -8, 0, 0, 55, 41, 0, 22, 45, 92, 87, 22, 0, 77, 91, 1, 14, 3, 11, 0, 0, 31, 0, 14, 23, 48, 10, 8, 14, 43, 37, 0, 7, 6, 5, 0, 13, 35, 0, 0, 112, 14, 31, 13, 15, 27, 67, 79, 5, 0, 2, 0, 0, 28, 124, 58, 10, 35, 41, 15, 8, 31, 34, 16, 40, 56, 4, 0, 13, 86, 107, 0, 0, 0, 6, 12, 13, 0, 19, -2, -4, 0, 3, 49, 120, 131, 83, 57, 0, 0, 0, 0, 0, 4, 30, 29, 13, 8, 5, 4, 0, 0, 0, 103, 141, 105, 36, 0, 6, 0, 0, 39, -4, 4, 0, 32, 106, 43, 20, 58, 0, 5, 56, 52, 21, 8, 0, 0, 0, 9, 4, 34, 34, 120, 132, 91, 59, 11, 0, 51, 47, 11, 22, 25, 35, 72, 72, 0, 8, 8, 90, 68, 0, -1, 11, 80, 0, 18, 30, 25, 43, 73, 67, 0, 8, 25, 80, 61, 0, 41, 0, 48, 0, 0, 0, 81, 74, 29, 12, 0, 6, 38, 42, 39, 56, 67, 62, 58, 77, 86, 35, 0, 0, 5, 19, -3, 0, 24, 9, 52, 34, 0, 0, 20, 108, 114, 0, -8, 2, 70, 23, 9, 10, 12, 99, 70, 73, 109, 19, 0, 2, 104, 0, 52, 91, 102, 103, 104, 104, 60, 67, 57, 32, 0, 1, 64, 0, 3, 22, 117, 125, 128, 71, 0, 4, 0, 99, 101, 43, 29, 16, 0, 0, 63, 0, 0, 15, 80, 11, 58, 111, 119, 72, 3, 0, 82, 100, 6, 0, 8, 55, 114, 41, 10, 0, 83, 52, 16, 55, 89, 96, 83, 66, 42, 29, 0, 0, 5, 116, 103, 44, 0, 36, 71, 73, 65, 14, 24, 16, 0, 0, 81, 86, 63, 53, 46, 41, 22, 0, 0, 8, 27, 103, 0, 4, 19, 94, 99, 56, 88, 76, 75, 124, 38, 31, 104, 12, 92, 15, 0, 26, 86, 65, 61, -8, 74, 76, 85, 8, 89, 81, -10, 0, 0, 105, 70, 47, 77, 8, 2, 0, 98, 99, 64, 0, 37, 7, 1, 68, 23, 11, 9, 57, 16, 33, 38, -2, 45, 58, 61, 5, 8, 0, 21, 49, -1, 59, 19, 17, 14, 13, 53, 20, 45, 58, 0, 35, 55, 46, 32, 25, 24, 67, 63, 0, 36, 30, 28, 23, 15, 10, 4, 31, 71, 118, 37, 13, 20, 95, -5, 144, 34, 29, 37, 30, 28, 20, 38, 74, -11, -6, 143, 0, 37, 29, 32, 29, 54, 31, 34, 21, 11, 52, 35, 43, 37, 51, 31, 30, 18, 0, 74, 65, 12, 25, 8, 127, 43, 33, 57, 60, 4, 8, 4, 68, 56, 14, 34, 27, 24, 21, 55, 71, 103, 95, 49, 82, 93, 17, 133, 0, 77, 0, 5, 53, 119, 84, 59, 15, 6, 119, -5, 63, 113, 47, 69, 193, 64, 61, 5, 89, 78, 125, 22, 37, 112, 108, 97, -4, 41, 53, 58, 68, 61, 161, 0, 17, 87, 32, 50, 64, 12, 17, 50, 164, 25, 38, 5, 15, 89, 104, 90, 76, 0, 26, 132, 28, 7, 144, 41, 110, 60, 42, 9, 50, 30, 129, 127, 124, -4, 22, 28, 103, 6, 71, 60, 24, 22, 69, 94, 104, 98, 87, 73, 75, -2, 0, 7, 106, 26, 63, 22, 33, -28, 0, 12, 226, 230, 42, 62, 8, 42, 24, -7, 9, 80, 90, 38, -34, 114, 214, 58, 71, 43, 84, 84, 54, 67, 200, 67, 32, 24, 20, 49, 51, 50, 57, 6, 18, 4, 84, 14, 55, 132, 189, 196, 70, 51, 46, 17, 24, 60, 66, 53, 3, 0, 103, 3, 58, 5, 145, 43, 55, 59, 72, 40, 32, 21, 84, 125, 70, 145, 152, 162, 91, 28, 2, 35, 21, 0, 76, 48, 0, 72, 116, 137, 55, 57, 57, 47, 7, 17, -25, -3, 72, 42, 38, 77, 110, 116, 124, 18, 54, -24, 19, -5, 13, 6, 27, 10, 2, 0, 0, 118, 136, 60, 58, 99, 21, 26, 38, 143, -39, 59, 62, 88, 103, 67, 101, 95, 3, 26, 37, 108, 67, 27, 11, 0, 91, 111, 59, -15, 19, 41, 79, 1, 56, 76, 79, 88, 127, 46, -14, 18, 25, 1, 57, 98, 25, 44, 91, 74, 110, 62, 51, 37, 24, 61, 14, 0, 43, 56, 99, 158, 32, 38, 64, 44, -64, 41, 0, 49, 102, 116, -20, 0, 25, -21, 27, 33, 63, 134, 21, -31, 22, 60, 54, 5, 72, 14, 26, 40, 59, 68, 93, 103, 111, -2, -5, -14, 52, 41, 32, 0, 57, 106, 4, 96, 259, 22, 35, 72, 82, 118, 128, 7, 22, 92, 80, 18, 14, 61, 93, 130, 27, 41, 0, 89, 110, 17, 136, 21, 8, -25, 0, 12, 25, 32, 109, 0, 78, 23, 7, 261, 2, 16, 83, 13, 30, 82, 19, 24, 9, 169, -9, 80, 15, 97, 31, 80, 9, 226, 65, 29, 99, 0, 9, 10, 18, 20, 103, 65, 32, 37, 50, 45, 11, 8, 137, 17, 87, 101, 44, -51, 19, 4, 192, 23, 3, 115, 34, -4, 85, 5, 9, 24, 13, 18, 90, 197, 204, 60, 109, 18, 72, 8, 0, 0, 0, 169, 22, 60, 60, 16, 31, 21, -3, -3, 112, 21, 0, 8, 71, -2, 0, 10, 30, -33, 0, 0, 0, 74, 29, 25, 15, 0, 13, 22, -20, -2, 0, 0, 33, 30, 11, 7, 1, 208, 108, 109, -2, -3, 244, -63, 37, 0, 20, 0, 1, 13, 4, 0, 43, 4, 4, 227, 13, -24, 48, 33, 3, 0, 11, 6, 7, 45, 0, 0, 1, -43, 34, 12, 111, 25, 20, -1, 263, 196, -1, 8, 7, -3, 9, -18, -24, -39, 38, 59, 35, 111, 17, 19, 124, 76, 37, 3, 46, 17, 4, 0, 15, 65, 68, 61, 46, 21, 52, -4, 0, 15, 1, -20, 61, 18, 0, 3, 330, 47, 6, 0, 0, 0, 3, 9, 18, 0, 0, 0, 75, 7, 9, 100, -6, 13, 61, 3, 12, 20, 40, 61, 176, 0, 0, -1, -8, -7, -10, 74, 97, 97, -3, -12, 167, 179, 297, 33, 16, 30, 12, -12, 0, -4, 106, -2, -16, 4, -8, 5, 10, 53, -16, -2, 0, -5, -2, 97, -3, -4, 23, -13, 27, 41, 49, 157, 11, 123, 98, 119, 3, 109, 50, 8, 13, 46, 52, 11, 179, 166, 0, 126, 0, -3, 3, 21, 168, -4, 0, 34, 25, 18, 0, -18, -123, 4, 67, 91, 102, 4, 129, 5, 10, 13, -33, -62, 24, 7, 0, 0, 76, 135, -27, 20, 0, 172, 272, 58, 0, 29, 33, 56, 87, 50, 20, -1, 2, 23, -5, 5, 113, 4, -4, 72, -4, 62, 105, 92, 62, 54, 18, -7, 64, 4, 29, 10, 40, 83, 30, 123, 0, 42, 51, 23, -4, 13, 83, 91, 56, 89, 81, 33, 41, 16, 0, 78, 54, 11, 29, 0, 316, 11, -3, 27, 52, 94, 95, 57, 66, -3, 70, 56, 43, 38, -8, 91, 9, 5, 0, -1, 64, 64, 58, 18, 8, 36, 35, 0, 32, 59, 46, 41, 69, 24, 30, 28, 22, 8, 71, 41, 0, 61, 82, 72, 49, 45, 39, 55, 242, -20, 13, 24, 41, 44, 62, 61, 58, 28, 5, 1, 0, 26, 22, 0, 22, 98, 54, 51, 50, 45, 19, 21, 69, 48, 29, 25, 22, 0, 0, -1, 40, 14, 95, 91, 107, 29, -14, 15, 78, -1, 78, 65, 48, 0, 96, 127, 124, 4, 65, 70, 60, -3, 85, 94, 8, 3, 90, 100, 155, 79, 70, 42, 83, 69, -7, -5, 0, 0, 0, 16, 0, 0, 4, 0, 3, 0, 95, 27, 33, 95, 80, -10, -7, -12, 25, -4, 0, 0, -5, 15, 78, 76, 208, 112, 32, 32, 40, -9, 46, 37, 99, -3, -5, 0, 73, 79, 53, 81, 0, -13, 36, 36, 39, 62, 41, -15, 97, 110, 0, 14, 121, 29, 83, 68, 52, 34, 19, -13, -34, 102, 103, 116, 10, 23, 8, 112, 0, 61, 51, 19, -15, -40, 70, 116, 15, 5, 39, 81, 0, 62, 55, 34, 37, 13, 11, 55, 25, 37, 70, 133, 33, 21, 21, 15, 9, 37, 56, 83, 53, 73, 78, 89, 61, 23, -27, -46, 50, 29, 32, 21, 41, 44, 4, 46, 80, -28, -33, 34, 16, 55, 96, 51, 26, 27, 37, 40, 307, -9, 125, -30, -22, 91, 35, 112, 49, 83, 57, 31, 17, 16, 37, -2, 161, 98, 50, 96, 62, 17, 20, 95, 24, -2, 40, 11, -38, 178, 117, -44, 12, 13, 73, 82, 27, 14, 109, 72, 31, 20, 21, 61, 82, 72, 162, 103, 98, 0, 44, 108, 75, 20, 76, 28, 1, 58, 70, 76, 334, 85, 116, 93, 128, 116, 10, 54, 126, 109, 103, 40, 30, -2, 89, 40, 70, 109, 75, 17, 155, 107, 95, 27, 8, 8, 24, 16, 11, 86, 69, 73, 153, 136, 123, 225, 183, 169, 158, 148, 97, 112, 100, 0, 21, -4, 67, 48, 34, 15, 16, 86, 174, 165, -12, 146, 4, 181, 26, 3, 2, -2, 9, -6, 0, 78, 108, 126, 107, 87, 27, 37, 29, 8, 12, 9, -2, 23, 75, 5, 0, 5, 102, 3, 205, 214, 45, 3, 57, 1, 35, 21, 11, 7, 75, 0, 12, 118, 119, 234, 179, 146, 37, 22, 209, 211, 205, 36, 183, 110, 42, 99, 104, 21, 18, 3, 71, 9, 6, 0, 22, 81, 130, 170, 124, 168, 141, 74, 73, 83, 176, 162, 201, 51, 42, 20, 42, 54, 47, 41, 14, 1, 237, 191, 69, 25, 165, 135, 24, 70, 205, 101, 70, 72, 0, 105, 56, 46, 33, 33, 0, -16, 28, 129, 228, 214, 178, 149, 12, 214, 148, 72, 18, 90, 78, 213, 90, 61, 52, 41, 0, 3, 31, 34, 25, 195, 83, 76, 101, 51, 32, 0, -2, 18, 37, 48, 73, 180, -81, 238, 229, 141, 29, 31, 38, 92, 228, 211, 14, 85, 107, 101, 51, 44, 44, 0, 15, 21, 25, 78, 116, 237, 175, 125, 10, 95, 33, 55, 98, 151, 106, 149, 54, 58, -6, 0, -11, 12, 27, 80, 245, 229, 161, 92, 61, 101, 79, 65, 47, 10, 10, 182, 129, 55, 210, 92, 63, 0, 7, 63, 90, 300, 264, 70, 116, 25, 0, 8, 40, 88, 176, 105, 68, 109, -3, 105, 131, 115, 103, 114, 99, 58, 60, 65, 84, 77, 0, 4, 5, 232, 217, 53, 64, 72, 123, 54, 35, 1, 100, 97, 144, 129, 290, -4, 0, 5, 0, 1, 17, 202, 47, 0, 121, 108, 65, 70, 70, 116, 46, 1, 194, 93, 93, 89, 210, 201, 180, 149, 55, -12, 17, 18, 135, 112, 205, 59, 84, 161, 78, 222, 41, 23, 19, 0, 0, 12, 28, 33, 118, 74, 133, 162, 199, 171, 166, 148, 134, 294, 4, -2, 0, 6, 7, 23, 86, 76, 9, -6, 70, 173, 55, 60, 48, 40, 15, 22, 126, 70, 26, 50, 37, 61, 87, 79, 96, 33, 2, 1, -2, 68, 158, 194, 40, 38, 1, 0, 7, 12, 23, 0, 1, 72, 84, 91, 18, -4, -6, 56, 59, 0, 0, 0, 20, 18, 96, 119, 115, 60, 2, 152, 84, 96, 26, 4, -1, 52, 90, 1, 0, -1, 19, 33, 21, 32, 103, 111, 0, 99, -3, 15, 20, 22, 20, 16, 31, 5, 18, 88, 97, 101, 105, 102, 91, 0, 157, 190, 176, 248, 81, 1, 0, 1, 0, 3, 8, 12, 18, 28, 37, 0, 12, 18, 51, 110, 3, 7, 0, 185, 162, 162, 164, 0, 26, 25, 109, 90, -6, 18, 114, 70, 0, 0, 181, 189, 158, 158, 155, 9, 9, 5, 16, 68, 156, 116, 14, 72, -5, 181, 189, 152, 151, 99, 42, 58, 35, 0, 9, 8, 19, 16, 19, 108, 132, 106, 120, 48, 0, 227, 203, 144, 20, 12, 12, 18, 18, 17, 15, 26, 111, 23, 39, 0, 180, 143, 142, 136, 20, 10, 17, 23, 16, 21, 0, 101, 72, 15, 23, 0, 8, 34, 0, 0, 176, 98, 98, 31, 17, 22, 20, 70, 7, 163, -11, 22, 0, 18, 0, 165, 169, 170, 131, 131, 250, 35, 20, 65, 128, 72, 65, 151, 184, 11, 15, 2, -1, 171, 123, 177, 101, 33, 6, 3, 22, 30, 29, 10, 147, 16, 5, -2, 0, 148, 148, 168, 114, 48, 123, 107, 7, 12, 25, 32, 55, 6, 43, 143, 106, 88, 125, 33, -3, -3, -1, 40, 40, 33, 46, 108, 58, 35, 98, 44, 30, 157, 96, 86, 30, 5, 39, 46, 32, 29, 0, -2, 97, 26, 2, -34, 130, 149, 0, -3, 20, 45, 38, 37, 23, 125, 48, 112, 43, 0, 38, 44, 45, 50, 0, 105, 95, 14, 205, 68, 235, 21, -1, -3, 3, 21, 118, 106, 72, 20, 153, 22, 199, 199, 199, 187, 62, 76, 12, 27, 82, 44, 23, 95, 11, 82, 19, 0, 142, 89, 40, 128, 35, -1, -2, 173, 163, 14, 121, 25, 52, 57, 78, 110, 82, 127, -38, 202, 157, 85, 59, 7, 108, -2, 118, 83, 69, 59, 122, 75, 32, 93, 120, 34, 4, 88, 224, 20, 80, 79, 23, 7, 81, 53, 54, -15, 0, 15, 237, 225, 208, -8, 35, 119, 194, 195, 54, 44, 117, 105, 99, -3, 7, 37, 67, 221, 242, 99, -17, 124, 79, 55, 37, 1, 46, 252, 189, 25, 191, 219, 165, -1, 73, 164, 28, 25, 104, 3, 41, 60, 98, 40, 93, 91, 206, 53, 157, 148, 129, -13, 93, 27, 20, 17, 2, 16, 12, 226, 104, 77, 54, 39, 45, 38, 188, 140, 37, -7, 168, 181, 146, 134, 4, 10, 256, 222, 141, 111, 82, 74, 39, 39, 99, 197, 231, 228, 1, 58, 57, 54, 54, 107, 109, 113, 105, 97, 73, 109, 115, 28, 4, 243, 210, 204, 157, 59, 114, 39, 41, 94, 226, 187, 39, 52, 55, 49, 95, 94, 217, 71, 58, 6, 4, 4, 199, 119, 36, 38, 35, 98, 192, 219, 218, 153, -20, 28, 124, 47, 106, 96, 212, 191, 133, 57, 83, 105, 30, 10, 230, 224, 193, 1, 120, 35, 213, 42, 31, 38, 47, 139, 83, 83, 62, 96, 40, 36, 0, 0, 5, 215, 204, 193, 180, 95, 98, 34, 31, 83, 194, 103, 48, 35, 38, 23, 203, 184, 119, 48, 1, 94, 39, 27, 163, 125, 85, 295, 288, 123, 45, 15, 53, 8, 0, 44, 191, 53, 8, 1, 88, 62, 76, 89, 214, 106, 52, 22, 1, 2, 61, 12, 37, 37, -3, 107, 140, 86, 104, 78, 60, 179, 185, 203, 259, 208, 21, 21, 53, 0, 170, 163, 0, 38, 33, 44, 16, 24, 55, 58, 167, 187, 52, 0, 30, 7, -10, 91, 239, 67, 22, 9, 4, 0, 38, 19, 26, 31, 38, 7, 17, 66, 158, 141, 12, 8, 109, 103, 27, 109, 60, 46, 38, 154, 61, 163, 204, 222, 226, 33, -4, 11, 0, 0, 84, 24, 7, 37, -3, 43, 190, 187, 122, -3, -4, 0, 30, 28, 92, 67, 162, 0, 41, 35, 16, 0, 1, 89, 80, 86, -15, 19, 16, 17, 200, 22, 39, 47, 44, 178, 167, 92, 46, 30, 15, 84, 12, 82, 101, 5, 8, 97, 68, 43, 183, 17, 45, 49, 40, 158, 98, 41, 90, 106, 199, 136, 143, 82, 74, 48, 4, 5, 6, 3, 12, 18, 43, 105, 5, 192, 32, 79, -18, -14, 17, 0, 29, 139, 117, 111, 75, 7, 18, 142, 5, 31, 137, 0, 0, 3, 5, 0, 39, 64, 21, 44, 72, 51, 26, 6, 23, 78, 0, 133, 0, 0, 47, 176, 93, 41, 52, 24, 12, 85, -3, 31, 26, 9, 0, 44, 74, 66, 48, 76, 37, 215, 115, 0, -2, 57, 142, 33, 39, 106, 62, 124, 19, 35, 170, 1, 0, 164, 43, 96, 29, 46, 11, 80, 49, 6, 25, 104, 112, 3, 3, 35, 151, 48, 58, 116, 19, 6, 45, 4, 18, 50, 41, 32, 0, 6, 74, 100, 0, 88, 135, 186, 25, 71, 43, 5, 61, 19, 27, 66, 61, 66, 132, 96, 2, 46, 81, 88, 45, 57, 123, 36, 53, 110, 11, 32, -2, 45, 33, 67, 47, -10, 47, 56, 39, -9, 177, 207, 100, 64, 91, 56, 64, 24, -7, 68, 84, 150, 50, 38, 19, -2, 66, 107, 123, 67, 38, 40, 58, -12, 39, 39, 36, 56, 76, 84, 93, 117, 53, 28, 45, 76, 50, 59, 91, 112, 23, -28, 53, 45, 69, 1, 3, 48, 55, 8, 59, 2, 93, 103, 76, 3, 61, 43, 0, 57, 41, 53, 61, 4, -5, 110, 21, 6, 56, 55, 47, 72, 51, 47, 61, 0, 13, 32, 40, 17, 11, 0, 41, 57, 61, 1, 157, 40, 49, 77, 80, 0, 10, 27, 90, 140, 118, 0, 25, 44, 3, 14, 53, 36, 13, 21, 11, 82, 217, 0, 103, 29, 12, 8, 218, 53, 2, 62, 136, 64, 72, 56, -26, -4, 54, 1, 100, 106, 40, 48, 56, 0, 91, 138, -47, 36, 104, 0, 10, 1, 0, 0, 35, 59, 69, 20, -14, 21, 78, 11, -1, 22, 33, 3, 34, 28, 31, 130, 8, 9, 20, 4, 38, 88, 0, -3, 17, -23, 190, 3, 134, 106, 264, 3, 15, 13, 174, 24, 13, 243, 84, 39, 212, 11, 5, 13, -13, 9, -1, 14, 45, 73, 99, 75, 26, 40, -8, 7, 117, 156, 26, 29, 85, 114, 33, 13, 12, 134, -10, 74, 39, -13, 87, 4, 94, 20, 57, 66, 121, 0, 0, 214, 53, 279, 82, 50, 0, 8, 192, 149, 12, 17, 52, 175, 96, 0, 21, -5, 2, -2, 52, 52, 42, 0, 2, 14, 222, 47, 174, 134, 13, 68, 267, 95, 196, 51, -4, 46, 50, 6, 194, 0, 54, 18, 24, 50, 48, 75, 69, 53, 54, 188, 177, 46, 46, 74, -3, 167, 61, 27, 13, 59, 219, 138, 213, 0, 61, 92, 15, 202, 138, 39, 196, 37, 14, 10, 61, 19, 55, -2, 28, 52, 220, 176, 22, 27, 2, 63, 53, 9, 12, 33, 195, 180, 90, -3, -5, 53, 47, 66, 172, -2, 4, 12, 148, 25, 0, 238, 19, 49, 144, -4, -6, 1, 16, 166, -13, 0, -37, 47, 87, 84, 70, 31, 4, 0, 11, 33, -4, 124, 33, -1, 148, 68, 29, 154, 0, 15, 29, 41, 143, 14, -2, 108, -1, 5, 9, 13, 26, 40, 21, 66, 77, 49, 49, 72, 46, 42, 38, 13, -34, 58, 12, 67, 2, 147, 34, 29, 0, -32, 3, 0, -6, 168, 45, 1, 42, 6, 71, 83, 40, 130, 7, 10, 0, -67, 25, 112, 91, 157, 2, 57, 159, 88, 84, 83, 0, 67, 21, 0, 149, 87, 125, 81, 41, 30, 10, -14, -1, 1, 101, 14, 22, 137, 182, 81, 116, 13, 80, 38, 47, 8, 134, 172, 192, 60, 107, 104, 50, 71, -10, -13, -10, 0, 73, 3, 101, -4, -4, -26, 85, -1, 132, 142, 61, 80, 15, -4, -5, -11, 1, 86, 136, 14, -7, 66, 69, 85, 111, 79, 12, 43, 64, 154, 185, 82, 120, 111, 106, 85, 78, 18, 31, 79, -1, 14, 20, 2, 17, 6, 0, -9, -12, 165, 175, 112, 29, 105, -11, 0, 15, 59, 0, 21, 114, 116, 67, 23, 4, -2, -14, 185, 185, 17, 39, 0, 4, 61, 25, 11, 124, 110, 16, 2, -3, 4, 186, 0, 35, 115, 105, 3, 1, 0, 10, 13, 77, -1, 8, -1, 8, 90, 92, 118, 114, 105, 0, 3, 161, 163, 176, 196, 18, 46, 105, 0, 0, 0, 4, 0, 75, 88, 85, 127, 30, 167, 167, 18, 116, 110, 108, 70, 15, 11, 0, 177, 20, 40, -36, -2, 42, 34, 72, 8, 0, 0, 188, 192, 92, 0, 0, 157, 198, 5, 104, 2, 0, 124, 143, 49, 70, 70, 69, 155, 14, -70, 0, 30, 107, 64, 57, 1, 2, 2, 1, 121, 60, 0, 67, 40, 0, 223, 223, 68, 57, 51, 6, 41, 0, 3, 0, 80, 77, 79, 2, 7, 61, 52, 40, 46, 87, 3, 0, 82, 49, 26, 59, 3, 1, 98, 46, 27, 21, 102, 2, 5, 17, -3, 23, 110, 108, -1, 0, 123, 108, 93, 166, 27, 26, 19, 116, 114, -1, 66, 200, 176, 6, 17, 18, 38, 7, 130, 0, 208, 2, 31, 19, 35, 25, -28, 131, -6, 0, 105, 115, 127, 111, 12, 8, 6, 104, 18, 12, 60, 19, 8, 127, 143, 0, 57, 0, 23, 33, 164, 10, 107, 115, 148, 229, 8, 4, 0, 0, -3, -19, 78, 21, 7, 114, 237, 236, 238, 244, 244, 260, 18, 8, 41, 2, 28, 31, 45, 30, 1, 11, 6, 0, 0, 109, 17, 16, 1, 14, 12, 17, 257, 88, 30, 14, 0, 4, 10, -7, 24, 85, 21, -12, -2, 37, -4, -1, 22, 37, 274, 16, 0, 48, -4, 4, 8, 2, 42, 78, 8, 289, 290, 38, 5, 11, 19, 39, 66, 63, 1, 0, 10, 38, 6, 22, 1, 5, 16, 9, 8, 7, 15, 40, 22, 43, 17, 8, 24, 25, 45, 25, 4, 17, 12, 30, 36, 37, 4, 7, 32, 36, 29, 34, 294, -17, 46, 0, 325, 268, 265, 17, 4, 4, -7, 333, 18, 1, -14, 5, 11, 51, 124, 2, 136, 142, 246, -38, 153, 0, 260, 285, 76, 8, 46, 48, 30, 80, 212, 256, -78, 260, 260, 69, 299, 77, 28, 49, 279, 319, 318, 318, -55, 10, 228, 264, 279, 61, 266, 285, 115, 231, 276, 297, 345, 49, 72, 31, 3, 197, 211, -4, 231, 238, 307, 354, 368, 44, -18, 62, 83, -42, 214, 346, 22, 30, 204, 363, 401, 129, -30, 56, 228, 403, 233, -20, 0, 41, 109, 233, 374, 40, 23, 74, 382, 51, 387, 390, 49, 243, 135, 34, 393, 46, 42, 236, 91, 91, -13, 397, 397, -2, 72, 73, 207, 148, 222, 198, 134, 13, 134, 131, 39, 132, 143, 141, 60, 239, 110, 57, 32, 264, 260, 228, 98, -5, 12, 250, 244, 234, -2, 188, 282, 233, -80, 17, 38, -30, 294, 238, -16, 16, 30, 94, 176, 291, 305, 207, -12, 216, 194, 191, 144, 133, 114, 175, 318, 338, 143, 115, 286, 293, 116, 107, 91, 28, 104, 222, 257, 52, 4, 112, 104, 88, 342, 92, 85, 79, 35, 59, 48, 45, 301, 57, 17, 275, -3, 19, 290, 53, 9, 1, 8, 0, 321, 22, -25, 162, 0, 1, 28, -53, 194, 18, 11, 1, 2, 11, 8, 9, 12, 344, 8, -2, 142, 28, 19, 17, 0, 25, 166, -37, 19, 85, -11, 163, -2, -2, 2, 26, 301, 36, 32, -2, 7, 9, 9, -2, 21, 75, 53, 8, 21, -8, 0, 3, 8, -1, 66, 64, 88, 55, 40, 4, 34, 75, -8, 17, 41, 56, 222, -8, 2, 40, 2, 26, 50, -6, 21, 6, -18, 92, 77, 10, 3, 0, 31, 11, 88, 25, -4, 9, 29, 31, 1, 126, 0, 8, 0, 43, 15, 94, 0, 7, 8, 45, -4, -9, 0, 43, 12, 23, 62, 50, 53, 148, 0, 0, 0, 363, 115, 155, 29, -7, 8, 64, -1, 53, 271, 0, 4, 3, 60, 38, 31, 71, -3, 0, 63, 14, 1, 66, 64, 53, 0, 0, 7, 181, 52, 27, 64, 134, 25, 1, 25, 24, 11, 7, 140, 16, 30, 22, 0, 15, 27, 50, 121, 82, 5, 30, 0, 5, 8, 0, 7, 108, 22, 43, 39, 94, 55, 86, 68, 57, 46, 174, 5, 10, 53, 44, 22, 0, 5, 21, 170, 34, 42, 94, 223, 50, -27, -5, 121, 16, 78, 78, 59, 54, 132, 22, 4, 50, 52, 0, 9, 62, 40, 26, 49, 64, 87, 80, 46, 1, 178, 13, 76, 57, 48, 45, 40, 51, 109, 197, 41, 31, 18, 3, 55, -4, 194, 0, 183, 65, 55, 46, 20, 43, 65, 26, 0, 26, 63, 37, 15, 5, 69, 55, 92, 5, 5, -31, -36, 194, 75, 48, 39, 30, 28, 173, 25, 81, 59, 35, 4, 182, 146, 5, 15, 11, 5, 119, 51, 50, 34, 22, 6, 4, 26, 14, 8, 46, 69, 41, 131, 62, 85, 108, 12, -3, 0, 25, 176, 22, -7, 59, 14, 18, 105, 85, 8, 80, 122, 66, 12, 10, 9, 43, 5, 12, 130, 37, 6, -6, 30, 127, 2, 139, 121, 133, 46, -34, 188, 173, 95, 32, 26, 9, -22, 21, 95, -2, 144, 256, 62, 27, 38, 54, 4, 147, 40, 10, -8, 183, 183, 20, 9, -3, 37, 212, 165, 27, 96, 91, 92, 4, 82, 60, 39, 216, 198, 190, 59, 88, 5, 111, 79, 24, 110, 53, 81, 0, -3, 66, 43, 33, 188, 178, 79, 4, 90, 58, 24, 80, 44, 40, 157, 6, 30, 45, 13, 181, 170, 153, 5, 100, 214, 84, 235, 30, 0, 50, 57, 64, 32, 39, 137, 107, 99, -70, 190, 39, 111, 55, 25, -71, 186, 109, 17, 211, -65, 204, 74, 38, 91, -74, 175, 126, 157, 85, 56, 27, 71, 57, 80, 90, 53, 137, 16, -65, 105, -62, -55, 126, 182, 17, 18, 51, 30, 9, -37, 143, 71, 7, 66, 6, 104, 52, 44, 92, 98, -52, -44, 138, 172, -23, 51, 31, 32, 45, 31, -46, -28, -28, 34, 32, -41, -20, 40, -8, 8, -19, -23, 48, 49, -9, 64, 186, -33, -8, -5, 106, 0, 39, -13, -20, 0, 99, -1, 0, 39, 2, 51, 48, 49, -16, -1, 8, 5, -3, 174, -11, -6, 213, 33, 62, 0, -10, 49, 63, 218, 18, -11, 74, 4, 9, 47, 87, 163, 64, 3, 3, 82, 75, 25, 26, 53, 71, 63, 47, 52, 2, 22, 60, 77, 22, 52, 10, 48, 73, 73, 31, 73, 52, 48, 43, 150, 24, 79, 74, 73, 15, 33, 195, 35, 17, 0, 0, 21, 18, 18, 85, 20, 13, 4, 45, 142, 47, 114, 18, 134, 15, 0, 27, 138, 3, -1, 170, 2, 6, 9, 164, 183, 60, 5, 0, 13, 0, 183, 80, 61, -5, 204, 0, 78, 0, 79, 71, 69, 57, 207, 159, 136, 73, 61, 0, 85, 71, 66, 85, 120, 118, 206, 124, 146, 86, 8, -34, 178, 187, 123, 3, 71, 38, 129, 195, 201, 78, -22, 0, 186, 209, 212, 221, 36, -15, 0, 192, 214, 210, 50, 71, 9, 0, 25, 68, -1, 54, 0, 26, 219, 135, 0, 15, 59, 146, 0, 14, 46, 62, 141, 0, 0, 9, 21, 49, 144, 131, 10, 7, 48, -1, 68, 7, 160, 0, 67, -13, 22, 24, 4, 40, 0, 33, 84, 0, 53, 125, 18, 22, 22, 35, 60, 20, 28, 21, 26, 53, 0, 31, 10, 58, 11, 56, 17, 14, 77, 77, 105, 86, 39, 18, 4, 91, 7, 74, 87, 104, 159, -1, 11, 75, -46, 94, 86, 75, 29, 82, 276, 41, 47, -7, 9, 8, 39, 248, 23, 28, 0, 0, 13, -21, 93, 92, -2, -1, 138, -20, 77, 152, 41, 85, 17, 58, 61, 145, 55, 41, -2, -6, 172, 67, 94, 9, 25, 73, -4, 70, 87, 0, -6, -37, -11, 0, -8, -6, 5, 0, -1, 134, 12, 35, -23, 102, 45, -9, 238, 42, 14, 33, 21, 30, -146, 7, 20, 3, 10, 8, 1, -8, -7, 48, 82, 71, 116, 99, 106, 30, -28, 100, 10, 0, 25, 13, 15, 16, 7, 23, 42, 52, 33, -45, 22, -4, -20, 1, 0, -3, 0, 2, 0, 0, -41, -23, 0, 0, 0, 0, 0, 6, 2, 0, 75, -23, -53, 0, 168, 130, 58, 8, 31, 67, 24, 0, 18, 38, 0, 167, -77, 79, 2, -14, -3, -49, 29, 63, 28, 28, -29, 49, 27, 76, 45, 0, 0, 22, 21, 7, 5, 0, 12, 0, 2, 38, 8, 0, 5, 0, -6, 0, 0, -1, 0, 0, 4, 52, -10, 54, 0, 53, 0, 0, 44, -16, 8, 8, 0, 20, 25, 0, 60, 16, 7, 22, 107, 50, 0, 23, 8, 0, 0, 125, 62, 6, 3, -13, 66, 57, 41, 1, 3, 10, 6, 9, 48, 27, 61, 67, 12, 10, 2, 12, 28, 8, 7, 73, 56, 27, 29, 39, 104, 88, 48, 67, 0, 86, 2, 23, 40, 16, 9, 0, 51, 25, 8, 33, 34, 130, 37, 5, 33, 75, 0, 0, 0, 61, 0, 43, 124, 22, 5, 36, 41, 36, 20, 38, 27, 36, 12, 25, 41, 0, 92, 166, 84, 2, 152, -6, 7, -2, 167, 119, 44, 93, -1, 0, 0, 0, 55, 68, 19, 10, 61, 172, 0, 0, 12, -10, 3, 183, 58, 35, 0, 8, 58, 73, 152, 50, 39, 50, 6, -5, 77, 73, 15, 129, 143, 91, -13, 0, 110, 8, 0, 47, 79, 64, 32, -3, -22, 41, 0, 5, 86, -16, 0, 71, 26, 57, 40, 89, 13, 102, 3, 32, 0, 117, 13, 12, 30, 25, 51, 69, 53, 2, -3, 100, 52, 54, -2, 3, -1, 43, 27, 6, 89, 108, 26, 5, 0, 10, 24, 124, 60, 30, 0, 54, 79, 54, -1, 7, 149, 91, 73, 108, 83, 31, 28, 4, 0, 154, 152, 62, 12, 15, 28, 0, 133, 148, 138, 90, -13, 0, 99, 39, 67, 9, 18, 11, 9, 45, 50, 71, 51, 68, -6, 103, 129, 31, 65, 2, -1, 0, 47, 67, 46, 29, 8, 60, -1, 107, 39, 0, 54, 0, 0, 61, 0, 0, 0, 18, 58, 26, 0, 45, 55, 18, 12, 13, 14, 15, 55, 92, -1, 54, 5, 23, 21, -4, 0, 20, 6, 63, -7, -12, -15, -5, 0, 5, 29, 69, 0, 0, 2, 0, 42, 32, 4, 12, 0, 16, -10, 22, 53, 5, -8, -8, -4, -17, 3, 116, 8, 0, 3, 0, 54, 24, 5, 56, 0, 3, 13, 14, 44, 35, 2, 8, 24, -1, 71, 17, 71, 68, 13, 24, -6, -16, 0, 61, 40, 19, 88, 76, 86, 67, 52, 46, 42, 41, 5, 95, 48, 0, 123, 74, 83, 108, 60, 64, 88, 97, 92, 44, 67, 108, 0, 83, 29, 10, 97, 176, 148, 113, -11, -34, 134, 146, 15, -20, -14, 71, 79, 75, 143, 73, 61, 68, -9, 24, 92, 198, 20, -2, 32, 68, 191, 12, 27, -8, -5, 11, 65, 168, 179, 19, 60, 77, -6, -7, 0, 0, -1, 90, 0, 0, 14, 17, 22, 0, 0, 5, 95, 6, 5, -1, -5, 63, 0, 170, 6, 110, 108, 0, 9, 0, -14, 50, 101, 0, -26, 7, 13, 9, 17, -11, 104, 191, 13, 21, 0, 22, 117, 141, 158, 1, -36, 128, -47, 26, 66, 114, -67, -81, 34, 87, 94, -78, -103, 105, -2, 92, 51, -77, 93, 97, 37, 47, 112, 73, 67, 46, 78, 57, 76, 77, 72, 6, 0, 38, 56, 42, 34, 102, -12, 22, 15, 22, 83, 107, 34, 11, 100, 48, 20, 3, 9, 84, 40, 0, 63, 28, 92, 48, 10, 2, 90, 72, 120, 94, 69, 153, 99, 70, 2, 189, -82, -78, 51, -39, -16, -41, 0, 43, 39, 48, 42, -22, -30, -28, 4, 32, -24, -17, 22, 30, -33, -26, 0, 34, -1, 0, 30, -14, 0, 0, 7, 11, 2, -7, -8, 28, -29, -14, 23, -10, -7, 23, 28, 28, 21, -24, 27, 3, 34, 34, 221, -39, 34, 30, -36, -33, -5, 3, -26, -49, 187, 120, -56, 3, 275, 99, -78, -27, 113, 231, 133, 128, 140, 86, 61, 251, 54, 49, 71, 69, 68, 36, 10, 23, 6, -8, -2, 5, 0, 69, 92, 18, -9, 4, 14, 0, 7, -1, 7, 6, -1, -4, 63, 28, 33, 92, 33, 17, 0, 4, 13, 85, 7, 21, 3, 43, 23, -8, 30, 37, 0, 108, 0, 0, 6, 46, 82, 42, 10, 28, 31, 49, 0, 16, 49, 3, 34, 20, 134, 3, 24, 36, 0, 32, 1, 102, 9, 28, 1, 82, 0, 137, 170, 9, 180, 39, 15, 16, 0, 10, 8, 37, 30, 145, 49, 68, 0, 66, 0, 99, 17, 85, 0, 61, 151, 83, 30, 71, 32, 27, 210, 20, 23, 6, 13, 33, 0, 52, 2, 13, 23, 10, 17, 45, -32, 89, 16, 53, -37, 7, 10, 20, 24, 39, 56, -7, 13, -5, 85, 5, -42, 112, 124, 111, 103, 25, 19, 13, 70, 66, 43, 31, 12, 64, 47, 14, 0, -2, 26, 7, 11, 19, -8, 58, 17, 0, 34, 29, 10, 5, 84, 27, 36, 14, -2, 60, 75, 29, 30, 85, 6, 39, 51, 32, 0, 58, 31, 20, 1, 13, 9, 27, 10, 78, 25, 26, -3, 99, 75, 89, 5, 42, 52, 16, 13, 49, 8, 12, 41, 13, 86, 87, 14, 27, 9, 4, 34, 80, 69, 3, 40, 43, 12, 5, 26, 24, 46, 57, 17, 30, 58, 4, 0, 17, 32, 58, 4, 0, 166, 21, 4, 18, 95, 6, 46, 21, 58, 174, 54, 54, 85, 62, 52, 98, 121, 0, 95, 101, -2, 34, 62, 36, -25, 71, 5, 58, 11, 0, -2, 57, 0, 83, 45, 23, 114, 18, 3, 0, 0, 26, 19, -5, 11, -11, 4, -1, 33, 54, 17, 3, 95, 12, 14, 66, 8, 40, 0, 43, 10, 46, 1, 30, -6, 12, 42, 54, 90, 3, 4, 8, 10, 36, 7, -4, 3, 2, 14, 40, 60, 13, 25, 45, 33, 7, 15, 8, 49, 10, 0, -54, 64, -1, 14, -2, 38, 37, 7, 14, 40, -6, 40, 39, 25, 55, 22, 25, 7, 109, 39, 62, 41, 9, 22, 7, 11, 19, 5, 0, 0, 5, 22, 43, 55, 66, 0, 43, 78, 2, 108, 1, 6, 1, -21, 20, 7, 114, 2, 36, 66, 0, 0, 10, -1, 0, 94, 48, -8, 124, 5, 11, -29, 23, 77, 12, 0, 177, 53, 76, 3, 71, 21, 86, 33, 110, 39, 77, 170, 89, 45, 0, 5, 187, 137, 62, 191, -61, 55, 18, 9, 118, 42, 35, 2, 105, -75, 93, 79, 5, 45, 81, 114, 136, 2, 15, 53, 24, 59, 10, 34, 76, 48, -18, 107, 81, 67, 14, 3, 126, 19, 25, 85, 69, 2, 8, 33, 0, 0, 22, 27, 26, 2, 0, 3, 3, 72, 61, 0, 10, 56, 9, 9, 63, 40, -1, 50, 38, 12, 179, 16, 97, 137, 96, 96, 10, 13, 97, 64, 62, -8, 161, 73, 163, 72, 0, 184, 166, 35, 168, 0, 41, 66, -17, 0, 0, 10, -6, 52, -15, 73, 25, 0, 0, 27, 20, 57, 110, -25, 138, 63, -18, 14, 8, 90, 48, 2, 9, 11, 0, -114, 21, 0, 25, 0, 26, 76, 3, 1, 20, 66, 107, -15, 58, 19, 28, 62, -3, -15, -8, 20, 15, -5, 5, -19, -38, 8, 13, 5, 11, 24, 56, 7, 0, 109, 6, 4, 21, 37, 26, -12, 4, 176, 23, 16, 142, 120, 81, 34, 215, 33, -16, 158, 28, -79, 71, 0, 164, 2, 0, -32, 42, 69, 2, -3, 66, 9, 149, 0, 1, 207, -1, 71, 1, 155, 91, 206, 79, 69, 0, 50, 35, 547, 33, 63, 51, 41, 75, 40, 0, 73, 52, 54, 98, 118, 108, 0, 62, 48, 20, 78, 112, 38, 3, 67, 73, 54, 74, 141, 86, 6, 0, 61, 34, 74, -5, 23, 54, 180, 20, 32, 63, 81, 6, 71, 61, 68, 21, 60, 27, 96, 184, 67, -5, 54, 0, 49, 73, 26, 104, 11, 137, 129, 5, 83, 260, 146, 76, 44, 29, 155, 129, 0, 10, 188, 50, 6, 0, 185, 11, 127, 246, 32, 58, 0, 21, 83, 133, 0, 7, 10, 49, 6, 10, 59, 50, 3, 0, 56, 43, 3, 45, 16, 34, 32, 3, 36, 5, 66, 0, 0, 82, 0, 17, 0, 82, 65, 48, 100, 0, 0, 0, 0, 0, 10, 77, 40, 10, -6, 147, 47, 25, -1, 19, 23, 0, -97, 60, 41, 78, 38, 14, 7, -15, 96, 73, 16, 8, 43, 62, 40, 7, 12, 20, 141, 0, 53, 74, 44, 44, 49, -2, 35, 61, 13, 2, 64, 43, 98, 92, -6, 84, 0, 99, 108, 27, 0, 67, 10, 37, 20, 16, 228, 47, 19, 10, 2, -6, 27, 13, 12, 78, 230, 114, 24, 20, 2, 0, 0, 7, -4, 33, 62, 144, 73, 182, 38, 98, -3, 0, -10, 82, 23, 29, 106, 232, 25, 0, 0, 0, 3, 88, 32, 29, -16, 6, 29, 45, 42, 24, 33, 11, -1, 15, 7, -10, 94, 0, 4, 0, -21, 0, 7, 16, 14, 81, 96, 169, -2, 0, -11, -43, 107, 62, 11, 0, 0, 174, 71, -12, -1, -7, 19, -28, 9, 64, 27, 13, 97, 29, 33, 46, 7, 0, 0, 40, -40, 9, -1, 0, 9, 53, 60, 8, 0, 2, 91, 43, -48, 83, 0, 0, 22, -30, 148, 0, 6, 26, -3, 23, 19, -22, 0, 14, 56, 69, 90, 26, 6, 105, 100, -26, 22, 115, 103, 3, 117, 12, 161, 139, 41, -36, 5, 0, 0, -43, 76, 75, 5, 0, 176, 60, 0, 0, -8, 0, 0, 60, 0, 16, 132, 96, 14, -6, 54, 37, 9, 11, 0, 19, 4, 17, 20, 3, 41, 50, 17, -10, 102, -2, 52, 133, 66, 37, 32, 3, 29, 30, 167, 11, 48, 0, 222, 58, 50, 39, 0, 162, -1, 30, 49, 0, 48, -29, 38, 56, 39, 23, 23, 0, 0, 105, 96, 11, 38, 14, 0, 92, 49, 0, 73, 12, 114, 0, -3, 77, 51, 0, 109, 47, 114, 92, 55, 185, 166, 72, 0, -11, 33, 120, 86, 144, 126, 105, 85, 64, 49, 26, -51, 125, 195, 140, 10, 82, 65, 3, 0, 0, 0, 0, 0, 231, 213, 188, 52, 89, 57, 58, 65, 43, 57, 0, 0, 65, 0, 59, 108, 61, 74, 164, 159, 121, 71, 0, 0, 99, 163, 85, 65, 48, 38, 158, 98, 63, 48, 13, 114, 68, 49, 47, 67, 17, 34, 82, 53, 232, 222, 188, 99, 75, 9, 0, 3, 1, 33, 106, 24, 0, 0, 34, 25, 28, 75, 42, 25, 8, 223, 92, 38, 81, 15, 0, 0, 0, 0, 0, 48, 36, -14, 0, 0, 205, 34, 14, 46, 50, 195, 129, 71, 57, 0, 0, 23, 193, 162, 22, 5, 8, 62, 40, 20, 184, 0, 0, 59, 86, 117, 29, -115, 0, 0, 18, 179, 52, 39, 12, 68, 0, 0, 8, 90, 225, 21, 30, 5, 0, 168, 85, 47, 30, -12, 6, 74, 2, 9, 11, 0, 0, 20, 6, 207, -9, 21, 91, 20, 10, 124, 128, -18, 21, -1, 31, 66, 57, 3, 5, 246, 165, 101, 0, 109, 76, 0, 78, 81, 0, 250, 153, 145, 122, 136, 152, 103, 110, 101, 89, 0, 114, 71, 83, 3, 0, 0, 33, 46, 115, 130, 108, 93, 89, 21, 0, 91, 76, 0, 109, 0, 0, 12, 75, 28, 0, 0, 0, 102, 26, 55, 0, 0, 6, 89, 65, 0, 22, 12, 1, -2, -8, 0, 16, 69, 107, 93, 79, 48, 48, 67, 0, 70, 71, 91, 0, 0, 15, 3, 21, 118, 0, 124, 29, 40, 86, 0, 0, 2, 137, 128, 119, 120, 112, 60, 0, 0, 120, 30, 53, 0, 0, 0, 13, 7, 0, 0, -6, 98, 148, 120, 90, 76, 27, 11, 4, 0, 70, 63, 0, 64, 45, 27, 0, 0, 0, 205, 14, 3, 12, 0, 0, 0, 7, 177, 184, 147, 122, 80, 71, 13, 0, 0, 86, 84, -1, -12, 5, 0, 0, 188, 98, 45, 59, 17, 62, 3, 11, 31, 21, 14, 0, 236, 156, 134, 123, 112, 90, 84, 19, 0, 77, 59, -4, 125, 0, 0, 0, 4, 212, 214, 166, 116, 55, 0, 10, 0, 0, 0, 70, 0, 11, 0, 0, 1, 4, 172, 81, 43, 52, 20, 23, 29, 5, 5, 49, 70, 101, 0, 30, 0, 0, 52, 159, 127, 69, 51, 0, 79, 36, 5, 66, 165, 99, 40, 0, 0, 150, 194, 71, 0, 156, 99, 87, 59, 42, 0, 13, 22, 0, 70, 41, 0, 0, 74, 162, 180, 85, 175, 128, 89, 86, 65, 0, 0, 39, 0, 79, 0, 0, 99, 18, 0, 0, 0, 12, 158, 131, 68, 37, 0, 0, 33, 136, 46, 0, 0, 91, 176, 134, 58, 0, -3, 125, 31, 12, 15, 0, 0, 12, 2, 196, 170, 102, 57, 43, 0, 121, 2, 58, 89, 125, 87, 9, -3, 34, 31, 189, 106, 18, 24, 8, 0, 0, 186, 5, 19, 50, 16, 124, 0, 70, 0, 179, 183, 165, 125, 116, 88, 92, 67, 58, 14, 0, -25, 25, 195, 122, 109, 56, 0, 4, 0, 0, 153, 53, 32, 2, 10, 4, 2, 66, 0, 4, 0, 224, 234, 152, 128, 102, 55, 0, 0, 24, 24, 0, 106, 61, 0, 15, 11, 31, 0, 187, 151, 67, 0, 16, 64, 35, 50, 60, 201, 0, 184, 132, 17, 6, 5, 22, 10, 30, 27, 57, 9, 0, -3, 47, 52, 30, 197, 0, 0, 13, 170, 42, 0, 0, 46, 145, 19, 26, 143, 47, 61, 0, 29, 58, 237, 139, 65, 5, 34, 0, 59, 86, 60, 6, 33, 0, 42, 20, 140, 125, 67, 11, 3, 4, 0, 16, 57, 34, 0, 64, 34, 151, 0, 0, 11, 14, 179, 37, 10, 86, 0, 81, 115, 0, 149, 67, 0, 8, 56, 0, 52, 13, 42, 58, 23, 31, 68, 141, 166, 112, 6, 0, 0, 80, 45, 95, 18, 144, 0, 106, 158, 13, 0, 0, 16, 46, 114, 30, 86, 50, 140, 31, 30, 26, 22, 30, 0, 43, 0, 7, 137, 14, 0, -12, -3, 0, 48, 36, 0, 49, 167, 243, 0, 136, 114, 22, 55, 153, 0, 0, 0, 0, 65, 7, 20, 18, 151, 144, 85, 39, 19, 4, 16, 147, 21, 12, 168, 6, 2, 1, 119, 60, 128, 0, 147, 119, 56, 13, 12, 52, 0, 8, 24, -2, 0, 37, 14, 20, 60, 0, 0, 0, 15, 42, 58, 102, 123, 14, 128, 18, 10, 167, 49, 28, 0, 59, 125, 7, 133, 84, 0, 40, 160, 128, 8, 179, 11, 0, 163, 9, 74, 215, 0, 117, 137, 22, 0, 25, 130, 3, 91, 207, 83, 0, 2, 36, 18, 5, 51, 68, 129, 0, 91, 26, 15, 72, 102, 0, 89, 107, 9, 70, 128, 85, 164, 171, 0, 32, 94, 0, 32, 96, 160, 6, 71, 84, 73, 272, 0, 153, 211, 67, 12, 0, 90, 85, 167, 170, 21, 104, 28, 0, 28, 0, 7, 40, 5, 65, 73, 14, 0, 19, 71, 96, 4, 0, 0, 90, 35, 10, -2, 23, 14, 146, 0, 19, -7, 0, 75, 6, 72, 44, 88, 183, 86, 69, 126, 23, 73, 222, 0, 29, 0, 15, 120, 151, 49, 3, 25, 6, 0, 113, 0, 12, 25, 23, 68, 125, 183, 88, 29, 31, 0, 20, 0, 53, 5, 38, 31, 0, 0, 38, 39, 0, 22, 31, 36, 0, 16, 97, 13, 73, 19, 0, 18, 110, 0, -22, 49, 0, 0, 57, 43, 13, 91, 114, 80, 16, 49, 45, 9, 0, 0, 167, 32, 73, 0, 16, 6, 18, 0, 50, 32, 6, 14, 56, 62, 119, 49, -5, 42, 138, 98, -4, 24, 8, 22, 12, 0, 14, 75, 30, 93, 0, 16, 0, 0, 46, 44, 76, 16, 0, 67, 41, 22, 0, 52, 1, 51, 28, 92, 9, 61, -1, 4, 0, 0, 88, 41, 0, 55, 22, 11, 102, 0, 40, 83, 67, 39, 70, 10, 62, 8, 10, 28, 21, 0, 30, -18, 74, -1, 0, 8, 106, 14, 70, 0, 0, 0, 0, 31, 31, 0, 50, 70, 81, 127, 29, 2, 26, 93, 73, 68, 0, 61, 67, 8, 0, 28, 82, 0, 8, 0, 0, -2, 0, 18, 67, 19, 67, 0, 19, 6, 0, 57, 58, 36, 0, 0, 0, 44, 9, 0, 11, 99, 0, 26, -1, 81, 0, 14, -4, 5, 25, 2, 0, 0, 21, 113, 7, 0, 2, 13, 29, 25, 0, 47, 33, 0, 90, 25, 100, 80, 21, 84, 0, -19, 18, 39, 79, 1, 12, -5, 0, -3, 78, 79, 56, 0, 0, 17, 13, 8, 7, 6, 3, 53, 21, 7, 29, 21, 20, 16, 11, 0, 9, 28, 19, 67, 30, 23, 22, 54, 83, 48, 95, 27, 49, 66, 52, 14, 5, 75, 55, 64, 1, 39, 1, 47, 44, 14, 34, 76, 72, 26, 69, 72, 21, 37, 46, 22, 20, 44, 40, 109, 23, 50, 17, 87, 252, 257, 248, 8, 61, 65, 59, 79, 81, 0, 100, 98, 72, 206, 214, 90, -1, 15, 79, 46, 52, 3, 75, 15, 31, 110, 62, 160, 169, 8, 35, 141, 30, 52, 56, 62, 144, 56, 1, 40, 89, 56, 0, 54, 0, 128, 62, 10, 0, 116, 5, 29, 5, 35, 93, -3, -9, 26, 0, 133, 0, 50, 73, 60, 56, 17, -10, -24, 68, 260, 1, 57, 5, 151, -3, -17, -27, -35, 22, 13, 13, 126, 38, 38, 77, 0, 0, 0, 25, 123, 23, -15, -28, 45, 93, 9, 7, 41, 33, 0, 73, 4, 154, 77, 38, 0, 138, 12, 11, -43, 24, 82, 58, 169, 0, 5, 17, 47, -1, -1, 0, 69, 9, 113, 2, 4, 131, -2, 0, 1, 0, 8, 0, 0, 0, 22, 40, 0, 4, 0, 0, 46, 8, 7, 77, 15, 43, 24, 7, -34, 51, 110, 23, 23, -25, 73, 38, 28, -6, 16, -7, 91, 2, 0, 9, 16, 37, 26, 10, -3, -3, 89, 110, 33, 41, 10, 0, 43, 33, 13, 411, 45, 92, 9, 0, 14, 167, 15, 74, 102, 90, 16, 35, 118, 0, 0, 127, -2, -1, -2, -4, 0, 44, -20, 19, 19, 86, 42, 16, -14, 37, 35, 65, 71, 3, 223, 23, -2, 71, 8, 73, 36, 69, 77, 84, 99, 42, 10, 9, 46, 60, 104, 93, 287, 12, 28, 7, 10, 0, 90, 52, 56, -2, 44, 6, 1, 11, 220, 35, 74, 82, -2, 0, 0, 0, 26, 104, -6, 4, 196, 12, 49, 81, 57, 97, 54, 12, -13, -22, 94, 67, 29, 4, 34, 186, 48, 1, 121, 122, 2, 40, 82, 1, 21, 27, 33, 161, 79, 116, 38, 59, 138, 39, 50, -20, 131, 135, 140, 3, 22, 36, 86, 2, 23, 70, 111, 1, 3, 20, 76, 88, 0, -2, 106, 76, 74, 39, 9, 40, 116, 68, 29, 30, 26, 76, 3, 80, 171, 31, 76, 53, 193, 120, 198, 178, 135, 110, 92, -3, 0, 184, 99, 68, 23, 26, 12, 184, 141, 51, 16, 9, -11, 70, 51, 80, 241, 134, 28, 3, 219, 108, -9, 91, 225, 205, 171, 58, 63, 64, 76, 130, 98, 25, 24, 214, 18, 131, 87, 110, 41, 31, 118, 49, 226, 118, 7, 9, 4, 134, 126, 19, 72, 4, 5, 102, 70, 100, 0, 0, 95, 90, 14, 14, 35, 134, 50, 96, -3, 101, 47, 112, -2, 101, -32, 245, 68, 130, 73, 70, 19, 18, 226, 75, 37, 5, -4, 76, 139, 2, 50, 35, 128, 76, -6, 12, 82, 11, 72, 86, 87, 48, 66, 0, 0, 120, 37, 74, 90, 11, 0, -3, 55, 21, 3, 0, -1, 19, 29, 16, 35, 85, 119, 55, 3, 2, 75, 93, 109, 109, 108, 189, 192, 104, 5, 7, 26, 129, 0, 114, 7, 167, 141, 118, 23, 121, 5, 26, 38, -5, 176, 35, 11, 24, -4, 124, -2, -20, 117, 37, -3, 11, 22, 101, -5, 0, 139, 67, 179, 128, 114, 0, 115, 22, 6, 16, 166, 127, 6, 9, 26, -21, 136, 61, 11, 118, 11, 43, 36, 29, 44, 145, 75, 14, 67, 145, 38, 68, 194, 194, 164, 20, 56, 105, 186, 44, 103, 41, 63, 209, 33, 102, 68, 99, 216, 102, 92, -18, -21, 141, -36, 72, 241, 42, 40, 123, 6, 0, -24, 28, 130, 76, 189, 96, 39, 21, 262, 89, 207, 71, 199, 148, 120, 96, 118, 234, 16, 43, 208, 117, 2, 40, 107, 68, 73, 95, 123, 145, 20, 153, 137, 4, 42, 201, 191, 34, 33, 98, 123, 110, 19, 255, 207, 35, 29, 0, 22, 14, 83, 188, 79, 59, 84, 166, 44, 128, 110, 17, 258, 47, 26, 121, 115, 29, 131, 69, 53, 66, 0, 122, 35, 3, 19, 13, 74, 23, 24, 48, 204, -8, 6, 33, 21, 56, 26, 102, 56, 204, 13, 40, 10, 24, 0, 15, 0, 59, 0, 174, 11, 5, 21, 53, 7, 85, 108, 12, 19, 208, 127, 0, 4, 38, 73, 72, 15, 0, 35, 0, 0, 7, 6, -2, 0, 19, 42, 99, 111, 66, 24, 106, 64, 113, 14, 8, 232, 0, 30, 37, 103, 0, 73, 4, 22, 0, 7, 0, 56, 1, 62, 54, 78, 24, 115, 8, 51, 115, 23, 73, 62, 56, -2, 100, 72, 155, 203, -30, -2, -10, 43, 69, 81, 28, 62, 68, 51, 48, 89, 19, 42, 46, 76, 0, 29, 46, 49, 69, 70, 0, -1, 46, 0, 63, 8, 119, 1, 124, 0, 0, 36, 83, 91, 37, 2, 13, 24, 44, 149, 104, 28, 217, 1, 10, 2, 2, 6, -7, 1, 80, 10, 11, 78, 144, 205, 26, -1, 16, -2, -6, 71, 176, 61, 184, 54, 162, 167, 176, 183, 0, 148, 106, 136, 1, 52, 17, 100, 182, 38, 39, 171, 69, 68, 173, 76, 42, 140, 179, 50, 46, 21, 58, 3, 31, 65, 25, 53, -8, 59, -6, 55, 5, 95, 111, -26, 20, 42, 15, -44, 43, 6, 81, -17, 27, 79, -2, 38, -13, 0, 12, 78, 92, -6, -65, 26, 199, 1, 188, 79, 16, 60, 97, 35, 67, -1, -6, -25, -8, 36, 144, 132, 145, 0, 0, 18, 47, -4, 33, 167, 57, 20, -6, 49, 53, 88, 120, 38, 93, 26, -12, 173, 44, 29, 18, 3, 95, -15, 51, 4, 147, 181, 23, 133, 49, 167, 26, 42, 123, 56, 5, 169, 61, 20, 102, -7, -10, 6, 43, 23, -27, 72, 6, 1, 212, 122, 58, 1, 231, 68, 0, 48, 43, 42, 4, 36, 108, 11, 99, 0, 101, 35, 28, 57, 0, 199, -3, 124, 134, 118, 132, 6, 19, 29, 11, 147, 24, 131, 239, 27, 45, 98, 110, 5, 20, 11, 0, 262, 11, 24, 35, 9, 264, 5, 6, 0, -7, 34, 8, 0, 29, 36, 45, 44, 65, 46, 40, 39, 37, 0, 34, 35, 27, 17, 19, 95, 50, 319, 317, 316, 49, 47, 55, 327, -13, -4, 42, 294, -76, 264, 262, 270, 77, 256, 328, 261, 267, 275, 294, 309, 186, 45, 188, 315, -4, 208, 356, 359, 369, 49, 395, 73, 227, 197, 188, 142, 57, 107, -46, 254, 215, 71, 244, 228, 224, 149, 0, 34, 122, 42, 20, 34, 145, -2, 53, 3, 47, 30, 350, 272, 0, 69, -7, 7, 24, 3, 33, 183, 0, 27, 0, 43, 88, 69, 7, 0, 3, 0, 80, -8, 34, 13, 5, 9, 13, 18, 33, 5, 0, 9, 20, 6, 5, 13, 11, 36, -5, 5, 0, 3, 2, -29, 45, 0, 36, 170, 5, 0, 48, 4, 5, 163, 0, 33, 57, 230, 38, 57, 89, 25, 120, 52, 82, -38, 49, 166, 189, -20, -17, -40, 31, -20, 30, 10, 125, 0, 21, 5, 54, 8, 98, 79, 0, -6, 91, 19, 124, 159, 61, 66, 97, 0, 7, 35, 30, 113, 56, 45, 15, 23, 10, -8, 46, 200, 29, 20, 35, 14, 6, 42, 45, 44, 49, 219, 34, 35, 13, -2, 60, 36, 34, 75, 63, 59, 96, 35, 180, 10, 251, 63, -1, 20, 39, 120, 79, 28, -65, -2, 190, 0, -24, 65, 18, 178, -10, 1, 53, 75, 137, 75, 18, 22, 62, 92, 4, -3, 213, 169, 214, 0, 90, 69, 212, -3, 19, 8, 5, 71, 50, 80, 56, 23, 91, 30, 53, 35, 54, 78, 232, 17, 108, 126, -2, 0, 177, 180, 14, 26, 144, 22, 20, 193, 152, 48, 12, 4, 8, 203, 69, 208, 215, 183, 200, 215, 211, 112, 226, 136, 21, 0, 211, 154, 19, 214, 49, 19, 63, 5, -6, 75, 36, 119, 7, 10, 65, 0, 46, 68, 51, 26, 22, 39, 19, 15, 0, 107, 81, -2, 110, 0, 0, 30, 49, 19, 1, -3, 92, 73, -7, 0, -6, 40, 32, -12, -9, 29, -5, 47, 191, -89, 79, 16, 92, 34, 11, -10, 71, 17, 8, 0, 71, -2, 0, 28, 0, 139, -13, 33, 3, -15, 3, 0, 30, 35, 0, 16, 2, 6, 14, 0, 37, 0, 84, -5, -6, 39, 10, 0, 0, 55, -25, 3, 2, 16, 23, 12, 47, 59, 0, 43, 0, 9, 52, 59, 48, 26, 89, -1, 23, 46, 47, 24, 32, 27, 6, 0, 47, 8, 0, 5, 0, 0, 65, 88, 33, 0, 10, 3, 7, 122, -7, 5, 17, 0, 182, 77, 170, 9, 135, 67, 157, 39, 91, 36, 13, 0, 59, 7, 25, 77, 1, 0, 0, 0, 10, 74, 12, 0, 103, 0, 32, 67, 7, 4, 55, 100, 26, 25, 13, 39, 39, 11, 0, 29, 0, 0, 49, 5, 126, 4, 6, 0, 34, 3, 74, 0, 5, 33, 7, 38, 0, 3, 19, -3, 27, 5, -13, 0, 6, 14, 12, 0, -1, 0, 35, 0, 48, 4, 22, 14, 5, 58, 24, 74, 78, 34, 31, 71, 92, 55, 64, 95, 140, 101, 87, 34, 19, 5, 113, 43, 7, 8, 108, 13, 0, 91, -23, 68, 154, 20, 19, 66, 0, 48, 18, 29, 63, 9, -4, -11, 13, 28, 27, 0, 1, 194, -3, -9, 0, 148, 115, 0, 0, 0, 0, 5, 50, 14, 135, 119, 39, -62, 23, 33, 60, 48, 4, 92, 50, 79, -6, 29, 14, 10, 6, 12, 28, 63, 92, 12, 58, 122, 0, 20, 10, 77, 41, 45, 110, 118, -22, -23, -10, -33, 38, 35, 0, -6, 0, 1, 10, 0, 10, 24, 7, 195, 117, 101, 78]}, {"marker": {"color": "rgb(5, 79, 174)"}, "name": "Horizontal Distance", "type": "box", "y": [258, 212, 268, 242, 153, 300, 270, 234, 240, 247, 180, 371, 150, 150, 67, 42, 120, 85, 95, 85, 60, 216, 192, 124, 120, 0, 30, 323, 212, 127, 67, 120, 124, 150, 162, 242, 67, 182, 124, 0, 0, 0, 150, 30, 95, 210, 212, 234, 95, 85, 60, 85, 30, 192, 175, 30, 190, 180, 175, 162, 134, 42, 134, 153, 150, 127, 134, 90, 218, 180, 242, 277, 108, 0, 240, 201, 134, 30, 382, 134, 531, 67, 150, 216, 210, 175, 30, 560, 418, 67, 272, 30, 67, 108, 426, 295, 85, 543, 0, 190, 351, 150, 150, 190, 170, 162, 30, 134, 309, 0, 408, 242, 342, 400, 228, 469, 240, 295, 190, 330, 301, 30, 150, 30, 85, 30, 551, 234, 67, 216, 0, 30, 85, 95, 0, 162, 30, 570, 42, 240, 658, 60, 272, 180, 218, 242, 190, 424, 175, 242, 30, 210, 295, 150, 424, 60, 488, 218, 285, 497, 335, 85, 268, 350, 127, 324, 510, 60, 228, 384, 67, 319, 384, 300, 366, 488, 484, 247, 201, 458, 618, 30, 600, 30, 424, 553, 150, 42, 360, 277, 283, 268, 67, 579, 60, 313, 212, 618, 190, 67, 150, 648, 218, 484, 120, 0, 759, 127, 153, 323, 604, 342, 443, 474, 85, 618, 85, 984, 30, 458, 90, 720, 150, 255, 484, 124, 603, 285, 600, 335, 190, 30, 180, 495, 150, 67, 750, 997, 42, 457, 201, 319, 503, 870, 201, 85, 247, 570, 792, 120, 510, 418, 362, 360, 30, 815, 842, 618, 811, 384, 743, 30, 67, 301, 391, 0, 690, 210, 424, 362, 631, 0, 0, 120, 60, 60, 153, 182, 240, 931, 323, 134, 42, 497, 626, 60, 0, 789, 60, 42, 201, 516, 849, 849, 306, 108, 153, 210, 659, 180, 313, 323, 85, 134, 730, 212, 42, 366, 764, 371, 192, 612, 134, 362, 335, 277, 283, 481, 268, 418, 30, 30, 658, 488, 212, 400, 212, 0, 693, 210, 192, 342, 630, 212, 458, 180, 297, 162, 134, 218, 480, 192, 323, 255, 134, 242, 42, 268, 300, 421, 484, 85, 426, 752, 780, 391, 395, 361, 570, 362, 210, 408, 127, 240, 466, 680, 192, 210, 242, 256, 247, 108, 60, 400, 696, 815, 330, 67, 351, 421, 85, 258, 192, 170, 350, 300, 242, 175, 120, 335, 268, 255, 201, 277, 242, 228, 175, 30, 0, 95, 162, 30, 0, 134, 190, 85, 30, 67, 124, 201, 362, 127, 95, 67, 0, 30, 495, 0, 124, 67, 0, 30, 124, 518, 150, 0, 67, 42, 0, 0, 360, 268, 285, 95, 201, 0, 30, 42, 190, 228, 201, 201, 0, 268, 180, 150, 376, 390, 295, 0, 150, 0, 150, 258, 283, 150, 350, 42, 124, 216, 134, 218, 108, 371, 513, 470, 247, 170, 108, 0, 108, 390, 228, 268, 85, 60, 228, 641, 95, 150, 558, 424, 212, 234, 212, 124, 674, 309, 192, 513, 379, 437, 242, 90, 247, 324, 350, 300, 437, 618, 30, 108, 497, 120, 60, 700, 175, 175, 120, 297, 451, 134, 95, 150, 484, 0, 437, 270, 400, 335, 932, 0, 514, 30, 408, 616, 127, 0, 60, 60, 0, 170, 42, 350, 638, 390, 485, 366, 451, 547, 433, 592, 390, 270, 384, 319, 553, 446, 458, 258, 182, 433, 300, 488, 210, 335, 362, 285, 150, 458, 379, 335, 306, 170, 42, 283, 95, 451, 170, 564, 631, 390, 524, 324, 170, 365, 277, 277, 30, 120, 90, 518, 60, 210, 85, 134, 297, 242, 285, 60, 30, 212, 283, 458, 277, 0, 421, 30, 90, 124, 67, 30, 319, 85, 446, 362, 67, 30, 376, 467, 0, 335, 42, 60, 30, 0, 301, 30, 228, 361, 30, 247, 499, 30, 30, 342, 108, 42, 0, 0, 272, 60, 127, 391, 331, 30, 30, 0, 127, 85, 0, 0, 67, 153, 175, 446, 124, 255, 190, 134, 60, 85, 150, 342, 228, 182, 124, 95, 0, 240, 180, 95, 361, 201, 659, 481, 67, 0, 228, 570, 150, 283, 424, 30, 402, 430, 300, 30, 42, 323, 541, 379, 85, 234, 30, 437, 524, 531, 60, 532, 433, 351, 430, 42, 108, 120, 95, 0, 30, 30, 67, 42, 366, 531, 591, 120, 60, 182, 579, 228, 30, 60, 190, 741, 201, 190, 218, 446, 433, 242, 547, 616, 242, 633, 511, 295, 531, 531, 85, 30, 30, 120, 480, 67, 234, 485, 495, 390, 218, 418, 190, 390, 30, 108, 371, 323, 212, 175, 30, 300, 277, 134, 371, 509, 150, 319, 216, 124, 277, 192, 127, 323, 247, 306, 0, 67, 42, 108, 201, 277, 30, 242, 120, 60, 443, 319, 256, 255, 162, 319, 67, 391, 0, 67, 484, 90, 67, 60, 424, 242, 85, 120, 234, 306, 234, 0, 150, 342, 42, 175, 361, 67, 67, 0, 30, 285, 300, 210, 391, 108, 212, 324, 42, 277, 127, 295, 120, 108, 85, 0, 0, 30, 120, 175, 30, 258, 85, 323, 30, 120, 582, 150, 342, 240, 30, 247, 190, 124, 90, 42, 255, 162, 201, 234, 42, 120, 242, 95, 323, 391, 124, 0, 150, 658, 60, 153, 124, 42, 42, 285, 30, 60, 247, 582, 30, 60, 85, 540, 0, 67, 350, 342, 150, 85, 30, 42, 108, 134, 418, 30, 212, 420, 212, 272, 474, 618, 95, 30, 42, 330, 0, 30, 108, 134, 365, 150, 218, 108, 124, 175, 60, 85, 242, 42, 418, 95, 700, 212, 295, 216, 268, 134, 192, 319, 30, 42, 162, 190, 0, 85, 201, 277, 301, 700, 421, 134, 170, 30, 0, 216, 242, 210, 30, 218, 277, 376, 182, 242, 361, 175, 30, 60, 330, 458, 454, 300, 120, 170, 30, 150, 30, 42, 124, 67, 722, 400, 127, 242, 258, 60, 306, 67, 642, 85, 30, 60, 180, 649, 67, 30, 30, 60, 90, 547, 42, 0, 90, 30, 42, 127, 180, 60, 218, 201, 95, 242, 330, 182, 134, 30, 108, 295, 216, 272, 127, 30, 95, 319, 0, 0, 258, 124, 324, 277, 42, 309, 210, 108, 295, 67, 30, 30, 60, 604, 212, 0, 218, 60, 324, 42, 30, 127, 537, 313, 469, 524, 323, 85, 210, 124, 108, 417, 42, 90, 182, 120, 408, 350, 830, 95, 371, 330, 277, 30, 277, 162, 771, 430, 180, 484, 190, 95, 212, 124, 108, 702, 150, 0, 30, 190, 85, 216, 67, 42, 127, 659, 182, 30, 162, 240, 180, 0, 42, 30, 426, 300, 210, 175, 485, 67, 108, 283, 127, 560, 30, 258, 30, 134, 201, 124, 228, 234, 443, 680, 558, 446, 30, 402, 524, 192, 30, 210, 400, 0, 408, 42, 108, 300, 812, 30, 108, 60, 607, 242, 90, 297, 531, 30, 430, 212, 210, 218, 192, 95, 134, 283, 85, 499, 297, 150, 408, 240, 234, 553, 108, 247, 300, 60, 182, 691, 85, 175, 404, 67, 212, 134, 242, 390, 323, 134, 0, 300, 283, 450, 240, 212, 384, 218, 313, 201, 150, 90, 95, 323, 295, 30, 190, 127, 283, 30, 342, 210, 351, 120, 153, 485, 360, 330, 300, 120, 108, 450, 30, 371, 420, 180, 85, 277, 0, 421, 391, 331, 182, 95, 256, 484, 306, 218, 162, 134, 170, 577, 342, 313, 201, 127, 108, 295, 268, 124, 361, 95, 218, 551, 134, 90, 30, 395, 600, 524, 443, 457, 42, 408, 301, 212, 277, 247, 190, 134, 459, 488, 400, 313, 256, 242, 306, 30, 90, 30, 335, 466, 350, 242, 309, 182, 212, 153, 95, 272, 162, 335, 247, 190, 228, 42, 256, 175, 323, 180, 150, 120, 90, 270, 150, 212, 268, 360, 300, 210, 120, 67, 210, 162, 67, 0, 153, 134, 335, 277, 134, 430, 180, 400, 256, 95, 247, 190, 162, 175, 150, 180, 120, 90, 95, 402, 180, 212, 90, 331, 247, 134, 258, 228, 150, 566, 319, 127, 212, 362, 150, 67, 201, 324, 67, 300, 0, 180, 323, 210, 150, 564, 90, 210, 108, 60, 319, 150, 150, 30, 589, 454, 30, 457, 170, 474, 446, 30, 60, 366, 319, 90, 446, 361, 0, 108, 342, 120, 313, 301, 175, 42, 150, 175, 150, 335, 306, 190, 0, 150, 212, 218, 90, 150, 108, 170, 218, 134, 258, 108, 108, 67, 0, 258, 335, 234, 0, 255, 216, 319, 242, 170, 124, 60, 192, 162, 175, 350, 300, 277, 120, 30, 443, 153, 60, 216, 190, 313, 285, 201, 120, 210, 268, 212, 247, 335, 218, 190, 153, 319, 268, 124, 108, 108, 323, 242, 134, 175, 256, 150, 190, 134, 216, 134, 108, 108, 309, 175, 228, 218, 430, 170, 182, 335, 430, 210, 162, 242, 390, 228, 90, 42, 90, 255, 201, 228, 108, 361, 60, 190, 95, 42, 60, 30, 162, 150, 85, 573, 150, 134, 306, 108, 95, 0, 120, 90, 277, 592, 180, 85, 0, 67, 85, 541, 108, 170, 85, 212, 324, 95, 240, 402, 379, 85, 319, 532, 201, 95, 175, 313, 424, 124, 95, 242, 212, 190, 30, 331, 335, 330, 60, 150, 108, 85, 150, 150, 120, 95, 90, 190, 30, 228, 150, 60, 210, 120, 127, 255, 362, 67, 190, 90, 277, 162, 30, 424, 124, 301, 270, 150, 67, 124, 85, 295, 175, 85, 270, 240, 216, 108, 95, 124, 0, 150, 108, 362, 268, 120, 67, 30, 90, 391, 150, 242, 0, 30, 85, 0, 180, 67, 0, 30, 0, 42, 67, 335, 242, 85, 240, 591, 150, 134, 60, 60, 124, 360, 0, 60, 162, 120, 150, 150, 120, 60, 210, 300, 242, 150, 90, 270, 150, 90, 124, 30, 60, 240, 240, 90, 270, 270, 240, 120, 90, 124, 210, 127, 60, 201, 256, 124, 240, 175, 150, 277, 210, 201, 182, 162, 240, 95, 60, 212, 60, 42, 30, 124, 124, 30, 30, 30, 0, 42, 30, 175, 90, 240, 95, 90, 60, 210, 120, 42, 120, 402, 127, 127, 150, 162, 108, 150, 182, 150, 255, 190, 300, 283, 234, 190, 210, 90, 120, 150, 270, 270, 270, 210, 120, 300, 127, 170, 240, 247, 90, 210, 95, 242, 153, 180, 212, 180, 180, 150, 127, 0, 255, 162, 30, 331, 124, 150, 150, 175, 134, 42, 0, 270, 270, 454, 108, 360, 360, 242, 90, 192, 228, 390, 335, 309, 124, 268, 60, 60, 60, 85, 42, 376, 255, 418, 362, 234, 30, 60, 366, 255, 240, 201, 30, 0, 30, 120, 361, 268, 216, 162, 134, 42, 30, 124, 150, 120, 30, 446, 450, 283, 256, 201, 466, 134, 85, 30, 90, 30, 30, 450, 323, 120, 182, 30, 60, 60, 108, 120, 67, 511, 443, 390, 120, 379, 210, 365, 247, 30, 0, 420, 300, 120, 391, 212, 0, 0, 30, 450, 30, 30, 306, 134, 108, 60, 0, 450, 134, 582, 240, 120, 90, 30, 277, 0, 480, 0, 60, 457, 162, 778, 272, 42, 150, 30, 108, 60, 85, 60, 67, 256, 150, 180, 721, 60, 351, 124, 313, 95, 42, 384, 391, 408, 391, 351, 618, 408, 525, 351, 120, 60, 285, 228, 30, 499, 636, 319, 417, 319, 277, 85, 446, 350, 277, 108, 404, 424, 564, 510, 30, 30, 192, 170, 85, 60, 382, 360, 330, 175, 295, 192, 95, 124, 319, 228, 0, 67, 408, 564, 571, 301, 283, 108, 216, 124, 95, 335, 0, 30, 201, 0, 127, 60, 67, 95, 42, 85, 309, 532, 799, 408, 0, 0, 0, 127, 127, 120, 30, 108, 192, 319, 30, 60, 210, 201, 228, 488, 543, 124, 30, 90, 30, 30, 108, 228, 170, 0, 90, 30, 437, 240, 90, 218, 85, 30, 150, 180, 150, 313, 285, 85, 525, 150, 0, 60, 379, 127, 182, 153, 95, 376, 30, 319, 339, 60, 313, 120, 60, 458, 162, 192, 743, 182, 212, 450, 0, 212, 513, 180, 618, 306, 85, 711, 342, 150, 342, 30, 451, 108, 573, 361, 339, 210, 420, 0, 268, 210, 270, 300, 365, 108, 120, 180, 270, 42, 30, 277, 42, 212, 150, 636, 591, 153, 30, 0, 247, 0, 134, 247, 277, 90, 255, 234, 60, 85, 402, 60, 85, 301, 0, 242, 295, 285, 175, 285, 607, 256, 342, 430, 30, 124, 30, 42, 0, 60, 408, 30, 90, 240, 323, 283, 134, 466, 42, 30, 30, 0, 30, 150, 210, 190, 272, 301, 228, 150, 180, 240, 30, 60, 0, 120, 134, 180, 424, 67, 162, 268, 240, 450, 30, 60, 30, 108, 30, 0, 120, 0, 247, 108, 228, 313, 270, 120, 525, 153, 124, 90, 0, 175, 228, 162, 95, 323, 120, 365, 454, 30, 0, 42, 95, 190, 0, 242, 170, 150, 124, 30, 283, 150, 124, 371, 120, 90, 30, 175, 95, 0, 127, 218, 67, 190, 277, 342, 313, 256, 228, 30, 295, 379, 408, 0, 30, 319, 124, 30, 0, 0, 30, 85, 85, 127, 190, 335, 108, 85, 0, 150, 270, 300, 323, 0, 67, 127, 42, 361, 124, 0, 430, 201, 42, 67, 95, 153, 120, 242, 42, 30, 0, 0, 30, 30, 0, 124, 150, 30, 342, 270, 240, 150, 524, 0, 134, 30, 30, 60, 268, 67, 42, 0, 30, 0, 0, 60, 95, 212, 300, 330, 240, 124, 95, 30, 162, 60, 216, 95, 60, 42, 30, 134, 60, 323, 342, 182, 210, 150, 255, 60, 150, 30, 0, 150, 30, 0, 268, 0, 95, 247, 190, 201, 175, 0, 108, 180, 134, 150, 30, 90, 150, 162, 150, 150, 120, 85, 124, 400, 553, 192, 150, 124, 60, 67, 85, 95, 150, 120, 60, 0, 0, 67, 216, 190, 175, 295, 95, 120, 216, 382, 228, 42, 30, 180, 85, 228, 95, 228, 382, 30, 30, 150, 190, 210, 228, 256, 212, 228, 175, 85, 67, 150, 95, 95, 190, 295, 134, 108, 30, 30, 0, 90, 67, 127, 42, 30, 30, 180, 218, 240, 404, 309, 242, 240, 240, 228, 180, 180, 153, 108, 277, 534, 309, 283, 30, 218, 0, 150, 175, 424, 30, 42, 450, 256, 108, 67, 134, 124, 120, 67, 30, 134, 30, 30, 0, 258, 30, 0, 120, 150, 67, 90, 300, 300, 300, 182, 180, 180, 124, 67, 124, 120, 90, 90, 90, 175, 335, 201, 182, 376, 30, 285, 277, 108, 430, 350, 153, 134, 124, 30, 67, 60, 60, 67, 0, 234, 150, 60, 192, 300, 360, 361, 379, 360, 360, 192, 170, 120, 120, 30, 85, 67, 60, 30, 30, 30, 42, 30, 30, 228, 108, 90, 60, 30, 60, 270, 361, 371, 242, 216, 95, 90, 67, 0, 0, 0, 0, 0, 0, 30, 0, 120, 90, 376, 382, 0, 210, 30, 0, 60, 342, 390, 256, 175, 60, 458, 0, 0, 30, 30, 0, 30, 30, 60, 658, 361, 162, 30, 175, 212, 277, 361, 30, 0, 162, 42, 391, 404, 404, 446, 277, 218, 42, 30, 0, 0, 0, 60, 60, 60, 42, 30, 42, 0, 212, 180, 127, 0, 30, 42, 60, 42, 134, 247, 247, 342, 295, 330, 242, 0, 30, 67, 60, 60, 90, 67, 90, 90, 268, 218, 60, 180, 210, 42, 285, 60, 573, 480, 0, 30, 42, 90, 134, 240, 30, 0, 150, 268, 95, 190, 319, 30, 60, 30, 30, 67, 85, 124, 272, 108, 120, 283, 85, 228, 309, 342, 234, 612, 30, 0, 0, 0, 85, 85, 67, 60, 30, 190, 301, 301, 120, 153, 30, 42, 0, 175, 120, 95, 150, 376, 255, 600, 95, 67, 42, 0, 90, 30, 0, 0, 331, 134, 192, 30, 0, 0, 309, 258, 457, 365, 30, 42, 60, 60, 60, 108, 127, 124, 120, 175, 30, 0, 382, 90, 180, 210, 0, 0, 0, 0, 30, 42, 212, 277, 300, 153, 295, 127, 277, 134, 150, 30, 277, 190, 210, 240, 255, 0, 0, 0, 30, 30, 0, 0, 30, 0, 30, 60, 127, 170, 30, 180, 342, 127, 85, 134, 120, 90, 30, 0, 108, 42, 153, 240, 150, 0, 0, 30, 30, 0, 30, 0, 42, 331, 365, 90, 485, 192, 134, 153, 108, 0, 0, 67, 192, 270, 295, 120, 0, 0, 30, 0, 30, 30, 0, 0, 30, 150, 30, 247, 170, 300, 150, 42, 335, 255, 0, 150, 150, 30, 67, 30, 0, 0, 0, 0, 90, 30, 0, 60, 95, 228, 150, 162, 277, 120, 242, 342, 234, 182, 192, 0, 0, 30, 30, 108, 576, 330, 0, 0, 124, 85, 270, 30, 210, 277, 212, 240, 150, 67, 30, 42, 0, 0, 67, 67, 30, 360, 360, 390, 390, 390, 330, 42, 30, 0, 85, 124, 228, 362, 330, 150, 300, 247, 153, 30, 0, 67, 30, 319, 30, 360, 127, 30, 60, 150, 60, 842, 721, 726, 309, 285, 270, 0, 60, 30, 212, 0, 395, 408, 420, 420, 182, 120, 134, 30, 150, 0, 0, 342, 95, 153, 108, 741, 301, 301, 67, 0, 0, 95, 212, 85, 150, 127, 134, 30, 234, 242, 108, 150, 108, 218, 361, 124, 268, 272, 0, 421, 400, 285, 256, 175, 216, 127, 201, 30, 0, 268, 335, 182, 361, 85, 655, 648, 630, 641, 787, 242, 240, 30, 379, 295, 228, 134, 0, 30, 85, 350, 60, 42, 247, 313, 685, 210, 0, 30, 30, 212, 150, 170, 212, 0, 30, 0, 391, 0, 153, 258, 150, 42, 30, 120, 150, 67, 573, 216, 0, 30, 0, 0, 0, 0, 0, 270, 162, 127, 30, 0, 0, 153, 417, 0, 277, 190, 90, 242, 362, 841, 153, 30, 30, 0, 0, 30, 67, 150, 150, 124, 30, 30, 0, 306, 134, 30, 319, 228, 532, 564, 134, 0, 0, 30, 0, 0, 0, 67, 30, 0, 95, 127, 95, 228, 371, 150, 484, 0, 85, 0, 0, 319, 342, 30, 0, 108, 134, 30, 0, 162, 234, 170, 450, 454, 30, 30, 150, 297, 30, 150, 361, 295, 285, 150, 124, 67, 42, 150, 85, 60, 324, 212, 150, 192, 295, 424, 175, 277, 0, 30, 258, 218, 162, 134, 108, 85, 30, 0, 216, 42, 120, 255, 277, 335, 210, 395, 90, 134, 162, 268, 339, 0, 270, 90, 210, 124, 95, 42, 95, 277, 218, 553, 379, 573, 674, 60, 240, 170, 90, 30, 162, 0, 319, 335, 430, 402, 379, 0, 0, 60, 270, 306, 361, 330, 240, 180, 120, 60, 60, 95, 134, 277, 190, 283, 309, 268, 283, 0, 30, 95, 182, 150, 67, 390, 210, 180, 150, 120, 60, 30, 242, 306, 592, 361, 0, 95, 499, 400, 376, 350, 323, 295, 268, 212, 67, 42, 30, 0, 268, 404, 309, 95, 0, 182, 474, 30, 0, 192, 216, 351, 379, 390, 335, 283, 175, 150, 127, 108, 85, 67, 30, 0, 210, 85, 67, 42, 228, 242, 342, 258, 228, 300, 360, 234, 285, 256, 30, 30, 0, 319, 339, 361, 30, 0, 210, 192, 212, 234, 258, 335, 362, 323, 268, 216, 190, 162, 134, 85, 42, 30, 0, 175, 30, 190, 201, 417, 391, 360, 306, 60, 297, 228, 0, 0, 30, 201, 234, 277, 300, 324, 350, 366, 350, 335, 313, 285, 256, 228, 175, 150, 95, 67, 42, 0, 0, 124, 242, 277, 247, 218, 108, 175, 376, 300, 234, 297, 190, 162, 134, 108, 255, 300, 0, 30, 85, 443, 524, 518, 0, 339, 228, 212, 218, 242, 258, 319, 366, 361, 309, 295, 277, 190, 134, 85, 60, 30, 0, 0, 134, 108, 335, 371, 30, 190, 216, 277, 277, 270, 270, 272, 0, 170, 258, 350, 390, 0, 0, 0, 467, 0, 175, 201, 400, 283, 268, 256, 247, 242, 240, 242, 247, 256, 300, 339, 361, 361, 339, 319, 268, 242, 182, 153, 124, 42, 30, 0, 30, 108, 124, 108, 234, 127, 108, 272, 319, 150, 162, 192, 255, 240, 0, 150, 350, 466, 443, 0, 60, 342, 324, 309, 285, 272, 285, 295, 309, 342, 361, 361, 339, 319, 258, 242, 228, 218, 201, 150, 60, 0, 60, 108, 150, 108, 108, 212, 420, 268, 150, 212, 210, 153, 30, 108, 361, 0, 30, 60, 402, 446, 424, 384, 350, 335, 323, 313, 306, 300, 306, 313, 323, 335, 366, 234, 216, 201, 108, 162, 95, 127, 330, 127, 190, 180, 182, 120, 60, 30, 0, 0, 90, 408, 30, 30, 60, 426, 391, 351, 342, 335, 330, 342, 362, 366, 342, 297, 277, 255, 212, 162, 153, 124, 95, 42, 30, 0, 150, 30, 67, 0, 30, 0, 0, 0, 0, 182, 95, 42, 417, 134, 150, 170, 162, 153, 162, 108, 60, 457, 0, 95, 180, 365, 360, 360, 0, 30, 60, 120, 391, 450, 433, 402, 390, 379, 365, 361, 361, 365, 379, 350, 324, 277, 255, 234, 192, 170, 150, 124, 85, 60, 0, 175, 319, 342, 30, 0, 30, 0, 30, 30, 150, 612, 446, 108, 150, 124, 120, 120, 150, 497, 0, 30, 212, 268, 0, 0, 30, 30, 0, 0, 330, 492, 458, 395, 400, 324, 309, 283, 192, 170, 150, 108, 85, 67, 42, 0, 192, 0, 0, 0, 85, 631, 420, 541, 85, 90, 30, 182, 240, 437, 30, 30, 30, 0, 0, 450, 499, 516, 499, 470, 457, 430, 420, 404, 382, 361, 339, 319, 300, 283, 268, 170, 150, 108, 60, 30, 0, 108, 306, 180, 301, 30, 0, 0, 0, 0, 0, 0, 319, 361, 42, 85, 60, 60, 60, 85, 90, 90, 391, 390, 0, 60, 0, 391, 541, 525, 510, 497, 485, 459, 454, 433, 384, 361, 339, 319, 277, 258, 242, 228, 201, 175, 127, 67, 42, 30, 30, 30, 0, 0, 0, 0, 150, 67, 60, 85, 67, 30, 0, 0, 42, 0, 192, 247, 319, 361, 30, 60, 42, 395, 437, 300, 306, 360, 42, 30, 0, 0, 85, 365, 514, 543, 537, 503, 495, 417, 366, 342, 319, 297, 277, 234, 201, 108, 85, 67, 0, 0, 247, 90, 210, 67, 0, 0, 424, 258, 277, 330, 331, 30, 0, 0, 30, 430, 0, 30, 277, 331, 335, 30, 30, 0, 0, 0, 0, 0, 430, 488, 518, 551, 537, 524, 497, 484, 457, 402, 376, 350, 300, 277, 212, 192, 175, 150, 124, 95, 67, 42, 0, 95, 272, 127, 0, 0, 108, 42, 474, 300, 300, 484, 376, 0, 362, 0, 60, 350, 42, 30, 0, 30, 510, 417, 362, 258, 134, 124, 108, 85, 60, 0, 0, 335, 339, 268, 108, 0, 247, 134, 0, 180, 446, 300, 0, 268, 330, 30, 0, 30, 67, 516, 499, 470, 457, 430, 417, 390, 295, 242, 216, 192, 150, 95, 85, 67, 30, 30, 0, 258, 242, 42, 150, 309, 0, 85, 319, 417, 270, 270, 0, 124, 379, 0, 0, 30, 108, 492, 474, 430, 228, 201, 150, 108, 85, 67, 42, 30, 0, 175, 134, 190, 216, 268, 324, 0, 30, 124, 417, 366, 192, 30, 190, 323, 277, 242, 60, 60, 60, 0, 30, 30, 124, 120, 376, 313, 190, 134, 108, 85, 67, 30, 0, 0, 362, 30, 67, 124, 242, 67, 42, 42, 240, 108, 127, 228, 218, 210, 210, 256, 30, 0, 0, 30, 30, 30, 60, 30, 0, 30, 60, 120, 150, 192, 255, 277, 277, 255, 67, 42, 30, 0, 0, 0, 0, 30, 228, 330, 60, 180, 295, 212, 30, 0, 0, 30, 0, 0, 0, 0, 0, 30, 30, 0, 30, 60, 85, 190, 497, 350, 268, 212, 134, 108, 30, 0, 30, 234, 212, 0, 30, 0, 42, 295, 67, 319, 153, 0, 418, 0, 0, 0, 0, 30, 30, 0, 0, 0, 0, 0, 30, 30, 0, 0, 30, 67, 458, 324, 295, 283, 212, 150, 67, 42, 30, 0, 0, 0, 0, 95, 190, 30, 0, 417, 0, 228, 127, 120, 120, 0, 0, 285, 30, 0, 30, 0, 42, 30, 0, 0, 30, 0, 0, 30, 60, 201, 218, 361, 339, 300, 283, 256, 242, 216, 201, 190, 150, 127, 108, 0, 0, 0, 0, 30, 108, 0, 0, 0, 234, 192, 67, 212, 67, 85, 90, 0, 0, 0, 0, 30, 0, 0, 30, 30, 0, 67, 30, 0, 30, 0, 30, 42, 247, 272, 404, 361, 297, 228, 216, 201, 190, 175, 153, 127, 108, 85, 30, 0, 0, 0, 0, 162, 212, 67, 0, 0, 127, 0, 60, 85, 258, 108, 60, 0, 30, 0, 0, 30, 0, 0, 0, 0, 90, 0, 384, 404, 361, 339, 255, 234, 201, 175, 162, 150, 134, 124, 42, 0, 0, 67, 60, 0, 42, 0, 85, 134, 0, 134, 0, 67, 361, 339, 108, 124, 0, 0, 60, 0, 0, 0, 30, 30, 0, 0, 0, 0, 0, 30, 30, 42, 426, 404, 382, 361, 339, 319, 234, 212, 175, 134, 124, 108, 90, 67, 30, 120, 127, 0, 150, 30, 0, 30, 108, 150, 175, 283, 108, 42, 60, 162, 395, 255, 175, 0, 0, 0, 0, 90, 0, 30, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 30, 60, 150, 0, 30, 60, 404, 382, 361, 339, 277, 255, 234, 212, 150, 134, 124, 108, 85, 67, 60, 60, 42, 0, 30, 42, 67, 175, 361, 90, 0, 0, 0, 0, 446, 0, 30, 0, 0, 0, 120, 120, 395, 430, 404, 382, 361, 319, 297, 277, 255, 212, 170, 127, 60, 42, 30, 30, 30, 0, 0, 124, 134, 120, 95, 150, 0, 0, 0, 42, 170, 30, 0, 42, 256, 60, 60, 67, 134, 242, 150, 127, 0, 0, 0, 42, 60, 0, 30, 30, 0, 0, 153, 190, 0, 30, 437, 446, 424, 382, 361, 339, 297, 277, 255, 234, 212, 192, 60, 42, 30, 30, 0, 0, 0, 60, 90, 67, 42, 0, 30, 0, 0, 201, 228, 234, 60, 60, 335, 306, 42, 30, 42, 30, 95, 212, 272, 170, 108, 95, 85, 0, 0, 0, 0, 30, 30, 30, 67, 30, 30, 0, 180, 180, 0, 0, 446, 426, 404, 382, 319, 297, 277, 234, 212, 170, 127, 42, 30, 30, 0, 0, 95, 42, 150, 108, 30, 0, 134, 0, 60, 0, 60, 30, 124, 0, 0, 60, 180, 240, 300, 67, 30, 0, 108, 60, 30, 0, 0, 0, 0, 0, 0, 0, 0, 190, 0, 446, 424, 404, 384, 361, 339, 319, 212, 192, 127, 42, 30, 0, 85, 192, 175, 134, 108, 0, 0, 175, 212, 0, 0, 134, 295, 124, 192, 212, 283, 67, 90, 30, 218, 488, 95, 60, 319, 108, 0, 127, 30, 0, 30, 67, 0, 0, 30, 30, 182, 242, 283, 0, 424, 404, 382, 319, 255, 234, 212, 192, 150, 108, 85, 42, 30, 30, 30, 247, 162, 0, 170, 201, 216, 0, 0, 30, 0, 162, 42, 42, 85, 30, 67, 90, 182, 324, 0, 342, 0, 0, 0, 426, 382, 361, 339, 319, 277, 255, 234, 212, 170, 150, 108, 67, 0, 30, 42, 134, 67, 216, 0, 0, 0, 180, 30, 0, 95, 124, 175, 201, 0, 134, 153, 30, 30, 42, 67, 283, 134, 90, 60, 0, 300, 60, 134, 108, 300, 433, 408, 384, 339, 319, 297, 277, 212, 192, 170, 108, 85, 67, 42, 30, 0, 30, 0, 108, 190, 0, 67, 134, 170, 180, 0, 242, 234, 30, 0, 0, 631, 30, 90, 150, 134, 60, 67, 150, 175, 228, 268, 150, 108, 420, 30, 30, 30, 0, 175, 0, 212, 485, 402, 120, 30, 0, 0, 0, 201, 256, 366, 297, 277, 255, 216, 192, 170, 127, 108, 85, 67, 30, 0, 42, 134, 255, 30, 0, 277, 0, 150, 175, 212, 150, 0, 90, 42, 30, 95, 134, 170, 268, 295, 0, 0, 0, 0, 0, 153, 228, 0, 0, 319, 360, 446, 30, 0, 0, 0, 0, 0, 85, 379, 457, 430, 350, 277, 212, 127, 108, 85, 67, 42, 30, 0, 0, 30, 42, 124, 301, 0, 0, 30, 192, 297, 297, 228, 0, 30, 255, 0, 150, 268, 0, 108, 30, 170, 0, 150, 0, 0, 67, 30, 0, 180, 0, 0, 30, 124, 323, 0, 418, 532, 418, 390, 362, 309, 192, 150, 108, 85, 67, 42, 30, 0, 42, 85, 190, 300, 234, 0, 277, 277, 162, 30, 30, 255, 234, 30, 180, 480, 175, 268, 42, 0, 108, 0, 150, 212, 404, 459, 30, 0, 134, 0, 0, 0, 437, 379, 323, 268, 216, 170, 95, 67, 42, 30, 42, 67, 0, 30, 85, 30, 0, 0, 42, 228, 258, 180, 240, 247, 150, 42, 495, 242, 90, 0, 0, 95, 60, 0, 150, 0, 30, 127, 402, 446, 459, 342, 285, 228, 201, 175, 150, 108, 85, 67, 30, 0, 30, 85, 212, 90, 30, 0, 0, 108, 0, 0, 0, 85, 85, 0, 0, 0, 240, 0, 30, 124, 30, 330, 655, 95, 120, 85, 30, 0, 120, 0, 480, 537, 30, 108, 170, 162, 408, 42, 153, 395, 365, 306, 218, 190, 162, 134, 108, 85, 67, 30, 0, 30, 120, 150, 0, 42, 95, 124, 212, 90, 0, 0, 0, 212, 301, 85, 0, 0, 67, 60, 30, 247, 30, 42, 85, 150, 162, 0, 564, 60, 67, 134, 488, 371, 297, 516, 450, 421, 331, 212, 153, 124, 67, 42, 30, 0, 0, 30, 85, 95, 277, 30, 0, 67, 190, 255, 277, 150, 30, 0, 0, 0, 240, 270, 0, 0, 247, 212, 67, 67, 60, 182, 242, 212, 60, 0, 0, 180, 190, 30, 67, 90, 150, 0, 0, 60, 426, 408, 390, 360, 330, 270, 240, 210, 120, 90, 60, 0, 0, 0, 150, 285, 67, 0, 277, 85, 42, 30, 30, 0, 0, 42, 30, 150, 270, 30, 30, 0, 127, 228, 247, 108, 150, 95, 30, 30, 0, 108, 297, 600, 706, 309, 190, 67, 234, 0, 361, 570, 0, 162, 228, 492, 384, 366, 350, 210, 150, 90, 30, 234, 277, 319, 90, 120, 95, 0, 339, 0, 0, 0, 30, 270, 0, 0, 134, 85, 120, 175, 0, 85, 0, 150, 228, 268, 182, 510, 376, 361, 342, 324, 309, 295, 240, 210, 90, 30, 0, 60, 30, 85, 150, 170, 361, 382, 277, 228, 0, 0, 0, 0, 150, 90, 30, 30, 170, 95, 30, 0, 42, 108, 703, 285, 366, 85, 170, 242, 0, 216, 268, 543, 0, 85, 120, 134, 190, 309, 30, 192, 437, 495, 335, 300, 283, 256, 240, 150, 120, 90, 60, 30, 216, 134, 30, 0, 67, 0, 42, 30, 272, 0, 162, 258, 30, 67, 540, 600, 127, 60, 108, 234, 175, 0, 0, 108, 552, 30, 0, 268, 0, 0, 285, 309, 277, 242, 216, 201, 175, 150, 120, 90, 60, 0, 85, 0, 180, 0, 95, 67, 180, 90, 30, 67, 30, 95, 277, 30, 540, 256, 60, 0, 30, 30, 600, 537, 150, 240, 242, 0, 134, 510, 283, 255, 234, 216, 201, 162, 134, 60, 30, 0, 95, 192, 255, 0, 30, 0, 150, 0, 134, 60, 30, 0, 134, 124, 120, 67, 95, 0, 335, 592, 42, 30, 0, 201, 0, 421, 297, 242, 212, 175, 162, 150, 67, 42, 30, 0, 30, 180, 319, 127, 85, 162, 0, 190, 0, 30, 30, 30, 212, 175, 90, 309, 0, 480, 510, 228, 268, 108, 67, 201, 90, 0, 42, 277, 0, 0, 190, 255, 190, 170, 150, 134, 124, 108, 85, 30, 180, 242, 67, 0, 85, 127, 0, 85, 182, 95, 85, 134, 30, 30, 886, 162, 301, 335, 30, 342, 201, 0, 0, 134, 150, 153, 162, 390, 0, 212, 192, 162, 150, 127, 95, 85, 42, 153, 90, 108, 0, 339, 342, 0, 150, 0, 30, 124, 201, 256, 90, 120, 350, 362, 365, 306, 67, 60, 150, 297, 450, 182, 228, 218, 108, 90, 240, 67, 182, 30, 591, 242, 0, 0, 391, 216, 170, 150, 108, 30, 30, 175, 90, 30, 0, 134, 85, 0, 30, 335, 361, 382, 30, 0, 0, 108, 67, 30, 0, 90, 313, 134, 150, 600, 624, 201, 182, 175, 30, 0, 0, 90, 90, 192, 513, 430, 256, 228, 201, 175, 150, 127, 108, 95, 67, 42, 30, 162, 0, 108, 201, 182, 351, 192, 108, 256, 319, 300, 0, 323, 306, 295, 0, 170, 242, 424, 516, 0, 247, 190, 162, 134, 108, 85, 67, 60, 0, 0, 283, 0, 255, 366, 0, 30, 0, 295, 210, 0, 85, 446, 153, 216, 234, 30, 360, 127, 0, 0, 495, 0, 0, 30, 295, 212, 182, 124, 67, 42, 0, 67, 309, 190, 0, 30, 0, 30, 95, 212, 0, 285, 201, 42, 60, 124, 402, 95, 127, 108, 0, 30, 255, 108, 85, 0, 162, 361, 210, 150, 120, 90, 60, 30, 0, 95, 42, 60, 0, 0, 60, 228, 283, 0, 85, 90, 0, 30, 67, 162, 218, 212, 228, 85, 192, 85, 30, 30, 0, 497, 497, 0, 240, 210, 150, 120, 90, 60, 30, 0, 234, 216, 402, 323, 67, 0, 0, 85, 134, 0, 150, 175, 256, 277, 60, 0, 242, 240, 124, 42, 242, 192, 150, 255, 297, 361, 366, 85, 150, 0, 216, 300, 513, 210, 150, 120, 90, 60, 0, 67, 234, 192, 0, 0, 258, 175, 85, 0, 95, 0, 0, 330, 361, 295, 272, 285, 212, 150, 210, 376, 95, 0, 0, 256, 210, 180, 0, 85, 127, 175, 228, 30, 0, 42, 170, 175, 0, 30, 162, 335, 30, 0, 424, 301, 150, 192, 175, 175, 0, 42, 162, 0, 30, 0, 30, 67, 268, 0, 0, 30, 127, 30, 201, 175, 108, 0, 90, 446, 408, 391, 362, 201, 306, 335, 295, 350, 42, 67, 371, 0, 30, 192, 228, 218, 0, 256, 180, 0, 124, 42, 30, 0, 30, 85, 379, 0, 210, 360, 306, 162, 0, 335, 300, 0, 60, 108, 0, 127, 60, 201, 190, 256, 247, 85, 0, 162, 0, 0, 0, 268, 319, 384, 309, 170, 0, 180, 395, 295, 402, 295, 283, 124, 30, 190, 42, 108, 30, 42, 153, 0, 201, 124, 0, 30, 0, 210, 330, 90, 60, 108, 108, 655, 573, 0, 180, 300, 360, 0, 0, 85, 60, 30, 150, 162, 234, 42, 0, 0, 0, 240, 330, 450, 421, 60, 124, 0, 408, 361, 60, 30, 120, 108, 67, 42, 30, 0, 30, 256, 124, 30, 60, 242, 0, 120, 210, 0, 0, 90, 240, 420, 566, 424, 365, 0, 60, 543, 524, 466, 30, 42, 474, 42, 30, 0, 180, 201, 190, 0, 210, 0, 85, 134, 190, 524, 42, 42, 513, 153, 124, 390, 361, 421, 0, 0, 60, 270, 60, 300, 0, 30, 180, 0, 210, 180, 0, 201, 474, 450, 300, 0, 335, 108, 182, 503, 391, 212, 108, 42, 60, 212, 85, 60, 42, 0, 0, 30, 201, 0, 268, 323, 450, 150, 589, 418, 212, 150, 335, 391, 242, 285, 366, 0, 124, 30, 0, 268, 120, 0, 120, 240, 360, 417, 0, 30, 30, 474, 150, 175, 134, 85, 150, 85, 0, 108, 323, 342, 247, 95, 67, 272, 120, 170, 0, 175, 120, 150, 330, 360, 408, 0, 153, 60, 120, 351, 492, 242, 150, 418, 30, 67, 175, 192, 391, 124, 335, 42, 153, 85, 95, 127, 192, 216, 242, 258, 0, 400, 60, 210, 510, 598, 30, 90, 150, 342, 342, 124, 95, 503, 408, 295, 268, 0, 95, 212, 258, 201, 90, 0, 0, 0, 42, 0, 430, 85, 510, 0, 30, 153, 150, 234, 335, 484, 256, 175, 0, 216, 297, 324, 277, 216, 134, 120, 85, 150, 0, 300, 510, 301, 350, 30, 108, 256, 30, 175, 67, 60, 60, 150, 0, 255, 170, 85, 0, 270, 361, 240, 210, 0, 0, 433, 255, 150, 85, 30, 60, 30, 108, 162, 85, 0, 212, 180, 108, 30, 42, 192, 180, 0, 0, 573, 603, 604, 300, 450, 210, 134, 0, 192, 212, 175, 0, 268, 30, 268, 192, 150, 60, 270, 350, 42, 0, 459, 234, 212, 240, 0, 95, 150, 234, 153, 0, 192, 42, 60, 0, 67, 42, 30, 0, 323, 277, 270, 85, 60, 0, 295, 437, 624, 417, 234, 182, 190, 108, 511, 228, 150, 90, 42, 212, 170, 134, 95, 0, 324, 30, 30, 210, 0, 0, 240, 0, 503, 350, 323, 218, 42, 30, 201, 30, 67, 150, 170, 0, 306, 297, 258, 67, 0, 0, 30, 108, 212, 0, 30, 90, 180, 360, 481, 306, 30, 283, 0, 153, 95, 30, 190, 0, 0, 335, 283, 0, 85, 153, 391, 601, 134, 295, 295, 408, 30, 42, 42, 175, 0, 0, 331, 0, 247, 85, 162, 201, 124, 30, 242, 95, 0, 30, 85, 180, 0, 228, 190, 0, 0, 518, 127, 170, 182, 382, 295, 212, 234, 90, 90, 90, 0, 0, 124, 466, 376, 95, 95, 95, 85, 42, 180, 210, 67, 182, 201, 192, 0, 42, 484, 330, 0, 0, 0, 30, 42, 42, 0, 67, 30, 30, 0, 30, 228, 400, 552, 242, 192, 30, 30, 30, 0, 0, 30, 190, 67, 108, 60, 60, 30, 0, 0, 30, 379, 552, 469, 201, 0, 454, 0, 0, 108, 212, 30, 0, 150, 446, 242, 134, 384, 0, 30, 170, 201, 170, 30, 0, 0, 0, 30, 60, 150, 170, 531, 488, 366, 283, 30, 0, 297, 300, 42, 42, 67, 175, 192, 258, 0, 60, 90, 324, 283, 0, 150, 360, 339, 0, 30, 67, 85, 108, 182, 283, 0, 60, 120, 283, 256, 0, 351, 0, 216, 0, 0, 0, 218, 272, 190, 127, 0, 60, 150, 153, 212, 258, 319, 339, 382, 426, 297, 134, 0, 0, 120, 190, 30, 0, 42, 30, 127, 192, 0, 0, 180, 458, 433, 0, 42, 42, 408, 42, 30, 42, 30, 285, 342, 382, 391, 108, 0, 60, 323, 0, 150, 242, 268, 485, 430, 402, 255, 234, 212, 124, 0, 67, 242, 0, 30, 60, 313, 342, 371, 255, 67, 30, 0, 503, 474, 150, 108, 85, 0, 0, 234, 0, 0, 30, 175, 30, 162, 277, 306, 268, 30, 0, 469, 466, 30, 0, 42, 242, 480, 95, 30, 0, 201, 108, 30, 201, 531, 430, 371, 256, 150, 85, 0, 0, 134, 450, 459, 90, 0, 67, 127, 216, 170, 30, 67, 42, 0, 0, 516, 553, 247, 190, 162, 134, 85, 0, 0, 60, 85, 421, 0, 30, 42, 192, 234, 95, 335, 492, 558, 571, 124, 190, 454, 120, 390, 108, 0, 67, 170, 124, 108, 134, 443, 319, 360, 446, 234, 150, 60, 0, 0, 335, 351, 268, 330, 240, 120, 30, 319, 190, 90, 0, 362, 95, 67, 300, 446, 272, 212, 95, 30, 60, 95, 42, 134, 492, 270, 85, 42, 0, 134, 108, 127, 285, 285, 256, 201, 127, 85, 30, 285, 175, 0, 120, 324, 210, 495, 351, 323, 108, 180, 85, 201, 390, 362, 283, 150, 108, 60, 67, 180, 360, 212, 175, 150, 309, 134, 711, 499, 553, 485, 376, 300, 192, 85, 180, 201, 30, 324, 0, 474, 576, 391, 297, 120, 255, 134, 450, 108, 272, 108, 484, 516, 591, 319, 297, 190, 0, 255, 192, 90, 180, 67, 738, 525, 576, 604, 297, 30, 42, 42, 295, 247, 120, 618, 268, 256, 124, 212, 190, 270, 272, 592, 283, 309, 95, 576, 0, 268, 0, 30, 390, 534, 210, 618, 255, 30, 268, 391, 182, 582, 124, 190, 659, 644, 658, 85, 330, 342, 256, 120, 360, 558, 525, 417, 30, 108, 124, 134, 150, 162, 626, 0, 351, 446, 90, 150, 633, 124, 60, 150, 402, 67, 90, 90, 268, 499, 509, 469, 324, 127, 60, 745, 212, 60, 541, 85, 511, 255, 212, 42, 268, 180, 722, 684, 649, 649, 268, 258, 306, 30, 319, 297, 90, 120, 391, 499, 474, 450, 382, 324, 277, 210, 0, 42, 551, 120, 268, 150, 335, 306, 0, 85, 638, 658, 362, 120, 272, 350, 120, 540, 67, 499, 484, 192, 309, 366, 576, 90, 150, 285, 376, 300, 234, 255, 534, 400, 242, 190, 180, 108, 120, 108, 95, 134, 242, 30, 402, 127, 192, 402, 552, 604, 531, 365, 371, 691, 256, 677, 426, 335, 30, 0, 256, 437, 268, 67, 408, 541, 446, 424, 350, 295, 192, 120, 424, 661, 382, 424, 510, 537, 277, 90, 30, 633, 124, 90, 417, 192, 30, 268, 342, 446, 242, 283, 300, 404, 67, 134, 541, 339, 361, 242, 342, 295, 319, 339, 382, 470, 366, 306, 153, 30, 85, 67, 780, 67, 30, 0, 0, 417, 458, 277, 342, 474, 170, 192, 210, 362, 571, 330, 255, 255, 319, 277, 524, 488, 30, 150, 180, 457, 300, 300, 256, 0, 277, 376, 390, 488, 60, 175, 601, 30, 170, 234, 255, 309, 384, 342, 466, 192, 240, 30, 170, 309, 395, 335, 443, 295, 210, 201, 175, 124, 67, 256, 42, 0, 255, 212, 283, 362, 361, 331, 201, 201, 742, 780, 0, 182, 258, 277, 408, 0, 60, 342, 108, 127, 175, 283, 285, 417, 60, 240, 255, 335, 362, 67, 108, 124, 150, 162, 192, 212, 228, 313, 323, 351, 361, 258, 95, 0, 120, 283, 42, 301, 499, 424, 190, 150, 170, 228, 256, 134, 240, 255, 201, 60, 67, 150, 153, 255, 404, 120, 0, 175, 216, 240, 390, 60, 30, 150, 0, 42, 67, 85, 216, 0, 180, 190, 42, 553, 466, 108, 162, 180, 60, 150, 228, 120, 30, 306, 400, 150, 120, 150, 218, 201, 42, 616, 361, 591, 390, 0, 60, 85, 190, 85, 300, 765, 443, 67, 108, 242, 67, 90, 182, 228, 395, 283, 212, 713, 95, 402, 306, 60, 216, 402, 85, 323, 457, 42, 60, 67, 85, 309, 306, 285, 313, 242, 417, 60, 134, 60, 0, 0, 0, 511, 150, 330, 201, 67, 309, 446, 30, 30, 402, 210, 0, 30, 277, 30, 0, 85, 295, 150, 0, 0, 0, 400, 256, 361, 404, 0, 90, 150, 150, 30, 0, 0, 510, 234, 67, 42, 30, 516, 390, 360, 85, 30, 778, 720, 391, 0, 134, 60, 42, 108, 150, 0, 323, 42, 67, 446, 300, 457, 255, 134, 90, 0, 85, 319, 566, 228, 0, 0, 309, 532, 510, 120, 454, 162, 134, 42, 1045, 774, 42, 67, 210, 30, 42, 379, 408, 495, 525, 277, 150, 571, 153, 67, 255, 437, 124, 30, 384, 42, 30, 0, 127, 283, 268, 201, 180, 201, 537, 497, 30, 67, 162, 365, 488, 67, 0, 30, 1025, 283, 67, 470, 0, 30, 30, 60, 85, 0, 0, 124, 240, 30, 170, 247, 60, 162, 516, 30, 67, 95, 150, 150, 418, 0, 60, 90, 210, 270, 330, 531, 313, 301, 108, 95, 404, 300, 570, 134, 67, 234, 85, 30, 0, 240, 351, 30, 60, 42, 67, 90, 95, 190, 420, 30, 0, 60, 360, 430, 30, 30, 162, 30, 150, 175, 190, 418, 350, 454, 470, 350, 42, 324, 120, 30, 150, 201, 216, 67, 395, 390, 0, 430, 0, 85, 30, 175, 361, 297, 0, 124, 120, 120, 85, 240, 690, 30, 256, 228, 300, 150, 371, 42, 150, 216, 277, 390, 190, 42, 0, 0, 362, 306, 240, 201, 0, 384, 904, 510, 60, 268, 270, 285, 350, 216, 108, 30, 42, 108, 90, 67, 633, 67, 30, 335, 30, 300, 277, 313, 256, 247, 190, 120, 516, 108, 190, 85, 323, 324, 270, 277, 0, 268, 408, 256, 60, 124, 371, 402, 376, 201, 180, 150, 127, 95, 0, 247, 301, 313, 240, 0, 450, 150, 30, 268, 371, 430, 443, 362, 124, 30, 218, 331, 309, 268, 30, 742, 108, 30, 0, 30, 418, 408, 351, 67, 30, 90, 67, 0, 175, 216, 360, 323, 277, 153, 376, 255, 216, 67, 446, 108, 0, 150, 443, 408, 390, 351, 342, 212, 938, 90, 201, 297, 470, 497, 446, 418, 390, 234, 134, 42, 0, 95, 67, 0, 95, 309, 420, 420, 420, 420, 258, 277, 466, 295, 216, 192, 170, 0, 127, 150, 446, 120, 255, 283, 418, 134, 120, 192, 658, 60, 342, 510, 270, 0, 360, 480, 467, 162, 711, 592, 365, 95, 560, 569, 60, 30, 330, 242, 351, 576, 537, 404, 659, 421, 134, 150, 0, 0, 0, 67, 0, 0, 30, 0, 30, 0, 277, 258, 319, 700, 510, 162, 175, 192, 255, 30, 0, 0, 30, 67, 270, 285, 892, 295, 769, 319, 402, 127, 242, 395, 612, 30, 30, 0, 247, 268, 382, 424, 0, 67, 361, 382, 402, 552, 190, 192, 648, 589, 85, 85, 509, 295, 616, 342, 256, 150, 90, 175, 319, 484, 655, 598, 90, 95, 42, 361, 0, 706, 242, 120, 201, 342, 360, 636, 120, 30, 150, 234, 0, 509, 258, 162, 457, 150, 150, 323, 85, 190, 330, 631, 182, 180, 180, 42, 30, 127, 391, 511, 443, 499, 534, 573, 277, 210, 324, 417, 228, 210, 85, 60, 127, 283, 30, 384, 525, 323, 384, 309, 551, 256, 255, 124, 67, 85, 242, 256, 895, 210, 247, 361, 342, 390, 90, 402, 270, 285, 127, 67, 42, 67, 228, 201, 313, 175, 417, 577, 306, 256, 228, 234, 67, 30, 124, 30, 270, 351, 216, 366, 108, 446, 510, 541, 330, 108, 297, 170, 85, 108, 150, 162, 600, 600, 376, 216, 192, 150, 350, 618, 379, 60, 309, 95, 30, 170, 190, 216, 874, 630, 488, 426, 256, 228, 300, 390, 420, 300, 270, 150, 134, 42, 400, 85, 192, 446, 350, 306, 295, 190, 162, 153, 633, 649, 330, 240, 216, 255, 192, 192, 793, 752, 713, 693, 457, 342, 285, 256, 525, 301, 481, 0, 150, 335, 201, 180, 180, 60, 60, 234, 551, 277, 569, 454, 150, 828, 365, 60, 30, 30, 42, 30, 42, 150, 319, 433, 153, 285, 679, 488, 400, 127, 85, 67, 30, 95, 404, 30, 0, 30, 242, 182, 400, 342, 256, 60, 573, 30, 408, 216, 134, 67, 324, 0, 42, 309, 335, 900, 684, 569, 192, 192, 424, 395, 306, 589, 560, 362, 190, 488, 518, 153, 85, 42, 339, 67, 42, 30, 67, 150, 350, 607, 524, 242, 182, 492, 511, 600, 513, 485, 900, 513, 376, 153, 192, 228, 210, 124, 67, 60, 916, 700, 216, 127, 240, 180, 162, 484, 636, 366, 297, 421, 0, 418, 524, 366, 258, 150, 0, 268, 95, 408, 849, 750, 636, 510, 67, 330, 210, 525, 150, 607, 319, 1050, 350, 484, 384, 210, 0, 30, 85, 150, 120, 300, 693, 295, 443, 342, 108, 0, 190, 67, 127, 162, 190, 618, 360, 420, 360, 210, 272, 595, 531, 721, 671, 644, 162, 806, 1047, 457, 295, 175, 150, 190, 90, 150, 150, 216, 258, 390, 270, 180, 30, 690, 624, 417, 750, 534, 957, 663, 313, 268, 60, 0, 175, 85, 190, 362, 421, 361, 242, 735, 376, 924, 216, 339, 192, 42, 67, 541, 424, 124, 335, 800, 350, 0, 85, 842, 824, 1124, 1112, 361, 551, 85, 0, 42, 162, 268, 499, 800, 295, 446, 272, 853, 457, 424, 424, 1045, 474, 319, 339, 361, 484, 391, 0, 30, 67, 379, 351, 702, 361, 247, 488, 499, 470, 30, 902, 883, 510, 1188, 1170, 67, 30, 60, 162, 108, 67, 335, 258, 67, 874, 911, 552, 258, 210, 446, 458, 90, 979, 949, 940, 912, 768, 745, 700, 524, 247, 67, 124, 134, 454, 376, 350, 830, 960, 577, 426, 990, 201, 150, 85, 0, 30, 120, 175, 180, 960, 283, 408, 824, 807, 684, 591, 553, 540, 1260, 60, 30, 0, 67, 150, 300, 474, 268, 256, 175, 1019, 644, 558, 607, 216, 150, 95, 216, 497, 285, 150, 175, 300, 994, 400, 256, 255, 361, 268, 95, 30, 1020, 607, 1343, 495, 175, 90, 0, 85, 127, 218, 60, 162, 902, 722, 424, 67, 60, 90, 1065, 525, 60, 0, 42, 268, 283, 480, 524, 541, 242, 42, 912, 841, 451, 67, 30, 42, 1121, 638, 42, 60, 60, 234, 270, 295, 417, 511, 552, 180, 750, 30, 212, 255, 323, 335, 350, 150, 85, 90, 990, 871, 841, 361, 301, 190, 30, 700, 1124, 1092, 1087, 551, 30, 0, 42, 67, 120, 150, 170, 277, 331, 277, 0, 90, 95, 408, 752, 108, 95, 0, 1100, 999, 960, 755, 30, 339, 379, 510, 417, 201, 60, 636, 150, 0, 85, 1061, 1131, 984, 932, 782, 150, 180, 182, 234, 300, 918, 524, 67, 192, 42, 1022, 1115, 1008, 990, 698, 277, 228, 90, 0, 212, 218, 277, 470, 134, 228, 240, 972, 735, 150, 0, 834, 765, 532, 85, 162, 242, 300, 466, 474, 618, 134, 218, 67, 120, 0, 1148, 1149, 1092, 871, 210, 42, 272, 424, 488, 301, 0, 182, 150, 120, 201, 0, 124, 90, 67, 30, 1047, 618, 874, 190, 67, 95, 384, 589, 270, 330, 295, 242, 0, 60, 0, 949, 969, 1134, 1123, 960, 1165, 212, 426, 553, 240, 124, 313, 900, 1129, 30, 60, 42, 67, 1024, 1015, 1173, 459, 216, 120, 120, 108, 201, 361, 598, 330, 85, 30, 30, 0, 845, 849, 939, 1064, 750, 488, 607, 95, 42, 335, 384, 560, 42, 201, 824, 511, 800, 511, 218, 60, 67, 60, 162, 190, 457, 510, 212, 446, 256, 234, 162, 134, 1003, 469, 510, 182, 30, 127, 150, 390, 525, 60, 108, 190, 120, 60, 256, 765, 1099, 0, 30, 85, 268, 503, 513, 395, 285, 192, 735, 242, 0, 212, 234, 309, 446, 0, 218, 190, 30, 1159, 684, 1127, 108, 30, 60, 60, 85, 268, 242, 95, 402, 306, 108, 1154, 1124, 1095, 918, 570, 371, 90, 306, 540, 175, 150, 418, 85, 162, 95, 0, 1124, 466, 577, 735, 175, 60, 360, 331, 301, 67, 277, 108, 108, 150, 446, 569, 365, 459, 420, 390, 270, 150, 190, 67, 636, 182, 418, 331, 272, 256, 210, 564, 127, 741, 819, 268, 361, 391, 450, 67, 484, 551, 285, 240, 335, 295, 277, 210, 0, 201, 467, 417, 390, 170, 797, 875, 866, 859, 212, 218, 228, 201, 182, 42, 30, 108, 430, 864, 755, 175, 216, 722, 618, 301, 212, 30, 300, 450, 361, 636, 825, 836, 247, 67, 592, 592, 342, 335, 301, 85, 210, 180, 190, 702, 631, 577, 771, 390, 566, 537, 485, 430, 272, 120, 120, 90, 95, 240, 210, 391, 201, 150, 124, 690, 641, 630, 726, 180, 351, 175, 882, 595, 525, 497, 30, 180, 457, 376, 255, 212, 162, 134, 600, 603, 582, 731, 450, 390, 30, 499, 516, 534, 553, 1064, 1075, 443, 418, 408, 390, 335, 331, 182, 42, 443, 335, 309, 258, 124, 630, 607, 570, 488, 540, 210, 361, 430, 443, 492, 1006, 1015, 1036, 365, 361, 30, 42, 120, 323, 600, 560, 553, 540, 430, 693, 511, 421, 182, 295, 319, 875, 531, 927, 942, 1006, 573, 450, 331, 285, 270, 210, 108, 458, 433, 335, 30, 570, 524, 543, 60, 366, 446, 509, 787, 940, 969, 283, 240, 120, 120, 0, 0, 30, 417, 391, 342, 319, 484, 690, 404, 446, 875, 552, 350, 277, 285, 124, 95, 430, 350, 201, 450, 85, 600, 247, 404, 684, 752, 926, 685, 674, 404, 247, 67, 150, 42, 0, 424, 335, 120, 120, 30, 793, 836, 886, 899, 864, 361, 228, 67, 30, 30, 330, 85, 391, 95, 42, 671, 900, 642, 711, 767, 816, 874, 866, 847, 607, 566, 180, 182, 175, 42, 437, 390, 0, 360, 390, 134, 306, 360, 779, 787, 845, 837, 201, 0, 201, 85, 60, 819, 577, 234, 85, 60, 30, 0, 300, 424, 541, 534, 484, 134, 301, 741, 787, 366, 30, 108, 201, 175, 309, 499, 180, 256, 270, 495, 633, 759, 735, 730, 666, 150, 60, 42, 0, 42, 134, 270, 30, 525, 150, 636, 713, 470, 309, 42, 30, 0, 524, 552, 433, 306, 418, 30, 268, 175, 150, 0, 60, 210, 150, 558, 255, 180, 255, 277, 534, 180, 579, 589, 598, 671, 361, 228, 127, 127, 85, 190, 150, 424, 323, 85, 153, 443, 484, 560, 641, 120, 170, 210, 470, 850, 395, 532, 607, 618, 607, 458, 313, 150, 134, 120, 210, 42, 90, 90, 90, 228, 485, 631, 60, 976, 162, 120, 228, 285, 268, 108, 120, 569, 524, 247, 201, 42, 124, 470, 30, 277, 532, 0, 30, 30, 42, 30, 437, 484, 120, 150, 247, 228, 60, 30, 127, 300, 0, 524, 0, 0, 430, 503, 270, 190, 108, 335, 134, 255, 30, 175, 379, 30, 0, 180, 201, 570, 481, 201, 210, 1041, 573, 0, 30, 182, 443, 85, 108, 256, 342, 395, 351, 488, 551, 90, 0, 430, 127, 247, 255, 450, 30, 256, 437, 120, 295, 362, 376, 30, 30, 242, 454, 402, 300, 366, 108, 30, 192, 618, 430, 481, 391, 331, 30, 362, 285, 324, 30, 330, 342, 577, 124, 210, 277, 42, 247, 67, 85, 212, 277, 297, 365, 459, 30, 212, 228, 242, 124, 134, 390, 362, 524, 418, 330, 150, 42, 182, 108, 297, 216, 365, 300, 153, 450, 272, 810, 930, 626, 190, 361, 376, 350, 481, 569, 201, 339, 469, 450, 553, 67, 67, 201, 404, 446, 379, 573, 270, 218, 201, 636, 379, 268, 342, 256, 268, 283, 446, 323, 564, 595, 335, 331, 240, 295, 335, 127, 418, 600, 150, 331, 30, 108, 134, 170, 42, 607, 170, 514, 362, 492, 30, 295, 573, 0, 424, 212, 228, 242, 60, 90, 301, 696, 30, 247, 446, 418, 319, 488, 509, 190, 0, 240, 268, 272, 67, 95, 0, 301, 323, 725, 42, 713, 323, 342, 426, 481, 0, 134, 391, 716, 631, 258, 0, 234, 424, 30, 242, 150, 323, 258, 283, 120, 537, 815, 0, 854, 212, 120, 90, 953, 180, 67, 277, 845, 182, 446, 626, 618, 547, 743, 30, 600, 190, 247, 306, 270, 0, 391, 685, 671, 450, 342, 0, 95, 30, 0, 0, 124, 124, 242, 361, 417, 150, 256, 85, 95, 108, 256, 67, 210, 400, 190, 430, 30, 95, 180, 30, 240, 361, 0, 42, 120, 256, 757, 270, 384, 430, 570, 30, 268, 67, 474, 376, 90, 624, 335, 170, 497, 216, 120, 430, 162, 95, 150, 85, 361, 212, 594, 258, 67, 108, 134, 285, 616, 404, 67, 30, 256, 382, 212, 175, 67, 361, 218, 268, 120, 247, 150, 150, 180, 60, 350, 323, 297, 0, 42, 371, 360, 743, 201, 134, 0, 120, 418, 330, 150, 124, 390, 301, 570, 60, 124, 42, 60, 127, 212, 379, 270, 60, 30, 134, 424, 531, 285, 270, 170, 153, 516, 600, 382, 228, 30, 234, 270, 42, 384, 0, 811, 323, 365, 295, 323, 330, 330, 376, 474, 366, 376, 732, 153, 301, 60, 342, 594, 295, 331, 503, 430, 212, 552, 0, 361, 750, 295, 390, 234, 120, 722, 371, 247, 256, 384, 108, 150, 30, 190, 384, 437, 323, 90, 90, 85, 607, 242, 60, 242, 242, 371, 342, 175, 285, 300, 277, 350, 485, 306, 30, 90, 85, 242, 67, 0, 484, 120, 351, 240, 30, 30, 170, 150, 301, 60, 0, 297, 306, 190, 742, 395, 190, 30, 0, 297, 212, 90, 395, 182, 30, 323, 192, 134, 420, 0, 150, 234, 295, 433, 85, 30, 450, 234, 134, 720, 134, 335, 247, 134, 295, 553, 361, 313, 175, 285, 256, 228, 108, 162, 589, 150, 242, 42, 450, 216, 192, 175, 127, 30, 108, 127, 566, 283, 240, 258, 42, 402, 201, 190, 481, 212, 190, 180, 362, 134, 424, 365, 492, 30, 153, 516, 513, 459, 283, 0, 256, 170, 0, 484, 297, 446, 283, 212, 192, 153, 228, 242, 180, 390, 30, 90, 417, 582, 391, 362, 240, 330, 153, 134, 134, 384, 564, 624, 636, 551, 418, 306, 228, 134, 182, 124, 0, 382, 124, 537, 85, 150, 175, 404, 90, 382, 404, 579, 552, 268, 108, 120, 120, 85, 384, 633, 67, 60, 552, 541, 524, 335, 277, 60, 201, 323, 467, 600, 495, 430, 361, 342, 268, 256, 85, 234, 180, 30, 134, 150, 42, 190, 134, 108, 60, 95, 509, 531, 417, 295, 212, 30, 0, 108, 283, 0, 180, 558, 541, 335, 228, 67, 108, 60, 573, 594, 558, 180, 0, 30, 242, 190, 108, 553, 484, 216, 30, 150, 30, 573, 124, 547, 531, 443, 95, 42, 42, 552, 492, 400, 60, 85, 60, 663, 497, 774, 644, 564, 524, 60, 30, 376, 457, 516, 573, 552, 514, 679, 0, 30, 0, 30, 0, 390, 732, 757, 750, 234, 297, 366, 120, 659, 577, 551, 283, 170, 127, 0, 457, 573, 488, 242, 30, 480, 242, 700, 42, 67, 0, 551, 577, 466, 0, 0, 268, 591, 180, 664, 30, 67, 242, 256, 618, 404, 361, 319, 655, 300, 335, 0, 285, 256, 404, 525, 67, 162, 42, 67, 218, 499, 0, 285, 192, 175, 531, 595, 537, 283, 295, 153, 497, 30, 150, 170, 636, 470, 342, 42, 30, 218, 376, 277, 255, 513, 60, 0, 789, 190, 446, 342, 127, 60, 725, 309, 134, 408, 379, 67, 30, 218, 175, 313, 459, 371, 85, 0, 700, 780, 870, 882, 283, 180, 170, 424, 365, 60, 242, 902, 942, 175, 277, 153, 323, 108, 511, 0, 940, 60, 180, 85, 342, 216, 450, 450, 90, 127, 492, 534, 854, 960, 108, 95, 153, 488, 258, 170, 446, 127, 153, 618, 785, 0, 458, 192, 467, 511, 636, 95, 525, 573, 806, 1027, 60, 30, 0, 0, 60, 255, 458, 120, 85, 592, 1071, 1061, 1050, 1106, 1087, 1082, 277, 242, 216, 120, 255, 277, 190, 182, 180, 153, 60, 0, 30, 534, 108, 150, 190, 175, 162, 108, 1154, 497, 162, 180, 162, 42, 124, 150, 120, 319, 120, 60, 42, 458, 90, 30, 95, 150, 1203, 170, 0, 417, 228, 30, 85, 95, 134, 331, 60, 1261, 1260, 201, 295, 95, 242, 366, 162, 134, 60, 283, 30, 446, 180, 120, 30, 295, 228, 297, 350, 42, 67, 408, 210, 212, 150, 376, 283, 192, 256, 150, 108, 90, 454, 234, 277, 324, 30, 67, 297, 277, 458, 361, 1055, 247, 283, 0, 1208, 1179, 1150, 108, 67, 67, 108, 1213, 60, 30, 153, 42, 60, 277, 390, 42, 418, 430, 1201, 228, 457, 30, 1154, 1087, 417, 120, 379, 437, 295, 361, 698, 1167, 679, 1087, 1129, 150, 1128, 360, 255, 127, 1087, 1032, 1008, 984, 433, 30, 793, 967, 1015, 212, 938, 987, 534, 832, 912, 969, 1041, 150, 234, 210, 67, 805, 816, 30, 875, 886, 902, 979, 1041, 270, 190, 283, 212, 402, 949, 942, 85, 127, 924, 1018, 1050, 658, 270, 242, 1033, 1047, 700, 324, 0, 108, 391, 1057, 1092, 272, 42, 324, 912, 1084, 870, 1032, 841, 626, 618, 153, 1015, 799, 782, 589, 658, 540, 170, 949, 973, 331, 1008, 984, 511, 573, 792, 480, 488, 297, 339, 466, 85, 509, 433, 339, 342, 446, 437, 390, 283, 488, 484, 408, 365, 306, 120, 890, 430, 390, 457, 573, 1003, 400, 446, 175, 313, 573, 1045, 418, 402, 134, 180, 300, 811, 1015, 1068, 330, 335, 816, 335, 301, 240, 240, 782, 313, 1091, 1140, 324, 190, 1060, 1073, 268, 247, 162, 153, 450, 511, 618, 175, 90, 277, 258, 180, 1183, 234, 201, 162, 90, 511, 120, 361, 1140, 134, 67, 679, 30, 90, 1140, 134, 67, 67, 42, 0, 1261, 216, 400, 285, 0, 30, 180, 577, 999, 201, 60, 30, 30, 150, 67, 67, 42, 799, 60, 30, 531, 190, 85, 67, 0, 127, 592, 120, 134, 170, 787, 1294, 85, 30, 67, 153, 658, 150, 234, 60, 60, 60, 85, 60, 127, 497, 162, 30, 124, 150, 0, 30, 108, 30, 180, 120, 430, 285, 319, 42, 150, 531, 402, 60, 297, 192, 721, 30, 42, 85, 30, 150, 150, 108, 60, 30, 684, 598, 376, 85, 42, 0, 242, 170, 350, 240, 42, 108, 162, 190, 170, 268, 120, 42, 0, 228, 67, 402, 0, 234, 30, 90, 618, 579, 42, 124, 30, 42, 150, 90, 301, 384, 42, 30, 60, 900, 488, 408, 120, 60, 150, 277, 30, 350, 488, 309, 201, 216, 120, 60, 190, 638, 67, 0, 283, 67, 752, 180, 124, 95, 420, 0, 170, 659, 108, 153, 484, 366, 60, 30, 67, 85, 192, 95, 306, 201, 85, 170, 0, 42, 175, 426, 268, 180, 30, 85, 0, 90, 150, 0, 124, 277, 60, 170, 446, 582, 228, 201, 150, 150, 335, 467, 30, 60, 192, 60, 30, 67, 384, 85, 589, 90, 255, 600, 573, 85, 60, 150, 661, 201, 240, 297, 124, 95, 684, 875, 150, 319, 150, 0, 60, 182, 95, 67, 350, 216, 240, 201, 95, 42, 391, 30, 247, 162, 134, 175, 228, 150, 646, 836, 630, 360, 180, 85, 192, 418, 824, 0, 641, 210, 180, 150, 124, 212, 175, 95, 0, 417, 210, 162, 85, 30, 457, 497, 247, 90, 150, 283, 309, 888, 960, 150, 120, 90, 120, 541, 124, 960, 630, 212, 30, 722, 360, 108, 60, 30, 192, 743, 120, 379, 150, 120, 42, 90, 457, 67, 30, 162, 361, 201, 547, 335, 360, 330, 60, 162, 408, 67, 612, 134, 309, 124, 655, 90, 902, 573, 124, 268, 716, 270, 90, 664, 283, 511, 721, 228, 883, 150, 30, 182, 268, 808, 201, 830, 541, 402, 210, 488, 997, 750, 270, 134, 120, 30, 513, 85, 932, 150, 537, 900, 210, 67, 162, 899, 67, 658, 120, 30, 240, 789, 760, 124, 85, 60, 124, 966, 752, 201, 537, 458, 247, 30, 242, 153, 636, 973, 924, 900, 446, 618, 60, 285, 228, 90, 306, 134, 492, 0, 42, 175, 124, 95, 890, 816, 607, 85, 242, 162, 85, 180, 127, 268, 732, 108, 247, 120, 162, 819, 765, 685, 42, 570, 706, 690, 792, 510, 0, 134, 277, 395, 272, 510, 514, 240, 382, 767, 787, 192, 832, 234, 192, 726, 779, 805, 324, 553, 685, 937, 577, 360, 443, 691, 618, 564, 534, 365, 313, 240, 510, 658, 450, 240, 335, 466, 480, 566, 390, 595, 659, 402, 691, 467, 95, 228, 120, 420, 649, 443, 607, 210, 404, 492, 469, 277, 216, 573, 446, 524, 537, 450, 658, 636, 255, 175, 175, 150, 335, 467, 525, 595, 323, 301, 470, 573, 162, 90, 108, 360, 511, 256, 283, 509, 218, 484, 297, 446, 484, 731, 0, 408, 323, 335, 457, 408, 30, 30, 150, 402, 458, 285, 150, 150, 283, 85, 30, 295, 693, 127, 201, 648, 342, 162, 0, 150, 283, 361, 607, 150, 180, 180, 85, 108, 242, 313, 524, 480, 134, 210, 218, 162, 85, 108, 162, 201, 300, 234, 390, 228, 247, 466, 150, 256, 362, 270, 182, 218, 210, 255, 309, 190, 201, 153, 481, 661, 277, 255, 216, 443, 120, 693, 424, 408, 474, 0, 90, 90, 162, 342, 95, 60, 60, 446, 277, 524, 552, 306, 362, 42, 0, 474, 390, 30, 30, 402, 67, 30, 42, 366, 470, 255, 30, 67, 60, 67, 433, 488, 342, 95, 551, 0, 402, 0, 516, 313, 306, 180, 541, 524, 525, 553, 666, 42, 283, 242, 240, 242, 256, 553, 552, 573, 474, 335, 90, 228, 323, 350, 234, 42, 816, 162, 552, 806, 658, 309, 134, 0, 361, 400, 443, 594, 210, 95, 0, 382, 484, 516, 255, 424, 30, 0, 134, 458, 85, 258, 0, 268, 551, 573, 0, 60, 376, 488, 30, 108, 335, 418, 531, 30, 0, 150, 216, 351, 569, 430, 212, 108, 255, 90, 424, 30, 509, 0, 342, 120, 124, 67, 800, 134, 0, 480, 484, 0, 270, 759, 60, 212, 285, 342, 242, 218, 277, 182, 212, 391, 0, 180, 511, 270, 540, 210, 228, 570, 270, 360, 633, 484, 297, 600, 466, 247, 30, 467, 216, 268, 633, 162, 228, 361, 484, 218, 182, 495, 134, 658, 870, 319, 277, 90, 42, 30, 242, 757, 192, 295, 0, 0, 108, 808, 258, 218, 30, 175, 759, 362, 658, 765, 295, 642, 395, 134, 212, 674, 324, 240, 153, 247, 600, 170, 297, 210, 90, 277, 30, 660, 319, 212, 30, 108, 60, 0, 170, 30, 30, 30, 607, 417, 134, 216, 201, 295, 212, 228, 807, 234, 162, 212, 726, 228, 418, 684, 190, 90, 192, 108, 67, 120, 474, 240, 309, 234, 323, 309, 285, 201, 361, 242, 170, 0, 256, 134, 162, 190, 67, 218, 175, 124, 150, 443, 426, 60, 210, 30, 30, 30, 0, 90, 60, 0, 150, 90, 0, 0, 0, 0, 0, 30, 30, 0, 190, 450, 90, 0, 607, 488, 532, 42, 127, 240, 497, 0, 153, 124, 0, 319, 466, 242, 30, 30, 30, 424, 270, 124, 67, 446, 366, 470, 424, 618, 192, 335, 0, 525, 192, 95, 85, 0, 153, 67, 67, 297, 190, 0, 30, 0, 30, 0, 0, 30, 0, 0, 30, 90, 60, 268, 0, 240, 0, 0, 127, 30, 182, 108, 0, 67, 85, 0, 175, 95, 180, 60, 361, 120, 0, 90, 30, 0, 0, 376, 150, 30, 42, 42, 212, 192, 108, 30, 30, 60, 60, 60, 150, 228, 182, 234, 90, 60, 95, 95, 120, 42, 30, 175, 150, 150, 90, 216, 234, 201, 150, 162, 0, 150, 42, 362, 90, 85, 30, 30, 182, 120, 60, 85, 67, 342, 153, 30, 170, 134, 30, 0, 0, 234, 0, 127, 234, 190, 60, 90, 134, 108, 67, 240, 85, 85, 85, 127, 228, 30, 228, 306, 384, 30, 361, 30, 30, 30, 446, 247, 175, 313, 30, 0, 0, 0, 124, 153, 150, 30, 330, 379, 30, 0, 42, 60, 42, 390, 283, 95, 0, 30, 108, 127, 391, 255, 95, 153, 30, 134, 210, 268, 42, 258, 297, 255, 153, 0, 216, 108, 95, 85, 201, 153, 150, 85, 30, 120, 30, 42, 170, 67, 0, 134, 60, 306, 150, 350, 42, 335, 67, 120, 0, 361, 30, 30, 95, 60, 120, 300, 324, 30, 170, 362, 180, 309, 30, 67, 192, 120, 60, 30, 134, 258, 30, 60, 0, 108, 210, 331, 95, 85, 0, 95, 192, 134, 30, 30, 323, 361, 134, 283, 212, 120, 247, 134, 0, 335, 342, 212, 30, 42, 175, 0, 228, 306, 331, 277, 30, 0, 277, 120, 210, 192, 108, 42, 42, 268, 295, 256, 120, 210, 60, 182, 228, 134, 124, 42, 30, 0, 277, 277, 95, 277, 120, 255, 95, 212, 127, 108, 234, 0, 0, 192, 30, 0, 0, 42, 124, 108, 0, 150, 192, 67, 30, 108, 67, 67, 180, 190, 30, 228, 42, 90, 108, 30, 0, 42, 30, 150, 42, 30, 30, 60, 0, 30, 60, 277, 0, 0, 42, 0, 150, 210, 60, 85, 0, 67, 30, 85, 201, 67, 60, 42, 30, 30, 270, 295, 90, 0, 30, 0, 134, 60, 240, 300, 0, 30, 124, 42, 153, 90, 30, 60, 120, 42, 190, 108, 170, 134, 60, 150, 60, 60, 0, 150, 120, 90, 190, 402, 210, 210, 180, 153, 153, 150, 30, 417, 242, 0, 300, 268, 240, 331, 120, 306, 400, 270, 270, 120, 457, 301, 300, 301, 90, 60, 300, 450, 390, 331, 240, 247, 418, 391, 234, 190, 268, 323, 268, 324, 360, 180, 300, 319, 216, 437, 272, 295, 150, 210, 60, 247, 268, 30, 212, 150, 150, 30, 180, 234, 297, 60, 108, 150, 90, 30, 0, 0, 60, 192, 30, 0, 108, 95, 285, 0, 0, 42, 234, 42, 95, 30, 60, 127, 0, 295, 335, 218, 242, 0, 30, 0, 60, 150, 309, 0, 90, 120, 150, 150, 60, 30, 170, 324, 180, 180, 0, 60, 228, 228, 319, 242, 180, 201, 228, 120, 124, 234, 242, 242, 153, 170, 192, 277, 270, 360, 30, 212, 268, 331, 330, 323, 124, 85, 256, 283, 134, 324, 270, 256, 212, 268, 182, 124, 0, 108, 153, 192, 85, 297, 42, 150, 60, 90, 309, 300, 216, 67, 376, 242, 85, 30, 42, 300, 127, 0, 170, 124, 417, 120, 42, 30, 335, 342, 258, 228, 150, 319, 242, 153, 124, 534, 180, 216, 446, 390, 335, 234, 0, 361, 467, 488, 509, 242, 216, 108, 30, 323, 108, 67, 85, 594, 514, 30, 0, 127, 601, 0, 417, 90, 0, 60, 85, 182, 95, 85, 90, 404, 518, 342, 95, 95, 124, 633, 700, 124, 674, 560, 175, 216, 190, 180, 671, 342, 210, 212, 323, 335, 324, 270, 301, 313, 582, 331, 408, 351, 598, 510, 450, 365, 285, 541, 503, 488, 454, 210, 175, 497, 390, 390, 390, 361, 175, 120, 30, 95, 30, 212, 153, 212, 0, 150, 242, 150, 30, 95, 153, 0, 60, 210, 85, 67, 60, 60, 150, 108, 67, 228, 127, 95, 42, 430, 85, 234, 108, 270, 30, 295, 301, 60, 134, 108, 0, 474, 108, 0, 30, 454, 180, 150, 67, 300, 153, 297, 30, 134, 277, 30, 234, 376, 319, 541, 323, 124, 0, 95, 212, 446, 216, 297, 153, 361, 212, 806, 566, 134, 600, 268, 67, 30, 0, 95, 42, 162, 190, 480, 127, 297, 67, 511, 0, 633, 67, 1020, 127, 485, 612, 534, 153, 480, 769, 150, 854, 371, 150, 134, 175, 242, 0, 663, 30, 350, 150, 90, 150, 331, 417, 579, 190, 277, 537, 309, 95, 108, 170, 454, 573, 384, 90, 540, 376, 480, 384, 726, 361, 283, 268, 175, 134, 85, 216, 192, 134, 85, 268, 216, 134, 90, 0, 192, 85, 60, 210, 170, 30, 300, 446, 0, 85, 127, 95, 170, 499, 272, 180, 42, 124, 212, 443, 120, 342, 658, 242, 342, 190, 335, 60, 272, 124, 277, 42, 212, 190, 212, 67, 579, 234, 228, 242, 722, 228, 218, 67, 127, 488, 127, 537, 552, 360, 258, 190, 120, 361, 402, 604, 120, 283, 60, 124, 175, 408, 30, 458, 108, 67, 30, 108, 60, 108, 255, 60, 474, 175, 300, 0, 216, 108, 297, 108, 0, 488, 134, 30, 85, 258, 30, 268, 162, 360, 430, 595, 134, 421, 700, 430, 210, 283, 30, 390, 446, 85, 90, 469, 319, 277, 285, 192, 366, 30, 30, 283, 433, 0, 382, 330, 150, 488, 67, 30, 0, 0, 150, 67, 30, 42, 360, 450, 30, 297, 361, 108, 541, 350, 85, 577, 323, 60, 268, 0, 218, 201, 256, 127, 150, 379, 108, 268, 331, 576, 42, 85, 42, 120, 153, 85, 30, 42, 67, 60, 366, 242, 60, 120, 420, 301, 60, 90, 108, 258, 67, 42, 661, 510, 30, 90, 67, 309, 395, 108, 175, 283, 120, 182, 268, 153, 210, 240, 134, 30, 674, 210, 283, 182, 90, 124, 162, 228, 170, 30, 0, 85, 30, 212, 323, 509, 499, 0, 319, 240, 190, 371, 67, 95, 30, 516, 85, 150, 900, 134, 150, 443, 0, 0, 190, 30, 0, 481, 607, 60, 569, 242, 95, 153, 277, 484, 60, 234, 481, 573, 376, 30, 295, 301, 595, 331, 395, 190, 360, 573, 285, 170, 95, 30, 589, 418, 234, 598, 541, 218, 351, 67, 371, 175, 150, 67, 323, 459, 283, 256, 85, 150, 258, 324, 524, 285, 404, 120, 108, 124, 228, 170, 162, 108, 295, 210, 150, 564, 408, 30, 240, 404, 424, 124, 371, 30, 150, 306, 90, 30, 150, 192, 362, 30, 60, 30, 30, 153, 124, 0, 175, 242, 120, 576, 134, 242, 30, 127, 90, 67, 342, 258, 228, 295, 216, 192, 319, 108, 228, 382, 424, 458, 256, 108, 256, 124, 247, 295, 268, 270, 258, 30, 577, 201, 127, 162, 0, 42, 30, 566, 124, 234, 402, 0, 0, 247, 240, 270, 192, 212, 258, 120, 240, 67, 319, 309, 90, 108, 124, 351, 42, 210, 210, 0, 201, 0, 270, 124, 90, 42, 95, 120, 242, 127, 108, 134, 277, 120, 30, 95, 120, 67, 108, 216, 85, 85, 90, 60, 85, 108, 85, 201, 301, 42, 240, 180, 212, 270, 210, 218, 180, 300, 330, 285, 108, 120, 216, 192, 390, 108, 362, 120, 30, 277, 190, 424, 391, 0, 339, 30, 0, 67, 120, 408, 67, 30, 150, 67, 330, 30, 60, 484, 67, 390, 90, 371, 510, 466, 124, 270, 0, 408, 108, 816, 192, 216, 192, 150, 335, 90, 0, 201, 361, 153, 285, 362, 335, 0, 270, 134, 108, 256, 451, 124, 42, 150, 190, 95, 134, 510, 150, 30, 30, 180, 120, 150, 30, 124, 95, 553, 60, 90, 85, 285, 42, 150, 120, 228, 459, 120, 67, 190, 474, 283, 67, 212, 0, 90, 319, 60, 228, 212, 256, 690, 30, 162, 552, 300, 170, 95, 67, 212, 277, 0, 30, 277, 192, 256, 0, 400, 342, 285, 511, 90, 170, 0, 150, 180, 285, 42, 108, 30, 124, 30, 30, 85, 134, 60, 0, 277, 90, 30, 255, 134, 90, 60, 42, 150, 324, 108, 0, 0, 182, 0, 30, 0, 134, 124, 95, 175, 0, 0, 0, 0, 0, 42, 120, 234, 150, 384, 330, 90, 242, 30, 42, 150, 42, 437, 124, 95, 120, 242, 180, 124, 192, 339, 295, 67, 30, 153, 255, 67, 124, 120, 124, 256, 30, 175, 210, 256, 192, 162, 108, 95, 212, 67, 30, 170, 108, 182, 382, 85, 175, 0, 218, 270, 212, 0, 180, 30, 90, 216, 170, 330, 216, 108, 42, 60, 60, 162, 30, 162, 240, 360, 335, 124, 95, 60, 0, 0, 30, 42, 108, 108, 283, 210, 390, 90, 295, 90, 0, 42, 180, 30, 362, 240, 696, 127, 0, 0, 0, 30, 301, 350, 134, 42, 30, 60, 201, 382, 124, 90, 60, 85, 42, 30, 382, 247, 0, 42, 0, 60, 0, 30, 67, 42, 228, 212, 382, 30, 0, 60, 108, 268, 306, 30, 0, 0, 295, 108, 295, 30, 67, 180, 450, 42, 95, 201, 30, 124, 85, 240, 180, 30, 0, 0, 150, 351, 60, 90, 0, 30, 150, 240, 30, 0, 42, 134, 67, 175, 201, 0, 0, 175, 162, 242, 0, 42, 30, 108, 319, 216, 180, 0, 67, 120, 134, 120, 30, 30, 300, 108, 240, 108, 390, 400, 30, 730, 60, 319, 153, 85, 124, 42, 0, 0, 247, 671, 277, 67, 0, 371, 283, 0, 0, 228, 0, 0, 127, 0, 30, 210, 127, 30, 240, 218, 268, 30, 42, 0, 150, 30, 30, 30, 30, 182, 180, 60, 67, 376, 201, 95, 162, 95, 150, 162, 30, 124, 201, 277, 42, 268, 0, 277, 85, 524, 660, 0, 297, 42, 256, 95, 0, 210, 382, 95, 108, 85, 30, 60, 0, 0, 234, 335, 85, 124, 30, 0, 330, 170, 0, 108, 30, 153, 0, 297, 552, 150, 0, 390, 212, 150, 379, 134, 270, 240, 90, 0, 330, 85, 192, 408, 242, 182, 153, 124, 95, 150, 283, 510, 201, 309, 228, 42, 150, 467, 30, 0, 0, 0, 0, 0, 376, 351, 295, 108, 242, 216, 255, 124, 277, 242, 0, 0, 108, 0, 228, 162, 108, 175, 446, 283, 212, 95, 0, 0, 170, 309, 270, 272, 277, 324, 277, 175, 120, 90, 30, 216, 350, 85, 216, 212, 60, 404, 301, 301, 361, 339, 297, 190, 134, 30, 0, 30, 30, 218, 484, 90, 0, 0, 446, 408, 362, 331, 331, 335, 351, 319, 175, 67, 323, 30, 0, 0, 0, 0, 636, 175, 134, 30, 0, 0, 366, 469, 417, 371, 360, 366, 212, 134, 120, 0, 0, 95, 319, 331, 474, 443, 430, 395, 390, 391, 342, 0, 0, 309, 350, 420, 120, 446, 0, 0, 446, 421, 85, 67, 60, 240, 0, 0, 30, 272, 511, 90, 90, 30, 0, 408, 150, 85, 95, 30, 30, 391, 60, 42, 30, 0, 0, 30, 30, 335, 524, 488, 162, 42, 30, 256, 300, 127, 90, 30, 90, 350, 360, 30, 30, 547, 430, 162, 0, 295, 124, 0, 382, 216, 0, 524, 541, 525, 497, 470, 443, 283, 212, 170, 323, 42, 450, 339, 192, 30, 0, 0, 60, 108, 443, 402, 323, 268, 170, 42, 0, 162, 192, 0, 350, 0, 0, 85, 108, 60, 0, 0, 0, 162, 67, 95, 0, 30, 30, 216, 162, 0, 210, 30, 30, 30, 42, 0, 60, 150, 228, 201, 175, 95, 108, 162, 0, 216, 283, 162, 30, 0, 42, 30, 90, 247, 0, 283, 60, 150, 242, 42, 0, 30, 404, 256, 242, 228, 218, 124, 0, 0, 212, 60, 201, 0, 0, 0, 42, 30, 0, 0, 30, 212, 361, 228, 182, 170, 60, 30, 30, 0, 150, 124, 0, 234, 162, 124, 0, 0, 0, 301, 42, 30, 30, 0, 0, 0, 30, 324, 342, 382, 277, 162, 150, 30, 0, 0, 242, 192, 60, 30, 30, 0, 30, 335, 216, 85, 90, 85, 162, 30, 30, 42, 60, 60, 0, 488, 424, 297, 277, 255, 192, 150, 42, 0, 124, 335, 90, 342, 0, 0, 0, 30, 408, 371, 424, 319, 95, 0, 30, 0, 30, 0, 150, 0, 30, 0, 0, 30, 42, 446, 150, 95, 85, 85, 67, 124, 30, 30, 124, 150, 268, 0, 60, 0, 0, 120, 404, 319, 127, 85, 0, 182, 108, 30, 162, 481, 268, 127, 0, 0, 297, 417, 90, 0, 361, 192, 150, 108, 67, 0, 30, 85, 0, 150, 127, 0, 0, 210, 319, 361, 150, 511, 297, 170, 150, 108, 0, 0, 124, 0, 162, 0, 0, 295, 120, 0, 0, 0, 30, 361, 297, 170, 67, 0, 0, 85, 283, 85, 0, 0, 108, 242, 300, 127, 0, 124, 216, 258, 42, 242, 0, 0, 30, 30, 484, 361, 234, 150, 127, 0, 301, 30, 283, 134, 190, 270, 42, 256, 124, 85, 391, 234, 42, 30, 85, 0, 0, 537, 242, 85, 182, 42, 360, 0, 124, 0, 295, 376, 324, 255, 234, 192, 242, 212, 150, 30, 0, 30, 42, 446, 234, 212, 134, 0, 67, 0, 0, 277, 190, 124, 42, 192, 30, 30, 162, 0, 30, 0, 395, 421, 295, 242, 192, 108, 0, 0, 60, 85, 0, 150, 218, 0, 85, 108, 190, 0, 430, 313, 127, 0, 153, 212, 67, 190, 234, 295, 0, 424, 277, 42, 30, 42, 60, 30, 60, 216, 192, 30, 0, 30, 153, 175, 67, 270, 0, 0, 60, 391, 95, 0, 0, 120, 218, 85, 162, 446, 362, 150, 0, 85, 95, 365, 300, 150, 30, 124, 0, 228, 240, 153, 85, 170, 42, 150, 42, 300, 270, 180, 60, 30, 30, 0, 30, 120, 90, 0, 228, 124, 256, 0, 0, 85, 108, 570, 170, 85, 618, 0, 283, 277, 0, 319, 180, 0, 30, 134, 0, 90, 30, 67, 150, 285, 150, 201, 228, 324, 258, 30, 0, 0, 216, 108, 503, 85, 247, 0, 277, 339, 85, 0, 0, 60, 120, 268, 162, 234, 210, 258, 134, 124, 95, 85, 134, 0, 228, 0, 60, 342, 30, 0, 30, 67, 150, 153, 108, 0, 85, 306, 457, 0, 234, 216, 95, 175, 255, 0, 0, 0, 0, 297, 30, 60, 124, 362, 234, 175, 108, 67, 30, 42, 242, 42, 108, 366, 30, 60, 30, 306, 201, 379, 0, 242, 192, 124, 60, 42, 210, 0, 30, 300, 649, 0, 85, 42, 85, 234, 0, 0, 0, 607, 457, 420, 212, 592, 42, 218, 42, 30, 319, 150, 85, 0, 323, 309, 108, 366, 150, 0, 67, 268, 242, 30, 256, 30, 0, 300, 30, 255, 420, 0, 240, 216, 85, 0, 85, 277, 127, 335, 361, 180, 0, 42, 127, 60, 42, 330, 270, 331, 0, 240, 124, 30, 192, 268, 0, 240, 390, 124, 285, 297, 277, 390, 446, 0, 60, 212, 0, 150, 240, 469, 95, 390, 285, 228, 553, 0, 351, 466, 201, 42, 0, 255, 150, 488, 342, 175, 201, 30, 0, 85, 0, 67, 162, 30, 180, 666, 30, 0, 60, 228, 272, 30, 0, 0, 153, 60, 67, 42, 60, 67, 382, 0, 150, 300, 0, 212, 30, 335, 108, 210, 474, 301, 182, 582, 360, 180, 420, 0, 85, 0, 42, 270, 450, 108, 30, 162, 30, 0, 300, 0, 30, 67, 162, 450, 272, 577, 459, 124, 192, 0, 30, 0, 295, 124, 190, 95, 0, 0, 127, 90, 0, 60, 90, 272, 0, 85, 216, 120, 90, 272, 0, 30, 270, 0, 470, 162, 0, 0, 150, 268, 108, 234, 450, 201, 42, 285, 150, 42, 0, 0, 360, 395, 331, 300, 85, 30, 124, 90, 534, 108, 67, 30, 150, 134, 391, 255, 30, 150, 270, 537, 150, 60, 85, 85, 277, 0, 124, 182, 335, 576, 201, 134, 0, 0, 270, 277, 361, 85, 0, 134, 124, 108, 0, 301, 42, 192, 120, 618, 95, 192, 95, 85, 0, 0, 361, 60, 0, 90, 85, 67, 234, 0, 175, 285, 417, 120, 162, 30, 150, 42, 30, 309, 134, 0, 108, 210, 234, 60, 0, 30, 362, 212, 190, 0, 134, 0, 0, 85, 170, 0, 242, 295, 323, 509, 404, 30, 108, 362, 362, 350, 0, 127, 309, 67, 0, 127, 391, 0, 95, 0, 0, 90, 124, 42, 210, 150, 242, 0, 60, 85, 67, 300, 216, 150, 95, 0, 0, 120, 30, 0, 30, 342, 0, 313, 150, 323, 0, 120, 60, 30, 42, 42, 0, 0, 120, 450, 67, 0, 30, 85, 175, 339, 30, 120, 162, 0, 360, 67, 212, 192, 42, 541, 342, 331, 85, 335, 330, 60, 30, 30, 0, 42, 300, 301, 85, 0, 0, 335, 247, 162, 108, 30, 30, 240, 342, 85, 180, 216, 192, 150, 108, 0, 30, 511, 170, 301, 150, 234, 67, 212, 210, 201, 300, 108, 537, 644, 120, 30, 67, 218, 551, 234, 30, 531, 30, 212, 552, 60, 108, 218, 210, 330, 162, 180, 60, 376, 124, 30, 85, 124, 95, 531, 234, 162, 212, 391, 708, 726, 703, 212, 277, 495, 450, 285, 485, 30, 616, 595, 120, 541, 595, 616, 30, 90, 443, 67, 212, 201, 417, 716, 90, 300, 285, 433, 469, 30, 95, 404, 525, 426, 534, 319, 497, 319, 30, 180, 342, 319, 0, 277, 0, 443, 319, 30, 30, 402, 30, 192, 180, 379, 323, 42, 60, 150, 0, 390, 0, 170, 309, 242, 190, 313, 371, 430, 324, 750, 268, 350, 30, 309, 335, 379, 402, 430, 108, 134, 162, 256, 339, 150, 319, 0, 0, 0, 90, 242, 247, 335, 376, 376, 339, 30, 42, 108, 212, 30, 335, 30, 420, 134, 90, 0, 454, 228, 30, 212, 376, 170, 127, 488, 0, 30, 182, 306, 30, 150, 0, 458, 30, 518, 60, 85, 426, 216, 0, 30, 0, 67, 30, 0, 0, 362, 150, 0, 60, 0, 0, 309, 30, 67, 541, 42, 153, 95, 42, 459, 180, 430, 395, 134, 395, 382, 297, 108, 182, 85, 240, 362, 120, 0, 67, 95, 108, 85, 42, 120, 210, 488, 283, 108, 192, 124, 0, 127, 108, 150, 994, 342, 283, 120, 0, 190, 598, 180, 295, 446, 300, 300, 212, 330, 120, 0, 443, 90, 30, 90, 30, 42, 256, 60, 42, 150, 376, 190, 120, 124, 335, 295, 313, 323, 30, 811, 212, 30, 295, 124, 162, 210, 182, 258, 424, 457, 192, 150, 162, 175, 376, 726, 382, 661, 150, 319, 30, 30, 0, 234, 525, 330, 127, 268, 30, 30, 30, 902, 350, 481, 582, 42, 0, 0, 0, 124, 582, 60, 42, 875, 300, 247, 362, 319, 319, 484, 90, 162, 234, 696, 212, 120, 30, 108, 872, 216, 124, 607, 579, 120, 150, 277, 162, 150, 108, 190, 799, 552, 741, 190, 210, 595, 376, 426, 256, 698, 671, 618, 30, 60, 242, 295, 30, 558, 335, 297, 30, 60, 85, 516, 402, 0, 42, 300, 192, 458, 150, 120, 323, 300, 210, 285, 451, 391, 216, 30, 319, 306, 108, 362, 108, 768, 408, 408, 295, 313, 577, 268, 30, 0, 285, 569, 540, 242, 120, 228, 703, 190, 553, 162, 120, 297, 283, 242, 240, 907, 532, 108, 42, 420, 503, 234, 192, 824, 726, 595, 175, 458, 466, 454, 684, 430, 210, 170, 330, 90, 713, 362, 497, 162, 108, 722, 228, 360, 755, 85, 67, 42, 446, 552, 175, 361, 30, 30, 384, 285, 760, 240, 0, 395, 450, 95, 95, 150, 283, 162, 492, 30, 874, 787, 451, 67, 335, 216, 1218, 424, 402, 430, 234, 390, 362, 1318, 268, 120, 42, 134, 990, 543, 170, 242, 277, 366, 990, 30, 108, 313, 175, 815, 371, 351, 283, 1022, 30, 0, 503, 182, 990, 272, 42, 162, 67, 1138, 957, 85, 0, 30, 658, 153, 60, 108, 162, 564, 1140, 30, 134, 339, 962, 514, 484, 395, 1084, 1170, 787, 67, 150, 361, 295, 0, 666, 60, 800, 787, 612, 319, 256, 60, 108, 162, 30, 992, 162, 30, 382, 85, 247, 30, 150, 577, 228, 30, 212, 424, 180, 60, 0, 270, 162, 696, 240, 210, 0, 1087, 492, 182, 95, 990, 558, 67, 42, 351, 488, 789, 306, 95, 242, 42, 466, 361, 430, 277, 1106, 124, 95, 488, 1104, 297, 153, 1150, 1090, 335, 277, 360, 180, 342, 258, 603, 331, 391, 360, 366, 541, 420, 674, 360, 180, 631, 295, 324, 240, 300, 451, 789, 256, 150, 1006, 300, 60, 319, 641, 750, 642, 931, 301, 150, 300, 458, 525, 779, 582, 342, 277, 228, 658, 840, 1082, 85, 603, 638, 457, 60, 589, 967, 180, 182, 170, 153, 469, 150, 228, 216, 85, 480, 666, 376, 234, 361, 819, 212, 190, 516, 837, 558, 212, 210, 0, 212, 108, 859, 859, 726, 757, 845, 450, 90, 247, 162, 180, 680, 175, 60, 242, 212, 323, 805, 659, 731, 752, 90, 350, 270, 30, 95, 85, 552, 256, 268, 624, 573, 30, 85, 120, 258, 566, 67, 485, 618, 655, 90, 210, 150, 350, 0, 180, 0, 604, 30, 618, 90, 127, 95, 190, 150, 541, 319, 90, 95, 616, 390, 30, 30, 430, 210, 190, 120, 0, 150, 0, 0, 30, 240, 60, 0, 162, 212, 242, 268, 247, 120, 390, 277, 268, 134, 295, 914, 0, 85, 170, 272, 30, 313, 85, 242, 0, 108, 60, 127, 30, 255, 361, 499, 95, 671, 108, 162, 424, 85, 216, 323, 256, 60, 335, 454, 760, 794, 309, 30, 569, 779, 120, 297, 210, 319, 342, 778, 731, 420, 277, 331, 297, 335, 0, 313, 351, 371, 474, 162, 0, 30, 212, 0, 872, 30, 376, 30, 175, 0, 0, 210, 256, 511, 362, 30, 60, 216, 342, 450, 525, 120, 641, 67, 180, 67, 85, 108, 182, 108, 547, 60, 255, 446, 335, 450, 430, 30, 90, 95, 30, 120, 509, 182, 679, 741, 283, 379, 731, 335, 30, 300, 573, 270, 30, 162, 433, 684, 342, 190, 484, 300, 301, 474, 309, 127, 319, 324, 335, 323, 300, 150, 192, 42, 234, 242, 170, 371, 60, 395, 60, 335, 127, 330, 258, 242, 361, 272, 95, 331, 153, 67, 192, 313, 67, 488, 60, 234, 247, 0, 134, 503, 342, 212, 335, 182, 649, 42, 616, 421, 162, 175, 418, 201, 242, 124, 201, 210, 124, 150, 716, 361, 426, 0, 0, 95, 153, 95, 228, 534, 218, 90, 90, 175, 247, 376, 558, 541, 285, 90, 60, 552, 524, 390, 234, 162, 443, 108, 295, 30, 376, 531, 90, 782, 277, 390, 124, 446, 666, 277, 42, 342, 589, 228, 566, 30, 67, 85, 467, 90, 270, 404, 108, 30, 604, 648, 577, 30, 618, 564, 0, 268, 350, 376, 170, 108, 510, 30, 499, 30, 323, 150, 210, 418, 85, 895, 127, 875, 864, 990, 658, 30, 342, 108, 108, 743, 382, 638, 1050, 228, 175, 467, 534, 162, 319, 153, 30, 1140, 60, 192, 499, 60, 1184, 42, 30, 0, 297, 190, 42, 0, 256, 134, 192, 309, 127, 300, 404, 228, 201, 60, 134, 150, 150, 120, 67, 268, 424, 1154, 1172, 1179, 531, 509, 446, 1189, 127, 192, 108, 1114, 658, 1077, 1131, 1103, 309, 860, 1074, 882, 899, 953, 1020, 1065, 847, 256, 870, 847, 124, 953, 957, 834, 892, 1062, 918, 891, 553, 510, 480, 558, 124, 234, 446, 459, 395, 721, 417, 395, 361, 300, 350, 216, 594, 212, 324, 162, 247, 295, 190, 313, 120, 270, 1199, 1087, 0, 430, 30, 42, 150, 42, 150, 323, 0, 124, 0, 201, 323, 256, 124, 0, 330, 85, 492, 234, 124, 277, 175, 67, 108, 134, 212, 150, 134, 67, 150, 127, 85, 67, 42, 201, 60, 30, 0, 95, 108, 875, 516, 0, 170, 582, 85, 0, 365, 60, 42, 481, 0, 127, 488, 566, 108, 270, 283, 127, 395, 212, 594, 285, 124, 755, 779, 134, 242, 268, 150, 362, 90, 30, 228, 175, 60, 201, 134, 60, 331, 270, 150, 228, 900, 67, 752, 547, 242, 700, 190, 0, 90, 285, 295, 904, 175, 162, 60, 277, 60, 242, 242, 658, 120, 90, 216, 60, 30, 277, 150, 190, 150, 437, 90, 90, 60, 42, 212, 124, 120, 730, 962, 258, 351, 201, 780, 42, 690, 150, 124, 67, 150, 443, 741, 540, 633, 743, 750, 743, 331, 324, 240, 600, 732, 42, 242, 270, 426, 497, 120, 175, 323, 418, 95, 342, 765, 524, 636, 0, 618, 350, 787, 247, 510, 256, 90, 190, 589, 285, 212, 270, 618, 755, 180, 175, 360, 283, 671, 240, 618, 277, 30, 0, 484, 443, 67, 95, 277, 618, 108, 446, 537, 234, 67, 85, 67, 509, 759, 509, 531, 342, 376, 426, 390, 255, 531, 509, 95, 0, 457, 424, 283, 474, 366, 150, 513, 30, 60, 553, 153, 499, 30, 30, 400, 0, 301, 391, 277, 134, 323, 240, 446, 190, 0, 391, 300, 443, 309, 0, 0, 175, 277, 192, 42, 666, 285, 277, 430, 0, 295, 150, 190, 30, 499, 212, 90, 242, 658, 300, 277, 108, 417, 153, 150, 30, 190, 309, 90, 0, 234, 60, 0, 551, 0, 323, 60, 190, 30, 108, 30, 0, 201, 201, 0, 67, 85, 134, 127, 0, 331, 0, 331, 182, 30, 420, 42, 0, 0, 120, 30, 124, 60, 95, 67, 42, 108, 201, 0, 218, 0, 30, 127, 180, 150, 134, 255, 30, 85, 153, 124, 190, 127, 124, 30, 0, 95, 404, 0, 60, 0, 0, 170, 162, 108, 0, 42, 42, 42, 351, 30, 60, 42, 0, 430, 301, 390, 95, 360, 175, 390, 85, 277, 108, 42, 0, 277, 42, 124, 277, 30, 0, 85, 0, 30, 190, 90, 0, 256, 0, 85, 234, 30, 67, 85, 309, 170, 60, 85, 212, 150, 90, 30, 201, 0, 0, 242, 85, 272, 30, 90, 30, 108, 134, 150, 0, 60, 120, 67, 95, 30, 30, 42, 30, 42, 42, 60, 0, 95, 95, 85, 0, 42, 0, 170, 30, 175, 90, 85, 67, 30, 150, 90, 362, 335, 120, 134, 376, 362, 180, 339, 277, 391, 272, 270, 90, 60, 30, 371, 95, 300, 30, 379, 277, 212, 323, 228, 342, 360, 212, 67, 285, 0, 212, 335, 277, 212, 60, 30, 120, 90, 67, 95, 0, 42, 376, 30, 30, 0, 255, 242, 0, 0, 0, 297, 150, 201, 67, 242, 190, 120, 240, 30, 42, 256, 342, 30, 240, 190, 255, 30, 175, 30, 134, 108, 60, 90, 134, 277, 42, 150, 319, 0, 67, 30, 192, 192, 234, 256, 467, 150, 283, 85, 421, 150, 524, 330, 120, 0, 67, 60, 124, 443, 624, 258, 633, 365, 218, 319]}],
                        {"autosize": false, "height": 500, "margin": {"b": 80, "l": 40, "r": 30, "t": 100}, "paper_bgcolor": "rgb(243, 243, 243)", "plot_bgcolor": "rgb(243, 243, 243)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Distance To Hydrology"}, "width": 700},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('ef689e60-fe28-4c15-aa89-b5f96cc3cc40');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>

{% endraw %}


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

{% raw %}
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="e8e384c4-b5f6-4ba5-ac9b-280d1579eaf8" class="plotly-graph-div" style="height:500px; width:700px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("e8e384c4-b5f6-4ba5-ac9b-280d1579eaf8")) {
                    Plotly.newPlot(
                        'e8e384c4-b5f6-4ba5-ac9b-280d1579eaf8',
                        [{"marker": {"color": "rgb(255,111,145)"}, "name": "Hillshade Noon", "type": "box", "y": [232, 235, 238, 238, 234, 237, 225, 230, 221, 219, 243, 240, 224, 247, 240, 225, 239, 227, 232, 228, 223, 253, 209, 222, 221, 237, 243, 236, 238, 210, 201, 231, 233, 227, 220, 195, 207, 229, 183, 219, 234, 235, 223, 197, 236, 225, 229, 232, 227, 231, 198, 205, 234, 215, 227, 212, 219, 230, 232, 227, 230, 226, 228, 224, 210, 221, 226, 214, 227, 225, 244, 208, 223, 223, 218, 221, 233, 234, 230, 231, 223, 235, 187, 210, 224, 206, 197, 226, 221, 234, 233, 206, 218, 246, 221, 218, 243, 231, 233, 221, 222, 220, 216, 219, 242, 245, 230, 203, 216, 237, 226, 227, 233, 231, 231, 222, 209, 227, 232, 211, 232, 220, 231, 220, 190, 228, 230, 222, 221, 229, 219, 233, 211, 200, 216, 234, 221, 226, 219, 224, 254, 232, 231, 224, 238, 238, 212, 251, 229, 228, 231, 228, 227, 217, 254, 253, 223, 233, 209, 212, 245, 242, 223, 213, 219, 250, 242, 226, 251, 250, 230, 211, 225, 199, 226, 227, 222, 233, 221, 232, 237, 236, 215, 221, 238, 231, 182, 221, 234, 242, 219, 246, 241, 233, 234, 227, 188, 242, 228, 232, 217, 231, 224, 205, 250, 230, 215, 227, 202, 219, 200, 245, 214, 224, 216, 204, 221, 213, 176, 244, 201, 245, 218, 252, 217, 231, 244, 222, 227, 249, 241, 243, 252, 211, 231, 205, 234, 210, 187, 217, 225, 238, 225, 234, 195, 215, 181, 252, 240, 208, 234, 234, 229, 245, 218, 229, 231, 210, 237, 192, 233, 231, 220, 224, 245, 217, 246, 239, 245, 251, 238, 233, 239, 240, 226, 237, 220, 225, 241, 238, 221, 217, 218, 227, 196, 223, 234, 223, 215, 225, 213, 220, 222, 240, 234, 216, 227, 234, 201, 203, 245, 248, 244, 229, 192, 219, 212, 247, 247, 230, 231, 200, 228, 225, 221, 238, 200, 233, 249, 230, 220, 231, 220, 225, 244, 236, 244, 214, 227, 241, 223, 252, 240, 240, 201, 240, 212, 219, 235, 224, 246, 230, 213, 217, 245, 238, 248, 211, 231, 234, 223, 203, 229, 242, 241, 230, 230, 235, 238, 193, 235, 234, 224, 222, 229, 234, 197, 228, 228, 247, 227, 240, 231, 250, 251, 248, 221, 231, 233, 249, 237, 224, 238, 246, 237, 234, 217, 200, 234, 234, 236, 235, 220, 234, 232, 231, 225, 220, 231, 244, 234, 231, 234, 247, 228, 208, 249, 244, 236, 218, 216, 225, 218, 210, 234, 235, 232, 208, 224, 217, 214, 211, 232, 231, 229, 213, 218, 208, 218, 214, 225, 235, 243, 218, 230, 214, 212, 209, 233, 229, 219, 213, 236, 241, 228, 234, 247, 239, 230, 226, 231, 240, 237, 235, 190, 236, 225, 214, 245, 225, 236, 237, 232, 228, 250, 224, 239, 215, 238, 219, 215, 228, 203, 245, 244, 232, 239, 245, 164, 219, 233, 241, 246, 244, 229, 225, 246, 214, 219, 230, 239, 242, 230, 237, 229, 230, 233, 241, 237, 222, 211, 230, 242, 242, 156, 254, 229, 225, 195, 222, 253, 229, 226, 228, 216, 239, 220, 233, 230, 242, 238, 239, 234, 224, 244, 248, 224, 229, 217, 218, 229, 217, 210, 246, 252, 207, 234, 225, 219, 230, 224, 248, 240, 229, 225, 209, 248, 223, 223, 208, 240, 224, 253, 213, 247, 199, 211, 242, 236, 227, 229, 209, 222, 200, 221, 203, 192, 235, 216, 213, 208, 245, 241, 215, 202, 232, 232, 220, 244, 241, 218, 231, 245, 234, 235, 230, 204, 208, 246, 230, 206, 240, 237, 246, 234, 235, 234, 247, 198, 247, 241, 222, 220, 214, 218, 224, 214, 225, 213, 228, 232, 208, 216, 216, 222, 231, 242, 225, 211, 212, 217, 219, 208, 209, 244, 233, 210, 213, 238, 235, 237, 230, 208, 233, 225, 235, 234, 230, 202, 228, 227, 227, 222, 215, 194, 246, 204, 188, 222, 223, 232, 225, 229, 231, 232, 216, 223, 224, 224, 234, 218, 227, 224, 192, 223, 214, 237, 231, 218, 229, 183, 236, 194, 233, 223, 188, 209, 233, 219, 241, 227, 236, 217, 237, 216, 226, 227, 186, 194, 201, 228, 176, 205, 203, 194, 188, 184, 202, 201, 210, 213, 231, 221, 232, 204, 229, 232, 210, 218, 215, 221, 238, 224, 212, 224, 222, 174, 247, 202, 204, 225, 233, 208, 216, 222, 210, 231, 232, 232, 244, 183, 242, 211, 203, 246, 234, 215, 245, 214, 221, 237, 242, 226, 206, 218, 227, 202, 252, 225, 221, 221, 241, 211, 247, 228, 223, 209, 226, 208, 225, 221, 245, 237, 227, 220, 251, 221, 243, 225, 219, 176, 235, 226, 215, 239, 233, 229, 227, 207, 206, 207, 234, 234, 199, 253, 234, 230, 236, 231, 223, 225, 217, 229, 230, 216, 229, 206, 228, 247, 239, 213, 226, 228, 231, 229, 225, 225, 209, 209, 240, 241, 217, 238, 233, 223, 226, 221, 207, 213, 230, 214, 236, 222, 220, 211, 241, 215, 230, 208, 214, 214, 209, 222, 211, 228, 229, 219, 231, 224, 234, 201, 216, 223, 229, 214, 222, 235, 241, 228, 217, 221, 219, 230, 225, 238, 234, 231, 209, 205, 253, 245, 236, 207, 233, 242, 233, 225, 223, 239, 235, 219, 251, 211, 223, 227, 226, 247, 218, 218, 213, 223, 222, 211, 220, 141, 228, 223, 240, 241, 234, 229, 237, 241, 244, 253, 218, 238, 238, 233, 244, 219, 226, 245, 230, 219, 212, 217, 234, 233, 239, 239, 219, 218, 223, 217, 238, 214, 216, 210, 230, 216, 231, 243, 246, 227, 247, 219, 211, 218, 224, 224, 234, 233, 221, 211, 221, 199, 213, 207, 209, 241, 241, 206, 211, 201, 217, 211, 228, 235, 215, 231, 234, 222, 222, 211, 216, 225, 245, 225, 234, 227, 213, 220, 231, 230, 231, 243, 219, 228, 228, 240, 242, 241, 230, 223, 235, 228, 231, 238, 225, 196, 204, 227, 237, 228, 219, 232, 209, 218, 230, 230, 250, 230, 231, 224, 200, 219, 208, 238, 198, 238, 237, 225, 251, 203, 225, 227, 220, 249, 220, 236, 233, 239, 232, 222, 225, 197, 233, 247, 229, 208, 199, 244, 232, 241, 214, 229, 230, 223, 201, 209, 204, 244, 228, 214, 232, 226, 245, 250, 218, 238, 226, 186, 247, 247, 231, 247, 215, 233, 205, 218, 247, 223, 242, 217, 246, 213, 220, 237, 238, 217, 239, 235, 224, 219, 246, 239, 224, 213, 240, 200, 220, 238, 236, 247, 198, 218, 234, 225, 227, 230, 247, 226, 207, 230, 246, 224, 241, 242, 216, 214, 221, 227, 228, 231, 251, 215, 229, 223, 216, 227, 231, 223, 223, 186, 241, 218, 235, 247, 233, 236, 218, 252, 229, 228, 224, 206, 232, 250, 206, 215, 224, 208, 214, 198, 229, 219, 252, 226, 226, 219, 223, 246, 229, 224, 234, 230, 231, 211, 219, 202, 225, 215, 215, 225, 226, 213, 219, 212, 231, 232, 217, 231, 220, 215, 200, 197, 203, 206, 196, 211, 221, 223, 219, 222, 217, 204, 239, 245, 196, 217, 222, 213, 216, 220, 189, 208, 219, 214, 217, 237, 230, 219, 223, 186, 218, 203, 247, 236, 183, 184, 193, 218, 207, 227, 184, 180, 213, 219, 216, 232, 196, 188, 179, 217, 215, 247, 182, 193, 216, 207, 245, 237, 220, 226, 230, 229, 232, 192, 208, 207, 231, 237, 212, 187, 213, 198, 200, 201, 214, 245, 194, 185, 163, 186, 184, 198, 239, 233, 236, 163, 247, 168, 179, 178, 226, 183, 192, 202, 245, 242, 172, 185, 184, 218, 205, 183, 182, 157, 176, 168, 182, 205, 222, 239, 241, 201, 166, 164, 171, 177, 242, 167, 248, 238, 221, 187, 237, 180, 180, 154, 196, 240, 213, 183, 184, 231, 164, 230, 163, 150, 178, 148, 158, 154, 213, 230, 236, 231, 204, 199, 147, 226, 198, 149, 241, 231, 245, 196, 236, 224, 235, 187, 244, 236, 230, 232, 173, 244, 210, 159, 233, 227, 237, 224, 167, 225, 239, 233, 163, 229, 234, 239, 226, 224, 229, 227, 238, 165, 233, 226, 165, 242, 205, 230, 187, 236, 199, 215, 233, 247, 218, 200, 185, 197, 222, 239, 193, 232, 229, 192, 178, 235, 173, 166, 211, 242, 226, 170, 175, 186, 227, 237, 205, 222, 141, 235, 156, 243, 205, 173, 187, 189, 237, 219, 189, 240, 187, 184, 182, 183, 233, 208, 193, 231, 222, 201, 180, 193, 208, 234, 215, 218, 216, 236, 170, 217, 221, 224, 226, 235, 107, 227, 231, 172, 201, 111, 183, 243, 113, 239, 218, 199, 103, 166, 224, 172, 111, 183, 245, 198, 99, 205, 169, 246, 247, 225, 220, 208, 113, 160, 223, 223, 186, 99, 183, 154, 211, 218, 130, 155, 153, 234, 227, 119, 172, 159, 213, 124, 118, 245, 116, 181, 165, 212, 229, 233, 211, 229, 121, 168, 162, 161, 239, 160, 227, 171, 165, 232, 226, 208, 225, 238, 154, 240, 232, 177, 156, 201, 242, 233, 162, 164, 226, 237, 171, 175, 223, 220, 216, 214, 165, 171, 217, 212, 168, 232, 180, 232, 170, 234, 230, 180, 226, 173, 184, 236, 125, 227, 191, 223, 233, 173, 180, 242, 234, 229, 230, 168, 184, 223, 237, 225, 210, 178, 185, 235, 237, 240, 230, 233, 179, 238, 161, 177, 214, 222, 162, 182, 186, 203, 233, 241, 221, 186, 179, 233, 186, 244, 174, 220, 230, 223, 194, 189, 234, 231, 230, 226, 244, 244, 236, 203, 184, 191, 234, 221, 207, 210, 186, 174, 196, 190, 242, 237, 223, 230, 221, 180, 234, 185, 177, 222, 235, 228, 171, 173, 176, 161, 166, 170, 181, 178, 225, 169, 235, 239, 210, 161, 240, 173, 169, 170, 200, 173, 167, 168, 168, 229, 167, 243, 243, 234, 225, 234, 222, 231, 224, 223, 155, 164, 232, 161, 218, 161, 238, 162, 158, 197, 160, 232, 231, 157, 246, 240, 236, 158, 165, 157, 193, 167, 164, 219, 217, 242, 236, 239, 157, 170, 158, 225, 240, 247, 248, 245, 178, 221, 236, 216, 216, 186, 162, 219, 244, 229, 164, 168, 238, 241, 246, 249, 239, 237, 231, 210, 163, 202, 216, 198, 248, 248, 244, 241, 222, 232, 240, 246, 229, 236, 186, 184, 171, 251, 243, 240, 190, 184, 181, 184, 241, 242, 244, 251, 252, 232, 191, 182, 212, 219, 240, 237, 249, 185, 186, 205, 214, 232, 231, 253, 242, 186, 191, 194, 197, 203, 238, 236, 249, 230, 217, 224, 200, 187, 194, 196, 198, 227, 236, 237, 247, 243, 227, 224, 211, 189, 204, 194, 223, 250, 248, 233, 238, 242, 199, 193, 196, 200, 194, 207, 208, 214, 243, 231, 198, 192, 197, 212, 221, 213, 242, 218, 222, 227, 242, 188, 186, 190, 234, 198, 225, 224, 205, 187, 181, 180, 207, 239, 238, 233, 225, 215, 205, 172, 230, 235, 213, 226, 224, 176, 199, 208, 218, 174, 228, 230, 223, 224, 211, 253, 221, 211, 176, 177, 174, 212, 200, 184, 188, 243, 217, 210, 189, 227, 233, 199, 206, 200, 208, 240, 231, 215, 203, 207, 224, 233, 211, 209, 179, 197, 194, 237, 234, 232, 236, 216, 212, 215, 220, 214, 205, 214, 235, 215, 219, 198, 180, 219, 219, 223, 227, 204, 181, 173, 202, 216, 236, 190, 170, 163, 235, 192, 220, 214, 240, 225, 203, 215, 239, 208, 197, 246, 219, 231, 220, 211, 223, 225, 223, 238, 192, 213, 199, 199, 190, 189, 224, 238, 173, 183, 224, 173, 167, 231, 214, 218, 223, 179, 195, 215, 225, 178, 250, 143, 150, 224, 184, 210, 205, 217, 221, 241, 156, 211, 216, 231, 187, 207, 184, 196, 230, 197, 185, 172, 188, 228, 225, 222, 195, 214, 222, 222, 188, 205, 208, 209, 213, 207, 193, 203, 230, 239, 183, 225, 210, 186, 246, 234, 234, 177, 221, 213, 192, 244, 219, 173, 211, 202, 199, 202, 155, 208, 242, 179, 186, 218, 213, 205, 192, 234, 230, 230, 189, 196, 241, 246, 229, 214, 230, 234, 224, 236, 217, 222, 214, 183, 183, 224, 224, 194, 207, 204, 240, 223, 245, 245, 194, 238, 217, 253, 219, 212, 214, 214, 201, 207, 192, 218, 228, 241, 229, 235, 213, 226, 212, 195, 206, 196, 223, 190, 209, 231, 243, 219, 211, 227, 226, 244, 215, 211, 215, 206, 182, 188, 205, 244, 203, 244, 229, 228, 197, 191, 203, 180, 204, 207, 196, 219, 215, 200, 230, 215, 234, 230, 218, 196, 188, 242, 213, 195, 202, 226, 212, 223, 188, 212, 218, 187, 211, 210, 188, 186, 252, 247, 193, 229, 224, 219, 208, 203, 210, 226, 187, 222, 211, 169, 162, 186, 196, 209, 234, 250, 218, 211, 221, 229, 216, 210, 207, 203, 232, 216, 216, 218, 226, 200, 209, 216, 176, 222, 248, 214, 219, 214, 221, 222, 214, 187, 200, 221, 197, 200, 200, 204, 182, 241, 224, 207, 225, 188, 210, 234, 195, 164, 211, 212, 214, 213, 239, 195, 210, 221, 231, 188, 220, 244, 219, 231, 212, 240, 208, 206, 237, 186, 183, 228, 196, 213, 207, 234, 189, 199, 186, 214, 233, 248, 230, 172, 229, 215, 251, 250, 177, 176, 168, 216, 210, 194, 210, 188, 203, 218, 171, 217, 224, 225, 200, 247, 228, 183, 173, 197, 204, 220, 194, 196, 209, 228, 219, 218, 214, 199, 168, 167, 202, 169, 179, 203, 207, 215, 171, 190, 173, 201, 185, 202, 217, 217, 228, 188, 197, 246, 157, 211, 207, 209, 210, 212, 236, 253, 231, 205, 175, 152, 177, 184, 223, 191, 248, 236, 190, 162, 203, 215, 203, 217, 209, 197, 198, 214, 203, 224, 137, 210, 177, 160, 149, 177, 219, 211, 195, 192, 252, 251, 250, 220, 178, 174, 180, 220, 189, 236, 210, 135, 168, 214, 234, 243, 143, 195, 181, 191, 194, 230, 224, 220, 221, 248, 248, 252, 253, 236, 152, 182, 175, 213, 231, 218, 203, 198, 215, 176, 130, 208, 241, 209, 201, 248, 249, 250, 246, 251, 207, 209, 212, 218, 245, 177, 222, 242, 133, 188, 249, 210, 171, 148, 228, 225, 220, 207, 204, 245, 247, 248, 251, 251, 204, 236, 246, 234, 227, 141, 184, 142, 207, 203, 195, 223, 220, 217, 200, 246, 251, 246, 244, 243, 197, 207, 214, 244, 193, 172, 178, 157, 162, 138, 232, 228, 229, 229, 185, 243, 244, 244, 244, 243, 247, 251, 244, 233, 205, 207, 212, 193, 206, 199, 207, 178, 154, 214, 207, 227, 229, 190, 242, 241, 239, 241, 243, 244, 242, 244, 217, 225, 230, 223, 226, 247, 202, 222, 204, 155, 167, 220, 199, 229, 197, 243, 244, 236, 227, 245, 248, 244, 205, 200, 200, 208, 231, 133, 174, 229, 224, 229, 203, 223, 190, 224, 227, 163, 165, 216, 221, 229, 231, 236, 239, 243, 239, 242, 242, 244, 249, 246, 217, 206, 217, 208, 214, 214, 193, 176, 203, 208, 194, 236, 237, 154, 178, 222, 226, 225, 204, 241, 241, 247, 243, 243, 241, 214, 217, 180, 141, 126, 207, 181, 204, 205, 175, 225, 217, 227, 251, 201, 220, 180, 188, 198, 154, 209, 189, 227, 222, 214, 214, 217, 226, 195, 216, 213, 183, 209, 172, 167, 190, 244, 232, 252, 220, 226, 226, 225, 226, 220, 220, 220, 218, 243, 242, 218, 215, 174, 211, 240, 234, 187, 179, 197, 180, 206, 242, 254, 245, 211, 199, 195, 215, 217, 243, 244, 247, 227, 155, 160, 244, 214, 228, 214, 254, 240, 222, 202, 219, 218, 216, 217, 221, 223, 218, 216, 253, 217, 231, 226, 212, 201, 191, 244, 247, 250, 225, 231, 233, 232, 198, 159, 173, 253, 254, 200, 253, 250, 213, 198, 174, 200, 186, 180, 220, 237, 218, 199, 217, 238, 252, 227, 238, 240, 237, 236, 235, 168, 187, 217, 249, 223, 238, 243, 222, 240, 240, 251, 251, 195, 184, 164, 254, 226, 214, 215, 244, 242, 239, 231, 193, 200, 246, 209, 246, 227, 219, 218, 212, 240, 244, 211, 197, 205, 190, 208, 237, 192, 223, 233, 234, 235, 230, 225, 140, 189, 193, 193, 245, 229, 243, 217, 183, 228, 190, 189, 248, 229, 178, 204, 216, 222, 222, 138, 178, 247, 198, 156, 171, 147, 140, 181, 252, 179, 251, 244, 223, 216, 226, 238, 241, 250, 208, 196, 185, 206, 247, 177, 120, 162, 162, 225, 245, 223, 214, 228, 227, 221, 235, 236, 241, 252, 202, 242, 203, 193, 222, 217, 208, 172, 187, 212, 245, 153, 194, 184, 166, 164, 243, 244, 228, 224, 223, 222, 233, 229, 187, 238, 183, 212, 206, 215, 222, 223, 236, 253, 253, 240, 233, 235, 226, 230, 248, 178, 216, 187, 198, 216, 211, 155, 211, 208, 231, 233, 224, 209, 190, 184, 175, 200, 238, 248, 234, 233, 231, 235, 237, 244, 177, 196, 237, 225, 221, 198, 172, 158, 127, 135, 163, 151, 203, 244, 245, 235, 209, 196, 174, 158, 154, 253, 210, 219, 211, 233, 210, 190, 163, 151, 239, 251, 252, 250, 251, 226, 245, 242, 223, 220, 195, 215, 210, 181, 203, 205, 206, 157, 201, 211, 151, 251, 249, 224, 244, 248, 200, 151, 173, 215, 244, 246, 193, 207, 229, 141, 224, 136, 153, 184, 137, 140, 195, 251, 238, 230, 242, 239, 241, 246, 246, 247, 169, 198, 176, 212, 241, 254, 245, 172, 216, 199, 160, 185, 225, 156, 213, 242, 229, 248, 243, 242, 240, 212, 167, 248, 221, 210, 243, 254, 184, 152, 202, 219, 230, 241, 244, 243, 217, 232, 239, 243, 242, 253, 167, 173, 192, 229, 149, 194, 154, 168, 170, 251, 242, 232, 228, 224, 249, 215, 210, 242, 214, 161, 193, 218, 181, 217, 161, 250, 251, 232, 237, 194, 192, 214, 193, 227, 213, 133, 150, 161, 170, 172, 220, 151, 220, 222, 167, 181, 161, 254, 243, 222, 192, 222, 236, 227, 172, 166, 152, 157, 164, 190, 209, 211, 149, 198, 170, 172, 192, 183, 253, 234, 228, 197, 196, 199, 235, 233, 236, 192, 150, 154, 168, 207, 213, 187, 220, 249, 249, 253, 232, 222, 223, 208, 198, 232, 235, 157, 181, 254, 237, 250, 209, 239, 230, 222, 203, 130, 146, 174, 179, 151, 137, 226, 162, 216, 155, 221, 194, 198, 142, 138, 237, 242, 215, 199, 253, 211, 135, 182, 178, 170, 169, 143, 157, 221, 250, 207, 212, 214, 215, 251, 166, 159, 160, 168, 170, 171, 177, 163, 168, 197, 233, 147, 241, 247, 211, 229, 225, 238, 230, 243, 248, 249, 150, 181, 190, 182, 174, 174, 177, 180, 183, 185, 187, 230, 225, 207, 184, 170, 224, 212, 210, 253, 161, 154, 249, 252, 214, 254, 250, 253, 245, 237, 175, 155, 146, 225, 234, 224, 214, 249, 252, 247, 173, 168, 191, 192, 159, 158, 166, 173, 191, 203, 225, 229, 190, 230, 208, 169, 230, 254, 242, 224, 199, 254, 221, 226, 229, 218, 231, 248, 247, 248, 239, 186, 169, 179, 186, 194, 207, 206, 193, 158, 160, 177, 189, 197, 228, 215, 221, 208, 200, 201, 203, 168, 223, 221, 232, 234, 239, 188, 180, 175, 185, 254, 252, 222, 147, 170, 168, 133, 126, 230, 225, 247, 191, 217, 234, 247, 254, 173, 175, 191, 195, 207, 191, 163, 177, 187, 202, 228, 224, 189, 191, 225, 240, 249, 245, 234, 242, 245, 240, 235, 227, 246, 252, 251, 238, 237, 229, 229, 219, 166, 240, 234, 229, 225, 251, 253, 252, 244, 239, 221, 196, 219, 232, 253, 248, 195, 173, 178, 185, 197, 208, 203, 179, 170, 186, 215, 232, 167, 194, 194, 229, 252, 210, 210, 240, 243, 244, 245, 239, 244, 234, 228, 254, 237, 156, 229, 239, 212, 245, 247, 251, 249, 249, 230, 236, 251, 252, 244, 192, 173, 173, 199, 200, 206, 208, 211, 191, 184, 225, 244, 235, 248, 189, 198, 253, 244, 230, 244, 242, 246, 193, 220, 253, 239, 232, 217, 187, 206, 239, 238, 249, 249, 251, 253, 246, 243, 205, 210, 227, 239, 252, 253, 194, 200, 207, 180, 161, 210, 249, 252, 248, 241, 245, 238, 203, 217, 222, 236, 238, 181, 169, 236, 251, 230, 235, 249, 251, 243, 248, 194, 225, 254, 251, 250, 166, 157, 167, 190, 209, 214, 214, 195, 180, 215, 241, 151, 167, 208, 190, 176, 177, 210, 228, 233, 207, 236, 242, 227, 216, 243, 239, 239, 242, 235, 208, 212, 247, 239, 191, 195, 175, 166, 175, 251, 179, 227, 223, 226, 219, 236, 253, 253, 242, 249, 233, 188, 207, 247, 251, 249, 169, 159, 168, 193, 199, 205, 213, 207, 194, 218, 210, 202, 202, 200, 189, 233, 232, 209, 182, 251, 218, 211, 249, 238, 238, 244, 246, 217, 249, 223, 220, 192, 162, 231, 231, 209, 223, 244, 227, 215, 214, 222, 206, 228, 254, 239, 199, 188, 196, 204, 209, 208, 215, 216, 223, 140, 217, 206, 177, 248, 221, 241, 246, 249, 250, 226, 169, 217, 216, 226, 222, 188, 224, 229, 240, 225, 209, 202, 242, 252, 253, 195, 206, 229, 252, 253, 250, 254, 227, 190, 190, 198, 211, 209, 234, 223, 236, 202, 142, 219, 211, 214, 207, 186, 196, 222, 187, 254, 245, 245, 240, 237, 244, 243, 233, 235, 209, 247, 223, 232, 200, 204, 218, 196, 199, 209, 237, 254, 253, 243, 201, 206, 228, 252, 253, 249, 222, 188, 174, 178, 183, 197, 206, 207, 217, 224, 229, 222, 205, 203, 219, 216, 200, 171, 161, 192, 174, 194, 223, 223, 234, 190, 248, 223, 251, 238, 235, 212, 240, 244, 229, 226, 224, 243, 241, 209, 216, 185, 222, 205, 183, 204, 251, 253, 210, 205, 222, 248, 251, 248, 220, 185, 191, 198, 204, 193, 201, 157, 204, 219, 192, 225, 219, 217, 253, 254, 248, 243, 243, 236, 235, 229, 236, 221, 223, 181, 231, 231, 242, 235, 221, 223, 228, 210, 212, 232, 247, 220, 194, 205, 225, 253, 254, 233, 220, 216, 206, 243, 251, 221, 209, 202, 195, 189, 182, 187, 200, 215, 239, 227, 184, 217, 217, 157, 215, 238, 246, 242, 238, 247, 240, 247, 225, 215, 225, 222, 241, 223, 223, 213, 223, 220, 249, 212, 210, 202, 185, 182, 221, 192, 152, 156, 153, 204, 202, 241, 183, 223, 252, 242, 243, 226, 186, 215, 234, 211, 188, 133, 219, 217, 229, 242, 249, 242, 229, 220, 244, 247, 241, 221, 217, 214, 199, 211, 144, 187, 189, 164, 204, 196, 230, 228, 175, 215, 241, 244, 248, 236, 192, 218, 238, 206, 189, 158, 219, 217, 220, 221, 241, 232, 209, 212, 218, 210, 213, 210, 254, 214, 225, 237, 235, 224, 228, 211, 188, 211, 223, 252, 216, 186, 204, 222, 227, 245, 250, 253, 242, 242, 200, 170, 157, 195, 182, 204, 229, 216, 206, 208, 225, 230, 215, 228, 219, 206, 244, 240, 196, 202, 230, 194, 254, 254, 229, 219, 232, 247, 179, 232, 240, 253, 249, 239, 234, 253, 224, 234, 234, 227, 169, 169, 168, 176, 177, 195, 195, 195, 196, 216, 232, 228, 216, 219, 230, 169, 249, 239, 234, 180, 233, 223, 234, 238, 237, 246, 253, 227, 213, 215, 228, 249, 242, 237, 231, 229, 159, 214, 195, 203, 208, 204, 208, 195, 219, 200, 186, 187, 213, 235, 221, 199, 221, 253, 247, 240, 225, 209, 251, 252, 247, 222, 225, 248, 211, 195, 200, 229, 241, 245, 226, 231, 241, 236, 238, 202, 218, 223, 221, 205, 207, 203, 199, 209, 198, 229, 240, 230, 205, 163, 155, 239, 240, 221, 220, 224, 240, 240, 241, 251, 206, 227, 250, 216, 225, 187, 230, 221, 220, 211, 228, 229, 236, 244, 187, 193, 190, 215, 225, 233, 229, 229, 221, 213, 204, 214, 209, 204, 201, 195, 205, 221, 252, 233, 227, 230, 222, 169, 169, 194, 250, 222, 192, 253, 231, 243, 249, 231, 183, 210, 244, 254, 252, 208, 145, 189, 184, 211, 220, 227, 243, 204, 163, 243, 242, 217, 229, 226, 225, 221, 217, 206, 203, 200, 200, 219, 231, 230, 231, 232, 250, 225, 227, 229, 233, 199, 214, 206, 202, 217, 252, 239, 227, 239, 237, 224, 201, 170, 241, 247, 213, 149, 217, 195, 240, 252, 250, 222, 225, 227, 225, 221, 219, 220, 219, 205, 228, 229, 247, 174, 218, 233, 225, 214, 229, 230, 189, 215, 226, 249, 218, 228, 252, 219, 232, 190, 226, 243, 246, 251, 249, 239, 238, 249, 224, 239, 146, 243, 249, 206, 204, 226, 244, 250, 251, 222, 224, 222, 220, 218, 222, 225, 194, 213, 254, 224, 225, 229, 224, 225, 213, 211, 194, 192, 205, 202, 195, 221, 180, 252, 209, 222, 249, 242, 246, 227, 214, 230, 219, 216, 231, 230, 223, 214, 207, 198, 248, 204, 232, 247, 157, 136, 192, 249, 247, 206, 200, 211, 232, 249, 234, 215, 220, 214, 218, 224, 219, 220, 225, 224, 219, 205, 224, 182, 220, 181, 186, 251, 248, 237, 243, 218, 233, 236, 224, 147, 224, 250, 251, 153, 172, 221, 209, 222, 209, 200, 218, 241, 251, 253, 222, 218, 209, 217, 221, 223, 225, 224, 219, 219, 216, 185, 204, 237, 247, 232, 227, 225, 210, 226, 197, 217, 212, 181, 245, 248, 247, 219, 208, 225, 233, 231, 244, 226, 222, 149, 203, 183, 211, 238, 239, 149, 134, 225, 246, 224, 231, 232, 217, 206, 198, 204, 225, 248, 254, 250, 233, 198, 205, 216, 226, 235, 227, 208, 188, 214, 198, 228, 233, 225, 229, 217, 180, 186, 221, 216, 198, 212, 212, 228, 230, 231, 244, 243, 245, 245, 215, 220, 220, 222, 223, 236, 225, 236, 241, 235, 248, 219, 153, 196, 228, 138, 128, 212, 240, 219, 221, 222, 219, 201, 199, 199, 243, 254, 236, 214, 200, 192, 204, 226, 222, 208, 173, 210, 216, 219, 226, 213, 219, 197, 204, 216, 235, 238, 244, 247, 242, 236, 241, 241, 219, 233, 226, 199, 188, 218, 241, 235, 242, 234, 252, 232, 248, 179, 168, 230, 214, 215, 221, 225, 223, 215, 204, 238, 254, 219, 211, 201, 217, 205, 188, 209, 224, 236, 233, 213, 218, 229, 230, 220, 180, 216, 168, 181, 161, 167, 217, 203, 222, 152, 199, 207, 233, 237, 221, 227, 199, 197, 221, 216, 242, 240, 250, 250, 207, 139, 170, 152, 226, 211, 209, 214, 216, 197, 196, 197, 228, 249, 216, 212, 217, 214, 230, 220, 176, 227, 233, 212, 217, 219, 218, 219, 215, 199, 179, 223, 230, 204, 221, 227, 237, 237, 234, 215, 213, 241, 184, 222, 211, 212, 215, 210, 209, 209, 204, 202, 196, 208, 244, 234, 211, 222, 234, 226, 189, 171, 220, 228, 231, 216, 141, 155, 172, 180, 176, 179, 179, 218, 237, 230, 226, 229, 232, 235, 228, 217, 219, 187, 217, 187, 242, 231, 188, 187, 220, 212, 209, 215, 211, 204, 207, 211, 205, 189, 243, 233, 218, 214, 216, 225, 234, 236, 204, 219, 227, 229, 222, 211, 143, 165, 176, 170, 180, 188, 204, 214, 221, 224, 228, 228, 225, 233, 226, 225, 222, 220, 215, 222, 213, 249, 247, 239, 236, 219, 229, 213, 218, 219, 249, 226, 237, 218, 231, 235, 240, 212, 209, 203, 202, 213, 215, 213, 193, 206, 228, 229, 225, 230, 224, 222, 212, 220, 230, 197, 228, 213, 216, 205, 138, 170, 186, 227, 218, 229, 230, 225, 211, 214, 229, 229, 235, 242, 242, 228, 209, 207, 232, 194, 188, 218, 229, 234, 226, 237, 214, 223, 185, 233, 246, 239, 218, 203, 202, 209, 205, 204, 214, 225, 230, 234, 233, 225, 224, 233, 202, 222, 217, 218, 202, 188, 173, 191, 231, 223, 206, 180, 197, 184, 180, 231, 226, 218, 227, 227, 230, 216, 207, 246, 240, 221, 230, 231, 236, 175, 193, 227, 226, 228, 236, 227, 225, 214, 201, 204, 206, 209, 210, 211, 221, 231, 225, 218, 222, 214, 225, 231, 189, 176, 217, 226, 226, 207, 194, 218, 211, 210, 232, 208, 222, 231, 208, 223, 212, 212, 169, 220, 240, 230, 239, 244, 248, 234, 240, 224, 224, 212, 206, 206, 199, 209, 218, 233, 232, 229, 189, 236, 205, 228, 223, 216, 226, 211, 211, 223, 216, 196, 217, 189, 158, 239, 217, 233, 232, 177, 199, 218, 220, 227, 241, 191, 210, 208, 245, 220, 219, 211, 207, 206, 207, 211, 205, 201, 226, 229, 229, 230, 207, 213, 227, 233, 234, 214, 224, 212, 210, 217, 202, 197, 187, 189, 213, 155, 169, 228, 241, 168, 224, 204, 224, 226, 228, 231, 166, 221, 168, 199, 211, 209, 219, 194, 195, 215, 216, 220, 213, 216, 214, 207, 201, 202, 206, 210, 206, 233, 234, 231, 221, 216, 228, 223, 224, 225, 232, 218, 229, 223, 206, 206, 205, 210, 196, 193, 191, 178, 171, 217, 238, 205, 200, 201, 213, 220, 200, 202, 195, 205, 197, 194, 168, 246, 240, 230, 203, 212, 197, 196, 207, 214, 223, 231, 229, 223, 227, 243, 229, 236, 231, 222, 227, 232, 231, 200, 232, 225, 224, 223, 194, 196, 193, 187, 213, 209, 173, 197, 211, 226, 222, 214, 234, 237, 234, 192, 215, 227, 224, 223, 166, 221, 225, 200, 234, 220, 211, 205, 198, 188, 184, 197, 195, 201, 208, 230, 231, 224, 197, 188, 218, 236, 227, 192, 210, 229, 225, 231, 226, 183, 185, 212, 190, 223, 221, 218, 172, 217, 225, 172, 163, 240, 232, 230, 207, 199, 222, 167, 212, 233, 216, 225, 213, 234, 214, 178, 234, 168, 155, 205, 211, 204, 201, 182, 198, 202, 228, 228, 226, 216, 203, 238, 212, 220, 190, 203, 189, 184, 235, 195, 225, 202, 169, 239, 225, 220, 231, 225, 233, 242, 250, 239, 166, 203, 217, 205, 200, 202, 198, 190, 187, 184, 202, 225, 231, 190, 228, 223, 206, 207, 243, 246, 163, 179, 216, 205, 183, 216, 230, 173, 192, 186, 160, 167, 235, 236, 190, 178, 236, 234, 221, 189, 207, 215, 227, 222, 232, 211, 236, 242, 243, 245, 170, 244, 244, 221, 222, 214, 207, 193, 192, 186, 195, 185, 193, 203, 216, 226, 165, 203, 212, 220, 228, 218, 181, 216, 208, 214, 172, 209, 236, 203, 146, 178, 223, 185, 190, 224, 215, 225, 227, 204, 203, 187, 221, 248, 212, 242, 177, 206, 180, 188, 198, 200, 196, 196, 199, 204, 212, 228, 212, 217, 214, 218, 200, 204, 218, 155, 232, 180, 190, 216, 160, 232, 154, 235, 223, 224, 222, 218, 234, 209, 240, 232, 187, 222, 236, 226, 207, 176, 179, 188, 194, 197, 202, 210, 216, 224, 215, 216, 224, 225, 202, 229, 224, 193, 242, 174, 231, 230, 234, 235, 234, 204, 187, 226, 203, 233, 159, 167, 200, 235, 235, 205, 222, 182, 182, 196, 206, 206, 210, 219, 233, 235, 226, 198, 169, 221, 229, 209, 234, 211, 226, 225, 193, 204, 206, 243, 194, 199, 180, 222, 189, 218, 241, 233, 239, 210, 209, 228, 211, 189, 157, 195, 226, 209, 189, 190, 196, 210, 217, 217, 213, 222, 197, 204, 222, 225, 224, 222, 200, 191, 192, 182, 147, 152, 225, 221, 191, 231, 220, 216, 217, 178, 189, 175, 213, 232, 253, 249, 247, 207, 212, 183, 174, 192, 195, 199, 219, 225, 211, 199, 218, 222, 221, 229, 237, 207, 232, 212, 217, 192, 191, 208, 216, 214, 200, 195, 223, 238, 165, 218, 234, 222, 213, 237, 240, 241, 233, 233, 234, 200, 196, 215, 234, 177, 225, 220, 194, 202, 178, 186, 204, 230, 218, 204, 211, 222, 220, 218, 210, 225, 221, 233, 230, 225, 207, 211, 205, 227, 225, 211, 211, 208, 221, 160, 223, 171, 172, 239, 239, 236, 220, 219, 236, 252, 250, 235, 235, 234, 224, 212, 197, 186, 183, 190, 195, 202, 210, 215, 228, 240, 224, 224, 235, 228, 211, 147, 204, 201, 221, 221, 222, 199, 204, 181, 221, 238, 240, 212, 229, 239, 224, 196, 188, 185, 191, 199, 208, 213, 232, 226, 158, 214, 216, 229, 230, 231, 221, 233, 224, 212, 222, 199, 157, 142, 229, 238, 197, 188, 226, 235, 232, 229, 235, 191, 219, 209, 196, 188, 197, 207, 227, 211, 202, 229, 220, 223, 226, 210, 203, 233, 206, 199, 167, 233, 230, 234, 216, 239, 229, 219, 224, 221, 192, 181, 150, 235, 248, 238, 211, 195, 194, 194, 196, 204, 216, 198, 222, 208, 231, 224, 226, 229, 226, 211, 199, 168, 206, 226, 248, 234, 181, 172, 166, 230, 232, 229, 184, 183, 190, 251, 246, 211, 235, 218, 197, 198, 198, 198, 200, 210, 178, 230, 231, 224, 213, 226, 224, 223, 223, 212, 196, 206, 233, 232, 170, 221, 160, 159, 226, 232, 232, 232, 230, 227, 232, 230, 222, 217, 207, 160, 244, 254, 248, 225, 201, 199, 202, 200, 199, 241, 203, 221, 204, 219, 229, 183, 206, 225, 226, 220, 204, 211, 208, 192, 159, 146, 151, 231, 224, 231, 230, 221, 222, 204, 228, 218, 189, 216, 207, 209, 210, 223, 231, 215, 219, 215, 227, 227, 229, 224, 218, 241, 211, 170, 160, 222, 229, 183, 222, 217, 177, 205, 229, 196, 211, 219, 223, 224, 220, 215, 167, 217, 212, 205, 156, 230, 240, 209, 210, 214, 211, 191, 237, 227, 238, 236, 246, 250, 243, 223, 190, 230, 234, 236, 216, 195, 193, 219, 179, 214, 209, 215, 223, 214, 215, 216, 235, 184, 199, 210, 242, 236, 212, 193, 244, 246, 236, 195, 179, 249, 231, 229, 236, 215, 223, 227, 213, 216, 218, 215, 190, 214, 216, 202, 226, 237, 219, 238, 231, 205, 203, 195, 251, 239, 212, 226, 214, 215, 234, 223, 199, 210, 227, 204, 231, 228, 225, 217, 218, 210, 184, 208, 233, 194, 253, 249, 254, 218, 230, 212, 207, 195, 236, 218, 187, 222, 234, 227, 228, 219, 221, 233, 245, 212, 170, 218, 225, 225, 224, 222, 251, 229, 220, 223, 220, 215, 209, 230, 214, 212, 192, 209, 214, 223, 201, 227, 229, 218, 223, 235, 235, 246, 232, 211, 155, 219, 172, 201, 226, 213, 243, 220, 228, 237, 209, 243, 218, 211, 194, 221, 223, 215, 210, 241, 207, 215, 228, 220, 224, 224, 231, 242, 192, 228, 213, 223, 230, 232, 202, 208, 230, 213, 195, 199, 209, 217, 207, 198, 202, 226, 251, 205, 232, 230, 224, 221, 230, 194, 178, 240, 215, 212, 204, 221, 223, 203, 217, 199, 214, 215, 206, 231, 244, 245, 229, 208, 209, 223, 231, 234, 244, 208, 214, 193, 214, 185, 170, 197, 201, 217, 219, 221, 216, 232, 243, 232, 246, 211, 217, 220, 217, 224, 221, 225, 232, 238, 232, 211, 210, 226, 220, 213, 220, 200, 197, 201, 208, 207, 217, 213, 226, 242, 245, 199, 215, 226, 213, 228, 230, 243, 189, 186, 239, 179, 229, 218, 232, 173, 160, 171, 207, 203, 198, 178, 196, 216, 198, 235, 247, 236, 241, 244, 197, 198, 211, 214, 211, 241, 209, 252, 245, 230, 246, 185, 212, 190, 231, 219, 215, 214, 201, 237, 216, 205, 244, 227, 237, 243, 211, 199, 236, 226, 247, 228, 210, 188, 238, 226, 228, 215, 210, 210, 195, 252, 229, 245, 247, 186, 222, 237, 230, 228, 223, 221, 216, 214, 213, 206, 222, 193, 187, 180, 197, 206, 235, 194, 196, 208, 225, 195, 202, 237, 184, 237, 239, 241, 228, 223, 206, 220, 208, 220, 214, 220, 211, 186, 206, 211, 229, 237, 236, 226, 204, 223, 227, 223, 220, 208, 212, 222, 218, 237, 228, 204, 201, 184, 185, 210, 213, 191, 236, 232, 208, 231, 241, 219, 239, 221, 233, 237, 220, 218, 236, 175, 195, 210, 209, 200, 193, 220, 239, 229, 197, 197, 235, 235, 224, 231, 227, 202, 229, 235, 220, 227, 225, 221, 228, 242, 218, 251, 246, 213, 214, 234, 234, 236, 229, 203, 195, 247, 227, 220, 213, 210, 231, 205, 227, 212, 196, 246, 213, 207, 211, 226, 228, 237, 229, 193, 199, 193, 224, 244, 226, 220, 213, 214, 218, 222, 213, 235, 216, 205, 218, 196, 241, 241, 198, 190, 236, 233, 195, 192, 215, 214, 214, 207, 218, 220, 209, 227, 186, 238, 210, 195, 184, 241, 196, 225, 218, 230, 209, 201, 207, 217, 228, 217, 224, 234, 212, 210, 215, 229, 231, 225, 202, 236, 224, 209, 230, 229, 201, 207, 228, 250, 236, 226, 228, 238, 205, 225, 219, 230, 242, 209, 204, 213, 208, 206, 199, 199, 221, 211, 224, 200, 196, 227, 241, 250, 253, 251, 247, 226, 220, 225, 234, 221, 204, 247, 225, 198, 206, 214, 226, 246, 248, 245, 235, 245, 234, 207, 236, 235, 233, 225, 218, 222, 206, 233, 225, 196, 185, 194, 231, 215, 216, 243, 206, 234, 221, 210, 202, 202, 231, 253, 237, 176, 214, 209, 218, 218, 224, 231, 229, 213, 219, 199, 197, 240, 217, 189, 208, 219, 216, 203, 217, 223, 197, 233, 226, 207, 211, 216, 213, 221, 205, 217, 223, 233, 230, 217, 186, 199, 196, 233, 235, 222, 205, 224, 213, 213, 219, 228, 225, 185, 245, 232, 208, 234, 229, 227, 190, 214, 222, 234, 209, 211, 203, 202, 204, 202, 198, 212, 233, 227, 225, 204, 219, 217, 229, 238, 237, 187, 232, 246, 205, 223, 226, 228, 250, 246, 203, 163, 226, 245, 179, 192, 249, 189, 227, 220, 213, 233, 225, 226, 229, 201, 237, 251, 239, 236, 240, 245, 246, 228, 226, 228, 219, 238, 227, 221, 246, 251, 254, 239, 237, 234, 184, 214, 226, 223, 232, 240, 229, 224, 222, 235, 232, 199, 232, 228, 199, 216, 194, 247, 247, 244, 191, 223, 224, 222, 235, 233, 226, 223, 232, 243, 197, 241, 227, 218, 189, 190, 189, 212, 227, 235, 222, 217, 214, 234, 228, 231, 239, 239, 191, 247, 204, 216, 206, 205, 169, 244, 212, 211, 211, 216, 225, 227, 228, 224, 224, 213, 224, 225, 234, 216, 244, 207, 249, 201, 220, 219, 171, 247, 212, 221, 224, 226, 183, 218, 222, 249, 217, 229, 196, 214, 172, 163, 243, 240, 241, 251, 230, 199, 170, 183, 197, 205, 236, 227, 226, 248, 230, 234, 235, 207, 188, 176, 187, 229, 234, 209, 245, 222, 227, 228, 185, 237, 239, 188, 189, 184, 232, 250, 223, 241, 187, 224, 192, 228, 242, 222, 222, 224, 220, 157, 156, 197, 206, 219, 213, 216, 241, 225, 226, 227, 178, 220, 178, 232, 225, 227, 230, 227, 228, 231, 190, 227, 210, 243, 233, 217, 214, 241, 205, 216, 221, 226, 226, 233, 226, 198, 236, 184, 222, 188, 243, 220, 221, 224, 231, 247, 189, 206, 185, 234, 237, 247, 218, 220, 218, 227, 228, 222, 223, 246, 186, 225, 225, 211, 228, 221, 216, 212, 174, 231, 183, 246, 230, 216, 223, 215, 226, 226, 240, 183, 254, 244, 237, 222, 244, 239, 240, 226, 240, 237, 241, 230, 172, 217, 244, 222, 228, 235, 171, 219, 212, 223, 252, 244, 199, 220, 218, 240, 226, 208, 182, 225, 238, 220, 225, 221, 223, 252, 249, 247, 244, 245, 225, 231, 239, 220, 251, 246, 223, 228, 237, 234, 189, 249, 246, 229, 250, 228, 218, 211, 219, 199, 247, 240, 252, 230, 207, 170, 217, 214, 202, 196, 217, 208, 246, 242, 248, 227, 222, 224, 235, 240, 208, 211, 235, 205, 231, 218, 217, 215, 203, 212, 196, 201, 221, 214, 237, 239, 207, 202, 211, 204, 237, 231, 248, 250, 207, 251, 228, 211, 232, 234, 246, 220, 221, 212, 208, 250, 249, 250, 245, 229, 212, 209, 208, 201, 249, 223, 209, 221, 218, 228, 240, 238, 243, 227, 236, 211, 216, 205, 251, 254, 245, 250, 249, 218, 218, 236, 195, 210, 250, 226, 246, 222, 240, 226, 226, 209, 253, 251, 251, 249, 242, 212, 236, 237, 218, 206, 229, 251, 250, 252, 245, 221, 232, 212, 233, 226, 222, 217, 221, 217, 254, 246, 243, 227, 226, 221, 229, 240, 248, 236, 223, 211, 215, 224, 254, 254, 250, 253, 247, 234, 219, 220, 215, 228, 219, 222, 232, 226, 221, 252, 252, 232, 217, 216, 225, 222, 207, 253, 246, 220, 218, 254, 248, 230, 223, 224, 234, 225, 207, 236, 215, 229, 208, 227, 250, 252, 207, 223, 210, 206, 224, 220, 254, 251, 249, 252, 243, 245, 228, 198, 229, 234, 254, 244, 245, 251, 242, 225, 185, 179, 189, 204, 204, 224, 233, 164, 207, 253, 238, 247, 253, 254, 189, 235, 249, 216, 227, 253, 237, 238, 230, 222, 205, 233, 253, 253, 249, 253, 225, 232, 235, 196, 237, 220, 215, 254, 254, 254, 253, 253, 253, 254, 254, 216, 214, 210, 242, 234, 192, 243, 232, 202, 235, 223, 253, 209, 218, 252, 252, 252, 253, 228, 232, 173, 186, 235, 238, 194, 198, 198, 213, 199, 225, 251, 252, 238, 186, 232, 196, 193, 228, 253, 253, 252, 252, 209, 197, 241, 233, 249, 216, 207, 252, 243, 166, 186, 244, 227, 211, 254, 236, 253, 244, 190, 241, 182, 201, 203, 240, 220, 226, 237, 246, 244, 226, 183, 234, 231, 228, 252, 252, 201, 243, 246, 202, 244, 237, 206, 242, 237, 228, 218, 202, 254, 240, 224, 250, 225, 210, 218, 203, 253, 245, 229, 202, 218, 217, 226, 212, 233, 250, 234, 235, 226, 235, 210, 236, 208, 191, 253, 233, 229, 232, 233, 208, 209, 214, 218, 207, 232, 243, 247, 236, 205, 224, 225, 219, 252, 240, 227, 226, 243, 233, 223, 205, 227, 236, 241, 224, 213, 205, 211, 237, 212, 213, 225, 208, 225, 181, 244, 240, 218, 209, 205, 229, 167, 193, 214, 229, 210, 211, 252, 231, 219, 205, 200, 198, 204, 220, 231, 215, 239, 222, 222, 234, 229, 229, 191, 220, 193, 196, 206, 198, 187, 220, 201, 253, 250, 250, 230, 228, 227, 226, 206, 186, 210, 195, 198, 155, 230, 186, 221, 218, 230, 235, 237, 240, 205, 200, 192, 184, 200, 230, 211, 228, 205, 236, 221, 218, 197, 215, 216, 192, 216, 218, 212, 227, 225, 225, 219, 210, 240, 241, 238, 185, 209, 165, 214, 182, 237, 235, 233, 222, 214, 204, 250, 245, 237, 245, 242, 236, 231, 221, 227, 210, 211, 183, 184, 214, 198, 221, 233, 230, 220, 221, 237, 245, 232, 221, 228, 178, 180, 216, 213, 223, 214, 221, 239, 247, 245, 220, 232, 209, 228, 234, 223, 226, 226, 224, 221, 230, 229, 236, 236, 208, 225, 216, 218, 214, 224, 224, 205, 245, 238, 246, 232, 237, 196, 229, 213, 243, 228, 248, 191, 192, 192, 186, 211, 220, 237, 209, 249, 214, 241, 245, 223, 198, 226, 204, 227, 224, 233, 226, 219, 210, 236, 252, 227, 232, 249, 207, 213, 236, 213, 218, 210, 219, 224, 241, 237, 224, 207, 223, 221, 218, 219, 215, 212, 254, 254, 253, 251, 208, 209, 213, 198, 216, 222, 214, 210, 212, 215, 212, 250, 227, 204, 218, 210, 227, 209, 215, 216, 218, 238, 244, 222, 218, 192, 245, 235, 214, 206, 220, 215, 221, 220, 220, 219, 209, 217, 222, 226, 241, 243, 230, 216, 208, 213, 198, 220, 215, 212, 214, 211, 213, 224, 226, 194, 253, 232, 215, 230, 253, 254, 219, 208, 211, 225, 237, 223, 225, 208, 213, 235, 211, 249, 239, 234, 226, 206, 213, 208, 216, 206, 240, 239, 235, 236, 227, 229, 226, 220, 210, 206, 206, 230, 240, 223, 254, 226, 238, 241, 222, 218, 236, 238, 223, 227, 219, 228, 224, 210, 218, 217, 217, 247, 254, 229, 210, 214, 186, 235, 225, 215, 214, 224, 233, 237, 242, 226, 221, 246, 213, 221, 227, 217, 231, 233, 245, 245, 250, 237, 173, 245, 235, 235, 225, 226, 221, 216, 234, 229, 224, 212, 239, 233, 235, 242, 227, 224, 205, 241, 241, 228, 212, 222, 216, 206, 211, 218, 244, 225, 237, 254, 251, 244, 202, 169, 240, 244, 237, 220, 215, 211, 227, 224, 226, 243, 248, 244, 242, 242, 189, 202, 240, 239, 238, 230, 219, 216, 226, 223, 217, 193, 215, 238, 226, 218, 214, 214, 206, 221, 226, 218, 224, 224, 208, 201, 252, 226, 235, 232, 217, 198, 221, 221, 224, 225, 206, 227, 184, 243, 232, 222, 216, 221, 217, 208, 205, 253, 224, 221, 210, 223, 205, 208, 204, 189, 226, 213, 232, 238, 227, 225, 222, 223, 211, 204, 210, 211, 206, 207, 180, 249, 252, 249, 222, 233, 225, 205, 209, 211, 211, 208, 196, 184, 163, 253, 251, 247, 202, 213, 189, 247, 210, 203, 223, 194, 206, 218, 190, 176, 183, 254, 248, 223, 178, 175, 233, 221, 211, 244, 247, 196, 185, 233, 225, 222, 201, 206, 184, 244, 245, 241, 223, 221, 205, 240, 224, 209, 188, 249, 250, 213, 204, 200, 221, 193, 195, 240, 232, 214, 202, 203, 205, 192, 230, 214, 213, 217, 213, 203, 209, 210, 192, 194, 239, 206, 208, 203, 196, 220, 220, 235, 239, 208, 212, 214, 197, 198, 207, 202, 200, 239, 207, 204, 234, 231, 224, 219, 192, 213, 218, 226, 233, 224, 228, 190, 200, 191, 235, 209, 226, 241, 227, 226, 188, 200, 212, 191, 223, 210, 203, 197, 204, 210, 214, 228, 231, 232, 221, 224, 195, 230, 246, 220, 206, 221, 168, 211, 191, 221, 221, 226, 226, 221, 227, 246, 250, 250, 238, 222, 209, 222, 218, 202, 228, 235, 239, 215, 213, 216, 220, 212, 239, 237, 217, 212, 213, 215, 231, 217, 212, 205, 231, 225, 178, 190, 206, 206, 203, 221, 221, 222, 210, 218, 248, 202, 212, 218, 209, 212, 226, 233, 211, 188, 188, 246, 233, 201, 201, 204, 207, 221, 214, 222, 195, 209, 252, 216, 231, 209, 247, 223, 206, 216, 185, 191, 234, 228, 207, 196, 206, 220, 222, 228, 219, 215, 215, 215, 245, 222, 218, 232, 242, 215, 229, 230, 207, 189, 188, 251, 236, 228, 205, 222, 238, 216, 214, 220, 247, 231, 229, 204, 204, 174, 209, 209, 229, 214, 190, 241, 199, 213, 221, 204, 212, 217, 181, 230, 203, 181, 240, 226, 229, 217, 197, 195, 204, 195, 197, 227, 210, 220, 211, 209, 248, 216, 212, 215, 213, 213, 172, 166, 245, 227, 226, 236, 225, 200, 207, 208, 213, 221, 217, 253, 221, 211, 222, 219, 234, 197, 178, 221, 237, 214, 193, 207, 229, 201, 212, 214, 203, 201, 179, 219, 236, 246, 248, 251, 205, 206, 201, 213, 221, 201, 249, 222, 225, 221, 223, 217, 238, 248, 244, 246, 201, 203, 210, 221, 224, 223, 220, 220, 231, 215, 247, 232, 211, 185, 172, 217, 226, 244, 245, 242, 240, 244, 245, 247, 207, 227, 214, 215, 241, 221, 232, 233, 189, 177, 247, 244, 219, 221, 221, 237, 245, 247, 241, 247, 249, 245, 247, 247, 230, 225, 226, 208, 204, 238, 225, 228, 237, 183, 193, 225, 246, 212, 238, 211, 207, 201, 201, 242, 224, 223, 239, 231, 230, 229, 245, 250, 234, 230, 212, 207, 216, 233, 217, 242, 228, 228, 211, 207, 241, 228, 247, 246, 245, 226, 225, 216, 216, 215, 216, 232, 246, 197, 231, 227, 230, 225, 223, 224, 248, 244, 247, 246, 246, 250, 202, 210, 220, 225, 224, 229, 232, 231, 230, 231, 236, 219, 217, 224, 218, 247, 247, 246, 226, 226, 229, 231, 230, 231, 241, 242, 245, 206, 233, 224, 216, 222, 186, 232, 245, 245, 221, 230, 224, 223, 234, 233, 230, 237, 244, 243, 225, 211, 230, 232, 234, 231, 231, 232, 229, 238, 236, 221, 225, 225, 225, 222, 252, 241, 248, 246, 242, 217, 214, 227, 217, 214, 217, 216, 232, 232, 235, 234, 204, 246, 248, 238, 236, 229, 248, 248, 240, 245, 245, 246, 236, 214, 236, 232, 239, 240, 225, 234, 226, 183, 227, 250, 244, 250, 246, 249, 237, 215, 215, 219, 231, 231, 237, 229, 248, 249, 238, 240, 250, 246, 246, 214, 232, 216, 224, 238, 215, 221, 230, 238, 195, 182, 232, 233, 236, 251, 251, 227, 222, 245, 224, 218, 211, 230, 229, 231, 238, 195, 189, 178, 253, 252, 246, 246, 245, 237, 232, 219, 210, 231, 231, 248, 209, 203, 216, 249, 227, 233, 254, 254, 252, 252, 246, 236, 250, 239, 231, 226, 228, 233, 249, 237, 243, 235, 240, 253, 250, 248, 246, 245, 245, 247, 232, 241, 225, 235, 194, 253, 246, 239, 242, 254, 250, 251, 253, 243, 250, 246, 210, 240, 239, 230, 203, 209, 212, 232, 243, 190, 251, 250, 250, 246, 246, 244, 247, 221, 203, 216, 216, 204, 211, 212, 234, 222, 251, 248, 223, 251, 212, 230, 243, 235, 210, 210, 214, 208, 242, 213, 245, 243, 171, 181, 194, 240, 232, 245, 222, 217, 214, 215, 224, 217, 216, 241, 218, 188, 218, 253, 248, 242, 227, 232, 215, 226, 221, 219, 234, 232, 253, 253, 221, 221, 224, 223, 223, 219, 232, 224, 211, 220, 246, 250, 230, 216, 228, 212, 209, 210, 199, 215, 193, 244, 99, 253, 246, 246, 245, 230, 220, 206, 213, 226, 244, 181, 240, 245, 228, 183, 201, 229, 248, 234, 248, 229, 217, 221, 217, 172, 169, 244, 124, 194, 254, 251, 240, 238, 253, 249, 247, 169, 170, 167, 244, 238, 248, 227, 230, 253, 236, 232, 171, 236, 192, 249, 249, 244, 245, 243, 188, 222, 235, 232, 244, 229, 233, 248, 253, 223, 224, 253, 194, 211, 201, 224, 230, 235, 243, 245, 249, 240, 226, 218, 161, 235, 213, 250, 243, 245, 248, 151, 251, 222, 245, 238, 252, 224, 231, 203, 199, 241, 242, 248, 145, 215, 245, 240, 240, 235, 237, 215, 233, 228, 182, 238, 221, 219, 251, 248, 234, 236, 237, 243, 244, 243, 228, 241, 202, 236, 244, 204, 194, 170, 172, 240, 238, 228, 251, 124, 249, 246, 230, 217, 234, 229, 226, 244, 232, 220, 180, 202, 190, 178, 233, 228, 225, 251, 245, 241, 217, 246, 245, 246, 249, 252, 249, 238, 230, 223, 239, 254, 254, 233, 233, 227, 233, 204, 149, 189, 242, 243, 235, 217, 247, 141, 245, 241, 243, 245, 243, 247, 251, 231, 235, 236, 244, 228, 239, 241, 245, 246, 223, 210, 250, 247, 245, 113, 251, 248, 245, 252, 221, 232, 250, 210, 232, 245, 254, 250, 208, 226, 236, 232, 227, 245, 242, 247, 248, 130, 243, 244, 251, 237, 245, 251, 252, 250, 240, 233, 237, 228, 222, 229, 225, 217, 223, 236, 237, 240, 247, 227, 214, 222, 220, 236, 223, 248, 235, 226, 199, 232, 234, 243, 244, 243, 232, 229, 252, 232, 232, 227, 217, 248, 161, 238, 235, 232, 240, 242, 243, 239, 208, 247, 248, 249, 243, 230, 219, 220, 234, 223, 216, 245, 246, 196, 241, 245, 240, 220, 235, 212, 247, 243, 240, 236, 237, 203, 226, 246, 213, 193, 244, 168, 237, 249, 237, 197, 236, 243, 246, 248, 242, 239, 237, 234, 202, 179, 219, 245, 236, 226, 251, 232, 223, 237, 247, 238, 230, 229, 228, 216, 242, 248, 235, 200, 246, 226, 236, 229, 240, 243, 243, 206, 246, 249, 220, 229, 226, 228, 243, 228, 248, 247, 242, 233, 214, 231, 245, 226, 215, 234, 223, 242, 197, 237, 230, 239, 231, 230, 236, 248, 254, 240, 249, 214, 245, 210, 196, 234, 234, 236, 237, 249, 240, 239, 249, 241, 241, 241, 242, 228, 241, 207, 199, 195, 221, 232, 247, 227, 207, 223, 231, 219, 203, 241, 227, 192, 199, 207, 231, 194, 206, 243, 231, 228, 233, 243, 247, 234, 236, 218, 243, 248, 243, 246, 244, 241, 230, 233, 249, 218, 223, 234, 224, 222, 234, 230, 244, 243, 198, 242, 221, 230, 207, 253, 244, 228, 242, 240, 243, 245, 242, 240, 249, 221, 212, 198, 197, 249, 223, 233, 248, 239, 218, 244, 238, 242, 239, 249, 183, 248, 227, 240, 228, 231, 228, 254, 251, 220, 217, 191, 181, 230, 230, 250, 208, 222, 226, 238, 242, 252, 249, 218, 214, 189, 204, 179, 231, 232, 253, 217, 231, 242, 205, 191, 205, 230, 249, 247, 197, 216, 249, 254, 254, 238, 236, 209, 180, 213, 243, 249, 238, 215, 179, 246, 248, 228, 233, 212, 240, 242, 250, 251, 234, 207, 223, 214, 218, 242, 216, 223, 245, 216, 212, 185, 207, 205, 225, 247, 236, 242, 249, 249, 218, 220, 218, 208, 234, 224, 237, 251, 236, 244, 208, 200, 245, 248, 236, 182, 250, 239, 235, 229, 216, 225, 204, 220, 251, 213, 212, 221, 207, 253, 232, 233, 197, 244, 222, 218, 226, 228, 234, 252, 231, 249, 221, 225, 236, 215, 224, 221, 219, 228, 215, 228, 231, 232, 243, 207, 222, 218, 190, 220, 230, 184, 227, 225, 229, 226, 232, 246, 229, 239, 201, 228, 223, 205, 243, 226, 251, 218, 235, 233, 231, 244, 246, 224, 221, 223, 183, 249, 220, 248, 231, 230, 250, 227, 234, 237, 239, 244, 213, 225, 240, 235, 219, 225, 201, 239, 240, 235, 228, 226, 245, 236, 247, 223, 252, 231, 251, 240, 225, 226, 221, 238, 227, 225, 224, 216, 206, 225, 224, 248, 212, 253, 222, 208, 216, 189, 225, 177, 223, 236, 228, 239, 210, 229, 237, 200, 240, 240, 163, 233, 238, 240, 238, 234, 212, 244, 254, 231, 232, 233, 204, 161, 172, 219, 244, 233, 227, 230, 223, 234, 234, 225, 205, 213, 234, 192, 236, 228, 218, 218, 241, 248, 237, 243, 252, 248, 241, 230, 171, 195, 219, 233, 235, 232, 207, 242, 227, 207, 211, 231, 207, 232, 223, 238, 250, 219, 250, 225, 242, 221, 198, 177, 221, 186, 220, 211, 187, 198, 235, 214, 212, 231, 247, 236, 218, 193, 216, 219, 222, 221, 196, 220, 211, 236, 253, 248, 238, 233, 230, 170, 229, 252, 206, 216, 207, 215, 196, 214, 229, 222, 245, 193, 209, 218, 238, 225, 238, 192, 250, 250, 249, 224, 228, 238, 207, 247, 208, 237, 194, 176, 240, 245, 236, 236, 244, 202, 254, 244, 230, 230, 213, 247, 243, 246, 239, 242, 243, 245, 244, 228, 217, 215, 233, 240, 243, 246, 239, 252, 245, 245, 211, 208, 234, 239, 222, 243, 246, 198, 198, 220, 230, 217, 232, 232, 226, 214, 199, 244, 245, 248, 219, 225, 240, 239, 229, 238, 166, 239, 245, 250, 194, 208, 173, 236, 237, 237, 248, 249, 216, 235, 213, 211, 192, 213, 224, 254, 249, 241, 188, 241, 241, 228, 246, 211, 206, 222, 236, 241, 162, 218, 207, 226, 233, 240, 246, 243, 238, 185, 226, 234, 254, 206, 207, 179, 228, 245, 184, 222, 214, 234, 232, 189, 240, 239, 235, 242, 216, 223, 221, 178, 250, 241, 237, 218, 219, 221, 218, 220, 239, 192, 245, 234, 220, 169, 216, 221, 240, 237, 250, 224, 210, 217, 216, 223, 217, 214, 232, 238, 233, 213, 215, 221, 235, 237, 227, 219, 185, 248, 247, 233, 209, 205, 219, 220, 229, 244, 239, 227, 202, 210, 226, 182, 203, 209, 222, 221, 220, 232, 213, 190, 236, 200, 216, 216, 227, 218, 202, 237, 236, 241, 205, 221, 224, 225, 232, 224, 177, 171, 232, 210, 231, 240, 236, 242, 238, 230, 230, 212, 236, 246, 199, 203, 227, 231, 215, 232, 228, 210, 238, 240, 251, 226, 217, 223, 224, 226, 227, 224, 206, 211, 248, 199, 207, 221, 240, 232, 233, 232, 231, 210, 194, 149, 235, 238, 242, 243, 236, 233, 234, 235, 243, 200, 201, 236, 218, 164, 208, 211, 237, 236, 246, 242, 251, 251, 233, 240, 249, 237, 221, 202, 203, 206, 233, 236, 221, 227, 251, 236, 251, 251, 242, 251, 235, 236, 202, 235, 217, 251, 251, 251, 252, 223, 205, 205, 216, 225, 228, 240, 206, 251, 243, 248, 252, 251, 251, 237, 205, 202, 205, 209, 206, 218, 230, 233, 239, 230, 237, 251, 249, 243, 245, 250, 228, 185, 202, 228, 249, 249, 250, 237, 246, 248, 248, 203, 206, 207, 206, 233, 229, 233, 246, 235, 248, 217, 201, 205, 251, 251, 230, 165, 201, 238, 248, 233, 231, 151, 140, 202, 207, 208, 210, 218, 200, 215, 232, 250, 174, 247, 246, 248, 249, 229, 223, 168, 207, 236, 224, 221, 250, 197, 200, 205, 222, 213, 225, 245, 246, 250, 249, 200, 208, 210, 227, 220, 242, 247, 247, 248, 204, 224, 221, 223, 243, 242, 247, 246, 215, 201, 244, 248, 239, 219, 212, 216, 214, 235, 240, 211, 218, 217, 219, 214, 201, 214, 210, 229, 243, 246, 213, 216, 205, 208, 215, 207, 242, 251, 249, 246, 250, 210, 221, 216, 232, 221, 250, 251, 245, 234, 212, 200, 218, 221, 224, 198, 206, 247, 248, 192, 218, 241, 245, 227, 208, 203, 222, 204, 210, 239, 246, 239, 236, 215, 207, 212, 218, 204, 213, 214, 203, 221, 226, 207, 243, 224, 207, 217, 212, 216, 215, 212, 215, 214, 202, 216, 238, 249, 243, 240, 237, 206, 212, 211, 208, 227, 229, 226, 237, 239, 237, 220, 220, 224, 211, 215, 230, 242, 240, 235, 247, 248, 215, 232, 217, 235, 243, 236, 228, 235, 218, 195, 192, 215, 225, 228, 238, 222, 235, 235, 224, 189, 230, 253, 216, 216, 227, 222, 222, 231, 228, 188, 251, 244, 226, 234, 226, 233, 233, 229, 227, 219, 251, 228, 240, 239, 231, 226, 236, 221, 229, 222, 215, 222, 248, 251, 249, 216, 214, 236, 234, 236, 226, 215, 215, 209, 214, 217, 252, 215, 241, 207, 205, 205, 222, 230, 239, 253, 221, 232, 232, 216, 237, 244, 231, 221, 231, 211, 214, 228, 248, 217, 233, 236, 223, 194, 238, 233, 233, 225, 234, 226, 226, 234, 237, 228, 253, 216, 217, 236, 206, 211, 204, 204, 204, 229, 241, 236, 236, 224, 233, 232, 220, 236, 234, 222, 215, 203, 211, 208, 222, 235, 245, 244, 252, 247, 247, 225, 193, 202, 227, 224, 219, 254, 228, 242, 221, 250, 250, 228, 208, 208, 218, 225, 198, 232, 213, 223, 229, 241, 253, 214, 238, 211, 249, 176, 240, 220, 213, 227, 217, 198, 220, 228, 240, 225, 209, 214, 217, 236, 220, 227, 237, 233, 228, 227, 224, 211, 243, 220, 209, 208, 230, 223, 229, 163, 236, 240, 219, 241, 213, 191, 223, 236, 239, 239, 223, 232, 237, 223, 229, 247, 224, 215, 220, 217, 249, 209, 238, 243, 253, 214, 238, 227, 247, 245, 242, 243, 222, 212, 240, 222, 198, 229, 237, 222, 205, 188, 224, 226, 220, 220, 216, 230, 226, 237, 233, 219, 225, 202, 221, 206, 241, 244, 226, 234, 215, 232, 237, 238, 229, 226, 227, 208, 213, 234, 228, 241, 205, 220, 241, 228, 226, 243, 232, 233, 239, 251, 242, 190, 209, 236, 212, 234, 231, 225, 236, 210, 219, 221, 239, 205, 207, 184, 194, 206, 222, 233, 183, 210, 200, 223, 216, 243, 232, 212, 179, 187, 248, 228, 238, 244, 219, 225, 205, 229, 234, 243, 241, 234, 234, 186, 206, 162, 222, 245, 233, 232, 239, 252, 229, 219, 200, 227, 222, 235, 230, 210, 210, 217, 214, 241, 189, 234, 230, 246, 180, 228, 219, 212, 217, 194, 222, 239, 194, 174, 233, 218, 213, 202, 194, 212, 115, 232, 236, 217, 217, 214, 221, 223, 212, 189, 216, 197, 228, 218, 217, 207, 209, 204, 228, 207, 166, 189, 226, 187, 210, 223, 165, 241, 230, 231, 227, 204, 166, 232, 175, 201, 204, 201, 200, 226, 204, 211, 214, 203, 179, 200, 243, 239, 234, 207, 207, 236, 239, 198, 193, 206, 232, 198, 221, 183, 210, 204, 159, 182, 247, 215, 209, 194, 213, 213, 199, 214, 179, 201, 184, 215, 197, 208, 191, 197, 180, 200, 210, 215, 240, 205, 215, 179, 215, 211, 172, 210, 212, 203, 227, 214, 187, 227, 221, 220, 182, 170, 211, 229, 221, 215, 195, 244, 210, 170, 202, 195, 216, 170, 200, 211, 248, 206, 209, 180, 214, 225, 195, 213, 235, 205, 214, 206, 196, 238, 233, 180, 228, 205, 228, 158, 180, 212, 213, 216, 205, 219, 206, 184, 189, 229, 227, 226, 195, 241, 227, 215, 198, 211, 217, 216, 217, 241, 244, 184, 221, 222, 201, 218, 193, 209, 214, 243, 226, 224, 221, 211, 223, 224, 216, 217, 216, 216, 215, 240, 209, 211, 221, 215, 186, 229, 223, 210, 221, 222, 218, 233, 209, 178, 214, 208, 218, 232, 209, 199, 231, 248, 216, 242, 242, 251, 214, 198, 204, 232, 208, 186, 191, 212, 226, 237, 240, 214, 221, 212, 211, 195, 203, 223, 215, 228, 212, 207, 185, 193, 226, 206, 202, 201, 220, 221, 238, 230, 195, 206, 193, 202, 234, 224, 217, 185, 204, 224, 235, 236, 236, 242, 237, 206, 187, 202, 228, 192, 215, 228, 201, 204, 206, 217, 230, 225, 226, 228, 228, 195, 215, 238, 250, 241, 193, 173, 210, 188, 199, 223, 215, 226, 230, 248, 252, 233, 179, 173, 225, 197, 197, 233, 230, 228, 202, 192, 195, 225, 232, 241, 237, 176, 203, 189, 201, 249, 230, 223, 224, 216, 211, 236, 206, 218, 202, 218, 201, 220, 249, 242, 186, 219, 244, 227, 230, 249, 208, 194, 190, 177, 223, 230, 227, 254, 243, 217, 230, 240, 254, 247, 202, 225, 242, 243, 189, 200, 232, 228, 237, 225, 231, 232, 214, 247, 251, 246, 206, 249, 216, 225, 238, 220, 226, 234, 216, 233, 248, 229, 229, 209, 227, 237, 226, 248, 248, 216, 250, 193, 198, 215, 230, 193, 237, 231, 225, 208, 237, 202, 214, 245, 244, 218, 232, 226, 237, 220, 226, 239, 229, 215, 237, 222, 214, 228, 220, 230, 225, 232, 241, 234, 225, 222, 247, 230, 212, 223, 223, 228, 193, 228, 230, 215, 228, 211, 229, 227, 201, 231, 214, 233, 239, 232, 206, 236, 220, 178, 224, 216, 221, 207, 227, 240, 242, 218, 218, 239, 228, 237, 184, 232, 228, 191, 217, 237, 218, 247, 235, 232, 230, 237, 235, 225, 193, 194, 232, 247, 222, 222, 204, 166, 232, 220, 224, 232, 226, 236, 224, 218, 226, 227, 220, 220, 244, 221, 244, 216, 212, 220, 245, 249, 233, 231, 230, 243, 227, 232, 196, 204, 230, 209, 218, 228, 238, 229, 222, 227, 210, 243, 198, 222, 220, 199, 188, 211, 208, 225, 226, 207, 220, 207, 203, 207, 222, 217, 197, 179, 227, 208, 215, 210, 174, 218, 230, 230, 189, 231, 226, 221, 232, 227, 207, 153, 222, 221, 211, 224, 206, 230, 225, 154, 190, 186, 212, 224, 219, 213, 211, 210, 209, 218, 230, 207, 203, 217, 214, 226, 213, 209, 224, 220, 224, 200, 237, 215, 223, 208, 230, 226, 241, 220, 234, 214, 216, 199, 232, 249, 216, 216, 227, 248, 246, 217, 180, 214, 221, 229, 245, 212, 229, 206, 216, 215, 207, 235, 215, 216, 224, 222, 210, 207, 212, 187, 213, 209, 231, 239, 226, 230, 221, 197, 220, 224, 223, 183, 218, 219, 219, 215, 226, 231, 209, 234, 208, 229, 204, 223, 228, 210, 207, 229, 245, 209, 226, 224, 219, 224, 227, 215, 220, 227, 221, 228, 208, 221, 237, 247, 228, 225, 230, 252, 220, 231, 234, 238, 227, 228, 242, 218, 220, 219, 231, 216, 250, 250, 244, 225, 224, 236, 230, 254, 247, 253, 252, 218, 226, 236, 252, 229, 232, 244, 224, 252, 226, 245, 214, 253, 226, 248, 232, 254, 251, 212, 208, 216, 209, 244, 221, 212, 239, 248, 252, 250, 244, 222, 252, 253, 246, 249, 244, 202, 244, 247, 246, 222, 238, 252, 222, 198, 245, 244, 235, 252, 244, 254, 248, 252, 251, 250, 251, 250, 252, 218, 238, 247, 250, 250, 250, 219, 245, 245, 252, 244, 249, 204, 248, 252, 253, 231, 244, 249, 227, 231, 240, 249, 254, 247, 246, 250, 233, 230, 215, 214, 232, 247, 242, 251, 232, 206, 206, 211, 251, 198, 226, 216, 211, 247, 210, 217, 254, 199, 194, 250, 230, 248, 184, 228, 192, 225, 226, 182, 223, 231, 218, 225, 166, 214, 218, 205, 241, 246, 203, 221, 231, 244, 210, 210, 231, 221, 229, 226, 233, 232, 225, 238, 243, 218, 217, 226, 183, 229, 205, 198, 227, 196, 221, 219, 238, 235, 253, 254, 252, 187, 233, 226, 175, 222, 247, 224, 249, 212, 246, 208, 204, 217, 214, 199, 253, 199, 213, 174, 241, 231, 213, 226, 220, 241, 228, 232, 181, 191, 206, 191, 229, 173, 245, 192, 163, 150, 161, 249, 215, 164, 168, 187, 163, 235, 250, 226, 217, 191, 252, 195, 240, 213, 222, 249, 155, 243, 247, 198, 252, 223, 172, 192, 254, 222, 201, 251, 253, 251, 229, 171, 251, 254, 248, 253, 223, 253, 249, 215, 236, 216, 251, 184, 253, 253, 215, 235, 231, 252, 214, 247, 240, 208, 247, 206, 187, 192, 250, 253, 252, 240, 187, 247, 247, 236, 224, 250, 189, 212, 242, 212, 188, 251, 248, 246, 249, 252, 251, 254, 248, 197, 242, 212, 220, 184, 252, 254, 224, 245, 248, 252, 176, 197, 254, 182, 253, 155, 194, 242, 235, 247, 251, 251, 178, 253, 246, 251, 196, 229, 246, 252, 190, 199, 188, 241, 183, 254, 220, 218, 166, 251, 182, 178, 251, 232, 235, 252, 181, 212, 246, 231, 190, 196, 234, 251, 244, 247, 215, 253, 244, 225, 204, 249, 233, 223, 201, 194, 200, 205, 254, 244, 232, 248, 201, 199, 217, 229, 227, 251, 243, 239, 187, 198, 253, 216, 246, 244, 246, 223, 209, 251, 252, 208, 213, 237, 209, 197, 254, 254, 220, 212, 187, 245, 243, 228, 238, 242, 183, 245, 215, 254, 246, 209, 184, 202, 216, 219, 222, 213, 237, 178, 236, 229, 215, 251, 191, 248, 240, 249, 247, 241, 205, 227, 250, 243, 166, 248, 227, 246, 171, 249, 249, 253, 220, 210, 246, 216, 242, 239, 214, 212, 196, 230, 228, 250, 199, 235, 248, 248, 228, 236, 240, 215, 246, 254, 227, 248, 207, 215, 227, 245, 214, 210, 204, 209, 224, 226, 254, 205, 198, 186, 232, 232, 216, 222, 201, 226, 248, 215, 203, 225, 194, 175, 179, 216, 201, 194, 219, 202, 240, 184, 180, 207, 209, 200, 187, 223, 182, 188, 187, 192, 250, 187, 184, 169, 177, 198, 188, 242, 246, 205, 202, 197, 210, 199, 211, 184, 252, 248, 208, 198, 208, 254, 210, 208, 215, 235, 214, 253, 253, 214, 214, 202, 209, 217, 254, 219, 208, 231, 209, 246, 227, 254, 194, 223, 238, 236, 234, 251, 252, 232, 233, 197, 251, 252, 246, 236, 236, 243, 200, 248, 251, 253, 181, 201, 210, 225, 201, 238, 246, 225, 216, 233, 242, 229, 252, 247, 239, 217, 199, 226, 250, 220, 213, 186, 188, 246, 253, 254, 192, 206, 235, 243, 254, 231, 185, 241, 246, 227, 216, 211, 203, 180, 253, 252, 251, 218, 221, 212, 185, 250, 247, 251, 196, 248, 246, 230, 194, 249, 251, 248, 247, 198, 248, 252, 238, 228, 253, 244, 251, 232, 249, 252, 251, 188, 247, 237, 238, 180, 245, 249, 165, 241, 233, 175, 200, 253, 252, 242, 236, 197, 155, 243, 194, 246, 251, 203, 146, 170, 198, 198, 165, 182, 226, 251, 156, 166, 240, 158, 152, 200, 156, 199, 218, 193, 215, 138, 154, 180, 198, 153, 190, 188, 195, 182, 159, 226, 211, 225, 165, 186, 226, 226, 227, 225, 191, 175, 155, 176, 218, 188, 159, 183, 211, 205, 163, 160, 187, 199, 175, 216, 216, 191, 169, 183, 219, 174, 192, 193, 224, 194, 252, 209, 184, 188, 214, 207, 213, 225, 199, 231, 194, 238, 212, 220, 250, 236, 229, 214, 220, 235, 232, 223, 211, 225, 222, 228, 232, 195, 215, 243, 217, 223, 171, 239, 239, 190, 225, 239, 221, 237, 254, 250, 222, 240, 241, 214, 224, 242, 242, 236, 230, 240, 219, 224, 236, 230, 222, 237, 232, 246, 225, 225, 223, 227, 228, 225, 227, 186, 217, 216, 225, 221, 224, 219, 227, 232, 227, 228, 240, 223, 242, 244, 228, 216, 197, 182, 226, 227, 241, 227, 229, 226, 224, 217, 250, 245, 226, 205, 214, 229, 253, 232, 214, 224, 183, 221, 231, 210, 216, 245, 225, 235, 211, 236, 226, 231, 190, 254, 195, 183, 183, 241, 220, 213, 183, 220, 248, 226, 196, 219, 244, 211, 249, 225, 229, 212, 226, 207, 227, 214, 241, 240, 230, 213, 254, 229, 241, 220, 233, 229, 231, 230, 231, 244, 246, 215, 200, 231, 252, 236, 211, 196, 221, 248, 243, 223, 235, 238, 212, 233, 225, 247, 222, 241, 251, 231, 205, 216, 237, 233, 232, 221, 228, 200, 210, 237, 229, 221, 247, 207, 238, 239, 236, 235, 247, 222, 236, 236, 222, 232, 228, 231, 237, 229, 225, 231, 203, 229, 204, 246, 233, 211, 232, 230, 231, 242, 218, 226, 219, 241, 215, 248, 214, 232, 223, 244, 244, 206, 244, 232, 238, 238, 177, 223, 236, 235, 238, 216, 203, 196, 199, 209, 233, 238, 226, 217, 223, 203, 241, 216, 226, 227, 220, 244, 202, 203, 242, 220, 217, 213, 224, 249, 223, 250, 209, 231, 221, 194, 206, 216, 215, 199, 214, 225, 207, 202, 216, 249, 235, 224, 202, 228, 223, 218, 216, 245, 212, 219, 221, 252, 247, 213, 232, 228, 213, 209, 217, 197, 231, 212, 228, 240, 243, 224, 224, 223, 205, 250, 201, 224, 229, 237, 236, 229, 233, 226, 227, 228, 231, 208, 229, 241, 221, 223, 248, 231, 239, 238, 208, 232, 198, 215, 230, 223, 235, 236, 248, 214, 225, 239, 224, 220, 209, 250, 228, 246, 230, 223, 225, 205, 235, 229, 232, 221, 222, 208, 233, 234, 229, 236, 238, 237, 208, 242, 240, 240, 231, 211, 200, 233, 200, 232, 221, 212, 211, 232, 223, 225, 231, 222, 250, 244, 228, 205, 233, 251, 234, 234, 236, 240, 229, 226, 221, 223, 215, 219, 171, 218, 253, 248, 229, 241, 225, 202, 228, 216, 205, 221, 229, 236, 227, 227, 229, 238, 215, 233, 224, 214, 202, 231, 198, 233, 234, 219, 235, 218, 232, 236, 191, 231, 205, 229, 195, 224, 233, 228, 223, 224, 202, 198, 185, 218, 235, 227, 192, 196, 215, 193, 180, 200, 228, 234, 174, 193, 199, 231, 163, 209, 183, 182, 236, 203, 180, 175, 197, 245, 228, 169, 230, 159, 232, 241, 155, 172, 226, 180, 150, 242, 241, 229, 187, 240, 214, 161, 226, 232, 223, 211, 172, 243, 226, 236, 239, 227, 216, 226, 229, 194, 196, 240, 235, 241, 233, 235, 166, 237, 218, 180, 187, 230, 194, 239, 169, 192, 184, 190, 239, 212, 195, 213, 217, 238, 191, 99, 188, 102, 231, 169, 180, 221, 198, 229, 211, 156, 241, 237, 164, 191, 186, 227, 159, 213, 236, 209, 204, 215, 218, 213, 165, 209, 181, 174, 230, 234, 229, 232, 189, 236, 227, 227, 230, 161, 203, 239, 232, 198, 232, 189, 234, 226, 223, 178, 181, 229, 168, 242, 244, 180, 239, 235, 216, 179, 246, 239, 202, 161, 151, 244, 204, 232, 225, 239, 242, 229, 227, 169, 234, 179, 228, 243, 247, 175, 182, 166, 236, 248, 167, 165, 225, 251, 189, 222, 222, 188, 222, 186, 173, 231, 192, 236, 228, 228, 234, 209, 235, 232, 199, 244, 190, 240, 233, 191, 216, 216, 227, 194, 238, 193, 179, 233, 215, 163, 200, 231, 224, 234, 225, 221, 254, 182, 224, 205, 198, 200, 208, 218, 213, 219, 242, 218, 219, 220, 242, 245, 218, 189, 234, 171, 247, 178, 207, 228, 218, 171, 190, 199, 229, 186, 181, 230, 190, 215, 248, 148, 233, 190, 214, 156, 236, 177, 224, 140, 179, 244, 191, 210, 222, 219, 228, 213, 195, 223, 207, 241, 224, 205, 207, 231, 197, 224, 221, 169, 195, 230, 208, 214, 202, 224, 212, 211, 245, 203, 247, 222, 252, 229, 207, 189, 230, 228, 235, 190, 194, 224, 169, 211, 251, 220, 228, 223, 189, 227, 224, 192, 202, 195, 224, 221, 219, 200, 202, 191, 222, 219, 185, 212, 233, 181, 183, 194, 193, 212, 235, 187, 185, 192, 252, 229, 227, 233, 222, 186, 235, 234, 169, 185, 215, 224, 192, 212, 147, 229, 247, 202, 205, 221, 219, 218, 222, 195, 217, 224, 249, 246, 251, 163, 245, 158, 197, 223, 214, 206, 228, 210, 190, 215, 233, 204, 165, 145, 197, 250, 181, 226, 155, 153, 232, 215, 184, 247, 253, 211, 213, 188, 205, 215, 203, 246, 249, 218, 189, 190, 211, 189, 203, 223, 221, 206, 202, 198, 248, 249, 213, 153, 232, 252, 159, 196, 212, 228, 207, 243, 232, 140, 182, 176, 211, 223, 213, 233, 243, 249, 238, 209, 208, 211, 244, 240, 207, 166, 211, 250, 249, 242, 198, 216, 245, 229, 227, 230, 221, 222, 228, 221, 202, 199, 250, 240, 182, 231, 224, 214, 203, 224, 229, 211, 215, 233, 147, 162, 236, 226, 223, 245, 188, 193, 162, 245, 241, 174, 202, 235, 204, 230, 243, 253, 168, 241, 209, 222, 223, 248, 171, 175, 192, 251, 235, 154, 183, 212, 219, 239, 234, 232, 224, 203, 214, 214, 187, 174, 246, 238, 229, 239, 234, 185, 193, 126, 185, 235, 165, 151, 234, 218, 193, 205, 224, 254, 222, 183, 230, 136, 137, 232, 203, 236, 165, 251, 236, 234, 246, 195, 220, 221, 205, 238, 230, 241, 159, 231, 221, 182, 197, 187, 176, 239, 237, 244, 248, 215, 203, 218, 173, 227, 247, 249, 244, 238, 246, 206, 210, 252, 133, 177, 244, 242, 239, 211, 198, 207, 187, 160, 191, 203, 203, 247, 214, 222, 243, 202, 152, 166, 241, 241, 218, 187, 173, 177, 186, 219, 222, 225, 161, 187, 242, 195, 169, 223, 197, 195, 236, 193, 184, 162, 227, 215, 250, 206, 236, 249, 183, 159, 200, 227, 168, 173, 158, 210, 156, 222, 227, 196, 172, 177, 176, 177, 170, 173, 174, 243, 214, 184, 165, 208, 228, 246, 227, 235, 240, 252, 229, 233, 184, 186, 200, 181, 189, 238, 253, 168, 252, 230, 239, 234, 176, 232, 232, 168, 172, 195, 167, 198, 210, 238, 232, 229, 175, 172, 220, 189, 213, 253, 194, 208, 175, 178, 194, 221, 246, 202, 239, 243, 187, 242, 251, 186, 249, 197, 162, 210, 203, 207, 235, 230, 232, 214, 248, 205, 238, 222, 232, 246, 253, 245, 187, 208, 243, 206, 204, 174, 197, 196, 195, 161, 230, 221, 226, 240, 198, 235, 231, 219, 220, 220, 248, 236, 189, 248, 185, 208, 215, 201, 200, 215, 180, 200, 206, 245, 253, 250, 189, 189, 248, 201, 230, 250, 235, 246, 237, 248, 236, 239, 248, 195, 209, 208, 210, 208, 195, 191, 216, 246, 236, 235, 222, 237, 239, 201, 191, 208, 207, 218, 231, 219, 235, 238, 223, 230, 218, 192, 218, 229, 228, 254, 185, 206, 218, 233, 219, 152, 178, 242, 192, 220, 246, 245, 225, 198, 218, 202, 204, 148, 200, 200, 219, 248, 210, 224, 213, 208, 229, 254, 242, 238, 238, 219, 236, 243, 200, 217, 249, 229, 226, 196, 192, 207, 249, 232, 222, 230, 231, 192, 235, 254, 183, 229, 243, 244, 228, 187, 191, 217, 230, 237, 218, 218, 194, 184, 230, 239, 223, 177, 229, 201, 241, 233, 237, 237, 246, 224, 212, 170, 189, 178, 183, 204, 254, 245, 219, 175, 225, 211, 229, 211, 226, 226, 225, 208, 234, 224, 214, 250, 228, 217, 190, 238, 209, 224, 218, 210, 205, 204, 187, 223, 157, 200, 219, 230, 225, 219, 183, 199, 217, 225, 234, 175, 222, 226, 201, 209, 203, 207, 251, 222, 170, 159, 186, 225, 224, 245, 221, 254, 232, 222, 168, 178, 225, 227, 206, 190, 217, 184, 187, 243, 223, 212, 205, 185, 224, 224, 184, 175, 229, 225, 235, 240, 244, 186, 228, 190, 174, 232, 232, 244, 234, 154, 169, 174, 213, 200, 199, 243, 226, 217, 224, 226, 193, 230, 155, 248, 223, 223, 239, 243, 216, 206, 236, 204, 220, 249, 216, 211, 229, 218, 215, 208, 188, 253, 199, 244, 188, 238, 246, 235, 215, 214, 212, 195, 219, 245, 214, 228, 217, 225, 219, 231, 198, 228, 232, 148, 224, 196, 216, 210, 235, 180, 216, 223, 218, 229, 214, 219, 243, 237, 217, 245, 180, 226, 214, 252, 221, 215, 211, 232, 190, 202, 231, 184, 220, 234, 240, 181, 210, 208, 144, 219, 200, 251, 233, 215, 198, 226, 218, 220, 175, 198, 234, 228, 236, 231, 211, 215, 206, 217, 206, 251, 214, 217, 237, 202, 203, 188, 222, 243, 162, 169, 213, 247, 230, 186, 179, 172, 224, 236, 217, 224, 212, 236, 220, 211, 211, 190, 225, 232, 186, 233, 199, 174, 177, 221, 206, 224, 217, 201, 216, 209, 224, 168, 168, 190, 232, 195, 212, 220, 226, 212, 173, 235, 203, 229, 205, 220, 208, 202, 199, 208, 183, 187, 163, 178, 234, 244, 212, 244, 203, 203, 204, 234, 222, 238, 221, 215, 203, 232, 185, 206, 220, 241, 246, 240, 254, 250, 195, 214, 218, 209, 206, 202, 222, 232, 232, 212, 231, 230, 177, 156, 179, 228, 229, 220, 235, 222, 208, 227, 208, 229, 217, 206, 249, 212, 228, 236, 218, 210, 223, 228, 230, 212, 212, 230, 225, 222, 229, 214, 211, 222, 202, 206, 161, 236, 209, 218, 202, 234, 193, 214, 213, 171, 194, 223, 217, 193, 226, 227, 217, 191, 194, 199, 222, 235, 229, 226, 188, 241, 228, 219, 231, 240, 204, 193, 184, 190, 211, 223, 213, 219, 186, 218, 148, 233, 153, 212, 192, 217, 227, 218, 214, 162, 223, 229, 224, 227, 249, 220, 223, 196, 186, 234, 211, 175, 183, 153, 174, 174, 172, 214, 225, 214, 213, 219, 183, 218, 220, 202, 195, 222, 218, 245, 201, 245, 176, 222, 204, 235, 220, 247, 246, 204, 212, 205, 232, 197, 202, 205, 206, 218, 220, 227, 231, 201, 240, 205, 187, 184, 189, 233, 240, 232, 191, 228, 165, 198, 229, 221, 185, 176, 218, 214, 211, 207, 225, 218, 212, 213, 148, 242, 228, 213, 207, 184, 211, 230, 223, 222, 179, 208, 226, 229, 201, 148, 235, 188, 237, 169, 219, 215, 184, 198, 226, 236, 238, 227, 230, 208, 188, 224, 209, 222, 220, 219, 230, 225, 229, 171, 231, 199, 200, 228, 249, 210, 210, 214, 215, 217, 214, 222, 211, 206, 218, 219, 214, 246, 175, 214, 226, 216, 210, 230, 220, 201, 213, 221, 197, 218, 231, 219, 212, 226, 248, 206, 224, 236, 201, 202, 203, 220, 215, 221, 228, 232, 226, 176, 220, 235, 211, 200, 202, 231, 230, 220, 217, 227, 222, 236, 223, 250, 250, 239, 213, 220, 212, 208, 227, 212, 230, 229, 228, 232, 236, 192, 250, 247, 226, 205, 224, 194, 187, 212, 193, 215, 211, 163, 207, 237, 218, 213, 207, 231, 238, 205, 184, 196, 207, 209, 230, 226, 225, 223, 252, 188, 211, 233, 218, 202, 193, 218, 226, 224, 218, 185, 224, 219, 201, 216, 241, 236, 240, 197, 229, 250, 244, 227, 217, 226, 212, 224, 242, 227, 215, 221, 242, 221, 249, 220, 240, 214, 220, 194, 235, 233, 249, 208, 219, 244, 185, 220, 237, 239, 215, 232, 233, 199, 212, 197, 217, 224, 220, 230, 237, 236, 225, 242, 162, 229, 227, 199, 212, 238, 183, 224, 226, 203, 233, 229, 222, 200, 248, 236, 214, 228, 195, 216, 220, 233, 206, 222, 198, 226, 219, 230, 195, 236, 218, 224, 222, 195, 203, 250, 188, 217, 221, 224, 215, 231, 235, 191, 242, 240, 248, 218, 232, 211, 217, 238, 230, 213, 213, 209, 216, 214, 233, 192, 208, 200, 212, 202, 207, 211, 229, 218, 211, 216, 225, 220, 196, 216, 187, 228, 215, 197, 197, 230, 194, 237, 208, 225, 223, 220, 228, 217, 225, 183, 240, 208, 207, 224, 228, 232, 213, 210, 200, 244, 213, 218, 229, 192, 224, 247, 197, 209, 233, 252, 214, 226, 246, 247, 242, 207, 216, 209, 199, 235, 192, 186, 199, 221, 197, 229, 224, 192, 252, 221, 220, 236, 230, 216, 228, 188, 236, 223, 226, 207, 189, 225, 215, 211, 227, 228, 207, 249, 238, 213, 247, 236, 195, 249, 233, 197, 228, 231, 230, 244, 228, 210, 162, 244, 223, 237, 228, 240, 235, 240, 216, 236, 202, 249, 216, 226, 209, 225, 252, 194, 217, 193, 219, 227, 229, 240, 215, 251, 243, 234, 163, 182, 188, 202, 251, 250, 147, 232, 188, 226, 225, 223, 221, 233, 225, 248, 225, 221, 254, 226, 226, 225, 223, 205, 185, 218, 228, 198, 205, 228, 218, 250, 208, 248, 245, 246, 223, 213, 216, 195, 247, 198, 225, 174, 229, 184, 243, 236, 182, 254, 252, 241, 237, 234, 248, 243, 234, 222, 201, 222, 194, 190, 246, 218, 252, 249, 232, 213, 248, 248, 250, 223, 209, 221, 203, 222, 151, 224, 245, 244, 250, 253, 247, 248, 246, 222, 241, 248, 210, 213, 244, 235, 235, 237, 221, 253, 250, 230, 237, 253, 253, 250, 204, 225, 253, 225, 247, 203, 207, 225, 232, 226, 228, 253, 221, 210, 223, 253, 221, 234, 209, 240, 252, 226, 213, 222, 226, 183, 234, 251, 236, 243, 252, 219, 220, 238, 223, 175, 201, 218, 219, 254, 218, 220, 229, 234, 231, 238, 215, 254, 211, 238, 239, 223, 223, 219, 254, 252, 233, 203, 215, 238, 183, 219, 253, 191, 226, 226, 222, 227, 207, 171, 210, 226, 234, 218, 233, 183, 234, 252, 252, 220, 225, 245, 242, 230, 236, 238, 237, 231, 207, 244, 216, 232, 179, 220, 219, 223, 230, 198, 216, 239, 224, 231, 215, 247, 218, 199, 202, 225, 222, 202, 233, 232, 190, 197, 199, 226, 185, 253, 217, 237, 220, 221, 250, 206, 237, 225, 234, 225, 190, 230, 221, 220, 184, 187, 202, 183, 235, 230, 211, 184, 233, 223, 243, 184, 184, 230, 210, 218, 188, 213, 236, 230, 193, 206, 203, 237, 165, 161, 210, 254, 234, 247, 245, 223, 197, 181, 224, 230, 230, 232, 169, 224, 203, 226, 223, 175, 209, 212, 210, 210, 236, 211, 215, 233, 218, 217, 238, 239, 254, 254, 218, 215, 214, 241, 243, 248, 206, 222, 251, 228, 235, 238, 213, 242, 239, 254, 224, 233, 216, 217, 223, 218, 250, 201, 234, 228, 224, 212, 229, 230, 246, 212, 222, 218, 213, 208, 189, 215, 189, 229, 245, 226, 193, 215, 216, 220, 191, 185, 205, 254, 210, 217, 220, 224, 219, 207, 182, 243, 226, 205, 210, 216, 237, 219, 227, 206, 196, 223, 233, 229, 214, 215, 216, 219, 244, 252, 231, 215, 234, 240, 198, 222, 228, 211, 250, 219, 216, 217, 233, 223, 213, 222, 206, 238, 179, 207, 197, 195, 229, 233, 193, 209, 212, 188, 220, 226, 204, 228, 221, 210, 179, 235, 243, 214, 242, 233, 176, 237, 198, 217, 244, 202, 229, 182, 233, 219, 215, 234, 229, 251, 243, 234, 214, 241, 220, 224, 220, 214, 214, 240, 211, 237, 186, 246, 235, 237, 226, 231, 216, 183, 182, 244, 226, 235, 221, 219, 234, 222, 218, 199, 240, 217, 242, 221, 220, 241, 246, 209, 219, 254, 209, 233, 216, 245, 246, 207, 220, 220, 204, 214, 222, 223, 222, 234, 213, 215, 211, 229, 229, 216, 210, 230, 201, 223, 236, 243, 247, 203, 229, 205, 234, 245, 246, 207, 238, 215, 232, 244, 250, 225, 231, 203, 233, 224, 222, 220, 220, 217, 243, 245, 244, 205, 218, 218, 213, 244, 246, 241, 237, 235, 230, 242, 236, 203, 244, 249, 252, 248, 247, 216, 229, 230, 227, 228, 228, 227, 241, 246, 227, 212, 216, 234, 239, 245, 227, 248, 245, 215, 225, 230, 181, 186, 251, 253, 252, 241, 204, 234, 233, 240, 184, 193, 241, 231, 239, 247, 217, 230, 199, 252, 251, 210, 177, 209, 200, 194, 252, 246, 219, 249, 191, 244, 215, 238, 207, 210, 254, 247, 248, 225, 215, 209, 226, 226, 233, 250, 245, 212, 244, 239, 247, 232, 157, 247, 245, 184, 247, 223, 173, 120, 239, 234, 246, 236, 169, 250, 246, 229, 250, 183, 166, 242, 229, 228, 172, 229, 244, 249, 248, 251, 252, 233, 216, 231, 245, 237, 250, 240, 240, 245, 189, 217, 219, 249, 245, 172, 192, 198, 234, 244, 251, 237, 239, 211, 229, 249, 245, 230, 247, 246, 189, 114, 230, 195, 176, 190, 233, 241, 251, 227, 252, 238, 210, 211, 202, 223, 246, 204, 228, 236, 225, 235, 207, 247, 237, 222, 227, 246, 206, 248, 235, 199, 204, 239, 239, 235, 241, 235, 240, 244, 219, 248, 246, 230, 184, 247, 222, 231, 227, 212, 244, 244, 229, 238, 225, 214, 206, 242, 219, 253, 222, 247, 231, 225, 246, 247, 224, 253, 225, 248, 242, 222, 225, 243, 240, 203, 199, 229, 223, 249, 214, 207, 207, 216, 244, 243, 245, 237, 228, 218, 239, 198, 240, 219, 213, 225, 230, 233, 215, 204, 187, 194, 214, 254, 253, 221, 192, 238, 219, 206, 229, 202, 189, 208, 239, 250, 238, 248, 239, 192, 218, 211, 234, 187, 230, 235, 183, 207, 253, 221, 210, 251, 219, 223, 207, 231, 214, 236, 207, 220, 249, 244, 229, 216, 223, 221, 233, 227, 227, 229, 222, 211, 227, 241, 246, 241, 223, 229, 240, 240, 237, 242, 229, 235, 250, 238, 225, 246, 202, 232, 243, 235, 254, 239, 211, 231, 232, 229, 239, 234, 230, 218, 224, 238, 190, 241, 239, 223, 232, 221, 230, 251, 241, 245, 225, 215, 211, 225, 230, 246, 222, 231, 175, 188, 211, 211, 224, 253, 180, 209, 229, 213, 242, 241, 238, 243, 239, 227, 238, 239, 233, 229, 253, 229, 246, 248, 225, 207, 249, 207, 207, 224, 234, 243, 209, 241, 234, 234, 237, 242, 217, 241, 220, 207, 247, 239, 216, 157, 211, 227, 223, 238, 223, 249, 220, 214, 231, 247, 204, 213, 232, 194, 231, 215, 189, 221, 234, 235, 215, 231, 208, 215, 224, 235, 227, 219, 182, 222, 226, 227, 229, 231, 250, 201, 206, 243, 236, 221, 161, 233, 241, 210, 224, 194, 210, 234, 254, 246, 247, 215, 235, 184, 235, 204, 214, 223, 184, 232, 247, 212, 230, 241, 201, 203, 248, 249, 222, 205, 218, 217, 249, 231, 236, 198, 217, 248, 249, 251, 242, 250, 207, 197, 206, 246, 248, 232, 201, 225, 203, 237, 211, 202, 224, 225, 245, 244, 250, 203, 223, 213, 226, 248, 221, 163, 234, 244, 207, 216, 218, 199, 195, 205, 220, 227, 241, 218, 208, 208, 238, 217, 212, 231, 205, 210, 208, 238, 235, 225, 223, 214, 234, 244, 229, 205, 213, 235, 224, 225, 247, 248, 224, 232, 231, 230, 231, 228, 173, 232, 229, 234, 235, 242, 222, 238, 241, 234, 247, 246, 230, 212, 206, 204, 228, 229, 231, 213, 254, 245, 136, 222, 235, 232, 217, 212, 207, 229, 206, 238, 237, 224, 213, 208, 246, 223, 245, 237, 223, 234, 217, 216, 215, 214, 234, 209, 216, 210, 206, 247, 191, 253, 244, 245, 217, 219, 245, 236, 247, 205, 211, 242, 253, 234, 203, 217, 236, 210, 225, 234, 213, 237, 235, 227, 231, 218, 208, 238, 236, 231, 228, 212, 221, 222, 226, 197, 219, 197, 195, 246, 222, 226, 199, 210, 188, 227, 202, 225, 216, 222, 204, 213, 240, 227, 216, 208, 206, 218, 203, 191, 206, 189, 196, 213, 243, 235, 233, 198, 235, 230, 218, 232, 215, 213, 233, 240, 242, 196, 217, 168, 188, 179, 205, 239, 216, 195, 177, 213, 216, 204, 196, 200, 216, 207, 191, 195, 211, 207, 241, 197, 212, 240, 181, 228, 245, 184, 216, 209, 215, 183, 196, 211, 211, 224, 163, 234, 235, 218, 208, 211, 193, 199, 209, 247, 238, 231, 231, 211, 201, 204, 231, 205, 215, 204, 206, 238, 204, 202, 202, 198, 230, 228, 237, 215, 223, 186, 250, 225, 183, 232, 211, 252, 201, 188, 239, 185, 177, 212, 254, 236, 230, 234, 234, 235, 247, 191, 247, 222, 244, 235, 228, 224, 182, 215, 241, 218, 215, 192, 228, 230, 232, 218, 223, 214, 243, 231, 227, 227, 238, 235, 224, 224, 242, 245, 219, 206, 202, 227, 222, 243, 207, 230, 237, 227, 205, 212, 235, 226, 213, 206, 240, 196, 164, 195, 207, 208, 233, 220, 218, 239, 222, 223, 221, 217, 213, 172, 201, 222, 204, 237, 218, 211, 220, 230, 220, 231, 222, 211, 241, 229, 223, 187, 192, 218, 210, 233, 208, 203, 232, 229, 236, 241, 214, 219, 225, 194, 228, 223, 229, 198, 216, 211, 219, 220, 230, 237, 220, 220, 223, 248, 245, 239, 224, 248, 250, 239, 249, 249, 254, 233, 202, 242, 253, 247, 254, 253, 239, 246, 215, 242, 252, 213, 247, 244, 251, 234, 250, 200, 244, 246, 253, 232, 209, 211, 228, 226, 219, 196, 244, 234, 239, 239, 242, 215, 245, 234, 240, 238, 223, 213, 184, 190, 202, 234, 188, 244, 200, 254, 168, 253, 160, 236, 211, 227, 222, 186, 175, 240, 237, 199, 158, 248, 230, 192, 207, 164, 227, 194, 252, 243, 240, 203, 242, 209, 253, 253, 253, 253, 251, 243, 250, 239, 199, 240, 223, 173, 182, 242, 246, 247, 238, 231, 248, 250, 249, 253, 231, 224, 164, 249, 202, 224, 232, 207, 253, 229, 241, 232, 217, 182, 251, 247, 246, 254, 253, 220, 194, 198, 245, 214, 187, 187, 190, 183, 239, 192, 220, 212, 224, 251, 192, 210, 235, 253, 210, 229, 179, 242, 236, 247, 228, 251, 248, 252, 240, 214, 248, 241, 254, 224, 204, 222, 187, 241, 198, 238, 202, 244, 225, 245, 217, 198, 185, 189, 194, 181, 199, 243, 213, 180, 203, 250, 231, 205, 216, 200, 243, 206, 195, 236, 213, 220, 253, 183, 216, 207, 227, 219, 206, 250, 220, 242, 251, 224, 231, 250, 228, 233, 246, 236, 251, 224, 225, 238, 209, 209, 169, 229, 217, 246, 190, 236, 247, 252, 204, 247, 242, 254, 251, 207, 227, 170, 250, 251, 196, 252, 247, 248, 229, 248, 186, 240, 241, 171, 250, 182, 192, 211, 210, 253, 246, 204, 145, 252, 201, 148, 243, 166, 186, 154, 225, 226, 150, 201, 182, 189, 158, 246, 194, 217, 236, 190, 197, 173, 202, 178, 219, 225, 251, 221, 220, 237, 244]}, {"marker": {"color": "rgb(132,94,194)"}, "name": "Hillshade 3pm", "type": "box", "y": [148, 151, 135, 122, 150, 140, 138, 144, 133, 124, 161, 136, 92, 170, 151, 137, 161, 133, 156, 144, 126, 179, 71, 130, 143, 142, 145, 120, 154, 71, 52, 130, 137, 147, 133, 100, 61, 131, 36, 157, 159, 158, 126, 184, 141, 130, 130, 136, 135, 131, 48, 192, 149, 106, 129, 200, 116, 139, 153, 141, 137, 116, 129, 138, 116, 124, 134, 142, 138, 127, 144, 116, 102, 151, 136, 93, 108, 166, 130, 102, 121, 119, 152, 140, 137, 151, 181, 148, 124, 184, 137, 137, 156, 149, 121, 135, 144, 141, 148, 138, 134, 113, 128, 100, 153, 166, 147, 102, 161, 158, 115, 177, 165, 137, 132, 101, 112, 137, 143, 138, 136, 153, 118, 92, 199, 119, 135, 153, 90, 146, 155, 186, 117, 98, 148, 133, 162, 150, 164, 124, 196, 222, 128, 212, 144, 150, 95, 179, 154, 146, 136, 220, 148, 140, 181, 172, 160, 153, 109, 123, 149, 135, 108, 125, 180, 177, 171, 240, 166, 160, 110, 118, 114, 149, 149, 152, 112, 215, 124, 149, 147, 165, 134, 113, 165, 124, 104, 174, 139, 156, 155, 146, 152, 133, 124, 125, 92, 170, 165, 120, 192, 147, 120, 115, 160, 141, 123, 98, 96, 130, 124, 166, 188, 194, 148, 104, 149, 143, 90, 144, 125, 183, 126, 177, 152, 147, 137, 144, 163, 158, 197, 169, 170, 115, 126, 152, 166, 130, 142, 142, 137, 154, 138, 146, 88, 107, 92, 191, 138, 100, 167, 130, 132, 169, 108, 121, 137, 96, 138, 94, 124, 150, 161, 150, 183, 131, 152, 139, 153, 173, 151, 156, 173, 180, 111, 161, 182, 183, 129, 150, 121, 123, 129, 133, 107, 123, 164, 175, 155, 122, 141, 111, 177, 191, 138, 108, 157, 142, 125, 105, 187, 178, 177, 187, 85, 144, 144, 177, 184, 132, 162, 96, 155, 133, 118, 135, 110, 121, 173, 138, 137, 131, 153, 135, 171, 157, 176, 124, 167, 173, 130, 198, 169, 192, 111, 153, 107, 127, 138, 123, 183, 149, 124, 161, 159, 126, 158, 118, 129, 138, 119, 121, 145, 184, 187, 148, 169, 169, 198, 94, 145, 128, 116, 112, 129, 151, 90, 148, 129, 149, 135, 145, 128, 217, 176, 181, 118, 178, 154, 209, 127, 134, 149, 158, 119, 115, 85, 58, 130, 113, 119, 114, 98, 119, 116, 111, 105, 103, 121, 136, 132, 129, 158, 165, 108, 72, 164, 140, 124, 123, 114, 97, 81, 70, 144, 123, 161, 114, 99, 79, 103, 120, 114, 151, 109, 98, 90, 72, 115, 122, 127, 134, 146, 123, 159, 136, 144, 142, 128, 132, 116, 146, 151, 182, 145, 136, 199, 161, 137, 128, 149, 150, 160, 147, 66, 140, 138, 112, 164, 167, 133, 147, 135, 129, 195, 141, 157, 73, 146, 157, 106, 129, 110, 152, 140, 110, 143, 201, 123, 165, 136, 142, 168, 161, 122, 130, 196, 135, 143, 136, 171, 195, 174, 159, 129, 138, 156, 156, 129, 156, 139, 163, 179, 139, 112, 185, 116, 106, 165, 152, 205, 115, 102, 165, 118, 143, 131, 180, 154, 166, 149, 163, 171, 120, 181, 193, 122, 129, 141, 129, 144, 161, 76, 176, 191, 124, 154, 150, 129, 142, 126, 181, 154, 135, 126, 116, 193, 141, 128, 84, 138, 121, 176, 107, 153, 88, 109, 146, 135, 119, 113, 125, 134, 108, 130, 104, 112, 133, 158, 134, 114, 161, 147, 116, 140, 127, 138, 114, 159, 173, 121, 160, 145, 139, 159, 145, 127, 119, 174, 112, 165, 173, 206, 172, 129, 132, 177, 149, 82, 147, 169, 97, 194, 81, 108, 134, 85, 103, 141, 164, 120, 119, 100, 88, 94, 109, 141, 93, 106, 105, 84, 122, 99, 87, 135, 151, 80, 88, 132, 141, 141, 106, 191, 147, 129, 136, 155, 160, 184, 144, 120, 144, 150, 76, 181, 177, 83, 181, 130, 137, 149, 151, 114, 160, 143, 123, 148, 160, 124, 153, 125, 119, 136, 147, 124, 112, 138, 127, 158, 140, 85, 197, 154, 168, 161, 157, 120, 113, 133, 162, 124, 177, 168, 171, 119, 167, 128, 86, 115, 165, 156, 87, 131, 143, 134, 111, 106, 103, 114, 120, 160, 152, 116, 135, 112, 213, 139, 177, 150, 105, 108, 155, 173, 135, 173, 154, 28, 171, 203, 98, 131, 168, 114, 129, 111, 203, 148, 125, 133, 195, 106, 192, 113, 101, 142, 163, 127, 139, 93, 216, 175, 137, 133, 105, 125, 131, 98, 190, 144, 107, 176, 215, 108, 168, 152, 134, 80, 123, 191, 124, 125, 147, 151, 151, 132, 185, 95, 153, 130, 109, 71, 138, 146, 154, 135, 127, 143, 152, 107, 103, 143, 143, 150, 106, 173, 137, 154, 154, 138, 110, 115, 149, 162, 137, 109, 128, 77, 129, 156, 144, 190, 157, 133, 144, 161, 148, 162, 150, 128, 157, 166, 116, 183, 123, 105, 123, 119, 60, 153, 164, 150, 168, 126, 118, 108, 153, 106, 115, 170, 120, 98, 106, 113, 106, 125, 132, 126, 105, 139, 138, 110, 124, 152, 129, 149, 180, 127, 165, 150, 115, 122, 133, 125, 156, 129, 124, 115, 153, 97, 205, 179, 147, 124, 139, 179, 158, 182, 115, 182, 159, 132, 164, 82, 104, 159, 158, 215, 172, 118, 144, 118, 118, 109, 116, 132, 183, 184, 158, 169, 148, 119, 159, 190, 189, 207, 162, 191, 172, 182, 164, 150, 152, 205, 159, 162, 103, 160, 148, 139, 174, 186, 155, 115, 130, 125, 182, 131, 129, 113, 198, 138, 156, 201, 164, 115, 161, 114, 104, 118, 147, 160, 122, 135, 153, 150, 146, 133, 165, 151, 104, 154, 196, 147, 95, 69, 161, 99, 177, 163, 123, 140, 123, 142, 144, 121, 132, 100, 190, 122, 177, 129, 89, 163, 144, 171, 173, 147, 165, 147, 144, 170, 174, 166, 129, 142, 149, 126, 145, 160, 188, 94, 113, 116, 178, 165, 143, 129, 101, 87, 172, 181, 185, 184, 127, 160, 136, 169, 98, 152, 109, 132, 131, 138, 169, 79, 120, 127, 131, 171, 132, 157, 143, 156, 214, 179, 148, 110, 133, 206, 158, 95, 149, 210, 145, 164, 141, 139, 187, 162, 88, 143, 79, 151, 159, 89, 140, 149, 162, 188, 156, 151, 128, 60, 180, 207, 126, 175, 104, 158, 101, 106, 154, 130, 146, 117, 179, 107, 163, 154, 185, 114, 146, 170, 134, 167, 195, 173, 141, 111, 177, 97, 128, 161, 146, 150, 126, 204, 128, 136, 157, 147, 153, 109, 121, 129, 190, 121, 193, 182, 143, 116, 128, 142, 133, 129, 202, 133, 121, 122, 97, 156, 129, 131, 132, 72, 173, 142, 120, 187, 144, 130, 116, 186, 150, 114, 128, 117, 156, 207, 104, 133, 138, 106, 148, 137, 155, 126, 179, 148, 189, 132, 114, 176, 139, 134, 156, 152, 152, 112, 138, 103, 138, 180, 121, 126, 143, 89, 153, 123, 158, 156, 124, 143, 166, 101, 84, 106, 91, 92, 84, 115, 134, 148, 115, 129, 135, 98, 182, 201, 80, 118, 157, 114, 121, 132, 79, 106, 122, 110, 120, 166, 135, 152, 142, 72, 125, 118, 216, 162, 73, 75, 93, 126, 100, 204, 70, 78, 119, 125, 118, 135, 88, 92, 74, 121, 119, 187, 69, 83, 130, 109, 185, 135, 108, 137, 181, 141, 131, 77, 92, 95, 140, 123, 113, 66, 105, 103, 103, 93, 105, 163, 103, 95, 56, 95, 82, 81, 186, 140, 183, 56, 175, 59, 67, 71, 102, 74, 84, 87, 187, 153, 59, 79, 75, 158, 133, 80, 69, 45, 62, 42, 62, 102, 114, 146, 137, 105, 61, 59, 58, 55, 146, 52, 199, 144, 138, 89, 145, 58, 65, 33, 99, 194, 99, 70, 77, 145, 51, 142, 53, 29, 65, 16, 28, 24, 107, 137, 165, 143, 95, 89, 12, 134, 88, 13, 156, 132, 203, 71, 144, 154, 128, 56, 173, 172, 136, 155, 43, 175, 102, 19, 150, 130, 127, 107, 31, 119, 134, 126, 36, 143, 149, 199, 136, 129, 143, 140, 136, 49, 168, 158, 60, 172, 82, 143, 77, 146, 97, 117, 146, 157, 124, 97, 62, 78, 129, 173, 80, 150, 125, 86, 61, 180, 58, 42, 99, 189, 135, 73, 73, 91, 167, 159, 108, 140, 0, 172, 15, 178, 86, 51, 80, 87, 152, 107, 83, 169, 73, 63, 60, 84, 165, 146, 108, 153, 150, 116, 57, 80, 129, 143, 98, 165, 136, 162, 64, 127, 124, 190, 141, 122, 0, 125, 185, 70, 91, 0, 105, 164, 0, 203, 134, 117, 0, 0, 179, 55, 0, 102, 175, 113, 0, 128, 63, 166, 178, 172, 110, 126, 0, 0, 157, 124, 100, 0, 70, 0, 152, 179, 0, 41, 45, 144, 154, 0, 0, 48, 99, 0, 0, 182, 0, 0, 57, 110, 145, 192, 96, 146, 0, 59, 51, 45, 130, 40, 128, 58, 102, 149, 131, 112, 135, 172, 19, 152, 132, 49, 20, 75, 161, 141, 26, 31, 148, 124, 34, 75, 141, 134, 143, 121, 32, 39, 121, 112, 31, 174, 47, 185, 28, 153, 143, 43, 143, 31, 52, 171, 0, 133, 66, 135, 162, 33, 47, 154, 213, 146, 121, 31, 57, 130, 175, 137, 113, 50, 63, 166, 223, 175, 135, 136, 59, 156, 31, 59, 113, 163, 33, 66, 75, 82, 198, 178, 138, 68, 53, 162, 69, 187, 39, 137, 153, 125, 67, 57, 163, 137, 136, 191, 199, 182, 157, 91, 44, 53, 159, 141, 109, 97, 50, 32, 58, 46, 184, 186, 112, 156, 114, 31, 157, 44, 32, 154, 212, 146, 30, 31, 32, 22, 24, 29, 37, 32, 154, 25, 180, 183, 125, 14, 188, 25, 20, 21, 104, 24, 17, 18, 22, 170, 20, 139, 172, 139, 130, 140, 150, 160, 149, 135, 18, 36, 168, 36, 124, 33, 150, 35, 23, 119, 114, 147, 178, 27, 169, 133, 153, 24, 31, 13, 111, 27, 21, 148, 155, 138, 131, 140, 12, 24, 14, 143, 187, 148, 151, 143, 35, 144, 147, 117, 119, 44, 12, 162, 144, 126, 24, 37, 169, 133, 145, 155, 152, 146, 114, 97, 37, 154, 128, 111, 152, 152, 162, 153, 95, 112, 130, 147, 143, 188, 44, 46, 42, 164, 138, 129, 49, 45, 43, 70, 179, 149, 148, 168, 179, 112, 53, 46, 138, 139, 150, 152, 155, 46, 51, 110, 181, 146, 146, 176, 136, 48, 54, 63, 70, 81, 149, 129, 154, 166, 109, 136, 67, 44, 57, 63, 67, 135, 145, 150, 145, 140, 126, 135, 81, 51, 77, 65, 112, 162, 153, 117, 132, 141, 57, 53, 59, 73, 93, 129, 91, 110, 157, 146, 61, 58, 73, 96, 122, 91, 154, 125, 117, 125, 164, 69, 67, 71, 165, 137, 119, 125, 80, 79, 60, 55, 125, 209, 172, 137, 145, 176, 78, 44, 227, 189, 118, 124, 115, 38, 142, 176, 141, 40, 175, 175, 152, 153, 122, 199, 163, 164, 36, 51, 60, 134, 102, 78, 168, 179, 117, 92, 82, 195, 136, 82, 93, 133, 161, 182, 129, 100, 76, 82, 121, 125, 82, 80, 80, 117, 103, 193, 205, 131, 142, 97, 90, 125, 211, 131, 79, 104, 163, 83, 97, 112, 71, 167, 168, 144, 223, 128, 161, 97, 88, 83, 227, 109, 43, 34, 205, 121, 161, 149, 158, 124, 94, 91, 166, 191, 196, 208, 160, 238, 179, 157, 96, 154, 163, 160, 54, 172, 192, 196, 109, 95, 201, 224, 134, 150, 185, 55, 64, 185, 110, 115, 123, 40, 160, 208, 172, 66, 210, 76, 32, 146, 91, 67, 127, 207, 211, 235, 31, 147, 109, 126, 62, 139, 161, 177, 197, 84, 66, 61, 103, 165, 192, 177, 93, 185, 132, 246, 34, 102, 78, 236, 204, 195, 121, 117, 212, 189, 59, 125, 86, 73, 163, 172, 235, 71, 104, 84, 74, 156, 104, 38, 205, 171, 111, 84, 31, 220, 130, 53, 61, 222, 212, 196, 93, 172, 107, 164, 76, 75, 144, 163, 226, 188, 157, 116, 146, 197, 95, 110, 87, 72, 69, 201, 216, 170, 196, 199, 177, 135, 164, 140, 54, 157, 129, 192, 222, 210, 196, 194, 197, 121, 165, 115, 242, 190, 156, 181, 90, 118, 87, 64, 189, 115, 172, 194, 95, 158, 173, 106, 73, 139, 130, 143, 216, 227, 221, 121, 189, 69, 63, 142, 64, 199, 125, 117, 66, 58, 192, 97, 166, 141, 180, 161, 120, 53, 148, 88, 170, 143, 115, 67, 48, 155, 217, 88, 182, 144, 67, 100, 49, 98, 110, 41, 221, 207, 113, 78, 167, 216, 38, 146, 99, 99, 80, 92, 119, 134, 40, 224, 221, 108, 84, 105, 180, 191, 152, 218, 98, 107, 147, 142, 96, 79, 71, 64, 122, 223, 222, 217, 172, 168, 192, 201, 0, 129, 204, 167, 86, 112, 106, 154, 88, 42, 110, 190, 179, 183, 181, 182, 104, 232, 115, 90, 103, 77, 146, 224, 158, 90, 225, 225, 221, 208, 212, 111, 136, 140, 143, 83, 128, 168, 124, 159, 102, 135, 74, 73, 145, 37, 119, 224, 120, 218, 211, 206, 104, 128, 143, 137, 158, 204, 245, 146, 137, 83, 193, 180, 100, 107, 100, 148, 145, 178, 169, 144, 114, 135, 108, 107, 125, 106, 61, 188, 110, 134, 93, 182, 100, 129, 42, 117, 135, 118, 125, 149, 85, 61, 85, 81, 97, 202, 203, 181, 163, 98, 89, 196, 77, 151, 103, 115, 143, 150, 145, 74, 63, 177, 231, 170, 195, 197, 184, 173, 130, 171, 116, 85, 72, 101, 71, 99, 102, 63, 165, 171, 151, 80, 152, 204, 175, 185, 138, 213, 189, 206, 62, 168, 60, 195, 177, 132, 55, 105, 100, 86, 70, 67, 172, 189, 191, 115, 113, 66, 72, 195, 0, 144, 227, 36, 68, 148, 179, 233, 54, 180, 169, 66, 69, 128, 106, 99, 110, 213, 209, 210, 195, 169, 75, 107, 82, 205, 182, 195, 196, 168, 210, 0, 32, 125, 148, 84, 74, 200, 195, 191, 210, 183, 154, 181, 186, 187, 146, 39, 118, 228, 66, 107, 218, 218, 135, 103, 120, 119, 112, 87, 80, 221, 210, 188, 191, 187, 122, 142, 153, 137, 176, 103, 128, 44, 100, 70, 65, 117, 101, 97, 72, 194, 199, 196, 204, 161, 85, 184, 179, 226, 140, 189, 110, 38, 160, 114, 142, 130, 117, 115, 48, 221, 221, 218, 189, 185, 199, 204, 199, 175, 158, 174, 175, 215, 68, 161, 127, 97, 62, 87, 72, 109, 116, 51, 191, 189, 186, 191, 196, 199, 203, 187, 188, 90, 211, 175, 103, 195, 105, 187, 166, 82, 73, 91, 54, 109, 70, 218, 202, 211, 195, 185, 180, 173, 107, 133, 90, 170, 106, 87, 146, 217, 179, 104, 0, 182, 72, 219, 217, 175, 130, 81, 89, 106, 107, 214, 217, 218, 186, 212, 208, 188, 208, 167, 130, 165, 163, 73, 148, 122, 146, 99, 56, 215, 66, 206, 227, 143, 23, 91, 107, 103, 60, 195, 197, 207, 186, 192, 184, 183, 171, 129, 102, 71, 207, 89, 87, 96, 19, 125, 112, 111, 197, 105, 166, 119, 113, 213, 112, 132, 105, 227, 122, 100, 92, 96, 105, 75, 232, 154, 127, 237, 97, 85, 123, 204, 221, 214, 86, 122, 138, 136, 130, 99, 98, 97, 89, 206, 204, 153, 167, 103, 164, 178, 214, 104, 154, 132, 109, 154, 229, 203, 140, 134, 104, 89, 118, 82, 203, 201, 210, 163, 76, 46, 142, 90, 137, 158, 197, 134, 87, 96, 140, 127, 112, 104, 93, 97, 87, 76, 198, 100, 133, 169, 160, 182, 139, 154, 160, 170, 118, 155, 179, 240, 176, 87, 136, 206, 197, 50, 198, 182, 102, 133, 82, 154, 92, 78, 87, 151, 170, 149, 139, 151, 169, 102, 144, 169, 184, 228, 236, 52, 123, 168, 186, 86, 122, 140, 106, 195, 192, 187, 217, 137, 49, 29, 195, 135, 189, 123, 156, 131, 163, 190, 109, 154, 143, 65, 153, 96, 81, 79, 104, 186, 182, 118, 182, 126, 57, 76, 175, 101, 134, 139, 151, 178, 192, 194, 0, 196, 181, 70, 159, 102, 185, 145, 91, 164, 52, 187, 193, 237, 93, 123, 137, 158, 192, 0, 0, 0, 178, 116, 205, 82, 48, 114, 201, 32, 173, 139, 91, 75, 133, 184, 182, 192, 171, 108, 50, 84, 190, 150, 78, 218, 89, 178, 141, 92, 75, 134, 114, 127, 186, 180, 181, 208, 135, 215, 163, 87, 112, 97, 91, 64, 39, 87, 183, 25, 230, 151, 137, 98, 150, 138, 99, 159, 106, 114, 186, 173, 129, 195, 101, 191, 63, 244, 109, 94, 142, 179, 174, 125, 109, 115, 128, 173, 198, 115, 145, 112, 151, 141, 114, 51, 237, 243, 195, 163, 160, 188, 39, 224, 228, 77, 133, 148, 112, 173, 187, 186, 208, 170, 70, 204, 214, 160, 171, 172, 34, 163, 48, 69, 103, 86, 79, 138, 138, 170, 135, 112, 80, 50, 31, 206, 209, 135, 74, 138, 130, 49, 101, 89, 122, 173, 180, 171, 163, 97, 140, 129, 198, 109, 89, 225, 200, 81, 229, 236, 57, 56, 196, 204, 72, 166, 153, 125, 182, 173, 143, 83, 182, 142, 183, 217, 184, 64, 201, 12, 243, 49, 110, 213, 58, 61, 74, 175, 120, 187, 182, 182, 182, 185, 184, 178, 165, 191, 99, 104, 158, 199, 228, 0, 171, 81, 45, 200, 192, 87, 71, 132, 213, 199, 189, 187, 164, 193, 83, 219, 131, 118, 161, 189, 82, 25, 206, 232, 187, 152, 163, 133, 196, 206, 195, 199, 194, 205, 43, 58, 137, 230, 97, 195, 111, 102, 62, 180, 196, 211, 183, 193, 214, 210, 118, 193, 222, 50, 48, 162, 213, 184, 109, 171, 179, 183, 192, 130, 109, 220, 79, 107, 168, 66, 29, 18, 14, 0, 185, 67, 230, 198, 109, 42, 63, 193, 231, 107, 76, 151, 200, 210, 117, 35, 0, 0, 0, 54, 139, 176, 33, 240, 112, 112, 92, 93, 187, 197, 150, 136, 70, 59, 150, 243, 230, 45, 0, 0, 0, 234, 187, 150, 87, 175, 175, 202, 138, 154, 100, 119, 98, 175, 152, 104, 131, 193, 238, 213, 70, 162, 221, 246, 161, 94, 61, 63, 37, 0, 0, 228, 90, 247, 82, 136, 118, 134, 84, 77, 126, 182, 234, 216, 194, 193, 56, 38, 27, 16, 17, 0, 0, 98, 197, 200, 168, 86, 167, 213, 78, 43, 33, 37, 28, 22, 23, 12, 33, 90, 179, 66, 154, 169, 176, 178, 175, 238, 120, 187, 186, 191, 60, 68, 51, 30, 24, 15, 18, 22, 30, 40, 50, 179, 221, 158, 82, 66, 231, 248, 239, 180, 80, 66, 175, 208, 113, 194, 159, 172, 141, 117, 127, 100, 73, 103, 149, 236, 207, 175, 194, 209, 58, 25, 34, 41, 1, 0, 0, 0, 41, 66, 125, 194, 143, 230, 236, 70, 147, 195, 195, 182, 82, 182, 170, 130, 176, 205, 156, 162, 222, 220, 125, 25, 0, 16, 25, 38, 58, 58, 38, 0, 3, 26, 42, 54, 165, 202, 224, 173, 181, 179, 179, 81, 87, 138, 162, 150, 124, 57, 38, 27, 41, 185, 183, 154, 120, 178, 52, 113, 97, 172, 168, 192, 86, 113, 121, 154, 187, 9, 12, 33, 39, 58, 35, 0, 24, 37, 64, 135, 195, 32, 35, 154, 144, 155, 143, 110, 130, 140, 135, 127, 114, 196, 207, 201, 224, 218, 136, 161, 151, 75, 166, 233, 241, 209, 187, 191, 197, 217, 209, 117, 55, 85, 108, 209, 149, 40, 0, 15, 23, 42, 61, 52, 20, 9, 36, 90, 176, 78, 141, 124, 244, 172, 64, 71, 210, 213, 140, 142, 124, 136, 123, 130, 198, 206, 123, 195, 183, 222, 173, 165, 171, 213, 203, 100, 114, 159, 205, 137, 37, 11, 12, 46, 47, 57, 61, 66, 37, 31, 130, 229, 241, 162, 36, 44, 186, 149, 131, 138, 131, 143, 43, 108, 203, 157, 135, 154, 143, 146, 194, 153, 162, 156, 167, 200, 225, 227, 58, 63, 95, 122, 168, 208, 40, 48, 59, 30, 96, 67, 171, 171, 150, 127, 138, 137, 60, 90, 107, 114, 151, 149, 112, 114, 165, 242, 122, 152, 210, 230, 222, 39, 91, 183, 216, 171, 6, 0, 0, 36, 65, 73, 74, 47, 32, 95, 184, 106, 124, 60, 83, 74, 51, 84, 120, 125, 96, 135, 146, 95, 178, 137, 126, 122, 136, 119, 75, 95, 221, 137, 142, 179, 139, 133, 144, 216, 50, 240, 239, 228, 96, 122, 172, 200, 225, 206, 111, 30, 60, 147, 213, 157, 3, 0, 0, 37, 48, 59, 72, 59, 44, 109, 215, 189, 184, 89, 60, 109, 152, 143, 97, 183, 124, 81, 154, 122, 122, 137, 141, 82, 219, 126, 161, 176, 152, 113, 150, 127, 150, 210, 149, 217, 97, 96, 60, 99, 199, 121, 46, 29, 41, 55, 64, 60, 72, 78, 130, 57, 96, 95, 81, 174, 137, 163, 193, 155, 158, 161, 179, 177, 156, 100, 186, 115, 176, 165, 218, 130, 84, 66, 142, 203, 209, 40, 59, 101, 168, 207, 216, 182, 94, 31, 32, 44, 66, 61, 119, 121, 230, 199, 107, 173, 89, 111, 125, 80, 76, 111, 85, 183, 163, 150, 131, 125, 138, 138, 115, 115, 61, 175, 153, 196, 152, 138, 226, 63, 59, 75, 131, 196, 192, 133, 50, 59, 98, 164, 199, 163, 87, 28, 0, 15, 22, 43, 57, 58, 76, 95, 130, 157, 145, 139, 165, 228, 97, 74, 41, 85, 69, 63, 158, 140, 146, 39, 177, 107, 186, 122, 115, 77, 149, 213, 190, 213, 140, 135, 141, 114, 98, 225, 217, 89, 47, 74, 214, 209, 63, 58, 87, 150, 188, 182, 88, 23, 38, 45, 54, 154, 141, 99, 215, 198, 108, 114, 165, 92, 173, 194, 168, 195, 150, 135, 123, 99, 219, 159, 183, 174, 130, 128, 132, 155, 126, 135, 157, 98, 117, 224, 155, 87, 52, 72, 110, 206, 200, 108, 80, 75, 59, 134, 175, 92, 64, 50, 44, 39, 29, 35, 53, 170, 231, 181, 113, 95, 141, 58, 115, 165, 173, 177, 198, 167, 139, 187, 167, 197, 192, 86, 146, 178, 166, 76, 88, 81, 157, 67, 63, 54, 32, 29, 110, 138, 96, 109, 90, 156, 149, 184, 107, 94, 178, 193, 153, 187, 184, 190, 113, 173, 168, 39, 82, 81, 114, 144, 154, 133, 98, 86, 137, 149, 131, 83, 76, 70, 46, 77, 82, 122, 79, 112, 189, 184, 160, 105, 177, 105, 165, 159, 194, 180, 191, 96, 120, 176, 183, 73, 94, 97, 128, 108, 132, 107, 61, 68, 79, 63, 69, 177, 202, 204, 169, 172, 181, 173, 159, 180, 92, 77, 107, 200, 219, 194, 202, 174, 171, 148, 161, 173, 148, 166, 197, 181, 110, 131, 111, 84, 103, 76, 58, 61, 91, 164, 137, 182, 182, 206, 230, 236, 93, 95, 117, 155, 199, 189, 118, 99, 132, 168, 196, 106, 128, 176, 184, 144, 163, 199, 95, 137, 166, 202, 78, 123, 164, 179, 178, 102, 85, 51, 50, 80, 121, 138, 157, 188, 183, 103, 157, 146, 138, 191, 159, 96, 124, 144, 146, 156, 175, 164, 157, 142, 144, 181, 166, 181, 179, 210, 159, 185, 124, 100, 70, 72, 73, 48, 96, 177, 106, 99, 118, 179, 186, 92, 107, 210, 180, 128, 169, 129, 160, 171, 167, 108, 158, 182, 162, 168, 110, 143, 149, 151, 99, 111, 186, 189, 209, 184, 125, 122, 111, 60, 64, 54, 48, 69, 169, 182, 189, 169, 203, 103, 41, 146, 126, 129, 104, 109, 140, 124, 131, 161, 113, 152, 157, 90, 146, 152, 171, 140, 119, 92, 195, 182, 175, 204, 127, 182, 166, 113, 108, 118, 104, 102, 85, 73, 59, 73, 64, 56, 151, 161, 164, 170, 188, 232, 153, 153, 186, 112, 95, 178, 218, 99, 134, 177, 115, 162, 155, 165, 157, 118, 163, 192, 183, 184, 86, 115, 88, 175, 166, 153, 194, 180, 116, 177, 165, 84, 101, 95, 92, 84, 78, 65, 66, 58, 55, 97, 182, 185, 186, 183, 161, 106, 127, 108, 155, 97, 115, 208, 190, 92, 193, 131, 97, 127, 166, 198, 206, 107, 161, 147, 185, 69, 144, 171, 141, 169, 159, 90, 95, 96, 91, 83, 81, 83, 83, 64, 132, 177, 204, 137, 183, 118, 111, 189, 187, 140, 195, 91, 128, 161, 98, 106, 168, 87, 153, 123, 96, 147, 170, 161, 153, 144, 145, 163, 127, 183, 86, 172, 214, 157, 80, 105, 135, 160, 164, 88, 94, 88, 84, 80, 86, 91, 42, 77, 189, 107, 130, 182, 119, 176, 188, 210, 198, 191, 158, 54, 39, 121, 33, 174, 71, 95, 202, 153, 161, 149, 180, 123, 160, 168, 157, 174, 166, 151, 155, 150, 186, 114, 172, 211, 115, 56, 89, 198, 225, 75, 62, 75, 109, 151, 113, 82, 92, 78, 81, 90, 80, 82, 90, 88, 80, 56, 108, 123, 106, 190, 123, 165, 174, 145, 154, 82, 152, 156, 136, 120, 168, 178, 163, 163, 124, 186, 157, 97, 75, 62, 89, 130, 160, 171, 97, 94, 70, 75, 83, 88, 91, 92, 102, 141, 223, 118, 176, 177, 183, 148, 163, 187, 197, 229, 50, 127, 127, 173, 173, 173, 175, 93, 127, 101, 117, 142, 163, 100, 174, 142, 69, 117, 134, 180, 225, 155, 36, 149, 197, 168, 138, 129, 99, 79, 65, 69, 98, 149, 183, 160, 115, 46, 55, 74, 93, 115, 139, 143, 165, 232, 146, 106, 145, 98, 165, 113, 130, 149, 183, 143, 171, 172, 124, 143, 160, 161, 165, 148, 154, 151, 82, 96, 98, 103, 135, 116, 92, 115, 132, 114, 211, 207, 23, 137, 215, 137, 154, 120, 169, 127, 123, 122, 112, 63, 57, 55, 137, 185, 116, 72, 48, 36, 55, 100, 149, 220, 112, 192, 137, 113, 143, 167, 148, 65, 96, 149, 173, 188, 165, 182, 154, 150, 151, 141, 106, 127, 143, 140, 99, 121, 151, 133, 155, 176, 188, 127, 149, 91, 171, 159, 128, 114, 114, 116, 106, 86, 65, 121, 178, 80, 66, 49, 87, 196, 172, 179, 137, 151, 149, 133, 162, 157, 147, 166, 164, 194, 51, 157, 113, 135, 75, 138, 155, 149, 165, 127, 152, 125, 91, 128, 137, 91, 170, 194, 232, 150, 176, 216, 162, 120, 130, 77, 144, 113, 100, 103, 81, 50, 50, 52, 101, 159, 75, 67, 75, 70, 141, 179, 98, 163, 174, 156, 154, 146, 100, 129, 180, 137, 196, 90, 130, 147, 146, 180, 181, 148, 123, 140, 206, 183, 77, 162, 117, 91, 93, 79, 71, 65, 61, 62, 58, 71, 135, 112, 65, 85, 140, 179, 174, 119, 166, 141, 171, 84, 54, 49, 82, 196, 195, 196, 184, 116, 121, 180, 175, 175, 176, 162, 105, 83, 152, 141, 165, 144, 233, 158, 86, 181, 133, 97, 81, 81, 73, 60, 64, 76, 69, 46, 133, 109, 78, 70, 75, 93, 150, 146, 174, 100, 111, 187, 217, 215, 49, 79, 163, 118, 78, 97, 79, 169, 180, 166, 177, 170, 176, 169, 145, 148, 126, 96, 80, 192, 124, 163, 152, 131, 136, 152, 118, 214, 158, 132, 189, 112, 184, 77, 167, 183, 228, 80, 71, 61, 59, 75, 78, 73, 47, 68, 99, 99, 91, 108, 167, 212, 153, 108, 134, 119, 145, 211, 219, 215, 22, 82, 207, 101, 153, 188, 156, 151, 109, 109, 133, 145, 154, 157, 150, 169, 157, 75, 154, 131, 190, 146, 102, 132, 96, 118, 106, 196, 176, 234, 162, 131, 89, 66, 62, 65, 59, 60, 75, 94, 106, 124, 145, 149, 161, 210, 98, 126, 143, 165, 174, 104, 52, 89, 156, 172, 215, 83, 210, 166, 80, 209, 139, 138, 174, 173, 121, 107, 193, 144, 132, 146, 110, 104, 183, 216, 158, 145, 233, 215, 121, 99, 96, 78, 58, 63, 62, 66, 69, 76, 97, 126, 159, 178, 186, 110, 128, 160, 133, 58, 104, 134, 185, 216, 191, 179, 218, 85, 154, 79, 109, 121, 175, 109, 192, 203, 129, 134, 163, 168, 221, 143, 149, 177, 137, 95, 92, 71, 61, 61, 51, 67, 85, 165, 169, 170, 134, 131, 60, 121, 152, 150, 189, 176, 227, 230, 213, 174, 124, 89, 101, 152, 94, 125, 158, 158, 51, 96, 203, 136, 158, 219, 150, 134, 164, 85, 81, 67, 61, 59, 59, 66, 56, 50, 100, 132, 159, 177, 187, 215, 118, 149, 177, 189, 119, 156, 147, 217, 55, 64, 80, 126, 217, 32, 96, 128, 144, 195, 107, 77, 156, 108, 131, 160, 148, 124, 132, 159, 195, 158, 186, 82, 144, 215, 206, 92, 78, 79, 72, 62, 51, 51, 58, 63, 57, 119, 138, 157, 214, 216, 137, 195, 193, 195, 196, 80, 105, 135, 136, 164, 211, 70, 67, 101, 170, 51, 39, 179, 144, 190, 190, 141, 114, 110, 153, 187, 151, 148, 132, 107, 99, 225, 152, 119, 69, 69, 45, 42, 59, 70, 91, 117, 162, 168, 219, 193, 111, 143, 146, 194, 168, 217, 224, 48, 107, 99, 112, 142, 155, 181, 70, 88, 228, 167, 49, 146, 132, 189, 89, 73, 126, 133, 150, 157, 105, 112, 205, 199, 184, 108, 172, 208, 150, 114, 95, 85, 76, 52, 36, 49, 42, 52, 65, 120, 146, 167, 150, 179, 115, 156, 215, 37, 66, 111, 108, 133, 134, 115, 94, 210, 178, 91, 96, 111, 131, 238, 217, 115, 99, 139, 131, 136, 82, 147, 234, 83, 193, 156, 78, 114, 122, 148, 206, 155, 149, 171, 210, 136, 112, 93, 84, 32, 49, 60, 172, 209, 223, 197, 145, 165, 123, 126, 44, 127, 114, 110, 191, 211, 110, 123, 144, 140, 202, 84, 151, 101, 152, 208, 204, 208, 139, 142, 145, 104, 83, 74, 59, 42, 37, 37, 68, 126, 153, 94, 175, 192, 181, 177, 211, 199, 0, 18, 129, 157, 55, 115, 223, 41, 70, 149, 67, 101, 136, 143, 88, 101, 133, 115, 200, 181, 185, 125, 122, 208, 216, 154, 156, 203, 189, 142, 159, 223, 189, 214, 204, 216, 93, 53, 43, 32, 47, 42, 56, 74, 104, 131, 130, 178, 97, 117, 190, 166, 157, 134, 198, 136, 97, 179, 148, 104, 21, 100, 101, 177, 176, 159, 123, 111, 130, 178, 129, 81, 123, 217, 109, 174, 56, 86, 29, 35, 52, 55, 49, 53, 66, 86, 112, 166, 125, 130, 151, 114, 101, 83, 179, 40, 182, 93, 47, 117, 84, 159, 32, 120, 120, 109, 147, 155, 175, 134, 164, 214, 134, 144, 240, 219, 74, 18, 22, 35, 45, 57, 72, 115, 132, 173, 113, 187, 163, 168, 55, 142, 142, 93, 191, 143, 139, 131, 115, 117, 114, 181, 185, 138, 213, 179, 53, 65, 113, 229, 164, 186, 117, 25, 26, 48, 66, 71, 99, 109, 144, 192, 210, 211, 117, 175, 132, 101, 145, 199, 143, 188, 42, 63, 216, 187, 54, 145, 46, 121, 70, 84, 136, 115, 135, 102, 71, 145, 161, 66, 76, 123, 164, 73, 35, 38, 47, 71, 84, 90, 94, 102, 216, 209, 119, 120, 213, 174, 55, 204, 76, 59, 115, 80, 133, 123, 57, 108, 128, 105, 105, 64, 130, 121, 125, 207, 187, 165, 163, 194, 169, 26, 13, 39, 43, 51, 88, 99, 82, 227, 132, 176, 105, 133, 171, 114, 177, 93, 152, 202, 197, 206, 204, 197, 168, 155, 199, 189, 107, 123, 132, 174, 124, 214, 131, 133, 113, 115, 134, 191, 82, 119, 188, 94, 155, 119, 143, 53, 18, 30, 61, 112, 100, 201, 113, 110, 112, 97, 80, 132, 160, 139, 136, 153, 72, 96, 111, 196, 92, 74, 89, 70, 177, 121, 122, 71, 70, 129, 124, 117, 181, 130, 148, 168, 164, 224, 117, 223, 92, 69, 44, 27, 24, 34, 42, 55, 70, 77, 104, 183, 121, 97, 212, 195, 178, 82, 198, 166, 120, 118, 139, 199, 96, 86, 137, 232, 186, 198, 157, 133, 90, 42, 30, 27, 35, 47, 62, 73, 130, 143, 100, 105, 175, 170, 127, 167, 146, 208, 158, 137, 216, 160, 110, 72, 188, 143, 112, 54, 152, 159, 109, 122, 168, 170, 120, 62, 42, 33, 48, 64, 124, 192, 205, 132, 85, 159, 117, 140, 163, 221, 134, 170, 136, 163, 156, 150, 130, 128, 108, 167, 132, 186, 170, 92, 73, 181, 209, 202, 66, 42, 42, 44, 49, 63, 86, 187, 99, 164, 143, 139, 183, 202, 155, 92, 165, 66, 121, 156, 215, 194, 92, 78, 131, 159, 118, 186, 151, 147, 128, 164, 159, 96, 111, 78, 47, 51, 55, 59, 72, 99, 104, 193, 172, 114, 69, 109, 133, 186, 96, 111, 154, 172, 215, 218, 64, 137, 84, 95, 149, 116, 125, 117, 111, 203, 200, 165, 119, 200, 148, 76, 223, 185, 190, 92, 58, 60, 71, 78, 104, 191, 215, 146, 104, 138, 185, 26, 60, 110, 107, 132, 113, 161, 139, 126, 97, 110, 101, 114, 205, 127, 102, 110, 136, 82, 107, 95, 99, 98, 187, 174, 165, 104, 146, 165, 149, 97, 127, 156, 199, 198, 131, 178, 158, 117, 55, 118, 109, 139, 173, 172, 105, 81, 194, 97, 89, 201, 160, 135, 169, 181, 157, 110, 161, 137, 49, 153, 221, 152, 141, 165, 158, 127, 194, 214, 172, 161, 227, 216, 207, 119, 83, 154, 162, 183, 154, 40, 39, 162, 153, 112, 111, 130, 170, 186, 200, 132, 235, 118, 142, 161, 148, 144, 106, 63, 185, 205, 162, 111, 68, 223, 193, 159, 186, 99, 178, 147, 134, 119, 142, 204, 157, 203, 185, 75, 135, 225, 144, 168, 113, 151, 146, 80, 177, 232, 97, 118, 87, 160, 144, 144, 104, 82, 139, 80, 152, 222, 198, 144, 156, 180, 63, 69, 144, 103, 179, 218, 197, 93, 171, 89, 87, 76, 213, 129, 44, 117, 155, 162, 169, 216, 196, 182, 143, 156, 60, 133, 136, 166, 89, 147, 215, 112, 92, 100, 104, 125, 98, 180, 110, 165, 140, 75, 125, 223, 107, 162, 139, 216, 215, 158, 135, 144, 108, 161, 23, 115, 0, 64, 145, 69, 199, 91, 107, 140, 219, 180, 156, 124, 55, 156, 202, 213, 180, 128, 150, 181, 191, 94, 105, 118, 217, 218, 133, 123, 220, 226, 172, 169, 84, 123, 157, 115, 64, 89, 147, 185, 180, 176, 114, 143, 192, 153, 168, 190, 209, 92, 103, 172, 171, 212, 157, 134, 71, 116, 132, 147, 114, 145, 189, 196, 187, 210, 139, 140, 112, 82, 205, 114, 116, 220, 181, 95, 176, 64, 121, 78, 107, 152, 189, 194, 196, 209, 143, 199, 176, 193, 213, 98, 92, 209, 85, 92, 208, 216, 216, 184, 225, 98, 102, 194, 163, 77, 115, 158, 181, 179, 182, 187, 193, 123, 216, 174, 176, 212, 245, 163, 86, 110, 104, 213, 109, 116, 206, 102, 186, 175, 139, 158, 146, 161, 143, 115, 108, 94, 159, 185, 76, 188, 167, 127, 170, 170, 63, 207, 222, 78, 79, 176, 89, 166, 144, 104, 221, 113, 88, 92, 160, 123, 88, 123, 173, 164, 202, 80, 172, 140, 158, 151, 83, 197, 119, 130, 150, 103, 75, 115, 216, 206, 184, 90, 113, 125, 185, 205, 153, 145, 175, 179, 118, 193, 224, 198, 111, 125, 133, 158, 174, 64, 119, 124, 98, 64, 125, 202, 168, 75, 80, 102, 167, 96, 78, 142, 76, 177, 197, 195, 214, 178, 74, 115, 213, 148, 131, 152, 178, 99, 87, 113, 170, 133, 131, 141, 170, 158, 100, 192, 156, 149, 154, 95, 117, 151, 158, 141, 123, 90, 68, 219, 175, 68, 164, 181, 70, 137, 128, 178, 207, 168, 202, 129, 109, 125, 142, 103, 120, 103, 89, 72, 91, 171, 143, 124, 68, 72, 156, 177, 165, 145, 112, 59, 156, 118, 123, 151, 112, 100, 216, 222, 177, 188, 168, 81, 113, 160, 138, 164, 201, 102, 115, 160, 157, 153, 69, 64, 152, 157, 153, 104, 157, 211, 160, 156, 95, 147, 125, 151, 133, 81, 111, 121, 201, 148, 168, 168, 182, 175, 179, 217, 153, 189, 91, 95, 181, 162, 207, 189, 86, 85, 160, 152, 97, 135, 180, 164, 124, 76, 138, 152, 143, 148, 144, 210, 147, 100, 98, 170, 86, 190, 98, 185, 80, 85, 165, 99, 119, 183, 201, 201, 158, 100, 192, 147, 148, 167, 134, 162, 193, 152, 187, 173, 171, 185, 107, 164, 120, 134, 169, 191, 166, 105, 109, 107, 228, 159, 162, 164, 156, 132, 157, 169, 187, 126, 170, 78, 76, 95, 128, 158, 170, 164, 144, 123, 186, 122, 136, 121, 133, 192, 196, 95, 71, 73, 96, 158, 156, 152, 173, 217, 209, 171, 139, 128, 124, 123, 131, 135, 140, 211, 116, 75, 105, 101, 156, 171, 170, 227, 158, 132, 138, 149, 88, 126, 169, 178, 147, 99, 168, 156, 101, 107, 119, 139, 151, 144, 133, 141, 140, 155, 115, 90, 120, 104, 108, 169, 122, 89, 199, 238, 192, 158, 155, 117, 137, 159, 97, 106, 118, 177, 144, 90, 101, 195, 168, 238, 235, 176, 142, 128, 157, 176, 91, 117, 117, 26, 169, 124, 124, 126, 170, 180, 46, 96, 109, 147, 144, 166, 147, 132, 121, 135, 127, 162, 179, 100, 96, 91, 127, 114, 114, 136, 134, 188, 218, 213, 78, 130, 154, 186, 181, 164, 102, 32, 156, 194, 82, 97, 213, 119, 187, 157, 153, 119, 96, 123, 157, 65, 141, 214, 214, 207, 170, 162, 150, 109, 103, 104, 89, 148, 161, 92, 188, 213, 201, 195, 196, 180, 45, 105, 132, 130, 149, 158, 112, 94, 94, 150, 154, 206, 189, 122, 199, 147, 141, 225, 207, 201, 55, 132, 135, 148, 143, 143, 132, 153, 116, 137, 73, 197, 107, 152, 120, 133, 60, 119, 121, 132, 103, 112, 84, 149, 143, 152, 126, 140, 66, 169, 216, 190, 146, 155, 108, 231, 154, 109, 113, 126, 124, 110, 122, 115, 111, 82, 142, 150, 120, 88, 152, 82, 170, 207, 173, 167, 110, 157, 151, 113, 115, 104, 70, 104, 94, 157, 128, 155, 208, 185, 117, 103, 155, 136, 127, 169, 134, 149, 120, 61, 92, 101, 153, 106, 109, 151, 148, 142, 141, 170, 154, 94, 84, 129, 141, 105, 140, 139, 147, 145, 82, 226, 229, 74, 89, 129, 127, 159, 132, 224, 77, 98, 78, 108, 133, 143, 142, 141, 130, 78, 99, 143, 124, 130, 211, 137, 134, 144, 148, 148, 99, 198, 76, 129, 144, 142, 143, 131, 132, 141, 99, 206, 185, 179, 149, 115, 128, 132, 142, 127, 135, 143, 136, 143, 134, 100, 213, 76, 89, 73, 195, 129, 137, 132, 141, 213, 193, 117, 78, 183, 135, 147, 121, 133, 140, 140, 137, 125, 146, 220, 222, 178, 166, 99, 192, 136, 128, 139, 66, 165, 125, 179, 171, 101, 112, 133, 132, 131, 214, 201, 192, 140, 123, 139, 211, 212, 131, 110, 134, 126, 149, 147, 115, 98, 138, 142, 132, 200, 46, 104, 193, 116, 185, 136, 100, 142, 142, 187, 143, 94, 47, 169, 156, 110, 114, 103, 133, 183, 163, 148, 136, 141, 123, 160, 168, 103, 172, 142, 115, 139, 191, 166, 149, 157, 146, 140, 214, 110, 115, 102, 105, 116, 148, 158, 212, 185, 101, 45, 114, 88, 59, 161, 138, 143, 154, 142, 158, 158, 136, 155, 123, 196, 89, 92, 218, 80, 126, 122, 118, 112, 90, 97, 115, 128, 127, 133, 119, 204, 89, 63, 144, 138, 145, 162, 151, 155, 118, 164, 145, 114, 187, 144, 174, 123, 127, 81, 140, 206, 153, 166, 143, 165, 104, 91, 82, 120, 156, 138, 128, 143, 131, 97, 133, 134, 149, 192, 175, 126, 113, 77, 190, 189, 144, 162, 154, 121, 124, 142, 211, 95, 173, 128, 152, 119, 213, 119, 164, 100, 179, 166, 174, 180, 160, 133, 131, 145, 110, 102, 124, 163, 163, 169, 218, 110, 172, 120, 193, 130, 117, 172, 136, 143, 187, 143, 143, 157, 152, 127, 118, 134, 165, 174, 119, 77, 123, 145, 192, 181, 157, 203, 192, 217, 114, 110, 140, 145, 116, 184, 125, 134, 137, 169, 198, 149, 127, 131, 195, 185, 109, 174, 158, 116, 171, 198, 150, 150, 131, 132, 122, 195, 111, 170, 133, 129, 128, 139, 157, 197, 127, 121, 155, 213, 150, 144, 198, 164, 153, 179, 217, 206, 118, 107, 125, 138, 180, 202, 213, 199, 141, 126, 114, 74, 132, 133, 137, 149, 216, 25, 169, 174, 206, 212, 197, 188, 127, 195, 160, 129, 131, 187, 208, 208, 124, 125, 212, 175, 175, 179, 209, 192, 140, 124, 207, 115, 207, 127, 123, 183, 184, 195, 207, 209, 207, 202, 197, 114, 117, 107, 132, 113, 64, 169, 221, 155, 211, 199, 203, 147, 104, 199, 199, 170, 176, 106, 150, 104, 126, 213, 212, 218, 203, 206, 126, 73, 138, 182, 173, 129, 81, 211, 131, 82, 138, 174, 179, 185, 176, 142, 200, 141, 220, 156, 144, 148, 181, 144, 71, 202, 144, 167, 164, 196, 183, 182, 142, 205, 138, 112, 149, 120, 186, 147, 140, 149, 151, 140, 126, 129, 232, 160, 148, 191, 184, 93, 136, 147, 217, 144, 190, 164, 216, 186, 115, 126, 173, 190, 141, 188, 218, 139, 93, 157, 117, 181, 146, 137, 180, 201, 171, 203, 191, 215, 192, 113, 132, 130, 148, 152, 206, 125, 100, 174, 138, 110, 129, 128, 197, 86, 129, 121, 118, 132, 144, 152, 135, 68, 135, 118, 111, 212, 142, 106, 103, 160, 182, 201, 73, 140, 128, 140, 107, 150, 125, 133, 157, 153, 150, 169, 131, 144, 54, 185, 163, 115, 138, 118, 147, 35, 97, 141, 161, 138, 138, 183, 110, 98, 139, 136, 113, 138, 141, 175, 142, 124, 114, 135, 119, 142, 154, 107, 138, 106, 118, 118, 105, 43, 85, 144, 192, 162, 162, 115, 114, 138, 160, 133, 111, 137, 109, 142, 47, 109, 116, 98, 149, 102, 116, 145, 130, 134, 130, 127, 110, 81, 157, 114, 162, 163, 125, 122, 164, 91, 128, 151, 63, 95, 126, 115, 135, 151, 162, 169, 167, 137, 133, 132, 115, 105, 47, 111, 80, 180, 152, 120, 175, 174, 150, 202, 142, 122, 153, 145, 157, 152, 138, 169, 147, 144, 85, 81, 95, 223, 185, 183, 116, 150, 169, 123, 149, 161, 149, 141, 67, 68, 116, 133, 159, 166, 104, 130, 161, 163, 137, 157, 111, 132, 162, 107, 183, 188, 184, 104, 109, 160, 156, 155, 127, 112, 96, 103, 131, 187, 187, 133, 141, 131, 149, 151, 150, 89, 111, 136, 136, 105, 152, 91, 98, 101, 97, 100, 166, 125, 174, 160, 77, 131, 144, 140, 103, 146, 71, 187, 118, 133, 140, 125, 107, 114, 177, 129, 131, 219, 63, 124, 125, 154, 175, 182, 162, 99, 127, 152, 140, 96, 100, 158, 115, 114, 123, 163, 191, 188, 181, 167, 103, 106, 106, 91, 122, 116, 98, 131, 148, 102, 120, 191, 125, 92, 161, 96, 140, 126, 184, 151, 131, 121, 139, 107, 107, 58, 155, 119, 111, 94, 145, 105, 248, 111, 123, 176, 138, 144, 147, 132, 129, 149, 109, 118, 110, 133, 79, 159, 119, 120, 120, 116, 184, 138, 128, 119, 171, 105, 84, 147, 171, 188, 134, 109, 118, 118, 170, 118, 124, 111, 186, 112, 99, 160, 136, 135, 144, 106, 122, 85, 92, 99, 172, 178, 174, 178, 141, 142, 129, 114, 92, 113, 117, 102, 125, 132, 191, 103, 133, 142, 130, 124, 177, 182, 131, 129, 112, 132, 121, 110, 126, 123, 137, 154, 179, 110, 116, 79, 69, 172, 190, 113, 171, 157, 133, 140, 139, 136, 125, 189, 124, 174, 132, 118, 182, 191, 149, 176, 166, 132, 44, 189, 166, 177, 186, 129, 125, 113, 119, 131, 139, 102, 134, 165, 124, 156, 130, 146, 78, 176, 176, 190, 119, 125, 110, 87, 96, 123, 149, 148, 131, 188, 162, 157, 72, 39, 156, 182, 169, 123, 152, 147, 191, 126, 131, 156, 165, 149, 150, 233, 59, 97, 179, 180, 183, 189, 127, 109, 183, 169, 140, 112, 119, 182, 191, 123, 117, 122, 121, 127, 118, 155, 183, 171, 115, 156, 168, 98, 151, 144, 127, 86, 127, 115, 121, 186, 114, 170, 129, 133, 156, 187, 127, 128, 140, 135, 142, 208, 90, 124, 102, 171, 114, 146, 148, 93, 183, 104, 208, 172, 181, 196, 177, 200, 115, 84, 105, 107, 112, 139, 91, 151, 169, 161, 184, 175, 195, 97, 110, 107, 127, 115, 134, 100, 78, 175, 183, 161, 87, 89, 66, 163, 115, 118, 169, 57, 134, 156, 109, 79, 83, 187, 161, 102, 61, 79, 186, 188, 132, 176, 172, 80, 79, 136, 103, 199, 72, 92, 95, 167, 168, 158, 192, 190, 105, 139, 134, 119, 81, 155, 172, 167, 86, 79, 113, 99, 94, 162, 205, 184, 99, 85, 148, 97, 131, 169, 167, 186, 123, 90, 87, 102, 72, 79, 142, 91, 123, 117, 103, 156, 165, 152, 158, 213, 170, 104, 72, 80, 102, 70, 58, 138, 146, 147, 142, 143, 125, 134, 90, 173, 115, 127, 134, 113, 149, 111, 97, 80, 193, 121, 143, 145, 129, 118, 92, 115, 120, 82, 96, 169, 69, 91, 108, 123, 130, 135, 145, 149, 124, 127, 89, 114, 160, 196, 159, 105, 17, 126, 140, 137, 139, 143, 141, 131, 128, 166, 169, 178, 150, 191, 172, 121, 164, 126, 121, 185, 201, 135, 136, 136, 141, 123, 145, 147, 206, 170, 165, 112, 127, 96, 87, 98, 177, 174, 45, 90, 116, 118, 136, 138, 140, 141, 119, 117, 178, 88, 104, 116, 211, 200, 164, 132, 77, 41, 43, 163, 197, 106, 111, 122, 146, 147, 136, 141, 84, 78, 175, 83, 109, 161, 154, 116, 80, 99, 40, 43, 138, 195, 128, 108, 130, 158, 150, 152, 149, 139, 135, 130, 165, 102, 85, 133, 137, 106, 122, 111, 106, 49, 41, 179, 125, 166, 127, 138, 160, 143, 140, 143, 169, 170, 192, 81, 172, 26, 122, 137, 160, 133, 73, 145, 74, 186, 106, 199, 205, 107, 67, 174, 74, 40, 166, 167, 171, 131, 112, 104, 127, 142, 122, 162, 133, 135, 115, 81, 148, 166, 181, 205, 198, 105, 30, 22, 189, 172, 164, 174, 135, 123, 137, 110, 129, 127, 100, 180, 110, 179, 147, 85, 154, 64, 42, 159, 176, 133, 121, 145, 114, 92, 170, 105, 61, 68, 37, 167, 172, 159, 174, 163, 121, 130, 134, 149, 153, 88, 163, 100, 153, 177, 109, 163, 169, 154, 151, 149, 131, 121, 139, 143, 122, 156, 148, 150, 182, 116, 147, 159, 170, 55, 37, 158, 155, 157, 143, 138, 134, 163, 141, 149, 134, 149, 131, 110, 181, 127, 126, 128, 77, 45, 166, 159, 162, 167, 159, 160, 148, 149, 136, 166, 184, 141, 148, 150, 129, 121, 133, 130, 142, 135, 183, 182, 133, 25, 54, 158, 145, 141, 127, 137, 115, 102, 107, 140, 150, 133, 184, 182, 167, 150, 146, 180, 134, 152, 132, 119, 139, 138, 121, 148, 107, 170, 160, 101, 136, 145, 159, 166, 148, 157, 110, 132, 110, 106, 109, 150, 154, 73, 117, 157, 168, 166, 151, 142, 154, 141, 162, 165, 164, 170, 118, 106, 131, 107, 110, 140, 162, 156, 156, 115, 147, 162, 158, 154, 84, 166, 166, 152, 135, 114, 142, 152, 138, 138, 184, 143, 145, 77, 182, 180, 170, 151, 38, 113, 161, 150, 93, 149, 138, 135, 169, 162, 138, 153, 169, 136, 115, 150, 132, 158, 168, 138, 139, 143, 184, 179, 199, 156, 176, 175, 151, 143, 174, 153, 181, 155, 130, 100, 154, 120, 127, 130, 140, 130, 168, 163, 166, 144, 118, 172, 191, 175, 168, 155, 172, 176, 149, 152, 156, 155, 126, 126, 150, 136, 131, 133, 138, 206, 155, 80, 95, 174, 149, 178, 152, 163, 129, 136, 130, 137, 160, 121, 195, 155, 172, 168, 138, 135, 174, 151, 150, 123, 129, 98, 111, 177, 134, 148, 155, 154, 79, 65, 234, 158, 164, 180, 190, 108, 92, 140, 109, 140, 132, 152, 133, 138, 146, 79, 66, 30, 173, 189, 154, 155, 151, 127, 132, 97, 133, 143, 139, 154, 120, 74, 239, 164, 157, 126, 195, 181, 182, 190, 158, 128, 165, 138, 168, 161, 141, 135, 160, 132, 181, 169, 153, 174, 165, 176, 169, 152, 149, 149, 156, 148, 138, 132, 59, 191, 173, 196, 162, 194, 173, 176, 181, 156, 158, 143, 137, 150, 126, 108, 78, 122, 132, 143, 148, 53, 183, 171, 167, 154, 155, 163, 150, 92, 134, 116, 99, 164, 138, 129, 134, 107, 165, 157, 100, 160, 138, 136, 140, 129, 89, 137, 136, 128, 141, 81, 183, 164, 0, 0, 0, 139, 124, 140, 119, 163, 144, 136, 142, 120, 118, 145, 91, 51, 100, 173, 158, 143, 138, 142, 161, 143, 137, 124, 143, 118, 176, 175, 128, 118, 135, 136, 127, 112, 145, 100, 78, 184, 152, 159, 133, 123, 136, 135, 149, 161, 73, 90, 64, 157, 0, 172, 154, 154, 152, 110, 98, 96, 117, 133, 187, 0, 146, 159, 143, 43, 91, 114, 160, 120, 154, 105, 124, 130, 82, 77, 63, 190, 0, 82, 176, 165, 128, 135, 185, 152, 151, 99, 42, 17, 189, 188, 157, 131, 111, 172, 145, 136, 24, 155, 63, 171, 190, 137, 142, 133, 112, 103, 119, 107, 138, 129, 112, 154, 175, 125, 132, 177, 91, 79, 56, 179, 160, 180, 153, 158, 160, 133, 97, 82, 6, 185, 143, 191, 150, 170, 176, 23, 162, 110, 152, 122, 166, 127, 104, 73, 49, 169, 151, 171, 32, 86, 149, 129, 142, 131, 124, 133, 110, 97, 28, 166, 144, 144, 171, 160, 118, 116, 118, 132, 152, 133, 102, 129, 113, 121, 141, 63, 47, 17, 21, 165, 156, 148, 165, 0, 164, 144, 104, 87, 116, 103, 154, 156, 106, 87, 26, 55, 39, 25, 154, 152, 157, 171, 146, 137, 110, 147, 144, 146, 159, 172, 163, 127, 107, 92, 127, 190, 191, 114, 124, 96, 116, 71, 0, 37, 163, 160, 159, 155, 150, 0, 147, 138, 141, 143, 134, 148, 176, 106, 124, 124, 158, 141, 123, 167, 162, 165, 168, 134, 167, 153, 150, 0, 162, 162, 165, 177, 84, 106, 176, 80, 119, 140, 189, 160, 120, 136, 118, 108, 94, 186, 170, 167, 157, 0, 146, 144, 176, 126, 145, 183, 167, 159, 124, 109, 132, 113, 157, 99, 91, 76, 87, 176, 176, 135, 157, 96, 83, 90, 91, 122, 89, 148, 113, 93, 46, 149, 117, 176, 144, 145, 137, 105, 183, 105, 106, 97, 92, 151, 68, 195, 166, 145, 139, 229, 153, 151, 63, 152, 175, 183, 211, 105, 91, 85, 128, 148, 145, 155, 172, 129, 130, 179, 166, 128, 119, 69, 166, 193, 192, 204, 116, 66, 97, 147, 146, 100, 136, 203, 167, 187, 162, 119, 145, 150, 160, 168, 200, 197, 120, 148, 106, 0, 114, 203, 116, 101, 163, 106, 163, 168, 162, 167, 152, 136, 133, 109, 142, 176, 212, 65, 143, 155, 119, 101, 162, 137, 229, 122, 151, 211, 87, 214, 196, 160, 136, 104, 160, 148, 137, 162, 76, 177, 167, 128, 123, 124, 209, 132, 59, 142, 139, 147, 143, 139, 121, 150, 181, 149, 185, 140, 156, 112, 145, 118, 115, 175, 189, 171, 164, 165, 155, 141, 145, 151, 157, 218, 126, 74, 63, 167, 164, 116, 173, 106, 93, 122, 120, 82, 57, 156, 219, 138, 154, 138, 142, 106, 114, 161, 221, 224, 207, 135, 150, 112, 115, 85, 182, 179, 170, 176, 176, 166, 137, 146, 221, 141, 156, 110, 168, 168, 153, 159, 167, 155, 125, 136, 86, 197, 95, 197, 181, 132, 151, 149, 154, 155, 138, 154, 172, 197, 170, 186, 148, 154, 95, 108, 202, 233, 149, 156, 146, 150, 216, 189, 121, 160, 96, 161, 115, 131, 124, 178, 186, 92, 152, 173, 179, 185, 139, 159, 119, 197, 132, 150, 149, 165, 188, 184, 133, 173, 159, 106, 171, 145, 174, 137, 136, 131, 118, 183, 162, 135, 152, 160, 124, 134, 202, 198, 187, 130, 135, 159, 100, 107, 229, 157, 131, 167, 83, 193, 165, 132, 143, 89, 150, 137, 209, 163, 175, 176, 193, 134, 148, 233, 131, 134, 221, 195, 195, 87, 171, 168, 202, 174, 160, 225, 205, 181, 194, 193, 184, 170, 143, 90, 116, 198, 147, 222, 188, 150, 144, 205, 176, 111, 159, 123, 198, 178, 117, 186, 174, 166, 176, 89, 112, 184, 165, 175, 156, 143, 113, 143, 194, 179, 178, 144, 143, 205, 207, 183, 179, 172, 183, 147, 210, 189, 186, 167, 156, 174, 140, 107, 210, 131, 192, 166, 128, 149, 143, 85, 129, 161, 190, 201, 165, 189, 142, 128, 108, 193, 132, 86, 193, 129, 203, 122, 200, 162, 153, 176, 154, 89, 125, 142, 100, 186, 150, 197, 132, 132, 159, 152, 166, 200, 193, 205, 119, 143, 189, 165, 157, 142, 116, 192, 172, 163, 161, 183, 184, 200, 185, 151, 181, 163, 186, 153, 151, 157, 89, 189, 165, 163, 196, 149, 132, 145, 187, 178, 123, 169, 125, 146, 137, 127, 168, 92, 100, 182, 171, 184, 171, 147, 185, 90, 176, 186, 63, 154, 170, 190, 181, 177, 145, 159, 202, 146, 155, 158, 65, 27, 124, 162, 169, 180, 140, 124, 119, 189, 172, 148, 127, 143, 179, 55, 207, 133, 182, 121, 190, 185, 204, 155, 181, 185, 200, 140, 53, 118, 195, 195, 130, 191, 73, 135, 178, 107, 221, 185, 74, 142, 141, 149, 166, 105, 162, 108, 135, 104, 117, 85, 187, 110, 124, 132, 98, 76, 212, 145, 183, 193, 173, 124, 89, 104, 133, 206, 174, 135, 131, 148, 234, 201, 175, 149, 148, 126, 140, 39, 177, 210, 203, 201, 122, 134, 179, 155, 141, 120, 187, 43, 138, 155, 186, 154, 128, 72, 206, 173, 156, 148, 110, 128, 147, 151, 62, 118, 84, 56, 217, 142, 119, 141, 154, 113, 182, 197, 108, 162, 124, 159, 160, 156, 136, 144, 151, 171, 176, 107, 224, 130, 128, 135, 153, 222, 141, 178, 181, 178, 78, 64, 142, 149, 157, 145, 188, 56, 43, 96, 136, 144, 105, 106, 161, 83, 154, 157, 193, 178, 96, 106, 134, 126, 141, 193, 64, 184, 180, 172, 54, 79, 0, 119, 119, 127, 184, 187, 96, 135, 81, 156, 47, 73, 118, 186, 157, 176, 40, 149, 142, 135, 184, 79, 112, 118, 150, 176, 0, 140, 67, 131, 162, 174, 178, 192, 144, 30, 125, 133, 195, 88, 160, 21, 125, 156, 81, 150, 199, 120, 139, 39, 142, 141, 195, 187, 91, 127, 125, 85, 209, 136, 185, 84, 123, 126, 118, 117, 128, 184, 173, 113, 141, 14, 118, 124, 141, 121, 193, 103, 80, 178, 89, 106, 124, 93, 183, 120, 108, 123, 99, 103, 124, 144, 136, 126, 60, 220, 160, 112, 188, 133, 145, 93, 146, 187, 126, 101, 171, 81, 196, 51, 77, 85, 114, 130, 101, 125, 97, 75, 114, 162, 177, 101, 120, 134, 70, 211, 117, 137, 155, 184, 175, 172, 180, 138, 39, 42, 146, 103, 132, 201, 195, 139, 190, 135, 114, 88, 196, 152, 143, 149, 173, 179, 142, 123, 111, 112, 169, 193, 166, 105, 99, 174, 177, 173, 118, 100, 99, 142, 222, 149, 164, 168, 181, 124, 125, 112, 108, 75, 67, 11, 133, 168, 138, 166, 119, 113, 116, 132, 166, 146, 148, 181, 140, 26, 103, 109, 132, 160, 165, 203, 164, 161, 112, 128, 163, 130, 107, 153, 156, 159, 110, 152, 163, 167, 167, 202, 164, 163, 133, 186, 128, 130, 148, 166, 111, 164, 165, 173, 185, 141, 157, 156, 163, 115, 159, 140, 103, 168, 137, 155, 167, 162, 183, 132, 143, 139, 149, 152, 158, 149, 172, 129, 137, 159, 184, 180, 167, 136, 143, 164, 100, 142, 141, 179, 157, 160, 163, 119, 150, 161, 177, 144, 154, 158, 81, 154, 121, 168, 146, 124, 163, 143, 144, 151, 201, 187, 185, 161, 147, 138, 153, 138, 173, 132, 122, 145, 153, 152, 147, 140, 95, 81, 132, 158, 79, 149, 145, 157, 175, 151, 137, 130, 152, 145, 109, 100, 177, 140, 142, 148, 97, 86, 137, 143, 157, 183, 176, 143, 151, 147, 148, 157, 130, 163, 159, 153, 142, 148, 169, 145, 133, 133, 145, 179, 136, 117, 185, 154, 126, 153, 135, 129, 85, 122, 134, 142, 147, 137, 139, 129, 117, 138, 153, 157, 188, 182, 145, 142, 115, 152, 145, 155, 187, 170, 176, 150, 163, 136, 145, 149, 135, 148, 187, 169, 146, 163, 133, 113, 134, 157, 163, 123, 139, 152, 159, 114, 149, 137, 147, 172, 134, 130, 160, 142, 102, 175, 141, 189, 178, 135, 126, 142, 155, 145, 151, 131, 97, 111, 115, 62, 139, 175, 115, 103, 151, 147, 152, 154, 147, 154, 151, 113, 139, 198, 144, 181, 175, 112, 135, 133, 95, 120, 142, 142, 161, 176, 170, 130, 131, 139, 189, 152, 147, 134, 179, 172, 179, 185, 113, 125, 160, 133, 142, 168, 177, 130, 116, 165, 178, 159, 154, 168, 135, 170, 171, 193, 132, 175, 131, 199, 159, 164, 168, 177, 168, 182, 179, 170, 207, 148, 172, 194, 178, 144, 176, 128, 178, 167, 176, 170, 201, 200, 180, 178, 129, 99, 171, 180, 171, 143, 157, 163, 163, 171, 166, 189, 184, 168, 191, 176, 168, 172, 181, 148, 179, 167, 199, 137, 130, 130, 162, 156, 178, 183, 161, 180, 163, 132, 174, 148, 218, 203, 161, 197, 197, 164, 160, 195, 162, 162, 145, 87, 161, 162, 162, 155, 224, 199, 143, 167, 169, 146, 175, 152, 174, 200, 113, 156, 145, 145, 143, 165, 220, 202, 169, 170, 138, 173, 173, 204, 201, 177, 172, 142, 170, 156, 160, 169, 207, 210, 172, 210, 203, 183, 132, 138, 152, 165, 194, 181, 161, 202, 165, 157, 161, 196, 148, 142, 141, 158, 97, 195, 139, 103, 168, 175, 174, 195, 193, 136, 184, 75, 203, 163, 138, 168, 135, 114, 109, 105, 208, 143, 113, 123, 102, 114, 94, 163, 157, 154, 163, 158, 158, 93, 196, 180, 88, 73, 173, 207, 104, 152, 210, 125, 207, 135, 92, 36, 102, 184, 124, 123, 107, 117, 178, 125, 197, 147, 99, 73, 164, 170, 151, 154, 146, 143, 207, 178, 121, 97, 176, 140, 135, 200, 192, 183, 131, 95, 137, 106, 129, 87, 56, 134, 102, 198, 186, 84, 84, 198, 209, 120, 110, 92, 92, 105, 185, 168, 136, 182, 94, 110, 87, 205, 116, 119, 107, 100, 135, 196, 126, 215, 100, 130, 170, 121, 129, 209, 93, 136, 135, 125, 144, 186, 132, 47, 212, 143, 130, 172, 184, 230, 121, 123, 143, 153, 151, 85, 103, 40, 164, 130, 128, 168, 33, 93, 77, 180, 89, 193, 106, 120, 188, 141, 214, 105, 136, 160, 150, 193, 65, 168, 127, 135, 129, 110, 143, 109, 187, 93, 154, 139, 117, 124, 123, 169, 139, 132, 83, 167, 121, 167, 165, 146, 132, 127, 124, 193, 85, 191, 109, 151, 30, 158, 167, 209, 131, 90, 166, 134, 177, 112, 141, 97, 105, 115, 84, 170, 98, 117, 145, 164, 173, 148, 153, 121, 95, 58, 163, 99, 174, 137, 137, 108, 115, 117, 183, 125, 77, 147, 225, 77, 107, 174, 47, 152, 139, 143, 172, 129, 44, 107, 28, 127, 177, 118, 111, 200, 98, 127, 139, 95, 41, 138, 175, 142, 169, 179, 85, 197, 178, 84, 67, 129, 185, 128, 151, 58, 172, 185, 99, 59, 164, 175, 145, 127, 125, 234, 87, 169, 105, 114, 76, 120, 138, 68, 56, 135, 130, 110, 118, 136, 169, 116, 123, 68, 152, 129, 157, 133, 101, 89, 194, 131, 99, 176, 157, 157, 27, 15, 118, 136, 186, 121, 71, 215, 111, 96, 51, 79, 123, 142, 104, 89, 218, 62, 66, 148, 153, 139, 153, 97, 141, 130, 99, 88, 71, 195, 109, 39, 235, 58, 142, 166, 58, 92, 100, 107, 145, 167, 59, 139, 72, 154, 139, 136, 108, 205, 96, 161, 116, 108, 107, 103, 107, 127, 197, 25, 90, 137, 150, 107, 109, 149, 155, 180, 111, 89, 138, 120, 125, 139, 117, 155, 106, 106, 107, 208, 142, 74, 159, 143, 140, 163, 186, 65, 122, 127, 122, 169, 198, 74, 90, 73, 99, 163, 158, 137, 132, 190, 103, 129, 193, 192, 104, 46, 54, 162, 92, 161, 127, 115, 121, 185, 212, 109, 120, 148, 145, 165, 136, 175, 73, 139, 114, 59, 135, 90, 92, 151, 121, 140, 182, 170, 160, 103, 124, 161, 144, 104, 201, 192, 109, 101, 126, 181, 198, 202, 204, 149, 128, 165, 98, 105, 207, 89, 80, 129, 154, 99, 103, 165, 126, 118, 107, 115, 117, 132, 80, 132, 172, 148, 101, 97, 125, 100, 97, 190, 79, 102, 113, 205, 191, 128, 104, 96, 175, 94, 97, 184, 130, 114, 104, 87, 89, 120, 131, 166, 136, 93, 102, 119, 175, 220, 177, 97, 176, 178, 126, 164, 104, 122, 96, 164, 148, 121, 154, 146, 75, 132, 223, 115, 188, 192, 143, 93, 91, 58, 111, 142, 118, 178, 214, 164, 113, 145, 184, 186, 79, 113, 147, 176, 70, 108, 133, 111, 188, 91, 104, 127, 111, 165, 173, 219, 80, 203, 217, 104, 150, 219, 135, 145, 192, 125, 159, 173, 173, 120, 120, 150, 180, 168, 148, 125, 162, 68, 64, 88, 171, 134, 151, 145, 206, 151, 118, 70, 86, 142, 135, 136, 128, 132, 146, 141, 143, 167, 131, 90, 131, 168, 152, 131, 107, 140, 144, 214, 217, 138, 137, 154, 154, 122, 177, 124, 125, 157, 73, 135, 149, 118, 149, 111, 216, 129, 71, 143, 113, 155, 205, 198, 82, 156, 139, 111, 142, 133, 129, 183, 125, 225, 148, 130, 148, 143, 136, 152, 75, 231, 113, 107, 140, 168, 93, 168, 134, 153, 153, 124, 119, 146, 103, 116, 196, 159, 108, 126, 123, 27, 108, 87, 115, 152, 93, 114, 93, 218, 193, 153, 136, 147, 148, 84, 160, 151, 120, 189, 160, 154, 110, 121, 159, 134, 99, 109, 175, 96, 155, 170, 173, 146, 123, 104, 144, 140, 114, 141, 154, 180, 130, 108, 162, 159, 168, 161, 189, 118, 112, 146, 174, 133, 111, 166, 100, 111, 153, 156, 145, 139, 157, 155, 191, 185, 137, 135, 148, 149, 123, 104, 74, 160, 168, 157, 155, 178, 75, 183, 111, 163, 166, 173, 158, 146, 100, 146, 142, 138, 112, 94, 137, 195, 165, 140, 147, 135, 89, 99, 210, 141, 155, 56, 201, 86, 175, 197, 165, 182, 207, 121, 211, 138, 150, 80, 205, 199, 151, 140, 206, 208, 182, 145, 88, 152, 207, 191, 194, 148, 151, 165, 202, 77, 94, 163, 127, 76, 142, 161, 86, 81, 162, 104, 161, 200, 164, 159, 139, 183, 141, 176, 96, 189, 181, 157, 176, 186, 183, 180, 163, 177, 181, 152, 180, 153, 170, 103, 158, 177, 101, 129, 149, 162, 164, 128, 165, 149, 163, 171, 163, 106, 189, 199, 78, 98, 147, 156, 117, 156, 181, 167, 110, 121, 119, 163, 100, 126, 132, 179, 183, 127, 152, 108, 218, 167, 143, 117, 140, 122, 137, 190, 198, 176, 193, 117, 93, 137, 214, 114, 106, 212, 105, 193, 103, 175, 95, 194, 97, 159, 115, 194, 176, 88, 63, 89, 88, 137, 103, 98, 140, 219, 177, 202, 136, 89, 167, 173, 176, 162, 152, 84, 148, 149, 155, 94, 135, 172, 137, 85, 151, 148, 125, 171, 140, 191, 157, 170, 162, 165, 172, 155, 174, 139, 132, 149, 162, 161, 158, 133, 145, 140, 167, 139, 158, 111, 161, 193, 174, 135, 153, 196, 138, 139, 150, 193, 183, 153, 150, 159, 146, 138, 149, 137, 166, 178, 144, 174, 149, 140, 91, 118, 203, 143, 151, 104, 113, 156, 125, 158, 186, 101, 116, 162, 130, 165, 115, 120, 177, 191, 120, 118, 122, 128, 146, 152, 84, 127, 144, 113, 160, 159, 90, 126, 136, 158, 125, 120, 151, 94, 129, 116, 126, 125, 120, 136, 164, 88, 247, 121, 112, 114, 155, 68, 96, 94, 110, 93, 155, 169, 169, 178, 170, 56, 136, 160, 31, 110, 214, 120, 213, 148, 182, 128, 85, 89, 138, 110, 181, 74, 110, 76, 182, 161, 125, 155, 146, 223, 118, 124, 79, 123, 160, 117, 120, 91, 218, 67, 94, 96, 114, 221, 124, 86, 106, 172, 121, 120, 214, 192, 127, 50, 201, 51, 214, 86, 125, 155, 13, 214, 181, 106, 191, 134, 36, 54, 178, 109, 85, 171, 185, 190, 121, 0, 214, 185, 189, 206, 104, 186, 223, 91, 128, 89, 159, 98, 181, 173, 102, 115, 105, 180, 98, 151, 129, 99, 218, 132, 61, 66, 164, 210, 207, 133, 104, 156, 145, 132, 111, 174, 61, 92, 133, 98, 85, 180, 182, 150, 186, 164, 211, 198, 154, 85, 143, 115, 94, 45, 214, 201, 118, 152, 161, 213, 64, 101, 201, 29, 206, 30, 59, 129, 118, 184, 195, 213, 36, 173, 204, 213, 58, 100, 151, 172, 51, 82, 65, 139, 55, 183, 171, 103, 19, 201, 47, 41, 162, 143, 133, 169, 41, 97, 201, 139, 90, 80, 153, 203, 178, 195, 110, 176, 195, 121, 94, 202, 162, 92, 56, 88, 100, 72, 189, 148, 125, 191, 77, 90, 99, 132, 117, 183, 143, 136, 53, 60, 208, 107, 144, 137, 150, 110, 85, 165, 197, 112, 93, 163, 89, 84, 197, 181, 121, 120, 77, 214, 160, 119, 129, 135, 72, 151, 76, 190, 162, 132, 81, 77, 124, 111, 128, 101, 128, 70, 115, 161, 93, 217, 86, 152, 127, 199, 172, 149, 90, 106, 198, 192, 42, 184, 113, 145, 67, 152, 158, 175, 93, 95, 147, 93, 143, 124, 69, 89, 51, 124, 114, 173, 107, 129, 152, 223, 126, 142, 145, 146, 149, 178, 101, 147, 163, 119, 100, 140, 103, 99, 115, 104, 119, 134, 185, 61, 50, 104, 164, 156, 129, 154, 84, 98, 168, 142, 130, 167, 77, 104, 117, 137, 133, 96, 127, 65, 134, 121, 116, 149, 133, 95, 67, 110, 114, 209, 135, 120, 180, 121, 126, 109, 127, 127, 149, 233, 182, 130, 121, 129, 213, 132, 121, 38, 201, 184, 134, 119, 63, 182, 127, 134, 127, 121, 87, 180, 200, 139, 131, 114, 134, 134, 196, 140, 130, 147, 112, 142, 96, 197, 64, 86, 123, 140, 146, 198, 171, 132, 116, 198, 205, 169, 202, 149, 124, 145, 160, 196, 162, 171, 139, 182, 167, 102, 64, 135, 171, 126, 157, 147, 131, 114, 168, 166, 142, 103, 119, 161, 168, 96, 130, 100, 203, 172, 193, 179, 143, 128, 166, 137, 190, 120, 170, 164, 146, 111, 118, 123, 113, 55, 180, 200, 166, 126, 157, 189, 109, 204, 153, 174, 43, 150, 151, 101, 146, 182, 174, 149, 145, 84, 152, 178, 121, 100, 171, 156, 178, 109, 188, 167, 160, 110, 202, 132, 120, 105, 204, 152, 101, 132, 116, 105, 102, 204, 179, 135, 119, 154, 52, 157, 141, 142, 165, 124, 46, 53, 113, 114, 49, 87, 145, 165, 24, 80, 159, 68, 68, 111, 90, 119, 135, 189, 121, 42, 73, 115, 174, 69, 105, 89, 61, 103, 72, 128, 209, 184, 74, 103, 133, 128, 129, 122, 105, 86, 95, 106, 121, 182, 86, 134, 99, 115, 73, 65, 135, 96, 76, 127, 198, 84, 75, 118, 144, 95, 89, 95, 116, 49, 210, 115, 85, 106, 120, 86, 124, 112, 54, 165, 90, 187, 131, 88, 203, 127, 115, 70, 81, 241, 114, 87, 66, 96, 86, 245, 107, 97, 75, 230, 78, 87, 13, 121, 124, 51, 91, 122, 84, 117, 193, 208, 87, 130, 129, 148, 173, 133, 168, 164, 128, 150, 90, 110, 156, 128, 119, 143, 143, 145, 144, 137, 136, 139, 134, 99, 132, 28, 142, 107, 166, 121, 129, 105, 156, 146, 141, 182, 148, 116, 151, 136, 107, 123, 143, 62, 164, 153, 128, 216, 126, 113, 149, 152, 171, 182, 137, 146, 134, 142, 181, 105, 123, 156, 136, 124, 184, 105, 83, 173, 144, 158, 104, 150, 163, 164, 84, 182, 79, 63, 84, 151, 128, 143, 53, 104, 178, 106, 111, 152, 178, 126, 163, 167, 151, 124, 107, 136, 128, 136, 170, 141, 158, 153, 179, 124, 156, 94, 121, 143, 157, 135, 143, 140, 152, 111, 124, 115, 182, 161, 115, 104, 132, 153, 136, 124, 176, 154, 125, 129, 127, 150, 128, 158, 172, 115, 62, 81, 131, 134, 122, 91, 109, 63, 76, 134, 112, 90, 165, 132, 154, 128, 118, 139, 151, 99, 149, 181, 112, 108, 115, 139, 132, 144, 141, 120, 167, 144, 97, 215, 121, 112, 166, 143, 129, 138, 150, 115, 116, 132, 113, 159, 117, 133, 104, 139, 165, 105, 141, 118, 160, 121, 76, 93, 199, 136, 143, 128, 110, 86, 82, 131, 135, 155, 125, 149, 104, 113, 129, 99, 103, 156, 92, 179, 157, 76, 131, 101, 185, 74, 120, 153, 121, 158, 149, 137, 124, 96, 118, 123, 91, 149, 134, 120, 173, 199, 125, 199, 147, 114, 191, 195, 206, 179, 145, 141, 213, 79, 134, 188, 160, 131, 215, 166, 119, 82, 112, 143, 124, 191, 166, 162, 140, 143, 119, 149, 146, 169, 65, 149, 157, 136, 131, 197, 168, 168, 142, 163, 152, 116, 141, 147, 209, 108, 170, 156, 127, 199, 126, 148, 155, 134, 136, 167, 130, 164, 206, 106, 153, 199, 151, 136, 131, 206, 127, 186, 157, 159, 172, 78, 193, 122, 124, 113, 180, 106, 145, 176, 133, 154, 181, 160, 144, 162, 166, 165, 174, 117, 136, 132, 75, 163, 130, 131, 113, 175, 130, 138, 162, 119, 199, 151, 130, 109, 108, 179, 153, 154, 151, 131, 141, 156, 128, 116, 116, 113, 122, 134, 197, 188, 145, 170, 132, 123, 149, 124, 151, 125, 165, 135, 167, 135, 163, 153, 101, 144, 131, 122, 116, 129, 85, 159, 135, 122, 131, 147, 202, 128, 95, 145, 114, 154, 71, 126, 146, 145, 164, 136, 99, 94, 79, 121, 153, 128, 81, 87, 117, 79, 49, 97, 136, 139, 76, 90, 97, 149, 56, 115, 84, 77, 150, 102, 68, 64, 85, 150, 135, 45, 189, 35, 149, 138, 43, 57, 141, 71, 25, 179, 190, 140, 73, 161, 120, 21, 143, 156, 157, 153, 28, 163, 184, 164, 160, 122, 120, 111, 173, 68, 70, 148, 139, 182, 143, 153, 51, 182, 127, 79, 149, 182, 93, 170, 45, 64, 75, 86, 157, 96, 86, 122, 118, 187, 85, 0, 77, 0, 168, 51, 74, 86, 123, 142, 119, 14, 138, 123, 52, 82, 80, 151, 35, 96, 133, 99, 105, 133, 131, 124, 32, 97, 57, 40, 158, 194, 142, 190, 60, 165, 132, 142, 184, 30, 107, 159, 192, 96, 159, 50, 139, 130, 179, 29, 36, 142, 22, 184, 170, 29, 134, 146, 93, 26, 172, 153, 130, 37, 21, 187, 153, 165, 125, 157, 136, 125, 125, 30, 132, 35, 162, 136, 150, 35, 43, 35, 145, 152, 38, 37, 149, 179, 49, 173, 92, 53, 122, 51, 116, 152, 56, 147, 115, 107, 139, 88, 150, 131, 73, 140, 53, 163, 153, 59, 177, 107, 143, 74, 150, 62, 55, 174, 127, 51, 168, 238, 118, 150, 143, 160, 194, 46, 166, 89, 67, 72, 194, 91, 81, 122, 226, 207, 115, 91, 161, 224, 121, 183, 208, 36, 218, 0, 84, 172, 144, 17, 45, 203, 159, 72, 38, 146, 57, 210, 221, 37, 190, 62, 207, 34, 188, 49, 202, 15, 104, 168, 154, 81, 130, 202, 117, 211, 82, 121, 105, 151, 120, 78, 246, 187, 79, 223, 192, 25, 90, 119, 124, 104, 60, 142, 142, 193, 179, 78, 225, 212, 176, 149, 70, 58, 224, 146, 160, 111, 175, 174, 70, 65, 214, 127, 146, 103, 43, 141, 103, 58, 180, 141, 136, 103, 108, 215, 154, 126, 213, 194, 146, 123, 161, 39, 40, 152, 138, 192, 125, 161, 175, 103, 180, 138, 135, 120, 94, 44, 158, 131, 137, 108, 136, 138, 50, 199, 116, 121, 176, 183, 161, 134, 97, 169, 110, 151, 96, 247, 186, 167, 191, 85, 204, 13, 155, 104, 92, 77, 129, 77, 101, 173, 133, 91, 62, 54, 77, 197, 70, 132, 55, 110, 124, 190, 75, 224, 170, 184, 202, 164, 82, 98, 79, 189, 204, 152, 52, 90, 111, 193, 68, 113, 104, 84, 78, 71, 202, 181, 165, 90, 237, 209, 157, 75, 79, 114, 80, 220, 237, 0, 193, 42, 100, 95, 81, 211, 202, 180, 212, 207, 69, 71, 197, 196, 165, 164, 78, 206, 194, 184, 144, 80, 224, 178, 125, 126, 102, 98, 118, 198, 102, 136, 218, 141, 20, 130, 128, 99, 56, 212, 188, 71, 88, 154, 104, 50, 136, 125, 106, 176, 138, 152, 118, 182, 129, 136, 102, 185, 180, 105, 143, 173, 78, 197, 100, 103, 174, 159, 128, 85, 48, 160, 118, 151, 163, 68, 80, 188, 143, 108, 90, 99, 89, 220, 196, 124, 151, 122, 101, 182, 122, 142, 160, 0, 196, 180, 65, 102, 115, 118, 92, 114, 116, 179, 110, 136, 236, 109, 63, 168, 102, 133, 136, 173, 117, 176, 182, 99, 133, 184, 193, 205, 164, 200, 79, 176, 208, 32, 240, 210, 161, 121, 117, 209, 217, 108, 125, 111, 147, 230, 168, 152, 135, 220, 206, 158, 209, 211, 79, 110, 135, 132, 179, 113, 149, 180, 90, 124, 137, 226, 193, 162, 71, 203, 203, 100, 16, 64, 198, 212, 201, 130, 47, 33, 186, 189, 86, 169, 108, 149, 211, 167, 54, 192, 143, 107, 129, 236, 0, 10, 185, 164, 172, 103, 119, 221, 73, 8, 124, 196, 41, 34, 3, 105, 0, 192, 245, 116, 20, 20, 18, 20, 13, 84, 20, 231, 224, 35, 10, 96, 202, 166, 142, 153, 157, 178, 123, 198, 24, 26, 49, 26, 126, 144, 198, 0, 174, 101, 202, 148, 173, 129, 128, 7, 0, 47, 53, 43, 64, 238, 125, 145, 150, 185, 88, 33, 69, 204, 39, 61, 18, 21, 50, 202, 171, 50, 142, 132, 136, 147, 175, 27, 156, 54, 4, 64, 59, 76, 165, 101, 107, 117, 186, 66, 125, 150, 129, 142, 178, 142, 28, 60, 133, 67, 57, 20, 154, 50, 67, 44, 113, 109, 193, 126, 52, 151, 109, 141, 243, 109, 149, 230, 31, 221, 25, 64, 74, 68, 76, 89, 155, 154, 76, 146, 192, 173, 32, 33, 224, 78, 136, 172, 118, 163, 121, 200, 136, 166, 222, 40, 62, 59, 63, 147, 106, 97, 83, 147, 214, 118, 85, 122, 151, 51, 35, 60, 95, 91, 124, 99, 123, 129, 189, 160, 179, 93, 212, 197, 119, 184, 29, 57, 81, 211, 171, 28, 76, 156, 168, 105, 200, 157, 137, 53, 78, 51, 154, 56, 200, 161, 104, 163, 133, 90, 70, 65, 109, 184, 134, 119, 126, 80, 162, 145, 69, 108, 174, 118, 151, 182, 199, 206, 156, 106, 87, 105, 105, 39, 146, 179, 108, 150, 220, 158, 164, 174, 208, 77, 118, 171, 154, 190, 200, 192, 107, 189, 199, 65, 112, 150, 170, 221, 157, 165, 174, 100, 215, 104, 69, 50, 57, 70, 200, 195, 173, 90, 108, 225, 136, 148, 100, 102, 226, 80, 156, 191, 198, 163, 157, 139, 115, 129, 159, 107, 90, 73, 61, 59, 152, 122, 96, 178, 103, 102, 137, 182, 103, 120, 167, 122, 174, 143, 148, 97, 55, 66, 56, 66, 180, 101, 97, 70, 75, 108, 107, 141, 142, 178, 109, 221, 105, 92, 91, 140, 167, 102, 191, 165, 176, 174, 95, 70, 69, 138, 175, 146, 124, 91, 131, 152, 114, 133, 185, 154, 99, 46, 147, 117, 197, 141, 219, 92, 88, 109, 107, 148, 88, 133, 97, 81, 94, 94, 42, 164, 116, 154, 101, 103, 140, 157, 81, 187, 151, 168, 112, 151, 75, 171, 192, 134, 147, 75, 130, 174, 141, 173, 132, 174, 218, 143, 84, 74, 67, 199, 166, 192, 147, 131, 203, 221, 128, 166, 151, 136, 148, 102, 110, 61, 77, 68, 128, 129, 175, 112, 208, 204, 79, 93, 133, 125, 179, 189, 119, 144, 96, 167, 84, 74, 67, 134, 118, 166, 136, 162, 89, 140, 157, 151, 181, 184, 115, 162, 57, 164, 108, 71, 47, 145, 127, 102, 155, 118, 158, 105, 119, 153, 112, 171, 165, 98, 63, 164, 70, 83, 159, 169, 198, 197, 132, 169, 218, 159, 74, 151, 105, 181, 163, 109, 184, 147, 156, 132, 176, 164, 172, 79, 73, 47, 97, 118, 106, 133, 82, 197, 193, 99, 131, 119, 84, 247, 86, 70, 89, 69, 55, 103, 142, 167, 116, 90, 160, 188, 135, 169, 76, 136, 111, 89, 73, 64, 59, 68, 72, 98, 68, 39, 147, 145, 223, 146, 62, 61, 62, 152, 168, 145, 142, 225, 156, 204, 83, 137, 197, 137, 201, 142, 187, 156, 124, 147, 80, 65, 62, 59, 108, 145, 218, 183, 146, 226, 137, 46, 118, 133, 145, 116, 132, 87, 62, 109, 213, 217, 75, 64, 169, 85, 158, 127, 80, 64, 92, 164, 219, 178, 192, 121, 169, 86, 119, 153, 199, 215, 58, 214, 33, 140, 205, 93, 52, 145, 115, 216, 235, 117, 123, 180, 141, 165, 120, 214, 157, 86, 69, 46, 93, 215, 170, 163, 156, 198, 130, 84, 134, 141, 203, 58, 36, 39, 80, 113, 89, 130, 74, 98, 63, 128, 104, 185, 229, 115, 176, 122, 166, 57, 91, 121, 142, 146, 211, 114, 120, 67, 40, 163, 101, 136, 51, 37, 62, 154, 96, 136, 100, 197, 187, 122, 28, 131, 135, 117, 173, 99, 165, 179, 178, 181, 55, 175, 98, 131, 111, 198, 212, 164, 97, 176, 235, 52, 72, 87, 94, 126, 209, 129, 180, 80, 194, 173, 64, 53, 84, 143, 153, 109, 186, 117, 91, 162, 116, 173, 32, 16, 100, 218, 66, 96, 163, 105, 93, 104, 81, 207, 202, 214, 63, 28, 74, 112, 115, 225, 169, 197, 183, 180, 59, 56, 159, 122, 121, 64, 147, 76, 25, 49, 98, 120, 175, 116, 185, 171, 99, 97, 67, 212, 180, 190, 134, 138, 132, 67, 178, 118, 189, 154, 162, 65, 69, 99, 87, 161, 84, 118, 188, 146, 179, 127, 199, 154, 167, 117, 94, 78, 234, 122, 119, 63, 85, 114, 191, 153, 103, 244, 170, 134, 215, 137, 145, 186, 178, 54, 105, 160, 183, 96, 167, 203, 189, 46, 143, 113, 112, 89, 159, 213, 157, 204, 157, 167, 179, 202, 113, 155, 194, 169, 239, 158, 104, 170, 222, 149, 161, 149, 110, 117, 160, 0, 156, 179, 148, 67, 127, 159, 102, 154, 124, 152, 158, 10, 101, 228, 122, 102, 177, 181, 227, 89, 70, 85, 93, 192, 138, 157, 159, 119, 199, 71, 168, 197, 205, 69, 174, 158, 105, 233, 109, 42, 123, 180, 66, 211, 146, 118, 126, 71, 116, 207, 192, 134, 85, 165, 76, 219, 179, 104, 191, 94, 133, 138, 199, 146, 215, 82, 143, 171, 221, 113, 151, 153, 97, 145, 169, 182, 118, 161, 97, 105, 126, 77, 170, 190, 150, 197, 174, 135, 161, 199, 192, 176, 27, 216, 116, 182, 195, 173, 45, 116, 165, 79, 220, 144, 143, 113, 166, 120, 176, 198, 56, 113, 133, 138, 185, 197, 184, 206, 121, 127, 104, 156, 232, 179, 121, 98, 81, 186, 78, 130, 116, 183, 208, 146, 155, 129, 172, 218, 209, 121, 129, 201, 177, 148, 136, 159, 90, 122, 154, 152, 179, 93, 90, 161, 92, 154, 108, 145, 141, 96, 185, 124, 207, 92, 93, 113, 67, 152, 175, 74, 70, 237, 166, 213, 194, 158, 186, 176, 108, 227, 92, 57, 198, 161, 162, 159, 117, 122, 157, 105, 167, 144, 146, 168, 191, 95, 102, 147, 66, 123, 153, 214, 147, 133, 185, 206, 204, 90, 108, 131, 140, 213, 102, 91, 91, 91, 98, 120, 120, 133, 207, 123, 117, 138, 158, 109, 125, 127, 204, 180, 109, 122, 66, 120, 150, 150, 111, 100, 114, 180, 144, 122, 213, 140, 142, 163, 131, 171, 133, 118, 157, 185, 133, 83, 95, 212, 132, 154, 194, 128, 158, 138, 192, 226, 65, 193, 137, 224, 78, 106, 183, 205, 174, 148, 223, 140, 153, 173, 102, 167, 133, 138, 90, 64, 85, 106, 162, 158, 63, 170, 81, 144, 145, 137, 130, 207, 178, 152, 149, 133, 177, 138, 134, 133, 132, 124, 95, 134, 134, 130, 220, 138, 163, 219, 110, 219, 209, 199, 134, 144, 131, 84, 195, 112, 133, 59, 125, 70, 136, 225, 73, 192, 177, 130, 227, 156, 150, 134, 114, 122, 103, 237, 56, 119, 143, 115, 213, 186, 154, 100, 159, 160, 158, 196, 90, 107, 113, 194, 0, 142, 141, 138, 162, 171, 147, 149, 190, 153, 142, 163, 81, 153, 156, 141, 121, 161, 200, 181, 161, 103, 205, 173, 173, 188, 99, 120, 172, 145, 144, 90, 111, 137, 156, 107, 129, 190, 132, 151, 140, 189, 83, 191, 118, 214, 188, 114, 93, 119, 137, 112, 151, 161, 131, 205, 196, 136, 102, 130, 166, 89, 90, 126, 115, 189, 104, 115, 119, 125, 208, 138, 135, 178, 116, 206, 192, 128, 132, 132, 187, 172, 143, 101, 122, 133, 149, 133, 179, 110, 154, 153, 133, 128, 122, 102, 215, 160, 137, 120, 215, 39, 138, 176, 176, 143, 140, 146, 149, 118, 130, 155, 138, 110, 129, 148, 128, 140, 32, 158, 125, 144, 130, 85, 131, 170, 119, 152, 193, 168, 134, 108, 127, 163, 99, 135, 136, 106, 121, 110, 92, 138, 116, 196, 98, 134, 124, 167, 176, 162, 138, 155, 159, 154, 84, 128, 156, 178, 75, 79, 96, 94, 167, 161, 118, 77, 143, 161, 140, 78, 81, 156, 81, 125, 99, 151, 151, 152, 73, 132, 132, 143, 129, 134, 174, 191, 113, 148, 153, 119, 101, 95, 137, 154, 190, 107, 0, 118, 115, 102, 151, 57, 96, 151, 182, 169, 160, 77, 92, 142, 117, 103, 121, 123, 189, 200, 191, 136, 89, 173, 179, 152, 108, 126, 181, 154, 163, 181, 116, 131, 122, 199, 180, 128, 116, 104, 113, 133, 191, 71, 169, 132, 160, 73, 136, 128, 163, 113, 166, 129, 151, 85, 62, 117, 114, 114, 148, 183, 89, 114, 103, 146, 99, 112, 121, 180, 89, 126, 132, 172, 153, 138, 123, 136, 128, 115, 96, 170, 145, 198, 176, 114, 131, 127, 180, 177, 103, 178, 173, 160, 176, 165, 115, 151, 137, 174, 104, 142, 136, 122, 159, 181, 129, 121, 133, 121, 119, 192, 124, 143, 55, 119, 106, 62, 147, 146, 97, 109, 97, 89, 166, 111, 219, 118, 110, 157, 63, 239, 202, 116, 155, 168, 47, 182, 110, 139, 151, 99, 123, 36, 180, 137, 132, 135, 123, 169, 149, 136, 84, 173, 131, 187, 141, 98, 201, 137, 100, 142, 58, 154, 183, 196, 131, 156, 137, 47, 42, 165, 133, 164, 146, 133, 172, 118, 134, 71, 139, 108, 141, 164, 87, 134, 162, 152, 131, 183, 62, 122, 157, 153, 159, 139, 159, 115, 174, 164, 95, 190, 157, 143, 165, 158, 135, 171, 182, 101, 122, 156, 50, 155, 150, 142, 157, 105, 163, 100, 128, 160, 147, 121, 135, 99, 163, 144, 170, 105, 160, 75, 153, 171, 146, 138, 140, 161, 156, 162, 163, 82, 161, 140, 78, 165, 164, 140, 123, 170, 138, 156, 124, 81, 192, 187, 182, 169, 169, 134, 162, 113, 163, 155, 155, 151, 142, 163, 105, 132, 133, 143, 237, 185, 156, 178, 147, 121, 151, 147, 70, 41, 187, 187, 187, 137, 68, 189, 150, 137, 59, 50, 146, 117, 140, 160, 137, 143, 62, 179, 174, 82, 0, 150, 65, 61, 205, 150, 132, 158, 53, 153, 81, 130, 160, 138, 184, 154, 153, 120, 87, 163, 125, 122, 136, 171, 145, 73, 194, 161, 149, 142, 27, 156, 152, 86, 146, 135, 28, 0, 140, 128, 143, 116, 66, 161, 144, 110, 163, 44, 17, 131, 124, 110, 25, 210, 160, 185, 148, 186, 179, 120, 133, 233, 180, 163, 164, 129, 142, 141, 62, 99, 150, 162, 145, 32, 46, 51, 168, 195, 181, 123, 156, 81, 107, 156, 163, 102, 149, 143, 36, 0, 112, 133, 13, 31, 154, 181, 166, 96, 172, 131, 66, 67, 53, 150, 219, 68, 108, 119, 142, 170, 94, 178, 192, 86, 100, 196, 74, 150, 119, 62, 113, 126, 123, 114, 130, 117, 166, 169, 84, 172, 176, 112, 104, 165, 139, 108, 107, 202, 167, 170, 106, 125, 126, 182, 142, 170, 83, 210, 125, 187, 214, 170, 158, 172, 127, 180, 117, 157, 210, 115, 227, 172, 172, 184, 168, 143, 228, 153, 165, 176, 126, 137, 179, 139, 153, 118, 98, 164, 145, 176, 160, 144, 166, 118, 148, 136, 129, 166, 179, 163, 103, 191, 209, 91, 180, 187, 95, 91, 123, 183, 88, 193, 147, 214, 138, 211, 167, 117, 121, 188, 156, 92, 199, 188, 94, 102, 206, 189, 177, 169, 185, 177, 122, 179, 154, 155, 119, 107, 190, 144, 180, 149, 246, 87, 204, 169, 173, 162, 142, 155, 169, 185, 160, 130, 144, 181, 187, 173, 169, 231, 142, 137, 195, 172, 189, 192, 133, 149, 230, 154, 197, 186, 123, 188, 168, 150, 207, 183, 173, 156, 164, 187, 111, 157, 184, 179, 149, 166, 137, 194, 136, 159, 167, 127, 73, 93, 240, 183, 166, 157, 42, 121, 125, 204, 128, 204, 110, 155, 124, 68, 150, 127, 135, 134, 159, 117, 124, 128, 115, 108, 181, 105, 144, 178, 103, 60, 180, 208, 80, 148, 118, 132, 94, 141, 130, 192, 142, 184, 142, 190, 117, 121, 160, 131, 78, 0, 79, 129, 123, 130, 163, 179, 86, 113, 204, 203, 89, 120, 134, 89, 121, 152, 53, 113, 139, 116, 192, 115, 171, 109, 115, 115, 152, 147, 50, 118, 128, 128, 140, 127, 167, 147, 154, 145, 120, 96, 26, 130, 157, 160, 110, 80, 110, 150, 204, 201, 151, 167, 121, 71, 133, 154, 163, 148, 48, 139, 154, 107, 104, 144, 137, 147, 162, 160, 88, 142, 181, 140, 155, 105, 132, 141, 148, 199, 158, 180, 149, 184, 155, 154, 93, 147, 157, 150, 143, 154, 147, 138, 119, 145, 128, 99, 143, 141, 184, 140, 115, 171, 117, 178, 139, 62, 117, 149, 125, 143, 140, 131, 124, 139, 158, 167, 143, 123, 99, 146, 187, 160, 160, 147, 115, 145, 141, 178, 186, 138, 150, 163, 183, 195, 173, 156, 151, 178, 172, 137, 147, 149, 110, 141, 183, 183, 185, 185, 161, 171, 176, 187, 183, 160, 192, 199, 182, 173, 205, 176, 182, 153, 142, 136, 176, 173, 182, 151, 192, 163, 139, 150, 168, 156, 150, 151, 149, 185, 148, 205, 202, 165, 162, 149, 198, 177, 201, 186, 155, 194, 158, 185, 161, 138, 162, 160, 99, 89, 81, 184, 148, 190, 219, 142, 96, 149, 144, 115, 148, 55, 64, 210, 206, 111, 103, 137, 136, 79, 111, 166, 70, 183, 199, 208, 129, 87, 88, 119, 176, 159, 110, 216, 137, 118, 164, 103, 173, 106, 77, 162, 144, 148, 148, 118, 36, 142, 86, 169, 161, 174, 110, 99, 151, 176, 160, 109, 92, 99, 68, 111, 98, 102, 128, 132, 159, 191, 175, 63, 123, 173, 137, 149, 142, 145, 127, 235, 132, 50, 139, 62, 127, 57, 157, 175, 137, 112, 146, 148, 107, 170, 150, 95, 119, 155, 137, 137, 98, 106, 228, 91, 110, 200, 66, 148, 140, 0, 105, 98, 143, 97, 103, 113, 99, 103, 49, 182, 139, 80, 61, 150, 124, 119, 92, 148, 197, 113, 117, 116, 105, 105, 124, 98, 114, 160, 108, 210, 102, 70, 94, 93, 176, 114, 198, 141, 180, 101, 174, 190, 75, 185, 109, 204, 89, 81, 157, 112, 58, 217, 185, 144, 130, 137, 129, 137, 170, 55, 145, 184, 143, 192, 101, 201, 197, 100, 139, 91, 99, 66, 176, 149, 147, 230, 142, 111, 152, 143, 168, 190, 218, 152, 173, 141, 148, 174, 99, 121, 165, 181, 91, 135, 119, 148, 203, 96, 170, 122, 128, 128, 163, 151, 127, 99, 180, 91, 92, 133, 192, 91, 103, 151, 106, 150, 148, 148, 139, 169, 137, 154, 136, 226, 142, 153, 93, 162, 131, 143, 214, 116, 219, 204, 155, 91, 68, 150, 115, 192, 162, 161, 183, 118, 190, 184, 73, 183, 183, 104, 125, 178, 184, 144, 168, 136, 157, 172, 156, 172, 186, 151, 139, 222, 141, 130, 114, 152, 160, 130, 173, 156, 194, 125, 99, 131, 170, 144, 194, 184, 126, 155, 121, 134, 170, 135, 165, 159, 186, 143, 181, 102, 144, 150, 174, 121, 125, 113, 152, 157, 109, 128, 164, 124, 143, 157, 159, 116, 204, 137, 159, 156, 151, 135, 51, 36, 56, 242, 37, 151, 88, 176, 62, 176, 46, 147, 98, 110, 141, 37, 79, 212, 226, 138, 84, 190, 102, 53, 81, 110, 108, 53, 214, 146, 220, 104, 154, 112, 175, 198, 174, 206, 170, 146, 160, 128, 93, 233, 102, 117, 102, 148, 142, 219, 122, 117, 164, 157, 154, 206, 131, 109, 0, 154, 66, 105, 117, 74, 206, 165, 167, 120, 150, 65, 207, 150, 151, 188, 207, 83, 61, 76, 229, 97, 44, 49, 75, 35, 134, 54, 103, 99, 113, 163, 78, 102, 113, 178, 86, 118, 74, 136, 131, 152, 98, 180, 147, 185, 157, 92, 153, 128, 192, 118, 118, 131, 108, 128, 136, 156, 88, 200, 105, 170, 92, 125, 107, 112, 67, 119, 141, 195, 133, 128, 132, 181, 131, 150, 163, 132, 187, 121, 117, 125, 72, 87, 206, 40, 143, 102, 197, 143, 121, 158, 149, 131, 214, 141, 112, 204, 129, 110, 194, 146, 182, 120, 115, 131, 64, 62, 150, 118, 139, 159, 89, 130, 160, 186, 145, 175, 149, 184, 162, 137, 183, 78, 198, 187, 112, 168, 212, 193, 101, 199, 102, 135, 128, 75, 174, 78, 156, 130, 157, 175, 142, 153, 50, 165, 143, 54, 160, 111, 119, 66, 163, 152, 69, 169, 99, 96, 91, 189, 133, 117, 197, 101, 93, 91, 98, 89, 99, 112, 214, 91, 83, 119, 164]}],
                        {"autosize": false, "height": 500, "margin": {"b": 80, "l": 40, "r": 30, "t": 100}, "paper_bgcolor": "rgb(243, 243, 243)", "plot_bgcolor": "rgb(243, 243, 243)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Hillshade 3pm and Noon"}, "width": 700},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('e8e384c4-b5f6-4ba5-ac9b-280d1579eaf8');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>
{% endraw %}

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


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_26_0.png)


Let's take a look our categorical categorical variables soil types and wilderness areas.


```python
soil_types = train.iloc[:,14:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, palette="rocket")
plt.xticks(rotation= 75)
plt.ylabel('Total')
plt.title('Count of Soil Types With Value 1',color = 'darkred',fontsize=12)
```




    Text(0.5, 1.0, 'Count of Soil Types With Value 1')




![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_28_1.png)


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


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_30_0.png)


I wonder how many (y) labels we have in each class. I'll take a look the last column (cover type) for this.


```python
import plotly.express as px

cover_type = train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', height=400, width=650)
fig.show()
```

{% raw %}
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="8c8d5cbb-c68a-4919-9e47-e4937395b0a7" class="plotly-graph-div" style="height:400px; width:650px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("8c8d5cbb-c68a-4919-9e47-e4937395b0a7")) {
                    Plotly.newPlot(
                        '8c8d5cbb-c68a-4919-9e47-e4937395b0a7',
                        [{"alignmentgroup": "True", "hoverlabel": {"namelength": 0}, "hovertemplate": "CoverType=%{x}<br>Total=%{y}", "legendgroup": "", "marker": {"color": "#636efa"}, "name": "", "offsetgroup": "", "orientation": "v", "showlegend": false, "textposition": "auto", "type": "bar", "x": [7, 6, 5, 4, 3, 2, 1], "xaxis": "x", "y": [2160, 2160, 2160, 2160, 2160, 2160, 2160], "yaxis": "y"}],
                        {"barmode": "relative", "height": 400, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 650, "xaxis": {"anchor": "y", "domain": [0.0, 0.98], "title": {"text": "CoverType"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Total"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('8c8d5cbb-c68a-4919-9e47-e4937395b0a7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>
{% endraw %}

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


![png](/assets/files/Learn-Together-with-Sarfaraz_files/Learn-Together-with-Sarfaraz_34_0.png)


### Pandas Profiling
Actually there is a faster way for exploratory data analysis. Pandas provides you powerful HTML profiling reports with pandas-profiling. It's like a magic! You can click "Overview", "Variables" etc tabs for a quick run.


```python
import pandas_profiling as pp

report = pp.ProfileReport(train)
report.to_file("report.html")

report
```

## Modeling

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

    Train Shape: (11340, 54), (11340,)
    Validation Shape: (3780, 54), (3780,)



```python
# Let's fit the Decision Tree Model first

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# Fit the model
model.fit(train_X, train_y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')




```python
# Get predictions for the validation set

predictions = model.predict(val_X)
```


```python
# Let's check our categorization accuracy

from sklearn.metrics import accuracy_score

accuracy_score(val_y, predictions)
```




    0.7603174603174603



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15122</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15123</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15125</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



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

    Max leaf nodes: 5 - - Accuracy: 0.557936507936508
    Max leaf nodes: 55 - - Accuracy: 0.7092592592592593
    Max leaf nodes: 105 - - Accuracy: 0.7394179894179894
    Max leaf nodes: 155 - - Accuracy: 0.7497354497354497
    Max leaf nodes: 205 - - Accuracy: 0.7579365079365079
    Max leaf nodes: 255 - - Accuracy: 0.7584656084656085
    Max leaf nodes: 305 - - Accuracy: 0.7619047619047619
    Max leaf nodes: 355 - - Accuracy: 0.762962962962963
    Max leaf nodes: 405 - - Accuracy: 0.7658730158730159
    Max leaf nodes: 455 - - Accuracy: 0.7714285714285715
    Max leaf nodes: 505 - - Accuracy: 0.7698412698412699
    Max leaf nodes: 555 - - Accuracy: 0.7701058201058201
    Max leaf nodes: 605 - - Accuracy: 0.7690476190476191
    Max leaf nodes: 655 - - Accuracy: 0.7738095238095238
    Max leaf nodes: 705 - - Accuracy: 0.7748677248677248
    Max leaf nodes: 755 - - Accuracy: 0.7753968253968254
    Max leaf nodes: 805 - - Accuracy: 0.7761904761904762
    Max leaf nodes: 855 - - Accuracy: 0.7764550264550265
    Max leaf nodes: 905 - - Accuracy: 0.7753968253968254
    Max leaf nodes: 955 - - Accuracy: 0.7746031746031746
    Max leaf nodes: 1005 - - Accuracy: 0.7738095238095238
    Max leaf nodes: 1055 - - Accuracy: 0.773015873015873
    Max leaf nodes: 1105 - - Accuracy: 0.7746031746031746
    Max leaf nodes: 1155 - - Accuracy: 0.7746031746031746
    Max leaf nodes: 1205 - - Accuracy: 0.7711640211640212
    Max leaf nodes: 1255 - - Accuracy: 0.7706349206349207
    Max leaf nodes: 1305 - - Accuracy: 0.7695767195767196
    Max leaf nodes: 1355 - - Accuracy: 0.7674603174603175
    Max leaf nodes: 1405 - - Accuracy: 0.7666666666666667
    Max leaf nodes: 1455 - - Accuracy: 0.7669312169312169
    Max leaf nodes: 1505 - - Accuracy: 0.7653439153439153
    Max leaf nodes: 1555 - - Accuracy: 0.7637566137566137
    Max leaf nodes: 1605 - - Accuracy: 0.7634920634920634
    Max leaf nodes: 1655 - - Accuracy: 0.7603174603174603
    Max leaf nodes: 1705 - - Accuracy: 0.7600529100529101
    Max leaf nodes: 1755 - - Accuracy: 0.7600529100529101
    Max leaf nodes: 1805 - - Accuracy: 0.7600529100529101
    Max leaf nodes: 1855 - - Accuracy: 0.7600529100529101
    Max leaf nodes: 1905 - - Accuracy: 0.7600529100529101
    Max leaf nodes: 1955 - - Accuracy: 0.7600529100529101
    Best Max Leaf Nodes Parameter: (855, 0.7764550264550265)


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

    Random Forest Accuracy:  0.814021164021164


    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/forest.py:245: FutureWarning:

    The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.



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
    'n_estimators': [50, 100, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [8, 9, 10, 11, 12],
    'criterion' :['gini', 'entropy']
}

model_CV = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)

# We do not need validation set that's why we will train it on full training set
model_CV.fit(train, y)

# Let's check the best parameters
model_CV.best_params_
```

    CPU times: user 0 ns, sys: 2 µs, total: 2 µs
    Wall time: 5.48 µs





    {'criterion': 'entropy',
     'max_depth': 12,
     'max_features': 'auto',
     'n_estimators': 500}



Okay so let's train with the best parameters now


```python
model = RandomForestClassifier(random_state=0, max_features='auto', criterion='entropy', max_depth=10, n_estimators=500)

model.fit(train_X, train_y)

predictions = model.predict(val_X)

print("Best Parameters Random Forest Accuracy Is: ", accuracy_score(val_y, predictions))
```

    Best Parameters Random Forest Accuracy Is:  0.7962962962962963


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
