---
title: Learn Together Kaggle Competition with Sarfaraz
excerpt: "In this competition you’ll predict what types of trees there are in an area based on various geographic features.

The competition datasets comes from a study conducted in four wilderness areas within the beautiful Roosevelt National Forest of northern Colorado. These areas represent forests with very little human disturbances – the existing forest cover types there are more a result of ecological processes rather than forest management practices.

The data is in raw form and contains categorical data such as wilderness areas and soil type."
header:
  overlay_image: "https://storage.googleapis.com/kaggle-competitions/kaggle/15767/logos/header.png?t=2019-08-21-16-25-52"
---


# Learn With Other Kaggle Users
### Classify forest types based on information about the area
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


![png]("/assets/images/Learn-Together-with-Sarfaraz/15_0.png")


It seems the most important correlations are between "Horizontal Distance To Hydrology" and "Vertical Distance To Hydrology" with 70%; between "Aspect" and "Hillshade 3pm" with 60%; between "Hillshade Noon" and "Hillshade 3pm" with %60; between "Elevation" and "Horizontal Distance To Roadways" with %60. Let's see how they are looking.


```python
train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='yellow', figsize = (12,9))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()
```


![png]("/assets/images/Learn-Together-with-Sarfaraz/17_0.png")



```python
train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', alpha=0.5, color='maroon', figsize = (12,9))
plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")
plt.show()
```


![png]("/assets/images/Learn-Together-with-Sarfaraz/18_0.png")



```python
train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', alpha=0.5, color='purple', figsize = (12,9))
plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")
plt.show()
```


![png]("/assets/images/Learn-Together-with-Sarfaraz/19_0.png")


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


![png]("/assets/images/Learn-Together-with-Sarfaraz/26_0.png")


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




![png]("/assets/images/Learn-Together-with-Sarfaraz/28_1.png")


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


![png]("/assets/images/Learn-Together-with-Sarfaraz/30_0.png")


I wonder how many (y) labels we have in each class. I'll take a look the last column (cover type) for this.


```python
import plotly.express as px

cover_type = train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', height=400, width=650)
fig.show()
```


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


![png]("/assets/images/Learn-Together-with-Sarfaraz/34_0.png")


### Pandas Profiling
Actually there is a faster way for exploratory data analysis. Pandas provides you powerful HTML profiling reports with pandas-profiling. It's like a magic! You can click "Overview", "Variables" etc tabs for a quick run.


```python
import pandas_profiling as pp

report = pp.ProfileReport(train)
report.to_file("report.html")

report
```




<meta charset="UTF-8">

<style>

        .variablerow {
            border: 1px solid #e1e1e8;
            border-top: hidden;
            padding-top: 2em;
            padding-bottom: 2em;
            padding-left: 1em;
            padding-right: 1em;
        }

        .headerrow {
            border: 1px solid #e1e1e8;
            background-color: #f5f5f5;
            padding: 2em;
        }
        .namecol {
            margin-top: -1em;
            overflow-x: auto;
        }

        .dl-horizontal dt {
            text-align: left;
            padding-right: 1em;
            white-space: normal;
        }

        .dl-horizontal dd {
            margin-left: 0;
        }

        .ignore {
            opacity: 0.4;
        }

        .container.pandas-profiling {
            max-width:975px;
        }

        .col-md-12 {
            padding-left: 2em;
        }

        .indent {
            margin-left: 1em;
        }

        .center-img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block;
        }

        /* Table example_values */
            table.example_values {
                border: 0;
            }

            .example_values th {
                border: 0;
                padding: 0 ;
                color: #555;
                font-weight: 600;
            }

            .example_values tr, .example_values td{
                border: 0;
                padding: 0;
                color: #555;
            }

        /* STATS */
            table.stats {
                border: 0;
            }

            .stats th {
                border: 0;
                padding: 0 2em 0 0;
                color: #555;
                font-weight: 600;
            }

            .stats tr {
                border: 0;
            }

            .stats td{
                color: #555;
                padding: 1px;
                border: 0;
            }


        /* Sample table */
            table.sample {
                border: 0;
                margin-bottom: 2em;
                margin-left:1em;
            }
            .sample tr {
                border:0;
            }
            .sample td, .sample th{
                padding: 0.5em;
                white-space: nowrap;
                border: none;

            }

            .sample thead {
                border-top: 0;
                border-bottom: 2px solid #ddd;
            }

            .sample td {
                width:100%;
            }


        /* There is no good solution available to make the divs equal height and then center ... */
            .histogram {
                margin-top: 3em;
            }
        /* Freq table */

            table.freq {
                margin-bottom: 2em;
                border: 0;
            }
            table.freq th, table.freq tr, table.freq td {
                border: 0;
                padding: 0;
            }

            .freq thead {
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;

            }

            td.fillremaining{
                width:auto;
                max-width: none;
            }

            td.number, th.number {
                text-align:right ;
            }

        /* Freq mini */
            .freq.mini td{
                width: 50%;
                padding: 1px;
                font-size: 12px;

            }
            table.freq.mini {
                 width:100%;
            }
            .freq.mini th {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 5em;
                font-weight: 400;
                text-align:right;
                padding-right: 0.5em;
            }

            .missing {
                color: #a94442;
            }
            .alert, .alert > th, .alert > td {
                color: #a94442;
            }


        /* Bars in tables */
            .freq .bar{
                float: left;
                width: 0;
                height: 100%;
                line-height: 20px;
                color: #fff;
                text-align: center;
                background-color: #337ab7;
                border-radius: 3px;
                margin-right: 4px;
            }
            .other .bar {
                background-color: #999;
            }
            .missing .bar{
                background-color: #a94442;
            }
            .tooltip-inner {
                width: 100%;
                white-space: nowrap;
                text-align:left;
            }

            .extrapadding{
                padding: 2em;
            }

            .pp-anchor{

            }

</style>

<div class="container pandas-profiling">
    <div class="row headerrow highlight">
        <h1>Overview</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-6 namecol">
        <p class="h4">Dataset info</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Number of variables</th>
                <td>56 </td>
            </tr>
            <tr>
                <th>Number of observations</th>
                <td>15120 </td>
            </tr>
            <tr>
                <th>Total Missing (%)</th>
                <td>0.0% </td>
            </tr>
            <tr>
                <th>Total size in memory</th>
                <td>2.0 MiB </td>
            </tr>
            <tr>
                <th>Average record size in memory</th>
                <td>140.3 B </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-6 namecol">
        <p class="h4">Variables types</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Numeric</th>
                <td>12 </td>
            </tr>
            <tr>
                <th>Categorical</th>
                <td>42 </td>
            </tr>
            <tr>
                <th>Boolean</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Date</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Text (Unique)</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Rejected</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Unsupported</th>
                <td>0 </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-12" style="padding-left: 1em;">

        <p class="h4">Warnings</p>
        <ul class="list-unstyled"><li><a href="#pp_var_Horizontal_Distance_To_Hydrology"><code>Horizontal_Distance_To_Hydrology</code></a> has 1590 / 10.5% zeros <span class="label label-info">Zeros</span></li><li><a href="#pp_var_Soil_Type15"><code>Soil_Type15</code></a> has constant value 0 <span class="label label-primary">Rejected</span></li><li><a href="#pp_var_Soil_Type7"><code>Soil_Type7</code></a> has constant value 0 <span class="label label-primary">Rejected</span></li><li><a href="#pp_var_Vertical_Distance_To_Hydrology"><code>Vertical_Distance_To_Hydrology</code></a> has 1890 / 12.5% zeros <span class="label label-info">Zeros</span></li> </ul>
    </div>
</div>
    <div class="row headerrow highlight">
        <h1>Variables</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Aspect">Aspect<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>361</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>2.4%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>156.68</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>360</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.7%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram4491030725958535617">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAATdJREFUeJzt2sFtAjEUQEFAlJQi0lPO9EQR9OQ0ED0B0rKOd%2Ba%2Bki9P3/b6PMYYJ%2BBPl70XADO77r2AvXz93F/%2B5nH73mAlzMwEgbDEBHlnGsAzTBAIAoEgEAgCgbDEIf1TXr0McC38/00XiBspZmKLBUEgEKbbYrGOFZ7zmCAQBAJBIBAEAkEgEAQCQSAQ/AfhKUd9AmSCQBAIBIFAcAbZ0ApvkY7OBIEgEAgCgSAQCA7pk3Gwn4sJAkEgEGyxDuio76reYYJAEAgEW6wF2DJtxwSBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCOcxxth7ETArEwSCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCL%2BVfHqSJKoJtAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives4491030725958535617,#minihistogram4491030725958535617"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives4491030725958535617">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles4491030725958535617"
                                                  aria-controls="quantiles4491030725958535617" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram4491030725958535617" aria-controls="histogram4491030725958535617"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common4491030725958535617" aria-controls="common4491030725958535617"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme4491030725958535617" aria-controls="extreme4491030725958535617"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles4491030725958535617">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>13</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>65</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>126</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>261</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>344</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>360</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>360</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>196</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>110.09</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.70263</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.1502</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>156.68</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>95.394</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.45094</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>2368951</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>12119</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram4491030725958535617">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3XtcVXW%2B//E3F6EUQbeFzUETx1vCBvNeaAeUwsq0vCLnYWZHy4xyZLQsranJzmgP8ZgTHouaOsfGKfIyeRlvY15Ss6apo23xchL1ocNDB0p2BIEirN8f/dwzOxBvX9Z2s1/Px4OH%2Bv3utdbn415782atxdpBlmVZAgAAgDHBvi4AAACgsSFgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDQn1dQKAoLv7e%2BDqDg4PkcDTT6dPlqqmxjK//WhNo/Ur0HAg9B1q/UuD1HGj9StdWzzfe2Nwn2%2BUIlh8LDg5SUFCQgoODfF2KLQKtX4meA0Gg9SsFXs%2BB1q8UmD3/FAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwL9XUBwLXqnld3%2BbqEy7J%2Baj9flwAA%2BP84ggUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYXwWIWzjb5/tBwDAleIIFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYJjfBqzCwkJlZmaqb9%2B%2BSkpK0jPPPKPS0lJJ0oEDBzR27Fj17NlTaWlpevvtt72WXbdunYYMGaLu3btr%2BPDh2rlzp2eupqZGCxYsUGpqqnr37q0JEyboxIkTtvYGAAD8m98GrMcee0yRkZHasmWLVq5cqa%2B//lqvvPKKKisrNWnSJN12223asWOHFixYoDfeeEObNm2S9GP4mjFjhqZPn65PP/1U48eP1xNPPKFTp05JkpYuXao1a9YoNzdXW7duVWxsrDIzM2VZli/bBQAAfsQvA1ZpaamcTqemTZumZs2a6aabbtKwYcP017/%2BVdu2bVNVVZUmT56spk2bKj4%2BXqNGjVJeXp4kadmyZUpOTlZycrLCw8M1dOhQde7cWatXr5Yk5eXlafz48erQoYMiIiKUlZWlgoIC7d2715ctAwAAP%2BKXASsyMlJz5szRDTfc4Bk7efKkoqOjlZ%2Bfry5duigkJMQzFxcXp3379kmS8vPzFRcX57W%2BuLg4uVwuVVZW6vDhw17zERERateunVwuVwN3BQAAGotQXxdggsvl0u9//3stXrxY69evV2RkpNd8ixYt5Ha7VVNTI7fbraioKK/5qKgoHT58WN99950sy6pzvqSk5JLrKSoqUnFxsddYaGhTRUdHX2Zn9QsJCfb6E4EtNLRx7AeBtl8HWr9S4PUcaP1KgdnzT/l9wPriiy80efJkTZs2TUlJSVq/fn2djwsKCvL8/WLXU13t9VZ5eXnKycnxGsvMzNSUKVOuar0XEhl5fYOsF/6lZctmvi7BqEDbrwOtXynweg60fqXA7Pk8vw5YW7Zs0VNPPaXnn39eDzzwgCTJ4XDo2LFjXo9zu91q0aKFgoOD1bJlS7nd7lrzDofD85i65lu1anXJdaWnp2vgwIFeY6GhTVVSUn4Z3V1cSEiwIiOvV2lphaqra4yuG/7H9P7lK4G2Xwdav1Lg9Rxo/UrXVs%2B%2B%2BuHTbwPWl19%2BqRkzZmjhwoXq37%2B/Z9zpdOq9997TuXPnFBr6Y3sul0vdunXzzJ%2B/Hus8l8ulwYMHKzw8XJ06dVJ%2Bfr769Okj6ccL6o8fP67ExMRLri06OrrW6cDi4u917lzD7GTV1TUNtm74j8a2DwTafh1o/UqB13Og9SsFZs/n%2BeXJ0XPnzum5557T9OnTvcKVJCUnJysiIkKLFy9WRUWF9u7dq%2BXLlysjI0OSNHr0aH3yySfatm2bzpw5o%2BXLl%2BvYsWMaOnSoJCkjI0NLlixRQUGBysrKlJ2dra5duyohIcH2PgEAgH/yyyNYe/bsUUFBgV5%2B%2BWW9/PLLXnMbNmzQ66%2B/rhdeeEG5ubm64YYblJWVpZSUFElS586dlZ2drTlz5qiwsFAdO3bUG2%2B8oRtvvFGSNGbMGBUXF%2BvBBx9UeXm5%2BvbtW%2Bt6KgAAgPoEWdxB0xbFxd8bX2doaLBatmymkpJyvzgEe8%2Bru3xdAq4R66f2u%2BCcv%2B3XVyvQ%2BpUCr%2BdA61e6tnq%2B8cbmPtmuX54iBAAAuJYRsAAAAAzzy2uw8A%2B9Zm3wdQkAAOAnOIIFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGBYqK8LAAAADeOeV3f5uoRLtn5qP1%2BXYBRHsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAY5tcBa8eOHUpKSlJWVpbX%2BMqVK3XLLbcoISHB6%2Burr76SJNXU1GjBggVKTU1V7969NWHCBJ04ccKzvNvt1tSpU5WUlKT%2B/ftr1qxZqqystLU3AADgv/w2YL355pt6%2BeWX1a5duzrne/fuLZfL5fWVmJgoSVq6dKnWrFmj3Nxcbd26VbGxscrMzJRlWZKk559/XhUVFVq7dq1WrFihgoICZWdn29YbAADwb34bsMLDw7V8%2BfILBqz65OXlafz48erQoYMiIiKUlZWlgoIC7d27V9988402b96srKwsORwOtW7dWo8//rhWrFihqqqqBugEAAA0Nn4bsMaNG6fmzZtfcP7kyZN6%2BOGH1bt3b6WmpmrVqlWSpMrKSh0%2BfFhxcXGex0ZERKhdu3ZyuVw6cOCAQkJC1KVLF898fHy8fvjhBx05cqThGgIAAI1GqK8LaAgOh0OxsbH65S9/qY4dO%2BrPf/6znn76aUVHR%2BvnP/%2B5LMtSVFSU1zJRUVEqKSlRixYtFBERoaCgIK85SSopKbmk7RcVFam4uNhrLDS0qaKjo6%2ByM28hIX6bjxHgQkMvvO%2Be368DZf8OtH6lwOs50Pq9UvW9L/ijRhmwUlJSlJKS4vn34MGD9ec//1krV67U9OnTJclzvVVd6pu7FHl5ecrJyfEay8zM1JQpU65qvUBj0bJls4s%2BJjLyehsquXYEWr9S4PUcaP1erkt5X/AnjTJg1SUmJkb79u1TixYtFBwcLLfb7TXvdrvVqlUrORwOlZWVqbq6WiEhIZ45SWrVqtUlbSs9PV0DBw70GgsNbaqSknIDnfwDPw3BX9X3WggJCVZk5PUqLa1QdXWNjVX5RqD1KwVez4HW75Uy/T3yPF8Ft0YZsN577z1FRUXp3nvv9YwVFBSobdu2Cg8PV6dOnZSfn68%2BffpIkkpLS3X8%2BHElJiYqJiZGlmXp4MGDio%2BPlyS5XC5FRkaqffv2l7T96OjoWqcDi4u/17lzvLAASZf0Wqiurgmo10yg9SsFXs%2BB1u/lamz/N43yEMjZs2c1e/ZsuVwuVVVVae3atfr44481ZswYSVJGRoaWLFmigoIClZWVKTs7W127dlVCQoIcDocGDRqkV199VadPn9apU6e0aNEijRw5UqGhjTKPAgAAw/w2MSQkJEiSzp07J0navHmzpB%2BPNo0bN07l5eX6xS9%2BoeLiYrVp00aLFi2S0%2BmUJI0ZM0bFxcV68MEHVV5err59%2B3pdM/XSSy/phRdeUGpqqpo0aaL77ruv1s1MAQAALiTIutorui/TwIEDNXz4cI0YMUI/%2B9nP7Ny0TxUXf298naGhwbore4fx9QINbf3UfhecCw0NVsuWzVRSUt7oThnUJdD6lQKvZ1/2e8%2Bru2zd3tWo733hatx444Vv6dSQbD9FOGLECK1bt0533nmnJk6cqE2bNnmOQgEAADQGtgeszMxMrVu3Th988IE6deqk3/zmN0pOTta8efN09OhRu8sBAAAwzmcXucfHx2vGjBnaunWrZs6cqQ8%2B%2BED33nuvJkyY4PlQZgAAAH/ks4BVVVWldevW6ZFHHtGMGTPUunVrPfvss%2BratavGjx%2BvNWvW%2BKo0AACAq2L7bxEWFBRo%2BfLl%2BvDDD1VeXq5Bgwbpf/7nf9SzZ0/PY3r37q0XX3xRQ4YMsbs8AACAq2Z7wBo8eLDat2%2BvSZMm6YEHHlCLFi1qPSY5OVmnT5%2B2uzQAAAAjbA9YS5Ys8dxBvT579%2B61oRoAAADzbL8Gq0uXLnrsscc8NwaVpP/%2B7//WI488UuvzAQEAAPyR7QFrzpw5%2Bv7779WxY0fPWEpKimpqajR37ly7ywEAADDO9lOEO3fu1Jo1a9SyZUvPWGxsrLKzs3XffffZXQ4AAIBxth/BqqysVHh4eO1CgoNVUVFhdzkAAADG2X4Eq3fv3po7d66mTZumqKgoSdLf//53vfLKK163agAA4FrjT5/tB9%2ByPWDNnDlT//7v/67bb79dERERqqmpUXl5udq2bat3333X7nIAAACMsz1gtW3bVn/605/08ccf6/jx4woODlb79u3Vv39/hYSE2F0OAACAcbYHLEkKCwvTnXfe6YtNAwAANDjbA9aJEyc0f/58ff3116qsrKw1/9FHH9ldEgAAgFE%2BuQarqKhI/fv3V9OmTe3ePAAAQIOzPWDt27dPH330kRwOh92bBgAAsIXt98Fq1aoVR64AAECjZnvAmjRpknJycmRZlt2bBgAAsIXtpwg//vhjffnll1q5cqXatGmj4GDvjPf%2B%2B%2B/bXRIAAIBRtgesiIgI/eu//qvdmwUAALCN7QFrzpw5dm8SAADAVrZfgyVJR44c0WuvvaZnn33WM/a///u/vigFAADAONsD1u7duzV06FBt2rRJa9eulfTjzUfHjRvHTUYBAECjYHvAWrBggZ566imtWbNGQUFBkn78fMK5c%2Bdq0aJFdpcDAABgnO0B6//%2B7/%2BUkZEhSZ6AJUl33323CgoK7C4HAADAONsDVvPmzev8DMKioiKFhYXZXQ4AAIBxtgesHj166De/%2BY3Kyso8Y0ePHtWMGTN0%2B%2B23210OAACAcbbfpuHZZ5/VQw89pL59%2B6q6ulo9evRQRUWFOnXqpLlz59pdDgAAgHG2B6ybbrpJa9eu1fbt23X06FFdd911at%2B%2Bvfr16%2Bd1TRYAAIC/sj1gSVKTJk105513%2BmLTAAAADc72gDVw4MB6j1RxLywAAODvbA9Y9957r1fAqq6u1tGjR%2BVyufTQQw/ZXQ4AAIBxtges6dOn1zm%2BceNGffbZZzZXAwAAYJ5PrsGqy5133qlf/epX%2BtWvfuXrUgA0sHte3eXrEi7L%2Bqn9fF0CAD/jkw97rsv%2B/ftlWZavywAAALhqth/BGjNmTK2xiooKFRQUKC0tze5yAAAAjLM9YMXGxtb6LcLw8HCNHDlSo0aNsrscAAAA42wPWNytHQAANHa2B6wPP/zwkh/7wAMPNGAlAAAADcP2gDVr1izV1NTUuqA9KCjIaywoKIiABQAA/JLtAeutt97S22%2B/rccee0xdunSRZVk6dOiQ3nzzTY0dO1Z9%2B/a1uyQAAACjfHINVm5urlq3bu0Z69Wrl9q2basJEyZo7dq1dpcEAABglO33wTp27JiioqJqjUdGRqqwsNDucgAAAIyzPWDFxMRo7ty5Kikp8YyVlpZq/vz5uvnmm%2B0uBwAAwDjbTxHOnDlT06ZNU15enpo1a6bg4GCVlZXpuuuu06JFi%2BwuBwAAwDjbA1b//v21bds2bd%2B%2BXadOnZJlWWrdurXuuOMONW/e3O5yAAAAjPPJhz1ff/31Sk1N1alTp9S2bVtflAAAANBgbL8Gq7KyUjNmzFD37t11zz33SPrxGqyJEyeqtLTU7nIAAACMsz1gzZs3TwcOHFB2draCg/%2Bx%2BerqamVnZ9tdDgAAgHG2B6yNGzfqt7/9re6%2B%2B27Phz5HRkZqzpw52rRpk93lAAAAGGd7wCovL1dsbGytcYfDoR9%2B%2BMHucgAAAIyzPWDdfPPN%2BuyzzyTJ67MHN2zYoH/5l3%2BxuxwAAADjbP8twn/7t3/Tk08%2BqREjRqimpkbvvPOO9u3bp40bN2rWrFl2lwMAAGCc7Uew0tPTNWPGDH366acKCQnR66%2B/rsLCQmVnZysjI%2BOy1rVjxw4lJSUpKyur1ty6des0ZMgQde/eXcOHD9fOnTs9czU1NVqwYIFSU1PVu3dvTZgwQSdOnPDMu91uTZ06VUlJSerfv79mzZqlysrKK28aAAAEFNuPYJ0%2BfVojRozQiBEjrmo9b775ppYvX6527drVmjtw4IBmzJihnJwc3Xbbbdq4caOeeOIJbdiwQTfddJOWLl2qNWvW6M0331Tr1q21YMECZWZmatWqVQoKCtLzzz%2Bvs2fPau3ataqqqtIvfvELZWdn67nnnruqmgEAQGCw/QhWamqq17VXVyo8PPyCAWvZsmVKTk5WcnKywsPDNXToUHXu3FmrV6%2BWJOXl5Wn8%2BPHq0KGDIiIilJWVpYKCAu3du1fffPONNm/erKysLDkcDrVu3VqPP/64VqxYoaqqqquuGwAANH62B6y%2Bfftq/fr1V72ecePGXfCjdfLz8xUXF%2Bc1FhcXJ5fLpcrKSh0%2BfNhrPiIiQu3atZPL5dKBAwcUEhKiLl26eObj4%2BP1ww8/6MiRI1ddNwAAaPxsP0X4s5/9TP/xH/%2Bh3Nxc3XzzzWrSpInX/Pz58696G263W1FRUV5jUVFROnz4sL777jtZllXnfElJiVq0aKGIiAjPPbrOz0lSSUnJJW2/qKhIxcXFXmOhoU0VHR19Je1cUEiI7fkYCEihoQ33Wjv/Og6k13Mg9oyLa8jXmS/YHrAOHz6sn//855IuPbBciYudhqxv/mpPYebl5SknJ8drLDMzU1OmTLmq9QLwjZYtmzX4NiIjr2/wbVxrArFnXJgdrzM72RawsrKytGDBAr377ruesUWLFikzM9P4tlq2bCm32%2B015na75XA41KJFCwUHB9c536pVKzkcDpWVlam6ulohISGeOUlq1arVJW0/PT1dAwcO9BoLDW2qkpLyK22pTvz0B9jD9Gv3n4WEBCsy8nqVllaourqmwbZzLQnEnnFxDfU681Vwsy1gbdmypdZYbm5ugwQsp9Opffv2eY25XC4NHjxY4eHh6tSpk/Lz89WnTx9JP37Y9PHjx5WYmKiYmBhZlqWDBw8qPj7es2xkZKTat29/SduPjo6udTqwuPh7nTvHGwngj%2Bx47VZX1wTce0Qg9owLa2z7gm2HQOo67WbitwnrMnr0aH3yySfatm2bzpw5o%2BXLl%2BvYsWMaOnSoJCkjI0NLlixRQUGBysrKlJ2dra5duyohIUEOh0ODBg3Sq6%2B%2BqtOnT%2BvUqVNatGiRRo4cqdBQ28%2BoAgAAP2RbYvjni8brG7tUCQkJkqRz585JkjZv3izpx6NNnTt3VnZ2tubMmaPCwkJ17NhRb7zxhm688UZJ0pgxY1RcXKwHH3xQ5eXl6tu3r9c1Uy%2B99JJeeOEFpaamqkmTJrrvvvvqvJkpAABAXfz2kIzL5ap3Pi0tTWlpaXXOBQUFacqUKRe86Lx58%2Bb6z//8z6uuEQAABCaukgYAADDMtiNYVVVVmjZt2kXHTNwHCwAAwJdsC1g9e/ZUUVHRRccAAAD8nW0B65/vfwUAANCY%2Be1F7gBgl3te3eXrEi7L%2Bqn9fF0CEPC4yB0AAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLBQXxcAAAhc97y6y9clAA2CI1gAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGBYqK8LAACYdc%2Bru3xdAhDwOIIFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLBGG7C6dOkip9OphIQEz9fs2bMlSbt379bIkSPVo0cPDR48WKtXr/ZadsmSJRo0aJB69OihjIwM7du3zxctAAAAP9WoP4tww4YNatOmjddYUVGRHn/8cc2aNUtDhgzRF198ocmTJ6t9%2B/ZKSEjQli1b9Nprr%2Bmtt95Sly5dtGTJEj322GPatGmTmjZt6qNOAACAP2m0R7AuZM2aNYqNjdXIkSMVHh6upKQkDRw4UMuWLZMk5eXlafjw4erWrZuuu%2B46TZw4UZK0detWX5YNAAD8SKMOWPPnz1dKSop69eql559/XuXl5crPz1dcXJzX4%2BLi4jynAX86HxwcrK5du8rlctlaOwAA8F%2BN9hThrbfeqqSkJL3yyis6ceKEpk6dql//%2Btdyu91q3bq112NbtGihkpISSZLb7VZUVJTXfFRUlGf%2BUhQVFam4uNhrLDS0qaKjo6%2Bwm7qFhDTqfAwACCChoY3re1qjDVh5eXmev3fo0EHTp0/X5MmT1bNnz4sua1nWVW87JyfHaywzM1NTpky5qvUCANBYtWzZzNclGNVoA9ZPtWnTRtXV1QoODpbb7faaKykpkcPhkCS1bNmy1rzb7VanTp0ueVvp6ekaOHCg11hoaFOVlJRfYfV14wgWAKCxMP098jxfBbdGGbD279%2Bv1atX65lnnvGMFRQUKCwsTMnJyfrjH//o9fh9%2B/apW7dukiSn06n8/HwNGzZMklRdXa39%2B/dr5MiRl7z96OjoWqcDi4u/17lzNVfaEgAAjVpj%2Bx7ZKA%2BBtGrVSnl5ecrNzdXZs2d19OhRLVy4UOnp6br//vtVWFioZcuW6cyZM9q%2Bfbu2b9%2Bu0aNHS5IyMjL04Ycfas%2BePaqoqNDixYsVFhamlJQU3zYFAAD8RpB1tRccXaM%2B//xzzZ8/X4cOHVJYWJiGDRumrKwshYeH6/PPP9fLL7%2BsgoICxcTEaNq0aUpLS/Ms%2B4c//EG5ubn69ttvlZCQoBdffFGdO3e%2BqnqKi7%2B/2pZqCQ0N1l3ZO4yvFwAAu62f2q9B1nvjjc0bZL0X02gD1rWGgAUAwIU1toDVKE8RAgAA%2BBIBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgFWHwsJCPfroo%2Brbt68GDBigefPmqaamxtdlAQAAPxHq6wKuRU8%2B%2BaTi4%2BO1efNmffvtt5o0aZJuuOEGPfzww74uDQAA%2BAGOYP2Ey%2BXSwYMHNX36dDVv3lyxsbEaP3688vLyfF0aAADwExzB%2Bon8/HzFxMQoKirKMxYfH6%2BjR4%2BqrKxMERERF11HUVGRiouLvcZCQ5sqOjraaK0hIeRjAEDjEBrauL6nEbB%2Bwu12KzIy0mvsfNgqKSm5pICVl5ennJwcr7EnnnhCTz75pLlC9WOQe%2Bimr5Wenm48vF2LioqKlJeXFzD9SvQcCD0HWr9S4PUcaP1KgdnzTzWuuGiIZVlXtXx6erpWrlzp9ZWenm6oun8oLi5WTk5OraNljVWg9SvRcyAItH6lwOs50PqVArPnn%2BII1k84HA653W6vMbfbraCgIDkcjktaR3R0dMAmdgAAwBGsWpxOp06ePKnTp097xlwulzp27KhmzZr5sDIAAOAvCFg/ERcXp4SEBM2fP19lZWUqKCjQO%2B%2B8o4yMDF%2BXBgAA/ETIiy%2B%2B%2BKKvi7jW3HHHHVq7dq1mz56tP/3pTxo5cqQmTJigoKAgX5dWS7NmzdSnT5%2BAOboWaP1K9BwIAq1fKfB6DrR%2BpcDs%2BZ8FWVd7RTcAAAC8cIoQAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQDhPhYtAAAKnElEQVQAYBgBCwAAwDAClh8qLCzUo48%2Bqr59%2B2rAgAGaN2%2BeampqfF2WUV26dJHT6VRCQoLna/bs2ZKk3bt3a%2BTIkerRo4cGDx6s1atX%2B7jaK7Njxw4lJSUpKyur1ty6des0ZMgQde/eXcOHD9fOnTs9czU1NVqwYIFSU1PVu3dvTZgwQSdOnLCz9Ct2oZ5XrlypW265xev5TkhI0FdffSXJf3suLCxUZmam%2Bvbtq6SkJD3zzDMqLS2VJB04cEBjx45Vz549lZaWprfffttr2fr2gWvZhXr%2B29/%2Bpi5dutR6jn/3u995lvXHng8ePKiHHnpIPXv2VFJSkqZOnari4mJJF3%2BvWrJkiQYNGqQePXooIyND%2B/bt80ULl%2B1CPX/22Wd1Psfr16/3LOuvPV8RC35n2LBh1nPPPWeVlpZaR48etdLS0qy3337b12UZ1blzZ%2BvEiRO1xv/%2B979bt956q7Vs2TKrsrLS2rVrl5WYmGh99dVXPqjyyuXm5lppaWnWmDFjrKlTp3rN7d%2B/33I6nda2bdusyspKa9WqVVa3bt2skydPWpZlWUuWLLEGDBhgHT582Pr%2B%2B%2B%2Btl156yRoyZIhVU1Pji1YuWX09r1ixwho7duwFl/XXnu%2B77z7rmWeescrKyqyTJ09aw4cPt2bOnGlVVFRYd9xxh/Xaa69Z5eXl1r59%2B6w%2BffpYGzdutCzr4vvAtexCPZ84ccLq3LnzBZfzx57PnDlj3X777VZOTo515swZ69tvv7XGjh1rPf744xd9r/roo4%2BsXr16WXv27LEqKiqsN954w%2BrXr59VXl7u467qV1/Pn376qTVgwIALLuuvPV8pjmD5GZfLpYMHD2r69Olq3ry5YmNjNX78eOXl5fm6NFusWbNGsbGxGjlypMLDw5WUlKSBAwdq2bJlvi7tsoSHh2v58uVq165drblly5YpOTlZycnJCg8P19ChQ9W5c2fPT795eXkaP368OnTooIiICGVlZamgoEB79%2B61u43LUl/PF%2BOPPZeWlsrpdGratGlq1qyZbrrpJg0bNkx//etftW3bNlVVVWny5Mlq2rSp4uPjNWrUKM/r%2BGL7wLWqvp4vxh97rqioUFZWliZNmqSwsDA5HA7ddddd%2Bvrrry/6XpWXl6fhw4erW7duuu666zRx4kRJ0tatW33Z0kXV1/PF%2BGvPV4qA5Wfy8/MVExOjqKgoz1h8fLyOHj2qsrIyH1Zm3vz585WSkqJevXrp%2BeefV3l5ufLz8xUXF%2Bf1uLi4OL87zDxu3Dg1b968zrkL9ehyuVRZWanDhw97zUdERKhdu3ZyuVwNWvPVqq9nSTp58qQefvhh9e7dW6mpqVq1apUk%2BW3PkZGRmjNnjm644QbP2MmTJxUdHa38/Hx16dJFISEhnrl/3o/r2weuZfX1fN7TTz%2Bt/v3767bbbtP8%2BfNVVVUlyT97joqK0qhRoxQaGipJOnLkiP74xz/qnnvuueh71U/ng4OD1bVr12u6X6n%2BniWpvLzcc4r4jjvu0DvvvCPLsiT5b89XioDlZ9xutyIjI73GzoetkpISX5TUIG699VYlJSVp06ZNysvL0549e/TrX/%2B6zv5btGjRqHp3u91eAVr68TkuKSnRd999J8uyLjjvrxwOh2JjY/XUU09p165d%2BuUvf6mZM2dq9%2B7djaZnl8ul3//%2B95o8efIF92O3262ampp69wF/8s89h4WFqXv37rrrrru0detW5ebmavXq1fqv//ovSfXv99e6wsJCOZ1O3XvvvUpISNCUKVMu%2Bl7lz/1KdfccERGhzp0766GHHtKOHTs0Z84c5eTkaMWKFZL8v%2BfLRcDyQ%2Bd/GmjM8vLyNGrUKIWFhalDhw6aPn261q5d6/lpt7G72HPc2PaBlJQUvfXWW4qLi1NYWJgGDx6su%2B66SytXrvQ8xp97/uKLLzRhwgRNmzZNSUlJF3xcUFCQ5%2B/%2B3K9Uu%2Bfo6Gi9//77uuuuu9SkSRMlJiZq0qRJjeI5jomJkcvl0oYNG3Ts2DE9/fTTl7Scv/Yr1d1zfHy83n33XfXp00dhYWHq37%2B/xowZ0yie4ytBwPIzDodDbrfba8ztdisoKEgOh8NHVTW8Nm3aqLq6WsHBwbX6LykpaVS9t2zZss7n2OFwqEWLFnX%2BH7jdbrVq1crOMhtcTEyMioqK/L7nLVu26NFHH9XMmTM1btw4ST%2B%2Bjn/6U7vb7fb0Wt8%2B4A/q6rkuMTEx%2Buabb2RZlt/3HBQUpNjYWGVlZWnt2rUKDQ2t973K3/uVavd8%2BvTpWo85/zqWGkfPl4OA5WecTqdOnjzptSO7XC517NhRzZo182Fl5uzfv19z5871GisoKFBYWJiSk5NrXW%2B1b98%2BdevWzc4SG5TT6azVo8vlUrdu3RQeHq5OnTopPz/fM1daWqrjx48rMTHR7lKNee%2B997Ru3TqvsYKCArVt29ave/7yyy81Y8YMLVy4UA888IBn3Ol06tChQzp37pxn7PxzfH7%2BQvvAte5CPe/evVuLFy/2euyRI0cUExOjoKAgv%2Bx59%2B7dGjRokNdtcoKDf/y2mpiYWO97ldPp9Nqnq6urtX///mu6X6n%2Bnrdv364//OEPXo8/cuSI2rZtK8l/e75SBCw/ExcXp4SEBM2fP19lZWUqKCjQO%2B%2B8o4yMDF%2BXZkyrVq2Ul5en3NxcnT17VkePHtXChQuVnp6u%2B%2B%2B/X4WFhVq2bJnOnDmj7du3a/v27Ro9erSvyzZm9OjR%2BuSTT7Rt2zadOXNGy5cv17FjxzR06FBJUkZGhpYsWaKCggKVlZUpOztbXbt2VUJCgo8rv3Jnz57V7Nmz5XK5VFVVpbVr1%2Brjjz/WmDFjJPlnz%2BfOndNzzz2n6dOnq3///l5zycnJioiI0OLFi1VRUaG9e/dq%2BfLlntfxxfaBa1V9PTdv3lyLFi3SqlWrVFVVJZfLpd/97nd%2B3bPT6VRZWZnmzZuniooKnT59Wq%2B99pp69eqljIyMet%2BrMjIy9OGHH2rPnj2qqKjQ4sWLFRYWppSUFN82dRH19dy8eXO98sor2rlzp6qqqrRr1y6tWLHC8xz7a89XzDd3h8DVOHnypDVx4kQrMTHRSkpKsn77299e8/cDulx/%2BctfrPT0dOvWW2%2B1%2BvTpY82ZM8eqrKz0zA0dOtSKj4%2B30tLSPPcO8idOp9NyOp3WLbfcYt1yyy2ef5%2B3ceNGKy0tzYqPj7fuv/9%2B6y9/%2BYtnrqamxlq4cKF1%2B%2B23W4mJidYjjzxyTd8r6Lz6eq6pqbEWLVpkDRgwwHI6ndbdd99tbdmyxbOsP/b8%2BeefW507d/b0%2Bc9ff/vb36xDhw5ZY8aMsZxOp5WSkmItXbrUa/n69oFr1cV63rRpkzV06FArMTHR6tevn/X6669b1dXVnuX9seeDBw9aY8eOtRITE63bbrvNmjp1qnXq1CnLsi7%2BXrV06VIrOTnZcjqdVkZGhnXo0CFftHDZ6uv5/ffft9LS0qyEhARrwIAB1gcffOC1rL/2fCWCLCuArjgDAACwAacIAQAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMCw/wfZEIW4HDpaPgAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common4491030725958535617">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">45</td>
        <td class="number">117</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">0</td>
        <td class="number">110</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">90</td>
        <td class="number">109</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">63</td>
        <td class="number">89</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">76</td>
        <td class="number">87</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">27</td>
        <td class="number">82</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">315</td>
        <td class="number">81</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">75</td>
        <td class="number">80</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">108</td>
        <td class="number">79</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">117</td>
        <td class="number">78</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (351)</td>
        <td class="number">14208</td>
        <td class="number">94.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme4491030725958535617">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">110</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">48</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:44%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">50</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:46%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">54</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:49%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">51</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:46%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">356</td>
        <td class="number">50</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:86%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">357</td>
        <td class="number">58</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">358</td>
        <td class="number">47</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:81%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">359</td>
        <td class="number">33</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:57%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">360</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Cover_Type">Cover_Type<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>7</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>4</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>7</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram5929972347266019934">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAAQpJREFUeJzt1bENwkAQRUGDKIki6ImYniiCnpYcoSfZEtjBTH7anzzdaWZmAb467z0Ajuyy94BP1/vzL3dej9vqN2u3/ePGFkfdtSzbtv2SHwSCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIJxmZvYeAUflB4EgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoHwBl7zFI9axWTDAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives5929972347266019934,#minihistogram5929972347266019934"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives5929972347266019934">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles5929972347266019934"
                                                  aria-controls="quantiles5929972347266019934" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram5929972347266019934" aria-controls="histogram5929972347266019934"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common5929972347266019934" aria-controls="common5929972347266019934"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme5929972347266019934" aria-controls="extreme5929972347266019934"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles5929972347266019934">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>2</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>6</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>7</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>7</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>6</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>4</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>2.0001</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.50002</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.25</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>1.7143</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>60480</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>4.0003</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram5929972347266019934">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3X1UlHX%2B//EXMEIpAo4GblpiaiRCd4q05K4mrZama2qhbSmtlTcUyeqGeXPsxg074ppHPCXuqV03T81mbireZKZ281t3z3ZnI5qrpCdjNUiZCIIUmN8ffp1twrvyw3XNTM/HOZ6zfS6Yec%2BHa/Y8nbkYw7xer1cAAAAwJtzuAQAAAEINgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGCYw%2B4BfioqK782fpvh4WFyOtvo2LFaNTV5jd9%2BKGCPzo79OTf26NzYo7Njf86tJffokkvaGr2988UrWEEsPDxMYWFhCg8Ps3uUgMUenR37c27s0bmxR2fH/pxbKO4RgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGAYgQUAAGCYw%2B4BcGH6zNpk9wgha%2BPUG%2B0e4Qe59Zn/Z/cI5y3Y9pbnWcsJtnMhmJ5nwea9P9xi9whG8QoWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYUEbWOXl5crJyVF6eroyMjI0Y8YMVVdXS5L27Nmju%2B%2B%2BW71799agQYP0/PPP%2B33vhg0bNGzYMF133XUaOXKk3n33Xd%2BxpqYmLVq0SJmZmUpLS9OECRN06NAhSx8bAAAIbkEbWJMmTVJMTIy2bt2q1atXa9%2B%2BfXr66adVX1%2BviRMn6oYbbtA777yjRYsWadmyZdq8ebOkk/GVn5%2Bv6dOn65///Keys7P14IMP6siRI5KklStXat26dSouLta2bduUmJionJwceb1eOx8uAAAIIkEZWNXV1UpJSdG0adPUpk0bdezYUbfffrvee%2B89bd%2B%2BXSdOnNDkyZPVunVr9erVS3fccYdcLpck6ZVXXlH//v3Vv39/RUVFafjw4bryyiu1du1aSZLL5VJ2dra6deum6Oho5eXlqaysTDt37rTzIQMAgCASlIEVExOjgoICdejQwbd2%2BPBhxcfHq7S0VElJSYqIiPAdS05O1q5duyRJpaWlSk5O9ru95ORkud1u1dfXa//%2B/X7Ho6Oj1aVLF7nd7hZ%2BVAAAIFQ47B7ABLfbrRdffFHPPvusNm7cqJiYGL/jcXFx8ng8ampqksfjUWxsrN/x2NhY7d%2B/X1999ZW8Xu9pj1dVVZ33PBUVFaqsrPRbczhaKz4%2B/gc%2BsrOLiAjKPg4aDgf721KCaW95nrWsYDoX0PJC6fkW9IH1/vvva/LkyZo2bZoyMjK0cePG035dWFiY73%2Bf63qqC73eyuVyqaioyG8tJydHubm5F3S7sFa7dm3sHiFksbc4hXMB3xUTc7HdIxgT1IG1detW/f73v9ecOXM0YsQISZLT6dTBgwf9vs7j8SguLk7h4eFq166dPB5Ps%2BNOp9P3Nac73r59%2B/OeKysrSwMHDvRbczhaq6qq9gc8unMLpdIPRKZ/XvifYNpbnmctK5jOBbS86uo6NTY2Gb1NuyI%2BaAPrgw8%2BUH5%2BvhYvXqx%2B/fr51lNSUvTSSy%2BpoaFBDsfJh%2Bd2u3XNNdf4jp%2B6HusUt9utoUOHKioqSj169FBpaan69u0r6eQF9Z999pmuvvrq854tPj6%2B2duBlZVfq6HB7EmDlsXPq%2BWwtziFcwHf1djYFDLnRFD%2B1ayhoUGzZ8/W9OnT/eJKkvr376/o6Gg9%2B%2Byzqqur086dO7Vq1SqNHTtWknTnnXfqH//4h7Zv365vv/1Wq1at0sGDBzV8%2BHBJ0tixY7VixQqVlZWppqZGhYWF6tmzp1JTUy1/nAAAIDgF5StYH330kcrKyjRv3jzNmzfP79imTZv03HPPae7cuSouLlaHDh2Ul5enAQMGSJKuvPJKFRYWqqCgQOXl5erevbuWLVumSy65RJI0ZswYVVZW6p577lFtba3S09ObXU8FAABwNkEZWH369NHevXvP%2BjUvvfTSGY8NGjRIgwYNOu2xsLAw5ebmckE6AAD40YLyLUIAAIBARmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYFtSB9c477ygjI0N5eXl%2B66tXr9ZVV12l1NRUvz8ff/yxJKmpqUmLFi1SZmam0tLSNGHCBB06dMj3/R6PR1OnTlVGRob69eunWbNmqb6%2B3tLHBgAAglfQBtby5cs1b948denS5bTH09LS5Ha7/f5cffXVkqSVK1dq3bp1Ki4u1rZt25SYmKicnBx5vV5J0pw5c1RXV6eSkhK9%2BuqrKisrU2FhoWWPDQAABLegDayoqCitWrXqjIF1Ni6XS9nZ2erWrZuio6OVl5ensrIy7dy5U19%2B%2BaW2bNmivLw8OZ1OJSQkaMqUKXr11Vd14sSJFngkAAAg1DjsHuDHGjdu3FmPHz58WPfee6927dqlmJgY5ebm6te//rXq6%2Bu1f/9%2BJScn%2B742OjpaXbp0kdvt1tdff62IiAglJSX5jvfq1UvffPONPv30U7/1M6moqFBlZaXfmsPRWvHx8T/wUZ5dRETQ9nFQcDjY35YSTHvL86xlBdO5gJYXSs%2B3oA2ss3E6nUpMTNTvfvc7de/eXW%2B88YYeeeQRxcfH64orrpDX61VsbKzf98TGxqqqqkpxcXGKjo5WWFiY3zFJqqqqOq/7d7lcKioq8lvLyclRbm7uBT4yWKlduzZ2jxCy2FucwrmA74qJudjuEYwJycAaMGCABgwY4PvvoUOH6o033tDq1as1ffp0SfJdb3U6Zzt2PrKysjRw4EC/NYejtaqqai/odr8vlEo/EJn%2BeeF/gmlveZ61rGA6F9Dyqqvr1NjYZPQ27Yr4kAys0%2BnUqZN27dqluLg4hYeHy%2BPx%2BB33eDxq3769nE6nampq1NjYqIiICN8xSWrfvv153Vd8fHyztwMrK79WQ4PZkwYti59Xy2FvcQrnAr6rsbEpZM6JkPyr2UsvvaQNGzb4rZWVlemyyy5TVFSUevToodLSUt%2Bx6upqffbZZ7r66qvVs2dPeb1effLJJ77jbrdbMTEx6tq1q2WPAQAABK%2BQDKzjx4/rySeflNvt1okTJ1RSUqK3335bY8aMkSSNHTtWK1asUFlZmWpqalRYWKiePXsqNTVVTqdTgwcP1jPPPKNjx47pyJEjWrp0qUaPHi2H4yfzgh8AALgAQVsMqampkqSGhgZJ0pYtWySdfLVp3Lhxqq2t1cMPP6zKykp17txZS5cuVUpKiiRpzJgxqqys1D333KPa2lqlp6f7XZT%2BxBNPaO7cucrMzFSrVq102223NfswUwAAgDMJ2sByu91nPBYWFqYpU6ZoypQpZzyem5t7xt/qa9u2rf74xz8amRMAAPz0hORbhAAAAHYisAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAyzPLAGDhyooqIiHT582Oq7BgAAsITlgTVq1Cht2LBBN998s%2B677z5t3rxZDQ0NVo8BAADQYiwPrJycHG3YsEF/%2B9vf1KNHDz311FPq37%2B/FixYoAMHDlg9DgAAgHG2XYPVq1cv5efna9u2bZo5c6b%2B9re/aciQIZowYYI%2B/vhju8YCAAC4YLYF1okTJ7Rhwwbdf//9ys/PV0JCgh599FH17NlT2dnZWrdunV2jAQAAXBCH1XdYVlamVatW6bXXXlNtba0GDx6sv/zlL%2Brdu7fva9LS0vTYY49p2LBhVo8HAABwwSwPrKFDh6pr166aOHGiRowYobi4uGZf079/fx07dszq0QAAAIywPLBWrFihvn37nvPrdu7cacE0AAAA5ll%2BDVZSUpImTZqkLVu2%2BNb%2B/Oc/6/7775fH47F6HAAAAOMsD6yCggJ9/fXX6t69u29twIABampq0vz5860eBwAAwDjL3yJ89913tW7dOrVr1863lpiYqMLCQt12221WjwMAAGCc5a9g1dfXKyoqqvkg4eGqq6uzehwAAADjLA%2BstLQ0zZ8/X1999ZVv7YsvvtDjjz/u91ENAAAAwcrytwhnzpyp3/72t/r5z3%2Bu6OhoNTU1qba2Vpdddpn%2B%2Bte/Wj0OAACAcZYH1mWXXab169fr7bff1meffabw8HB17dpV/fr1U0REhNXjAAAAGGd5YElSZGSkbr75ZjvuGgAAoMVZHliHDh3SwoULtW/fPtXX1zc7/uabb1o9EgAAgFG2XINVUVGhfv36qXXr1lbfPQAAQIuzPLB27dqlN998U06n0%2Bq7BgAAsITlH9PQvn17XrkCAAAhzfLAmjhxooqKiuT1eq2%2BawAAAEtY/hbh22%2B/rQ8%2B%2BECrV69W586dFR7u33gvv/yy1SMBAAAYZXlgRUdH65e//KXVdwsAAGAZywOroKDA6rsEAACwlOXXYEnSp59%2BqiVLlujRRx/1rX344Yd2jAIAAGCc5YG1Y8cODR8%2BXJs3b1ZJSYmkkx8%2BOm7cOD5kFAAAhATLA2vRokX6/e9/r3Xr1iksLEzSyX%2BfcP78%2BVq6dKnV4wAAABhneWD95z//0dixYyXJF1iSdMstt6isrMzqcQAAAIyzPLDatm172n%2BDsKKiQpGRkVaPAwAAYJzlgXX99dfrqaeeUk1NjW/twIEDys/P189//nOrxwEAADDO8o9pePTRRzV%2B/Hilp6ersbFR119/verq6tSjRw/Nnz/f6nEAAACMszywOnbsqJKSEr311ls6cOCALrroInXt2lU33nij3zVZAAAAwcrywJKkVq1a6eabb7bjrgEAAFqc5YE1cODAs75SxWdhAQCAYGd5YA0ZMsQvsBobG3XgwAG53W6NHz/e6nEAAACMszywpk%2Bfftr1119/Xf/6178sngYAAMA8W/4twtO5%2BeabtX79ervHAAAAuGABE1i7d%2B%2BW1%2Bu1ewwAAIALZvlbhGPGjGm2VldXp7KyMg0aNMjqcQAAAIyzPLASExOb/RZhVFSURo8erTvuuMPqcQAAAIyzPLD4tHYAABDqLA%2Bs11577by/dsSIES04CQAAQMuwPLBmzZqlpqamZhe0h4WF%2Ba2FhYURWAAAIChZHlh/%2BtOf9Pzzz2vSpElKSkqS1%2BvV3r17tXz5ct19991KT0%2B3eiQAAACjbLkGq7i4WAkJCb61Pn366LLLLtOECRNUUlJi9UgAAABGWf45WAcPHlRsbGyz9ZiYGJWXl1s9DgAAgHGWB1anTp00f/58VVVV%2Bdaqq6u1cOFCXX755VaPAwAAYJzlgTVz5kxt3LhRGRkZ6tOnj/r27asbbrhBq1ev1owZM37Qbb3zzjvKyMhQXl5es2MbNmzQsGHDdN1112nkyJF69913fceampq0aNEiZWZmKi0tTRMmTNChQ4d8xz0ej6ZOnaqMjAz169dPs2bNUn19/Y9/0AAA4CfF8muw%2BvXrp%2B3bt%2Butt97SkSNH5PV6lZCQoF/84hdq27bted/O8uXLtWrVKnXp0qXZsT179ig/P19FRUW64YYb9Prrr%2BvBBx/Upk2b1LFjR61cuVLr1q3T8uXLlZCQoEWLFiknJ0dr1qxRWFiY5syZo%2BPHj6ukpEQnTpzQww8/rMLCQs2ePdvkVgAAgBBly79FePHFFyszM1OZmZm69957NWTIkB8UV9LJT38/U2C98sor6t%2B/v/r376%2BoqCgNHz5cV155pdauXStJcrlcys7OVrdu3RQdHa28vDyVlZVp586d%2BvLLL7Vlyxbl5eXJ6XQqISFBU6ZM0auvvqoTJ04YefwAACC0WR5Y9fX1ys/P13XXXadbb71V0slrsO677z5VV1ef9%2B2MGzfujFFWWlqq5ORkv7Xk5GS53W7V19dr//79fsejo6PVpUsXud1u7dmzRxEREUpKSvId79Wrl7755ht9%2BumnP%2BShAgCAnyjL3yJcsGCB9uzZo8LCQj3yyCO%2B9cbGRhUWFuqJJ5644PvweDzNflMxNjZW%2B/fv11dffSWv13va41VVVYqLi1N0dLTfv5d46mu/e2H%2B2VRUVKiystJvzeForfj4%2BB/zcM4oIsKWFyB/MhwO9relBNPe8jxrWcF0LqDlhdLzzfLAev311/Xiiy8qMTFR%2Bfn5kk5%2BRENBQYFGjBhhJLAkNfuk%2BB9y/Fzfey4ul0tFRUV%2Bazk5OcrNzb2g24W12rVrY/cIIYu9xSmcC/iumJiL7R7BGMsDq7a2VomJic3WnU6nvvnmGyP30a5dO3k8Hr81j8cjp9OpuLg4hYeHn/Z4%2B/bt5XQ6VVNTo8bGRkVERPiOSVL79u3P6/6zsrI0cOBAvzWHo7Wqqmp/7EM6rVAq/UBk%2BueF/wmmveV51rKC6VxAy6uurlNjY5PR27Qr4i0PrMsvv1z/%2Bte/lJ6e7vdK0aZNm3TppZcauY%2BUlBTt2rXLb83tdmvo0KGKiopSjx49VFpaqr59%2B0o6eQ3YZ599pquvvlqdOnWS1%2BvVJ598ol69evm%2BNyYmRl27dj2v%2B4%2BPj2/2dmBl5ddqaDB70qBl8fNqOewtTuFcwHc1NjaFzDlh%2BV/N7rrrLj300EN6%2Bumn1dTUpBdeeEHTpk3TzJkzNX78eCP3ceedd%2Bof//iHtm/frm%2B//VarVq3SwYMHNXz4cEnS2LFjtWLFCpWVlammpkaFhYXq2bOnUlNT5XQ6NXjwYD3zzDM6duyYjhw5oqVLl2r06NFyOCzvUQAAEIQsL4asrCw5HA69%2BOKLioiI0HPPPaeuXbuqsLBQt9xyy3nfTmpqqiSpoaFBkrRlyxZJJ19tuvLKK1VYWKiCggKVl5ere/fuWrZsmS655BJJ0pgxY1RZWal77rlHtbW1Sk9P97tm6oknntDcuXOVmZmpVq1a6bbbbjvth5kCAACcjuWBdezYMY0aNUqjRo26oNtxu91nPT5o0CANGjTotMfCwsKUm5t7xovO27Ztqz/%2B8Y8XNB8AAPjpsvwtwszMzAv%2BLT0AAIBAZnlgpaena%2BPGjVbfLQAAgGUsf4vwZz/7mf7whz%2BouLhYl19%2BuVq1auV3fOHChVaPBAAAYJTlgbV//35dccUVks7/k9EBAACCiWWBlZeXp0WLFumvf/2rb23p0qXKycmxagQAAABLWHYN1tatW5utFRcXW3X3AAAAlrEssE73m4P8NiEAAAhFlgVWWFjYea0BAAAEO/4VUwAAAMMILAAAAMMs%2By3CEydOaNq0aedc43OwAABAsLMssHr37q2KiopzrgEAAAQ7ywLru59/BQAAEMq4BgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMCwkA2spKQkpaSkKDU11ffnySeflCTt2LFDo0eP1vXXX6%2BhQ4dq7dq1ft%2B7YsUKDR48WNdff73Gjh2rXbt22fEQAABAkHLYPUBL2rRpkzp37uy3VlFRoSlTpmjWrFkaNmyY3n//fU2ePFldu3ZVamqqtm7dqiVLluhPf/qTkpKStGLFCk2aNEmbN29W69atbXokAAAgmITsK1hnsm7dOiUmJmr06NGKiopSRkaGBg4cqFdeeUWS5HK5NHLkSF1zzTW66KKLdN9990mStm3bZufYAAAgiIT0K1gLFy7Uhx9%2BqJqaGt16662aMWOGSktLlZyc7Pd1ycnJ2rhxoySptLRUQ4YM8R0LDw9Xz5495Xa7NXTo0PO634qKClVWVvqtORytFR8ff4GPyF9ExE%2Bujy3lcLC/LSWY9pbnWcsKpnMBLS%2BUnm8hG1jXXnutMjIy9PTTT%2BvQoUOaOnWqHn/8cXk8HiUkJPh9bVxcnKqqqiRJHo9HsbGxfsdjY2N9x8%2BHy%2BVSUVGR31pOTo5yc3N/5KOBHdq1a2P3CCGLvcUpnAv4rpiYi%2B0ewZiQDSyXy%2BX73926ddP06dM1efJk9e7d%2B5zf6/V6L%2Bi%2Bs7KyNHDgQL81h6O1qqpqL%2Bh2vy%2BUSj8Qmf554X%2BCaW95nrWsYDoX0PKqq%2BvU2Nhk9DbtiviQDazv69y5sxobGxUeHi6Px%2BN3rKqqSk6nU5LUrl27Zsc9Ho969Ohx3vcVHx/f7O3Aysqv1dBg9qRBy%2BLn1XLYW5zCuYDvamxsCplzIiT/arZ7927Nnz/fb62srEyRkZHq379/s49d2LVrl6655hpJUkpKikpLS33HGhsbtXv3bt9xAACAcwnJwGrfvr1cLpeKi4t1/PhxHThwQIsXL1ZWVpZ%2B/etfq7y8XK%2B88oq%2B/fZbvfXWW3rrrbd05513SpLGjh2r1157TR999JHq6ur07LPPKjIyUgMGDLD3QQEAgKARkm8RJiQkqLi4WAsXLvQF0u233668vDxFRUVp2bJlmjdvnh5//HF16tRJCxYs0FVXXSVJ%2BuUvf6nf/e53mjp1qo4eParU1FQVFxfroosusvlRAQCAYBGSgSVJaWlpevnll894bM2aNWf83rvuukt33XVXS40GAABCXEi%2BRQgAAGAnAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAus0ysvL9cADDyg9PV033XSTFixYoKamJrvHAgAAQcJh9wCB6KGHHlKvXr20ZcsWHT16VBMnTlSHDh1077332j0aAAAIAryC9T1ut1uffPKJpk%2BfrrZt2yoxMVHZ2dlyuVx2jwYAAIIEr2B9T2lpqTp16qTY2FjfWq9evXTgwAHV1NQoOjr6nLdRUVGhyspKvzWHo7Xi4%2BONzhoRQR%2B3JIeD/W0pwbS3PM9aVjCdC2h5ofR8I7C%2Bx%2BPxKCYmxm/tVGxVVVWdV2C5XC4VFRX5rT344IN66KGHzA2qkyE3vuM%2BZWVlGY%2B3UFFRUSGXy/WT2KP3/nDLD/6en9L%2B/Fg8z87tp3Qe8TxrGRUVFVqyZElI7VHopKJBXq/3gr4/KytLq1ev9vuTlZVlaLr/qaysVFFRUbNXy/A/7NHZsT/nxh6dG3t0duzPuYXiHvEK1vc4nU55PB6/NY/Ho7CwMDmdzvO6jfj4%2BJApcAAA8MPxCtb3pKSk6PDhwzp27Jhvze12q3v37mrTpo2NkwEAgGBBYH1PcnKyUlNTtXDhQtXU1KisrEwvvPCCxo4da/doAAAgSEQ89thjj9k9RKD5xS9%2BoZKSEj355JNav369Ro8erQkTJigsLMzu0Zpp06aN%2Bvbty6trZ8EenR37c27s0bmxR2fH/pxbqO1RmPdCr%2BgGAACAH94iBAAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAClLvvPOOMjIylJeXZ/coAam8vFw5OTlKT09XRkaGZsyYoerqarvHCiiffPKJxo8fr969eysjI0NTp05VZWWl3WMFrKeeekpJSUl2jxFQkpKSlJKSotTUVN%2BfJ5980u6xAs6zzz6rfv366dprr1V2drY%2B//xzu0cKGP/%2B97/9zp/U1FR3D8s%2BAAAFMUlEQVSlpKSExHONwApCy5cv17x589SlSxe7RwlYkyZNUkxMjLZu3arVq1dr3759evrpp%2B0eK2AcP35cv/3tb9W3b1/t2LFDJSUlOnr0qPinSU9vz549WrNmjd1jBKRNmzbJ7Xb7/syZM8fukQLKypUrtXbtWq1YsULvvvuuunfvrj//%2Bc92jxUw0tLS/M4ft9utBx98ULfeeqvdo10wAisIRUVFadWqVQTWGVRXVyslJUXTpk1TmzZt1LFjR91%2B%2B%2B1677337B4tYNTV1SkvL08TJ05UZGSknE6nfvWrX2nfvn12jxZwmpqaNHfuXGVnZ9s9CoLQ888/r7y8PF1xxRWKjo7W7NmzNXv2bLvHClj//e9/9cILL%2BiRRx6xe5QLRmAFoXHjxqlt27Z2jxGwYmJiVFBQoA4dOvjWDh8%2BrPj4eBunCiyxsbG644475HA4JEmffvqp/v73v4fE3xpNe/nllxUVFaVhw4bZPUpAWrhwoQYMGKA%2Bffpozpw5qq2ttXukgPHFF1/o888/11dffaUhQ4YoPT1dubm5OnbsmN2jBazFixdr1KhRuvTSS%2B0e5YIRWAh5brdbL774oiZPnmz3KAGnvLxcKSkpGjJkiFJTU5Wbm2v3SAHlyy%2B/1JIlSzR37ly7RwlI1157rTIyMrR582a5XC599NFHevzxx%2B0eK2AcOXJE0sm3UV944QWtWbNGR44c4RWsM/j888%2B1efNm3XvvvXaPYgSBhZD2/vvva8KECZo2bZoyMjLsHifgdOrUSW63W5s2bdLBgwdD4mV5kwoKCjRy5Eh1797d7lECksvl0h133KHIyEh169ZN06dPV0lJiY4fP273aAHB6/VKku677z4lJCSoY8eOeuihh7R161Z9%2B%2B23Nk8XeFauXKlBgwbpkksusXsUIwgshKytW7fqgQce0MyZMzVu3Di7xwlYYWFhSkxMVF5enkpKSnj74v/s2LFDH374oXJycuweJWh07txZjY2NOnr0qN2jBIRTlynExMT41jp16iSv18sencbrr7%2BugQMH2j2GMQQWQtIHH3yg/Px8LV68WCNGjLB7nICzY8cODR48WE1NTb618PCT/3fQqlUru8YKKGvXrtXRo0d10003KT09XSNHjpQkpaena/369TZPZ7/du3dr/vz5fmtlZWWKjIzkesf/07FjR0VHR2vPnj2%2BtfLycrVq1Yo9%2Bp49e/aovLxcN954o92jGOOwewDAtIaGBs2ePVvTp09Xv3797B4nIKWkpKimpkYLFixQbm6u6urqtGTJEvXp04dfoPg/M2bM0MMPP%2Bz77yNHjigrK0tr1qxRbGysjZMFhvbt28vlcsnpdCo7O1vl5eVavHixsrKyFBERYfd4AcHhcGj06NF67rnnlJaWpujoaC1dulTDhg3z/YIJTtq9e7fi4uIUHR1t9yjGhHlPvUmMoJGamirpZEhI8j1R3W63bTMFkvfee0%2B/%2Bc1vFBkZ2ezYpk2b1KlTJxumCjx79%2B7VvHnz9PHHH6t169a64YYbNGPGDCUkJNg9WkD6/PPPlZmZqb1799o9SsD497//rYULF2rv3r2KjIzU7bffrry8PEVFRdk9WsA4fvy4CgoKtH79ep04cUKDBw/WnDlz1KZNG7tHCyjLli3TunXrVFJSYvcoxhBYAAAAhnENFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGH/H7nv%2BjV72eWhAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common5929972347266019934">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">7</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme5929972347266019934">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">3</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7</td>
        <td class="number">2160</td>
        <td class="number">14.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Elevation">Elevation<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>1665</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>11.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>2749.3</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1863</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>3849</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-2338006694815494869">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAATFJREFUeJzt3cFNxDAQQFEWURJF0BNneqIIejINoC8WKZshfu8eyZHy5YkPyW2ttZ6AHz2fvQCY7OXsBZzl9f3z7mu%2BPt4OWAmTbRvII4jw/zNiQRAIBIFAEAgEgUAQCATHvBty/Px7AhnGwzuLEQuCQCAIBMIl3kH%2BMrdfye73f6RLBPIoHsT9GLEgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgjPuyoq8XMokdBIJAIIwbsZhp1z9f2UEgCASCEYvDXGEsu6211tmLgKmMWBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBC%2BAXV1IDuq9Y7uAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-2338006694815494869,#minihistogram-2338006694815494869"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-2338006694815494869">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-2338006694815494869"
                                                  aria-controls="quantiles-2338006694815494869" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-2338006694815494869" aria-controls="histogram-2338006694815494869"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-2338006694815494869" aria-controls="common-2338006694815494869"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-2338006694815494869" aria-controls="extreme-2338006694815494869"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-2338006694815494869">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1863</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>2117</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>2376</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>2752</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>3104</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3397</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>3849</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>1986</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>728</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>417.68</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.15192</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.0821</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>2749.3</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>356.67</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.07564</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>41569757</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>174460</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-2338006694815494869">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl8Tfe%2B//F3BqJEQrThXlUcU2UyNxraGIqqckqN50GrtEVTSmmp4ZwOWlTcVEUVHU612gZ1TTVVDS112ns66JbiHMGljp5EZQshZPj%2B/ui1f7YYor72zt5ez8cjD3y/a%2B39%2BWTtLO%2BstfbaAcYYIwAAAFgT6O0CAAAA/A0BCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYFuztAm4UWVknvF2CNYGBAYqIqKBjx3JVVGS8XY5V9Oab6M03%2BWtv/tqX5Ju93XJLRa88L0ewcNUCAwMUEBCgwMAAb5diHb35JnrzTf7am7/2Jfl3b7YRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABY5rMB6/Dhw0pKSlJ8fLwSEhI0btw45eTk6Oeff1aDBg0UGxvr9vX222%2B71l29erW6du2qJk2aqEePHtq6datrrqioSCkpKWrfvr1atGihwYMH69ChQ95oEQAA%2BCifDVhDhw5VWFiYNm7cqKVLl%2Bqf//ynpk2b5pp3OBxuX4MHD5Yk7dq1S2PHjtWYMWP0t7/9TQMHDtSTTz6pX375RZK0cOFCrVy5UvPmzdOmTZtUq1YtJSUlyRjf%2BFBLAADgfT4ZsHJychQTE6PRo0erQoUKqlatmrp3766///3vV1x38eLFSkxMVGJiokJCQtStWzfVr19fK1askCSlpaVp4MCBqlOnjkJDQzVq1ChlZGRox44d17stAADgJ4K9XcDvERYWpilTpriNHTlyRJGRka5/P/vss/rqq69UUFCgXr16acSIESpTpozS09OVmJjotm5UVJQcDofy8vK0d%2B9eRUVFueZCQ0NVs2ZNORwONW7cuET1ZWZmKisry20sOLi8W32%2BLCgo0O1Pf3J%2Bbx2Sv/RyNVfnszF3XXb%2BRtlu/obefI%2B/9iX5d2%2B2%2BWTAupDD4dAHH3ygOXPmqGzZsmrSpIk6dOigl19%2BWbt27dLw4cMVHBysp556Sk6nU%2BHh4W7rh4eHa%2B/evTp%2B/LiMMRedz87OLnE9aWlpSk1NdRtLSkrSiBEjfn%2BTpVBY2E3eLuG68cXeKleuUKLlfLG3kqI33%2BSvvflrX5J/92aLzwesb7/9VsOGDdPo0aOVkJAgSfr4449d83FxcRoyZIjmzp2rp556SpKueD3VtV5v1adPH7Vr185tLDi4vLKzc6/pcUuLoKBAhYXdpJyc0yosLPJ2OVad35uvudLr60bZbvTmO/y1N3/tS/LN3kr6y6dtPh2wNm7cqGeeeUaTJk3SAw88cMnlqlevrqNHj8oYo8qVK8vpdLrNO51ORUREqFKlSgoMDLzofJUqVUpcV2RkZLHTgVlZJ1RQ4BsvxpIqLCzyu57O8ZUdx/lKui38fbvRm%2B/x1978tS/Jv3uzxWdPon733XcaO3asZs6c6Rautm/frjlz5rgtu2/fPlWvXl0BAQGKiYnRzp073eYdDocaNWqkkJAQ1atXT%2Bnp6a65nJwcHTx4UHFxcde3IQAA4Dd8MmAVFBRo4sSJGjNmjFq3bu02V7FiRc2ePVvLly9Xfn6%2BHA6H3n77bfXr10%2BS1Lt3b3311VfavHmzzpw5oyVLlujAgQPq1q2bJKlfv35asGCBMjIydPLkSSUnJ6thw4aKjY31eJ8AAMA3%2BeQpwh9%2B%2BEEZGRmaPHmyJk%2Be7Da3du1apaSkKDU1VX/%2B859VsWJFDRgwQA8//LAkqX79%2BkpOTtaUKVN0%2BPBh1a1bV3PnztUtt9wiSerbt6%2BysrI0YMAA5ebmKj4%2BvtgF6wCuTefXtnm7hKuyZmQrb5cAwMf4ZMBq3ry59uzZc8n56tWrq0OHDpec79ixozp27HjRuYCAAI0YMcLv3vEHAAA8xydPEQIAAJRmBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYFe7sAAMCNq/mEtd4u4aqsGdnK2yXAR3AECwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACzz2YB1%2BPBhJSUlKT4%2BXgkJCRo3bpxycnIkSbt27VL//v3VrFkzdezYUe%2B8847buqtXr1bXrl3VpEkT9ejRQ1u3bnXNFRUVKSUlRe3bt1eLFi00ePBgHTp0yKO9AQAA3%2Bazt2kYOnSoYmJitHHjRp04cUJJSUmaNm2aJk2apCFDhqh3796aN2%2Be9u/fr0GDBunWW29Vx44dtWvXLo0dO1apqalq2bKl1q1bpyeffFJr165VtWrVtHDhQq1cuVLz589X1apVlZKSoqSkJC1fvlwBAQHebhuAF3R%2BbZu3S7gq3EoA8D6fPIKVk5OjmJgYjR49WhUqVFC1atXUvXt3/f3vf9fmzZuVn5%2BvYcOGqXz58oqOjlavXr2UlpYmSVq8eLESExOVmJiokJAQdevWTfXr19eKFSskSWlpaRo4cKDq1Kmj0NBQjRo1ShkZGdqxY4c3WwYAAD7EJwNWWFiYpkyZoptvvtk1duTIEUVGRio9PV0NGjRQUFCQay4qKko7d%2B6UJKWnpysqKsrt8aKiouRwOJSXl6e9e/e6zYeGhqpmzZpyOBzXuSsAAOAvfPYU4fkcDoc%2B%2BOADzZkzR2vWrFFYWJjbfKVKleR0OlVUVCSn06nw8HC3%2BfDwcO3du1fHjx%2BXMeai89nZ2SWuJzMzU1lZWW5jwcHlFRkZeZWdlU5BQYFuf/oTX%2B4tOPjyNftyb7g6V3otlBa%2B%2BFosyffWn3/W/Lk323w%2BYH377bcaNmyYRo8erYSEBK1Zs%2Baiy51//ZQx5rKPeaX5K0lLS1NqaqrbWFJSkkaMGHFNj1vahIXd5O0Srhtf7K1y5QolWs4Xe8PVKelrAVfvar63/vyz5s%2B92eLTAWvjxo165plnNGnSJD3wwAOSpIiICB04cMBtOafTqUqVKikwMFCVK1eW0%2BksNh8REeFa5mLzVapUKXFdffr0Ubt27dzGgoPLKzs79yq6K72CggIVFnaTcnJOq7CwyNvlWHV%2Bb77mSq8vf95ucOcr%2BxpfPApSku%2BtP/%2Bs%2BWJv3vqFw2cD1nfffaexY8dq5syZat26tWs8JiZGH330kQoKChQc/Ft7DodDjRo1cs2fux7rHIfDoS5duigkJET16tVTenq67rjjDkm/XVB/8OBBxcXFlbi2yMjIYqcDs7JOqKDAN16MJVVYWOR3PZ3jKzuO85V0W/jzdsNv2L7Xz9V8b/35Z82fe7PF9359kFRQUKCJEydqzJgxbuFKkhITExUaGqo5c%2Bbo9OnT2rFjh5YsWaJ%2B/fpJknr37q2vvvpKmzdv1pkzZ7RkyRIdOHBA3bp1kyT169dPCxYsUEZGhk6ePKnk5GQ1bNhQsbGxHu8TAAD4Jp88gvXDDz8oIyNDkydP1uTJk93m1q5dqzfffFN/%2BctfNG/ePN18880aNWqU2rRpI0mqX7%2B%2BkpOTNWXKFB0%2BfFh169bV3Llzdcstt0iS%2Bvbtq6ysLA0YMEC5ubmKj48vdj0VAADA5fhkwGrevLn27Nlz2WU%2B%2BuijS8517NhRHTt2vOhcQECARowY4XcXpAMAAM/xyVOEAAAApRkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwLNjbBeDG0fm1bd4uAQAAj%2BAIFgAAgGUELAAAAMs4RQgAfobT8YD3cQQLAADAMgIWAACAZZwiBPwEp4UAoPTgCBYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFjm8YDVrl07paam6siRI55%2BagAAAI/weMB68MEHtXr1at1zzz169NFHtX79ehUUFHi6DAAAgOvG4wErKSlJq1ev1qJFi1SvXj298sorSkxM1PTp07V//35PlwMAAGCd167Bio6O1tixY7Vp0yaNHz9eixYt0n333afBgwfrxx9/9FZZAAAA18xrASs/P1%2BrV6/WY489prFjx6pq1ap67rnn1LBhQw0cOFArV670VmkAAADXJNjTT5iRkaElS5Zo2bJlys3NVadOnfTee%2B%2BpWbNmrmVatGih559/Xl27dvV0eQAAANfM40ewunTpos2bN2vIkCH64osvNH36dLdwJUmJiYk6duzYFR/ryy%2B/VEJCgkaNGuU2vnTpUt1%2B%2B%2B2KjY11%2Bzp36rGoqEgpKSlq3769WrRoocGDB%2BvQoUOu9Z1Op0aOHKmEhAS1bt1aEyZMUF5enoXuAQDAjcDjR7AWLFigO%2B6444rL7dix47Lz8%2BfP15IlS1SzZs2Lzrdo0ULvv//%2BRecWLlyolStXav78%2BapatapSUlKUlJSk5cuXKyAgQJMmTdLZs2e1atUq5efn66mnnlJycrImTpx45QYBAMANz%2BNHsBo0aKChQ4dqw4YNrrG//vWveuyxx%2BR0Okv8OCEhIZcNWJeTlpamgQMHqk6dOgoNDdWoUaOUkZGhHTt26OjRo9qwYYNGjRqliIgIVa1aVU888YQ%2B%2BeQT5efnX/VzAQCAG4/HA9aUKVN04sQJ1a1b1zXWpk0bFRUVaerUqSV%2BnIceekgVK1a85PyRI0f0yCOPqEWLFmrfvr2WL18uScrLy9PevXsVFRXlWjY0NFQ1a9aUw%2BHQrl27FBQUpAYNGrjmo6OjderUKe3bt%2B9qWgUAADcoj58i3Lp1q1auXKnKlSu7xmrVqqXk5GTdf//9Vp4jIiJCtWrV0tNPP626devqs88%2B07PPPqvIyEj94Q9/kDFG4eHhbuuEh4crOztblSpVUmhoqAICAtzmJCk7O7tEz5%2BZmamsrCy3seDg8oqMjLzGzkqHoKBAtz8B4EYRHHzl/Z4/7yP9uTfbPB6w8vLyFBISUmw8MDBQp0%2BftvIcbdq0UZs2bVz/7tKliz777DMtXbpUY8aMkSQZYy65/uXmSiItLU2pqaluY0lJSRoxYsQ1PW5pExZ2k7dLAACPqly5QomX9ed9pD/3ZovHA1aLFi00depUjR492nVk6N///remTZtW7N2ENlWvXl07d%2B5UpUqVFBgYWOx6L6fTqSpVqigiIkInT55UYWGhgoKCXHOSVKVKlRI9V58%2BfdSuXTu3seDg8srOzrXQifcFBQUqLOwm5eScVmFhkbfLAQCPKcl%2B3J/3kb7Y29WEYps8HrDGjx%2BvQYMG6c4771RoaKiKioqUm5urGjVqXPJdf1fro48%2BUnh4uO677z7XWEZGhmrUqKGQkBDVq1dP6enprncz5uTk6ODBg4qLi1P16tVljNHu3bsVHR0tSXI4HAoLC1Pt2rVL9PyRkZHFTgdmZZ1QQYFvvBhLqrCwyO96AoDLuZp9nj/vI/25N1s8HrBq1KihTz/9VF988YUOHjyowMBA1a5dW61bt3YdMbpWZ8%2Be1UsvvaQaNWro9ttv17p16/TFF19o0aJFkqR%2B/fpp3rx5uvvuu1W1alUlJyerYcOGio2NlSR16tRJr732mqZNm6azZ89q9uzZ6tmzp4KDPf7tAgAAPsgriaFs2bK65557rukxzoWhgoICSXLd9sHhcOihhx5Sbm6unnrqKWVlZenWW2/V7NmzFRMTI0nq27evsrKyNGDAAOXm5io%2BPt7tmqkXX3xRf/nLX9S%2BfXuVKVNG999/f7GbmQIAAFxKgLnWK7qv0qFDhzRjxgz985//vOjd0T///HNPluMxWVknvF2CNcHBgapcuYKys3Ov6hBx59e2XceqAOD6WzOy1RWX%2Bb37SF/gi73dcsulb%2Bl0PXnlGqzMzEy1bt1a5cuX9/TTAwAAXHceD1g7d%2B7U559/roiICE8/NQAAgEd4/E5hVapU4cgVAADwax4PWEOGDFFqauo138wTAACgtPL4KcIvvvhC3333nZYuXapbb71VgYHuGe/jjz/2dEkAAABWeTxghYaG6u677/b00wIAAHiMxwPWlClTPP2UAAAAHuWVj8Pet2%2BfZs2apeeee8419v3333ujFAAAAOs8HrC2b9%2Bubt26af369Vq1apWk324%2B%2BtBDD/ntTUYBAMCNxeMBKyUlRc8884xWrlypgIAASb99PuHUqVM1e/ZsT5cDAABgnccD1j/%2B8Q/169dPklwBS5LuvfdeZWRkeLocAAAA6zwesCpWrHjRzyDMzMxU2bJlPV0OAACAdR4PWE2bNtUrr7yikydPusb279%2BvsWPH6s477/R0OQAAANZ5/DYNzz33nB5%2B%2BGHFx8ersLBQTZs21enTp1WvXj1NnTrV0%2BUAAABY5/GAVa1aNa1atUpbtmzR/v37Va5cOdWuXVutWrVyuyYLAADAV3k8YElSmTJldM8993jjqQEAAK47jwesdu3aXfZIFffCAgAAvs7jAeu%2B%2B%2B5zC1iFhYXav3%2B/HA6HHn74YU%2BXAwAAYJ3HA9aYMWMuOr5u3Tp9/fXXHq4GAADAPq98FuHF3HPPPfr000%2B9XQYAAMA1KzUB66effpIxxttlAAAAXDOPnyLs27dvsbHTp08rIyNDHTt29HQ5AAAA1nk8YNWqVavYuwhDQkLUs2dP9erVy9PlAAAAWOfxgMXd2gEAgL/zeMBatmxZiZd94IEHrmMlAAAA14fHA9aECRNUVFRU7IL2gIAAt7GAgAACFgAA8EkeD1hvvfWW3nnnHQ0dOlQNGjSQMUZ79uzR/Pnz1b9/f8XHx3u6JAAAAKu8cg3WvHnzVLVqVddY8%2BbNVaNGDQ0ePFirVq3ydEkAAABWefw%2BWAcOHFB4eHix8bCwMB0%2BfNjT5QAAAFjn8YBVvXp1TZ06VdnZ2a6xnJwczZgxQ7fddpunywEAALDO46cIx48fr9GjRystLU0VKlRQYGCgTp48qXLlymn27NmeLgcAAMA6jwes1q1ba/PmzdqyZYt%2B%2BeUXGWNUtWpV3XXXXapYsaKnywEAALDO4wFLkm666Sa1b99ev/zyi2rUqOGNEgAAAK4bj1%2BDlZeXp7Fjx6pJkybq3LmzpN%2BuwXr00UeVk5Pj6XIAAACs83jAmj59unbt2qXk5GQFBv7/py8sLFRycrKnywEAALDO4wFr3bp1ev3113Xvvfe6PvQ5LCxMU6ZM0fr16z1dDgAAgHUeD1i5ubmqVatWsfGIiAidOnXK0%2BUAAABY5/GAddttt%2Bnrr7%2BWJLfPHly7dq3%2B8z//09PlAAAAWOfxdxH%2B6U9/0vDhw/Xggw%2BqqKhI7777rnbu3Kl169ZpwoQJni4HAADAOo8HrD59%2Big4OFgffPCBgoKC9Oabb6p27dpKTk7Wvffe6%2BlyAAAArPN4wDp27JgefPBBPfjgg55%2BagAAAI/w%2BDVY7du3d7v2CgAAwN94PGDFx8drzZo1nn5aAAAAj/H4KcL/%2BI//0Msvv6x58%2BbptttuU5kyZdzmZ8yY4emSAAAArPJ4wNq7d6/%2B8Ic/SJKys7M9/fQAAADXnccC1qhRo5SSkqL333/fNTZ79mwlJSV5qgQAAACP8Ng1WBs3biw2Nm/ePE89PQAAgMd4LGBd7J2DvJsQAAD4I48FrHMf7HylMQAAAF/n8ds0AAAA%2BDufDlhffvmlEhISNGrUqGJzq1evVteuXdWkSRP16NFDW7dudc0VFRUpJSVF7du3V4sWLTR48GAdOnTINe90OjVy5EglJCSodevWmjBhgvLy8jzSEwAA8H0eexdhfn6%2BRo8efcWxkt4Ha/78%2BVqyZIlq1qxZbG7Xrl0aO3asUlNT1bJlS61bt05PPvmk1q5dq2rVqmnhwoVauXKl5s%2Bfr6pVqyolJUVJSUlavny5AgICNGnSJJ09e1arVq1Sfn6%2BnnrqKSUnJ2vixIm//xsAAABuGB47gtWsWTNlZma6fV1srKRCQkIuGbAWL16sxMREJSYmKiQkRN26dVP9%2BvW1YsUKSVJaWpoGDhyoOnXqKDQ0VKNGjVJGRoZ27Niho0ePasOGDRo1apQiIiJUtWpVPfHEE/rkk0%2BUn59v7fsBAAD8l8eOYJ1//ysbHnrooUvOpaenKzEx0W0sKipKDodDeXl52rt3r6KiolxzoaGhqlmzphwOh06cOKGgoCA1aNDANR8dHa1Tp05p3759buOXkpmZqaysLLex4ODyioyMLGl7pVpQUKDbnwBwowgOvvJ%2Bz5/3kf7cm20ev5O7JzidToWHh7uNhYeHa%2B/evTp%2B/LiMMRedz87OVqVKlRQaGur2Dsdzy5b0zvNpaWlKTU11G0tKStKIESN%2BTzulVljYTd4uAQA8qnLlCiVe1p/3kf7cmy1%2BGbCkK99j63Lz13p/rj59%2Bqhdu3ZuY8HB5ZWdnXtNj1taBAUFKizsJuXknFZhYZG3ywEAjynJftyf95G%2B2NvVhGKb/DJgVa5cWU6n023M6XQqIiJClSpVUmBg4EXnq1SpooiICJ08eVKFhYUKCgpyzUlSlSpVSvT8kZGRxU4HZmWdUEGBb7wYS6qwsMjvegKAy7mafZ4/7yP9uTdb/PIkakxMjHbu3Ok25nA41KhRI4WEhKhevXpKT093zeXk5OjgwYOKi4tTw4YNZYzR7t273dYNCwtT7dq1PdYDAADwXX4ZsHr37q2vvvpKmzdv1pkzZ7RkyRIdOHBA3bp1kyT169dPCxYsUEZGhk6ePKnk5GQ1bNhQsbGxioiIUKdOnfTaa6/p2LFj%2BuWXXzR79mz17NlTwcF%2BecAPAABY5rOJITY2VpJUUFAgSdqwYYOk34421a9fX8nJyZoyZYoOHz6sunXrau7cubrlllskSX379lVWVpYGDBig3NxcxcfHu12U/uKLL%2Bovf/mL2rdvrzJlyuj%2B%2B%2B%2B/6M1MAQAALibA8InLHpGVdcLbJVgTHByoypUrKDs796rOwXd%2Bbdt1rAoArr81I1tdcZnfu4/0Bb7Y2y23VPTK8/rlKUIAAABvImABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABY5rcBq0GDBoqJiVFsbKzr66WXXpIkbd%2B%2BXT179lTTpk3VpUsXrVixwm3dBQsWqFOnTmratKn69eunnTt3eqMFAADgo4K9XcD1tHbtWt16661uY5mZmXriiSc0YcIEde3aVd9%2B%2B62GDRum2rVrKzY2Vhs3btSsWbP01ltvqUGDBlqwYIGGDh2q9evXq3z58l7qBAAA%2BBK/PYJ1KStXrlStWrXUs2dPhYSEKCEhQe3atdPixYslSWlpaerRo4caNWqkcuXK6dFHH5Ukbdq0yZtlAwAAH%2BLXR7BmzJih77//XidPnlTnzp01btw4paenKyoqym25qKgorVmzRpKUnp6u%2B%2B67zzUXGBiohg0byuFwqEuXLiV63szMTGVlZbmNBQeXV2Rk5DV2VDoEBQW6/QkAN4rg4Cvv9/x5H%2BnPvdnmtwGrcePGSkhI0LRp03To0CGNHDlSL7zwgpxOp6pWreq2bKVKlZSdnS1JcjqdCg8Pd5sPDw93zZdEWlqaUlNT3caSkpI0YsSI39lN6RQWdpO3SwAAj6pcuUKJl/XnfaQ/92aL3wastLQ019/r1KmjMWPGaNiwYWrWrNkV1zXGXNNz9%2BnTR%2B3atXMbCw4ur%2Bzs3Gt63NIiKChQYWE3KSfntAoLi7xdDgB4TEn24/68j/TF3q4mFNvktwHrQrfeeqsKCwsVGBgop9PpNpedna2IiAhJUuXKlYvNO51O1atXr8TPFRkZWex0YFbWCRUU%2BMaLsaQKC4v8ricAuJyr2ef58z7Sn3uzxS9Pov7000%2BaOnWq21hGRobKli2rxMTEYrdd2Llzpxo1aiRJiomJUXp6umuusLBQP/30k2seAADgSvwyYFWpUkVpaWmaN2%2Bezp49q/3792vmzJnq06eP/vjHP%2Brw4cNavHixzpw5oy1btmjLli06mZhuAAAQy0lEQVTq3bu3JKlfv35atmyZfvjhB50%2BfVpz5sxR2bJl1aZNG%2B82BQAAfIZfniKsWrWq5s2bpxkzZrgCUvfu3TVq1CiFhIRo7ty5mjx5sl544QVVr15d06dP1%2B233y5Juvvuu/X0009r5MiR%2BvXXXxUbG6t58%2BapXLlyXu4KAAD4igBzrVd0o0Sysk54uwRrgoMDVblyBWVn517VOfjOr227jlUBwPW3ZmSrKy7ze/eRvsAXe7vllopeeV6/PEUIAADgTQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMv88rMIbyR8/AwAAKUPR7AAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwL9nYBAAD4is6vbfN2CVdlzchW3i7hhsURLAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAy4K9XQAAALg%2BOr%2B2zdsllNiaka28XYJVHMECAACwjIAFAABgGQELAADAMgLWRRw%2BfFiPP/644uPj1bZtW02fPl1FRUXeLgsAAPgILnK/iOHDhys6OlobNmzQr7/%2BqiFDhujmm2/WI4884u3SAACAD%2BAI1gUcDod2796tMWPGqGLFiqpVq5YGDhyotLQ0b5cGAAB8BEewLpCenq7q1asrPDzcNRYdHa39%2B/fr5MmTCg0NveJjZGZmKisry20sOLi8IiMjrdcLAIA/CA72r2M%2BBKwLOJ1OhYWFuY2dC1vZ2dklClhpaWlKTU11G3vyySc1fPhwe4X%2Bn7%2B/fK/1x7ySzMxMpaWlqU%2BfPn4XGunNN9Gbb/LX3vy1L8m/e7PNv%2BKiJcaYa1q/T58%2BWrp0qdtXnz59LFXnfVlZWUpNTS12lM4f0Jtvojff5K%2B9%2BWtfkn/3ZhtHsC4QEREhp9PpNuZ0OhUQEKCIiIgSPUZkZCTJHgCAGxhHsC4QExOjI0eO6NixY64xh8OhunXrqkKFCl6sDAAA%2BAoC1gWioqIUGxurGTNm6OTJk8rIyNC7776rfv36ebs0AADgI4Kef/75571dRGlz1113adWqVXrppZf06aefqmfPnho8eLACAgK8XVqpUaFCBd1xxx1%2BeVSP3nwTvfkmf%2B3NX/uS/Ls3mwLMtV7RDQAAADecIgQAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIB1gzt8%2BLCSkpIUHx%2BvhIQEjRs3Tjk5OZKkXbt2qX///mrWrJk6duyod955x23d1atXq2vXrmrSpIl69OihrVu3uuaKioqUkpKi9u3bq0WLFho8eLAOHTpUanrbvXu3Bg4cqObNm%2Bvuu%2B/Wyy%2B/rLNnz0qSvv76azVo0ECxsbFuX2vWrHE99oIFC9SpUyc1bdpU/fr1086dO0tFbz///PNFa3/77bdd6/rqdlu2bFmxvmJiYtSuXTtJ0tKlS3X77bcXW%2BbHH38sNb3t3r1bDz/8sJo1a6aEhASNHDlSWVlZkqTt27erZ8%2Beatq0qbp06aIVK1a4rXu519yZM2f05z//WXfffbfi4%2BM1YsQIZWdnl5revvnmG/Xp00dNmzZVu3bt9MYbb7jW8%2BXtdq37itK83d54441ifUVHR2vAgAGSpFmzZqlhw4bFljl69Gip6c3rDG5o999/vxk3bpw5efKkOXLkiOnRo4cZP368OX36tLnrrrvMrFmzTG5urtm5c6e54447zLp164wxxvz0008mJibGbN682eTl5Znly5ebRo0amSNHjhhjjFmwYIFp27at2bt3rzlx4oR58cUXTdeuXU1RUZHXezt58qRp1aqV%2Ba//%2Bi9z5swZs3fvXtO2bVsze/ZsY4wxf/vb30zbtm0v%2Bbiff/65ad68ufnhhx/M6dOnzdy5c02rVq1Mbm6up1q7ZG%2BHDh0y9evXv%2BR6vrzdLmbixInm1VdfNcYY88knn5j%2B/ftf8nG93duZM2fMnXfeaVJTU82ZM2fMr7/%2Bavr372%2BeeOIJ8%2B9//9s0btzYLF682OTl5Zlt27aZuLg48%2BOPPxpjrvyamzJliunRo4f517/%2BZbKzs82TTz5phgwZ4pG%2BrtTb4cOHTePGjc2HH35ozp49a3bs2GGaNWtmli1bZozx7e12rfuK0rzdLmbQoEFm4cKFxhhjXn/9dTN27NhLPra3eysNCFg3sOPHj5tx48aZrKws19j7779vOnbsaNasWWNatmxpCgoKXHPTp083gwYNMsYY88ILL5ikpCS3x%2BvVq5eZO3euMcaYLl26mPfee881d%2BLECRMVFWW%2B//7769mSy%2BV6%2B9///V8zbtw4k5%2Bf75qbOnWqeeSRR4wxVw5Yjz/%2BuHnllVdc/y4sLDStWrUyq1atug6dFHe53q4UsHx5u11ox44dpnXr1ubEiRPGmCv/R%2B3t3pxOp1m0aJHb6%2B69994zHTp0MG%2B99ZZ54IEH3JYfOXKkmTRpkjHm8q%2B5/Px806xZM7NhwwbX/N69e02DBg3ML7/8cp27%2Bs3letuxY4eZPHmy2/LDhw83EydONMb49na7ln1Fad9uF1qzZo3p2rWr6/%2BEywWs0tBbacApwhtYWFiYpkyZoptvvtk1duTIEUVGRio9PV0NGjRQUFCQay4qKsp1eDs9PV1RUVFujxcVFSWHw6G8vDzt3bvXbT40NFQ1a9aUw%2BG4zl395nK93XbbbZoyZYqCg4Pd5qpWrer6d25urus01V133aV3331X5v8%2BF/3C3gMDA9WwYcNS0ds5zz77rFq3bq2WLVtqxowZys/Pv2jtku9stwu9%2BuqrGjp0qEJDQ92WfeSRR9SiRQu1b99ey5cvl6RS0Vt4eLh69erlet3t27dP//3f/63OnTtfcrtc6uft/NfcwYMHdeLECUVHR7vm69Spo3Llyik9Pd0DnV2%2Bt7i4OE2YMMFt%2BQt/3nx1u0m/f19R2rfb%2BQoLC5WcnKzRo0e7/Z%2BwZ88e9e3b13Va%2B9zlBqWht9KAgAUXh8OhDz74QMOGDZPT6VRYWJjbfKVKleR0OlVUVCSn06nw8HC3%2BfDwcGVnZ%2Bv48eMyxlxy3hvO7%2B1Cn3/%2BuTZt2qRBgwZJ%2Bm0HXr9%2BfT388MP68ssvNWXKFKWmpuqTTz6RpMv27g3n91a2bFk1adJEHTp00KZNmzRv3jytWLHCdc2Lv2y3b7/9VgcOHFDPnj1dYxEREapVq5aeeeYZbdu2TU8//bTGjx%2Bv7du3l6reDh8%2BrJiYGN13332KjY3ViBEjLvnzdq62y203p9MpScXWDwsLKxW9Xej999/XwYMH1bdvX0m%2Bvd2uZV/hS9tt1apVCg0NVWJiomusWrVqqlGjhqZNm6Zt27apV69eGjp0qPbt21eqevMmAhYk/fYf1uDBgzV69GglJCRccrmAgADX38/9lnYpV5r3lMv1tn79eo0ZM0avvvqq6tWrJ0mKjo7W%2B%2B%2B/rzvuuENly5ZV69at1bdvXy1dutS1XmntLTIyUh9//LE6dOigMmXKKC4uTkOGDLmq2ktrb%2Bd777331Lt3b4WEhLjG2rRpo7feektRUVEqW7asunTpog4dOpS67Va9enU5HA6tXbtWBw4c0LPPPlui9Xxhu12ptw8%2B%2BEAzZ87UG2%2B84TpK6cvbzca%2BorT2dr733nvPdXH7Ob169dLrr7%2BumjVr6qabbtLAgQPVsGFDtzdnlIbevImABW3cuFGPP/64xo8fr4ceekjSb79VXvibhtPpVKVKlRQYGKjKlSu7fks5fz4iIsK1zMXmq1Spcn2bucDFejsnLS1NEyZM0KxZs9SpU6fLPk716tWVmZkpSZft3ZMu19v5qlevrqNHj8oY4xfb7fTp09qyZYvr3YOXc267labepN9%2BUalVq5ZGjRqlVatWKTg4uFht2dnZrtfU5bbbuWUunD9%2B/Hip6O3YsWOSpJSUFL355ptasGCBmjVrdtnH8JXtdq6385V0X%2BEr2%2B3QoUPatWuX2rZte8XHONd7aevNWwhYN7jvvvtOY8eO1cyZM/XAAw%2B4xmNiYrRnzx4VFBS4xhwOhxo1auSav/DWBOfmQ0JCVK9ePbdz7Tk5OTp48KDi4uKuc0f/36V6k6S1a9cqJSVFCxYsUOvWrd3m1qxZow8//NBtbN%2B%2BfapRo4ak33o/v7fCwkL99NNPru%2BNJ1yqt%2B3bt2vOnDluy%2B7bt0/Vq1dXQECAz283Sdq2bZvKlSvndn2HJH300UdavXq121hGRoZq1KhRKnrbvn27OnXqpKKiItdYYOBvu%2BC4uLhi22Xnzp1uP2%2BXes3VqFFD4eHhbvP/%2BMc/dPbsWcXExFzPllwu11uZMmX07rvvatWqVUpLSyt2rZkvb7ctW7b87n2FL2w36bdLKBo2bFjsF8g33nhD27dvdxs7t91KQ2%2Blgocvqkcpkp%2Bfbzp37mw%2B/vjjYnNnzpwxbdu2Na%2B//ro5deqU%2BeGHH0zz5s3Npk2bjDHG7Nmzx8TGxppNmzaZvLw8s3jxYtOkSROTmZlpjDHmww8/NG3atHG9tXrSpEnmwQcfLBW95eTkmPj4ePPFF19cdN3PPvvMxMXFmS%2B//NKcPXvWbN261TRu3Nh1i4otW7aYZs2ame%2B//96cOnXKzJo1yyQmJprTp09f157OuVxvDofDREdHm2XLlpmzZ8%2BaH3/80bRq1cq88847xhjf3m7nzJw503Tv3r3Y%2BF//%2BlfTsmVL8%2BOPP5qzZ8%2BalStXmoYNGxqHw2GM8X5vOTk5JiEhwUydOtWcOnXK/Prrr2bw4MHmT3/6kzl69Khp0qSJWbRokcnLyzObN282cXFxZteuXcaYK7/mpk%2Bfbrp3727%2B9a9/mWPHjpkhQ4aY4cOHl4reDh48aBo3bmz27Nlz0XV9ebtd676iNG%2B3c5599tmL1vTyyy%2BbTp06mYyMDJOXl2fefvttExcX57rli7d7Kw0IWDew//mf/zH169c3MTExxb5%2B/vlns2fPHtO3b18TExNj2rRp47r/yTnr1q0zHTt2NNHR0eaPf/yj%2Beabb1xzRUVFZubMmebOO%2B80cXFx5rHHHnP94Hm7t6VLl15y7pyPP/7YdOzY0cTGxpq2bduaRYsWuT3%2BwoULTWJioomJiTH9%2BvW75H8enu7t559/NuvXrzfdunUzcXFxplWrVubNN980hYWFrvV9dbv9/PPPxhhjJk2aZB5//PFi6xYVFZnZs2ebtm3bmpiYGHPvvfeajRs3lprejDFm9%2B7dpn///iYuLs60bNnSjBw50vW29W%2B%2B%2BcZ069bNREdHm44dO7r%2Bkz7ncq%2B5M2fOmOeff960aNHCNGnSxDz99NMmJyenVPSWmppqGjRoUGx7nrv1hq9vt2vZV5Tm7XbOoEGDzAsvvFBsvby8PPPyyy%2Bbu%2B66y8TGxpru3bub7777zjVfGnrztgBjbvCr0AAAACzjGiwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsOz/AVKqb763prdaAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-2338006694815494869">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2290</td>
        <td class="number">25</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2830</td>
        <td class="number">25</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3371</td>
        <td class="number">24</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3244</td>
        <td class="number">23</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2820</td>
        <td class="number">23</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2955</td>
        <td class="number">23</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2795</td>
        <td class="number">23</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2952</td>
        <td class="number">23</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2962</td>
        <td class="number">22</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2304</td>
        <td class="number">22</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (1655)</td>
        <td class="number">14887</td>
        <td class="number">98.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-2338006694815494869">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1863</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1874</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1879</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1888</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1889</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">3842</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3844</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3846</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3848</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3849</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Hillshade_3pm">Hillshade_3pm<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>247</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>1.6%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>135.09</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>248</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.6%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-2719820631468411957">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAAS9JREFUeJzt3cEJwkAUQMEolmQR9uTZnizCnuJZkEciBJdk5i7s5fGT3ZWc5nmeJ%2BCr878XACO7/HsBfLren6t/83rcNlgJ02SCQBIIBIFAEAgEgUCwi7UDa3e%2B7HotZ4JAEAgEgUAQCASBQBAIBIFAcA6yoV9u5jIWEwSCQCAIBIJAIAgEgl2sA/K/9%2BVMEAgCgSAQCAKB4CV9BVdHjscEgSAQCAKBIBAIAoEgEAgCgSAQCKejfqPQod/29nAD2ASBIBAIAoEgEAgCgbCL6%2B52pNiKCQJBIBAEAmG4k3TvE8c22um7CQJBIBCGe8SCkZggEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEN434SPEZZ55fwAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-2719820631468411957,#minihistogram-2719820631468411957"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-2719820631468411957">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-2719820631468411957"
                                                  aria-controls="quantiles-2719820631468411957" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-2719820631468411957" aria-controls="histogram-2719820631468411957"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-2719820631468411957" aria-controls="common-2719820631468411957"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-2719820631468411957" aria-controls="extreme-2719820631468411957"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-2719820631468411957">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>53</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>106</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>138</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>167</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>207</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>248</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>248</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>61</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>45.895</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.33973</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.087344</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>135.09</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>36.486</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-0.34083</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>2042591</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>2106.4</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-2719820631468411957">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xt8k/Xd//F329gibdMDUNxdGXWcpLRUQayrSDkITBkoWCjdrcJEB1itraAoiDLFoaOselNutB52j83NDGQKCMIcB0%2B4eZiaBlEpMFkfxUZJLK0t0Da/P/gRF4pQ6DcJqa/n49FH6febfPO5PqRX3lzXRRLm8Xg8AgAAgDHhwS4AAACgvSFgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDLMEu4PvC6TxofM3w8DAlJkbrwIE6NTd7jK//fUd//Yv%2B%2Bhf99S/6618m%2B9ulS6yhqk5PyB7B2rlzp6ZMmaKBAwcqKytLhYWFcjqd%2Bvvf/64%2BffooPT3d52vDhg3e%2B65YsUKjR4/WgAEDlJeXp/Lycu/coUOHdP/992vIkCHKzMxUQUGBXC5XMDbxlMLDwxQWFqbw8LBgl9Iu0V//or/%2BRX/9i/76V3vob0gGrMOHD%2Bumm27SpZdequ3bt2vdunX66quvtGDBAklScnKy7Ha7z9dVV10lSdq8ebOWLl2qX//613rrrbc0bNgwzZgxQ998840kqaSkRA6HQzabTRs3bpTH49G9994brE0FAAAhKCQDVn19vYqKijR9%2BnRFRkYqMTFRI0eO1GeffXbK%2B9psNk2YMEEZGRnq0KGDbr75ZknSli1b1NjYqFWrVunWW2/VD37wA8XHx6uwsFBbt27VF1984e/NAgAA7URIXoMVFxeniRMnen/evXu3/vKXv3iPUtXV1Sk/P1/vvvuuIiMjddNNN2nq1KkKCwuTw%2BHQ1Vdf7b1veHi4%2BvbtK7vdrr59%2B%2BrgwYPq16%2Bfd75Hjx7q0KGDHA6Hunbt2qr6qqur5XQ6fcYslo5KSkpqy2a3EBER7vMdZtFf/6K//kV//Yv%2B%2Bld76G9IBqxjKisrNXr0aDU2NmrSpEkqKCjQzp071bt3b02ZMkUlJSX6xz/%2BoTvuuEOxsbHKycmR2%2B1WXFyczzpxcXFyuVxyu92SJKvV6jNvtVpP6zosm82m0tJSn7H8/HwVFBSc4ZaenNV6rl/WxVH017/or3/RX/%2Biv/4Vyv0N6YB17Fqrf/3rX7r//vt19913a8mSJfr973/vvc3gwYM1efJkrV69Wjk5OZIkj%2Bfk/yPhVPOnkpubq%2BHDh/uMWSwd5XLVtWnd40VEhMtqPVc1NfVqamo2ujbor7/RX/%2Biv/5Ff/3LZH8TEqINVXV6QjpgSVJYWJhSUlJUVFSkyZMna968eUpMTPS5TXJysjZu3ChJSkhI8B6pOsbtdqtXr17e%2B7ndbkVHf/sX8vXXX6tTp06trikpKanF6UCn86AaG/3zS9jU1Oy3tUF//Y3%2B%2Bhf99S/661%2Bh3N%2BQPLm5fft2jR49Ws3N3zY9PPzopmzbtk1//OMffW6/e/dudevWTZKUlpYmh8PhnWtqatKOHTuUkZGhbt26KS4uzmf%2B008/1eHDh5WWlubPTQIAAO1ISAastLQ01dbWavHixaqvr9eBAwe0dOlSXXLJJYqNjdWjjz6qN954Q0eOHNGbb76pF154QXl5eZKkvLw8vfjii/rggw9UX1%2Bv5cuXKzIyUkOHDlVERIQmTZqkJ554QlVVVXK5XPrNb36jkSNHqnPnzkHeagAAECpC8hRhbGysnn32WS1cuFCXXXaZOnbsqMsuu0wPP/ywunbtqrlz5%2Bqhhx5SVVWVOnfurLlz52rUqFGSpCFDhujOO%2B9UYWGhvvrqK6Wnp6usrEwdOnSQJBUUFKiurk7XXHONGhsbNWzYMO/7awEAALRGmKetV3SjVfzxUTkWS7gSEqLlctWF7Dnqsxn99S/661/017/or3%2BZ7C8flQMAANBOELAAAAAMI2ABAAAYFpIXuQNAIF312JvBLuG0bCi8PNglAN97HMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhoVswNq5c6emTJmigQMHKisrS4WFhXI6nZKk7du3KycnRwMGDNCYMWO0Zs0an/uuWLFCo0eP1oABA5SXl6fy8nLv3KFDh3T//fdryJAhyszMVEFBgVwuV0C3DQAAhLaQDFiHDx/WTTfdpEsvvVTbt2/XunXr9NVXX2nBggWqrq7WrbfeqsmTJ2v79u2aN2%2Be5s%2BfL7vdLknavHmzli5dql//%2Btd66623NGzYMM2YMUPffPONJKmkpEQOh0M2m00bN26Ux%2BPRvffeG8zNBQAAISYkA1Z9fb2Kioo0ffp0RUZGKjExUSNHjtRnn32mtWvXKiUlRTk5OYqKilJWVpaGDx%2BulStXSpJsNpsmTJigjIwMdejQQTfffLMkacuWLWpsbNSqVat066236gc/%2BIHi4%2BNVWFiorVu36osvvgjmJgMAgBBiCXYBZyIuLk4TJ070/rx792795S9/0VVXXSWHw6HU1FSf26empmrDhg2SJIfDoauvvto7Fx4err59%2B8put6tv3746ePCg%2BvXr553v0aOHOnToIIfDoa5du7aqvurqau/pymMslo5KSko67W09mYiIcJ/vMIv%2B%2Bhf99R%2BLJZz%2B%2Bhn99a/20N%2BQDFjHVFZWavTo0WpsbNSkSZNUUFCgW265pUUQio%2BP915H5Xa7FRcX5zMfFxcnl8slt9stSbJarT7zVqv1tK7DstlsKi0t9RnLz89XQUFBq9c4HVbruX5ZF0fRX/%2Biv%2BYlJER7/0x//Yv%2B%2Blco9zekA1ZycrLsdrv%2B9a9/6f7779fdd9/dqvt5PJ42zZ9Kbm6uhg8f7jNmsXSUy1XXpnWPFxERLqv1XNXU1Kupqdno2qC//kZ//eeSea8Eu4RW%2B%2BvsK4Jdwhnh%2BetfJvv7n//gCKSQDliSFBYWppSUFBUVFWny5MnKzs72Hok6xuVyKTExUZKUkJDQYt7tdqtXr17e27jdbkVHf/sX8vXXX6tTp06trikpKanF6UCn86AaG/3zS9jU1Oy3tUF//Y3%2Bfr%2BF%2Bt89z1//CuX%2BhuTJze3bt2v06NFqbv626eHhRzelf//%2BPm%2B7IEnl5eXKyMiQJKWlpcnhcHjnmpqatGPHDmVkZKhbt26Ki4vzmf/00091%2BPBhpaWl%2BXOTAABAOxKSASstLU21tbVavHix6uvrdeDAAS1dulSXXHKJ8vLyVFlZqZUrV%2BrQoUPatm2btm3bpkmTJkmS8vLy9OKLL%2BqDDz5QfX29li9frsjISA0dOlQRERGaNGmSnnjiCVVVVcnlcuk3v/mNRo4cqc6dOwd5qwEAQKgIyVOEsbGxevbZZ7Vw4UJddtll6tixoy677DI9/PDD6tSpk5588kktXLhQv/zlL5WcnKzFixfrwgsvlCQNGTJEd955pwoLC/XVV18pPT1dZWVl6tChgySpoKBAdXV1uuaaa9TY2Khhw4ZpwYIFQdxaAAAQasI8bb2iG63idB40vqbFEq6EhGi5XHUhe476bEZ//SuU%2BnvVY28Gu4R2a0Ph5cEu4YyE0vM3FJnsb5cusYaqOj0heYoQAADgbEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADAsZANWZWWl8vPzlZmZqaysLN1zzz2qqanRv//9b/Xp00fp6ek%2BX88884z3vuvXr9fYsWN18cUXa8KECXrjjTe8c83NzSopKdGIESM0aNAgTZs2Tfv27QvGJgIAgBAVsgFrxowZslqt2rx5s1avXq3PPvtMjz76qHfebrf7fE2bNk2S9PHHH2vOnDmaPXu23n77bU2dOlW33Xab9u/fL0l67rnntHbtWpWVlWnLli1KSUlRfn6%2BPB5PULYTAACEnpAMWDU1NUpLS9OsWbMUHR2t8847T%2BPHj9e77757yvuuXLlS2dnZys7OVlRUlMaNG6fevXtrzZo1kiSbzaapU6eqR48eiomJUVFRkSoqKvThhx/6e7MAAEA7YQl2AWfCarVq0aJFPmNVVVVKSkry/nz33XfrrbfeUmNjoyZOnKiCggKdc845cjgcys7O9rlvamqq7Ha7GhoatGvXLqWmpnrnYmJi1L17d9ntdl100UWtqq%2B6ulpOp9NnzGLp6FOfCRER4T7fYRb99S/6C0myWELz75/nr3%2B1h/6GZMA6nt1u1x/%2B8ActX75ckZGRuvjiizVy5Eg9/PDD%2Bvjjj3X77bfLYrHojjvukNvtVlxcnM/94%2BLitGvXLn399dfyeDwnnHe5XK2ux2azqbS01GcsPz9fBQUFZ76RJ2G1nuuXdXEU/fUv%2Bvv9lpAQHewS2oTnr3%2BFcn9DPmC99957mjlzpmbNmqWsrCxJ0vPPP%2B%2Bd79%2B/v6ZPn64nn3xSd9xxhySd8nqqtl5vlZubq%2BHDh/uMWSwd5XLVtWnd40VEhMtqPVc1NfVqamo2ujbor7/RX0gyvl8MFJ6//mWyv8EK8SEdsDZv3qy77rpL8%2BfP17XXXvudt0tOTtaXX34pj8ejhIQEud1un3m3263ExETFx8crPDz8hPOdOnVqdV1JSUktTgc6nQfV2OifX8Kmpma/rQ3662/09/st1P/uef76Vyj3N2RPbr7//vuaM2eOHn/8cZ9wtX37di1fvtzntrt371ZycrLCwsKUlpam8vJyn3m73a6MjAxFRUWpV69ecjgc3rmamhp9/vnn6t%2B/v383CAAAtBshGbAaGxt13333afbs2Ro8eLDPXGxsrJYtW6aXXnpJR44ckd1u1zPPPKO8vDxJ0qRJk/TWW29p69atOnTokFatWqW9e/dq3LhxkqS8vDytWLFCFRUVqq2tVXFxsfr27av09PSAbycAAAhNIXmK8IMPPlBFRYUWLlyohQsX%2Bsy98sorKikpUWlpqe6//37Fxsbqhhtu0JQpUyRJvXv3VnFxsRYtWqTKykr17NlTTz75pLp06SJJmjx5spxOp2644QbV1dUpMzOzxQXrAAAAJxPm4R00A8LpPGh8TYslXAkJ0XK56kL2HPXZjP76Vyj196rH3gx2Ce3WhsLLg13CGQml528oMtnfLl1iDVV1ekLyFCEAAMDZLCRPEQIIbRwRAtDecQQLAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwLOABa/jw4SotLVVVVVWgHxoAACAgAh6wrrvuOq1fv15XXnmlbr75Zm3atEmNjY2BLgMAAMBvAh6w8vPztX79ev35z39Wr1699Ktf/UrZ2dlavHix9uzZE%2BhyAAAAjAvaNVj9%2BvXTnDlztGXLFs2dO1d//vOfdfXVV2vatGn66KOPglUWAABAmwUtYB05ckTr16/XLbfcojlz5qhr166699571bdvX02dOlVr164NVmkAAABtYgn0A1ZUVGjVqlV68cUXVVdXp9GjR%2Bt3v/udBg4c6L3NoEGDtGDBAo0dOzbQ5QEAALRZwAPWmDFjdMEFF2j69Om69tprFR8f3%2BI22dnZOnDgQKBLAwAAMCLgAWvFihW69NJLT3m7Dz/8MADVAAAAmBfwa7D69OmjGTNm6NVXX/WO/d///Z9uueUWud3uQJcDAABgXMAD1qJFi3Tw4EH17NnTOzZ06FA1NzfrkUceafU6lZWVys/PV2ZmprKysnTPPfeopqZGkvTxxx/r%2Buuv18CBAzVq1Cg9%2B%2ByzPvddv369xo4dq4svvlgTJkzQG2%2B84Z1rbm5WSUmJRowYoUGDBmnatGnat29fG7caAAB8nwQ8YL3xxhsqLS1VSkqKdywlJUXFxcV6/fXXW73OjBkzZLVatXnzZq1evVqfffaZHn30UTU0NGj69Om67LLL9Prrr6ukpERPPvmkNm3aJOlo%2BJozZ45mz56tt99%2BW1OnTtVtt92m/fv3S5Kee%2B45rV27VmVlZdqyZYtSUlKUn58vj8djtA8AAKD9CnjAamhoUFRUVMtCwsNVX1/fqjVqamqUlpamWbNmKTo6Wuedd57Gjx%2Bvd999V1u3btWRI0c0c%2BZMdezYUf369dPEiRNls9kkSStXrlR2drays7MVFRWlcePGqXfv3lqzZo0kyWazaerUqerRo4diYmJUVFSkiooKrgkDAACtFvCL3AcNGqRHHnlEs2bNUlxcnCTpiy%2B%2B0KOPPurzVg0nY7VatWjRIp%2BxqqoqJSUlyeFwqE%2BfPoqIiPDOpaamauXKlZIkh8Oh7Oxsn/umpqbKbreroaFBu3btUmpqqncuJiZG3bt3l91u10UXXdSq%2Bqqrq%2BV0On3GLJaOSkpKatX9WysiItznO8yiv4D/WSyh%2BfvF/sG/2kN/Ax6w5s6dq5tuukk//vGPFRMTo%2BbmZtXV1albt276/e9/f0Zr2u12/eEPf9Dy5cu1YcMGWa1Wn/n4%2BHi53W41NzfL7XZ7g90xcXFx2rVrl77%2B%2Bmt5PJ4TzrtcrlbXY7PZVFpa6jOWn5%2BvgoKC09yy1rFaz/XLujiK/gL%2Bk5AQHewS2oT9g3%2BFcn8DHrC6deuml19%2BWa%2B99po%2B//xzhYeH64ILLtDgwYN9jjq11nvvvaeZM2dq1qxZysrK0oYNG054u7CwMO%2BfT3U9VVuvt8rNzdXw4cN9xiyWjnK56tq07vEiIsJltZ6rmpp6NTU1G10b9BcIBNP7xUBh/%2BBfJvsbrBAf8IAlSZGRkbryyivbvM7mzZt11113af78%2Bbr22mslSYmJidq7d6/P7dxut%2BLj4xUeHq6EhIQWbwfhdruVmJjovc2J5jt16tTqupKSklqcDnQ6D6qx0T%2B/hE1NzX5bG/QX8KeRxa3/z01ngw2Fl/v8zP7Bv0K5vwEPWPv27dOSJUv02WefqaGhocX83/72t1at8/7772vOnDl6/PHHNXjwYO94Wlqa/vSnP6mxsVEWy9HNs9vtysjI8M6Xl5f7rGW32zVmzBhFRUWpV69ecjgc3jdDramp0eeff67%2B/fuf0fYCAIDvn6Bcg1VdXa3BgwerY8eOZ7RGY2Oj7rvvPs2ePdsnXElHP2YnJiZGy5cv180336xPP/1Uq1at0uLFiyVJkyZNUk5OjrZu3aof//jHWrt2rfbu3atx48ZJkvLy8lRWVqYhQ4aoa9euKi4uVt%2B%2BfZWent62DQcAAN8bAQ9Y5eXl%2Btvf/qbExMQzXuODDz5QRUWFFi5cqIULF/rMvfLKK3riiSf0wAMPqKysTJ07d1ZRUZGGDh0qSerdu7eKi4u1aNEiVVZWqmfPnnryySfVpUsXSdLkyZPldDp1ww03qK6uTpmZmS0uWAcAADiZME%2BA30Hzyiuv1Lp169ShQ4dAPmzQOZ0Hja9psYQrISFaLlddyJ6jPpvRX/%2B56rE3g10CcEaOXYPF/sG/TPa3S5dYQ1WdnoC/wcT06dNVWlrKO6MDAIB2K%2BCnCF977TW9//77Wr16tc4//3yFh/tmvOeffz7QJQEAABgV8IAVExOjIUOGBPphAQAAAibgAev4j7gBAABob4LyIT%2B7d%2B/W0qVLde%2B993rH/vnPfwajFAAAAOMCHrC2b9%2BucePGadOmTVq3bp2ko28%2BeuONN7b6TUYBAADOZgEPWCUlJbrrrru0du1a7%2BcDduvWTY888oiWLVsW6HIAAACMC3jA%2BvTTT5WXlyfJ9wOYf/KTn6iioiLQ5QAAABgX8IAVGxt7ws8grK6uVmRkZKDLAQAAMC7gAWvAgAH61a9%2BpdraWu/Ynj17NGfOHP34xz8OdDkAAADGBfxtGu69915NmTJFmZmZampq0oABA1RfX69evXrpkUceCXQ5AAAAxgU8YJ133nlat26dtm3bpj179qhDhw664IILdPnll/tckwUAABCqAh6wJOmcc87RlVdeGYyHBgAA8LuAB6zhw4ef9EgV74UFAABCXcAD1tVXX%2B0TsJqamrRnzx7Z7XZNmTIl0OUAAAAYF/CANXv27BOOb9y4UX//%2B98DXA0AAIB5QfkswhO58sor9fLLLwe7DAAAgDY7awLWjh075PF4gl0GAABAmwX8FOHkyZNbjNXX16uiokKjRo0KdDkAAADGBTxgpaSktPhfhFFRUcrJydHEiRMDXQ4AAIBxAQ9YvFs7AABo7wIesF588cVW3/baa6/1YyUAAAD%2BEfCANW/ePDU3N7e4oD0sLMxnLCwsjIAFAABCUsAD1tNPP61nn31WM2bMUJ8%2BfeTxePTJJ5/oqaee0vXXX6/MzMxAlwQAAGBUUK7BKisrU9euXb1jl1xyibp166Zp06Zp3bp1gS4JAADAqIC/D9bevXsVFxfXYtxqtaqysjLQ5QAAABgX8ICVnJysRx55RC6XyztWU1OjJUuW6Ic//GGgywEAADAu4KcI586dq1mzZslmsyk6Olrh4eGqra1Vhw4dtGzZskCXAwAAYFzAA9bgwYO1detWbdu2Tfv375fH41HXrl11xRVXKDY2NtDlAAAAGBfwgCVJ5557rkaMGKH9%2B/erW7duwSgBAADAbwIesBoaGvTAAw/o5ZdfliSVl5erpqZGd955p37zm9/IarUGuiSgXbjqsTeDXQIA4P8L%2BEXuixcv1scff6zi4mKFh3/78E1NTSouLg50OQAAAMYFPGBt3LhR//M//6Of/OQn3g99tlqtWrRokTZt2hTocgAAAIwLeMCqq6tTSkpKi/HExER98803gS4HAADAuIAHrB/%2B8If6%2B9//Lkk%2Bnz34yiuv6L/%2B679Oa63XX39dWVlZKioq8hlfvXq1LrzwQqWnp/t8ffTRR5Kk5uZmlZSUaMSIERo0aJCmTZumffv2ee/vdrtVWFiorKwsDR48WPPmzVNDQ8OZbjIAAPieCfhF7j/72c90%2B%2B2367rrrlNzc7N%2B%2B9vfqry8XBs3btS8efNavc5TTz2lVatWqXv37iecHzRokH7/%2B9%2BfcO65557T2rVr9dRTT6lr164qKSlRfn6%2BXnrpJYWFhWn%2B/Pk6fPiw1q1bpyNHjuiOO%2B5QcXGx7rvvvjPaZgAA8P0S8CNYubm5mjNnjt5%2B%2B21FREToiSeeUGVlpYqLi5WXl9fqdaKiok4asE7GZrNp6tSp6tGjh2JiYlRUVKSKigp9%2BOGH%2BvLLL/Xqq6%2BqqKhIiYmJ6tq1q2699Va98MILOnLkyGk/FgAA%2BP4J%2BBGsAwcO6LrrrtN1113XpnVuvPHGk85XVVXp5z//ucrLy2W1WlVQUKBrrrlGDQ0N2rVrl1JTU723jYmJUffu3WW323Xw4EFFRESoT58%2B3vl%2B/frpm2%2B%2B0e7du33Gv0t1dbWcTqfPmMXSUUlJSae5lScXERHu8x1m0V8Ax7NYfPcL7B/8oz30N%2BABa8SIEXr//fe9/4PQHxITE5WSkqI777xTPXv21F//%2BlfdfffdSkpK0o9%2B9CN5PJ4WHzgdFxcnl8ul%2BPh4xcTE%2BNR37Lb/%2BfmJJ2Oz2VRaWuozlp%2Bfr4KCgjZu2YlZref6ZV0cRX8BHJOQEO3zM/sH/wrl/gY8YGVmZmrDhg26%2Buqr/fYYQ4cO1dChQ70/jxkzRn/961%2B1evVqzZ49W5LvBfbHO9lca%2BTm5mr48OE%2BYxZLR7lcdW1a93gREeGyWs9VTU29mpqaja4N%2BgugpWP7cfYP/mWyv8eH4kAJeMD6wQ9%2BoIcfflhlZWX64Q9/qHPOOcdnfsmSJX553OTkZJWXlys%2BPl7h4eFyu90%2B8263W506dVJiYqJqa2vV1NSkiIgI75wkderUqVWPlZSU1OJ0oNN5UI2N/vklbGpq9tvaoL8AvnX8voD9g3%2BFcn8DHrB27dqlH/3oR5Jaf8rtdP3pT39SXFycz1GyiooKdevWTVFRUerVq5ccDocuvfRSSVJNTY0%2B//xz9e/fX8nJyfJ4PNq5c6f69esnSbLb7bJarbrgggv8Ui8AAGhfAhawioqKVFJS4vPWCcuWLVN%2Bfr7xxzp8%2BLAeeughdevWTRdeeKE2btyo1157TX/%2B858lSXl5eSorK9OQIUPUtWtXFRcXq2/fvkpPT5ckjR49Wo899pgeffRRHT58WMuWLVNOTo4slqB8NjYAAAgxAUsMmzdvbjFWVlZ2xgHrWBhqbGyUJL366quSjh5tuvHGG1VXV6c77rhDTqdT559/vpYtW6a0tDRJ0uTJk%2BV0OnXDDTeorq5OmZmZPhelP/jgg3rggQc0YsQInXPOOfrpT3/a4s1MAQAAvkuYp61XdLdS//79ve%2BkfrKx9srpPGh8TYslXAkJ0XK56kL2HPXZLNT6e9VjbwY9wYLWAAAURElEQVS7BKDd21B4uaTQ2z%2BEGpP97dIl1lBVpydgbzBxordl8OdbNQAAAARL6L6DFwAAwFmKgAUAAGBYwC5yP3LkiGbNmnXKMX%2B9DxYAAECgBCxgDRw4UNXV1accAwAACHUBC1j/%2Bf5XAAAA7RnXYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMCwkA5Yr7/%2BurKyslRUVNRibv369Ro7dqwuvvhiTZgwQW%2B88YZ3rrm5WSUlJRoxYoQGDRqkadOmad%2B%2Bfd55t9utwsJCZWVlafDgwZo3b54aGhoCsk0AACD0hWzAeuqpp7Rw4UJ17969xdzHH3%2BsOXPmaPbs2Xr77bc1depU3Xbbbdq/f78k6bnnntPatWtVVlamLVu2KCUlRfn5%2BfJ4PJKk%2BfPnq76%2BXuvWrdMLL7ygiooKFRcXB3T7AABA6ArZgBUVFaVVq1adMGCtXLlS2dnZys7OVlRUlMaNG6fevXtrzZo1kiSbzaapU6eqR48eiomJUVFRkSoqKvThhx/qyy%2B/1KuvvqqioiIlJiaqa9euuvXWW/XCCy/oyJEjgd5MAAAQgizBLuBM3Xjjjd8553A4lJ2d7TOWmpoqu92uhoYG7dq1S6mpqd65mJgYde/eXXa7XQcPHlRERIT69Onjne/Xr5%2B%2B%2BeYb7d6922f8u1RXV8vpdPqMWSwdlZSU1NrNa5WIiHCf7zCL/gI4nsXiu19g/%2BAf7aG/IRuwTsbtdisuLs5nLC4uTrt27dLXX38tj8dzwnmXy6X4%2BHjFxMQoLCzMZ06SXC5Xqx7fZrOptLTUZyw/P18FBQVnsjmnZLWe65d1cRT9BXBMQkK0z8/sH/wrlPvbLgOWJO/1VGcyf6r7nkpubq6GDx/uM2axdJTLVdemdY8XEREuq/Vc1dTUq6mp2ejaoL8AWjq2H2f/4F8m%2B3t8KA6UdhmwEhIS5Ha7fcbcbrcSExMVHx%2Bv8PDwE8536tRJiYmJqq2tVVNTkyIiIrxzktSpU6dWPX5SUlKL04FO50E1Nvrnl7Cpqdlva4P%2BAvjW8fsC9g/%2BFcr9Dd2TmyeRlpam8vJynzG73a6MjAxFRUWpV69ecjgc3rmamhp9/vnn6t%2B/v/r27SuPx6OdO3f63NdqteqCCy4I2DYAAIDQ1S4D1qRJk/TWW29p69atOnTokFatWqW9e/dq3LhxkqS8vDytWLFCFRUVqq2tVXFxsfr27av09HQlJiZq9OjReuyxx3TgwAHt379fy5YtU05OjiyWdnnADwAAGBayiSE9PV2S1NjYKEl69dVXJR092tS7d28VFxdr0aJFqqysVM%2BePfXkk0%2BqS5cukqTJkyfL6XTqhhtuUF1dnTIzM30uSn/wwQf1wAMPaMSIETrnnHP005/%2B9IRvZgoAAHAiYZ62XtGNVnE6Dxpf02IJV0JCtFyuupA9R302u%2BqxN4NdAgC0yYbCy4Ndwhkx%2BfrWpUusoapOT7s8RQgAABBMBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw9ptwOrTp4/S0tKUnp7u/XrooYckSdu3b1dOTo4GDBigMWPGaM2aNT73XbFihUaPHq0BAwYoLy9P5eXlwdgEAAAQoizBLsCfXnnlFZ1//vk%2BY9XV1br11ls1b948jR07Vu%2B9955mzpypCy64QOnp6dq8ebOWLl2qp59%2BWn369NGKFSs0Y8YMbdq0SR07dgzSlgAAgFDSbo9gfZe1a9cqJSVFOTk5ioqKUlZWloYPH66VK1dKkmw2myZMmKCMjAx16NBBN998syRpy5YtwSwbAACEkHZ9BGvJkiX65z//qdraWl111VW655575HA4lJqa6nO71NRUbdiwQZLkcDh09dVXe%2BfCw8PVt29f2e12jRkzplWPW11dLafT6TNmsXRUUlJSG7fIV0REuM93AAD%2Bk8USmq8P7eH1rd0GrIsuukhZWVl69NFHtW/fPhUWFuqXv/yl3G63unbt6nPb%2BPh4uVwuSZLb7VZcXJzPfFxcnHe%2BNWw2m0pLS33G8vPzVVBQcIZbc3JW67l%2BWRcAENoSEqKDXUKbhPLrW7sNWDabzfvnHj16aPbs2Zo5c6YGDhx4yvt6PJ42PXZubq6GDx/uM2axdJTLVdemdY8XEREuq/Vc1dTUq6mp2ejaAIDQZ/p1J1BMvr4FK2S224B1vPPPP19NTU0KDw%2BX2%2B32mXO5XEpMTJQkJSQktJh3u93q1atXqx8rKSmpxelAp/OgGhv9E4Kampr9tjYAIHSF%2BmtDKL%2B%2BtcuAtWPHDq1Zs0b33HOPd6yiokKRkZHKzs7WX/7yF5/bl5eXKyMjQ5KUlpYmh8Oh8ePHS5Kampq0Y8cO5eTkBG4D2qmrHnsz2CUAABAQoXv12El06tRJNptNZWVlOnz4sPbs2aPHH39cubm5uuaaa1RZWamVK1fq0KFD2rZtm7Zt26ZJkyZJkvLy8vTiiy/qgw8%2BUH19vZYvX67IyEgNHTo0uBsFAABCRpinrRccnaXeeecdLVmyRJ988okiIyM1fvx4FRUVKSoqSu%2B8844WLlyoiooKJScna9asWRo1apT3vn/84x9VVlamr776Sunp6VqwYIF69%2B7dpnqczoNt3aQWLJZwJSREy%2BWqC4lDqBzBAoDA2lB4ebBLOCMmX9%2B6dIk1VNXpabcB62xDwCJgAUCgEbCCF7Da5SlCAACAYCJgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwzBLsAgAAgH9c9dibwS6h1TYUXh7sEowiYIW4S%2Ba9EuwSAADAcThFCAAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbBOoLKyUr/4xS%2BUmZmpYcOGafHixWpubg52WQAAIERYgl3A2ej2229Xv3799Oqrr%2Bqrr77S9OnT1blzZ/385z8PdmkAACAEcATrOHa7XTt37tTs2bMVGxurlJQUTZ06VTabLdilAQCAEMERrOM4HA4lJycrLi7OO9avXz/t2bNHtbW1iomJOeUa1dXVcjqdPmMWS0clJSUZrTUignwMAGgfLJZvX9OOvb6F8uscAes4brdbVqvVZ%2BxY2HK5XK0KWDabTaWlpT5jt912m26//XZzhepokJty3mfKzc01Ht5wtL82m43%2B%2Bgn99S/661/017%2Bqq6v1u989HdL9Dd1o6Ecej6dN98/NzdXq1at9vnJzcw1V9y2n06nS0tIWR8tgBv31L/rrX/TXv%2Bivf7WH/nIE6ziJiYlyu90%2BY263W2FhYUpMTGzVGklJSSGbuAEAQNtxBOs4aWlpqqqq0oEDB7xjdrtdPXv2VHR0dBArAwAAoYKAdZzU1FSlp6dryZIlqq2tVUVFhX77298qLy8v2KUBAIAQEbFgwYIFwS7ibHPFFVdo3bp1euihh/Tyyy8rJydH06ZNU1hYWLBLayE6OlqXXnopR9f8hP76F/31L/rrX/TXv0K9v2Getl7RDQAAAB%2BcIgQAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAVgiorK/WLX/xCmZmZGjZsmBYvXqzm5uZglxXS%2BvTpo7S0NKWnp3u/HnroIUnS9u3blZOTowEDBmjMmDFas2ZNkKs9%2B73%2B%2BuvKyspSUVFRi7n169dr7NixuvjiizVhwgS98cYb3rnm5maVlJRoxIgRGjRokKZNm6Z9%2B/YFsvSQ8F39Xb16tS688EKf53F6ero%2B%2BugjSfS3tSorK5Wfn6/MzExlZWXpnnvuUU1NjSTp448/1vXXX6%2BBAwdq1KhRevbZZ33ue7LnN476rv7%2B%2B9//Vp8%2BfVo8f5955hnvfUOqvx6EnPHjx3vuu%2B8%2BT01NjWfPnj2eUaNGeZ599tlglxXSevfu7dm3b1%2BL8S%2B%2B%2BMJz0UUXeVauXOlpaGjwvPnmm57%2B/ft7PvrooyBUGRrKyso8o0aN8kyePNlTWFjoM7djxw5PWlqaZ%2BvWrZ6GhgbPSy%2B95MnIyPBUVVV5PB6PZ8WKFZ5hw4Z5du3a5Tl48KDnwQcf9IwdO9bT3NwcjE05K52svy%2B88ILn%2Buuv/8770t/W%2BelPf%2Bq55557PLW1tZ6qqirPhAkTPHPnzvXU19d7rrjiCs/SpUs9dXV1nvLycs%2Bll17q2bhxo8fjOfXzG0d9V3/37dvn6d2793feL9T6yxGsEGO327Vz507Nnj1bsbGxSklJ0dSpU2Wz2YJdWru0du1apaSkKCcnR1FRUcrKytLw4cO1cuXKYJd21oqKitKqVavUvXv3FnMrV65Udna2srOzFRUVpXHjxql3797eo4I2m01Tp05Vjx49FBMTo6KiIlVUVOjDDz8M9GactU7W31Ohv6dWU1OjtLQ0zZo1S9HR0TrvvPM0fvx4vfvuu9q6dauOHDmimTNnqmPHjurXr58mTpzo3f%2Be6vmNk/f3VEKtvwSsEONwOJScnKy4uDjvWL9%2B/bRnzx7V1tYGsbLQt2TJEg0dOlSXXHKJ5s%2Bfr7q6OjkcDqWmpvrcLjU1VeXl5UGq8ux34403KjY29oRz39VPu92uhoYG7dq1y2c%2BJiZG3bt3l91u92vNoeRk/ZWkqqoq/fznP9egQYM0YsQIvfTSS5JEf1vJarVq0aJF6ty5s3esqqpKSUlJcjgc6tOnjyIiIrxz/7k/ONnzG0edrL/H3H333Ro8eLAuu%2BwyLVmyREeOHJEUev0lYIUYt9stq9XqM3YsbLlcrmCU1C5cdNFFysrK0qZNm2Sz2fTBBx/ol7/85Qn7HR8fT6/PkNvt9vnHgXT0%2BetyufT111/L4/F85zxOLTExUSkpKbrrrrv05ptv6s4779TcuXO1fft2%2BnuG7Ha7/vCHP2jmzJnfuT9wu91qbm4%2B6fMbJ/af/Y2MjNTFF1%2BskSNHasuWLSorK9OaNWv0v//7v5JOvv84GxGwQpDH4wl2Ce2OzWbTxIkTFRkZqR49emj27Nlat26d919OMOdUz1%2Be32du6NChevrpp5WamqrIyEiNGTNGI0eO1OrVq723ob%2Bt995772natGmaNWuWsrKyvvN2YWFh3j/T39Y7vr9JSUl6/vnnNXLkSJ1zzjnq37%2B/pk%2BfHrLPXwJWiElMTJTb7fYZc7vdCgsLU2JiYpCqan/OP/98NTU1KTw8vEW/XS4XvT5DCQkJJ3z%2BJiYmKj4%2B/oT9drvd6tSpUyDLbFeSk5NVXV1Nf0/T5s2b9Ytf/EJz587VjTfeKOno/vf4oyVut9vb25M9v%2BHrRP09keTkZH355ZfyeDwh118CVohJS0tTVVWVDhw44B2z2%2B3q2bOnoqOjg1hZ6NqxY4ceeeQRn7GKigpFRkYqOzu7xfVW5eXlysjICGSJ7UZaWlqLftrtdmVkZCgqKkq9evWSw%2BHwztXU1Ojzzz9X//79A11qSPrTn/6k9evX%2B4xVVFSoW7du9Pc0vP/%2B%2B5ozZ44ef/xxXXvttd7xtLQ0ffLJJ2psbPSOHXv%2BHpv/ruc3vvVd/d2%2BfbuWL1/uc9vdu3crOTlZYWFhIddfAlaISU1NVXp6upYsWaLa2lpVVFTot7/9rfLy8oJdWsjq1KmTbDabysrKdPjwYe3Zs0ePP/64cnNzdc0116iyslIrV67UoUOHtG3bNm3btk2TJk0KdtkhadKkSXrrrbe0detWHTp0SKtWrdLevXs1btw4SVJeXp5WrFihiooK1dbWqri4WH379lV6enqQKw8Nhw8f1kMPPSS73a4jR45o3bp1eu211zR58mRJ9Lc1Ghsbdd9992n27NkaPHiwz1x2drZiYmK0fPly1dfX68MPP9SqVau8%2B99TPb9x8v7GxsZq2bJleumll3TkyBHZ7XY988wzIdvfME8ondCEJGn//v2aP3%2B%2B/vGPfygmJkaTJ0/Wbbfd5nMdAE7PO%2B%2B8oyVLluiTTz5RZGSkxo8fr6KiIkVFRemdd97RwoULVVFRoeTkZM2aNUujRo0KdslnrWMv1sf%2BlW%2BxWCTJ%2Bz99Nm3apCVLlqiyslI9e/bUvHnzNGjQIElHr69YunSpnn/%2BedXV1SkzM1MPPvigzjvvvCBsydnpZP31eDxavny5Vq1aJafTqfPPP1933323hg0bJon%2Btsa7776r//7v/1ZkZGSLuVdeeUV1dXV64IEHVF5ers6dO%2BuWW27Rz372M%2B9tTvb8xqn7u2PHDpWWlmrv3r2KjY3VDTfcoFtuuUXh4UePB4VSfwlYAAAAhnGKEAAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAM%2B3/c74bpm5Tk4AAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-2719820631468411957">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">143</td>
        <td class="number">182</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">149</td>
        <td class="number">161</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">132</td>
        <td class="number">156</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">133</td>
        <td class="number">154</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">142</td>
        <td class="number">154</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">136</td>
        <td class="number">154</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">137</td>
        <td class="number">152</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">138</td>
        <td class="number">148</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">154</td>
        <td class="number">148</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">152</td>
        <td class="number">145</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (237)</td>
        <td class="number">13566</td>
        <td class="number">89.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-2719820631468411957">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">88</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">244</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:75%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">245</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">246</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">247</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">248</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Hillshade_9am">Hillshade_9am<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>176</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>1.2%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>212.7</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>254</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-1931379802470597327">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAASFJREFUeJzt28ENgkAUQEE1lmQR9uTZnizCntYGzIuQEBBm7iR74OXvEvY8xhgn4KvL2guALbuuvQD26/Z4TX7m/bwvsJL5TBAIAoEgEAgCgeCQzk/mHLj3wASBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgeBG4QEd9XbgHCYIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQPCz4g74%2BXA5JggEgUAQCASBQBAIBF%2BxNsYXqW0xQSAIBIIt1oJsl/6fQCbwwh/PeYwx1l4EbJUzCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCIQP/owUBQrKmIoAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-1931379802470597327,#minihistogram-1931379802470597327"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-1931379802470597327">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-1931379802470597327"
                                                  aria-controls="quantiles-1931379802470597327" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-1931379802470597327" aria-controls="histogram-1931379802470597327"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-1931379802470597327" aria-controls="common-1931379802470597327"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-1931379802470597327" aria-controls="extreme-1931379802470597327"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-1931379802470597327">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>151</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>196</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>220</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>235</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>250</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>254</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>254</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>39</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>30.561</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.14368</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.2188</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>212.7</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>24.046</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-1.0937</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>3216089</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>933.99</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-1931379802470597327">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3XtcVXW%2B//E3l4Dkjoo1ZOHxQiJoakpDJoqplWlpJDKni5OZt%2BIBaVmaZeWkM0rW0Y7KdDlZ85j2QBcvafooL12kOVMz2WZrF0mPxtGg3FuE8ALs3x/93DM7b3T87rXd8Ho%2BHjzQ73ft7/6sjwt4u9Zi7yC32%2B0WAAAAjAn2dwEAAAAtDQELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABgW6u8CWovq6sPG1wwODlJCQqQOHqxTU5Pb%2BPqtFX31DfrqG/TVPHrqG/7qa/v20ZY917/iDFYACw4OUlBQkIKDg/xdSotCX32DvvoGfTWPnvpGa%2BsrAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADAv1dwEAAMA3rn/mI3%2BX0GzrC672dwlGcQYLAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDAjZgpaSkKC0tTenp6Z6PJ598UpJUVlamnJwc9enTRyNGjNDq1au9Hrty5UoNHz5cffr0UV5ensrLyz1zR48e1aOPPqqBAwcqIyND%2Bfn5cjqdlu4bAAAIbKH%2BLuBcvPPOO7rkkku8xqqqqjR16lTNnj1bI0eO1KeffqopU6aoU6dOSk9P16ZNm7RkyRI9//zzSklJ0cqVKzV58mRt3LhRbdq00eLFi%2BVwOGSz2XThhRdqzpw5evjhh7V8%2BXI/7SUAAAg0AXsG63TWrFmj5ORk5eTkKDw8XJmZmcrOzlZJSYkkyWazacyYMerVq5ciIiJ09913S5I2b96shoYGlZaWaurUqbr44osVFxengoICbdmyRd99950/dwsAAASQgA5YRUVFGjRokK688krNmTNHdXV1cjgcSk1N9douNTXVcxnw5/PBwcHq3r277Ha79u7dq8OHD6tHjx6e%2Bc6dOysiIkIOh8OanQIAAAEvYC8RXnHFFcrMzNTvf/977du3TwUFBXr88cflcrnUoUMHr23j4uI891G5XC7FxsZ6zcfGxsrpdMrlckmSYmJivOZjYmJ%2B0X1YVVVVqq6u9hoLDW2jxMTEZq/RHCEhwV6fYQZ99Q366hv01Tx66h%2BhoS2r3wEbsGw2m%2BfPnTt31owZMzRlyhT17dv3rI91u93nNN%2Bc2pYuXeo1Nm3aNOXn55/TuqcTE3OhT9Zt7eirb9BX36Cv5tFTa8XHR/q7BKMCNmD93CWXXKLGxkYFBwd7zkSd4HQ6lZCQIEmKj48/ad7lcqlr166ebVwulyIj//kPfejQIbVt27bZteTm5io7O9trLDS0jZzOul%2B0T2cTEhKsmJgLVVNTr8bGJqNrt2b01Tfoq2/QV/PoqX%2BY/hl5gr%2BCW0AGrB07dmj16tV66KGHPGMVFRUKCwtTVlaW3nzzTa/ty8vL1atXL0lSWlqaHA6HRo8eLUlqbGzUjh07lJOTo44dOyo2NlYOh0NJSUmSpK%2B%2B%2BkrHjh1TWlpas%2BtLTEw86XJgdfVhNTT45gu1sbHJZ2u3ZvTVN%2Birb9BX8%2BiptVparwMyYLVt21Y2m00JCQkaP368Kisr9eyzzyo3N1c33XSTli5dqpKSEo0aNUoff/yxtm7d6rmkmJeXp/vvv1833nijUlJS9MILLygsLEyDBg1SSEiIxo4dq%2BXLlys9PV0RERF6%2BumnNXToULVr187Pew0A8Lfrn/nI3yUgQARkwOrQoYOKi4tVVFSkZcuWKSwsTKNHj1ZhYaHCw8O1YsUKzZs3T48//riSkpK0cOFCXX755ZKkgQMH6v7771dBQYF%2B%2BOEHpaenq7i4WBEREZKk/Px81dXV6aabblJDQ4MGDx6suXPn%2BnFvAQBAoAlyn%2Bsd3WiW6urDxtcMDQ1WfHyknM66Fndq1Z/oq2/QV9%2Bgr%2BadqaecwfKd9QVX%2B2Td9u2jfbLu2bSs34kEAAA4DxCwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgWIsIWE899ZRSUlI8fy8rK1NOTo769OmjESNGaPXq1V7br1y5UsOHD1efPn2Ul5en8vJyz9zRo0f16KOPauDAgcrIyFB%2Bfr6cTqdl%2BwIAAAJfwAesnTt3atWqVZ6/V1VVaerUqRo3bpzKyso0e/ZszZkzR3a7XZK0adMmLVmyRH/4wx%2B0bds2DR48WJMnT9aPP/4oSVq8eLEcDodsNps2bNggt9uthx9%2B2C/7BgAAAlNAB6ympiY99thjGj9%2BvGdszZo1Sk5OVk5OjsLDw5WZmans7GyVlJRIkmw2m8aMGaNevXopIiJCd999tyRp8%2BbNamhoUGlpqaZOnaqLL75YcXFxKigo0JYtW/Tdd9/5YxcBAEAACuiA9dprryk8PFwjR470jDkcDqWmpnptl5qa6rkM%2BPP54OBgde/eXXa7XXv37tXhw4fVo0cPz3znzp0VEREhh8Ph470BAAAtRai/C/i/%2Bv7777VkyRK98sorXuMul0sdOnTwGouLi/PcR%2BVyuRQbG%2Bs1HxsbK6fTKZfLJUmKiYnxmo%2BJiflF92FVVVWpurraayw0tI0SExObvUZzhIQEe32GGfTVN%2Birb9BX8%2Bipf4SGtqx%2BB2zAmj9/vsaMGaMuXbro22%2B//UWPdbvd5zR/NjabTUuXLvUamzZtmvLz889p3dOJibnQJ%2Bu2dvTVN%2Birb9BX8%2BipteLjI/1dglEBGbDKysr0j3/8Q2vXrj1pLj4%2B3nMm6gSn06mEhITTzrtcLnXt2tWzjcvlUmTkP/%2BhDx06pLZt2za7vtzcXGVnZ3uNhYa2kdNZ1%2Bw1miMkJFgxMReqpqZejY1NRtduzeirb9BX36Cv5tFT/zD9M/IEfwW3gAxYq1ev1g8//KDBgwdL%2BucZp4yMDN11110nBa/y8nL16tVLkpSWliaHw6HRo0dLkhobG7Vjxw7l5OSoY8eOio2NlcPhUFJSkiTpq6%2B%2B0rFjx5SWltbs%2BhITE0%2B6HFhdfVgNDb75Qm1sbPLZ2q0ZffUN%2Buob9NU8emqtltbrgLzg%2BdBDD2nDhg1atWqVVq1apeLiYknSqlWrNHLkSFVWVqqkpERHjx7V1q1btXXrVo0dO1aSlJeXp7feekufffaZ6uvrtWzZMoWFhWnQoEEKCQnR2LFjtXz5cu3fv19Op1NPP/20hg4dqnbt2vlzlwEAQAAJyDNYsbGxXjeqNzQ0SJIuuugiSdKKFSs0b948Pf7440pKStLChQt1%2BeWXS5IGDhyo%2B%2B%2B/XwUFBfrhhx%2BUnp6u4uJiRURESJLy8/NVV1enm266SQ0NDRo8eLDmzp1r7Q4CAICAFuQ%2B1zu60SzV1YeNrxkaGqz4%2BEg5nXUt7tSqP9FX36CvvkFfzTtTT69/5iM/VdXyrS%2B42ifrtm8f7ZN1zyYgLxECAACczwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwywPWNnZ2Vq6dKn2799v9VMDAABYwvKAdcstt2jdunW69tprdffdd2vjxo1qaGiwugwAAACfsTxgTZs2TevWrdNf/vIXde3aVU899ZSysrK0cOFC7d692%2BpyAAAAjPPbPVg9evTQzJkztXnzZs2aNUt/%2BctfdMMNN2jChAn6/PPP/VUWAADAOfNbwDp%2B/LjWrVuniRMnaubMmerQoYMefvhhde/eXePHj9eaNWv8VRoAAMA5CbX6CSsqKlRaWqq33npLdXV1Gj58uF5%2B%2BWX17dvXs02/fv00d%2B5cjRw50uryAAAAzpnlAWvEiBHq1KmTJk2apJtvvllxcXEnbZOVlaWDBw9aXRoAAIARlgeslStXqn///mfdbvv27RZUAwAAYJ7l92ClpKRo8uTJevfddz1j//Vf/6WJEyfK5XJZXQ4AAIBxlges%2BfPn6/Dhw%2BrSpYtnbNCgQWpqatKCBQusLgcAAMA4yy8Rfvjhh1qzZo3i4%2BM9Y8nJyVq0aJFuvPFGq8sBAAAwzvIzWEeOHFF4ePjJhQQHq76%2B3upyAAAAjLM8YPXr108LFizQoUOHPGPfffedHn/8ca%2BXagAAAAhUll8inDVrlu666y79%2Bte/VlRUlJqamlRXV6eOHTvqlVdesbocAAAA4ywPWB07dtTbb7%2Bt999/X3v37lVwcLA6deqkAQMGKCQkxOpyAAAAjLM8YElSWFiYrr32Wn88NQAAgM9ZHrD27dunoqIiff311zpy5MhJ8%2B%2B9957VJQEAABjll3uwqqqqNGDAALVp08bqpwcAAPA5ywNWeXm53nvvPSUkJFj91AAAAJaw/GUa2rZty5krAADQolkesCZNmqSlS5fK7Xaf0zpffPGF7rzzTvXt21eZmZkqKChQdXW1JKmsrEw5OTnq06ePRowYodWrV3s9duXKlRo%2BfLj69OmjvLw8lZeXe%2BaOHj2qRx99VAMHDlRGRoby8/PldDrPqVYAANC6WB6w3n//fb355pu6%2BuqrNXbsWI0bN87rozmOHTumu%2B66S/3791dZWZnWrl2rH374QXPnzlVVVZWmTp2qcePGqaysTLNnz9acOXNkt9slSZs2bdKSJUv0hz/8Qdu2bdPgwYM1efJk/fjjj5KkxYsXy%2BFwyGazacOGDXK73Xr44Yd91g8AANDyWB6woqKiNHDgQGVlZalz587q1KmT10dz1NfXq7CwUJMmTVJYWJgSEhI0dOhQff3111qzZo2Sk5OVk5Oj8PBwZWZmKjs7WyUlJZIkm82mMWPGqFevXoqIiNDdd98tSdq8ebMaGhpUWlqqqVOn6uKLL1ZcXJwKCgq0ZcsWfffddz7rCQAAaFksv8l9/vz557xGbGysbr31Vs/fv/nmG7355pu6/vrr5XA4lJqa6rV9amqq1q9fL0lyOBy64YYbPHPBwcHq3r277Ha7unfvrsOHD6tHjx6e%2Bc6dOysiIkIOh0MdOnQ459oBAEDL55cXGv3mm2/09ttv63//9389gesf//iHevfu/YvWqays1PDhw9XQ0KCxY8cqPz9fEydOPCkIxcXFee6jcrlcio2N9ZqPjY2V0%2BmUy%2BWSJMXExHjNx8TE/KL7sKqqqjz3g50QGtpGiYmJzV6jOUJCgr0%2Bwwz66hv01Tfoq3n01D9CQ1tWvy0PWGVlZZo4caI6deqkPXv2aP78%2Bdq3b5/uuOMOPfPMMxoyZEiz10pKSpLdbtf//M//6NFHH9WDDz7YrMed7Qb7c70B32azaenSpV5j06ZNU35%2B/jmtezoxMRf6ZN3Wjr76Bn31DfpqHj21Vnx8pL9LMMrygLV48WI98MADuvPOO9WzZ09JP70/4YIFC/Tcc8/9ooAlSUFBQUpOTlZhYaHGjRunrKwsz5moE5xOp%2Bd1t%2BLj40%2Bad7lc6tq1q2cbl8ulyMh//kMfOnRIbdu2bXZNubm5ys7O9hoLDW0jp7PuF%2B3b2YSEBCsm5kLV1NSrsbHJ6NqtGX31DfrqG/TVPHrqH6Z/Rp7gr%2BBmecD66quv9Oqrr0r6KRydcN1112nWrFnNWqOsrExz587V%2BvXrFRz80ynFE5979uypDRs2eG1fXl6uXr16SZLS0tLkcDg0evRoSVJjY6N27NihnJwcdezYUbGxsXI4HEpKSvLUe%2BzYMaWlpTV7HxMTE0%2B6HFhdfVgNDb75Qm1sbPLZ2q0ZffUN%2Buob9NU8emqtltZryy94RkdHn/I9CKuqqhQWFtasNdLS0lRbW6uFCxeqvr5eBw8e1JIlS3TllVcqLy9PlZWVKikp0dGjR7V161Zt3bpVY8eOlSTl5eXprbfe0meffab6%2BnotW7ZMYWFhGjRokEJCQjR27FgtX75c%2B/fvl9Pp1NNPP62hQ4eqXbt2RvsAAABaLssDVp8%2BffTUU0%2BptrbWM7Z7927NnDlTv/71r5u1RnR0tF588UWVl5frqquu0ogRIxQdHa2nn35abdu21YoVK/Tqq6%2Bqb9%2B%2Beuqpp7Rw4UJdfvnlkqSBAwfq/vvvV0FBgfr3769t27apuLhYERERkqT8/Hz16tVLN910k4YMGaLIyEj97ne/M98IAADQYgW5z/WO7l/owIEDuvPOO/Xtt9%2BqsbFRbdq0UX19vbp27arly5frV7/6lZXlWKa6%2BrDxNUNDgxUfHymns67FnVr1J/rqG/TVN%2BireWfq6fXPfOSnqlq%2B9QVX%2B2Td9u2jfbLu2Vh%2BD9ZFF12ktWvXauvWrdq9e7ciIiLUqVMnXX311V73ZAEAAAQqv7wO1gUXXKBrr73WH08NAADgc5YHrOzs7DOeqXrvvfcsrAYAAMA8ywPWDTfc4BWwGhsbtXv3btntdt15551WlwMAAGCc5QFrxowZpxzfsGGD/vrXv1pcDQAAgHnnzRv/XHvttXr77bf9XQYAAMA5O28C1o4dO875PQABAADOB5ZfIhw3btxJY/X19aqoqNCwYcOsLgcA4Ee8rhRaKssDVnJy8km/RRgeHq6cnBzdeuutVpcDAABgnOUBa8GCBVY/JQAAgKUsD1hvvfVWs7e9%2BeabfVgJAACAb1gesGbPnq2mpqaTbmgPCgryGgsKCiJgAQCAgGR5wHr%2B%2Bef14osvavLkyUpJSZHb7daXX36pP/7xj7rtttuUkZFhdUkAAABG%2BeUerOLiYnXo0MEzduWVV6pjx46aMGGC1q5da3VJAAAARln%2BOlh79uxRbGzsSeMxMTGqrKy0uhwAAADjLA9YSUlJWrBggZxOp2espqZGRUVFuvTSS60uBwAAwDjLLxHOmjVL06dPl81mU2RkpIKDg1VbW6uIiAg999xzVpcDAABgnOUBa8CAAdqyZYu2bt2qAwcOyO12q0OHDrrmmmsUHR1tdTkAAADGWR6wJOnCCy/UkCFDdODAAXXs2NEfJQAAAPiM5fdgHTlyRDNnzlTv3r11/fXXS/rpHqy7775bNTU1VpcDAABgnOUBa%2BHChdq5c6cWLVqk4OB/Pn1jY6MWLVpkdTkAAADGWR6wNmzYoP/4j//Qdddd53nT55iYGM2fP18bN260uhwAAADjLA9YdXV1Sk5OPmk8ISFBP/74o9XlAAAAGGd5wLr00kv117/%2BVZK83nvwnXfe0a9%2B9SurywEAADDO8t8i/M1vfqP77rtPt9xyi5qamvTSSy%2BpvLxcGzZs0OzZs60uBwAAwDjLA1Zubq5CQ0P16quvKiQkRMuXL1enTp20aNEiXXfddVaXAwAAYJzlAevgwYO65ZZbdMstt1j91AAAAJaw/B6sIUOGeN17BQAA0NJYHrAyMjK0fv16q58WAADAMpZfIrz44ov1u9/9TsXFxbr00kt1wQUXeM0XFRVZXRIAAIBRlgesXbt26d/%2B7d8kSU6n0%2BqnBwAA8DnLAlZhYaEWL16sV155xTP23HPPadq0aVaVAAAAYAnL7sHatGnTSWPFxcVWPT0AAIBlLAtYp/rNQX6bEAAAtESWBawTb%2Bx8tjEAAIBAZ/nLNAAAALR0BCwAAADDLPstwuPHj2v69OlnHeN1sAAAQKCzLGD17dtXVVVVZx0DAAAIdJYFrH99/SsAAICWjHuwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLCADViVlZWaNm2aMjIylJmZqYceekg1NTWSpJ07d%2Bq2225T3759NWzYML344otej123bp1Gjhyp3r17a8yYMfrwww89c01NTVq8eLGGDBmifv36acKECdq3b5%2Bl%2BwYAAAJbwAasyZMnKyYmRps2bdIbb7yhr7/%2BWr///e915MgRTZo0SVdddZU%2B%2BOADLV68WCtWrNDGjRsl/RS%2BZs6cqRkzZujjjz/W%2BPHjde%2B99%2BrAgQOSpD/96U9as2aNiouLtXnzZiUnJ2vatGm8MTUAAGi2gAxYNTU1SktL0/Tp0xUZGamLLrpIo0eP1ieffKItW7bo%2BPHjmjJlitq0aaMePXro1ltvlc1mkySVlJQoKytLWVlZCg8P16hRo9StWzetXr1akmSz2TR%2B/Hh17txZUVFRKiwsVEVFhbZv3%2B7PXQYAAAEkIANWTEyM5s%2Bfr3bt2nnG9u/fr8TERDkcDqWkpCgkJMQzl5qaqvLyckmSw%2BFQamqq13qpqamy2%2B06cuSIdu3a5TUfFRWlyy67THa73cd7BQAAWgrLXsndl%2Bx2u1599VUtW7ZM69evV0xMjNd8XFycXC6Xmpqa5HK5FBsb6zUfGxurXbt26dChQ3K73aecdzqdza6nqqpK1dXVXmOhoW2UmJj4C/fszEJCgr0%2Bwwz66hv01TfoK1qK0NCWdQwHfMD69NNPNWXKFE2fPl2ZmZlav379KbcLCgry/Pls91Od6/1WNptNS5cu9RqbNm2a8vPzz2nd04mJudAn67Z29NU36Ktv0FcEuvj4SH%2BXYFRAB6xNmzbpgQce0Jw5c3TzzTdLkhISErRnzx6v7Vwul%2BLi4hQcHKz4%2BHi5XK6T5hMSEjzbnGq%2Bbdu2za4rNzdX2dnZXmOhoW3kdNb9gr07u5CQYMXEXKiamno1NjYZXbs1o6%2B%2BQV99g76ipTD9M/IEfwW3gA1Yf//73zVz5kw9%2B%2ByzGjBggGc8LS1Nf/7zn9XQ0KDQ0J92z263q1evXp75E/djnWC32zVixAiFh4era9eucjgc6t%2B/v6Sfbqjfu3evevbs2ezaEhMTT7ocWF19WA0Nvvnm19jY5LO1WzP66hv01TfoKwJdSzt%2BA/KCZ0NDgx555BHNmDHDK1xJUlZWlqKiorRs2TLV19dr%2B/btKi0tVV5eniRp7Nix2rZtm7Zs2aKjR4%2BqtLRUe/bs0ahRoyRJeXl5WrlypSoqKlRbW6tFixape/fuSk9Pt3w/AQBAYArIM1ifffaZKioqNG/ePM2bN89r7p133tHy5cv12GOPqbi4WO3atVNhYaEGDRokSerWrZsWLVqk%2BfPnq7KyUl26dNGKFSvUvn17SdK4ceNUXV2t22%2B/XXV1dcrIyDjpfioAAIAzCXLzCpqWqK4%2BbHzN0NBgxcdHyumsa3GnVv2JvvoGffWNQO/r9c985O8ScJ5YX3C1T9Zt3z7aJ%2BueTUBeIgQAADifEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLBQfxcAADDr%2Bmc%2B8ncJQKvHGSwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgWEAHrA8%2B%2BECZmZkqLCw8aW7dunUaOXKkevfurTFjxujDDz/0zDU1NWnx4sUaMmSI%2BvXrpwkTJmjfvn2eeZfLpYKCAmVmZmrAgAGaPXu2jhw5Ysk%2BAQCAwBewAeuPf/yj5s2bp8suu%2BykuZ07d2rmzJmaMWOGPv74Y40fP1733nuvDhw4IEn605/%2BpDVr1qi4uFibN29WcnKypk2bJrfbLUmaM2eO6uvrtXbtWr3%2B%2BuuqqKjQokWLLN0/AAAQuAI2YIWHh6u0tPSUAaukpERZWVnKyspSeHi4Ro0apW7dumn16tWSJJvNpvHjx6tz586KiopSYWGhKioqtH37dn3//fd69913VVhYqISEBHXo0EFTp07V66%2B/ruPHj1u9mwAAIAAFbMC64447FB0dfco5h8Oh1NRUr7HU1FTZ7XYdOXJEu3bt8pqPiorSZZddJrvdrp07dyokJEQpKSme%2BR49eujHH3/UN99845udAQAALUqLfKscl8ul2NhYr7HY2Fjt2rVLhw4dktvtPuW80%2BlUXFycoqKiFBQU5DUnSU6ns1nPX1VVperqaq%2Bx0NA2SkxM/L/szmmFhAR7fYYZ9NU36Ktv0Fe0FKGhLesYbpEBS5Lnfqr/y/zZHns2NptNS5cu9RqbNm2a8vPzz2nd04mJudAn67Z29NXjur6IAAANXUlEQVQ36Ktv0FcEuvj4SH%2BXYFSLDFjx8fFyuVxeYy6XSwkJCYqLi1NwcPAp59u2bauEhATV1taqsbFRISEhnjlJatu2bbOePzc3V9nZ2V5joaFt5HTW/V936ZRCQoIVE3Ohamrq1djYZHTt1oy%2B%2BgZ99Q36ipbC9M/IE/wV3FpkwEpLS1N5ebnXmN1u14gRIxQeHq6uXbvK4XCof//%2BkqSamhrt3btXPXv2VFJSktxut7744gv16NHD89iYmBh16tSpWc%2BfmJh40uXA6urDamjwzTe/xsYmn63dmtFX36CvvkFfEeha2vHbsi54/n9jx47Vtm3btGXLFh09elSlpaXas2ePRo0aJUnKy8vTypUrVVFRodraWi1atEjdu3dXenq6EhISNHz4cD3zzDM6ePCgDhw4oOeee045OTkKDW2ReRQAABgWsIkhPT1dktTQ0CBJevfddyX9dLapW7duWrRokebPn6/Kykp16dJFK1asUPv27SVJ48aNU3V1tW6//XbV1dUpIyPD656pJ554Qo899piGDBmiCy64QDfeeOMpX8wUAADgVILc53pHN5qluvqw8TVDQ4MVHx8pp7OuxZ1a9Sf66hv01TdO1dfrn/nIz1UBv9z6gqt9sm779qd%2BSSdfa5GXCAEAAPyJgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLBQfxcAAOe765/5yN8lAAgwnMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhof4uAEDrc/0zH/m7BADwKc5gAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEErFOorKzUPffco4yMDA0ePFgLFy5UU1OTv8sCAAABgpdpOIX77rtPPXr00LvvvqsffvhBkyZNUrt27fTb3/7W36UBAIAAQMD6Gbvdri%2B%2B%2BEIvvfSSoqOjFR0drfHjx%2Bvll18mYOG8xmtLAcD5g4D1Mw6HQ0lJSYqNjfWM9ejRQ7t371Ztba2ioqLOukZVVZWqq6u9xkJD2ygxMdForSEhwV6fYQZ9BQDrhYa2rO%2B5BKyfcblciomJ8Ro7EbacTmezApbNZtPSpUu9xu69917dd9995grVT0Hu5ZefV25urvHw1poFal8/%2Bd11/i7hjKqqqmSz2QKur%2Bc7%2BmoePfWN1tbXlhUXDXG73ef0%2BNzcXL3xxhteH7m5uYaq%2B6fq6motXbr0pLNlODf01Tfoq2/QV/PoqW%2B0tr5yButnEhIS5HK5vMZcLpeCgoKUkJDQrDUSExNbRToHAACnxhmsn0lLS9P%2B/ft18OBBz5jdbleXLl0UGRnpx8oAAECgIGD9TGpqqtLT01VUVKTa2lpVVFTopZdeUl5enr9LAwAAASJk7ty5c/1dxPnmmmuu0dq1a/Xkk0/q7bffVk5OjiZMmKCgoCB/l3aSyMhI9e/fn7NrhtFX36CvvkFfzaOnvtGa%2BhrkPtc7ugEAAOCFS4QAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAlBlZaXuueceZWRkaPDgwVq4cKGampr8XVZASklJUVpamtLT0z0fTz75pCSprKxMOTk56tOnj0aMGKHVq1f7udrz1wcffKDMzEwVFhaeNLdu3TqNHDlSvXv31pgxY/Thhx965pqamrR48WINGTJE/fr104QJE7Rv3z4rSz%2Bvna6vb7zxhi6//HKv4zY9PV2ff/65JPp6JpWVlZo2bZoyMjKUmZmphx56SDU1NZKknTt36rbbblPfvn01bNgwvfjii16PPdOx3Nqdrq/ffvutUlJSTjpWX3jhBc9jW2xf3Qg4o0ePdj/yyCPumpoa9%2B7du93Dhg1zv/jii/4uKyB169bNvW/fvpPGv/vuO/cVV1zhLikpcR85csT90UcfuXv27On%2B/PPP/VDl%2Ba24uNg9bNgw97hx49wFBQVeczt27HCnpaW5t2zZ4j5y5Ih71apV7l69ern379/vdrvd7pUrV7oHDx7s3rVrl/vw4cPuJ554wj1y5Eh3U1OTP3blvHKmvr7%2B%2Buvu22677bSPpa%2Bnd%2BONN7ofeughd21trXv//v3uMWPGuGfNmuWur693X3PNNe4lS5a46%2Brq3OXl5e7%2B/fu7N2zY4Ha7z34st3an6%2Bu%2Bffvc3bp1O%2B3jWnJfOYMVYOx2u7744gvNmDFD0dHRSk5O1vjx42Wz2fxdWouyZs0aJScnKycnR%2BHh4crMzFR2drZKSkr8Xdp5Jzw8XKWlpbrssstOmispKVFWVpaysrIUHh6uUaNGqVu3bp6zgTabTePHj1fnzp0VFRWlwsJCVVRUaPv27VbvxnnnTH09G/p6ajU1NUpLS9P06dMVGRmpiy66SKNHj9Ynn3yiLVu26Pjx45oyZYratGmjHj166NZbb/V8bz3bsdyanamvZ9OS%2B0rACjAOh0NJSUmKjY31jPXo0UO7d%2B9WbW2tHysLXEVFRRo0aJCuvPJKzZkzR3V1dXI4HEpNTfXaLjU1VeXl5X6q8vx1xx13KDo6%2BpRzp%2Buj3W7XkSNHtGvXLq/5qKgoXXbZZbLb7T6tORCcqa%2BStH//fv32t79Vv379NGTIEK1atUqS6OsZxMTEaP78%2BWrXrp1nbP/%2B/UpMTJTD4VBKSopCQkI8c//6NX%2BmY7m1O1NfT3jwwQc1YMAAXXXVVSoqKtLx48cltey%2BErACjMvlUkxMjNfYibDldDr9UVJAu%2BKKK5SZmamNGzfKZrPps88%2B0%2BOPP37KPsfFxdHjX8jlcnn9Z0D66Xh1Op06dOiQ3G73aedxegkJCUpOTtYDDzygjz76SPfff79mzZqlsrIy%2BvoL2O12vfrqq5oyZcppv%2BZdLpeamprOeCzD27/2NSwsTL1799bQoUO1efNmFRcXa/Xq1frP//xPSWf%2BHhHoCFgByO12%2B7uEFsNms%2BnWW29VWFiYOnfurBkzZmjt2rWe/13h3J3teOV4/uUGDRqk559/XqmpqQoLC9OIESM0dOhQvfHGG55t6OuZffrpp5owYYKmT5%2BuzMzM024XFBTk%2BTM9Pbuf9zUxMVGvvfaahg4dqgsuuEA9e/bUpEmTWsWxSsAKMAkJCXK5XF5jLpdLQUFBSkhI8FNVLccll1yixsZGBQcHn9Rnp9NJj3%2Bh%2BPj4Ux6vCQkJiouLO2WfXS6X2rZta2WZLUJSUpKqqqroazNs2rRJ99xzj2bNmqU77rhD0k/fW39%2B1sTlcnn6eaZjGT85VV9PJSkpSd9//73cbneL7isBK8CkpaVp//79OnjwoGfMbrerS5cuioyM9GNlgWfHjh1asGCB11hFRYXCwsKUlZV10v1W5eXl6tWrl5UlBry0tLST%2Bmi329WrVy%2BFh4era9eucjgcnrmamhrt3btXPXv2tLrUgPLnP/9Z69at8xqrqKhQx44d6etZ/P3vf9fMmTP17LPP6uabb/aMp6Wl6csvv1RDQ4Nn7MSxemL%2BdMcyTt/XsrIyLVu2zGvbb775RklJSQoKCmrRfSVgBZjU1FSlp6erqKhItbW1qqio0EsvvaS8vDx/lxZw2rZtK5vNpuLiYh07dky7d%2B/Ws88%2Bq9zcXN10002qrKxUSUmJjh49qq1bt2rr1q0aO3asv8sOKGPHjtW2bdu0ZcsWHT16VKWlpdqzZ49GjRolScrLy9PKlStVUVGh2tpaLVq0SN27d1d6erqfKz%2B/HTt2TE8%2B%2BaTsdruOHz%2ButWvX6v3339e4ceMk0dfTaWho0COPPKIZM2ZowIABXnNZWVmKiorSsmXLVF9fr%2B3bt6u0tNTzvfVsx3Jrdqa%2BRkdH67nnntOqVat0/Phx2e12vfDCC62ir0HulnrxswU7cOCA5syZo//%2B7/9WVFSUxo0bp3vvvdfrXgE0z9/%2B9jcVFRXpyy%2B/VFhYmEaPHq3CwkKFh4frb3/7m%2BbNm6eKigolJSVp%2BvTpGjZsmL9LPu%2Bc%2BKF94n/%2BoaGhkuT5LaCNGzeqqKhIlZWV6tKli2bPnq1%2B/fpJ%2BuneiyVLlui1115TXV2dMjIy9MQTT%2Biiiy7yw56cX87UV7fbrWXLlqm0tFTV1dW65JJL9OCDD2rw4MGS6OvpfPLJJ/r3f/93hYWFnTT3zjvvqK6uTo899pjKy8vVrl07TZw4Ub/5zW8825zpWG7NztbXHTt2aOnSpdqzZ4%2Bio6N1%2B%2B23a%2BLEiQoO/ukcT0vtKwELAADAMC4RAgAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBh/w/RR7S6vAAJMAAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-1931379802470597327">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">226</td>
        <td class="number">279</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">229</td>
        <td class="number">269</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">224</td>
        <td class="number">265</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">228</td>
        <td class="number">261</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">230</td>
        <td class="number">260</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">233</td>
        <td class="number">248</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">223</td>
        <td class="number">245</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">219</td>
        <td class="number">242</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">231</td>
        <td class="number">239</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">225</td>
        <td class="number">236</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (166)</td>
        <td class="number">12576</td>
        <td class="number">83.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-1931379802470597327">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">58</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">59</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">65</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">73</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">250</td>
        <td class="number">192</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:96%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">251</td>
        <td class="number">174</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:87%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">252</td>
        <td class="number">189</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:94%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">253</td>
        <td class="number">200</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">254</td>
        <td class="number">190</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:95%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Hillshade_Noon">Hillshade_Noon<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>141</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.9%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>218.97</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>99</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>254</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-210129124746512168">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAASRJREFUeJzt28sJwkAUQFEVS7IIe3JtTxZhT2MDclEh5nfOPjCLXB6TR45jjHEA3jrNfQBYsvPcB2AdLrfH188879cJTvJfJggEgUAQCASBQBAIBIFAEAgEgUCwKNyhX5Z%2Be2WCQBAIBIFAEAgEgUAQCASfeZnMFv4hMUEgCASCQCAIBIJAIAgEgkAgCASCQCDYpG%2BAH6CmY4JAEAgEgUAQCASBQBAIBIFAEAgEgUCwSV8YW/FlMUEgCASCQCAIBIJL%2BoRcuNfPBIEgEAgCgeAO8gV3iv3ZbSBedj5xHGOMuQ8BS%2BUOAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAuEF69sXB77ABKEAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-210129124746512168,#minihistogram-210129124746512168"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-210129124746512168">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-210129124746512168"
                                                  aria-controls="quantiles-210129124746512168" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-210129124746512168" aria-controls="histogram-210129124746512168"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-210129124746512168" aria-controls="common-210129124746512168"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-210129124746512168" aria-controls="extreme-210129124746512168"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-210129124746512168">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>99</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>175</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>207</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>223</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>235</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>250</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>254</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>155</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>28</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>22.802</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.10413</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.1535</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>218.97</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>17.735</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-0.95323</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>3310760</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>519.93</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-210129124746512168">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3XtYVWWix/Efl8QUAVHBDmPZeBsRJK8kmXgpGTWdUlKZ05RHbbpQHMgmvIxlZeqkRg32lNTTnOxiBF285O2YYjdrummAdpHs5OFBIWWLIpjAOn903NMOVJSXtdm77%2Bd5fMz33Xut99eC7c%2B91l74WJZlCQAAAMb4unsBAAAA3oaCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAM83f3An4tysqONev2fX19FBraVkeOVKquzmrWfbkD%2BTyft2ckn2fz9nyS92c8U75Ondq5Zz1u2SuM8/X1kY%2BPj3x9fdy9lGZBPs/n7RnJ59m8PZ/k/RlbWj4KFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAY5u/uBQAA4CnGPP6%2Bu5dwXjamXuXuJfxq8Q4WAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYJhXFKxFixapV69ezj/v3LlTiYmJ6t%2B/v8aNG6e1a9e6PH7VqlVKSEhQ//79lZSUpIKCAufcyZMndf/992vYsGGKjY1VSkqKysvLbcsCAAA8n8cXrL1792rNmjXOP5eWlurOO%2B/U1KlTtXPnTs2bN0/z589Xfn6%2BJGnbtm3KzMzUo48%2Bqg8%2B%2BEAjRozQ7bffrhMnTkiSMjIyVFhYqOzsbG3evFmWZWnOnDluyQYAADyTRxesuro6PfDAA5o2bZpzbN26deratasSExMVEBCguLg4jRw5Ujk5OZKk7OxsTZw4UTExMWrdurVmzpwpSdq%2BfbtqamqUm5urO%2B%2B8U5dccolCQkKUmpqqvLw8HTp0yB0RAQCAB/J39wKa4pVXXlFAQIDGjx%2Bvxx9/XJJUWFioyMhIl8dFRkZq48aNzvmxY8c653x9fdW7d2/l5%2Berd%2B/eOnbsmPr06eOc79atm1q3bq3CwkKFh4c3al2lpaUqKytzGfP3b6OwsLALytkYfn6%2BLr97G/J5Pm/PSD7P5q35/P3/lcdbM57W0vJ5bMH64YcflJmZqRdeeMFl3OFw1CtCISEhzuuoHA6HgoODXeaDg4NVXl4uh8MhSQoKCnKZDwoKOq/rsLKzs7VixQqXseTkZKWkpDR6GxcqKOjiZt%2BHO5HP83l7RvJ5Nm/L175923pj3pbxl1pKPo8tWIsXL9bEiRPVvXt3/e///u95PdeyrCbNn8uUKVM0cuRIlzF//zYqL69s0nbPxs/PV0FBF6uiokq1tXXNth93IZ/n8/aM5PNs3prv53/veGvG086Ur6GSaQePLFg7d%2B7U559/rvXr19eba9%2B%2BvfOdqNPKy8sVGhp6xnmHw6EePXo4H%2BNwONS27b8OyNGjR9WhQ4dGry8sLKze6cCysmOqqWn%2BL%2Bja2jpb9uMu5PN83p6RfJ7N2/I1lMXbMv5SS8nXMk5Unqe1a9fq8OHDGjFihGJjYzVx4kRJUmxsrHr27Oly2wVJKigoUExMjCQpKipKhYWFzrna2lrt2bNHMTEx6tKli4KDg13mv/76a/3444%2BKioqyIRkAAPAGHlmwZs%2Berc2bN2vNmjVas2aNsrKyJElr1qzR%2BPHjVVxcrJycHJ08eVI7duzQjh07NHnyZElSUlKS3nzzTe3atUtVVVV66qmn1KpVKw0fPlx%2Bfn6aPHmynn76aZWUlKi8vFyPPfaYrr32WnXs2NGdkQEAgAfxyFOEwcHBLheq19TUSJI6d%2B4sSVq5cqUWLlyoBx98UBEREVq6dKl%2B97vfSZKGDRume%2B65R6mpqTp8%2BLCio6OVlZWl1q1bS5JSUlJUWVmpP/zhD6qpqdGIESO0YMECewMCAACP5mM19YpuNEpZ2bFm3b6/v6/at2%2Br8vLKFnHu2TTyeT5vz0g%2Bz9bYfGMef9/GVTXdxtSrnP/9az2GnTq1c8t6PPIUIQAAQEtGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADPN39wIAAL9eYx5/391LAJoF72ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZ5bMH68ssvdcstt2jAgAGKi4tTamqqysrKJEk7d%2B5UYmKi%2Bvfvr3Hjxmnt2rUuz121apUSEhLUv39/JSUlqaCgwDl38uRJ3X///Ro2bJhiY2OVkpKi8vJyW7MBAADP5pEF68cff9T06dM1ePBg7dy5U%2BvXr9fhw4e1YMEClZaW6s4779TUqVO1c%2BdOzZs3T/Pnz1d%2Bfr4kadu2bcrMzNSjjz6qDz74QCNGjNDtt9%2BuEydOSJIyMjJUWFio7Oxsbd68WZZlac6cOe6MCwAAPIy/uxdwIaqqqpSWlqYbbrhB/v7%2BCg0N1bXXXqsXX3xR69atU9euXZWYmChJiouL08iRI5WTk6Po6GhlZ2dr4sSJiomJkSTNnDlTq1at0vbt25WQkKDc3Fz97W9/0yWXXCJJSk1N1bhx43To0CGFh4e7LTMAAOdrzOPvu3sJjbYx9Sp3L8EojyxYwcHBuvHGG51//vbbb/XGG29ozJgxKiwsVGRkpMvjIyMjtXHjRklSYWGhxo4d65zz9fVV7969lZ%2Bfr969e%2BvYsWPq06ePc75bt25q3bq1CgsLG12wSktLnacrT/P3b6OwsLDzztpYfn6%2BLr97G/J5Pm/PSD6gafz9m/a11dK%2BRj2yYJ1WXFyshIQE1dTUaPLkyUpJSdGtt95arwiFhIQ4r6NyOBwKDg52mQ8ODlZ5ebkcDockKSgoyGU%2BKCjovK7Dys7O1ooVK1zGkpOTlZKS0uhtXKigoIubfR/uRD7P5%2B0ZyQdcmPbt2xrZTkv5GvXoghUREaH8/Hz9z//8j%2B6//37dd999jXqeZVlNmj%2BXKVOmaOTIkS5j/v5tVF5e2aTtno2fn6%2BCgi5WRUWVamvrmm0/7kI%2Bz%2BftGckHNE1T/44809eoqeJ2vjy6YEmSj4%2BPunbtqrS0NE2dOlXx8fHOd6JOKy8vV2hoqCSpffv29eYdDod69OjhfIzD4VDbtv86IEePHlWHDh0avaawsLB6pwPLyo6ppqb5X5Rqa%2Bts2Y%2B7kM/zeXtG8gEXxtTXVUv5Gm0ZJyrP086dO5WQkKC6un/9D/T1/SlK3759XW67IEkFBQXOi9qjoqJUWFjonKutrdWePXsUExOjLl26KDg42GX%2B66%2B/1o8//qioqKjmjAQAALyIRxasqKgoHT9%2BXEuXLlVVVZWOHDmizMxMDRw4UElJSSouLlZOTo5OnjypHTt2aMeOHZo8ebIkKSkpSW%2B%2B%2BaZ27dqlqqoqPfXUU2rVqpWGDx8uPz8/TZ48WU8//bRKSkpUXl6uxx57TNdee606duzo5tQAAMBTeOQpwnbt2um5557TwoULdeWVV6pNmza68sor9cgjj6hDhw5auXKlFi5cqAcffFARERFaunSpfve730mShg0bpnvuuUepqak6fPiwoqOjlZWVpdatW0uSUlJSVFlZqT/84Q%2BqqanRiBEjtGDBAjemBQAAnsbHauoV3WiUsrJjzbp9f39ftW/fVuXllS3i3LNp5PN83p6RfBfGk%2B7ThObV1PtgnelrtFOndk1d2gXxyFOEAAAALRkFCwAAwDDbC9bIkSO1YsUKlZSU2L1rAAAAW9hesCZNmqQNGzbommuu0cyZM7VlyxbV1NTYvQwAAIBmY3vBSk5O1oYNG/Tqq6%2BqR48eWrRokeLj47V06VLt37/f7uUAAAAY57ZrsPr06aP09HRt375dc%2BfO1auvvqqxY8dqxowZ%2BuKLL9y1LAAAgCZzW8E6deqUNmzYoFtvvVXp6ekKDw/XnDlz1Lt3b02bNk3r1q1z19IAAACaxPYbjRYVFSk3N1dvvvmmKisrlZCQoOeff14DBgxwPmbQoEFasGCBxo8fb/fyAAAAmsz2gjVu3Dhdfvnluu2223T99dcrJCSk3mPi4%2BN15MgRu5cGAABghO0Fa9WqVRo8ePA5H7d7924bVgMAAGCe7ddg9erVS7fffru2bt3qHPuv//ov3XrrrXI4HHYvBwAAwDjbC9bixYt17Ngxde/e3Tk2fPhw1dXVacmSJXYvBwAAwDjbTxG%2B9957Wrdundq3b%2B8c69q1q5YtW6brrrvO7uUAAAAYZ/s7WNXV1QoICKi/EF9fVVVV2b0cAAAA42wvWIMGDdKSJUt09OhR59ihQ4f04IMPutyqAQAAwFPZfopw7ty5mj59uoYMGaLAwEDV1dWpsrJSXbp00QsvvGD3cgAAAIyzvWB16dJFb731lt555x19//338vX11eWXX66hQ4fKz8/P7uUAAAAYZ3vBkqRWrVrpmmuucceuAQAAmp3tBevAgQNavny5vvnmG1VXV9ebf/vtt%2B1eEgAAgFFuuQartLRUQ4cOVZs2bezePQAAQLOzvWAVFBTo7bffVmhoqN27BgAAsIXtt2no0KED71wBAACvZnvBuu2227RixQpZlmX3rgEAAGxh%2BynCd955R5999plef/11/eY3v5Gvr2vHe%2BWVV%2BxeEgAAgFG2F6zAwEANGzbM7t0CAADYxvaCtXjxYrt3CQAAYCvbr8GSpG%2B//VaZmZmaM2eOc%2Bzzzz93x1IAAACMs71g7dy5UxMmTNCWLVu0fv16ST/dfPTmm2/mJqMAAMAr2F6wMjIy9Je//EXr1q2Tj4%2BPpJ9%2BPuGSJUv05JNP2r0cAAAA42wvWF9//bWSkpIkyVmwJOn3v/%2B9ioqK7F4OAACAcbYXrHbt2jX4MwhLS0vVqlUru5cDAABgnO0Fq3///lq0aJGOHz/uHNu/f7/S09M1ZMgQu5cDAABgnO23aZgzZ45uueUWxcbGqra2Vv3791dVVZV69OihJUuW2L0cAAAA42wvWJ07d9b69eu1Y8cO7d%2B/X61bt9bll1%2Buq666yuWaLAAAAE9le8GSpIsuukjXXHONO3YNAADQ7GwvWCNHjjzrO1XcCwsAAHg62wvW2LFjXQpWbW2t9u/fr/z8fN1yyy12LwcAAMA42wvWvffe2%2BD45s2b9dFHH9m8GgAAAPPc8rMIG3LNNdforbfecvcyAAAAmqzFFKw9e/bIsix3LwMAAKDJbD9FOHXq1HpjVVVVKioq0ujRo%2B1eDgAAgHG2F6yuXbvW%2BxRhQECAEhMTdeONN9q9HAAAAONsL1jcrR0AAHg72wvWm2%2B%2B2ejHXn/99c24EgAAgOZhe8GaN2%2Be6urq6l3Q7uPj4zLm4%2BNDwQIAAB7J9oL17LPP6rnnntPtt9%2BuXr16ybIsffXVV3rmmWd00003KTY21u4lAQAAGOWWa7CysrIUHh7uHBs4cKC6dOmiGTNmaP369XYvCQAAwCjb74P13XffKTg4uN54UFCQiouL7V4OAACAcbYXrIiICC1ZskTl5eXOsYqKCi1fvlyXXnqp3csBAAAwzvZThHPnztWsWbOUnZ2ttm3bytfXV8ePH1fr1q315JNP2r0cAAAA42wvWEOHDlVeXp527NihgwcPyrIshYeH6%2Bqrr1a7du3sXg4AAIBxthcsSbr44os1atQoHTx4UF26dHHHEgAAAJqN7ddgVVdXKz09Xf369dOYMWMk/XQN1syZM1VRUWH3cgAAAIyzvWAtXbpUe/fu1bJly%2BTr%2B6/d19bWatmyZXYvBwAAwDjbC9bmzZv197//Xb///e%2BdP/Q5KChIixcv1pYtW%2BxeDgAAgHG2F6zKykp17dq13nhoaKhOnDhh93IAAACMs71gXXrppfroo48kyeVnD27atEn/9m//1ujtFBcXKzk5WbGxsYqLi9Ps2bOd13Dt3btXN910kwYMGKDRo0frueeec3nuhg0bNH78ePXr108TJ07Ue%2B%2B955yrq6tTRkaGRo0apUGDBmnGjBk6cOBAUyIDAIBfGds/RfjHP/5Rd999tyZNmqS6ujr94x//UEFBgTZv3qx58%2BY1eju33367oqKitG3bNh07dkzJycn629/%2Bpvnz5%2Bu2227T5MmTlZWVpf3792v69On6zW9%2Bo9GjR2vv3r1KT0/XihUrdOWVV2rz5s266667tGnTJnXu3FkvvfSS1q1bp2eeeUbh4eHKyMhQcnKy1qxZ4zylCQAt2ZjH33f3EoBfPdvfwZoyZYrS09P14Ycfys/PT08//bSKi4u1bNkyJSUlNWobFRUVioqK0qxZs9S2bVt17txZN9xwgz755BPl5eXp1KlTuuOOO9SmTRv16dNHN954o7KzsyVJOTk5io%2BPV3x8vAICAjRhwgT17NlTa9eulSRlZ2dr2rRp6tatmwIDA5WWlqaioiLt3r272f6fAAAA72L7O1hHjhzRpEmTNGnSpAvexumL4n%2BupKREYWFhKiwsVK9eveTn5%2Beci4yMVE5OjiSpsLBQ8fHxLs%2BNjIxUfn6%2BqqurtW/fPkVGRjrnAgMDddlllyk/P19XXHHFBa8ZAAD8ethesEaNGqXPPvvM6Om2/Px8vfjii3rqqae0ceNGBQUFucyHhITI4XCorq5ODoej3g%2BbDg4O1r59%2B3T06FFZltXg/M9/duK5lJaWqqyszGXM37%2BNwsLCzjNZ4/n5%2Bbr87m3I5/m8PaO35wOam79/0753Wtr3oO0FKzY2Vhs3btTYsWONbO/TTz/VHXfcoVmzZikuLk4bN25s8HE/L3Q/v7i%2BIeeaP5fs7GytWLHCZSw5OVkpKSlN2m5jBAVd3Oz7cCfyeT5vz%2Bjt%2BYDm0r59WyPbaSnfg7YXrEsuuUSPPPKIsrKydOmll%2Bqiiy5ymV%2B%2BfHmjt7Vt2zb95S9/0fz583X99ddL%2Bul2D999953L4xwOh0JCQuTr66v27dvL4XDUmw8NDXU%2BpqH5Dh06NHpdU6ZM0ciRI13G/P3bqLy8stHbOF9%2Bfr4KCrpYFRVVqq2ta7b9uAv5PJ%2B3Z/T2fEBza%2BrfkWf6HjRV3M6X7QVr3759%2Bu1vfytJ53Xa7Zc%2B%2B%2Bwzpaen64knntDQoUOd41FRUVq9erVqamrk7/9TvPz8fMXExDjnCwoKXLaVn5%2BvcePGKSAgQD169FBhYaEGDx4s6acL6r///nv17du30WsLCwurdzqwrOyYamqa/0W3trbOlv24C/k8n7dn9PZ8QHMx9X3TUr4HbStYaWlpysjI0AsvvOAce/LJJ5WcnHze26qpqdFf//pX3XvvvS7lSpLi4%2BMVGBiop556SjNnztTXX3%2Bt3NxcLV26VJI0efJkJSYmKi8vT0OGDNG6dev03XffacKECZKkpKQkZWVladiwYQoPD9eyZcvUu3dvRUdHNyE9AAD4NbGtYG3btq3eWFZW1gUVrF27dqmoqEgLFy7UwoULXeY2bdqkp59%2BWg888ICysrLUsWNHpaWlafjw4ZKknj17atmyZVq8eLGKi4vVvXt3rVy5Up06dZIkTZ06VWVlZfrTn/6kyspKxcbG1rueCgAA4GxsK1gNXTh%2BoReTDxw4UF999dVZH7N69eozzo0ePVqjR49ucM7Hx0cpKSm2XJAOAAC8k22fZWzotgzcGR0AAHijlnGzCAAAAC9CwQIAADDMtmuwTp06pVmzZp1z7HzugwUAANAS2VawBgwYoNLS0nOOAQAAeDrbCtbP738FAADgzbgGCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEeXbDeffddxcXFKS0trd7chg0bNH78ePXr108TJ07Ue%2B%2B955yrq6tTRkaGRo0apUGDBmnGjBk6cOCAc97hcCg1NVVxcXEaOnSo5s2bp%2BrqalsyAQAAz%2BexBeuZZ57RwoULddlll9Wb27t3r9LT03Xvvffqww8/1LRp03TXXXfp4MGDkqSXXnpJ69atU1ZWlrZv366uXbsqOTlZlmVJkubPn6%2BqqiqtX79er732moqKirRs2TJb8wEAAM/lsQUrICBAubm5DRasnJwcxcfHKz4%2BXgEBAZowYYJ69uyptWvXSpKys7M1bdo0devWTYGBgUpLS1NRUZF2796tH374QVu3blVaWppCQ0MVHh6uO%2B%2B8U6%2B99ppOnTpld0wAAOCBPLZg3XzzzWrXrl2Dc4WFhYqMjHQZi4yMVH5%2Bvqqrq7Vv3z6X%2BcDAQF122WXKz8/X3r175efnp169ejnn%2B/TpoxMnTujbb79tnjAAAMCr%2BLt7Ac3B4XAoODjYZSw4OFj79u3T0aNHZVlWg/Pl5eUKCQlRYGCgfHx8XOYkqby8vFH7Ly0tVVlZmcuYv38bhYWFXUicRvHz83X53duQz/N5csZrl73r7iUAXs/fv2mvDS3tNcYrC5Yk5/VUFzJ/rueeS3Z2tlasWOEylpycrJSUlCZttzGCgi5u9n24E/k8368hI4Dz1759WyPbaSmvMV5ZsNq3by%2BHw%2BEy5nA4FBoaqpCQEPn6%2BjY436FDB4WGhur48eOqra2Vn5%2Bfc06SOnTo0Kj9T5kyRSNHjnQZ8/dvo/LyyguNdE5%2Bfr4KCrpYFRVVqq2ta7b9uAv5PN%2BvISOAC9fUvyPP9BpjqridL68sWFFRUSooKHAZy8/P17hx4xQQEKAePXqosLBQgwcPliRVVFTo%2B%2B%2B/V9%2B%2BfRURESHLsvTll1%2BqT58%2BzucGBQXp8ssvb9T%2Bw8LC6p0OLCs7ppqa5v9Lpba2zpb9uAv5PN%2BvISOA82fqdaGlvMa0jBOVhk2ePFkffPCB8vLydPLkSeXm5uq7777ThAkTJElJSUlatWqVioqKdPz4cS1btky9e/dWdHS0QkNDlZCQoMcff1xHjhzRwYMH9eSTTyoxMVH%2B/l7ZRwEAgGEe2xiio6MlSTU1NZKkrVu3Svrp3aaePXtq2bJlWrx4sYqLi9W9e3etXLlSnTp1kiRNnTpVZWVl%2BtOf/qTKykrFxsa6XDP10EMP6YEHHtCoUaN00UUX6brrrmvwZqYAAAAN8bGaekU3GqWs7Fizbt/f31ft27dVeXlli3hr1DTyeT5Pzjjm8ffdvQTA621MvapJzz/Ta0ynTg3f0qm5eeUpQgAAAHeiYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAzzd/cCAPz6jHn8fXcvAQCaFe9gAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYf7uXgAAM8Y8/r67lwAA%2BH%2B8gwUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGHdyb0BxcbEefPBB7d69W23atNHYsWM1a9Ys%2BfrSR39NuDM6AOBCUbAacPfdd6tPnz7aunWrDh8%2BrNtuu00dO3bUf/zHf7h7aQAAwAPwlswv5Ofn68svv9S9996rdu3aqWvXrpo2bZqys7PdvTQAAOAheAfrFwoLCxUREaHg4GDnWJ8%2BfbR//34dP35cgYGB59xGaWmpysrKXMb8/dsoLCzM%2BHqvXfau8W0CAGA3f/%2Bmvefj5%2Bfr8ru7UbB%2BweFwKCgoyGUrcNZcAAANYUlEQVTsdNkqLy9vVMHKzs7WihUrXMbuuusu3X333eYW%2Bv8%2BeeT3kn4qddnZ2ZoyZUqzFDl3I5/n8/aM5PNs3p5P8v6MpaWlev75Z1tMvpZR81oYy7Ka9PwpU6bo9ddfd/k1ZcoUQ6trWFlZmVasWFHvnTNvQT7P5%2B0ZyefZvD2f5P0ZW1o%2B3sH6hdDQUDkcDpcxh8MhHx8fhYaGNmobYWFhLaI9AwAA9%2BAdrF%2BIiopSSUmJjhw54hzLz89X9%2B7d1bZtWzeuDAAAeAoK1i9ERkYqOjpay5cv1/Hjx1VUVKR//OMfSkpKcvfSAACAh/BbsGDBAncvoqW5%2BuqrtX79ej388MN66623lJiYqBkzZsjHx8fdSzurtm3bavDgwV77Thv5PJ%2B3ZySfZ/P2fJL3Z2xJ%2BXyspl7RDQAAABecIgQAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjILlAd59913FxcUpLS2t3tyGDRs0fvx49evXTxMnTtR7773nnKurq1NGRoZGjRqlQYMGacaMGTpw4ICdS2%2BUs%2BXbsmWLJkyYoH79%2BikhIUGvvvqqy/yqVauUkJCg/v37KykpSQUFBXYt%2B7ycLeNplZWVGj58uGbPnu0c84ZjeOjQId1xxx264oorFBcXp%2BXLl6uurk6Sd%2BR76aWXlJCQ4PwafeGFF5xznpKvuLhYycnJio2NVVxcnGbPnq2KigpJ0t69e3XTTTdpwIABGj16tJ577jmX557tNailOFu%2BL7/8UtOmTdPAgQM1bNgwPfLII/rxxx%2Bdz925c6cSExPVv39/jRs3TmvXrnVXjLM6W8afS05O1siRI13GPP0Ynjp1SosWLVJsbKz69%2B%2BvlJQUORwO53Pdls9Ci5aVlWWNHj3amjp1qpWamuoyt2fPHisqKsrKy8uzqqurrTVr1lgxMTFWSUmJZVmWtWrVKmvEiBHWvn37rGPHjlkPPfSQNX78eKuurs4dURp0tny7d%2B%2B2oqOjrf/%2B7/%2B2Tp06ZeXl5Vl9%2BvSxPv74Y8uyLOvtt9%2B2Bg4caO3atcuqqqqyVq5caV111VVWZWWlO6Kc0dky/tzixYutAQMGWOnp6c4xTz%2BGdXV1VmJiovXwww9bx44ds/bt22dNmjTJ%2BuCDDyzL8vx8eXl5VkxMjLVr1y6rtrbW2rVrlxUTE2Nt377dsizPyGdZlnXddddZs2fPto4fP26VlJRYEydOtObOnWtVVVVZV199tZWZmWlVVlZaBQUF1uDBg63NmzdblnXu16CW4kz5jh8/bl111VXWY489Zp08edLat2%2BfNWLECOvJJ5%2B0LMuyDh06ZF1xxRVWTk6OVV1dbb3//vtW3759rS%2B%2B%2BMLNieo7U8af27ZtmzVgwABrxIgRzjFPP4aWZVlLliyxkpKSrIMHD1qHDx%2B2UlNTrZUrV1qW5d58vIPVwgUEBCg3N1eXXXZZvbmcnBzFx8crPj5eAQEBmjBhgnr27On8F1Z2dramTZumbt26KTAwUGlpaSoqKtLu3bvtjnFGZ8vncDh022236ZprrpG/v7/i4%2BPVs2dPffLJJ5J%2Byjdx4kTFxMSodevWmjlzpiRp%2B/bttmY4l7NlPO3LL7/U%2BvXrdcMNN7iMe/ox/Pjjj3XgwAHdd999CgwMVLdu3ZSbm6shQ4ZI8vx8BQUF6tGjh2JiYuTr66uYmBj17NlTe/bskeQZ%2BSoqKhQVFaVZs2apbdu26ty5s2644QZ98sknysvL06lTp3THHXeoTZs26tOnj2688UZlZ2dLOvdrUEtwtnyHDx/W1VdfrbvvvlutWrVSt27dlJCQ4HyNWbdunbp27arExEQFBAQoLi5OI0eOVE5OjptTuTpbxtOqqqr08MMPa/r06S7P9fRjWF1drdWrV2vevHkKDw9XaGioMjIy9Oc//1mSe/NRsFq4m2%2B%2BWe3atWtwrrCwUJGRkS5jkZGRys/PV3V1tfbt2%2BcyHxgYqMsuu0z5%2BfnNuubzcbZ8w4YNU3JysvPPNTU1KisrU3h4uKT6%2BX19fdW7d%2B8WlU86e0ZJsixLCxYsUFpamoKCgpzj3nAMP/30U/Xs2VMZGRmKjY3VqFGjnKeYvCHf1VdfrX379umjjz7Sjz/%2BqM8//1xFRUUaOnSox%2BQLCgrS4sWL1bFjR%2BdYSUmJwsLCVFhYqF69esnPz885FxkZ6TwVf7bXoJbibPkuvfRSLV68WP7%2B/i5zZ3qNkVzztxRny3jaihUrNGjQIA0YMMDluZ5%2BDAsLC1VTU6NvvvlGo0aN0pAhQ/TXv/5VJ06ckOTefBQsD%2BZwOBQcHOwyFhwcrPLych09elSWZZ1x3hMtW7ZMbdq00dixYyWdPb8nyc7Olo%2BPjyZOnOgy7g3H8ODBg9q1a5c6dOigvLw83X///crIyNDWrVu9Il/fvn01Z84cTZ8%2BXdHR0brpppuUmpqqvn37emy%2B/Px8vfjii7rjjjvkcDhcSr8khYSEyOFwqK6uziO/B3%2Be75fefvttbd%2B%2B3fkuz5nyt%2BR8Uv2MX3/9td544w3dd9999R7r6cfw0KFDkn66TvK1117Tiy%2B%2BqH/%2B85/KyMiQ5N58FCwPZ1lWk%2BY9gWVZWrp0qdavX6%2BnnnpKAQEBLnOe7PDhw3riiSe0YMEC%2Bfj4NPgYT85oWZZCQ0M1c%2BZMXXzxxYqPj9e1116rjRs3ujzGU3344Ydavny5nn32WX3xxRd6/vnn9fTTT2vr1q3Ox3hSvk8//VQzZszQrFmzFBcXd8bH/fxr1VvybdmyRffee68effRR9ejRw00rbLpfZjz9Dvldd92lDh06NPgcTz6GlmXp1KlTSk1NVUhIiLp166bp06e3iNcYCpYHa9%2B%2BvcsnJaSf2npoaKhCQkLk6%2Bvb4PyZvslaorq6Os2ePVvbtm3T6tWr9dvf/tY5d7b8nmLJkiW6/vrr1atXr3pz3nAMO3XqVO/0WkREhMrKyrwi3%2BrVqzV69GgNGTJEAQEBGjhwoMaNG6fc3FyPy7dt2zb9%2Bc9/1ty5c3XzzTdLkkJDQ%2Bv9S9/hcDizedL3YEP5TsvOzta8efOUmZmphIQE53hD%2BcrLy1tkPqnhjLm5uaqpqdHUqVMbfI6nH8PTpw1//joTERGhI0eOyLIst%2BajYHmwqKioetcC5OfnKyYmRgEBAerRo4cKCwudcxUVFfr%2B%2B%2B/Vt29fu5d6wRYtWqRvvvlGq1evVpcuXVzmoqKiXPLV1tZqz549iomJsXuZF2zt2rXKzc1VbGysYmNj9eyzz%2Bqtt95SbGysVxzDbt266cCBA6qsrHSOFRcXKyIiwivy1dXVqba21mXs9Ef8PSnfZ599pvT0dD3xxBO6/vrrneNRUVH66quvVFNT4xw7/Rpzev5Mr0EtyZnySdKmTZuUkZGhVatWaejQoS5z0dHR9fIVFBS0uHzSmTOuXbtW33zzjYYMGaLY2FjdeeedKikpUWxsrD799FOPP4bdunWTj4%2BP9u7d6xwrLi5W586d5ePj4958zf45RRiRnp5e7yPiX331lRUdHW1t377dqq6utnJycqx%2B/fpZpaWllmVZ1ssvv2wNHz7c%2BRHx%2BfPnW5MmTXLH8s%2BpoXyffPKJNWjQIKusrKzB5%2BzYscMaMGCA9fnnn1snTpywMjMzrfj4eKuqqsqOJZ%2B3hjKWlJS4/Fq0aJGVkpLi/Aixpx/D0x/zv//%2B%2B63Kykrrgw8%2BsKKjo61//vOflmV5fr7XX3/d6t%2B/v/Xxxx9bp06dsnbv3m0NHjzYys3NtSzLM/KdOnXKGjNmjPXKK6/Umzt58qQ1YsQI6%2B9//7t14sQJa9euXdbAgQOdt6E412tQS3C2fBUVFVZsbKz1zjvvNPjcH374werXr5/16quvWtXV1VZeXp7Vt29fa%2B/evc297PNytoyHDx92eY3ZsGGDNWzYMKukpMQ6efKkxx9Dy7Ks5ORkKzEx0SotLbW%2B//57a/To0VZmZqZlWe79GvWxLA86%2BforFB0dLUnOf0Ge/rTL6U9AbNmyRcuXL1dxcbG6d%2B%2BuefPmadCgQZJ%2BOu%2BcmZmpV155RZWVlYqNjdVDDz2kzp07uyFJw86Wb%2B7cuXrjjTdcPuEjSYMGDXJ%2BEu3ll19WVlaWDh8%2BrOjoaC1YsEA9e/a0McG5nesY/lxmZqaKi4u1ZMkSSZ5/DKWfLrB94IEHVFhYqNDQUP3nf/6n83YU3pDv%2Beef18svv6xDhw4pPDxckydP1vTp0%2BXj4%2BMR%2BT755BP9%2B7//u1q1alVvbtOmTaqsrNQDDzyggoICdezYUbfeeqv%2B%2BMc/Oh9zttegluBs%2BR566CHNnj27wbnTx/fjjz/WwoULVVRUpIiICM2aNUujR49u9nWfj3Mdw4iICOefP/roI82ZM0fbtm1zjnnyMdy0aZPatWunBQsWKC8vT35%2BfkpMTNQ999yjiy66SJL78lGwAAAADOMaLAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAw7P8ASO3RXGc3d6AAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-210129124746512168">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">225</td>
        <td class="number">327</td>
        <td class="number">2.2%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">229</td>
        <td class="number">324</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">226</td>
        <td class="number">320</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">224</td>
        <td class="number">313</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">230</td>
        <td class="number">311</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">223</td>
        <td class="number">303</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">232</td>
        <td class="number">298</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">222</td>
        <td class="number">297</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">228</td>
        <td class="number">294</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">218</td>
        <td class="number">293</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (131)</td>
        <td class="number">12040</td>
        <td class="number">79.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-210129124746512168">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">99</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">102</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">103</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">107</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">111</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">250</td>
        <td class="number">167</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:91%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">251</td>
        <td class="number">183</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">252</td>
        <td class="number">152</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:83%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">253</td>
        <td class="number">163</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:89%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">254</td>
        <td class="number">133</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:72%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Horizontal_Distance_To_Fire_Points">Horizontal_Distance_To_Fire_Points<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>2710</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>17.9%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1511.1</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>6993</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram6444980024487249378">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAATNJREFUeJzt3UGuAUEUQFHEkizi7%2BmP7cki7KlsQG4Q3V04Zy6pyc0r6UfvxxhjB9x12PoAMLPj1gd4h9P/5enPXM9/C5yEb2OCQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQJhum/eVzVxYigkCQSAQprtircWPrHiECQJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQfnZZ8RXPLjhabvx8JggEgUAQCASBQBAIBIFAEAgEgUAQCARP0hfkr4U%2BnwkCQSAQXLEm41o2FxMEggnyBdZ4ZcSvTimB8JC13tsyW4j7McbY%2BhAwK99BIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAINwA3aYfBIZMNx4AAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives6444980024487249378,#minihistogram6444980024487249378"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives6444980024487249378">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles6444980024487249378"
                                                  aria-controls="quantiles6444980024487249378" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram6444980024487249378" aria-controls="histogram6444980024487249378"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common6444980024487249378" aria-controls="common6444980024487249378"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme6444980024487249378" aria-controls="extreme6444980024487249378"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles6444980024487249378">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>296.9</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>730</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1256</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>1988.2</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3663</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>6993</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>6993</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1258.2</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>1099.9</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.72788</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>3.3854</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1511.1</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>818.9</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.6171</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>22848547</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>1209900</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram6444980024487249378">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl4lOWh//9PFklkyYYELFLxsJlkQtijAQkEBQFBZQvpZZUWFCWaE9QWhKJQqNACRQseIXr0iHouI7SVpQL%2BLEtd0h1tElBLxCOmwUSYIQtJIMn9%2B8Mv0w6BGuDODDPzfl1XLpv7nnnm/sxMyifP88yTEGOMEQAAAKwJ9fUCAAAAAg0FCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYFu7rBQSLiooq69sMDQ1RXFw7HT9eo6YmY337l6tgzS0Fb3ZyB1duKXizB2tuqfWyd%2BrUwdq2LgR7sPxYaGiIQkJCFBoa4uuleFWw5paCNzu5gyu3FLzZgzW3FHjZKVgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAlvltwerTp48cDoeSk5PdX0uXLpUkFRQUaMqUKRowYIDGjx%2BvrVu3etx348aNGjNmjAYMGKCsrCwVFRW55%2Brr6/X4449r%2BPDhSk1NVU5OjpxOp1ezAQAA/%2BbXf4tw586duuaaazzGysvLNWfOHC1cuFATJkzQX/7yFz3wwAO67rrrlJycrN27d2vt2rV6/vnn1adPH23cuFH333%2B/3nrrLbVt21Zr1qxRcXGx8vPzdeWVV2rRokV67LHHtH79eh%2BlBAAA/sZv92Cdz7Zt29S9e3dNmTJFERERSktLU0ZGhjZt2iRJys/P16RJk5SSkqLIyEjNmjVLkrRnzx41NDRo8%2BbNmjNnjq6%2B%2BmrFxMQoNzdXe/fu1ZdffunLWAAAwI/49R6s1atXa//%2B/aqurtbYsWM1f/58FRcXKzEx0eN2iYmJ2rFjhySpuLhY48aNc8%2BFhoYqISFBhYWFSkhIUFVVlZKSktzzPXr0UGRkpIqLi9W5c%2BcWrau8vFwVFRUeY%2BHhbRUfH3%2BxUc8pLCzU47/BIlhzS8GbndzBlVsK3uzBmlsKvOx%2BW7D69euntLQ0/fSnP9WRI0eUm5urJUuWyOVyNStCMTEx7vOoXC6XoqOjPeajo6PldDrlcrkkSVFRUR7zUVFRF3QeVn5%2BvtatW%2Bcxlp2drZycnBZv40JERV3ZKtu1bdDCnb5ewgX5809u9fUSzstfXnPbyB18gjV7sOaWAie73xas/Px89//u0aOHHn30UT3wwAMaOHDgN97XGHNJ898kMzNTGRkZHmPh4W3ldNZc0nbPFhYWqqioK1VZWavGxiar24asv142BOtrTu7gyi0Fb/ZgzS21XvbY2HbWtnUh/LZgne2aa65RY2OjQkND3XuiznA6nYqLi5MkxcbGNpt3uVzq1auX%2BzYul0vt2v3zBTlx4oQ6duzY4rXEx8c3OxxYUVGlhobW%2BWFpbGxqtW0Hs8v5OQ3W15zcwSdYswdrbilwsvvlgc4DBw5oxYoVHmMlJSVq06aN0tPTPS67IElFRUVKSUmRJDkcDhUXF7vnGhsbdeDAAaWkpKhbt26Kjo72mP/kk0906tQpORyOVkwEAAACiV8WrI4dOyo/P195eXk6deqUDh8%2BrKefflqZmZm6/fbbVVpaqk2bNqm%2Bvl779u3Tvn37NG3aNElSVlaW3njjDX3wwQeqra3Vs88%2BqzZt2mjEiBEKCwvTtGnTtH79epWVlcnpdOrnP/%2B5brnlFl111VU%2BTg0AAPyFXx4i7Ny5s/Ly8rR69Wp3Qbrzzjs1d%2B5cRUREaMOGDVq2bJmWLFmirl27auXKlbr%2B%2BuslScOHD9fDDz%2Bs3NxcHTt2TMnJycrLy1NkZKQkKScnRzU1Nbr99tvV0NCgkSNHavHixT5MCwAA/E2IudQzutEiFRVV1rcZHh6q2Nh2cjpr/OJ49din3vP1Ei7Ijtyhvl5CM/72mttC7uDKLQVv9mDNLbVe9k6dOljb1oXwy0OEAAAAlzMKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUBUbCefPJJ9enTx/19QUGBpkyZogEDBmj8%2BPHaunWrx%2B03btyoMWPGaMCAAcrKylJRUZF7rr6%2BXo8//riGDx%2Bu1NRU5eTkyOl0ei0LAADwf35fsA4ePKgtW7a4vy8vL9ecOXM0ffp0FRQUaOHChVq0aJEKCwslSbt379batWv1s5/9TO%2B//75Gjhyp%2B%2B%2B/XydPnpQkrVmzRsXFxcrPz9euXbtkjNFjjz3mk2wAAMA/%2BXXBampq0hNPPKEZM2a4x7Zt26bu3btrypQpioiIUFpamjIyMrRp0yZJUn5%2BviZNmqSUlBRFRkZq1qxZkqQ9e/aooaFBmzdv1pw5c3T11VcrJiZGubm52rt3r7788ktfRAQAAH4o3NcLuBSvvfaaIiIiNGHCBD311FOSpOLiYiUmJnrcLjExUTt27HDPjxs3zj0XGhqqhIQEFRYWKiEhQVVVVUpKSnLP9%2BjRQ5GRkSouLlbnzp1btK7y8nJVVFR4jIWHt1V8fPxF5TyfsLBQj//CrvDwy%2B95DdbXnNzBlVsK3uzBmlsKvOx%2BW7C%2B%2BuorrV27Vi%2B//LLHuMvlalaEYmJi3OdRuVwuRUdHe8xHR0fL6XTK5XJJkqKiojzmo6KiLug8rPz8fK1bt85jLDs7Wzk5OS3exoWIirqyVbYb7GJj2/l6CecVrK85uYNPsGYP1txS4GT324K1fPlyTZo0ST179tQXX3xxQfc1xlzS/DfJzMxURkaGx1h4eFs5nTWXtN2zhYWFKirqSlVW1qqxscnqtiHrr5cNwfqakzu4ckvBmz1Yc0utl91Xvyz7ZcEqKCjQ/v37tX379mZzsbGx7j1RZzidTsXFxZ133uVyqVevXu7buFwutWv3zxfkxIkT6tixY4vXFx8f3%2BxwYEVFlRoaWueHpbGxqdW2Hcwu5%2Bc0WF9zcgefYM0erLmlwMnulwc6t27dqmPHjmnkyJFKTU3VpEmTJEmpqanq3bu3x2UXJKmoqEgpKSmSJIfDoeLiYvdcY2OjDhw4oJSUFHXr1k3R0dEe85988olOnTolh8PhhWQAACAQ%2BGXBmj9/vnbt2qUtW7Zoy5YtysvLkyRt2bJFEyZMUGlpqTZt2qT6%2Bnrt27dP%2B/bt07Rp0yRJWVlZeuONN/TBBx%2BotrZWzz77rNq0aaMRI0YoLCxM06ZN0/r161VWVian06mf//znuuWWW3TVVVf5MjIAAPAjfnmIMDo62uNE9YaGBklSly5dJEkbNmzQsmXLtGTJEnXt2lUrV67U9ddfL0kaPny4Hn74YeXm5urYsWNKTk5WXl6eIiMjJUk5OTmqqanR7bffroaGBo0cOVKLFy/2bkAAAODXQsylntGNFqmoqLK%2BzfDwUMXGtpPTWeMXx6vHPvWer5dwQXbkDvX1Eprxt9fcFnIHV24peLMHa26p9bJ36tTB2rYuhF8eIgQAALicUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBl4b5eAC7NoIU7fb0EAABwFvZgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgWbivFwBcrsY%2B9Z6vl3BBduQO9fUSAAD/j9f3YGVkZGjdunUqKyvz9kMDAAB4hdcL1uTJk/Xmm2/q5ptv1qxZs/TWW2%2BpoaHB28sAAABoNV4vWNnZ2XrzzTf1%2Buuvq1evXnryySeVnp6ulStX6vDhw95eDgAAgHU%2BO8k9KSlJ8%2BbN0549e7RgwQK9/vrrGjdunGbOnKm//e1vvloWAADAJfNZwTp9%2BrTefPNN3XvvvZo3b546d%2B6sxx57TAkJCZoxY4a2bdvmq6UBAABcEq9/irCkpESbN2/WG2%2B8oZqaGo0ZM0YvvfSSBg4c6L7N4MGDtXjxYk2YMMHbywMAALhkXt%2BDNX78eO3du1ezZ8/W7373O61cudKjXElSenq6jh8//m%2B389FHH%2Bmee%2B7RwIEDlZaWptzcXFVUVEiSCgoKNGXKFA0YMEDjx4/X1q1bPe67ceNGjRkzRgMGDFBWVpaKiorcc/X19Xr88cc1fPhwpaamKicnR06n01J6AAAQDLxesDZu3KgdO3ZoxowZiomJOe/tPvzww/POnTp1St///vc1ZMgQFRQUaPv27Tp27JgWL16s8vJyzZkzR9OnT1dBQYEWLlyoRYsWqbCwUJK0e/durV27Vj/72c/0/vvva%2BTIkbr//vt18uRJSdKaNWtUXFys/Px87dq1S8YYPfbYY3afBAAAENC8XrD69Omj%2B%2B%2B/X2%2B//bZ77H/%2B53907733yuVytWgbtbW1mjt3rmbPnq02bdooLi5Ot9xyi/7%2B979r27Zt6t69u6ZMmaKIiAilpaUpIyNDmzZtkiTl5%2Bdr0qRJSklJUWRkpGbNmiVJ2rNnjxoaGrR582bNmTNHV199tWJiYpSbm6u9e/fqyy%2B/tP9kAACAgOT1c7CWL1%2Buqqoq9ezZ0z02YsQIvfPOO1qxYoVWrFjxjduIjo7W1KlT3d9/%2Bumn%2BvWvf62xY8equLhYiYmJHrdPTEzUjh07JEnFxcUaN26cey40NFQJCQkqLCxUQkKCqqqqlJSU5J7v0aOHIiMjVVxcrM6dO7coY3l5uftw5Rnh4W0VHx/fovu3VFgYf%2BkI/xQeHrjvhzPv9WB7zwdrbil4swdrbinwsnu9YL377rvatm2bYmNj3WPdu3fXqlWrdNttt13QtkpLSzVmzBg1NDRo2rRpysnJ0b333tusCMXExLjPo3K5XIqOjvaYj46OltPpdO9Bi4qK8piPioq6oPOw8vPztW7dOo%2Bx7Oxs5eTktHgbwIWKjW3n6yW0uqioK329BJ8I1txS8GYP1txS4GT3esGqq6tTREREs/HQ0FDV1tZe0La6du2qwsJC/d///Z8ef/xx/fCHP2zR/YwxlzT/TTIzM5WRkeExFh7eVk5nzSVt92yB0vJhh%2B331%2BUkLCxUUVFXqrKyVo2NTb5ejtcEa24peLMHa26p9bL76pdPrxeswYMHa8WKFXrkkUfce5K%2B/PJL/fSnP232acKWCAkJUffu3TV37lxNnz5d6enpzc7lcjqdiouLkyTFxsY2m3e5XOrVq5f7Ni6XS%2B3a/fMFOXHihDp27NjiNcXHxzc7HFhRUaWGhuD6YYF3BcP7q7GxKShyni1Yc0vBmz1Yc0uBk93ru0AWLFiggoIC3XjjjRoyZIgGDRqkESNGqKioSMuWLWvRNgoKCjRmzBg1Nf3zBQgN/TpK3759PS67IElFRUVKSUmRJDkcDhUXF7vnGhsbdeDAAaWkpKhbt26Kjo72mP/kk0906tQpORyOi84MAACCi9f3YHXr1k2/%2Bc1v9Lvf/U6ff/65QkNDdd1112nYsGEKCwtr0TYcDoeqq6u1cuVK5eTkqLa2VmvXrtWgQYOUlZWlF154QZs2bdLEiRP1%2B9//Xvv27VN%2Bfr4kKSsrSw8//LBuu%2B029enTR//93/%2BtNm3aaMSIEQoLC9O0adO0fv16JScnKzIyUj//%2Bc91yy236KqrrmrNpwUAAAQQrxcsSWrTpo1uvvnmi75/hw4d9MILL2jZsmW64YYb1LZtW91www36yU9%2Boo4dO2rDhg1atmyZlixZoq5du2rlypW6/vrrJUnDhw/Xww8/rNzcXB07dkzJycnKy8tTZGSkJCknJ0c1NTW6/fbb1dDQoJEjR2rx4sU2YgMAgCARYi71jO4LdOTIEa1evVp///vfVVdX12z%2Bt7/9rTeX4zUVFVXWtxkeHqpbVr1jfbvwTztyh/p6Ca0mPDxUsbHt5HTWBMS5GS0VrLml4M0erLml1sveqVMHa9u6EF7fg7VgwQKVl5dr2LBhatu2rbcfHgAAoNV5vWAVFRXpt7/9rfsTewAAAIHG658i7NixI3uuAABAQPN6wZo9e7bWrVt3yRfzBAAAuFx5/RDh7373O/31r3/Vr371K11zzTXu61ed8dprr3l7SQAAAFZ5vWC1b99ew4cP9/bDAgAAeI3XC9by5cu9/ZAAAABe5ZO/Fvzpp59q7dq1euyxx9xj%2B/fv98VSAAAArPN6wSooKNDEiRP11ltvafv27ZK%2Bvvjo3XffHbAXGQUAAMHF6wVrzZo1%2BsEPfqBt27YpJCRE0td/n3DFihV65plnvL0cAAAA67xesD755BNlZWVJkrtgSdKtt96qkpISby8HAADAOq8XrA4dOpzzbxCWl5erTZs23l4OAACAdV4vWAMGDNCTTz6p6upq99jhw4c1b9483Xjjjd5eDgAAgHVev0zDY489pnvuuUepqalqbGzUgAEDVFtbq169emnFihXeXg4AAIB1Xi9YXbp00fbt27Vv3z4dPnxYkZGRuu666zR06FCPc7IAAAD8ldcLliRdccUVuvnmm33x0AAAAK3O6wUrIyPj3%2B6p4lpYAADA33m9YI0bN86jYDU2Nurw4cMqLCzUPffc4%2B3lAAAAWOf1gvXoo4%2Bec3zXrl36wx/%2B4OXVAAAA2OeTv0V4LjfffLN%2B85vf%2BHoZAAAAl%2ByyKVgHDhyQMcbXywAAALhkXj9EOH369GZjtbW1Kikp0ejRo729HAAAAOu8XrC6d%2B/e7FOEERERmjJliqZOnert5QAAAFjn9YLF1doBAECg83rBeuONN1p82zvuuKMVVwIAANA6vF6wFi5cqKampmYntIeEhHiMhYSEULAAAIBf8nrBev755/XCCy/o/vvvV58%2BfWSM0ccff6znnntOd911l1JTU729JAAAAKt8cg5WXl6eOnfu7B4bNGiQunXrppkzZ2r79u3eXhIAAIBVXr8O1meffabo6Ohm41FRUSotLfX2cgAAAKzzesHq2rWrVqxYIafT6R6rrKzU6tWr9e1vf9vbywEAALDO64cIFyxYoEceeUT5%2Bflq166dQkNDVV1drcjISD3zzDPeXg4AAIB1Xi9Yw4YN0969e7Vv3z4dPXpUxhh17txZN910kzp06ODt5QAAAFjn9YIlSVdeeaVGjRqlo0ePqlu3br5YAgAAQKvx%2BjlYdXV1mjdvnvr376%2BxY8dK%2BvocrFmzZqmystLbywEAALDO6wVr5cqVOnjwoFatWqXQ0H8%2BfGNjo1atWuXt5QAAAFjn9YK1a9cu/eIXv9Ctt97q/qPPUVFRWr58ud566y1vLwcAAMA6rxesmpoade/evdl4XFycTp486e3lAAAAWOf1gvXtb39bf/jDHyTJ428P7ty5U9/61re8vRwAAADrvP4pwu985zt66KGHNHnyZDU1NenFF19UUVGRdu3apYULF3p7OQAAANZ5vWBlZmYqPDxcr7zyisLCwrR%2B/Xpdd911WrVqlW699VZvLwcAAMA6rxes48ePa/LkyZo8ebK3HxoAAMArvH4O1qhRozzOvQIAAAg0Xi9Yqamp2rFjh7cfFgAAwGu8fojw6quv1k9%2B8hPl5eXp29/%2Btq644gqP%2BdWrV3t7SQAAAFZ5vWAdOnRI//Ef/yFJcjqd3n54AACAVue1gjV37lytWbNGL7/8snvsmWeeUXZ2treWAAAA4BVeOwdr9%2B7dzcby8vK89fAAAABe47WCda5PDvJpQgAAEIi8VrDO/GHnbxoDAADwd16/TAMAAECg89uCVVpaquzsbKWmpiotLU3z589XZWWlJOngwYO66667NHDgQI0ePVovvPCCx33ffPNNTZgwQf3799ekSZP07rvvuueampq0Zs0ajRo1SoMHD9bMmTN15MgRr2YDAAD%2BzWufIjx9%2BrQeeeSRbxxr6XWw7r//fjkcDu3evVtVVVXKzs7WT3/6Uy1atEizZ8/WtGnTlJeXp8OHD%2Bv73/%2B%2BrrnmGo0ePVoHDx7UvHnztG7dOt1www3atWuXHnzwQe3cuVNdunTRq6%2B%2Bqm3btum5555T586dtWbNGmVnZ2vLli0c0gQAAC3itT1YAwcOVHl5ucfXucZaorKyUg6HQ4888ojatWunLl266M4779Sf//xn7d27V6dPn9YDDzygtm3bKikpSVOnTlV%2Bfr4kadOmTUpPT1d6eroiIiI0ceJE9e7dW1u3bpUk5efna8aMGerRo4fat2%2BvuXPnqqSkRB9%2B%2BGGrPTcAACCweG0P1r9e/%2BpSRUVFafny5R5jZWVlio%2BPV3Fxsfr06aOwsDD3XGJiojZt2iRJKi4uVnp6usd9ExMTVVhYqLq6Oh06dEiJiYnuufbt2%2Bvaa69VYWGh%2BvXr16L1lZeXq6KiwmMsPLyt4uPjLyjnNwkL89sjvGgF4eGB%2B344814Ptvd8sOaWgjd7sOaWAi%2B716/k3hoKCwv1yiuv6Nlnn9WOHTsUFRXlMR8TEyOXy6Wmpia5XC5FR0d7zEdHR%2BvQoUM6ceKEjDHnnL%2BQq87n5%2Bdr3bp1HmPZ2dnKycm5wGRAy8XGtvP1ElpdVNSVvl6CTwRrbil4swdrbilwsvt9wfrLX/6iBx54QI888ojS0tLO%2B4ek//X8qW%2B6/talXp8rMzNTGRkZHmPh4W3ldNZc0nbPFigtH3bYfn9dTsLCQhUVdaUqK2vV2Njk6%2BV4TbDmloI3e7Dmllovu69%2B%2BfTrgrV792794Ac/0KJFi3THHXdIkuLi4vTZZ5953M7lcikmJkahoaGKjY2Vy%2BVqNh8XF%2Be%2BzbnmO3bs2OJ1xcfHNzscWFFRpYaG4PphgXcFw/ursbEpKHKeLVhzS8GbPVhzS4GT3W93gfz1r3/VvHnz9PTTT7vLlSQ5HA59/PHHamhocI8VFhYqJSXFPV9UVOSxrTPzERER6tWrl4qLi91zlZWV%2Bvzzz9W3b99WTgQAAAKFXxashoYG/ehHP9Kjjz6qYcOGecylp6erffv2evbZZ1VbW6sPP/xQmzdvVlZWliRp2rRpev/997V3717V19dr8%2BbN%2BuyzzzRx4kRJUlZWljZu3KiSkhJVV1dr1apVSkhIUHJystdzAgAA/%2BSXhwg/%2BOADlZSUaNmyZVq2bJnH3M6dO7V%2B/Xo98cQTysvL01VXXaW5c%2BdqxIgRkqTevXtr1apVWr58uUpLS9WzZ09t2LBBnTp1kiRNnz5dFRUV%2Bu53v6uamhqlpqY2O2EdAADg3wkx/MVlr6ioqLK%2BzfDwUN2y6h3r24V/2pE71NdLaDXh4aGKjW0np7MmIM7NaKlgzS0Fb/ZgzS21XvZOnTpY29aF8MtDhAAAAJczChYAAIBlFCwAAADL/PIkdwDNjX3qPV8vocUC%2BXwxAJDYgwUAAGAdBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACzz64L1zjvvKC0tTXPnzm029%2Babb2rChAnq37%2B/Jk2apHfffdc919TUpDVr1mjUqFEaPHiwZs6cqSNHjrjnXS6XcnNzlZaWpmHDhmnhwoWqq6vzSiYAAOD//LZgPffcc1q2bJmuvfbaZnMHDx7UvHnz9Oijj%2Br3v/%2B9ZsyYoQcffFBHjx6VJL366qvatm2b8vLytGfPHnXv3l3Z2dkyxkiSFi1apNraWm3fvl2//OUvVVJSolWrVnk1HwAA8F9%2BW7AiIiK0efPmcxasTZs2KT09Xenp6YqIiNDEiRPVu3dvbd26VZKUn5%2BvGTNmqEePHmrfvr3mzp2rkpISffjhh/rqq6/09ttva%2B7cuYqLi1Pnzp01Z84c/fKXv9Tp06e9HRMAAPihcF8v4GLdfffd550rLi5Wenq6x1hiYqIKCwtVV1enQ4cOKTEx0T3Xvn17XXvttSosLFRVVZXCwsLUp08f93xSUpJOnjypTz/91GP8fMrLy1VRUeExFh7eVvHx8S2N1yJhYX7bjxHkwsMv7L175r06whgfAAARf0lEQVQebO/5YM0tBW/2YM0tBV52vy1Y/47L5VJ0dLTHWHR0tA4dOqQTJ07IGHPOeafTqZiYGLVv314hISEec5LkdDpb9Pj5%2Bflat26dx1h2drZycnIuJg4QcGJj213U/aKirrS8Ev8QrLml4M0erLmlwMkekAVLkvt8qouZ/6b7fpPMzExlZGR4jIWHt5XTWXNJ2z1boLR8BJ8L/VkICwtVVNSVqqysVWNjUyut6vITrLml4M0erLml1st%2Bsb/QXaqALFixsbFyuVweYy6XS3FxcYqJiVFoaOg55zt27Ki4uDhVV1ersbFRYWFh7jlJ6tixY4sePz4%2BvtnhwIqKKjU0BNcPC3A%2BF/uz0NjYFJQ/R8GaWwre7MGaWwqc7AG5C8ThcKioqMhjrLCwUCkpKYqIiFCvXr1UXFzsnqusrNTnn3%2Buvn37KiEhQcYYffTRRx73jYqK0nXXXee1DAAAwH8FZMGaNm2a3n//fe3du1f19fXavHmzPvvsM02cOFGSlJWVpY0bN6qkpETV1dVatWqVEhISlJycrLi4OI0ZM0ZPPfWUjh8/rqNHj%2BqZZ57RlClTFB4ekDv8AACAZX7bGJKTkyVJDQ0NkqS3335b0td7m3r37q1Vq1Zp%2BfLlKi0tVc%2BePbVhwwZ16tRJkjR9%2BnRVVFTou9/9rmpqapSamupxUvqPf/xjPfHEExo1apSuuOIK3Xbbbee8mCkAAMC5hJhLPaMbLVJRUWV9m%2BHhobpl1TvWtwu0th25Qy/o9uHhoYqNbSensyYgzs1oqWDNLQVv9mDNLbVe9k6dOljb1oUIyEOEAAAAvkTBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMCycF8vAEDwGfvUe75ewgW50D9ODQDswQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALAv39QIA4HI39qn3fL2EC7Ijd6ivlwAEPfZgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWhft6AQAAu8Y%2B9Z6vl9BiO3KH%2BnoJQKugYAEAfMafyqBEIUTLUbAAAGghCiFainOwAAAALKNgAQAAWEbBAgAAsIyCdQ6lpaW67777lJqaqpEjR2rlypVqamry9bIAAICf4CT3c3jooYeUlJSkt99%2BW8eOHdPs2bN11VVX6Xvf%2B56vlwYAAPwABesshYWF%2Buijj/Tiiy%2BqQ4cO6tChg2bMmKGXXnqJggUA8Cv%2B9KnH/%2B/Rm3y9BKsoWGcpLi5W165dFR0d7R5LSkrS4cOHVV1drfbt23/jNsrLy1VRUeExFh7eVvHx8VbXGhbGEV4AQGA4829aoPzbRsE6i8vlUlRUlMfYmbLldDpbVLDy8/O1bt06j7EHH3xQDz30kL2F6usid0%2BXvyszM9N6ebuclZeXKz8/P%2BhyS8GbndzBlVsK3uzBmlv6OvtLLz0fMNkDoyZaZoy5pPtnZmbqV7/6lcdXZmampdX9U0VFhdatW9dsb1mgC9bcUvBmJ3dw5ZaCN3uw5pYCLzt7sM4SFxcnl8vlMeZyuRQSEqK4uLgWbSM%2BPj4g2jcAALg47ME6i8PhUFlZmY4fP%2B4eKywsVM%2BePdWuXTsfrgwAAPgLCtZZEhMTlZycrNWrV6u6ulolJSV68cUXlZWV5eulAQAAPxG2ePHixb5exOXmpptu0vbt27V06VL95je/0ZQpUzRz5kyFhIT4emnNtGvXTkOGDAm6vWvBmlsK3uzkDq7cUvBmD9bcUmBlDzGXekY3AAAAPHCIEAAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyCpYfKi0t1X333afU1FSNHDlSK1euVFNTk6%2BXddHeeecdpaWlae7cuc3m3nzzTU2YMEH9%2B/fXpEmT9O6777rnmpqatGbNGo0aNUqDBw/WzJkzdeTIEfe8y%2BVSbm6u0tLSNGzYMC1cuFB1dXVeydQSpaWlys7OVmpqqtLS0jR//nxVVlZKkg4ePKi77rpLAwcO1OjRo/XCCy943PdSnhdf%2B%2Bijj3TPPfdo4MCBSktLU25urioqKiRJBQUFmjJligYMGKDx48dr69atHvfduHGjxowZowEDBigrK0tFRUXuufr6ej3%2B%2BOMaPny4UlNTlZOTI6fT6dVsLfXkk0%2BqT58%2B7u8DPXefPn3kcDiUnJzs/lq6dKmkwM/%2B7LPPatiwYerXr59mzJihL774QlJg5/7Tn/7k8VonJyfL4XC43/OBnN2Dgd%2B58847zY9%2B9CNTWVlpDh8%2BbEaPHm1eeOEFXy/rouTl5ZnRo0eb6dOnm9zcXI%2B5AwcOGIfDYfbu3Wvq6urMli1bTEpKiikrKzPGGLNx40YzcuRIc%2BjQIVNVVWV%2B/OMfmwkTJpimpiZjjDEPPvigue%2B%2B%2B8yxY8fM0aNHTWZmplm6dKnXM57PbbfdZubPn2%2Bqq6tNWVmZmTRpklmwYIGpra01N910k1m7dq2pqakxRUVFZsiQIWbXrl3GmEt/Xnypvr7e3HjjjWbdunWmvr7eHDt2zNx1111mzpw55ssvvzT9%2BvUzmzZtMnV1dea9994zffv2NX/729%2BMMcb89re/NYMGDTIffPCBqa2tNRs2bDBDhw41NTU1xhhjli9fbiZNmmT%2B8Y9/GKfTaR588EEze/ZsX8Y9pwMHDpghQ4aY3r17G2NMUOTu3bu3OXLkSLPxQM/%2ByiuvmFtvvdWUlJSYqqoqs3TpUrN06dKAz30uzz77rPnP//zPoMpOwfIzf/vb30xCQoJxuVzusf/93/81Y8aM8eGqLt5LL71kKisrzbx585oVrCVLlpjs7GyPsalTp5oNGzYYY4wZP368eemll9xzVVVVJjEx0ezfv99UVFSY66%2B/3hw8eNA9v2/fPtOvXz9z6tSpVkzUMidOnDDz5883FRUV7rGXX37ZjB492uzYscPccMMNpqGhwT23cuVK8/3vf98Yc2nPi6%2B5XC7z%2Buuvm9OnT7vHXnrpJXPLLbeY559/3txxxx0et8/NzTWLFi0yxhhz3333mSeffNI919jYaIYOHWq2b99uTp8%2BbQYOHGjefvtt9/yhQ4dMnz59zNGjR1s5Vcs1NjaaqVOnmv/6r/9yF6xgyH2%2BghXo2TMyMty/GP2rQM99ttLSUjNkyBBTWloaVNk5ROhniouL1bVrV0VHR7vHkpKSdPjwYVVXV/twZRfn7rvvVocOHc45V1xcrMTERI%2BxxMREFRYWqq6uTocOHfKYb9%2B%2Bva699loVFhbq4MGDCgsL8zgMk5SUpJMnT%2BrTTz9tnTAXICoqSsuXL9dVV13lHisrK1N8fLyKi4vVp08fhYWFuecSExPdu8kv5XnxtejoaE2dOlXh4eGSpE8//VS//vWvNXbs2PPmOl/u0NBQJSQkqLCwUJ9//rmqqqqUlJTknu/Ro4ciIyNVXFzshWQt89prrykiIkITJkxwjwVDbklavXq1RowYoUGDBmnRokWqqakJ6OxffvmlvvjiC504cULjxo1zH846fvx4QOc%2Bl6efflqTJ0/Wt771raDKTsHyMy6XS1FRUR5jZ8rWZXsc%2BiK5XC6PIil9ndXpdOrEiRMyxpx33uVyqX379goJCfGYky7P56mwsFCvvPKKHnjggXO%2BxjExMXK5XGpqarqk5%2BVyUVpaKofDoXHjxik5OVk5OTnnzX1m3f8ut8vlkqRm94%2BKirpscn/11Vdau3atnnjiCY/xQM8tSf369VNaWpreeust5efn64MPPtCSJUsCOvvRo0clSTt37tSLL76oLVu26OjRo/rRj34U0LnP9sUXX%2Bitt97S9773PUnB8X4/g4Llh4wxvl6C13xT1n837y/P01/%2B8hfNnDlTjzzyiNLS0s57u38ti5fyvFwOunbtqsLCQu3cuVOfffaZfvjDH7bofv6ce/ny5Zo0aZJ69ux5wff159ySlJ%2Bfr6lTp6pNmzbq0aOHHn30UW3fvl2nT5/%2Bxvv6a/Yz65o1a5Y6d%2B6sLl266KGHHtLu3bsv6P4XO3%2B5ePXVVzV69Gh16tSpxfcJlOwULD8TFxfnbvFnuFwuhYSEKC4uzkerah2xsbHnzBoXF6eYmBiFhoaec75jx46Ki4tTdXW1GhsbPeYkqWPHjq2/%2BBbavXu37rvvPi1YsEB33323pK9f47N/G3O5XO7Ml/K8XE5CQkLUvXt3zZ07V9u3b1d4eHizdTudTvf7%2Bt/lPnObs%2BdPnDhxWeQuKCjQ/v37lZ2d3WzuXLkCJff5XHPNNWpsbDznezVQsp85/P%2Bve1u6du0qY4xOnz4dsLnPtmvXLmVkZLi/D6b3OwXLzzgcDpWVlen48ePuscLCQvXs2VPt2rXz4crsczgcHh/Plb7OmpKSooiICPXq1cvjuHtlZaU%2B//xz9e3bVwkJCTLG6KOPPvK4b1RUlK677jqvZfh3/vrXv2revHl6%2Bumndccdd7jHHQ6HPv74YzU0NLjHzuQ%2BM3%2Bxz4uvFRQUaMyYMR6XFQkN/fr/hvr27dssV1FRkUfuf83V2NioAwcOKCUlRd26dVN0dLTH/CeffKJTp07J4XC0ZqQW2bp1q44dO6aRI0cqNTVVkyZNkiSlpqaqd%2B/eAZtbkg4cOKAVK1Z4jJWUlKhNmzZKT08P2OxdunRR%2B/btdfDgQfdYaWmprrjiioDO/a8OHjyo0tJSDR061D2WnJwcFNklcZkGfzR16lSzYMECU1VVZQ4dOmQyMjLMK6%2B84utlXZJzfYrw448/NsnJyWbPnj2mrq7ObNq0yfTv39%2BUl5cbY77%2B9OSIESPclyNYtGiRmTx5svv%2Bubm5ZtasWebYsWOmrKzMTJ482axYscKruc7n9OnTZuzYsea1115rNldfX29GjhxpfvGLX5iTJ0%2BaDz74wAwaNMjs2bPHGHPpz4svVVZWmrS0NLNixQpz8uRJc%2BzYMTNz5kzzne98x3z11Vemf//%2B5vXXXzd1dXVm7969pm/fvu5Pgu7bt88MHDjQ7N%2B/35w8edKsXbvWpKenm9raWmPM15%2B0vPPOO80//vEPc/z4cTN79mzz0EMP%2BTKum8vlMmVlZe6v/fv3m969e5uysjJTWloasLmNMebo0aOmX79%2BZsOGDaa%2Bvt58%2BumnZty4cWbp0qUB/ZobY8yTTz5pRo0aZT777DPz1VdfmczMTDN//vyAz33G5s2bzZAhQzzGgiW7MVymwS%2BVlZWZWbNmmb59%2B5q0tDTzi1/84rK4xtHFcDgcxuFwmOuvv95cf/317u/P2LVrlxk9erRJSkoyt99%2Bu/njH//onmtqajJPP/20ufHGG03fvn3Nvffe674WlDFf/2M%2Bd%2B5c069fPzN48GCzZMkSU19f79V85/OnP/3J9O7d2533X7%2B%2B%2BOIL8/HHH5vp06cbh8NhRowYYV599VWP%2B1/K8%2BJrH330kbnrrrtM3759zQ033GByc3PdH7H%2B4x//aCZOnGiSkpLM6NGjm33E/dVXXzXp6enG4XCYrKws8/HHH7vn6uvrzeLFi83gwYNN//79zcMPP2wqKyu9mq2ljhw54r5MgzGBn/uPf/yjyczMNP369TNDhgwxy5cvN3V1de65QM3%2Br%2Bvr16%2BfmTdvnqmurjbGBHbuM9avX2/Gjx/fbDwYshtjTIgxfnK2GAAAgJ/gHCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsOz/B%2BjYQU0haVtrAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common6444980024487249378">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">618</td>
        <td class="number">65</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">541</td>
        <td class="number">51</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">636</td>
        <td class="number">45</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">607</td>
        <td class="number">43</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">573</td>
        <td class="number">42</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">960</td>
        <td class="number">42</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">752</td>
        <td class="number">41</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">942</td>
        <td class="number">40</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">342</td>
        <td class="number">40</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">242</td>
        <td class="number">40</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (2700)</td>
        <td class="number">14671</td>
        <td class="number">97.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme6444980024487249378">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:10%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30</td>
        <td class="number">9</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:45%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">42</td>
        <td class="number">11</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:55%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">10</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">67</td>
        <td class="number">20</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">6661</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6686</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6723</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6853</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6993</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Horizontal_Distance_To_Hydrology">Horizontal_Distance_To_Hydrology<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>400</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>2.6%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>227.2</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>1343</td>
                </tr>
                <tr class="alert">
                    <th>Zeros (%)</th>
                    <td>10.5%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram2036734965981555444">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAAStJREFUeJzt3MFpwzAYgNE4ZKQO0Z167k4dojsp91A%2BYoNjxX3vbtDl45cs42WMMS7An65HLwBmdjt6AY8%2Bvn5WP/P7/bnDSsAEgSQQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCNP9tGELP3pgLyYIBIFAEAgEgUAQCASBQDjFa94tvBrmGSYIBIFAEAgEgUAQCASBQBAIBIFA%2BLcXhVusvVx0sfj%2BTBAIAoEgEAgCgeCQviNfDL8/EwSCQCDYYk3GtmwuAjmBLVG9whnCXcYY4%2BhFwKycQSAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCDcAeUFHZt1JaLVAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives2036734965981555444,#minihistogram2036734965981555444"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives2036734965981555444">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles2036734965981555444"
                                                  aria-controls="quantiles2036734965981555444" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram2036734965981555444" aria-controls="histogram2036734965981555444"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common2036734965981555444" aria-controls="common2036734965981555444"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme2036734965981555444" aria-controls="extreme2036734965981555444"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles2036734965981555444">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>67</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>180</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>330</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>631</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>1343</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>1343</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>263</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>210.08</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.92464</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>2.804</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>227.2</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>160.28</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.4881</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>3435199</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>44132</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram2036734965981555444">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl4lNXB/vE7yUgihGxg0B%2BlQlmzQNijAVmCgIIgYCTkeq1SpYJE0rC0gIBCpUDLpgVflvrSitiaQn0VUMSFpVpjV7TJgAuIVdJAIsyQEJJAkvP7w5epYxCCHGaY5Pu5Lq6Wc2aenHNnInee58kkyBhjBAAAAGuC/b0AAACA%2BoaCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAsc/h7AQ1FcXGp9WMGBwcpJqaJTpwoU02NsX78QEAGZCCRgUQGEhlIZCDVzuC665r6Zx1%2B%2BaiwIjg4SEFBQQoODvL3UvyGDMhAIgOJDCQykMhAunoyoGABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUOfy8Al6fnnFf9vYQ625Hdx99LAADAJziDBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWBXTBWrNmjfr27auuXbtq/PjxOnLkiCQpNzdXaWlp6t69u4YPH66tW7d6PW/jxo0aOnSounfvroyMDOXn53vmKisr9eijj6pfv35KTk5WVlaWXC6XT/cFAAACW8AWrOeee05bt27Vxo0b9fbbb6tdu3b6zW9%2Bo6KiIk2ePFnjxo1Tbm6u5syZo3nz5ikvL0%2BStGvXLq1atUq/%2BMUv9M4772jgwIGaNGmSTp8%2BLUlauXKlnE6ncnJytHPnThljNHv2bH9uFQAABJiALVgbNmzQ1KlT9b3vfU/h4eGaO3eu5s6dq23btql169ZKS0tTaGioUlJSlJqaqs2bN0uScnJyNGbMGCUlJSksLEwTJkyQJO3evVtVVVXasmWLJk%2BerBtuuEFRUVHKzs7Wnj17dOzYMX9uFwAABJCALFjHjh3TkSNHdPLkSQ0bNsxzKe/EiRNyOp2Kj4/3enx8fLznMuDX54ODgxUXF6e8vDx99tlnKi0tVUJCgme%2Bbdu2CgsLk9Pp9M3mAABAwHP4ewHfxtGjRyVJr776qn7961/LGKOsrCzNnTtXFRUVatGihdfjo6KiPPdRud1uRUZGes1HRkbK5XLJ7XZLkiIiIrzmIyIiLuk%2BrKKiIhUXF3uNORyNFRsbW%2Bdj1EVISGD1Y4fD/nrPZRBoWdhEBmQgkYFEBhIZSFdPBgFZsIwxkqQJEyZ4ytSUKVP0wx/%2BUCkpKXV%2B/redv5icnBytXr3aaywzM1NZWVmXddxAFx3d5IodOyLi2it27EBBBmQgkYFEBhIZSP7PICALVvPmzSV5n2lq2bKljDE6e/as50zUOS6XSzExMZKk6OjoWvNut1vt27f3PMbtdqtJk/%2BUgZMnT6pZs2Z1Xl96erpSU1O9xhyOxnK5yup8jLrwdzu/VLb3L32ZQUTEtSopKVd1dY314wcCMiADiQwkMpDIQKqdwZX85v5CArJgXX/99QoPD9eBAwc890sVFBTommuuUf/%2B/fXSSy95PT4/P19JSUmSpMTERDmdTo0ePVqSVF1drf379ystLU2tWrVSZGSknE6nWrZsKUn66KOPdObMGSUmJtZ5fbGxsbUuBxYXl6qqqmG%2B2M%2B5kvuvrq5p8PmSARlIZCCRgUQGkv8zCKxTIP/H4XAoLS1Na9eu1b/%2B9S8dP35cTz31lEaMGKHRo0eroKBAmzdvVmVlpfbu3au9e/dq7NixkqSMjAy9%2BOKLeu%2B991ReXq41a9aoUaNGGjBggEJCQjR27FitXbtWhYWFcrlcWrFihQYPHuw5awYAAHAxAXkGS5KmT5%2BuM2fO6O6779bZs2c1dOhQzZ07V02aNNG6deu0cOFCLViwQC1bttTSpUvVqVMnSVK/fv00bdo0ZWdn6/jx4%2BrcubPWr1%2BvsLAwSVJWVpbKysp05513qqqqSgMHDtT8%2BfP9uFMAABBogszl3tGNOikuLrV%2BTIcjWIOXvWX9uFfKjuw%2B1o/pcAQrOrqJXK6yBns6nAzIQCIDiQwkMpBqZ3DddU39so6AvEQIAABwNaNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMCygC1YHTt2VGJiojp37uz58/jjj0uScnNzlZaWpu7du2v48OHaunWr13M3btyooUOHqnv37srIyFB%2Bfr5nrrKyUo8%2B%2Bqj69eun5ORkZWVlyeVy%2BXRvAAAgsDn8vYDL8eqrr%2Bo73/mO11hRUZEmT56sOXPmaMSIEfr73/%2Buhx56SG3atFHnzp21a9curVq1Sk8//bQ6duyojRs3atKkSXrttdfUuHFjrVy5Uk6nUzk5Obr22ms1b948zZ49W2vXrvXTLgEAQKAJ2DNY32Tbtm1q3bq10tLSFBoaqpSUFKWmpmrz5s2SpJycHI0ZM0ZJSUkKCwvThAkTJEm7d%2B9WVVWVtmzZosmTJ%2BuGG25QVFSUsrOztWfPHh07dsyf2wIAAAEkoM9gLV%2B%2BXPv27dOpU6d0%2B%2B23a9asWXI6nYqPj/d6XHx8vHbs2CFJcjqdGjZsmGcuODhYcXFxysvLU1xcnEpLS5WQkOCZb9u2rcLCwuR0OtWiRYs6rauoqEjFxcVeYw5HY8XGxn7brZ5XSEhg9WOHw/56z2UQaFnYRAZkIJGBRAYSGUhXTwYBW7C6du2qlJQU/fznP9fnn3%2Bu7OxsLViwQG63u1YRioqK8txH5Xa7FRkZ6TUfGRkpl8slt9stSYqIiPCaj4iIuKT7sHJycrR69WqvsczMTGVlZdX5GPVRdHSTK3bsiIhrr9ixAwUZkIFEBhIZSGQg%2BT%2BDgC1YOTk5nv/ftm1bzZgxQw899JB69Ohx0ecaYy5r/mLS09OVmprqNeZwNJbLVXZZx/06f7fzS2V7/9KXGUREXKuSknJVV9dYP34gIAMykMhAIgOJDKTaGVzJb%2B4vJGAL1td95zvfUXV1tYKDgz1nos5xuVyKiYmRJEVHR9ead7vdat%2B%2BvecxbrdbTZr85xNy8uRJNWvWrM5riY2NrXU5sLi4VFVVDfPFfs6V3H91dU2Dz5cMyEAiA4kMJDKQ/J9BYJ0C%2BT/79%2B/XkiVLvMYOHTqkRo0aqX///l5vuyBJ%2Bfn5SkpKkiQlJibK6XR65qqrq7V//34lJSWpVatWioyM9Jr/6KOPdObMGSUmJl7BHQEAgPokIAtWs2bNlJOTo/Xr1%2BvMmTM6fPiwnnzySaWnp%2BvOO%2B9UQUGBNm/erMrKSu3du1d79%2B7V2LFjJUkZGRl68cUX9d5776m8vFxr1qxRo0aNNGDAAIWEhGjs2LFau3atCgsL5XK5tGLFCg0ePFjNmzf3864BAECgCMhLhC1atND69eu1fPlyT0EaPXq0pk6dqtDQUK1bt04LFy7UggUL1LJlSy1dulSdOnWSJPXr10/Tpk1Tdna2jh8/rs6dO2v9%2BvUKCwuTJGVlZamsrEx33nmnqqqqNHDgQM2fP9%2BPuwUAAIEmyFzuHd2ok%2BLiUuvHdDiCNXjZW9aPe6XsyO5j/ZgOR7Cio5vI5SprsPcbkAEZSGQgkYFEBlLtDK67rqlf1hGQlwgBAACuZhQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhWLwrWokWL1LFjR8/fc3NzlZaWpu7du2v48OHaunWr1%2BM3btyooUOHqnv37srIyFB%2Bfr5nrrKyUo8%2B%2Bqj69eun5ORkZWVlyeVy%2BWwvAAAg8AV8wTpw4IBeeuklz9%2BLioo0efJkjRs3Trm5uZozZ47mzZunvLw8SdKuXbu0atUq/eIXv9A777yjgQMHatKkSTp9%2BrQkaeXKlXI6ncrJydHOnTtljNHs2bP9sjcAABCYfF6wUlNTtXr1ahUWFl72sWpqavTYY49p/PjxnrFt27apdevWSktLU2hoqFJSUpSamqrNmzdLknJycjRmzBglJSUpLCxMEyZMkCTt3r1bVVVV2rJliyZPnqwbbrhBUVFRys7O1p49e3Ts2LHLXi8AAGgYHL7%2BgHfddZdefvllrVmzRjfffLPGjh2r1NRUORyXvpTnn39eoaGhGjFihJ544glJktPpVHx8vNfj4uPjtWPHDs/8sGHDPHPBwcGKi4tTXl6e4uLiVFpaqoSEBM9827ZtFRYWJqfTqRYtWtRpXUVFRSouLvYaczgaKzY29pL3eCEhIYF1AtLhsL/ecxkEWhY2kQEZSGQgkYFEBtLVk4HPC1ZmZqYyMzPldDq1fft2LVq0SAsWLNCoUaOUlpamNm3a1Ok4X3zxhVatWqVnn33Wa9ztdtcqQlFRUZ77qNxutyIjI73mIyMj5XK55Ha7JUkRERFe8xEREZd0H1ZOTo5Wr17tNZaZmamsrKw6H6M%2Bio5ucsWOHRFx7RU7dqAgAzKQyEAiA4kMJP9n4POCdU5CQoISEhL0k5/8RK%2B88ormz5%2BvDRs2KCUlRT/60Y/UpUuXCz5/8eLFGjNmjNq1a6cjR45c0sc2xlzW/MWkp6crNTXVa8zhaCyXq%2Byyjvt1/m7nl8r2/qUvM4iIuFYlJeWqrq6xfvxAQAZkIJGBRAYSGUi1M7iS39xfiN8K1tmzZ/X666/rhRde0LvvvqvWrVtrypQpKioq0vjx47VgwQKNGDHivM/Nzc3Vvn37tH379lpz0dHRnjNR57hcLsXExHzjvNvtVvv27T2PcbvdatLkP5%2BQkydPqlmzZnXeW2xsbK3LgcXFpaqqapgv9nOu5P6rq2safL5kQAYSGUhkIJGB5P8MfF6wDh06pC1btujFF19UWVmZhg4dqmeeeUY9evTwPKZXr16aP3/%2BNxasrVu36vjx4xo4cKCk/5xxSk5O1v3331%2BreOXn5yspKUmSlJiYKKfTqdGjR0uSqqurtX//fqWlpalVq1aKjIyU0%2BlUy5YtJUkfffSRzpw5o8TERLtBAACAesvnBWv48OFq06aNJk6cqFGjRikqKqrWY/r3768TJ0584zFmzZqlH/3oR56/Hz16VOnp6XrppZdUU1OjdevWafPmzRo5cqTeffdd7d27Vzk5OZKkjIwMTZs2TXfccYc6duyo//mf/1GjRo00YMAAhYSEaOzYsVq7dq06d%2B6ssLAwrVixQoMHD1bz5s3thwEAAOolnxesjRs3qnfv3hd93Pvvv/%2BNc5GRkV43qldVVUmSrr/%2BeknSunXrtHDhQi1YsEAtW7bU0qVL1alTJ0lSv379NG3aNGVnZ%2Bv48ePq3Lmz1q9fr7CwMElSVlaWysrKdOedd6qqqkoDBw7U/Pnzv%2B12AQBAAxRkLveO7kt08uRJzZw5U2lpabr11lslSb/5zW/0pz/9SUuXLj3vGa36oLi41PoxHY5gDV72lvXjXik7svtYP6bDEazo6CZyucoa7P0GZEAGEhlIZCCRgVQ7g%2Buua%2BqXdfj8x9AWL16s0tJStWvXzjM2YMAA1dTUaMmSJb5eDgAAgHU%2Bv0T49ttva9u2bYqOjvaMtW7dWsuWLdMdd9zh6%2BUAAABY5/MzWBUVFQoNDa29kOBglZeX%2B3o5AAAA1vm8YPXq1UtLlizRyZMnPWPHjh3TggULvN6qAQAAIFD5/BLhI488ovvvv18333yzwsPDVVNTo7KyMrVq1arWr70BAAAIRD4vWK1atdLLL7%2BsP/7xj/rss88UHBysNm3aqG/fvgoJCfH1cgAAAKzzy6/KadSokectGgAAAOobnxeszz//XMuXL9fHH3%2BsioqKWvNvvvmmr5cEAABglV/uwSoqKlLfvn3VuHFjX394AACAK87nBSs/P19vvvmmYmJifP2hAQAAfMLnb9PQrFkzzlwBAIB6zecFa%2BLEiVq9erV8/CsQAQAAfMbnlwj/%2BMc/6h//%2BIdeeOEFfec731FwsHfHe/755329JPjI7U/8yd9LuCRX4pdTAwAaBp8XrPDwcPXr18/XHxYAAMBnfF6wFi9e7OsPCQAA4FM%2BvwdLkj755BOtWrVKs2fP9ozt27fPH0sBAACwzucFKzc3VyNHjtRrr72m7du3S/ryzUfvvfde3mQUAADUCz4vWCtXrtSPf/xjbdu2TUFBQZK%2B/P2ES5Ys0VNPPeXr5QAAAFjn84L10UcfKSMjQ5I8BUuSbrvtNh06dMjXywEAALDO5wWradOm5/0dhEVFRWrUqJGvlwMAAGCdzwtW9%2B7dtWjRIp06dcozdvjwYc2cOVM333yzr5cDAABgnc/fpmH27Nm67777lJycrOrqanXv3l3l5eVq3769lixZ4uvlAAAAWOfzgnX99ddr%2B/bt2rt3rw4fPqywsDC1adNGffr08bonCwAAIFD5vGBJ0jXXXKNbb73VHx8aAADgivN5wUpNTb3gmSreCwsAAAQ6nxesYcOGeRWs6upqHT58WHl5ebrvvvt8vRwAAADrfF6wZsyYcd7xnTt36s9//rOPVwMAAGCfX34X4fnceuutevnll/29DAAAgMt21RSs/fv3yxjj72UAAABcNp9fIhw3blytsfLych06dEhDhgzx9XIAAACs83nBat26da2fIgwNDVVaWpruvvtuXy8HAADAOp8XLN6tHQAA1Hc%2BL1gvvvhinR87atSoK7gSAACAK8PnBWvOnDmqqampdUN7UFCQ11hQUBAFCwAABCSfF6ynn35aGzZs0KRJk9SxY0cZY/Thhx/qV7/6le655x4lJyf7ekkAAABW%2BeUerPXr16tFixaesZ49e6pVq1Z64IEHtH37dl8vCQAAwCqfvw/Wp59%2BqsjIyFrjERERKigoqPNxPvjgA913333q0aOHUlJSlJ2dreLiYklSbm6u0tLS1L17dw0fPlxbt271eu7GjRs1dOhQde/eXRkZGcrPz/fMVVZW6tFHH1W/fv2UnJysrKwsuVyub7lbAADQEPm8YLVs2VJLlizxKi0lJSVavny5vvvd79bpGGfOnNH999%2Bv3r17Kzc3V9u3b9fx48c1f/58FRUVafLkyRo3bpxyc3M1Z84czZs3T3l5eZKkXbt2adWqVfrFL36hd955RwMHDtSkSZN0%2BvRpSdLKlSvldDqVk5OjnTt3yhij2bNn2w8CAADUWz4vWI888oh27NihlJQU9ezZU71799ZNN92kF154QbNmzarTMcrLyzV16lRNnDhRjRo1UkxMjAYPHqyPP/5Y27ZtU%2BvWrZWWlqbQ0FClpKQoNTVVmzdvliTl5ORozJgxSkpKUlhYmCZMmCBJ2r17t6qqqrRlyxZNnjxZN9xwg6KiopSdna09e/bo2LFjVywTAABQv/j8Hqy%2Bfftqz5492rt3r44ePSpjjFq0aKFbbrlFTZs2rdMxIiMjvd6U9JNPPtH//u//6vbbb5fT6VR8fLzX4%2BPj47Vjxw5JktPp1LBhwzxzwcHBiouLU15enuLi4lRaWqqEhATPfNu2bRUWFian0%2Bl13xgAAMA38XnBkqRrr71WgwYN0tGjR9WqVatvfZyCggINHTpUVVVVGjt2rLKysvTDH/6wVhGKioryXJJ0u9217gGLjIyUy%2BWS2%2B2W9OX9YF8VERFxSfdhFRUVee4HO8fhaKzY2Ng6H6MuQkKuml8lWS85HIGR77nXQUN%2BPZABGUhkIJGBdPVk4POCVVFRoccee0wvv/yyJCk/P18lJSWaNm2aVqxYUavcXEjLli2Vl5enf/3rX3r00Uf1k5/8pE7Pu9gvlb7cXzqdk5Oj1atXe41lZmYqKyvrso4L34qObuLvJVySiIhr/b0EvyMDMpDIQCIDyf8Z%2BLxgLV26VAcOHNCyZcu8ClF1dbWWLVumn/70p5d0vKCgILVu3VpTp07VuHHj1L9/f8%2BZqHNcLpdiYmIkSdHR0bXm3W632rdv73mM2%2B1Wkyb/%2Bcf15MmTatasWZ3XlJ6ertTUVK8xh6OxXK6yS9rbxfi7ndd3tj9fV0pISLAiIq5VSUm5qqtr/L0cvyADMpDIQCIDqXYG/vpm2ecFa%2BfOndq0aZNat26tmTNnSvryEtzixYs1atSoOhWs3NxczZ8/Xzt27FBw8Jcl49z/dunSRTt37vR6fH5%2BvpKSkiRJiYmJcjqdGj16tKQvi93%2B/fuVlpamVq1aKTIyUk6nUy1btpQkffTRRzpz5owSExPrvMfY2NhalwOLi0tVVdUwX%2ByBKtA%2BX9XVNQG3ZtvIgAwkMpDIQPJ/Bj4/BVJWVqbWrVvXGo%2BJifG8VcLFJCYm6tSpU1q6dKnKy8t14sQJrVq1Sj179lRGRoYKCgq0efNmVVZWau/evdq7d6/Gjh0rScrIyNCLL76o9957T%2BXl5VqzZo0aNWqkAQMGKCQkRGPHjtXatWtVWFgol8ulFStWaPDgwWrevLnNGAAAQD3m84L13e9%2BV3/%2B858led/r9Oqrr%2Br//b//V6djNG3aVBs2bFB%2Bfr5uuukmDR8%2BXE2bNtWKFSvUrFkzrVu3Tps2bVKPHj20aNEiLV26VJ06dZIk9evXT9OmTVN2drZ69%2B6td955R%2BvXr1dYWJgkKSsrS0lJSbrzzjs1aNAgNWnSRD/72c8spwAAAOqzIHO5d3RfopycHC1fvlx33XWXnn32WU2fPl35%2BfnauXOn5syZo4yMDF8ux2eKi0utH9PhCNbgZW9ZPy6%2BtCO7j7%2BXUCcOR7Cio5vI5SprsJcEyIAMJDKQyECqncF119XtLaCsr8PXHzA9PV0Oh0ObNm1SSEiI1q5dqzZt2mjZsmW67bbbfL0cAAAA63xesE6cOKG77rpLd911l68/NAAAgE/4/B6sQYMGXfb7TAEAAFzNfF6wkpOTPb%2B2BgAAoD7y%2BSXCG264QT/72c%2B0fv16ffe739U111zjNb98%2BXJfLwkAAMAqnxesgwcP6nvf%2B54kXdLv9wMAAAgUPitYU6dO1cqVK/Xss896xp566illZmb6agkAAAA%2B4bN7sHbt2lVrbP369b768AAAAD7js4J1vp8c5KcJAQBAfeSzghUUFFSnMQAAgEDn87dpAAAAqO8oWAAAAJb57KcIz549q%2BnTp190jPfBwtXi9if%2B5O8lXJLXZ9zi7yUAAP6PzwpWjx49VFRUdNExAACAQOezgvXV978CAACoz7gHCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWBawBaugoECZmZlKTk5WSkqKZs2apZKSEknSgQMHdM8996hHjx4aMmSINmzY4PXcV155RSNGjFC3bt00ZswYvf322565mpoarVy5UoMGDVKvXr30wAMP6PPPP/fp3gAAQGAL2II1adIkRUREaNeuXXrhhRf08ccf6%2Bc//7kqKio0ceJE3XTTTXrrrbe0cuVKrVu3Tq%2B99pqkL8vXzJkzNWPGDL377rsaP368Hn74YR09elSS9Nxzz2nbtm1av369du/erdatWyszM1PGGH9uFwAABJCALFglJSVKTEzU9OnT1aRJE11//fUaPXq0/va3v2nPnj06e/asHnroITVu3FgJCQm6%2B%2B67lZOTI0navHmz%2Bvfvr/79%2Bys0NFQjR45Uhw4dtHXrVklSTk6Oxo8fr7Zt2yo8PFxTp07VoUOH9P777/tzywAAIIAEZMGKiIjQ4sWL1bx5c89YYWGhYmNj5XQ61bFjR4WEhHjm4uPjlZ%2BfL0lyOp2Kj4/3Ol58fLzy8vJUUVGhgwcPes2Hh4frxhtvVF5e3hXeFQAAqC8c/l6ADXl5edq0aZPWrFmjHTt2KCIiwms%2BKipKbrdbNTU1crvdioyM9JqPjIzUwYMHdfLkSRljzjvvcrnqvJ6ioiIVFxd7jTkcjRUbG3uJO7uwkJCA7Me4Qhry6%2BHc3smADL76vw0RGVw9GQR8wfr73/%2Buhx56SNOnT1dKSop27Nhx3scFBQV5/v/F7qe63PutcnJytHr1aq%2BxzMxMZWVlXdZxgQuJiLjW30vwOzIgA4kMJDKQ/J9BQBesXbt26cc//rHmzZunUaNGSZJiYmL06aefej3O7XYrKipKwcHBio6OltvtrjUfExPjecz55ps1a1bndaWnpys1NdU0MEA5AAAVnklEQVRrzOFoLJer7BJ2d3H%2Bbue4upSUlKu6usbfy/CLkJBgRURcSwZkQAZkUCuD6OgmfllHwBasf/zjH5o5c6aefPJJ9e3b1zOemJio3/3ud6qqqpLD8eX28vLylJSU5Jk/dz/WOXl5eRo%2BfLhCQ0PVvn17OZ1O9e7dW9KXN9R/9tln6tKlS53XFhsbW%2BtyYHFxqaqqGuaLHb5RXV3T4F9jZEAGEhlIZCD5P4OAPAVSVVWluXPnasaMGV7lSpL69%2B%2Bv8PBwrVmzRuXl5Xr//fe1ZcsWZWRkSJLGjh2rd955R3v27FFlZaW2bNmiTz/9VCNHjpQkZWRkaOPGjTp06JBOnTqlZcuWKS4uTp07d/b5PgEAQGAKyDNY7733ng4dOqSFCxdq4cKFXnOvvvqq1q5dq8cee0zr169X8%2BbNNXXqVA0YMECS1KFDBy1btkyLFy9WQUGB2rVrp3Xr1um6666TJI0bN07FxcX6/ve/r7KyMiUnJ9e6nwoAAOBCggzvoOkTxcWl1o/pcARr8LK3rB8Xgen1Gbc02EsCDkewoqObyOUqIwMyIAMy8Mrguuua%2BmUdAXmJEAAA4GpGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyh78XAMCOwcve8vcS6mxHdh9/LwEArijOYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAsC%2BiC9dZbbyklJUVTp06tNffKK69oxIgR6tatm8aMGaO3337bM1dTU6OVK1dq0KBB6tWrlx544AF9/vnnnnm3263s7GylpKSob9%2B%2BmjNnjioqKnyyJwAAEPgCtmD96le/0sKFC3XjjTfWmjtw4IBmzpypGTNm6N1339X48eP18MMP6%2BjRo5Kk5557Ttu2bdP69eu1e/dutW7dWpmZmTLGSJLmzZun8vJybd%2B%2BXX/4wx906NAhLVu2zKf7AwAAgStgC1ZoaKi2bNly3oK1efNm9e/fX/3791doaKhGjhypDh06aOvWrZKknJwcjR8/Xm3btlV4eLimTp2qQ4cO6f3339cXX3yhN954Q1OnTlVMTIxatGihyZMn6w9/%2BIPOnj3r620CAIAAFLC/i/Dee%2B/9xjmn06n%2B/ft7jcXHxysvL08VFRU6ePCg4uPjPXPh4eG68cYblZeXp9LSUoWEhKhjx46e%2BYSEBJ0%2BfVqffPKJ1/g3KSoqUnFxsdeYw9FYsbGxdd1enYSEBGw/RgPncNh97Z77WmjIXxNkQAYSGUhXTwYBW7AuxO12KzIy0mssMjJSBw8e1MmTJ2WMOe%2B8y%2BVSVFSUwsPDFRQU5DUnSS6Xq04fPycnR6tXr/Yay8zMVFZW1rfZDlDvREc3uSLHjYi49oocN5CQARlIZCD5P4N6WbAkee6n%2BjbzF3vuxaSnpys1NdVrzOFoLJer7LKO%2B3X%2BbufAt3UlvhYiIq5VSUm5qqtrrB47UJABGUhkINXO4Ep9Q3cx9bJgRUdHy%2B12e4253W7FxMQoKipKwcHB551v1qyZYmJidOrUKVVXVyskJMQzJ0nNmjWr08ePjY2tdTmwuLhUVVUN88UOfN2V%2Blqorq5p8F9nZEAGEhlI/s%2BgXp4CSUxMVH5%2BvtdYXl6ekpKSFBoaqvbt28vpdHrmSkpK9Nlnn6lLly6Ki4uTMUYffPCB13MjIiLUpk0bn%2B0BAAAErnpZsMaOHat33nlHe/bsUWVlpbZs2aJPP/1UI0eOlCRlZGRo48aNOnTokE6dOqVly5YpLi5OnTt3VkxMjIYOHaonnnhCJ06c0NGjR/XUU08pLS1NDke9POEHAAAsC9jG0LlzZ0lSVVWVJOmNN96Q9OXZpg4dOmjZsmVavHixCgoK1K5dO61bt07XXXedJGncuHEqLi7W97//fZWVlSk5OdnrpvSf/vSneuyxxzRo0CBdc801uuOOO877ZqYAAADnE2Qu945u1Elxcan1YzocwRq87C3rxwWutB3Zfawez%2BEIVnR0E7lcZQ32vhMyIAOJDKTaGVx3XVO/rKNeXiIEAADwJwoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJY5/L0AAA3P7U/8yd9LuCS2fzk1gPqPM1gAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUOfy8AAK52tz/xJ38v4ZLsyO7j7yUADR5nsAAAACyjYAEAAFhGwQIAALCMggUAAGAZN7kDQD0TSDflc0M%2B6ivOYAEAAFhGwQIAALCMS4QAAL8JpMuZEpc0UXecwTqPgoICPfjgg0pOTtbAgQO1dOlS1dTU%2BHtZAAAgQHAG6zymTJmihIQEvfHGGzp%2B/LgmTpyo5s2b6wc/%2BIG/lwYA8CPOuKGuOIP1NXl5efrggw80Y8YMNW3aVK1bt9b48eOVk5Pj76UBAIAAwRmsr3E6nWrZsqUiIyM9YwkJCTp8%2BLBOnTql8PDwix6jqKhIxcXFXmMOR2PFxsZaXWtICP0YAPDNHI6G9%2B/EuX8b/f1vJAXra9xutyIiIrzGzpUtl8tVp4KVk5Oj1atXe409/PDDmjJlir2F6ssid9/1Hys9Pd16eQsURUVFysnJIQMyIAMyIAORgfRlBs8887TfM2h41bYOjDGX9fz09HS98MILXn/S09Mtre4/iouLtXr16lpnyxoSMiADiQwkMpDIQCID6erJgDNYXxMTEyO32%2B015na7FRQUpJiYmDodIzY2tsF%2B5wAAADiDVUtiYqIKCwt14sQJz1heXp7atWunJk2a%2BHFlAAAgUFCwviY%2BPl6dO3fW8uXLderUKR06dEi//vWvlZGR4e%2BlAQCAABEyf/78%2Bf5exNXmlltu0fbt2/X444/r5ZdfVlpamh544AEFBQX5e2m1NGnSRL17927QZ9fIgAwkMpDIQCIDiQykqyODIHO5d3QDAADAC5cIAQAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAWggoICPfjgg0pOTtbAgQO1dOlS1dTU%2BHtZ1hUUFCgzM1PJyclKSUnRrFmzVFJSIkk6cOCA7rnnHvXo0UNDhgzRhg0bvJ77yiuvaMSIEerWrZvGjBmjt99%2B2x9bsGrRokXq2LGj5%2B%2B5ublKS0tT9%2B7dNXz4cG3dutXr8Rs3btTQoUPVvXt3ZWRkKD8/39dLtmbNmjXq27evunbtqvHjx%2BvIkSOSGk4G%2B/fv17333quePXuqT58%2BmjFjhucX0tfXDN566y2lpKRo6tSpteYu9PVdU1OjlStXatCgQerVq5ceeOABff755555t9ut7OxspaSkqG/fvpozZ44qKip8sqdLdaEMXnvtNY0cOVLdunXT0KFD9fvf/95r/kKf98rKSj366KPq16%2BfkpOTlZWVJZfLdcX3821cKINzysrKNGDAAM2aNcszdlW8DgwCzujRo83cuXNNSUmJOXz4sBkyZIjZsGGDv5dl3R133GFmzZplTp06ZQoLC82YMWPMI488YsrLy80tt9xiVq1aZcrKykx%2Bfr7p3bu32blzpzHGmP3795vExESzZ88eU1FRYV566SWTlJRkCgsL/byjb2///v2md%2B/epkOHDsYYY44dO2a6du1qNm/ebCoqKsyf/vQn06VLF/PPf/7TGGPMm2%2B%2BaXr27Gnee%2B89U15ebtatW2f69OljysrK/LmNb2XTpk3mtttuM4cOHTKlpaXm8ccfN48//niDyeDs2bOmT58%2BZvny5aaystKcOHHC/OAHPzBTpkyptxmsX7/eDBkyxIwbN85kZ2d7zV3s63vjxo1m4MCB5uDBg6a0tNT89Kc/NSNGjDA1NTXGGGMefvhh8%2BCDD5rjx4%2Bbo0ePmvT0dPP444/7fI8Xc6EM3n//fdO5c2fz%2Buuvm7Nnz5o9e/aYhIQE89e//tUYc/HP%2B%2BLFi82YMWPMv//9b%2BNyuczDDz9sJk6c6PM9XsyFMviqxYsXmx49epiZM2d6xq6G1wEFK8D885//NHFxccbtdnvGfvvb35qhQ4f6cVX2nTx50syaNcsUFxd7xp599lkzZMgQs2PHDnPTTTeZqqoqz9zSpUvN/fffb4wxZsGCBSYzM9PreHfffbdZt26dbxZvWXV1tbn77rvNf//3f3sK1tNPP21GjRrl9bjs7Gwzb948Y4wxDz74oFm0aJHXMfr06WO2b9/uu4Vbkpqa6inPX9VQMvj3v/9tOnToYA4ePOgZ%2B%2B1vf2tuvfXWepvBM888Y0pKSszMmTNr/cN6sa/v4cOHm2eeecYzV1paauLj482%2BfftMcXGx6dSpkzlw4IBnfu/evaZr167mzJkzV3BHl%2B5CGezdu9esXr3aa2z06NFmzZo1xpgLf97Pnj1revToYd544w3P/MGDB03Hjh3N0aNHr%2BCOLt2FMjjnwIEDpk%2BfPmbhwoVeBetqeB1wiTDAOJ1OtWzZUpGRkZ6xhIQEHT58WKdOnfLjyuyKiIjQ4sWL1bx5c89YYWGhYmNj5XQ61bFjR4WEhHjm4uPjPafAnU6n4uPjvY4XHx%2BvvLw83yzesueff16hoaEaMWKEZ%2Byb9vhNGQQHBysuLi7gMjh27JiOHDmikydPatiwYZ7LGSdOnGgwGbRo0UJxcXHKyclRWVmZjh8/rtdee00DBgyotxnce%2B%2B9atq06XnnLvT1XVFRoYMHD3rNh4eH68Ybb1ReXp4OHDigkJAQr0vtCQkJOn36tD755JMrs5lv6UIZ9OvXT5mZmZ6/V1VVqbi4WC1atJB04c/7Z599ptLSUiUkJHjm27Ztq7CwMDmdziu0m2/nQhlIkjFG8%2BfP19SpUxUREeEZv1peBxSsAON2u71eSJI8ZetqvYZuQ15enjZt2qSHHnrovBlERUXJ7XarpqZGbrfbq4BKX2YUiPl88cUXWrVqlR577DGv8W/K4Nwe60sGR48elSS9%2Buqr%2BvWvf62XXnpJR48e1dy5cxtMBsHBwVq1apXefPNNde/eXSkpKaqqqtL06dMbTAZfdaE9nTx5UsaYb5x3u90KDw9XUFCQ15wU2P/9XLZsmRo3bqxhw4ZJunBGbrdbkmq9biIiIgIug5ycHAUFBWnMmDFe41fL64CCFYCMMf5egk/9/e9/1wMPPKDp06crJSXlGx/31S%2BW%2BpLR4sWLNWbMGLVr1%2B6Sn1sfMji3hwkTJqhFixa6/vrrNWXKFO3ateuSnh/Izpw5o0mTJum2227T3/72N/3xj39U06ZNNWPGjDo9vz5k8HUX29OF5utTHsYYLV26VNu3b9eaNWsUGhrqNXex5way48eP68knn9T8%2BfO9/tv/Vf5%2BHVCwAkxMTIznO5Bz3G63goKCFBMT46dVXTm7du3Sgw8%2BqEceeUT33nuvpC8z%2BPp3GW63W1FRUQoODlZ0dPR5Mwq0fHJzc7Vv3z6vSwHnnG%2BPLpfLs8f6ksG5S8Rf/W67ZcuWMsbo7NmzDSKD3NxcHTlyRNOmTVPTpk3VokULZWVl6fXXX1dwcHCDyOCrLrSnc/8NON98s2bNFBMTo1OnTqm6utprTpKaNWt25RdvUU1NjWbNmqVdu3bpd7/7nb73ve955i6U0bnP/dfnT548GVAZLFmyRKNGjfK6zHfO1fI6oGAFmMTERBUWFnp%2BRFv68vJZu3bt1KRJEz%2BuzL5//OMfmjlzpp588kmNGjXKM56YmKgPP/xQVVVVnrG8vDwlJSV55r/%2Bo%2BhfnQ8UW7du1fHjxzVw4EAlJyd7ToMnJyerQ4cOtfaYn5/vlcFX76eorq7W/v37Ay6D66%2B/XuHh4Tpw4IBnrKCgQNdcc4369%2B/fIDKorq5WTU2N13fcZ86ckSSlpKQ0iAy%2B6kJf36GhoWrfvr3XnktKSvTZZ5%2BpS5cuiouLkzFGH3zwgddzIyIi1KZNG5/twYZFixbp448/1u9%2B9zu1atXKa%2B5Cn/dWrVopMjLSa/6jjz7SmTNnlJiY6LP1X66tW7dqy5YtSk5OVnJysp5%2B%2Bmm9/PLLSk5OvnpeB9Zul4fP3H333eaRRx4xpaWl5uDBgyY1NdVs2rTJ38uy6uzZs%2Bb22283zz//fK25yspKM3DgQPPLX/7SnD592rz33numZ8%2BeZvfu3cYYYz788EPTuXNns3v3blNRUWE2b95sunXrZoqKiny8i8vjdrtNYWGh58%2B%2BfftMhw4dTGFhoSkoKDDdunUzv//9701FRYXZs2eP6dKli%2BenYvbu3Wt69Ohh9u3bZ06fPm1WrVpl%2Bvfvb8rLy/28q0u3aNEiM2jQIPPpp5%2BaL774wqSnp5tZs2aZL774okFkcOLECdO7d2%2BzYsUKc/r0aXPixAkzadIk81//9V/1PoPz/fTYxb6%2Bf/vb35oBAwZ4fjx/3rx55q677vI8Pzs720yYMMEcP37cFBYWmrvuusssWbLEp/u6FOfL4G9/%2B5vp1auX109Zf9XFPu9Lly41o0ePNv/%2B97/NiRMnzMSJE82UKVOu%2BF6%2BrfNl8NX/NhYWFppFixaZrKwsz9t1XA2vAwpWACosLDQTJkwwXbp0MSkpKeaXv/yl57096ou//vWvpkOHDiYxMbHWnyNHjpgPP/zQjBs3ziQmJpoBAwaY5557zuv5O3fuNEOGDDEJCQnmzjvvNH/5y1/8tBN7Pv/8c8/bNBhjzF/%2B8hczcuRIk5CQYIYMGVLrrQyee%2B45079/f5OYmGgyMjLMhx9%2B6OslW1FZWWnmz59vevXqZbp27WpmzpxpTp06ZYxpOBnk5eWZe%2B65x/Ts2dOkpKSY7Oxsz4/U18cMzn2td%2BrUyXTq1Mnz93Mu9PVdU1NjnnzySXPzzTebLl26mB/%2B8Ide74FXUlJipk6darp27Wp69eplFixYYCorK326v7q4UAazZ8/2Gjv35wc/%2BIHn%2BRf6vH/1a6pbt25m2rRppqSkxOd7vJiLvQ6%2B6pe//KXX2zRcDa%2BDIGMC/E43AACAqwz3YAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZf8fRH7xHA0qJ1UAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common2036734965981555444">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">1590</td>
        <td class="number">10.5%</td>
        <td>
            <div class="bar" style="width:18%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30</td>
        <td class="number">1207</td>
        <td class="number">8.0%</td>
        <td>
            <div class="bar" style="width:14%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">150</td>
        <td class="number">497</td>
        <td class="number">3.3%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">490</td>
        <td class="number">3.2%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">42</td>
        <td class="number">452</td>
        <td class="number">3.0%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">67</td>
        <td class="number">411</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">85</td>
        <td class="number">381</td>
        <td class="number">2.5%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">108</td>
        <td class="number">361</td>
        <td class="number">2.4%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">90</td>
        <td class="number">284</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">120</td>
        <td class="number">283</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (390)</td>
        <td class="number">9164</td>
        <td class="number">60.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme2036734965981555444">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">1590</td>
        <td class="number">10.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30</td>
        <td class="number">1207</td>
        <td class="number">8.0%</td>
        <td>
            <div class="bar" style="width:76%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">42</td>
        <td class="number">452</td>
        <td class="number">3.0%</td>
        <td>
            <div class="bar" style="width:29%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">490</td>
        <td class="number">3.2%</td>
        <td>
            <div class="bar" style="width:31%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">67</td>
        <td class="number">411</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:26%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1260</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1261</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1294</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1318</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1343</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Horizontal_Distance_To_Roadways">Horizontal_Distance_To_Roadways<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>3250</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>21.5%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1714</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>6890</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram8647624807666160178">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAATBJREFUeJzt28FtwkAARUGMUlKKSE8501OKSE9LA%2BjJoDhe45k70l6evsXayxhjXICHrnsfAGb2sfcB/sLn98/Tv/m9fW1wEt6NBYEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEw3du8r7yZC1uxIBAEAkEgEAQCQSAQpvsX67/4jp01LAgEgUAQCASBQBAIBIFAEAgEgUAQCITT3qS/4tnbdzfvx2dBIAgEgkAgCASCQCAIBIJAILgH2ZCvFo/PgkCwIJOxOnMRyBvwCsx2PGJBsCAn5DFuPQsCwYKwyllXx4JAWMYYY%2B9DwKwsCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCIQ7mV0iBuHRtDgAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives8647624807666160178,#minihistogram8647624807666160178"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives8647624807666160178">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles8647624807666160178"
                                                  aria-controls="quantiles8647624807666160178" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram8647624807666160178" aria-controls="histogram8647624807666160178"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common8647624807666160178" aria-controls="common8647624807666160178"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme8647624807666160178" aria-controls="extreme8647624807666160178"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles8647624807666160178">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>242</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>764</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1316</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2270</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>4635.1</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>6890</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>6890</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1506</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>1325.1</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.77307</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.0224</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1714</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>1030.8</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.2478</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>25916031</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>1755800</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram8647624807666160178">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xt0VOW9//FPLhI0ITckaCkKh5sk4S5EA3JVEBFECIR0WaWCoERzErUFQRSLBU5BUcGD0C49IJ5lhFq5KOJSLtVKa1cFTUJACXDANJgIM%2BRCEkny/P7wx9QxKIE87GEy79dari6eZ2bn%2B5kk5cOenZ0gY4wRAAAArAn29QAAAABNDQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFgW6usBAkVJSZn1YwYHByk2NlwnTlSors5YP/6ljOyBmV0K7PxkD8zsUmDnb2z2Vq1aXISpzo0zWH4sODhIQUFBCg4O8vUojiN7YGaXAjs/2QMzuxTY%2Bf01OwULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACwL9fUACBwjn/urr0c4L1sy%2B/t6BACAn%2BIMFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwrEkUrAULFqhLly6eP%2B/atUspKSnq3bu3Ro0apY0bN3o9fs2aNRoxYoR69%2B6ttLQ05ebmevaqq6v1xBNPaODAgUpKSlJGRoZcLpdjWQAAgP/z%2B4KVn5%2BvDRs2eP5cXFysGTNmaNKkSdq1a5fmzJmjuXPnKicnR5K0bds2LVu2TL///e/18ccfa8iQIbr//vt16tQpSdLSpUuVl5en7Oxsbd26VcYYPfbYYz7JBgAA/JNfF6y6ujo9%2BeSTmjx5smdt06ZNateunVJSUhQWFqbk5GQNHTpU69atkyRlZ2dr3Lhx6tGjh5o3b66pU6dKkrZv366amhqtX79eM2bM0NVXX63o6GhlZmZqx44d%2Bvrrr30REQAA%2BCG/Llivv/66wsLCNHr0aM9aXl6e4uPjvR4XHx/veRvwh/vBwcHq2rWrcnJydOTIEZWVlSkhIcGz36FDBzVv3lx5eXkXOQ0AAGgqQn09wIX65ptvtGzZMr366qte6263W61bt/Zai46O9lxH5Xa7FRUV5bUfFRUll8slt9stSYqMjPTaj4yMPK/rsIqLi1VSUuK1Fhp6heLi4hp8jIYICQn2%2Bl/YFRp6ab6ugf55D%2BT8ZA/M7FJg5/fX7H5bsBYuXKhx48apY8eO%2Buqrr87rucaYRu2fS3Z2tpYvX%2B61lp6eroyMjEYd98dERl5%2BUY4b6GJiwn09wk8K9M97IOcne%2BAK5Pz%2Blt0vC9auXbu0e/dubd68ud5eTEyM50zUGS6XS7GxsT%2B673a71alTJ89j3G63wsP//ZfryZMn1bJlywbPl5qaqqFDh3qthYZeIZerosHHaIiQkGBFRl6u0tJK1dbWWT02ZP3zZUugf94DOT/ZAzO7FNj5G5vdV/9Y9suCtXHjRh0/flxDhgyR9O8zTklJSbr33nvrFa/c3Fz16NFDkpSYmKi8vDzdeeedkqTa2lrt3btXKSkpatu2raKiopSXl6c2bdpIkr744gt9%2B%2B23SkxMbPB8cXFx9d4OLCkpU03NxfmmqK2tu2jHDmSX%2Bmsa6J/3QM5P9sDMLgV2fn/L7l9vaP5/s2bN0tatW7VhwwZt2LBBq1atkiRt2LBBo0ePVmFhodatW6fq6mrt3LlTO3fu1MSJEyVJaWlpeuutt7Rnzx5VVlZqxYoVatasmQYPHqyQkBBNnDhRL730koqKiuRyufTss8/qlltu0ZVXXunLyAAAwI/45RmsqKgorwvVa2pqJElXXXWVJGnlypV6%2Bumn9dRTT6lNmzZavHixrrvuOknSwIED9fDDDyszM1PHjx9Xt27dtGrVKjVv3lySlJGRoYqKCt1xxx2qqanRkCFDNG/ePGcDAgAAvxZkGntFNxqkpKTM%2BjFDQ4MVExMul6vCL06bjnzur74e4bxsyezv6xHOyt8%2B77YFcn6yB2Z2KbDzNzZ7q1YtLsJU5%2BaXbxECAABcyihYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMCyUF8PgMa5fs67vh4BAAD8AGewAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlfluw9u3bp3vuuUd9%2BvRRcnKyMjMzVVJSIknatWuXUlJS1Lt3b40aNUobN270eu6aNWs0YsQI9e7dW2lpacrNzfXsVVdX64knntDAgQOVlJSkjIwMuVwuR7MBAAD/5pcF69tvv9W9996rfv36adeuXdq8ebOOHz%2BuefPmqbi4WDNmzNCkSZO0a9cuzZkzR3PnzlVOTo4kadu2bVq2bJl%2B//vf6%2BOPP9aQIUN0//3369SpU5KkpUuXKi8vT9nZ2dq6dauMMXrsscd8GRcAAPgZvyxYlZWVysrK0vTp09WsWTPFxsbqlltu0ZdffqlNmzapXbt2SklJUVhYmJKTkzV06FCtW7dOkpSdna1x48apR48eat68uaZOnSpJ2r59u2pqarR%2B/XrNmDFDV199taKjo5WZmakdO3bo66%2B/9mVkAADgR/yyYEVFRWnChAkKDQ2VJB08eFB//vOfNXLkSOXl5Sk%2BPt7r8fHx8Z63AX%2B4HxwcrK5duyonJ0dHjhxRWVmZEhISPPsdOnRQ8%2BbNlZeX50AyAADQFIT6eoDGKCws1IgRI1RTU6OJEycqIyND9913n1q3bu31uOjoaM91VG63W1FRUV77UVFRcrlccrvdkqTIyEiv/cjIyPO6Dqu4uNhzPdgZoaFXKC4ursHHaIiQEL/sx34jNPTSfH3PfN4D9fMfyPnJHpjZpcDO76/Z/bpgtWnTRjk5Ofq///s/PfHEE/rNb37ToOcZYxq1fy7Z2dlavny511p6eroyMjIadVw4KyYm3Ncj/KTIyMt9PYJPBXJ%2BsgeuQM7vb9n9umBJUlBQkNq1a6esrCxNmjRJgwYN8pyJOsPlcik2NlaSFBMTU2/f7XarU6dOnse43W6Fh//7L9eTJ0%2BqZcuWDZ4pNTVVQ4cO9VoLDb1CLlfFeWU7F39r8/7G9ufLlpCQYEVGXq7S0krV1tb5ehzHBXJ%2Bsgdmdimw8zc2u6/%2BseyXBWvXrl2aN2%2BetmzZouDg70rGmf/t3r27tm7d6vX43Nxc9ejRQ5KUmJiovLw83XnnnZKk2tpa7d27VykpKWrbtq2ioqKUl5enNm3aSJK%2B%2BOILffvtt0pMTGzwfHFxcfXeDiwpKVNNTWB9U/i7S/3zVVtbd8nPeDEFcn6yB2Z2KbDz%2B1t2x0%2BBDB06VMuXL1dRUdEFHyMxMVHl5eVavHixKisrdeLECS1btkzXX3%2B90tLSVFhYqHXr1qm6ulo7d%2B7Uzp07NXHiRElSWlqa3nrrLe3Zs0eVlZVasWKFmjVrpsGDByskJEQTJ07USy%2B9pKKiIrlcLj377LO65ZZbdOWVV9p6CQAAQBPneMEaP3683nnnHd18882aOnWq3nvvPdXU1JzXMVq0aKGXX35Zubm5uuGGGzRq1Ci1aNFCzz77rFq2bKmVK1dq7dq16tOnjxYsWKDFixfruuuukyQNHDhQDz/8sDIzM9WvXz99/PHHWrVqlZo3by5JysjIUI8ePXTHHXdo2LBhCg8P1%2B9%2B9zvrrwMAAGi6gkxjr%2Bi%2BQHl5edq8ebO2bNmi06dPa%2BzYsUpJSVH79u19Mc5FV1JSZv2YoaHBumXJh9aPi%2B9syezv6xHOKjQ0WDEx4XK5KvzqdLktgZyf7IGZXQrs/I3N3qpVi4sw1bn57CrphIQEzZw5U9u3b9fs2bP1xhtv6LbbbtOUKVP0%2Beef%2B2osAACARvNZwTp9%2BrTeeecd3XfffZo5c6Zat26txx57TF27dtXkyZO1adMmX40GAADQKI7/FGFBQYHWr1%2Bvt956SxUVFRoxYoRWr16tPn36eB7Tt29fzZs3T6NHj3Z6PAAAgEZzvGCNGjVK7du31/Tp0zV27FhFR0fXe8ygQYN04sQJp0cDAACwwvGCtWbNGvXr1%2B%2Bcj/vss88cmAYAAMA%2Bx6/B6tKli%2B6//369//77nrX/%2BZ//0X333VfvDusAAAD%2ByPGCtXDhQpWVlaljx46etcGDB6uurk6LFi1yehwAAADrHH%2BL8KOPPtKmTZsUExPjWWvXrp2WLFmi22%2B/3elxAAAArHP8DFZVVZXCwsLqDxIcrMrKSqfHAQAAsM7xgtW3b18tWrRIJ0%2Be9Kx9/fXXeuqpp7xu1QAAAOCvHH%2BLcPbs2br33nt14403KiIiQnV1daqoqFDbtm316quvOj0OAACAdY4XrLZt2%2Brtt9/WX/7yFx05ckTBwcFq3769BgwYoJCQEKfHAQAAsM7xgiVJzZo108033%2ByLDw002Mjn/urrEc7LpfrLqQEgEDlesI4ePapnnnlGX375paqqqurtf/DBB06PBAAAYJVPrsEqLi7WgAEDdMUVVzj94QEAAC46xwtWbm6uPvjgA8XGxjr9oQEAABzh%2BG0aWrZsyZkrAADQpDlesKZPn67ly5fLGOP0hwYAAHCE428R/uUvf9Gnn36qN998Uz//%2Bc8VHOzd8V5//XWnRwIAALDK8YIVERGhgQMHOv1hAQAAHON4wVq4cKHTHxIAAMBRjl%2BDJUkHDx7UsmXL9Nhjj3nWdu/e7YtRAAAArHO8YO3atUtjxozRe%2B%2B9p82bN0v67uajd999NzcZBQAATYLjBWvp0qX69a9/rU2bNikoKEjSd7%2BfcNGiRXrxxRedHgcAAMA6xwvWF198obS0NEnyFCxJuvXWW1VQUOD0OAAAANY5XrBatGhx1t9BWFxcrGbNmjk9DgAAgHWOF6zevXtrwYIFKi8v96wdOnRIM2fO1I033uj0OAAAANY5fpuGxx57TPfcc4%2BSkpJUW1ur3r17q7KyUp06ddKiRYucHgcAAMA6xwvWVVddpc2bN2vnzp06dOiQmjdvrvbt26t///5e12QBAAD4K8cLliRddtlluvnmm33xoQEAAC46xwvW0KFDf/JMFffCAgAA/s7xgnXbbbd5Faza2lodOnRIOTk5uueee5weBwAAwDrHC9ajjz561vWtW7fq73//u8PTAAAA2OeT30V4NjfffLPefvttX48BAADQaJdMwdq7d6%2BMMb4eAwAAoNEcf4tw0qRJ9dYqKytVUFCg4cOHOz0OAACAdY4XrHbt2tX7KcKwsDClpKRowoQJTo8DAABgneMFi7u1AwCAps7xgvXWW281%2BLFjx469iJMAAABcHI4XrDlz5qiurq7eBe1BQUFea0FBQRQsAADglxwvWH/84x/18ssv6/7771eXLl1kjNH%2B/fv1hz/8QXfddZeSkpKcHgkAAMAqn1yDtWrVKrVu3dqzdv3116tt27aaMmWKNm/e7PRIAAAAVjl%2BH6zDhw8rKiqq3npkZKQKCwudHgcAAMA6xwtWmzZttGjRIrlcLs9aaWmpnnnmGV1zzTVOjwMAAGCd428Rzp49W4888oiys7MVHh6u4OBglZeXq3nz5nrxxRedHgcAAMA6xwvWgAEDtGPHDu3cuVPHjh2TMUatW7fWTTfdpBYtWjg9DgAAgHWOFyxJuvzyyzVs2DAdO3ZMbdu29cUIAAAAF43j12BVVVVp5syZ6tWrl0aOHCnpu2uwpk6dqtLSUqfHAQAAsM7xgrV48WLl5%2BdryZIlCg7%2B94evra3VkiVLnB4HAADAOscL1tatW/XCCy/o1ltv9fzS58jISC1cuFDvvfee0%2BMAAABY53jBqqioULt27eqtx8bG6tSpU06PAwAAYJ3jBeuaa67R3//%2Bd0ny%2Bt2D7777rn72s585PQ4AAIB1jv8U4S9%2B8Qs99NBDGj9%2BvOrq6vTKK68oNzdXW7du1Zw5c5weBwAAwDrHC1ZqaqpCQ0O1du1ahYSE6KWXXlL79u21ZMkS3XrrrU6PAwAAYJ3jBevEiRMaP368xo8f7/SHBgAAcITj12ANGzbM69qrC1VYWKj09HQlJSUpOTlZs2bN8txHKz8/X3fddZf69Omj4cOH6%2BWXX/Z67jvvvKPRo0erV69eGjdunD766CPPXl1dnZYuXaphw4apb9%2B%2BmjJlio4ePdroeQEAQOBwvGAlJSVpy5YtjT7O/fffr8jISG3btk1vvvmmvvzyS/3Xf/2XqqqqNH36dN1www368MMPtXTpUq1cudJzC4j8/HzNnDlTjz76qP72t79p8uTJevDBB3Xs2DFJ0muvvaZNmzZp1apV2r59u9q1a6f09HQrpRAAAAQGx98ivPrqq/W73/1Oq1at0jXXXKPLLrvMa/%2BZZ5455zFKS0uVmJioRx55ROHh4QoPD9edd96pV199VTt27NDp06f1wAMPKCQkRAkJCZowYYKys7M1fPhwrVu3ToMGDdKgQYMkSWPGjNHatWu1ceNGTZs2TdnZ2Zo8ebI6dOggScrKylJSUpI%2B%2B%2Bwz9ezZ0/4LAgAAmhzHC9aBAwf0H//xH5Ikl8t1Qcc4c2PS7ysqKlJcXJzy8vLUpUsXhYSEePbi4%2BO1bt06SVJeXp6nXH1/PycnR1VVVTpw4IDi4%2BM9exEREbr22muVk5PT4IJVXFyskpISr7XQ0CsUFxd3XjnPJSTE8ROQuISFhgbG18OZr/tA/Pone2BmlwI7v79md6xgZWVlaenSpXr11Vc9ay%2B%2B%2BKLS09MbfeycnBytXbtWK1as0JYtWxQZGem1Hx0dLbfbrbq6OrndbkVFRXntR0VF6cCBAzp58qSMMWfdP58ymJ2dreXLl3utpaenKyMj4zyTAQ0XExPu6xEcFRl5ua9H8BmyB65Azu9v2R0rWNu2bau3tmrVqkYXrH/%2B85964IEH9Mgjjyg5OflHr%2B8682t5JJ3zeqrGXm%2BVmpqqoUOHeq2Fhl4hl6uiUcf9IX9r87i4bH99XapCQoIVGXm5SksrVVtb5%2BtxHEX2wMwuBXb%2Bxmb31T8%2BHStYZystjS0y27Zt069//WvNnTtXY8eOlfTdr9w5fPiw1%2BPcbreio6MVHBysmJgYud3uevuxsbGex5xtv2XLlg2eKy4urt7bgSUlZaqpCaxvCjgr0L6%2BamvrAi7zGWQPzOxSYOf3t%2ByOnQL5/hmkn1prqE8//VQzZ87U888/7ylXkpSYmKj9%2B/erpqbGs5aTk6MePXp49nNzc72OdWY/LCxMnTp1Ul5enmevtLRUR44cUffu3S94VgAAEFj88j2mmpoaPf7443r00Uc1YMAAr71BgwYpIiJCK1asUGVlpT777DOtX79eaWlpkqSJEyfq448/1o4dO1RdXa3169fr8OHDGjNmjCQpLS1Na9asUUFBgcrLy7VkyRJ17dpV3bp1czwnAADwT47/FKENe/bsUUFBgZ5%2B%2Bmk9/fTTXnvvvvuuXnrpJT355JNatWqVrrzySmVlZWnw4MGSpM6dO2vJkiVauHChCgsL1bFjR61cuVKtWrWSJE2aNEklJSX65S9/qYqKCiUlJdW7YB0AAOCnBBmH7qAZHx%2BvkSNHeq1t2bKl3lpD7oPlj0pKyqwfMzQ0WLcs%2BdD6ceGftmT29/UIjggNDVZMTLhcrgq/uh7DBrIHZnYpsPM3NnurVi0uwlTn5tgZrD59%2Bqi4uPicawAAAP7OsYL1/ftfAQAANGV%2BeZE7AADApYyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAsC/X1AADsGPncX309QoNtyezv6xEA4KLiDBYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALPPrgvXhhx8qOTlZWVlZ9fbeeecdjR49Wr169dK4ceP00Ucfefbq6uq0dOlSDRs2TH379tWUKVN09OhRz77b7VZmZqaSk5M1YMAAzZkzR1VVVY5kAgAA/s9vC9Yf/vAHPf3007r22mvr7eXn52vmzJl69NFH9be//U2TJ0/Wgw8%2BqGPHjkmSXnvtNW3atEmrVq3S9u3b1a5dO6Wnp8sYI0maO3euKisrtXnzZv3pT39SQUGBlixZ4mg%2BAADgv/y2YIWFhWn9%2BvVnLVjr1q3ToEGDNGjQIIWFhWnMmDHq3LmzNm7cKEnKzs7W5MmT1aFDB0VERCgrK0sFBQX67LPP9M033%2Bj9999XVlaWYmNj1bp1a82YMUN/%2BtOfdPr0aadjAgAAP%2BS3Bevuu%2B9WixYtzrqXl5en%2BPh4r7X4%2BHjl5OSoqqpKBw4c8NqPiIjQtddeq5ycHOXn5yskJERdunTx7CckJOjUqVM6ePDgxQkDAACalFBfD3AxuN1uRUVFea1FRUXpwIEDOnnypIwxZ913uVyKjo5WRESEgoKCvPYkyeVyNejjFxcXq6SkxGstNPQKxcXFXUicHxUS4rf9GAEuNPTCv3bPfN0H4tc/2QMzuxTY%2Bf01e5MsWJI811NdyP65nnsu2dnZWr58uddaenq6MjIyGnVcoKmIiQlv9DEiIy%2B3MIl/InvgCuT8/pa9SRasmJgYud1urzW3263Y2FhFR0crODj4rPstW7ZUbGysysvLVVtbq5CQEM%2BeJLVs2bJBHz81NVVDhw71WgsNvUIuV8WFRjorf2vzwBmN%2BV4ICQlWZOTlKi2tVG1tncWpLn1kD8zsUmDnb2x2G/%2BguxBNsmAlJiYqNzfXay0nJ0ejRo1SWFiYOnXqpLy8PPXr10%2BSVFpaqiNHjqh79%2B5q06aNjDHat2%2BfEhISPM%2BNjIxU%2B/btG/Tx4%2BLi6r0dWFJSppqawPqmAH6Mje%2BF2tq6gP2eIntgZpcCO7%2B/ZW%2BSp0AmTpyojz/%2BWDt27FB1dbXWr1%2Bvw4cPa8yYMZKktLQ0rVmzRgUFBSovL9eSJUvUtWtXdevWTbGxsRoxYoSee%2B45nThxQseOHdOLL76olJQUhYY2yT4KAAAs89vG0K1bN0lSTU2NJOn999%2BX9N3Zps6dO2vJkiVauHChCgsL1bFjR61cuVKtWrWSJE2aNEklJSX65S9/qYqKCiUlJXldM/Xb3/5WTz75pIYNG6bLLrtMt99%2B%2B1lvZgoAAHA2QaaxV3SjQUpKyqwfMzQ0WLcs%2BdD6cQF425LZ39cjSPruez4mJlwuV4VfvVViQyBnlwI7f2Ozt2p19ls6XWxN8i1CAAAAX6JgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlob4eAAAudSOf%2B6uvRzgvWzL7%2B3oEIOBxBgsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGTcaBYAmxp9ujMpNUdFUcQYLAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDLu5A4A8Bl/uuu8xJ3n0XCcwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZfyyZwAAGohfTo2G4gwWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIK1lkUFhZq2rRpSkpK0pAhQ7R48WLV1dX5eiwAAOAnuJP7WTz00ENKSEjQ%2B%2B%2B/r%2BPHj2v69Om68sor9atf/crXowEAAD9AwfqBnJwc7du3T6%2B88opatGihFi1aaPLkyVq9ejUFCwDgV/zpV/s0tV/rQ8H6gby8PLVp00ZRUVGetYSEBB06dEjl5eWKiIg45zGKi4tVUlL2Yrk9AAANF0lEQVTitRYaeoXi4uKszhoSwju8AICmITT07H%2Bnnfm7zt/%2BzqNg/YDb7VZkZKTX2pmy5XK5GlSwsrOztXz5cq%2B1Bx98UA899JC9QfVdkbvnqi%2BVmppqvbxd6oqLi5WdnU32AMsuBXZ%2Bsgdmdimw8xcXF2v16j/6XXb/qoMOMcY06vmpqal68803vf5LTU21NN2/lZSUaPny5fXOlgUCsgdmdimw85M9MLNLgZ3fX7NzBusHYmNj5Xa7vdbcbreCgoIUGxvboGPExcX5VcsGAAB2cQbrBxITE1VUVKQTJ0541nJyctSxY0eFh4f7cDIAAOAvKFg/EB8fr27duumZZ55ReXm5CgoK9MorrygtLc3XowEAAD8RMm/evHm%2BHuJSc9NNN2nz5s2aP3%2B%2B3n77baWkpGjKlCkKCgry9Wj1hIeHq1%2B/fgF5do3sgZldCuz8ZA/M7FJg5/fH7EGmsVd0AwAAwAtvEQIAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsHyQ4WFhZo2bZqSkpI0ZMgQLV68WHV1db4eq1E%2B/PBDJScnKysrq97eO%2B%2B8o9GjR6tXr14aN26cPvroI89eXV2dli5dqmHDhqlv376aMmWKjh496tl3u93KzMxUcnKyBgwYoDlz5qiqqsqRTA1VWFio9PR0JSUlKTk5WbNmzVJpaakkKT8/X3fddZf69Omj4cOH6%2BWXX/Z6bmNem0vBvn37dM8996hPnz5KTk5WZmamSkpKJEm7du1SSkqKevfurVGjRmnjxo1ez12zZo1GjBih3r17Ky0tTbm5uZ696upqPfHEExo4cKCSkpKUkZEhl8vlaLbzsWDBAnXp0sXz50DI3qVLFyUmJqpbt26e/%2BbPny8pMPKvWLFCAwYMUM%2BePTV58mR99dVXkpp29n/84x9en%2B9u3bopMTHR87Xf5LIb%2BJ0777zTPP7446a0tNQcOnTIDB8%2B3Lz88su%2BHuuCrVq1ygwfPtxMmjTJZGZmeu3t3bvXJCYmmh07dpiqqiqzYcMG06NHD1NUVGSMMWbNmjVmyJAh5sCBA6asrMz89re/NaNHjzZ1dXXGGGMefPBBM23aNHP8%2BHFz7Ngxk5qaaubPn%2B94xp9y%2B%2B23m1mzZpny8nJTVFRkxo0bZ2bPnm0qKyvNTTfdZJYtW2YqKipMbm6u6devn9m6dasxpvGvja9VV1ebG2%2B80SxfvtxUV1eb48ePm7vuusvMmDHDfP3116Znz55m3bp1pqqqyvz1r3813bt3N59//rkxxpgPPvjAXH/99WbPnj2msrLSrFy50vTv399UVFQYY4xZuHChGTdunPnXv/5lXC6XefDBB8306dN9GfdH7d271/Tr18907tzZGGMCJnvnzp3N0aNH660HQv61a9eaW2%2B91RQUFJiysjIzf/58M3/%2B/IDI/kMrVqww//mf/9kks1Ow/Mznn39uunbtatxut2ftf//3f82IESN8OFXjrF692pSWlpqZM2fWK1hPPfWUSU9P91qbMGGCWblypTHGmFGjRpnVq1d79srKykx8fLzZvXu3KSkpMdddd53Jz8/37O/cudP07NnTfPvttxcxUcOdPHnSzJo1y5SUlHjWXn31VTN8%2BHCzZcsWc8MNN5iamhrP3uLFi829995rjGnca3MpcLvd5o033jCnT5/2rK1evdrccsst5o9//KMZO3as1%2BMzMzPN3LlzjTHGTJs2zSxYsMCzV1tba/r37282b95sTp8%2Bbfr06WPef/99z/6BAwdMly5dzLFjxy5yqvNTW1trJkyYYP77v//bU7ACJfuPFaxAyD906FDPP5S%2BLxCyf19hYaHp16%2BfKSwsbJLZeYvQz%2BTl5alNmzaKioryrCUkJOjQoUMqLy/34WQX7u6771aLFi3OupeXl6f4%2BHivtfj4eOXk5KiqqkoHDhzw2o%2BIiNC1116rnJwc5efnKyQkxOutl4SEBJ06dUoHDx68OGHOU2RkpBYuXKgrr7zSs1ZUVKS4uDjl5eWpS5cuCgkJ8ezFx8d7Tos35rW5FERFRWnChAkKDQ2VJB08eFB//vOfNXLkyB/N9mPZg4OD1bVrV%2BXk5OjIkSMqKytTQkKCZ79Dhw5q3ry58vLyHEjWcK%2B//rrCwsI0evRoz1qgZJekZ555RoMHD9b111%2BvuXPnqqKiosnn//rrr/XVV1/p5MmTuu222zxvZ504caLJZ/%2Bh559/XuPHj9fPfvazJpmdguVn3G63IiMjvdbOlC2fv998Ebjdbq8yKX2X1%2BVy6eTJkzLG/Oi%2B2%2B1WRESEgoKCvPakS/e1ysnJ0dq1a/XAAw%2Bc9XMdHR0tt9uturq6Rr02l5LCwkIlJibqtttuU7du3ZSRkfGj2c/M/lPZ3W63JNV7fmRk5CWV/ZtvvtGyZcv05JNPeq0HQnZJ6tmzp5KTk/Xee%2B8pOztbe/bs0VNPPdXk8x87dkyS9O677%2BqVV17Rhg0bdOzYMT3%2B%2BONNPvv3ffXVV3rvvff0q1/9SlLT/LqnYPkhY4yvR3DUufL%2B1L4/vVb//Oc/NWXKFD3yyCNKTk7%2B0cd9vzA25rW5VLRp00Y5OTl69913dfjwYf3mN79p0PP8PfvChQs1btw4dezY8byf6%2B/ZJSk7O1sTJkxQs2bN1KFDBz366KPavHmzTp8%2Bfc7n%2BnP%2BM7NNnTpVrVu31lVXXaWHHnpI27ZtO6/nX%2Bj%2BpeK1117T8OHD1apVqwY/x9%2ByU7D8TGxsrKetn%2BF2uxUUFKTY2FgfTXXxxMTEnDVvbGysoqOjFRwcfNb9li1bKjY2VuXl5aqtrfXak6SWLVte/OHPw7Zt2zRt2jTNnj1bd999t6TvPtc//NeX2%2B325G7Ma3OpCQoKUrt27ZSVlaXNmzcrNDS03uwul8vzNf5T2c885of7J0%2BevGSy79q1S7t371Z6enq9vbNla0rZf8zPf/5z1dbWnvXrtinlP3M5wPfPtrRp00bGGJ0%2BfbpJZ/%2B%2BrVu3aujQoZ4/N8WvewqWn0lMTFRRUZFOnDjhWcvJyVHHjh0VHh7uw8kujsTERK8fxZW%2By9ujRw%2BFhYWpU6dOXu%2Bxl5aW6siRI%2Brevbu6du0qY4z27dvn9dzIyEi1b9/esQzn8umnn2rmzJl6/vnnNXbsWM96YmKi9u/fr5qaGs/amexn9i/0tbkU7Nq1SyNGjPC6xUhw8Hf/l9S9e/d62XJzc72yfz9bbW2t9u7dqx49eqht27aKiory2v/iiy/07bffKjEx8WJGarCNGzfq%2BPHjGjJkiJKSkjRu3DhJUlJSkjp37tyks0vS3r17tWjRIq%2B1goICNWvWTIMGDWrS%2Ba%2B66ipFREQoPz/fs1ZYWKjLLrusyWc/Iz8/X4WFherfv79nrVu3bk0vu6OX1MOKCRMmmNmzZ5uysjJz4MABM3ToULN27Vpfj9VoZ/spwv3795tu3bqZ7du3m6qqKrNu3TrTq1cvU1xcbIz57icoBw8e7LkVwdy5c8348eM9z8/MzDRTp041x48fN0VFRWb8%2BPFm0aJFjub6KadPnzYjR440r7/%2Ber296upqM2TIEPPCCy%2BYU6dOmT179pjrr7/ebN%2B%2B3RjT%2BNfG10pLS01ycrJZtGiROXXqlDl%2B/LiZMmWK%2BcUvfmG%2B%2BeYb06tXL/PGG2%2BYqqoqs2PHDtO9e3fPT4Tu3LnT9OnTx%2BzevducOnXKLFu2zAwaNMhUVlYaY777acs777zT/Otf/zInTpww06dPNw899JAv43pxu92mqKjI89/u3btN586dTVFRkSksLGzS2Y0x5tixY6Znz55m5cqVprq62hw8eNDcdtttZv78%2BU3%2Bc2%2BMMQsWLDDDhg0zhw8fNt98841JTU01s2bNCojsxhizfv16069fP6%2B1ppidguWHioqKzNSpU0337t1NcnKyeeGFFy6ZextdiMTERJOYmGiuu%2B46c91113n%2BfMbWrVvN8OHDTUJCgrnjjjvMJ5984tmrq6szzz//vLnxxhtN9%2B7dzX333ee5D5Qx3/0lnpWVZXr27Gn69u1rnnrqKVNdXe1ovp/yj3/8w3Tu3NmT%2Bfv/ffXVV2b//v1m0qRJJjEx0QwePNi89tprXs9vzGtzKdi3b5%2B56667TPfu3c0NN9xgMjMzPT9W/cknn5gxY8aYhIQEM3z48Ho/1v7aa6%2BZQYMGmcTERJOWlmb279/v2auurjbz5s0zffv2Nb169TIPP/ywKS0tdTTb%2BTh69KjnNg3GBEb2Tz75xKSmppqePXuafv36mYULF5qqqirPXlPO//0Ze/bsaWbOnGnKy8uNMU0/uzHGvPTSS2bUqFH11pta9iBjLrGrwgAAAPwc12ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGX/D7K3fEdqz9Z5AAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common8647624807666160178">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">150</td>
        <td class="number">88</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">120</td>
        <td class="number">56</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">390</td>
        <td class="number">47</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">618</td>
        <td class="number">45</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1110</td>
        <td class="number">43</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">700</td>
        <td class="number">41</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">108</td>
        <td class="number">38</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1273</td>
        <td class="number">37</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">900</td>
        <td class="number">37</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">212</td>
        <td class="number">37</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (3240)</td>
        <td class="number">14651</td>
        <td class="number">96.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme8647624807666160178">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:20%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30</td>
        <td class="number">15</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">42</td>
        <td class="number">5</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:34%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">60</td>
        <td class="number">11</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:73%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">67</td>
        <td class="number">13</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:86%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">6679</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6766</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6811</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6836</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6890</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Id">Id<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>15120</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>100.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>7560.5</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>15120</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-9120466649818541435">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAAPpJREFUeJzt1TENQgEQREE%2BQRIi8ESNJ0Tg6agh5LVHMaNgm5c9ZmZOwE/n7QHwzy7bA75d78/tCSx6PW7bEz54EAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBcMzMbI%2BAf%2BVBIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAILwB%2BwILj9G2cTUAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-9120466649818541435,#minihistogram-9120466649818541435"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-9120466649818541435">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-9120466649818541435"
                                                  aria-controls="quantiles-9120466649818541435" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-9120466649818541435" aria-controls="histogram-9120466649818541435"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-9120466649818541435" aria-controls="common-9120466649818541435"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-9120466649818541435" aria-controls="extreme-9120466649818541435"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-9120466649818541435">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>756.95</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>3780.8</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>7560.5</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>11340</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>14364</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>15120</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>15119</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>7559.5</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>4364.9</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.57733</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.2</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>7560.5</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>3780</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>114314760</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>19052000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-9120466649818541435">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xt0VOW9//FPkpFQCLmhCT2UCpWLuXAX0gbkEhQsCEJEQ1rLoQWFEskvAboAkYrKASqhVBMPNbZaUaopaBVSEYpcigfaaos6CWgF9Sg0mDFkIAkEcnl%2Bf7iY4xgQJA97zPB%2BrZW1zPPdl%2BfLODOf7L1nT4gxxggAAADWhAZ6AgAAAMGGgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALHMFegKXC4%2Bnyvo2Q0NDFBvbVkeP1qix0Vjf/tfF5dDn5dCjRJ/B5HLoUaLPYHDVVe0Csl%2BOYLVgoaEhCgkJUWhoSKCnckldDn1eDj1K9BlMLoceJfrExSNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlrkBPAM1z3cJXAj0FAACabVPOoEBPwSqOYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWNaiA9auXbuUmpqq3Nzccy5TU1OjYcOGaf78%2Bb6xxsZGrVq1SiNGjNCAAQM0depUffzxx7661%2BtVTk6OUlNTNXjwYC1cuFC1tbWXtBcAABA8WmzAevzxx7VkyRJdffXVX7pcfn6%2Bqqur/cbWrl2rjRs3qrCwUNu3b1fnzp2VlZUlY4wkadGiRTp58qSKi4v1/PPP6%2BDBg8rLy7tkvQAAgODSYgNWeHi41q9f/6UB65133lFxcbEmTJjgN15UVKQpU6bommuuUUREhHJzc3Xw4EG99dZb%2BvTTT7V161bl5uYqNjZW8fHxmjlzpp5//nnV1dVd6rYAAEAQaLEBa/LkyWrXrt0568YYLV68WLm5uYqMjPSN19bW6sCBA0pMTPSNRURE6Oqrr5bb7db%2B/fsVFhamHj16%2BOpJSUk6ceKE3n///UvTDAAACCquQE/gUikqKlJISIjS09NVUFDgGz927JiMMYqKivJbPioqSpWVlYqOjlZERIRCQkL8apJUWVl5QfsuLy%2BXx%2BPxG3O52iguLu5i2zmrsLAWm48BAPDjcgXXe1pQBqyKigo9/PDD%2Bt3vfucXlD7vzPVWX7V2IYqKivxCnSRlZWUpOzu7WdsFACBYxcS0DfQUrArKgLV8%2BXKNHz/e7zTfGdHR0QoNDZXX6/Ub93q9at%2B%2BvWJjY1VdXa2GhgaFhYX5apLUvn37C9p/RkaG0tLS/MZcrjaqrKy5mHbOiSNYAIBgYfs98oxABbegDFgbNmxQZGSkXnjhBUmfXXfV2Nio7du3629/%2B5u6deum0tJSDRw4UJJ0/PhxffTRR%2BrVq5c6duwoY4zeeecdJSUlSZLcbrciIyPVpUuXC9p/XFxck9OBHk%2BV6usbLXYJAEDwCLb3yKAMWDt37vT7/cknn9SRI0e0YMECSVJmZqYKCws1ZMgQxcfHKy8vTwkJCerZs6ckadSoUfrVr36lX/ziFzp9%2BrQeffRRTZw4US5XUP5zAQAAy1psYjgThurr6yVJW7dulfTZ0aYOHTr4LRsREaFvfOMbvvFJkybJ4/HoRz/6kWpqapSSkuJ3zdQDDzyg%2B%2B67TyNGjNAVV1yhm2%2B%2B%2BUtvZgoAAPB5Iaa5V3Tjgng8Vda36XKF6sa8Xda3CwCA0zblDLok273qqnPf0ulS4ippAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYFmLDVi7du1SamqqcnNzm9S2bNmicePGqW/fvho1apT%2B8Ic/%2BNXXrFmjUaNGqV%2B/fsrMzFRJSYmvdurUKf385z/XkCFDlJKSouzsbFVWVl7yfgAAQPBokQHr8ccf15IlS3T11Vc3qb399tuaO3eusrOz9frrr%2Buee%2B7RAw88oDfeeEOStG3bNuXn5%2Buhhx7S7t27NXz4cM2YMUMnTpyQJK1atUqlpaUqKirS5s2bZYzRggULHO0PAAC0bC0yYIWHh2v9%2BvVnDVher1fTp0/XDTfcIJfLpaFDh6p79%2B6%2BgFVUVKT09HT17t1brVu31rRp0yRJ27dvV319vdavX6%2BZM2fqm9/8pqKjo5WTk6MdO3bok08%2BcbRHAADQcrkCPYGLMXny5HPWhgwZoiFDhvh%2Br6%2Bvl8fjUXx8vCSptLRUo0eP9tVDQ0OVkJAgt9uthIQEVVVVKSkpyVe/5ppr1Lp1a5WWlvq2cT7l5eXyeDx%2BYy5XG8XFxV3Q%2BhcqLKxF5mMAAJpwuYLrPa1FBqyvIi8vT23atPGFKq/Xq6ioKL9loqKiVFlZKa/XK0mKjIz0q0dGRn6l67CKiopUUFDgN5aVlaXs7OyLaQEAgKAXE9M20FOwKmgDljFGeXl5Ki4u1po1axQeHu5XO9%2B6zZGRkaG0tDS/MZerjSora5q13S/iCBYAIFjYfo88I1DBLSgDVmNjoxYsWKC3335bzz77rDp16uSrxcTE%2BI5UneH1etWtWzfFxsb6fm/b9v8ekGPHjql9%2B/YXvP%2B4uLgmpwM9nirV1zdeTDsAAAS9YHuPDMpDIEuXLtV7773XJFxJUnJyskpLS32/NzQ0aN%2B%2Bferdu7c6deqkqKgov/q//vUvnT59WsnJyY7NHwAAtGxBF7D%2B8Y9/aMOGDSosLFR0dHSTemZmpl588UW9%2BeabOnnypFavXq1WrVpp2LBhCgsL0%2B23365f//rXKisrU2VlpX75y1/qxhtv1JVXXhmAbgAAQEvUIk8R9uzZU9JnnxCUpK1bt0qS3G63nn/%2BeVVVVWn48OF%2B6wwYMEBPPPGEhgwZotmzZysnJ0cVFRXq2bOnCgsL1bp1a0lSdna2ampqdMstt6i%2Bvl7Dhw/X4sWLnWsOAAC0eCGmuVd044J4PFXWt%2BlyherGvF3WtwsAgNM25Qy6JNu96qp2l2S75xN0pwgBAAACjYAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAy1p0wNq1a5dSU1OVm5vbpPbyyy9r7Nix6tu3r9LT0/Xaa6/5ao2NjVq1apVGjBihAQMGaOrUqfr44499da/Xq5ycHKWmpmrw4MFauHChamtrHekJAAC0fC02YD3%2B%2BONasmSJrr766ia1/fv3a968eZo7d67%2B%2Bte/asqUKbr77rt15MgRSdLatWu1ceNGFRYWavv27ercubOysrJkjJEkLVq0SCdPnlRxcbGef/55HTx4UHl5eY72BwAAWq4WG7DCw8O1fv36swasdevWaejQoRo6dKjCw8M1btw4de/eXRs2bJAkFRUVacqUKbrmmmsUERGh3NxcHTx4UG%2B99ZY%2B/fRTbd26Vbm5uYqNjVV8fLxmzpyp559/XnV1dU63CQAAWiBXoCdwsSZPnnzOWmlpqYYOHeo3lpiYKLfbrdraWh04cECJiYm%2BWkREhK6%2B%2Bmq53W5VVVUpLCxMPXr08NWTkpJ04sQJvf/%2B%2B37j51JeXi6Px%2BM35nK1UVxc3IW2d0HCwlpsPgYAwI/LFVzvaS02YH0Zr9erqKgov7GoqCgdOHBAx44dkzHmrPXKykpFR0crIiJCISEhfjVJqqysvKD9FxUVqaCgwG8sKytL2dnZF9MOAABBLyambaCnYFVQBixJvuupLqZ%2BvnXPJyMjQ2lpaX5jLlcbVVbWNGu7X8QRLABAsLD9HnlGoIJbUAasmJgYeb1evzGv16vY2FhFR0crNDT0rPX27dsrNjZW1dXVamhoUFhYmK8mSe3bt7%2Bg/cfFxTU5HejxVKm%2BvvFiWwIAIKgF23tkUB4CSU5OVklJid%2BY2%2B1W7969FR4erm7duqm0tNRXO378uD766CP16tVLCQkJMsbonXfe8Vs3MjJSXbp0cawHAADQcgVlwLr99tu1e/du7dixQ6dOndL69ev14Ycfaty4cZKkzMxMrVmzRgcPHlR1dbXy8vKUkJCgnj17KjY2VqNGjdKvfvUrHT16VEeOHNGjjz6qiRMnyuUKygN%2BAADAshabGHr27ClJqq%2BvlyRt3bpV0mdHm7p37668vDwtW7ZMhw8fVteuXfXYY4/pqquukiRNmjRJHo9HP/rRj1RTU6OUlBS/i9IfeOAB3XfffRoxYoSuuOIK3XzzzWe9mSkAAMDZhJjmXtGNC%2BLxVFnfpssVqhvzdlnfLgAATtuUM%2BiSbPeqq9pdku2eT1CeIgQAAAgkAhYAAIBljgestLQ0FRQUqKyszOldAwAAOMLxgHXrrbfq5Zdf1g033KBp06Zpy5YtvgvVAQAAgkHALnIvLS1VcXGxNm3apLq6Oo0fP14TJ04M2ntNcZE7AADnFmwXuQf8U4TGGL388stavHixqqurlZqaqv/3//6fevXqFchpWUfAAgDg3IItYAXsIve6ujq9/PLLuvPOOzVv3jzFx8drwYIFSkhI0JQpU7Rx48ZATQ0AAKBZHL/R6MGDB7V%2B/Xq9%2BOKLqqmp0ahRo/TUU0%2Bpf//%2BvmUGDBigxYsXa%2BzYsU5PDwAAoNkcD1hjxoxRly5dNH36dI0fP17R0dFNlhk6dKiOHj3q9NQAAACscDxgrVmzRgMHDjzvcm%2B99ZYDswEAALDP8WuwevTooRkzZvi%2BO1CSfve73%2BnOO%2B%2BU1%2Bt1ejoAAADWOR6wli1bpqqqKnXt2tU3NmzYMDU2Nmr58uVOTwcAAMA6x08Rvvbaa9q4caNiYmJ8Y507d1ZeXp5uvvlmp6cDAABgneNHsGpraxUeHt50IqGhOnnypNPTAQAAsM7xgDVgwAAtX75cx44d84198sknuv/%2B%2B/1u1QAAANBSOX6K8J577tFPfvITfe9731NERIQaGxtVU1OjTp066emnn3Z6OgAAANY5HrA6deqkP/3pT/rLX/6ijz76SKGhoerSpYsGDx6ssLAwp6cDAABgneMBS5JatWqlG264IRC7BgAAuOQcD1gff/yxVq5cqffee0%2B1tbVN6q%2B%2B%2BqrTUwIAALAqINdglZeXa/DgwWrTpo3TuwcAALjkHA9YJSUlevXVVxUbG%2Bv0rgEAABzh%2BG0a2rdvz5ErAAAQ1BwPWNOnT1dBQYGMMU7vGgAAwBGOnyL8y1/%2Bon/%2B85964YUX9K1vfUuhof4Z77nnnnN6SgAAAFY5HrAiIiI0ZMgQp3cLAADgGMcD1rJly5zeJQAAgKMcvwZLkt5//33l5%2BdrwYIFvrG9e/cGYioAAADWOR6w9uzZo3HjxmnLli0qLi6W9NnNRydPnsxNRgEAQFBwPGCtWrVKP/vZz7Rx40aFhIRI%2Buz7CZcvX65HH33U6ekAAABY53jA%2Bte//qXMzExJ8gUsSbrpppt08OBBa/vZt2%2BfJk%2BerOuuu06DBg3S3LlzdfToUUmfHUWbOHGi%2BvXrpzFjxmjDhg1%2B665Zs0ajRo1Sv379lJmZqZKSEmvzAgAAwc/xgNWuXbuzfgdheXm5WrVqZWUf9fX1uuuuu9SnTx/t3r1bxcXFOnr0qBYvXqzy8nLNnDlTkyZN0p49e7Rw4UItWrRIbrdbkrRt2zbl5%2BfroYce0u7duzV8%2BHDNmDFDJ06csDI3AAAQ/BwPWP369dPSpUtVXV3tG/vggw80b948fe9737OyD4/HI4/Ho1tuuUWtWrVSTEyMbrzxRu3fv18bN25U586dNXHiRIWHhys1NVVpaWlat26dJKmoqEjp6enq3bu3WrdurWnTpkmStm/fbmVuAAAg%2BDkesBYsWKC9e/cqJSVFp06dUr9%2B/TR69Gh5vV7Nnz/fyj7i4%2BOVkJCgoqIi1dTUqKKiQlu2bNGwYcNUWlqqxMREv%2BUTExN9pwG/WA8NDVVCQoLvCBcAAMD5OH4frA4dOqi4uFg7d%2B7UBx98oNatW6tLly4aNGiQ3zVZzREaGqr8/HxNmTJFTz31lCRp4MCBmjNnjmbOnKn4%2BHi/5aOjo1VZWSlJ8nq9ioqK8qtHRUX56heivLxcHo/Hb8zlaqO4uLiLaeecwsICcpcNAACsc7mC6z3N8YAlSVdccYVuuOGGS7b906dPa8aMGbrpppt810/df//9mjt37gWt39zvSSwqKlJBQYHfWFZWlrKzs5u1XQAAglVMTNtAT8EqxwNWWlralx6psnEvrD179ujQoUOaPXu2wsLC1K5dO2VnZ%2BuWW27R9ddfL6/X67d8ZWWlYmNjJUkxMTFN6l6vV926dbvg/WdkZCgtLc1vzOVqo8rKmovs6Ow4ggUACBa23yPPCFRwczxgjR492i9gNTQ06IMPPpDb7dZ//ud/WtlHQ0ODGhsb/Y5EnT59WpKUmpqqP/7xj37Ll5SUqHfv3pKk5ORklZaWasKECb5t7du3TxMnTrzg/cfFxTU5HejxVKm%2BvvGi%2BgEAINgF23uk4wHrXKfpNm/erL/97W9W9tG3b1%2B1adNG%2Bfn5mjFjhmpra7V69WoNGDBAt9xyiwoKCrRu3TqNGzdOf/3rX7Vz504VFRVJkjIzMzV79mzdfPPN6tGjh37729%2BqVatWGjZsmJW5AQCA4BdimnvBkSUNDQ1KTU21FrJKSkr0i1/8Qu%2B8845atWqlgQMHav78%2BYqPj9frr7%2BuJUuW6ODBg%2BrYsaPmzJmjkSNH%2Btb9/e9/r8LCQlVUVKhnz55avHixunfv3qz5eDxVzW2pCZcrVDfm7bK%2BXQAAnLYpZ9Al2e5VV7W7JNs9n69NwHK73Zo6dar%2B/ve/B3oqlwQBCwCAcwu2gOX4KcJJkyY1GTt58qQOHjzodxQJAACgpXI8YHXu3LnJpwjDw8M1ceJE3XbbbU5PBwAAwDrHA9by5cud3iUAAICjHA9YL7744gUvO378%2BEs4EwAAgEvD8YC1cOHCJveokqSQkBC/sZCQEAIWAABokRwPWL/5zW/0xBNPaMaMGerRo4eMMXr33Xf1%2BOOP64477lBKSorTUwIAALAqINdgFRYW%2Bn3h8nXXXadOnTpp6tSpKi4udnpKAAAAVjn%2BZXYffvihoqKimoxHRkbq8OHDTk8HAADAOscDVseOHbV8%2BXJVVlb6xo4fP66VK1fq29/%2BttPTAQAAsM7xU4T33HOP5syZo6KiIrVt21ahoaGqrq5W69at9eijjzo9HQAAAOscD1iDBw/Wjh07tHPnTh05ckTGGMXHx%2Bv6669Xu3aBuZ09AACATY4HLEn6xje%2BoREjRujIkSPq1KlTIKYAAABwyTh%2BDVZtba3mzZunvn376vvf/76kz67BmjZtmo4fP%2B70dAAAAKxzPGCtWLFC%2B/fvV15enkJD/2/3DQ0NysvLc3o6AAAA1jkesDZv3qxHHnlEN910k%2B9LnyMjI7Vs2TJt2bLF6ekAAABY53jAqqmpUefOnZuMx8bG6sSJE05PBwAAwDrHA9a3v/1t/e1vf5Mkv%2B8efOWVV/Qf//EfTk8HAADAOsc/RfiDH/xAs2bN0q233qrGxkY9%2BeSTKikp0ebNm7Vw4UKnpwMAAGCd4wErIyNDLpdLzzzzjMLCwvTrX/9aXbp0UV5enm666SanpwMAAGCd4wHr6NGjuvXWW3Xrrbc6vWsAAABHOH4N1ogRI/yuvQIAAAg2jgeslJQUbdq0yendAgAAOMbxU4Tf/OY39V//9V8qLCzUt7/9bV1xxRV%2B9ZUrVzo9JQAAAKscD1gHDhzQd77zHUlSZWWl07sHAAC45BwLWLm5uVq1apWefvpp39ijjz6qrKwsp6YAAADgCMeuwdq2bVuTscLCQqd2DwAA4BjHAtbZPjnIpwkBAEAwcixgnfli5/ONAQAAtHSO36YBAAAg2BGwAAAALHPsU4R1dXWaM2fOecds3gdr9erVWrt2raqrq9WnTx8tWbJE3/rWt7Rnzx6tXLlS77//vr75zW9q%2BvTpGjdunG%2B9NWvWaO3atfJ4POrRo4cWLlyo5ORka/MCAADBzbEjWP3791d5ebnfz9nGbFm7dq02bNigNWvW6LXXXlPXrl31u9/9TuXl5Zo5c6YmTZqkPXv2aOHChVq0aJHcbrekzz7tmJ%2Bfr4ceeki7d%2B/W8OHDNWPGDJ04ccLa3AAAQHALMUH6Ub4RI0Zo3rx5GjlypN/4b3/7WxUXF%2BuPf/yjbyw3N1ft2rXTAw88oOnTp6tz585asGCBJKmxsVFDhgzRggULNGbMmIuej8dTddHrnovLFaob83ZZ3y4AAE7blDPokmz3qqvaXZLtno/jd3J3wieffKJDhw7p2LFjGj16tCoqKpSSkqLFixertLRUiYmJfssnJib6vh%2BxtLRUo0eP9tVCQ0OVkJAgt9t9wQGrvLxcHo/Hb8zlaqO4uLhmduYvLIxL6AAAwcHlCq73tKAMWEeOHJEkvfLKK3ryySdljFF2drbuvfde1dbWKj4%2B3m/56Oho39f2eL1eRUVF%2BdWjoqK%2B0tf6FBUVqaCgwG8sKytL2dnZF9MOAABBLyambaCnYFVQBqwzZz2nTZvmC1OzZs3SnXfeqdTU1Ate/2JlZGQoLS3Nb8zlaqPKyppmbfeLOIIFAAgWtt8jzwhUcAvKgHXllVdKkiIjI31jHTt2lDFGdXV18nq9fstXVlYqNjZWkhQTE9Ok7vV61a1btwvef1xcXJPTgR5PlerrG79SHwAAXC6C7T0yKA%2BBdOjQQREREdq/f79v7PDhw7riiis0dOhQlZSU%2BC1fUlKi3r17S5KSk5NVWlrqqzU0NGjfvn2%2BOgAAwPkEZcByuVyaOHGifv3rX%2Bt///d/VVFRoUcffVRjx47VhAkTdPjwYa1bt06nTp3Szp07tXPnTt1%2B%2B%2B2SpMzMTL344ot68803dfLkSa1evVqtWrXSsGHDAtsUAABoMYLyFKEkzZkzR6dPn9Ztt92muro6jRo1Svfee6/atm2rxx57TEuWLNH999%2Bvjh07asWKFbr22mslSUOGDNHs2bOVk5OjiooK9ezZU4WFhWrdunWAOwIAAC1F0N4H6%2BuG%2B2ABAHBuwXYfrKA8RQgAABBIBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYdlkErKVLl6pHjx6%2B3/fs2aOJEyeqX79%2BGjNmjDZs2OC3/Jo1azRq1Cj169dPmZmZKikpcXrKAACgBQv6gLV//3699NJLvt/Ly8s1c%2BZMTZo0SXv27NHChQu1aNEiud1uSdK2bduUn5%2Bvhx56SLt379bw4cM1Y8YMnThxIlAtAACAFiaoA1ZjY6Puu%2B8%2BTZkyxTe2ceNGde7cWRMnTlR4eLhSU1OVlpamdevWSZKKioqUnp6u3r17q3Xr1po2bZokafv27YFoAQAAtECuQE/gUnruuecUHh6usWPH6le/%2BpUkqbS0VImJiX7LJSYmatOmTb766NGjfbXQ0FAlJCTI7XZrzJgxF7Tf8vJyeTwevzGXq43i4uKa004TYWFBnY8BAJcRlyu43tOCNmB9%2Bumnys/P19NPP%2B037vV6FR8f7zcWHR2tyspKXz0qKsqvHhUV5atfiKKiIhUUFPiNZWVlKTs7%2B6u0AADAZSMmpm2gp2BV0AasZcuWKT09XV27dtWhQ4e%2B0rrGmGbtOyMjQ2lpaX5jLlcbVVbWNGu7X8QRLABAsLD9HnlGoIJbUAasPXv2aO/evSouLm5Si4mJkdfr9RurrKxUbGzsOeter1fdunW74P3HxcU1OR3o8VSpvr7xgrcBAMDlJNjeI4PyEMiGDRtUUVGh4cOHKyUlRenp6ZKklJQUde/evcltF0pKStS7d29JUnJyskpLS321hoYG7du3z1cHAAA4n6AMWPPnz9fmzZv10ksv6aWXXlJhYaEk6aWXXtLYsWN1%2BPBhrVu3TqdOndLOnTu1c%2BdO3X777ZKkzMxMvfjii3rzzTd18uRJrV69Wq1atdKwYcMC2BEAAGhJgvIUYVRUlN%2BF6vX19ZKkDh06SJIee%2BwxLVmyRPfff786duyoFStW6Nprr5UkDRkyRLNnz1ZOTo4qKirUs2dPFRYWqnXr1s43AgAAWqQQ09wrunFBPJ4q69t0uUJ1Y94u69sFAMBpm3IGXZLtXnVVu0uy3fMJylOEAAAAgUTAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAA%2BcfpHAAAStUlEQVQAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGVBG7AOHz6srKwspaSkKDU1VfPnz9fx48clSfv379cdd9yh/v37a%2BTIkXriiSf81n355Zc1duxY9e3bV%2Bnp6XrttdcC0QIAAGihgjZgzZgxQ5GRkdq2bZteeOEFvffee/rFL36h2tpaTZ8%2BXd/97ne1a9curVq1So899pi2bNki6bPwNW/ePM2dO1d//etfNWXKFN199906cuRIgDsCAAAtRVAGrOPHjys5OVlz5sxR27Zt1aFDB02YMEFvvPGGduzYobq6Ov30pz9VmzZtlJSUpNtuu01FRUWSpHXr1mno0KEaOnSowsPDNW7cOHXv3l0bNmwIcFcAAKClCMqAFRkZqWXLlunKK6/0jZWVlSkuLk6lpaXq0aOHwsLCfLXExESVlJRIkkpLS5WYmOi3vcTERLndbmcmDwAAWjxXoCfgBLfbrWeeeUarV6/Wpk2bFBkZ6VePjo6W1%2BtVY2OjvF6voqKi/OpRUVE6cODABe%2BvvLxcHo/Hb8zlaqO4uLiLb%2BIswsKCMh8DAC5DLldwvacFfcD6xz/%2BoZ/%2B9KeaM2eOUlNTtWnTprMuFxIS4vtvY0yz9llUVKSCggK/saysLGVnZzdruwAABKuYmLaBnoJVQR2wtm3bpp/97GdatGiRxo8fL0mKjY3Vhx9%2B6Lec1%2BtVdHS0QkNDFRMTI6/X26QeGxt7wfvNyMhQWlqa35jL1UaVlTUX18g5cAQLABAsbL9HnhGo4Ba0Aeuf//yn5s2bp4cffliDBw/2jScnJ%2BvZZ59VfX29XK7P2ne73erdu7evfuZ6rDPcbrfGjBlzwfuOi4trcjrQ46lSfX3jxbYDAEBQC7b3yKA8BFJfX697771Xc%2BfO9QtXkjR06FBFRERo9erVOnnypN566y2tX79emZmZkqTbb79du3fv1o4dO3Tq1CmtX79eH374ocaNGxeIVgAAQAsUYpp7wdHX0BtvvKEf/vCHatWqVZPaK6%2B8opqaGt13330qKSnRlVdeqTvvvFM/%2BMEPfMts2bJFK1eu1OHDh9W1a1ctXLhQAwYMaNacPJ6qZq1/Ni5XqG7M22V9uwAAOG1TzqBLst2rrmp3SbZ7PkEZsL6OCFgAAJxbsAWsoDxFCAAAEEgELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgHUWhw8f1l133aWUlBQNHz5cK1asUGNjY6CnBQAAWghXoCfwdTRr1iwlJSVp69atqqio0PTp03XllVfqxz/%2BcaCnBgAAWgCOYH2B2%2B3WO%2B%2B8o7lz56pdu3bq3LmzpkyZoqKiokBPDQAAtBAcwfqC0tJSdezYUVFRUb6xpKQkffDBB6qurlZERMR5t1FeXi6Px%2BM35nK1UVxcnNW5hoWRjwEAwcHlCq73NALWF3i9XkVGRvqNnQlblZWVFxSwioqKVFBQ4Dd29913a9asWfYmqs%2BC3H92eE8ZGRnWw9vXSXl5uYqKioK6z8uhR4k%2Bg8nl0KNEn7h4wRUXLTHGNGv9jIwMvfDCC34/GRkZlmb3fzwejwoKCpocLQs2l0Ofl0OPEn0Gk8uhR4k%2BcfE4gvUFsbGx8nq9fmNer1chISGKjY29oG3ExcXxFwAAAJcxjmB9QXJyssrKynT06FHfmNvtVteuXdW2bdsAzgwAALQUBKwvSExMVM%2BePbVy5UpVV1fr4MGDevLJJ5WZmRnoqQEAgBYibPHixYsDPYmvm%2Buvv17FxcV68MEH9ac//UkTJ07U1KlTFRISEuipNdG2bVsNHDgw6I%2BuXQ59Xg49SvQZTC6HHiX6xMUJMc29ohsAAAB%2BOEUIAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBqwU6fPiw7rrrLqWkpGj48OFasWKFGhsbAz2tC3L48GFlZWUpJSVFqampmj9/vo4fPy5J2r9/v%2B644w71799fI0eO1BNPPOG37ssvv6yxY8eqb9%2B%2BSk9P12uvvearNTY2atWqVRoxYoQGDBigqVOn6uOPP3a0t7NZunSpevTo4ft9z549mjhxovr166cxY8Zow4YNfsuvWbNGo0aNUr9%2B/ZSZmamSkhJf7dSpU/r5z3%2BuIUOGKCUlRdnZ2aqsrHSsl7NZvXq1Bg8erD59%2BmjKlCk6dOiQpODqc9%2B%2BfZo8ebKuu%2B46DRo0SHPnzvV9GXxL7nPXrl1KTU1Vbm5uk1pznmter1c5OTlKTU3V4MGDtXDhQtXW1vrq53ueO9nnli1bNG7cOPXt21ejRo3SH/7wB796cx4/p1%2Bnv6zPM2pqajRs2DDNnz/fN9bSHs8WxaDFmTBhgrn33nvN8ePHzQcffGBGjhxpnnjiiUBP64LcfPPNZv78%2Baa6utqUlZWZ9PR0c88995iTJ0%2Ba66%2B/3uTn55uamhpTUlJiBg4caDZv3myMMWbfvn0mOTnZ7Nixw9TW1pqXXnrJ9O7d25SVlRljjFmzZo0ZPny4OXDggKmqqjIPPPCAGTt2rGlsbAxYr/v27TMDBw403bt3N8YY88knn5g%2BffqYdevWmdraWvM///M/plevXubtt982xhjz6quvmuuuu868%2Beab5uTJk%2Baxxx4zgwYNMjU1NcYYY5YtW2bS09PNv//9b1NZWWnuvvtuM3369ID198wzz5ibbrrJHDx40FRVVZkHH3zQPPjgg0HVZ11dnRk0aJBZuXKlOXXqlDl69Kj58Y9/bGbNmtWi%2BywsLDQjR440kyZNMjk5OX615j7X7r77bnPXXXeZiooKc%2BTIEZORkWEefPBBY4w57/PcyT7feust07NnT/PnP//Z1NXVmR07dpikpCTz%2BuuvG2Oa//g5%2BTr9ZX1%2B3rJly0z//v3NvHnzfGMt6fFsaQhYLczbb79tEhISjNfr9Y39/ve/N6NGjQrgrC7MsWPHzPz5843H4/GNPf3002bkyJFm06ZN5rvf/a6pr6/31VasWGF%2B8pOfGGOMuf/%2B%2B01WVpbf9m677Tbz2GOPGWOMGTNmjHnqqad8taqqKpOYmGj27t17KVs6p4aGBnPbbbeZ//7v//YFrN/85jdm/Pjxfsvl5OSYRYsWGWOMueuuu8zSpUv9tjFo0CBTXFxs6urqTP/%2B/c3WrVt99QMHDpgePXqYI0eOONBRU2lpaWd9IQ2mPv/973%2Bb7t27mwMHDvjGfv/735sbbrihRff51FNPmePHj5t58%2BY1eUNuznPN4/GYa6%2B91uzfv99X37lzp%2BnTp485ffr0eZ/ntn1Znzt37jQFBQV%2BYxMmTDCrV682xjTv8XP6dfrL%2Bjxj//79ZtCgQWbJkiV%2BAaslPZ4tDacIW5jS0lJ17NhRUVFRvrGkpCR98MEHqq6uDuDMzi8yMlLLli3TlVde6RsrKytTXFycSktL1aNHD4WFhflqiYmJvkPypaWlSkxM9NteYmKi3G63amtrdeDAAb96RESErr76arnd7kvc1dk999xzCg8P19ixY31j5%2BrhXD2GhoYqISFBbrdbH330kaqqqpSUlOSrX3PNNWrdurVKS0svcTdNffLJJzp06JCOHTum0aNH%2B06RHD16NKj6jI%2BPV0JCgoqKilRTU6OKigpt2bJFw4YNa9F9Tp48We3atTtrrTnPtf379yssLMzvtHhSUpJOnDih999//7zPc9u%2BrM8hQ4YoKyvL93t9fb08Ho/i4%2BMlNe/xc/p1%2Bsv6lCRjjBYvXqzc3FxFRkb6xlva49nSELBaGK/X6/cEkeR7Egf6epyvyu1265lnntFPf/rTs/YVHR0tr9erxsZGeb1evxcr6bO%2BKysrdezYMRljzll32qeffqr8/Hzdd999fuPn6vHMHL%2BsR6/XK0lN1o%2BMjAxIj0eOHJEkvfLKK3ryySf10ksv6ciRI7r33nuDqs/Q0FDl5%2Bfr1VdfVb9%2B/ZSamqr6%2BnrNmTMnqPr8vOY817xeryIiIhQSEuJXk%2BSrf9nzPJDy8vLUpk0bjR49WlLzHr%2Bv2%2Bt0UVGRQkJClJ6e7jcezI/n1wEBqwUyxgR6Cs32j3/8Q1OnTtWcOXOUmpp6zuU%2B/8Q%2BX99fl3%2BXZcuWKT09XV27dv3K67aUHs/MY9q0aYqPj1eHDh00a9Ysbdu27Sutf7F1p5w%2BfVozZszQTTfdpDfeeEN/%2Bctf1K5dO82dO/eC1m8pfX5Rc%2BZ9MT19/nnuNGOMVqxYoeLiYq1evVrh4eF%2BtfOtezE1J1VUVOjhhx/W4sWLz/nvHEyP59cJAauFiY2N9f31dIbX61VISIhiY2MDNKuvZtu2bbrrrrt0zz33aPLkyZI%2B6%2BuLf9l5vV5FR0crNDRUMTExZ%2B07NjbWt8zZ6u3bt7%2B0zXzBnj17tHfvXr9TD2ecrYfKykrf4/ZlPZ5Z5ov1Y8eOOd6jJN9p3s//9dqxY0cZY1RXVxc0fe7Zs0eHDh3S7Nmz1a5dO8XHxys7O1t//vOfz/r/XEvt8/Oa81yLjY1VdXW1Ghoa/GqSfPUve547rbGxUfPnz9e2bdv07LPP6jvf%2BY6v1pzH7%2Bv0Or18%2BXKNHz/e7zTfGcH2eH7d8C/QwiQnJ6usrMz3MXHps1NtXbt2Vdu2bQM4swvzz3/%2BU/PmzdPDDz%2Bs8ePH%2B8aTk5P17rvvqr6%2B3jfmdrvVu3dvX/2L5/XP1MPDw9WtWze/a1eOHz%2Bujz76SL169brEHfnbsGGDKioqNHz4cKWkpPgOyaekpKh79%2B5NeigpKfHr8fM9NDQ0aN%2B%2Bferdu7c6deqkqKgov/q//vUvnT59WsnJyQ505q9Dhw6KiIjQ/v37fWOHDx/WFVdcoaFDhwZNnw0NDWpsbPT7K/706dOSpNTU1KDp8/Oa81xLSEiQMUbvvPOO37qRkZHq0qXLeZ/nTlu6dKnee%2B89Pfvss%2BrUqZNfrTmP39fpdXrDhg1av369UlJSlJKSot/85jf605/%2BpJSUlKB7PL92HLqYHhbddttt5p577jFVVVXmwIEDJi0tzTzzzDOBntZ51dXVme9///vmueeea1I7deqUGT58uHnkkUfMiRMnzJtvvmmuu%2B46s337dmOMMe%2B%2B%2B67p2bOn2b59u6mtrTXr1q0zffv2NeXl5caYzz6hM2zYMN9HjRctWmRuvfVWJ9szxhjj9XpNWVmZ72fv3r2me/fupqyszBw%2BfNj07dvX/OEPfzC1tbVmx44dplevXr5P6OzcudP079/f7N2715w4ccLk5%2BeboUOHmpMnTxpjPvt0zoQJE8y///1vc/ToUTN9%2BnQza9Ysx3s8Y%2BnSpWbEiBHmww8/NJ9%2B%2BqnJyMgw8%2BfPN59%2B%2BmnQ9Hn06FEzcOBA88tf/tKcOHHCHD161MyYMcP88Ic/DIo%2Bz/aps%2BY%2B13Jycsy0adNMRUWFKSsrM7feeqtZvny5Meb8z3Mn%2B3zjjTfMgAED/D7V/HnNffwC8Tp9tj4//3pUVlZmli5darKzs3233WiJj2dLQcBqgcrKysy0adNMr169TGpqqnnkkUcCer%2BnC/X666%2Bb7t27m%2BTk5CY/hw4dMu%2B%2B%2B66ZNGmSSU5ONsOGDTNr1671W3/z5s1m5MiRJikpydxyyy3m73//u6/W2NhoHn74YfO9733P9OrVy9x5552%2BF5BA%2Bvjjj323aTDGmL///e9m3LhxJikpyYwcObLJbQ7Wrl1rhg4dapKTk01mZqZ59913fbVTp06ZxYsXmwEDBpi%2Bffua2bNnm%2BPHjzvWyxd9fj59%2BvQx8%2BbNM9XV1caY4OrT7XabO%2B64w1x33XUmNTXV5OTk%2BG6l0FL7PPO8u/baa821117r%2B/2M5jzXjh8/bnJzc02fPn3MgAEDzP33329OnTrlq5/vee5UnwsWLPAbO/Pz4x//2Ld%2Bcx4/J1%2Bnz/d4ft4jjzzid5uGlvR4tjQhxnxNrsQDAAAIElyDBQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACW/X8hjH1GxBkt3gAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-9120466649818541435">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2047</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6758</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">14978</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8833</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10880</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4727</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6774</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">629</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2676</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">12915</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (15110)</td>
        <td class="number">15110</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-9120466649818541435">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">15116</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15117</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15118</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15119</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15120</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Slope">Slope<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>52</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.3%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>16.502</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>52</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-8531695816899444010">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAATBJREFUeJzt3MFtg0AQQFFjuaQUkZ5yTk8uIj1tGoi%2BwBJhwe/dkfbyNWJWsIwxxg340/3oA8DMHkcf4CgfX8/Nz/x8f%2B5wEmZmgkAQCASBQBAIBIFAEAiES6x5X1nZwhomCIRLTJD/snVSuVg8PxMEgkAgCASCQCAIBIJAIAgEgkAgCASCm/Qd%2Baz3/EwQCAKBIBAIAoEgEAi2WJOx%2BZqLCQJBIBAEAkEgEAQCYbotln9cMRMTBIJAIAgEgkAgCASCQCBMt%2BZlOz/V3o8JAkEgEAQCQSAQBAJBIBCsed%2BQ797XEwirvGtUyxhjHH0ImJV3EAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAi/O%2BMgbc85rHYAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-8531695816899444010,#minihistogram-8531695816899444010"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-8531695816899444010">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-8531695816899444010"
                                                  aria-controls="quantiles-8531695816899444010" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-8531695816899444010" aria-controls="histogram-8531695816899444010"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-8531695816899444010" aria-controls="common-8531695816899444010"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-8531695816899444010" aria-controls="extreme-8531695816899444010"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-8531695816899444010">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>10</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>15</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>22</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>32</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>52</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>52</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>12</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>8.4539</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.51231</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.23831</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>16.502</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>6.9362</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.52366</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>249504</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>71.469</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-8531695816899444010">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3X1YlHW%2Bx/EPMIEJDE%2BK7SFXXJ9WBC1NKTLxITUzLQ1F2ko3a32gCNLN0iwrTVtxrSNeFrtbZ90626xmpSbqtj705O5mbTWMWom6tRyNWZ0JIXwA5vzR1ZwzQonyc4ah9%2Bu6uMjfb%2BY33/t73TN9uO%2BbmxCPx%2BMRAAAAjAkNdAEAAACtDQELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhmCXQBPxRO53Hja4aGhig%2BPlLHjlWrvt5jfP0fGvppFv00i36aQy/Naun9bN8%2BOiCvyxGsIBYaGqKQkBCFhoYEupRWgX6aRT/Nop/m0Euz6GfjCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw4I2YO3bt0%2BTJ09Wv379lJGRofz8fDmdTv3tb39Tjx49lJaW5vNVUlLife7q1as1cuRI9e3bVzk5OSotLfXOnTx5Ug8//LAGDRqk9PR05eXlyeVyBWITAQBAkArKgHXq1CndcccdGjBggHbt2qWNGzfq6NGjWrBggSQpKSlJdrvd52vUqFGSpG3btmnFihX61a9%2BpXfffVdDhgzR9OnT9fXXX0uSli9fLofDIZvNpi1btsjj8ejBBx8M1KYCAIAgFJQBq6amRgUFBZo2bZrCw8MVHx%2Bv4cOH67PPPjvrc202m8aPH68%2BffqoTZs2uvPOOyVJ27dvV21trdauXauZM2fqRz/6kWJjY5Wfn68dO3boyy%2B/vNCbBQAAWomgDFgxMTGaMGGCLJZv/lb1gQMH9Morr3iPUlVXVys3N1fp6em65ppr9Pzzz8vj%2BeYPUDocDqWkpHjXCg0NVc%2BePWW32/X555/r%2BPHj6tWrl3e%2BS5cuatOmjRwOhx%2B3EAAABDNLoAtojvLyco0cOVK1tbWaOHGi8vLytG/fPnXv3l2TJ0/W8uXL9fe//1333nuvoqOjlZWVJbfbrZiYGJ91YmJi5HK55Ha7JUlWq9Vn3mq1ntN1WBUVFXI6nT5jFktbJSYmnueWNi4sLNTnO5qHfppFP82in%2BbQS7PoZ%2BOCOmB9e63VP//5Tz388MO6//77tWzZMv3hD3/wPmbgwIGaNGmS1q1bp6ysLEnyHs36LmebPxubzaaioiKfsdzcXOXl5TVr3e9itV58Qdb9ofq2n1fM2xzgSs7N7kXXBbqERrF/mkU/zaGXZtFPX0EdsCQpJCREycnJKigo0KRJkzRv3jzFx8f7PCYpKUlbtmyRJMXFxXmPVH3L7XarW7du3ue53W5FRkZ657/66islJCQ0uabs7GwNHTrUZ8xiaSuXq/qctu1swsJCZbVerMrKGtXV1Rtd%2B4co2Ptpev9qrmDvZ0tDP82hl2a19H7GxUWe/UEXQFAGrF27dmnBggUqKSlRaOg3hyS//b5z507V1NTolltu8T7%2BwIED6tixoyQpNTVVDodD48aNkyTV1dVpz549ysrKUseOHRUTEyOHw6GkpCRJ0qeffqpTp04pNTW1yfUlJiY2OB3odB5Xbe2F2fHq6uov2No/RMHaz5Zac7D2s6Win%2BbQS7Pop6%2BgPGGampqqqqoqLV26VDU1NTp27JhWrFihK664QtHR0XryySf19ttv6/Tp03rnnXf08ssvKycnR5KUk5OjV199VR9%2B%2BKFqamq0atUqhYeHa/DgwQoLC9PEiRP1zDPP6PDhw3K5XPr1r3%2Bt4cOHq127dgHeagAAECyC8ghWdHS0nnvuOS1cuFBXXnml2rZtqyuvvFKLFi1Shw4dNHfuXD3%2B%2BOM6fPiw2rVrp7lz52rEiBGSpEGDBum%2B%2B%2B5Tfn6%2Bjh49qrS0NBUXF6tNmzaSpLy8PFVXV%2BvGG29UbW2thgwZ4r2/FgAAQFOEeJp7RTeaxOk8bnxNiyVUcXGRcrmqOSxrwJn9HPXUO4Eu6ZyU5F8d6BJ8sH%2BaRT/NoZdmtfR%2Btm8fHZDXDcpThAAAAC0ZAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgWNAGrH379mny5Mnq16%2BfMjIylJ%2BfL6fTKUnatWuXsrKy1LdvX40ePVrr16/3ee7q1as1cuRI9e3bVzk5OSotLfXOnTx5Ug8//LAGDRqk9PR05eXlyeVy%2BXXbAABAcAvKgHXq1CndcccdGjBggHbt2qWNGzfq6NGjWrBggSoqKjRz5kxNmjRJu3bt0rx58zR//nzZ7XZJ0rZt27RixQr96le/0rvvvqshQ4Zo%2BvTp%2BvrrryVJy5cvl8PhkM1m05YtW%2BTxePTggw8GcnMBAECQCcqAVVNTo4KCAk2bNk3h4eGKj4/X8OHD9dlnn2nDhg1KTk5WVlaWIiIilJGRoaFDh2rNmjWSJJvNpvHjx6tPnz5q06aN7rzzTknS9u3bVVtbq7Vr12rmzJn60Y9%2BpNjYWOXn52vHjh368ssvA7nJAAAgiFgCXcD5iImJ0YQJE7z/PnDggF555RWNGjVKDodDKSkpPo9PSUlRSUmJJMnhcOj666/3zoWGhqpnz56y2%2B3q2bOnjh8/rl69ennnu3TpojZt2sjhcKhDhw4XeMtat1FPvRPoEgAA8IugDFjfKi8v18iRI1VbW6uJEycqLy9Pd911V4MgFBsb672Oyu12KyYmxmc%2BJiZGLpdLbrdbkmS1Wn3mrVbrOV2HVVFR4b0e7FsWS1slJiY2eY2mCAsL9fmOHzaLpWXtB%2ByfZtFPc%2BilWfSzcUEdsJKSkmS32/XPf/5TDz/8sO6///4mPc/j8TRr/mxsNpuKiop8xnJzc5WXl9esdb%2BL1XrxBVkXwSUuLjLQJTSK/dMs%2BmkOvTSLfvoK6oAlSSEhIUpOTlZBQYEmTZqkzMxM75Gob7lcLsXHx0uS4uLiGsy73W5169bN%2Bxi3263IyP/7n9VXX32lhISEJteUnZ2toUOH%2BoxZLG3lclWf07adTVhYqKzWi1VZWaO6unqjayP4mN6/mov90yz6aQ69NKul9zNQP3wGZcDatWuXFixYoJKSEoWGfnNI8tvvvXv31pYtW3weX1paqj59%2BkiSUlNT5XA4NG7cOElSXV2d9uzZo6ysLHXs2FExMTFyOBxKSkqSJH366ac6deqUUlNTm1xfYmJig9OBTudx1dZemB2vrq7%2Bgq2N4NFS9wH2T7Popzn00iz66SsoT5impqaqqqpKS5cuVU1NjY4dO6YVK1boiiuuUE5OjsrLy7VmzRqdPHlSO3fu1M6dOzVx4kRJUk5Ojl599VV9%2BOGHqqmp0apVqxQeHq7BgwcrLCxMEydO1DPPPKPDhw/L5XLp17/%2BtYYPH6527doFeKsBAECwCMojWNHR0Xruuee0cOFCXXnllWrbtq2uvPJKLVq0SAkJCXr22We1cOFCPfroo0pKStLSpUv105/%2BVJI0aNAg3XfffcrPz9fRo0eVlpam4uJitWnTRpKUl5en6upq3XjjjaqtrdWQIUO0YMGCAG4tAAAINiGe5l7RjSZxOo8bX9NiCVVcXKRcruqgOCzLbRourJL8qwNdgo9g2z9bOvppDr00q6X3s3376IC8blCeIgQAAGjJCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwLChv0wCgoWD6Lc2W9huPAGAaR7AAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwLGgDVnl5uXJzc5Wenq6MjAw98MADqqys1L/%2B9S/16NFDaWlpPl%2B/%2B93vvM/dtGmTxowZo8svv1zjx4/X22%2B/7Z2rr6/X8uXLNWzYMPXv319Tp07VF198EYhNBAAAQSpoA9b06dNltVq1bds2rVu3Tp999pmefPJJ77zdbvf5mjp1qiRp7969mjNnjmbPnq2//vWvmjJliu6%2B%2B24dOXJEkvTiiy9qw4YNKi4u1vbt25WcnKzc3Fx5PJ6AbCcAAAg%2BQRmwKisrlZqaqlmzZikyMlKXXHKJxo0bp927d5/1uWvWrFFmZqYyMzMVERGhsWPHqnv37lq/fr0kyWazacqUKerSpYuioqJUUFCgsrIyffTRRxd6swAAQCthCXQB58NqtWrx4sU%2BY4cPH1ZiYqL33/fff7/effdd1dbWasKECcrLy9NFF10kh8OhzMxMn%2BempKTIbrfrxIkT2r9/v1JSUrxzUVFR6tSpk%2Bx2uy677LIm1VdRUSGn0%2BkzZrG09anPhLCwUJ/vQLCwWNhnzxXvd3PopVn0s3FBGbDOZLfb9cILL2jVqlUKDw/X5ZdfruHDh2vRokXau3ev7rnnHlksFt17771yu92KiYnxeX5MTIz279%2Bvr776Sh6Pp9F5l8vV5HpsNpuKiop8xnJzc5WXl3f%2BG/k9rNaLL8i6wIUSFxcZ6BKCFu93c%2BilWfTTV9AHrPfff18zZszQrFmzlJGRIUl66aWXvPO9e/fWtGnT9Oyzz%2Bree%2B%2BVpLNeT9Xc662ys7M1dOhQnzGLpa1crupmrXumsLBQWa0Xq7KyRnV19UbXBi4k0%2B%2BFHwLe7%2BbQS7Naej8D9QNdUAesbdu26Ze//KXmz5%2Bvm2666Tsfl5SUpH//%2B9/yeDyKi4uT2%2B32mXe73YqPj1dsbKxCQ0MbnU9ISGhyXYmJiQ1OBzqdx1Vbe2F2vLq6%2Bgu2NnAhsL%2BeP97v5tBLs%2Binr6A9YfrBBx9ozpw5evrpp33C1a5du7Rq1Sqfxx44cEBJSUkKCQlRamqqSktLfebtdrv69OmjiIgIdevWTQ6HwztXWVmpzz//XL17976wGwQAAFqNoAxYtbW1euihhzR79mwNHDjQZy46OlorV67Ua6%2B9ptOnT8tut%2Bt3v/udcnJyJEkTJ07Uu%2B%2B%2Bqx07dujkyZNau3atDh06pLFjx0qScnJytHr1apWVlamqqkqFhYXq2bOn0tLS/L6dAAAgOAXlKcIPP/xQZWVlWrhwoRYuXOgzt3nzZi1fvlxFRUV6%2BOGHFR0drdtuu02TJ0%2BWJHXv3l2FhYVavHixysvL1bVrVz377LNq3769JGnSpElyOp267bbbVF1drfT09AYXrAMAAHyfEA930PQLp/O48TUtllDFxUXK5aoOivPeo556J9AloIUoyb860CUEnWB7v7dk9NKslt7P9u2jA/K6QXmKEAAAoCUjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGOb3gDV06FAVFRXp8OHD/n5pAAAAv/B7wLr55pu1adMmXXvttbrzzju1detW1dbW%2BrsMAACAC8bvASs3N1ebNm3Sn/70J3Xr1k1PPPGEMjMztXTpUh08eNDf5QAAABgXsGuwevXqpTlz5mj79u2aO3eu/vSnP%2Bn666/X1KlT9fHHHweqLAAAgGYLWMA6ffq0Nm3apLvuuktz5sxRhw4d9OCDD6pnz56aMmWKNmzYEKjSAAAAmsXi7xcsKyvT2rVr9eqrr6q6ulojR47U73//e/Xr18/7mP79%2B2vBggUaM2aMv8sDAABoNr8HrNGjR6tz586aNm2abrrpJsXGxjZ4TGZmpo4dO%2Bbv0gAAAIzwe8BavXq1BgwYcNbHffTRR36oBgAAwDy/B6wePXpo%2BvTpysrK0rXXXitJ%2Bq//%2Bi%2B98847Wrp0aaNHtAC0LqOeeifQJZyTkvyrA10CgCDj94vcFy9erOPHj6tr167escGDB6u%2Bvl5LlizxdzkAAADG%2Bf0I1ttvv60NGzYoLi7OO5acnKzCwkLdcMMN/i4HAADAOL8fwTpx4oQiIiIaFhIaqpqaGn%2BXAwAAYJzfA1b//v21ZMkSffXVV96xL7/8Uo8%2B%2BqjPrRoAAACCld9PEc6dO1d33HGHrrrqKkVFRam%2Bvl7V1dXq2LGj/vCHP/i7HAAAAOP8HrA6duyo119/XW%2B%2B%2BaY%2B//xzhYaGqnPnzho4cKDCwsKavE55ebmeeOIJ7d69W2FhYRo0aJDmzp0rq9WqvXv3atGiRdq7d68SEhI0adIk3XHHHd7nbtq0SatWrdK//vUvde7cWffdd58GDhwoSaqvr9fTTz%2BtjRs3qrKyUr1799aCBQvUsWNH470AAACtU0D%2BVE54eLiuvfZa3XHHHZoyZYoyMzPPKVxJ0vTp02W1WrVt2zatW7dOn332mZ588kmdOHFC06ZN05VXXqm33npLy5cv17PPPqutW7dKkvbu3as5c%2BZo9uzZ%2Butf/6opU6bo7rvv1pEjRyRJL774ojZs2KDi4mJt375dycnJys3NlcfjMd4HAADQOvk9YH3xxRfKz8/X6NGjNWzYsAZfTVFZWanU1FTNmjVLkZGRuuSSSzRu3Djt3r1bO3bs0OnTpzVjxgy1bdtWvXr10oQJE2Sz2SRJa9asUWZmpjIzMxUREaGxY8eqe/fuWr9%2BvSTJZrNpypQp6tKli6KiolRQUKCysjJufAoAAJosINdgVVRUaODAgWrbtu15rWG1WrV48WKfscOHDysxMVEOh0M9evTwOSKWkpKiNWvWSJIcDocyMzN9npuSkiK73a4TJ05o//79SklJ8c5FRUWpU6dOstvtuuyyy86rXgAA8MPi94BVWlqqv/zlL4qPjze2pt1u1wsvvKBVq1appKREVqvVZz42NlZut1v19fVyu92KiYnxmY%2BJidH%2B/fv11VdfyePxNDrvcrmaXE9FRYWcTqfPmMXSVomJiee4Zd8vLCzU5zuAC8NiCfx7jPe7OfTSLPrZOL8HrISEhPM%2BctWY999/XzNmzNCsWbOUkZGhkpKSRh8XEhLi/e%2BzXU/V3OutbDabioqKfMZyc3OVl5fXrHW/i9V68QVZF8A34uIiA12CF%2B93c%2BilWfTTl98D1rRp01RUVKRZs2b5hJ7zsW3bNv3yl7/U/PnzddNNN0mS4uPjdejQIZ/Hud1uxcbGKjQ0VHFxcXK73Q3m4%2BPjvY9pbD4hIaHJdWVnZ2vo0KE%2BYxZLW7lc1eewdWcXFhYqq/ViVVbWqK6u3ujaAP6P6ffu%2BeD9bg69NKul9zNQPyD5PWC9%2Beab%2BuCDD7Ru3TpdeumlCg31PaT40ksvNWmdDz74QHPmzNHTTz/tvcWCJKWmpuqPf/yjamtrZbF8s3l2u119%2BvTxzpeWlvqsZbfbNXr0aEVERKhbt25yOBwaMGCApG8uqP/888/Vu3fvJm9jYmJig9OBTudx1dZemB2vrq7%2Bgq0NQC3q/cX73Rx6aRb99OX3gBUVFaVBgwY1a43a2lo99NBDmj17tk%2B4kqTMzExFRUVp1apVuvPOO/Xpp59q7dq1Wrp0qSRp4sSJysrK0o4dO3TVVVdpw4YNOnTokMaOHStJysnJUXFxsQYNGqQOHTqosLBQPXv2VFpaWrNqBgAAPxwhniC8wdPu3bv1s5/9TOHh4Q3mNm/erOrqaj3yyCMqLS1Vu3btdNddd%2BmWW27xPmbr1q1atmyZysvL1bVrV82bN0/9%2B/eX9M31VytWrNBLL72k6upqpaen67HHHtMll1zSrJqdzuPNen5jLJZQxcVFyuWqDoqfGkY99U6gSwDOS0n%2B1YEuIeje7y0ZvTSrpfezffvogLxuQALWgQMH9Prrr%2Bt//ud/vLdb%2BMc//qHLL7/c36X4DQGLgIXgRcBqXeilWS29n4EKWH7/ncpdu3Zp7Nix2rp1qzZu3Cjpm5uP3n777frLX/7i73IAAACM83vAWr58uX75y19qw4YN3t8i7Nixo5YsWaKVK1f6uxwAAADj/B6wPv30U%2BXk5EjyvTfVddddp7KyMn%2BXAwAAYJzfA1Z0dLROnDjRYLyioqLRi9YBAACCjd8DVt%2B%2BffXEE0%2BoqqrKO3bw4EHNmTNHV111lb/LAQAAMM7v98F68MEHNXnyZKWnp6uurk59%2B/ZVTU2NunXrpiVLlvi7HAAAAOP8HrAuueQSbdy4UTt37tTBgwfVpk0bde7cWVdffXWz/3QOAABAS%2BD3gCVJF110ka699tpAvDQAAMAF5/eANXTo0O89UsW9sAAAQLDze8C6/vrrfQJWXV2dDh48KLvdrsmTJ/u7HAAAAOP8HrBmz57d6PiWLVv0t7/9zc/VAAAAmOf32zR8l2uvvVavv/56oMsAAABothYTsPbs2aMA/N1pAAAA4/x%2BinDSpEkNxmpqalRWVqYRI0b4uxwAAADj/B6wkpOTG/wWYUREhLKysjRhwgR/lwMAAGCc3wMWd2sHAACtnd8D1quvvtrkx950000XsBIAAIALw%2B8Ba968eaqvr29wQXtISIjPWEhICAELAAAEJb8HrN/%2B9rd67rnnNH36dPXo0UMej0effPKJfvOb3%2BjWW29Venq6v0sCAAAwKiDXYBUXF6tDhw7esSuuuEIdO3bU1KlTtXHjRn%2BXBAAAYJTf74N16NAhxcTENBi3Wq0qLy/3dzkAAADG%2BT1gJSUlacmSJXK5XN6xyspKLVu2TD/%2B8Y/9XQ4AAIBxfj9FOHfuXM2aNUs2m02RkZEKDQ1VVVWV2rRpo5UrV/q7HAAAAOP8HrAGDhyoHTt2aOfOnTpy5Ig8Ho86dOiga665RtHR0f4uBwAAwDi/ByxJuvjiizVs2DAdOXJEHTt2DEQJAAAAF4zfr8E6ceKE5syZo8svv1yjRo2S9M01WHfeeacqKyv9XQ4AAIBxfj%2BCtXTpUu3du1eFhYW6//77veN1dXUqLCzUY4895u%2BSAOB7jXrqnUCXcE5K8q8OdAnAD57fj2Bt2bJF//mf/6nrrrvO%2B0efrVarFi9erK1bt/q7HAAAAOP8HrCqq6uVnJzcYDw%2BPl5ff/21v8sBAAAwzu8B68c//rH%2B9re/SZLP3x7cvHmz/uM//sPf5QAAABjn92uwbrnlFt1zzz26%2BeabVV9fr%2Beff16lpaXasmWL5s2b5%2B9yAAAAjPN7wMrOzpbFYtELL7ygsLAwPfPMM%2BrcubMKCwt13XXX%2BbscAAAA4/wesI4dO6abb75ZN998s79fGgAAwC/8fg3WsGHDfK69ao633npLGRkZKigo8Blft26dfvrTnyotLc3n6%2BOPP5Yk1dfXa/ny5Ro2bJj69%2B%2BvqVOn6osvvvA%2B3%2B12Kz8/XxkZGRo4cKDmzZunEydOGKkZAAC0fn4PWOnp6SopKWn2Or/5zW%2B0cOFCderUqdH5/v37y263%2B3z17t1bkvTiiy9qw4YNKi4u1vbt25WcnKzc3Fxv8Js/f75qamq0ceNGvfzyyyorK1NhYWGzawYAAD8Mfj9F%2BKMf/UiLFi1ScXGxfvzjH%2Buiiy7ymV%2B2bFmT1omIiNDatWu1aNEinTx58pxqsNlsmjJlirp06SJJKigoUHp6uj766CNdeumleuONN/TKK68oPj5ekjRz5kzde%2B%2B9mjNnToN6AQAAzuT3gLV//3795Cc/kSS5XK7zXuf222//3vnDhw/r5z//uUpLS2W1WpWXl6cbb7xRJ06c0P79%2B5WSkuJ9bFRUlDp16iS73a7jx48rLCxMPXr08M736tVLX3/9tQ4cOOAz/l0qKirkdDp9xiyWtkpMTDzHrfx%2BYWGhPt8BQJIsFj4Tvg%2BfnWbRz8b5LWAVFBRo%2BfLl%2BsMf/uAdW7lypXJzc42/Vnx8vJKTk3Xfffepa9eu%2BvOf/6z7779fiYmJ%2BslPfiKPx6OYmBif58TExMjlcik2NlZRUVHeu8x/Oyc1PRDabDYVFRX5jOXm5iovL6%2BZW9Y4q/XiC7IugOAUFxcZ6BKCAp%2BdZtFPX34LWNu2bWswVlxcfEEC1uDBgzV48GDvv0ePHq0///nPWrdunWbPni1J33uhfXMvws/OztbQoUN9xiyWtnK5qpu17pnCwkJltV6sysoa1dXVG10bQPAy/VnT2vDZaVZL72egfuDwW8BqLLSY%2Bm3CpkhKSlJpaaliY2MVGhoqt9vtM%2B92u5WQkKD4%2BHhVVVWprq5OYWFh3jlJSkhIaNJrJSYmNjgd6HQeV23thdnx6urqL9jaAIIPnwdNw2enWfTTl99OmP7/U27fN2bCH//4R23atMlnrKysTB07dlRERIS6desmh8PhnausrNTnn3%2Bu3r17q2fPnvJ4PNq3b5933m63y2q1qnPnzhekXgAA0Lq0yivSTp06pccff1x2u12nT5/Wxo0b9eabb2rSpEmSpJycHK1evVplZWWqqqpSYWGhevbsqbS0NMXHx2vkyJF66qmndOzYMR05ckQrV65UVlaWLBa//04AAAAIQkGbGNLS0iRJtbW1kqQ33nhD0jdHm26//XZVV1fCzWFXAAATfUlEQVTr3nvvldPp1KWXXqqVK1cqNTVVkjRp0iQ5nU7ddtttqq6uVnp6us9F6Y899pgeeeQRDRs2TBdddJFuuOGGBjczBQAA%2BC4hHj9dCJWSkqJRo0b5jJWUlDQYa%2Bp9sIKN03nc%2BJoWS6ji4iLlclUHxXnvUU%2B9E%2BgSgB%2BEkvyrA11CixZsn50tXUvvZ/v20QF5Xb8dwerXr58qKirOOgYAABDs/Baw/v/9rwAAAFqzVnmROwAAQCARsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYJgl0AWgea6YtznQJQAAgDNwBAsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgWFAHrLfeeksZGRkqKChoMLdp0yaNGTNGl19%2BucaPH6%2B3337bO1dfX6/ly5dr2LBh6t%2B/v6ZOnaovvvjCO%2B92u5Wfn6%2BMjAwNHDhQ8%2BbN04kTJ/yyTQAAIPgFbcD6zW9%2Bo4ULF6pTp04N5vbu3as5c%2BZo9uzZ%2Butf/6opU6bo7rvv1pEjRyRJL774ojZs2KDi4mJt375dycnJys3NlcfjkSTNnz9fNTU12rhxo15%2B%2BWWVlZWpsLDQr9sHAACCV9AGrIiICK1du7bRgLVmzRplZmYqMzNTERERGjt2rLp3767169dLkmw2m6ZMmaIuXbooKipKBQUFKisr00cffaR///vfeuONN1RQUKD4%2BHh16NBBM2fO1Msvv6zTp0/7ezMBAEAQCtqAdfvttys6OrrROYfDoZSUFJ%2BxlJQU2e12nThxQvv37/eZj4qKUqdOnWS327V3716FhYWpR48e3vlevXrp66%2B/1oEDBy7MxgAAgFalVf6xZ7fbrZiYGJ%2BxmJgY7d%2B/X1999ZU8Hk%2Bj8y6XS7GxsYqKilJISIjPnCS5XK4mvX5FRYWcTqfPmMXSVomJieezOd8pLCxo8zGAC8hi4bPh%2B3z72clnqBn0s3GtMmBJ8l5PdT7zZ3vu2dhsNhUVFfmM5ebmKi8vr1nrAkBTxMVFBrqEoGC1XhzoEloV%2BumrVQasuLg4ud1unzG32634%2BHjFxsYqNDS00fmEhATFx8erqqpKdXV1CgsL885JUkJCQpNePzs7W0OHDvUZs1jayuWqPt9NahQ/LQBojOnPmtYmLCxUVuvFqqysUV1dfaDLCXotvZ%2BB%2BoGjVQas1NRUlZaW%2BozZ7XaNHj1aERER6tatmxwOhwYMGCBJqqys1Oeff67evXsrKSlJHo9H%2B/btU69evbzPtVqt6ty5c5NePzExscHpQKfzuGprW96OB6D14bOmaerq6umVQfTTV6s8BDJx4kS9%2B%2B672rFjh06ePKm1a9fq0KFDGjt2rCQpJydHq1evVllZmaqqqlRYWKiePXsqLS1N8fHxGjlypJ566ikdO3ZMR44c0cqVK5WVlSWLpVXmUQAAYFjQJoa0tDRJUm1trSTpjTfekPTN0abu3bursLBQixcvVnl5ubp27apnn31W7du3lyRNmjRJTqdTt912m6qrq5Wenu5zzdRjjz2mRx55RMOGDdNFF12kG264odGbmQIAADQmxNPcK7rRJE7nceNrWiyhGl74lvF1AQS3kvyrA11Ci2axhCouLlIuVzWntAxo6f1s377xWzpdaK3yFCEAAEAgEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMswS6AACAWaOeeifQJTRZSf7VgS4BuCA4ggUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMa7UBq0ePHkpNTVVaWpr36/HHH5ck7dq1S1lZWerbt69Gjx6t9evX%2Bzx39erVGjlypPr27aucnByVlpYGYhMAAECQatX3wdq8ebMuvfRSn7GKigrNnDlT8%2BbN05gxY/T%2B%2B%2B9rxowZ6ty5s9LS0rRt2zatWLFCv/3tb9WjRw%2BtXr1a06dP19atW9W2bdsAbQkAAAgmrfYI1nfZsGGDkpOTlZWVpYiICGVkZGjo0KFas2aNJMlms2n8%2BPHq06eP2rRpozvvvFOStH379kCWDQAAgkirPoK1bNky/eMf/1BVVZVGjRqlBx54QA6HQykpKT6PS0lJUUlJiSTJ4XDo%2Buuv986FhoaqZ8%2BestvtGj16dJNet6KiQk6n02fMYmmrxMTEZm6Rr7CwH1w%2BBtDKWCz%2B/xz79rOTz1Az6GfjWm3Auuyyy5SRkaEnn3xSX3zxhfLz8/Xoo4/K7XarQ4cOPo%2BNjY2Vy%2BWSJLndbsXExPjMx8TEeOebwmazqaioyGcsNzdXeXl557k1ANA6xcVFBuy1rdaLA/barRH99NVqA5bNZvP%2Bd5cuXTR79mzNmDFD/fr1O%2BtzPR5Ps147OztbQ4cO9RmzWNrK5apu1rpn4qcFAMHO9OdiU4SFhcpqvViVlTWqq6v3%2B%2Bu3Ni29n4EK8a02YJ3p0ksvVV1dnUJDQ%2BV2u33mXC6X4uPjJUlxcXEN5t1ut7p169bk10pMTGxwOtDpPK7a2pa34wFAIAXyc7Gurp7PZYPop69WeQhkz549WrJkic9YWVmZwsPDlZmZ2eC2C6WlperTp48kKTU1VQ6HwztXV1enPXv2eOcBAADOplUGrISEBNlsNhUXF%2BvUqVM6ePCgnn76aWVnZ%2BvGG29UeXm51qxZo5MnT2rnzp3auXOnJk6cKEnKycnRq6%2B%2Bqg8//FA1NTVatWqVwsPDNXjw4MBuFAAACBohnuZecNRCvffee1q2bJk%2B%2BeQThYeHa9y4cSooKFBERITee%2B89LVy4UGVlZUpKStKsWbM0YsQI73P/%2B7//W8XFxTp69KjS0tK0YMECde/evVn1OJ3Hm7tJDVgsoRpe%2BJbxdQHAX0ryr/b7a1osoYqLi5TLVc0pLQNaej/bt48OyOu22oDV0hCwAKAhAlbwa%2Bn9DFTAapWnCAEAAAKJgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAyzBLoAAMAP16in3gl0CeekJP/qQJeAIMERLAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFiNKC8v1y9%2B8Qulp6dryJAhWrp0qerr6wNdFgAACBL8qZxG3HPPPerVq5feeOMNHT16VNOmTVO7du3085//PNClAQCAIEDAOoPdbte%2Bffv0/PPPKzo6WtHR0ZoyZYp%2B//vfE7AA4AeOv52IpiJgncHhcCgpKUkxMTHesV69eungwYOqqqpSVFTUWdeoqKiQ0%2Bn0GbNY2ioxMdForWFhnOEFAHy3YAqEf559TaBLMIqAdQa32y2r1eoz9m3YcrlcTQpYNptNRUVFPmN333237rnnHnOF6psgN/mSz5SdnW08vP0QVVRUyGaz0U9D6KdZ9NMcemkW/Wwch0Aa4fF4mvX87OxsrVu3zucrOzvbUHX/x%2Bl0qqioqMHRMpwf%2BmkW/TSLfppDL82in43jCNYZ4uPj5Xa7fcbcbrdCQkIUHx/fpDUSExNJ8QAA/IBxBOsMqampOnz4sI4dO%2BYds9vt6tq1qyIjIwNYGQAACBYErDOkpKQoLS1Ny5YtU1VVlcrKyvT8888rJycn0KUBAIAgEbZgwYIFgS6ipbnmmmu0ceNGPf7443r99deVlZWlqVOnKiQkJNClNRAZGakBAwZwdM0Q%2BmkW/TSLfppDL82inw2FeJp7RTcAAAB8cIoQAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACVhAqLy/XL37xC6Wnp2vIkCFaunSp6uvrA11WUHnrrbeUkZGhgoKCBnObNm3SmDFjdPnll2v8%2BPF6%2B%2B23A1BhcCkvL1dubq7S09OVkZGhBx54QJWVlZKkvXv36tZbb1W/fv00YsQIPffccwGutmXbt2%2BfJk%2BerH79%2BikjI0P5%2BflyOp2SpF27dikrK0t9%2B/bV6NGjtX79%2BgBXG1yeeOIJ9ejRw/tv%2BnnuevToodTUVKWlpXm/Hn/8cUn0swEPgs64ceM8Dz30kKeystJz8OBBz4gRIzzPPfdcoMsKGsXFxZ4RI0Z4Jk2a5MnPz/eZ27Nnjyc1NdWzY8cOz4kTJzyvvfaap0%2BfPp7Dhw8HqNrgcMMNN3geeOABT1VVlefw4cOe8ePHe%2BbOneupqanxXHPNNZ4VK1Z4qqurPaWlpZ4BAwZ4tmzZEuiSW6STJ096rrrqKk9RUZHn5MmTnqNHj3puvfVWz8yZMz1ffvml57LLLvOsWbPGc%2BLECc8777zj6d27t%2Bfjjz8OdNlBYc%2BePZ4BAwZ4unfv7vF4PPTzPHXv3t3zxRdfNBinnw1xBCvI2O127du3T7Nnz1Z0dLSSk5M1ZcoU2Wy2QJcWNCIiIrR27Vp16tSpwdyaNWuUmZmpzMxMRUREaOzYserevTs/iX2PyspKpaamatasWYqMjNQll1yicePGaffu3dqxY4dOnz6tGTNmqG3bturVq5cmTJjA/vodampqVFBQoGnTpik8PFzx8fEaPny4PvvsM23YsEHJycnKyspSRESEMjIyNHToUK1ZsybQZbd49fX1euSRRzRlyhTvGP00i342RMAKMg6HQ0lJSYqJifGO9erVSwcPHlRVVVUAKwset99%2Bu6KjoxudczgcSklJ8RlLSUmR3W73R2lByWq1avHixWrXrp137PDhw0pMTJTD4VCPHj0UFhbmnUtJSVFpaWkgSm3xYmJiNGHCBFksFknSgQMH9Morr2jUqFHfuW/Sy7N76aWXFBERoTFjxnjH6Of5W7ZsmQYPHqwrrrhC8%2BfPV3V1Nf1sBAEryLjdblmtVp%2Bxb8OWy%2BUKREmtitvt9gmv0jf9pbdNZ7fb9cILL2jGjBmN7q%2BxsbFyu91cN/g9ysvLlZqaquuvv15paWnKy8v7zl6yb36/f//731qxYoUeeeQRn3H6eX4uu%2BwyZWRkaOvWrbLZbPrwww/16KOP0s9GELCCkMfjCXQJrRr9PX/vv/%2B%2Bpk6dqlmzZikjI%2BM7HxcSEuLHqoJPUlKS7Ha7Nm/erEOHDun%2B%2B%2B8PdElBa/HixRo/fry6du0a6FJaBZvNpgkTJig8PFxdunTR7NmztXHjRp0%2BfTrQpbU4BKwgEx8fL7fb7TPmdrsVEhKi%2BPj4AFXVesTFxTXaX3p7dtu2bdMvfvELzZ07V7fffrukb/bXM3%2BCdbvdio2NVWgoHz/fJyQkRMnJySooKNDGjRtlsVga7Jsul4t983vs2rVL//jHP5Sbm9tgrrH3Ov08d5deeqnq6uoUGhpKP8/AJ1yQSU1N1eHDh3Xs2DHvmN1uV9euXRUZGRnAylqH1NTUBtcM2O129enTJ0AVBYcPPvhAc%2BbM0dNPP62bbrrJO56amqpPPvlEtbW13jH6%2Bd127dqlkSNH%2Bpw%2B/TaI9u7du8G%2BWVpaSi%2B/x/r163X06FENGTJE6enpGj9%2BvCQpPT1d3bt3p5/naM%2BePVqyZInPWFlZmcLDw5WZmUk/z0DACjIpKSlKS0vTsmXLVFVVpbKyMj3//PPKyckJdGmtwsSJE/Xuu%2B9qx44dOnnypNauXatDhw5p7NixgS6txaqtrdVDDz2k2bNna%2BDAgT5zmZmZioqK0qpVq1RTU6OPPvpIa9euZX/9DqmpqaqqqtLSpUtVU1OjY8eOacWKFbriiiuUk5Oj8vJyrVmzRidPntTOnTu1c%2BdOTZw4MdBlt1gPPPCAtmzZotdee02vvfaaiouLJUmvvfaaxowZQz/PUUJCgmw2m4qLi3Xq1CkdPHhQTz/9tLKzs3XjjTfSzzOEeLjgJOgcOXJE8%2BfP19///ndFRUVp0qRJuvvuu7mupYnS0tIkyXtU5dvf2Pr2NwW3bt2qZcuWqby8XF27dtW8efPUv3//wBQbBHbv3q2f/exnCg8PbzC3efNmVVdX65FHHlFpaanatWunu%2B66S7fccksAKg0On3zyiRYuXKiPP/5Ybdu21ZVXXqkHHnhAHTp00HvvvaeFCxeqrKxMSUlJmjVrlkaMGBHokoPGv/71Lw0bNkyffPKJJNHP8/Dee%2B9p2bJl%2BuSTTxQeHq5x48apoKBAERER9PMMBCwAAADDOEUIAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIb9L/KAs7zacoQ1AAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-8531695816899444010">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">11</td>
        <td class="number">740</td>
        <td class="number">4.9%</td>
        <td>
            <div class="bar" style="width:9%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">739</td>
        <td class="number">4.9%</td>
        <td>
            <div class="bar" style="width:9%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">13</td>
        <td class="number">717</td>
        <td class="number">4.7%</td>
        <td>
            <div class="bar" style="width:9%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">14</td>
        <td class="number">699</td>
        <td class="number">4.6%</td>
        <td>
            <div class="bar" style="width:9%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">12</td>
        <td class="number">677</td>
        <td class="number">4.5%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9</td>
        <td class="number">664</td>
        <td class="number">4.4%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15</td>
        <td class="number">664</td>
        <td class="number">4.4%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">16</td>
        <td class="number">640</td>
        <td class="number">4.2%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">17</td>
        <td class="number">598</td>
        <td class="number">4.0%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8</td>
        <td class="number">574</td>
        <td class="number">3.8%</td>
        <td>
            <div class="bar" style="width:7%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (42)</td>
        <td class="number">8408</td>
        <td class="number">55.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-8531695816899444010">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">5</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">78</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:26%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">134</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:44%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">210</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:69%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">305</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">47</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:60%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">48</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:20%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">49</td>
        <td class="number">5</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">50</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:20%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">52</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:20%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type1">Soil_Type1<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-2117104032486107786">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 97.7%">
            14765
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:3%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 2.3%">
            &nbsp;
        </div>
        355
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-2117104032486107786, #minifreqtable-2117104032486107786"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-2117104032486107786">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14765</td>
        <td class="number">97.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">355</td>
        <td class="number">2.3%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type10">Soil_Type10<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-5947398468550314928">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 85.8%">
            12978
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:17%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 14.2%">
            &nbsp;
        </div>
        2142
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-5947398468550314928, #minifreqtable-5947398468550314928"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-5947398468550314928">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">12978</td>
        <td class="number">85.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">2142</td>
        <td class="number">14.2%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type11">Soil_Type11<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-2298332004144779082">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 97.3%">
            14714
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:3%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 2.7%">
            &nbsp;
        </div>
        406
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-2298332004144779082, #minifreqtable-2298332004144779082"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-2298332004144779082">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14714</td>
        <td class="number">97.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">406</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type12">Soil_Type12<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-8460581745886543135">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 98.5%">
            14893
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:2%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 1.5%">
            &nbsp;
        </div>
        227
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-8460581745886543135, #minifreqtable-8460581745886543135"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-8460581745886543135">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14893</td>
        <td class="number">98.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">227</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type13">Soil_Type13<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-913449036818817083">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 96.9%">
            14644
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:4%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 3.1%">
            &nbsp;
        </div>
        476
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-913449036818817083, #minifreqtable-913449036818817083"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-913449036818817083">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14644</td>
        <td class="number">96.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">476</td>
        <td class="number">3.1%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type14">Soil_Type14<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable250171779957153945">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 98.9%">
            14951
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:2%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 1.1%">
            &nbsp;
        </div>
        169
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable250171779957153945, #minifreqtable250171779957153945"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable250171779957153945">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14951</td>
        <td class="number">98.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">169</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow ignore">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type15"><s>Soil_Type15</s><br/>
            <small>Constant</small>
        </p>
    </div><div class="col-md-3">
    <p><em>This variable is constant and should be ignored for analysis</em></p>
</div>
<div class="col-md-6">
    <table class="stats ">
        <tr>
            <th>Constant value</th>
            <td>0</td>
        </tr>
    </table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type16">Soil_Type16<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable7309902722191144092">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.2%">
            15006
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.8%">
            &nbsp;
        </div>
        114
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable7309902722191144092, #minifreqtable7309902722191144092"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable7309902722191144092">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15006</td>
        <td class="number">99.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">114</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type17">Soil_Type17<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable8356544300275442827">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 96.0%">
            14508
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.0%">
            &nbsp;
        </div>
        612
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable8356544300275442827, #minifreqtable8356544300275442827"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable8356544300275442827">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14508</td>
        <td class="number">96.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">612</td>
        <td class="number">4.0%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type18">Soil_Type18<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable2575496275498457323">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.6%">
            15060
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.4%">
            &nbsp;
        </div>
        60
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable2575496275498457323, #minifreqtable2575496275498457323"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable2575496275498457323">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15060</td>
        <td class="number">99.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">60</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type19">Soil_Type19<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable7075998845334459099">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.7%">
            15074
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.3%">
            &nbsp;
        </div>
        46
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable7075998845334459099, #minifreqtable7075998845334459099"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable7075998845334459099">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15074</td>
        <td class="number">99.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">46</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type2">Soil_Type2<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-416663181876624889">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.9%">
            14497
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.1%">
            &nbsp;
        </div>
        623
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-416663181876624889, #minifreqtable-416663181876624889"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-416663181876624889">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14497</td>
        <td class="number">95.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">623</td>
        <td class="number">4.1%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type20">Soil_Type20<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-2103513778641731213">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.1%">
            14981
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.9%">
            &nbsp;
        </div>
        139
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-2103513778641731213, #minifreqtable-2103513778641731213"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-2103513778641731213">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14981</td>
        <td class="number">99.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">139</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type21">Soil_Type21<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-6716775248193466362">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15104
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        16
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-6716775248193466362, #minifreqtable-6716775248193466362"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-6716775248193466362">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15104</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">16</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type22">Soil_Type22<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-2004837501397856249">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 97.7%">
            14775
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:3%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 2.3%">
            &nbsp;
        </div>
        345
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-2004837501397856249, #minifreqtable-2004837501397856249"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-2004837501397856249">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14775</td>
        <td class="number">97.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">345</td>
        <td class="number">2.3%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type23">Soil_Type23<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable2612731461151736288">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.0%">
            14363
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:6%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 5.0%">
            &nbsp;
        </div>
        757
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable2612731461151736288, #minifreqtable2612731461151736288"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable2612731461151736288">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14363</td>
        <td class="number">95.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">757</td>
        <td class="number">5.0%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type24">Soil_Type24<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable5465779069004785862">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 98.3%">
            14863
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:2%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 1.7%">
            &nbsp;
        </div>
        257
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable5465779069004785862, #minifreqtable5465779069004785862"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable5465779069004785862">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14863</td>
        <td class="number">98.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">257</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type25">Soil_Type25<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-6740042780537009202">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 100.0%">
            15119
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.0%">
            &nbsp;
        </div>
        1
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-6740042780537009202, #minifreqtable-6740042780537009202"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-6740042780537009202">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15119</td>
        <td class="number">100.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type26">Soil_Type26<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-4633616947891823572">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.6%">
            15066
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.4%">
            &nbsp;
        </div>
        54
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-4633616947891823572, #minifreqtable-4633616947891823572"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-4633616947891823572">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15066</td>
        <td class="number">99.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">54</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type27">Soil_Type27<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-7473609535174041940">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15105
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        15
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-7473609535174041940, #minifreqtable-7473609535174041940"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-7473609535174041940">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15105</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">15</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type28">Soil_Type28<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable8000655794318467313">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15111
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        9
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable8000655794318467313, #minifreqtable8000655794318467313"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable8000655794318467313">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15111</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">9</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type29">Soil_Type29<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-3565935104224671482">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 91.5%">
            13829
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:10%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 8.5%">
            &nbsp;
        </div>
        1291
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-3565935104224671482, #minifreqtable-3565935104224671482"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-3565935104224671482">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">13829</td>
        <td class="number">91.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">1291</td>
        <td class="number">8.5%</td>
        <td>
            <div class="bar" style="width:10%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type3">Soil_Type3<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-9209607627278080385">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 93.6%">
            14158
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:7%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 6.4%">
            &nbsp;
        </div>
        962
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-9209607627278080385, #minifreqtable-9209607627278080385"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-9209607627278080385">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14158</td>
        <td class="number">93.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">962</td>
        <td class="number">6.4%</td>
        <td>
            <div class="bar" style="width:7%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type30">Soil_Type30<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-6689001072464483665">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.2%">
            14395
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.8%">
            &nbsp;
        </div>
        725
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-6689001072464483665, #minifreqtable-6689001072464483665"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-6689001072464483665">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14395</td>
        <td class="number">95.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">725</td>
        <td class="number">4.8%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type31">Soil_Type31<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-8679216198536870443">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 97.8%">
            14788
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:3%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 2.2%">
            &nbsp;
        </div>
        332
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-8679216198536870443, #minifreqtable-8679216198536870443"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-8679216198536870443">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14788</td>
        <td class="number">97.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">332</td>
        <td class="number">2.2%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type32">Soil_Type32<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-3250711818858519272">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.4%">
            14430
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.6%">
            &nbsp;
        </div>
        690
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-3250711818858519272, #minifreqtable-3250711818858519272"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-3250711818858519272">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14430</td>
        <td class="number">95.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">690</td>
        <td class="number">4.6%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type33">Soil_Type33<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable128352044143450233">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.9%">
            14504
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.1%">
            &nbsp;
        </div>
        616
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable128352044143450233, #minifreqtable128352044143450233"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable128352044143450233">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14504</td>
        <td class="number">95.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">616</td>
        <td class="number">4.1%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type34">Soil_Type34<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable2241154372232424542">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15098
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        22
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable2241154372232424542, #minifreqtable2241154372232424542"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable2241154372232424542">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15098</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">22</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type35">Soil_Type35<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable9161102022663096076">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.3%">
            15018
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.7%">
            &nbsp;
        </div>
        102
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable9161102022663096076, #minifreqtable9161102022663096076"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable9161102022663096076">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15018</td>
        <td class="number">99.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">102</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type36">Soil_Type36<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable3939902560453011406">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15110
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        10
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable3939902560453011406, #minifreqtable3939902560453011406"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable3939902560453011406">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15110</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">10</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type37">Soil_Type37<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable1197792675611213553">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.8%">
            15086
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.2%">
            &nbsp;
        </div>
        34
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable1197792675611213553, #minifreqtable1197792675611213553"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable1197792675611213553">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15086</td>
        <td class="number">99.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">34</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type38">Soil_Type38<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-7706864425726872016">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.2%">
            14392
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:6%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.8%">
            &nbsp;
        </div>
        728
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-7706864425726872016, #minifreqtable-7706864425726872016"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-7706864425726872016">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14392</td>
        <td class="number">95.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">728</td>
        <td class="number">4.8%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type39">Soil_Type39<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable3414625673381870559">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.7%">
            14463
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.3%">
            &nbsp;
        </div>
        657
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable3414625673381870559, #minifreqtable3414625673381870559"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable3414625673381870559">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14463</td>
        <td class="number">95.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">657</td>
        <td class="number">4.3%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type4">Soil_Type4<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable2555300351204568778">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 94.4%">
            14277
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:6%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 5.6%">
            &nbsp;
        </div>
        843
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable2555300351204568778, #minifreqtable2555300351204568778"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable2555300351204568778">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14277</td>
        <td class="number">94.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">843</td>
        <td class="number">5.6%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type40">Soil_Type40<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable3605118407958853426">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 97.0%">
            14661
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:4%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 3.0%">
            &nbsp;
        </div>
        459
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable3605118407958853426, #minifreqtable3605118407958853426"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable3605118407958853426">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14661</td>
        <td class="number">97.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">459</td>
        <td class="number">3.0%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type5">Soil_Type5<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-8939565341666950220">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 98.9%">
            14955
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:2%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 1.1%">
            &nbsp;
        </div>
        165
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-8939565341666950220, #minifreqtable-8939565341666950220"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-8939565341666950220">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14955</td>
        <td class="number">98.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">165</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type6">Soil_Type6<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable5399315448322734463">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 95.7%">
            14470
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:5%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 4.3%">
            &nbsp;
        </div>
        650
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable5399315448322734463, #minifreqtable5399315448322734463"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable5399315448322734463">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14470</td>
        <td class="number">95.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">650</td>
        <td class="number">4.3%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow ignore">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type7"><s>Soil_Type7</s><br/>
            <small>Constant</small>
        </p>
    </div><div class="col-md-3">
    <p><em>This variable is constant and should be ignored for analysis</em></p>
</div>
<div class="col-md-6">
    <table class="stats ">
        <tr>
            <th>Constant value</th>
            <td>0</td>
        </tr>
    </table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type8">Soil_Type8<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable6951439447131137197">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 100.0%">
            15119
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.0%">
            &nbsp;
        </div>
        1
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable6951439447131137197, #minifreqtable6951439447131137197"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable6951439447131137197">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15119</td>
        <td class="number">100.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Soil_Type9">Soil_Type9<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable4539568940498085958">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.9%">
            15110
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.1%">
            &nbsp;
        </div>
        10
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable4539568940498085958, #minifreqtable4539568940498085958"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable4539568940498085958">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">15110</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">10</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Vertical_Distance_To_Hydrology">Vertical_Distance_To_Hydrology<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>423</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>2.8%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>51.077</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>-146</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>554</td>
                </tr>
                <tr class="alert">
                    <th>Zeros (%)</th>
                    <td>12.5%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram8538788201463328341">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAASZJREFUeJzt3dFpwzAUQFGndKQOkZ36nZ06RHZSFyiX2GCsqOf8G97PRRIy6DbGGBvwp4%2BrB4CZfV49wFW%2Bvn92f/N83E%2BYhJlZQSAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCv33%2B4Ii9TyZ4LuH9WUEgCASCQCAIBIJAIAgEgkAgCASCQCC4ST/R3pv3bXP7PhsrCASBQFhii3VkKwOvWCKQlTi3zEUgC/Ab/nluY4xx9RAwK4d0CAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCL/skxkAsv9b/QAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives8538788201463328341,#minihistogram8538788201463328341"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives8538788201463328341">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles8538788201463328341"
                                                  aria-controls="quantiles8538788201463328341" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram8538788201463328341" aria-controls="histogram8538788201463328341"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common8538788201463328341" aria-controls="common8538788201463328341"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme8538788201463328341" aria-controls="extreme8538788201463328341"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles8538788201463328341">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>-146</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>-4</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>32</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>79</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>176</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>554</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>700</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>74</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>61.239</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>1.199</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>3.4035</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>51.077</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>46.87</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.5378</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>772277</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>3750.3</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>118.2 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram8538788201463328341">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xt0VOWh/vEnF0kkITcw6KHUWG7mAuEiRCMSCAKCAorhki6qOQUFQdNEsKiAQLVCBUUFF5f2YKV6ltNQj1wE4SiXqsR6WoWTDIgS4QhZgUSYIRCSSJL9%2B8Mf045Bub1hZrK/n7VYlPed2ft9sif2Ye/NTpBlWZYAAABgTLCvFwAAANDcULAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGGhvl6AXVRUnPT1Ei5bcHCQ4uIidPx4lRoaLF8v54qya3a75pbsm53c9sotNf/s11zTyif79eszWB988IHS09OVn5/faG7jxo0aPny4evTooVGjRunDDz/0zDU0NGjx4sUaOHCgevfurQkTJujQoUOeebfbrby8PKWnp6tv376aOXOmampqPPN79%2B7V%2BPHj1atXLw0ePFirVq1q2qABIjg4SEFBQQoODvL1Uq44u2a3a27JvtnJba/ckr2zNyW/LVi///3v9cwzz%2Bj6669vNLd3717NmDFD06dP18cff6ycnBw9/PDDOnLkiCTpjTfe0Pr167Vy5Upt27ZNCQkJmjp1qizru2Y%2Be/ZsVVdXa8OGDfrLX/6ikpISLVq0SJJUU1OjSZMm6eabb9YHH3ygxYsXa8WKFdqyZcuVCw8AAAKa3xassLAwrVmz5pwFq6CgQBkZGcrIyFBYWJhGjBihzp07a926dZIkh8OhnJwcdejQQZGRkcrPz1dJSYl2796tb775Ru%2B9957y8/MVFxentm3basqUKfrLX/6iM2fOaPv27Tpz5oweeughtWzZUsnJyRo9erQcDseV/hIAAIAA5bf3YN13330/OOd0OpWRkeE1lpSUpKKiItXU1Gj//v1KSkryzEVGRur6669XUVGRTp48qZCQEHXp0sUzn5ycrNOnT%2Burr76S0%2BlUly5dFBIS4rXtgoKCC157eXm5KioqvMZCQ1sqPj7%2Bgrfhj0JCgr1%2BtxO7Zrdrbsm%2B2cltr9ySvbM3Jb8tWD/G7XYrOjraayw6Olr79%2B/XiRMnZFnWOeddLpdiYmIUGRmpoKAgrzlJcrlccrvdioqK8npvTEyM3G63GhoaFBx8/g%2Bgw%2BHQ0qVLvcamTp2q3Nzci8rpr6Kirvb1EnzGrtntmluyb3Zy24%2BdszeFgCxYkjz3U13K/Pneey7/WsjOZ%2BzYscrMzPQaCw1tKZer6qL3609CQoIVFXW1KiurVV/f4OvlXFF2zW7X3JJ9s5PbXrml5p89NjbCJ/sNyIIVGxsrt9vtNeZ2uxUXF6eYmBgFBwefc75169aKi4vTqVOnVF9f77kMePa1Z%2BcPHjzY6L1nt3sh4uPjG10OrKg4qbq65vHBra9vaDZZLpZds9s1t2Tf7OS2HztnbwoBecE1JSVFxcXFXmNFRUVKTU1VWFiYOnXqJKfT6ZmrrKzU119/rW7duikxMVGWZenzzz/3em9UVJRuuOEGpaSkaN%2B%2Bfaqrq2u0bQAAgAsRkAVrzJgx2rlzp7Zv367a2lqtWbNGBw8e1IgRIyRJ2dnZWr16tUpKSnTq1CktWrRIiYmJ6tq1q%2BLi4jRkyBC9%2BOKLOn78uI4cOaJXXnlFWVlZCg0NVUZGhiIjI7Vs2TJVV1dr9%2B7dWrNmjbKzs32cGgAABAq/vUTYtWtXSfKcSXrvvfckfXc2qXPnzlq0aJHmz5%2Bv0tJSdezYUStWrNA111wjSRo3bpwqKir0i1/8QlVVVUpLS/O66fw3v/mN5syZo4EDB%2Bqqq67SXXfd5XmYaYsWLbR8%2BXLNmTNHK1euVJs2bZSfn6/%2B/ftfwfQAACCQBVmXcsc3Llpz%2BFE5oaHBio2NkMtVZbvr9HbNbtfckn2zk9teuaXmn50flQMAANBMULAAAAAMo2ABAAAY5rc3uQO%2BNvTFj3y9hIuyKe9WXy8BAPD/cQYLAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgWMAWrD179ui%2B%2B%2B7TTTfdpFtvvVXTp0/X8ePHJUmFhYXKyspSz549deedd2rdunVe7129erWGDBminj17Kjs7W8XFxZ652tpaPfXUU%2BrXr5/S0tKUm5srl8t1RbMBAIDAFpAFq66uTg8%2B%2BKC6d%2B%2BunTt3asOGDTp%2B/Ljmzp2r8vJyTZkyRePGjVNhYaFmzpyp2bNnq6ioSJK0detWLVmyRM8995x27typAQMGaPLkyTp9%2BrQkafHixXI6nXI4HNq8ebMsy9ITTzzhy7gAACDABGTBqqioUEVFhUaOHKkWLVooNjZWgwYN0t69e7V%2B/XolJCQoKytLYWFhSk9PV2ZmpgoKCiRJDodDo0aNUmpqqsLDwzVx4kRJ0rZt21RXV6c1a9ZoypQpuu666xQTE6O8vDxt375dR48e9WVkAAAQQEJ9vYBL0bZtWyUmJsrhcOhXv/qVampqtGXLFvXv319Op1NJSUler09KStKmTZskSU6nU8OGDfPMBQcHKzExUUVFRUpMTNTJkyeVnJzsme/QoYPCw8PldDrVtm3bC1pfeXm5KioqvMZCQ1sqPj7%2BUiP7hZCQYK/f4V9CQ80fFzsfc7tmJ7e9ckv2zt6UArJgBQcHa8mSJcrJydFrr70mSerTp4%2BmTZumKVOmNCpCMTExnvuo3G63oqOjveajo6PlcrnkdrslSVFRUV7zUVFRF3UflsPh0NKlS73Gpk6dqtzc3Avehj%2BLirra10vAOcTGRjTZtu18zO2andz2Y%2BfsTSEgC9a3336ryZMn64477vDcPzVv3jxNnz79gt5vWdZlzZ/P2LFjlZmZ6TUWGtpSLlfVZW3X10JCghUVdbUqK6tVX9/g6%2BXge5ri82XnY27X7OS2V26p%2BWdvyr98/piALFiFhYU6fPiwHn30UYWEhKhVq1bKzc3VyJEjddttt3nORJ3lcrkUFxcnSYqNjW0073a71alTJ89r3G63IiL%2BeUBOnDih1q1bX/D64uPjG10OrKg4qbq65vHBra9vaDZZmpOmPCZ2PuZ2zU5u%2B7Fz9qYQkBdc6%2Bvr1dDQ4HWm6dtvv5Ukpaenez12QZKKi4uVmpoqSUpJSZHT6fTa1p49e5Samqr27dsrOjraa/6LL77Qt99%2Bq5SUlKaMBAAAmpGALFg9evRQy5YttWTJElVXV8vlcmnZsmXq3bu3Ro4cqdLSUhUUFKi2tlY7duzQjh07NGbMGElSdna23n77be3atUvV1dVatmyZWrRoof79%2ByskJERjxozR8uXLVVZWJpfLpRdeeEGDBg1SmzZtfJwaAAAEioC8RBgbG6v/%2BI//0O9%2B9zv169dPLVq0UJ8%2BfTR37ly1bt1aK1as0DPPPKN58%2BapXbt2WrhwoW688UZJUr9%2B/fToo48qLy9Px44dU9euXbVy5UqFh4dLknJzc1VVVaWRI0eqrq5OAwYM0Ny5c32YFgAABJog63Lv6MYFqag46eslXLbQ0GDFxkbI5aqyxXX6oS9%2B5OslXJRNebca36bdjvm/smt2ctsrt9T8s19zTSuf7DcgLxECAAD4MwoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhhkvWJmZmVq6dKnKyspMb7qRZcuWqW/fvurevbtycnJ0%2BPBhSVJhYaGysrLUs2dP3XnnnVq3bp3X%2B1avXq0hQ4aoZ8%2Beys7OVnFxsWeutrZWTz31lPr166e0tDTl5ubK5XI1eRYAANB8GC9Y9957rzZu3Kjbb79dEydO1JYtW1RXV2d6N3rjjTe0bt06rV69Wh9%2B%2BKE6duyoP/7xjyovL9eUKVM0btw4FRYWaubMmZo9e7aKiookSVu3btWSJUv03HPPaefOnRowYIAmT56s06dPS5IWL14sp9Mph8OhzZs3y7IsPfHEE8bXDwAAmi/jBWvq1KnauHGj/vznP6tTp0569tlnlZGRoYULF%2BrAgQPG9rNq1Srl5%2BfrZz/7mSIjIzVr1izNmjVL69evV0JCgrKyshQWFqb09HRlZmaqoKBAkuRwODRq1CilpqYqPDxcEydOlCRt27ZNdXV1WrNmjaZMmaLrrrtOMTExysvL0/bt23X06FFjawcAAM1baFNtODk5WcnJyfr1r3%2BtjRs3au7cuVq1apXS09P1q1/9St26dbvkbR89elSHDx/WiRMnNGzYMB07dkxpaWmaO3eunE6nkpKSvF6flJSkTZs2SZKcTqeGDRvmmQsODlZiYqKKioqUmJiokydPKjk52TPfoUMHhYeHy%2Bl0qm3bthe0vvLyclVUVHiNhYa2VHx8/KVG9gshIcFev8O/hIaaPy52PuZ2zU5ue%2BWW7J29KTVZwTpz5oz%2B%2B7//W2%2B99ZY%2B/vhjJSQk6JFHHlF5eblycnI0b948DR8%2B/JK2feTIEUnSu%2B%2B%2Bq1dffVWWZSk3N1ezZs1STU1NoyIUExPjuY/K7XYrOjraaz46Oloul0tut1uSFBUV5TUfFRV1UfdhORwOLV261Gts6tSpys3NveBt%2BLOoqKt9vQScQ2xsRJNt287H3K7ZyW0/ds7eFIwXrJKSEq1Zs0Zvv/22qqqqNGTIEL322mvq1auX5zW9e/fW3LlzL7lgWZYlSZo4caKnTD3yyCN64IEHlJ6efsHvv9T58xk7dqwyMzO9xkJDW8rlqrqs7fpaSEiwoqKuVmVlterrG3y9HHxPU3y%2B7HzM7Zqd3PbKLTX/7E35l88fY7xg3Xnnnbrhhhs0adIk3X333YqJiWn0moyMDB0/fvyS99GmTRtJ3mea2rVrJ8uydObMGc%2BZqLNcLpfi4uIkSbGxsY3m3W63OnXq5HmN2%2B1WRMQ/D8iJEyfUunXrC15ffHx8o8uBFRUnVVfXPD649fUNzSZLc9KUx8TOx9yu2cltP3bO3hSMX3BdvXq1Nm3apJycnHOWq7N27959yfu49tprFRkZqb1793rGSktLddVVVykjI8PrsQuSVFxcrNTUVElSSkqKnE6nZ66%2Bvl579uxRamqq2rdvr%2BjoaK/5L774Qt9%2B%2B61SUlIueb0AAMBejBesLl26aPLkyXrvvfc8Y3/84x/1wAMPNDpzdKlCQ0OVlZWl5cuX6//%2B7/907NgxvfLKKxo%2BfLjuuecelZaWqqCgQLW1tdqxY4d27NihMWPGSJKys7P19ttva9euXaqurtayZcvUokUL9e/fXyEhIRozZoyWL1%2BusrIyuVwuvfDCCxo0aJDnrBkAAMD5GC9Y8%2BfP18mTJ9WxY0fPWP/%2B/dXQ0KAFCxYY28%2B0adN02223afTo0br99tuVkJCgWbNmqXXr1lqxYoVef/119erVS88%2B%2B6wWLlyoG2%2B8UZLUr18/Pfroo8rLy1OfPn20c%2BdOrVy5UuHh4ZKk3NxcpaamauTIkRo4cKAiIiL029/%2B1ti6AQBA8xdkXe4d3d/Tt29frV%2B/XrGxsV7jLpdLd911lz766COTuwsYFRUnfb2EyxYaGqzY2Ai5XFW2uE4/9MXA%2BqxuyrvV%2BDbtdsz/lV2zk9teuaXmn/2aa1r5ZL/Gz2DV1NQoLCys8Y6Cg1VdXW16dwAAAH7HeMHq3bu3FixYoBMnTnjGjh49qnnz5nk9qgEAAKC5Mv6YhieffFK//OUvdcsttygyMlINDQ2qqqpS%2B/bt9ac//cn07gAAAPyO8YLVvn17vfPOO/rrX/%2Bqr7/%2BWsHBwbrhhhvUt29fhYSEmN4dAACA32mSH5XTokUL3X777U2xaQAAAL9nvGAdOnRIzz//vL788kvV1NQ0mn///fdN7xIAAMCvNMk9WOXl5erbt69atmxpevMAAAB%2Bz3jBKi4u1vvvv%2B/5uX4AAAB2Y/wxDa1bt%2BbMFQAAsDXjBWvSpElaunSpDD8gHgAAIGAYv0T417/%2BVZ9%2B%2Bqneeust/eQnP1FwsHeHe/PNN03vEgAAwK8YL1iRkZHq16%2Bf6c0CAAAEDOMFa/78%2BaY3CQAAEFCM34MlSV999ZWWLFmiJ554wjP22WefNcWuAAAA/I7xglVYWKgRI0Zoy5Yt2rBhg6TvHj5633338ZBRAABgC8YL1uLFi/XYY49p/fr1CgoKkvTdzydcsGCBXnnlFdO7AwAA8DvGC9YXX3yh7OxsSfIULEm64447VFJSYnp3AAAAfsd4wWrVqtU5fwZheXm5WrRoYXp3AAAAfsd4werZs6eeffZZnTp1yjN24MABzZgxQ7fccovp3QEAAPgd449peOKJJ3T//fcrLS1N9fX16tmzp6qrq9WpUyctWLDA9O4AAAD8jvGCde2112rDhg3asWOHDhw4oPDwcN1www269dZbve7JAgAAaK6MFyxJuuqqq3T77bc3xaYBAAD8nvGClZmZ%2BaNnqngWFgAAaO6MF6xhw4Z5Faz6%2BnodOHBARUVFuv/%2B%2B03vDgAAwO8YL1jTp08/5/jmzZv1t7/9zfTuAAAA/E6T/CzCc7n99tv1zjvvXKndAQAA%2BMwVK1h79uyRZVlXancAAAA%2BY/wS4bhx4xqNVVdXq6SkRIMHDza9OwAAAL9jvGAlJCQ0%2BleEYWFhysrK0ujRo03vDgAAwO8YL1g8rR0AANid8YL19ttvX/Br7777btO7BwAA8DnjBWvmzJlqaGhodEN7UFCQ11hQUBAFCwAANEvGC9Yf/vAHrVq1SpMnT1aXLl1kWZb27dun3//%2B9xo/frzS0tJM7xIAAMCvNMk9WCtXrlTbtm09YzfddJPat2%2BvCRMmaMOGDaZ3CQAA4FeMPwfr4MGDio6ObjQeFRWl0tJS07sDAADwO8YLVrt27bRgwQK5XC7PWGVlpZ5//nn99Kc/Nb07AAAAv2P8EuGTTz6padOmyeFwKCIiQsHBwTp16pTCw8P1yiuvmN4dAACA3zFesPr27avt27drx44dOnLkiCzLUtu2bXXbbbepVatWpncHAADgd4wXLEm6%2BuqrNXDgQB05ckTt27dvil0AAAD4LeP3YNXU1GjGjBnq0aOHhg4dKum7e7AmTpyoyspK07sDAADwO8YL1sKFC7V3714tWrRIwcH/3Hx9fb0WLVpkencAAAB%2Bx3jB2rx5s15%2B%2BWXdcccdnh/6HBUVpfnz52vLli2mdwcAAOB3jBesqqoqJSQkNBqPi4vT6dOnTe8OAADA7xgvWD/96U/1t7/9TZK8fvbgu%2B%2B%2Bq3/7t38zvTsAAAC/Y/xfEf785z/XI488onvvvVcNDQ169dVXVVxcrM2bN2vmzJmmdwcAAOB3jBessWPHKjQ0VK%2B//rpCQkK0fPly3XDDDVq0aJHuuOMO07sDAADwO8YL1vHjx3Xvvffq3nvvNb1pAACAgGD8HqyBAwd63XsFAABgN8YLVlpamjZt2mR6swAAAAHD%2BCXC6667Tr/97W%2B1cuVK/fSnP9VVV13lNf/888%2Bb3iUAAIBfMV6w9u/fr5/97GeSJJfLZXrzAAAAfs9YwcrPz9fixYv1pz/9yTP2yiuvaOrUqaZ2AQAAEBCM3YO1devWRmMrV640tfkf9eyzz6pLly6ePxcWFiorK0s9e/bUnXfeqXXr1nm9fvXq1RoyZIh69uyp7OxsFRcXe%2BZqa2v11FNPqV%2B/fkpLS1Nubi5n4gAAwEUxVrDO9S8Hr8S/Jty7d6/Wrl3r%2BXN5ebmmTJmicePGqbCwUDNnztTs2bNVVFQk6bsiuGTJEj333HPauXOnBgwYoMmTJ3t%2BjM/ixYvldDrlcDi0efNmWZalJ554oslzAACA5sNYwTr7g53PN2ZSQ0OD5syZo5ycHM/Y%2BvXrlZCQoKysLIWFhSk9PV2ZmZkqKCiQJDkcDo0aNUqpqakKDw/XxIkTJUnbtm1TXV2d1qxZoylTpui6665TTEyM8vLytH37dh09erRJswAAgObD%2BE3uV9Kbb76psLAwDR8%2BXC%2B%2B%2BKIkyel0Kikpyet1SUlJnkdHOJ1ODRs2zDMXHBysxMREFRUVKTExUSdPnlRycrJnvkOHDgoPD5fT6VTbtm0vaF3l5eWqqKjwGgsNban4%2BPhLyukvQkKCvX6HfwkNNX9c7HzM7Zqd3PbKLdk7e1MK2IL1zTffaMmSJV431UuS2%2B1uVIRiYmI891G53W5FR0d7zUdHR8vlcsntdkuSoqKivOajoqIu6j4sh8OhpUuXeo1NnTpVubm5F7wNfxYVdbWvl4BziI2NaLJt2/mY2zU7ue3HztmbgrGCdebMGU2bNu28Y6aegzV//nyNGjVKHTt21OHDhy/qvee7N%2Bxy7x0bO3asMjMzvcZCQ1vK5aq6rO36WkhIsKKirlZlZbXq6xt8vRx8T1N8vux8zO2andz2yi01/%2BxN%2BZfPH2OsYPXq1Uvl5eXnHTOhsLBQn332mTZs2NBoLjY21nMm6iyXy6W4uLgfnHe73erUqZPnNW63WxER/zwgJ06cUOvWrS94ffHx8Y0uB1ZUnFRdXfP44NbXNzSbLM1JUx4TOx9zu2Ynt/3YOXtTMFawvn%2BprimtW7dOx44d04ABAyT984xTWlqafvnLXzYqXsXFxUpNTZUkpaSkyOl06p577pEk1dfXa8%2BePcrKylL79u0VHR0tp9Opdu3aSZK%2B%2BOILffvtt0pJSblS8QAAQIALyDvaHn/8cW3evFlr167V2rVrPc/bWrt2rYYPH67S0lIVFBSotrZWO3bs0I4dOzRmzBhJUnZ2tt5%2B%2B23t2rVL1dXVWrZsmVq0aKH%2B/fsrJCREY8aM0fLly1VWViaXy6UXXnhBgwYNUps2bXwZGQAABJCAvMk9Ojra60b1uro6SdK1114rSVqxYoWeeeYZzZs3T%2B3atdPChQt14403SpL69eunRx99VHl5eTp27Ji6du2qlStXKjw8XJKUm5urqqoqjRw5UnV1dRowYIDmzp17ZQMCAICAFmRdiaeBQhUVJ329hMsWGhqs2NgIuVxVtrhOP/TFj3y9hIuyKe9W49u02zH/V3bNTm575Zaaf/Zrrmnlk/0G5CVCAAAAf0bBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMC/X1AgCYMfTFj3y9hAu2Ke9WXy8BAJoUZ7AAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMCxgC1ZpaammTp2qtLQ0paen6/HHH1dlZaUkae/evRo/frx69eqlwYMHa9WqVV7v3bhxo4YPH64ePXpo1KhR%2BvDDDz1zDQ0NWrx4sQYOHKjevXtrwoQJOnTo0BXNBgAAAlvAFqzJkycrKipKW7du1VtvvaUvv/xSv/vd71RTU6NJkybp5ptv1gcffKDFixdrxYoV2rJli6TvyteMGTM0ffp0ffzxx8rJydHDDz%2BsI0eOSJLeeOMNrV%2B/XitXrtS2bduUkJCgqVOnyrIsX8YFAAABJCALVmVlpVJSUjRt2jRFRETo2muv1T333KO///3v2r59u86cOaOHHnpILVu2VHJyskaPHi2HwyFJKigoUEZGhjIyMhQWFqYRI0aoc%2BfOWrdunSTJ4XAoJydHHTp0UGRkpPLz81VSUqLdu3f7MjIAAAggob5ewKWIiorS/PnzvcbKysoUHx8vp9OpLl26KCQkxDOXlJSkgoICSZLT6VRGRobXe5OSklRUVKSamhrt379fSUlJnrnIyEhdf/31KioqUvfu3S9ofeXl5aqoqPAaCw1tqfj4%2BIvK6W9CQoK9fgcuVWio/3%2BG7Pp5J7e9cktIROI1AAAPeklEQVT2zt6UArJgfV9RUZFef/11LVu2TJs2bVJUVJTXfExMjNxutxoaGuR2uxUdHe01Hx0drf379%2BvEiROyLOuc8y6X64LX43A4tHTpUq%2BxqVOnKjc39yKT%2BaeoqKt9vQQEuNjYCF8v4YLZ9fNObvuxc/amEPAF6x//%2BIceeughTZs2Tenp6dq0adM5XxcUFOT53%2Be7n%2Bpy77caO3asMjMzvcZCQ1vK5aq6rO36WkhIsKKirlZlZbXq6xt8vRwEsED4XrDr553c9sotNf/svvoLXUAXrK1bt%2Bqxxx7T7Nmzdffdd0uS4uLidPDgQa/Xud1uxcTEKDg4WLGxsXK73Y3m4%2BLiPK8513zr1q0veF3x8fGNLgdWVJxUXV3z%2BODW1zc0myzwjUD6/Nj1805u%2B7Fz9qYQsBdcP/30U82YMUMvvfSSp1xJUkpKivbt26e6ujrPWFFRkVJTUz3zxcXFXts6Ox8WFqZOnTrJ6XR65iorK/X111%2BrW7duTZwIAAA0FwFZsOrq6jRr1ixNnz5dffv29ZrLyMhQZGSkli1bpurqau3evVtr1qxRdna2JGnMmDHauXOntm/frtraWq1Zs0YHDx7UiBEjJEnZ2dlavXq1SkpKdOrUKS1atEiJiYnq2rXrFc8JAAACU0BeIty1a5dKSkr0zDPP6JlnnvGae/fdd7V8%2BXLNmTNHK1euVJs2bZSfn6/%2B/ftLkjp37qxFixZp/vz5Ki0tVceOHbVixQpdc801kqRx48apoqJCv/jFL1RVVaW0tLRGN6wDAAD8mCCLJ2heERUVJ329hMsWGhqs2NgIuVxVtrhOP/TFj3y9hGZrU96tvl7Cednt834Wue2VW2r%2B2a%2B5ppVP9huQlwgBAAD8GQULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGhfp6AQDsZ%2BiLH/l6CRdlU96tvl4CgADDGSwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGGhvl4A7GPoix/5egkAAFwRnMECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAw/hXhABwHoH2L2A35d3q6yUAtscZLAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwCtY5lJaW6sEHH1RaWpoGDBighQsXqqGhwdfLAgAAAYIHjZ7DI488ouTkZL333ns6duyYJk2apDZt2ujf//3ffb00AAAQAChY31NUVKTPP/9cr776qlq1aqVWrVopJydHr732GgULQEAIpCfP89R5NFcUrO9xOp1q166doqOjPWPJyck6cOCATp06pcjIyPNuo7y8XBUVFV5joaEtFR8fb3y9gxZ9YHybAHClBFIZlKT/nn6br5dgXEhIsNfvMIOC9T1ut1tRUVFeY2fLlsvluqCC5XA4tHTpUq%2Bxhx9%2BWI888oi5hf5/f//tHca3%2BUPKy8vlcDg0duzYJimL/syu2e2aW7JvdnLbK7f0XfbXXvuDLbM3JerqOViWdVnvHzt2rN566y2vX2PHjjW0Ot%2BpqKjQ0qVLG52dswO7Zrdrbsm%2B2cltr9ySvbM3Jc5gfU9cXJzcbrfXmNvtVlBQkOLi4i5oG/Hx8fwtAAAAG%2BMM1vekpKSorKxMx48f94wVFRWpY8eOioiI8OHKAABAoKBgfU9SUpK6du2q559/XqdOnVJJSYleffVVZWdn%2B3ppAAAgQITMnTt3rq8X4W9uu%2B02bdiwQU8//bTeeecdZWVlacKECQoKCvL10nwuIiJCffr0seXZPLtmt2tuyb7ZyW2v3JK9szeVIOty7%2BgGAACAFy4RAgAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwcIPKioq0qBBgzRmzJhGc4WFhcrKylLPnj115513at26dV7zq1ev1pAhQ9SzZ09lZ2eruLj4Si3bqNLSUj344INKS0vTgAEDtHDhQjU0NPh6WcZ88MEHSk9PV35%2BfqO5jRs3avjw4erRo4dGjRqlDz/80DPX0NCgxYsXa%2BDAgerdu7cmTJigQ4cOXcmlX5bS0lJNnTpVaWlpSk9P1%2BOPP67KykpJ0t69ezV%2B/Hj16tVLgwcP1qpVq7ze%2B2NfF3/3%2Beef6/7771evXr2Unp6uvLw8VVRUSLLP9/Szzz6rLl26eP7cnHN36dJFKSkp6tq1q%2BfX008/Lal55/YbFnAOa9eutTIyMqwJEyZYo0eP9po7evSo1b17d6ugoMCqqamxPvroI6tbt27W//7v/1qWZVnvv/%2B%2BddNNN1m7du2yqqurrRUrVli33nqrVVVV5Ysol%2BWee%2B6xZs2aZVVWVloHDhywBg8ebK1atcrXyzJi5cqV1uDBg61x48ZZeXl5XnN79uyxUlJSrO3bt1s1NTXW2rVrrdTUVKusrMyyLMtavXq1NWDAAGv//v3WyZMnrd/85jfW8OHDrYaGBl9EuWh33XWX9fjjj1unTp2yysrKrFGjRllPPvmkVV1dbd12223WkiVLrKqqKqu4uNjq06ePtXnzZsuyzv918We1tbXWLbfcYi1dutSqra21jh07Zo0fP96aMmWKbb6n9%2BzZY/Xp08fq3LmzZVnN/79lnTt3tg4dOtRovLnn9hecwcI51dbWyuFwKDU1tdHc%2BvXrlZCQoKysLIWFhSk9PV2ZmZkqKCiQJDkcDo0aNUqpqakKDw/XxIkTJUnbtm27ohkuV1FRkT7//HNNnz5drVq1UkJCgnJycuRwOHy9NCPCwsK0Zs0aXX/99Y3mCgoKlJGRoYyMDIWFhWnEiBHq3Lmz52%2B5DodDOTk56tChgyIjI5Wfn6%2BSkhLt3r37Sse4aJWVlUpJSdG0adMUERGha6%2B9Vvfcc4/%2B/ve/a/v27Tpz5oweeughtWzZUsnJyRo9erTnmJ/v6%2BLPqqurlZ%2Bfr0mTJqlFixaKi4vToEGD9OWXX9rie7qhoUFz5sxRTk6OZ8wOuc/FrrmvNAoWzmn06NFq27btOeecTqeSkpK8xpKSkjynkL8/HxwcrMTERBUVFTXdgpuA0%2BlUu3btFB0d7RlLTk7WgQMHdOrUKR%2BuzIz77rtPrVq1OufcDx3joqIi1dTUaP/%2B/V7zkZGRuv766wPiGEdFRWn%2B/Plq06aNZ6ysrEzx8fFyOp3q0qWLQkJCPHM/9tk%2BOx8IuaOjozV69GiFhoZKkr766iv913/9l4YOHWqL7%2Bk333xTYWFhGj58uGfMDrmff/559e/fXzfddJNmz56tqqoqW%2BT2BxQsXDS3262oqCivsZiYGLlcLs/8v5YS6bv/uJ%2BdDxTnynk2V6BluVg/dgxPnDghy7KaxTGWvjtT%2Bfrrr%2Buhhx76wc%2B22%2B1WQ0NDs/hsl5aWKiUlRcOGDVPXrl2Vm5vb7L%2Bnv/nmGy1ZskRz5szxGm/uubt376709HRt2bJFDodDu3bt0rx585p9bn9BwbKptWvXqkuXLuf89dZbb1329i3LMrBK32suOS7F%2BbI3h6/NP/7xD02YMEHTpk1Tenr6D74uKCjI878DPXe7du1UVFSkd999VwcPHtSvf/3rC3pfIOeeP3%2B%2BRo0apY4dO170ewM5t8Ph0OjRo9WiRQt16NBB06dP14YNG3TmzJnzvjeQc/uLUF8vAL4xcuRIjRw58pLeGxsbK7fb7TXmcrkUFxf3g/Nut1udOnW6tMX6SFxc3DlzBAUFebI2Vz90DOPi4hQTE6Pg4OBzzrdu3fpKLvOybN26VY899phmz56tu%2B%2B%2BW9J3x/zgwYNer3O73Z7MP/Z1CSRBQUFKSEhQfn6%2Bxo0bp4yMjGb7PV1YWKjPPvtMGzZsaDRnl/%2BWnfWTn/xE9fX15/z%2Bbc65fYUzWLhoXbt2bfRPdouLiz03xKekpMjpdHrm6uvrtWfPnnPeMO/PUlJSVFZWpuPHj3vGioqK1LFjR0VERPhwZU0vJSWl0TEuKipSamqqwsLC1KlTJ69jXFlZqa%2B//lrdunW70ku9JJ9%2B%2BqlmzJihl156yVOupO9y79u3T3V1dZ6xs7nPzv/Q18XfFRYWasiQIV6PGQkO/u7/Arp169Zsv6fXrVunY8eOacCAAUpLS9OoUaMkSWlpaercuXOzzb1nzx4tWLDAa6ykpEQtWrRQRkZGs83tV3z27xcREF5%2B%2BeVGj2n45ptvrB49elh//vOfrZqaGmv79u1Wt27drL1791qWZVk7duywevXqZX322WfW6dOnrSVLllgZGRlWdXW1LyJcltGjR1tPPvmkdfLkSWv//v1WZmam9frrr/t6WUbNmDGj0WMa9u3bZ3Xt2tXatm2bVVNTYxUUFFg9evSwysvLLcuyrP/8z/%2B0%2Bvfv73lMw%2BzZs617773XF8u/aGfOnLGGDh1qvfnmm43mamtrrQEDBlgvv/yydfr0aWvXrl3WTTfdZG3bts2yrPN/XfxZZWWllZ6ebi1YsMA6ffq0dezYMWvChAnWz3/%2B82b9Pe12u62ysjLPr88%2B%2B8zq3LmzVVZWZpWWljbb3EeOHLG6d%2B9urVixwqqtrbW%2B%2Buora9iwYdbTTz/drI%2B3P6Fg4ZwGDx5spaSkWImJiVaXLl2slJQUKyUlxTp8%2BLBlWZb1ySefWCNGjLCSk5OtwYMHe54TdNYbb7xhZWRkWCkpKVZ2dra1b98%2BX8S4bGVlZdbEiROtbt26Wenp6dbLL78cMM96Op%2Bzx/TGG2%2B0brzxRs%2Bfz9q8ebM1ePBgKzk52Ro5cqT1ySefeOYaGhqsl156ybrlllusbt26WQ888EBAPAvKsizrf/7nf6zOnTt78v7rr8OHD1v79u2zxo0bZ6WkpFj9%2B/e33njjDa/3/9jXxd99/vnn1vjx461u3bpZN998s5WXl2cdOXLEsiz7fE8fOnTI8xwsy2reuT/55BNr7NixVvfu3a0%2BffpY8%2BfPt2pqajxzzTW3vwiyLO5kAwAAMIl7sAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAsP8HkT6iA3me6AsAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common8538788201463328341">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">1890</td>
        <td class="number">12.5%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">217</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">206</td>
        <td class="number">1.4%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">200</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8</td>
        <td class="number">198</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7</td>
        <td class="number">182</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">176</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9</td>
        <td class="number">166</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">165</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">162</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (413)</td>
        <td class="number">11558</td>
        <td class="number">76.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme8538788201463328341">
            <p class="h4">Minimum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">-146</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-134</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-123</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-115</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-114</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">401</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">403</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">411</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">547</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">554</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Wilderness_Area1">Wilderness_Area1<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-5596561357875994416">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 76.2%">
            11523
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:31%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 23.8%">
            3597
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-5596561357875994416, #minifreqtable-5596561357875994416"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-5596561357875994416">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">11523</td>
        <td class="number">76.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">3597</td>
        <td class="number">23.8%</td>
        <td>
            <div class="bar" style="width:31%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Wilderness_Area2">Wilderness_Area2<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-237032388942349802">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 96.7%">
            14621
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:4%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 3.3%">
            &nbsp;
        </div>
        499
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-237032388942349802, #minifreqtable-237032388942349802"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-237032388942349802">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">14621</td>
        <td class="number">96.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">499</td>
        <td class="number">3.3%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Wilderness_Area3">Wilderness_Area3<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-8811097758180248530">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 58.0%">
            8771
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:72%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 42.0%">
            6349
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-8811097758180248530, #minifreqtable-8811097758180248530"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-8811097758180248530">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">8771</td>
        <td class="number">58.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">6349</td>
        <td class="number">42.0%</td>
        <td>
            <div class="bar" style="width:72%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_Wilderness_Area4">Wilderness_Area4<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="">
            <th>Distinct count</th>
            <td>2</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable4575659734140361982">
    <table class="mini freq">
        <tr class="">
    <th>0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 69.1%">
            10445
        </div>

    </td>
</tr><tr class="">
    <th>1</th>
    <td>
        <div class="bar" style="width:45%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 30.9%">
            4675
        </div>

    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable4575659734140361982, #minifreqtable4575659734140361982"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable4575659734140361982">

<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0</td>
        <td class="number">10445</td>
        <td class="number">69.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">4675</td>
        <td class="number">30.9%</td>
        <td>
            <div class="bar" style="width:45%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div>
    <div class="row headerrow highlight">
        <h1>Correlations</h1>
    </div>
    <div class="row variablerow">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAxMAAALICAYAAAAE1K0IAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3XlcVdX6x/HPYUZxIgFNGUxzRsOcM03UNM1Q0zTN6jplzkNdG%2BynJrdRbuY8dB0r86allXVLNBtt1ApNrMQUJ8AEFQSZzu8P4sRRVGRtBeP7fr18Hdx7r%2Bc8Z51pP3vttY/NbrfbERERERERuUwuJZ2AiIiIiIhcm1RMiIiIiIhIsaiYEBERERGRYlExISIiIiIixaJiQkREREREikXFhIiIiIiIFIuKCRERERERKRYVEyIiIiIiUiwqJkREREREpFhUTIiIiIiISLG4lXQCIiIi9erVu%2BA6d3d3KleuTOPGjenVqxddu3bFZrNdxexERORCbHa73V7SSYiISNmWX0w0adIEPz8/p3Vnz54lLi6OI0eOANCxY0fmzJmDh4fHVc9TREScqZgQEZESl19MzJ8/n86dOxe6TXR0NI8%2B%2BihnzpxhyJAhTJky5WqmKCIihdCcCRERuSZ07tyZyZMnA7BmzRrOnj1bwhmJiIjmTIiIyDXj9ttvZ%2BbMmaSnp/Pzzz8TFhbmtH779u289tpr/PDDD6SkpODj40P9%2BvW5%2B%2B676dmzZ6ExU1JSWL58OZ988gkHDhwgMzOTSpUq0aRJE/7xj3/QqlWr89qEh4dz%2BPBhli5dytmzZ3nppZeIj49nxYoV3HzzzQD88ssvLFu2jO%2B%2B%2B46EhATc3Nzw9/enZcuW3HfffYXOEzl9%2BjQrV65ky5YtHDhwgKysLKpWrUqzZs148MEHCQ0Nddr%2B0KFDdOrUCYCffvqJffv2sXDhQnbu3ElKSgr%2B/v506tSJiRMnUq5cuWL1uYjIxWhkQkRErhmVK1d2/H3q1CmndS%2B99BIPPvggmzdvxsfHh1atWuHr68v27dt55JFHmDBhAjk5OU5tkpKS6NOnD4sWLSIuLo6GDRvSunVrPDw8%2BPjjj3nggQdYv379BfOJi4tj4sSJuLu707p1a7y8vAD45ptvuPvuu3n77bc5e/YsN998M2FhYaSmpvLf//6Xe%2B65h%2B3btzvFOnz4ML1792bu3Ln89ttv1K9fn9atW2O323nvvfe45557WLdu3QVz2bFjBwMHDmT37t3Ur1%2BfWrVqcfjwYVatWsXYsWOL3MciIpfFLiIiUsLq1q1rr1u3rn3z5s0X3W7fvn2ObX/66SfH8o8%2B%2Bshet25de4sWLexff/21U5tvvvnG3rZtW3vdunXty5cvd1oXGRlpr1u3rr1r167248ePO5ZnZ2fb//Wvf9nr1q1rv/nmm%2B2pqalO7Tp27GivW7euPTw83L5w4cLz8uzXr5%2B9bt269ueee86ek5PjFPell16y161b137nnXc6tRk4cKC9bt269j59%2BtgTEhIcy3Nycuxz5861161b1964cWN7XFycY118fLyjP2677Tb7kiVL7Lm5uY71b7/9tmP9nj17Lta1IiLFopEJERG5Znz00UcAVKpUiQYNGjiWL1y4EIApU6bQsmVLpzYtWrTgn//8JwArV650Wufv70%2BPHj0YM2YM1113nWO5q6srEydOxMXFhdOnT/PDDz8Umk9ubi4jRow4b/mePXsA6NOnDy4uf33Vurq6Mm7cOMaNG8fgwYPJzMwE4Mcff%2BS7774D4IUXXsDf39/RxsXFhTFjxtCoUSMyMzN54403Cs3lxhtvZPjw4U6XzY2IiMDX19dxHyIiVtOcCRERuSZ88sknLF68GIBhw4bh5pb3FZaQkMDu3bsB6NKlS6FtO3fujM1m48iRI%2Bzfv59atWoBMHz48Aven7e3N9dddx1JSUkkJSUVuk3r1q2dioV8FStW5Pjx47z//vuMHz/eaZ2LiwujR492WvbZZ58BeVe1ql279gUfw%2B7du887PSpfYXNCbDYbgYGBnDhxguTk5ELbiYiYUDEhIiKlxuLFi3nrrbeclmVmZvL7778THx8P5B3tHzZsmGP9L7/84vj7scceu2BsNzc3srKy%2BP333x3FRH78Tz75hF27dpGYmMipU6ew/3nV9NOnTwN5IxCFCQgIKHT5Aw88QFRUFAsWLOCzzz7jzjvvpE2bNhf8cb7ffvsNyBtduJAbbrgByJunUZigoKBCl3t6egKQlZV1wdgiIsWlYkJEREqNn3766bxlrq6uVKlShU6dOtG/f386dOjgtP7kyZOOv7ds2XLJ%2B8gvEABiY2MZPXo0hw4dKla%2BlSpVKnR5/qlPixcvJiYmhpiYGAD8/Pzo1q0bDzzwAIGBgeflVLFixQveV4UKFYC8oiAjI8Mx2TuffsRPREqCigkRESk1LvajdReSP0fA3d2dmJgYpzkDF5ORkcHDDz/MkSNHCAkJYdSoUbRp04YqVarg7u4O/HUJ2Evdd2FGjBjBgAEDiI6O5pNPPuHLL78kKSmJ1atXs3btWl544QXuuOOOIj9Oe4HfmC3s1CoRkZKgYkJERK5p%2BZeLzcrKIjk52THh%2BFI%2B/fRTjhw5gs1mY8mSJQQHB5%2B3TUZGhlFuFStWpE%2BfPvTp04ecnBy%2B/PJL5s%2Bfz86dO3nyySdp3bo1VapUcYxwnHu524Ly13l5eWkUQkRKDR3aEBGRa1rdunUdf//6669Fbvf7778DeXMNCiskDh48yB9//GGcXz5XV1duvfVWVq1aRUBAAGlpaY7TuvLnShSc/3Gu/HUXm1chInK1qZgQEZFrmp%2BfH40aNQLgv//9b6Hb7N%2B/n4iICMclZOGvOQhnz54ttM38%2BfMdf5/7Y3cX8%2BOPP/LEE0%2BwbNmyQtd7eHhQtWpV4K%2BRj/x5IL/88kuhBUVubq7jsri33nprkXMREbnSVEyIiMg1b%2BTIkQC89957rFixwml%2BwYEDBxg3bhyxsbEcPXrUsbx%2B/foAHDt2jOjoaMfy9PR0IiMjiYmJISwsDLjwFZQK4%2Brqyvr165kzZw7btm07b/3mzZvZs2cP7u7uNGvWDICGDRvSrl07AJ544glOnDjh2D47O5sXX3yRffv2UaFCBQYMGFDkXERErjTNmRARkWve7bffzogRI1iyZAnPPvssq1at4oYbbiA5OZk9e/aQk5NDw4YNeeSRRxxtwsLCaNeuHZ9//jljx46ladOmeHp6smvXLtzc3Fi%2BfDmbNm1i586drFq1ir179/Lwww/TokWLi%2BbSuHFjhgwZwrJly3jooYeoUaMGwcHBuLi4cOjQIcfpVU8%2B%2BSR%2Bfn6Ods888wwPPPAAMTExhIeHU7duXTw9Pfn1119JTk7Gy8uLqKioC16OVkSkJKiYEBGRv4XJkyfTtm1bXn31VX744Qe2b9%2BOp6cnjRs3pnv37gwcOPC8icuzZ89m1qxZbNmyhV27duHv70/Xrl0ZOXIkQUFBVKtWjZ9//pkdO3bw66%2B/FvlKUVOmTKFFixZs2LCBXbt2sXPnTrKzs/Hz8%2BPOO%2B/kvvvuc4x65AsICGDdunWsWLGCzZs38%2Buvv5KdnU1AQAC33347Q4cOLXRuh4hISbLZC44Fi4iIiIiIFJHmTIiIiIiISLGomBARERERkWJRMSEiIiIiIsWiYkJEREREpBT57LPPaNu2LRMnTrzodrm5ubz00kt06tSJFi1aMHToUOLj4x3rU1JSmDBhAm3btqVdu3Y8%2BeSTjt%2B3sYqKCRERERGRUmLp0qVERkYW6eptr732Gu%2B%2B%2By5Llizh448/JiQkhNGjRzt%2Ba%2Bepp54iPT2d9957j/Xr17Nv3z5mzZplab4qJkRERERESglPT0/WrVtXpGJi7dq1PPjgg9SuXRsfHx8mTpzIvn37%2BPHHHzl%2B/DjR0dFMnDgRX19fAgICGDVqFOvXrycrK8uyfPU7EyIiIiIiFklMTCQpKclpmZ%2BfH/7%2B/kVqf//99xdpu4yMDH777TcaNmzoWObj40NwcDAxMTGcPn0aV1dX6tWr51jfqFEjzpw5Q1xcnNNyEyomRMqSIv7g1gXVqgW//go33gj795vF%2BuEHs/YAHh5Qvz7ExkJmplmsAh/GxebmBtnZxmGycC8tqeB%2B/KhZAFdX8PODpCTIyTFPqFw5s/Y2G1SoAKdPg%2BHPLKV7VDJOxdMTzp41TgVvr1L2k1EnTpi1d3GBypUhJQVyc81ieXmZtbfZwNsb0tONn6jkzPJG7V1coGJFOHXKvFsAqpw5XPzGrq4QEAAJCda8t2vUMI9RHKbfi4VYO2cO8%2BbNc1o2ZswYxo4da%2Bn9nDx5ErvdTqVKzp9FlSpVIjk5mcqVK%2BPj4%2BP0Y5v52yYnJ1uWh4oJESm6ypXzvkAqVy7pTPK4uuZ9Ebi6lnQmea7Al1JxlZpUbLa//pUGBfMpBb/ZWlq6BSg1fQKUrtdNKXrNlKZuwcUlLxEXF2uKib%2BR/v37Ex4e7rTMz8/vit3fxX5/%2Bmr8NrWKCREREREpm1ysnz7s7%2B9f5FOaTFSuXBkXFxdSUlKclqekpHDdddfh6%2BtLamoqOTk5uP550C1/2%2Buuu86yPDQBW0RERETkGuPp6cmNN97I7t27HctOnTrFwYMHadKkCQ0aNMButxMbG%2BtYHxMTQ8WKFalVq5ZleaiYEBEREZGyycXF%2Bn9XUEJCAt26dXP8lsS9997LqlWr2LdvH6mpqcyaNYsGDRoQGhqKr68vXbt2Zfbs2Zw4cYJjx44xf/58%2Bvbti5ubdScn6TQnERERESmbrvDOf3GEhoYCkP3nVTSio6OBvFGFrKws9u/fT%2BafFx0ZMGAASUlJDB48mLS0NFq1auU0%2Bfvpp59m2rRpdOrUCXd3d%2B68885L/hDe5VIxISIiIiJSSsTExFxwXc2aNdm7d6/j/zabjXHjxjFu3LhCt69QoQL//ve/Lc%2BxIBUTIiIiIlI2lcKRiWuNelBERERERIpFIxMiIiIiUjZpZMKYigkRERERKZtUTBhTD4qIiIiISLGomBC5BsyaNYvBgweXdBoiIiJ/L9fY70yURmXvEYuUUoMHD2bWrFklnYaIiIhIkWnOhIiIiIiUTWVwJMFqKiZESqGtW7fy/PPPk5iYSIcOHahatWpJpyQiIvL3o2LCmHpQpJQ5deoUEydO5L777uPrr7%2Bmd%2B/ebNiwoaTTEhERETmPRiZESpnPP/%2BccuXKMWjQIFxcXOjQoQPNmzcnLS3tsuIkJiaSlJTktMyvVi38K1cufnL16zvfmvD2No/h6el8K9ZzM/yayG9vGief6VHE/PYWHI202axpbxqnVHJ1taa9aRwwf64LPlGGsUwfjoUv3zzu7sVva%2BV7OyvLPEZxaWTCmIoJkVLm2LFjVK9eHZcCH3AhISHs3r37suKsXbuWefPmOS0bM348Y8ePN0/y9dfNY1gpOLikM/iLyZdzfggL0gBLUgE/PwuCAFWqWBPHKuXLG4fwsiANsKoWtqgisaqyMTloUVCFCtbEsYKX%2BTNe0YJjKAA%2BPtbEoaK/eQxfX/MYhw%2Bbx5ASo2JCpJTJzMwkJyfHaVlubu5lx%2Bnfvz/h4eFOy/x69oSVK4ufXP36eYXEwIEQG1v8OABvvGHWHvL2woKD4cABOHvWLFatWub5uLtbcoQty4JywqJUcE9JuvRGF%2BPmlldIJCdDdrZ5QqY7dC4ueYVEWhoU431VUIa72Y6uzZb3Ej57Fux2o1B4eRoGyE/INJF8J0%2BatXd1zSskTp%2BGcz4PL5tptWaz5b3uMjKM%2B%2BdUllk14eKSV0ikphq/fAGomJFY/MZubnmFxIkT1ry3S4pGJoypmBApZfz9/UlISMBut2P78yjhvn37ihXH3/%2Bco07791uRYl4hsXOnWYz0dGtygby9MSvjyV%2Bs2knIzrYmlhV7UPlxDGNZtd9tt1sXq9QwLQAKxjGNZfqayd/ZtNuNY1nVLbm5FsWy4ohDdnbJnqZkSsWEMfWgSCnTtm1bUlNTeeONN8jMzCQ6Opoff/yxpNMSEREROY%2BKCZFSplq1akRFRbFs2TJatmzJO%2B%2B8w8CBA0s6LRERkb8f/QK2MZ3mJFJKrF692vF3165d6dq1awlmIyIiInJpKiZEREREpGwqgyMJVlMxISIiIiJlk4oJY%2BpBEREREREpFo1MiIiIiEjZpJEJY%2BpBEREREREpFo1MiIiIiEjZpJEJYyomRERERKRsUjFhTD0oIiIiIiLFopEJERERESmbNDJhTD0oIiIiIiLFopEJkbLkhx/M2nt7592%2B8Qakp5vFuukms/YAYWGwYwcMGAA7dxqFysm2G6fjCuS4uBvHee8ds/aVKkF4OHz2GZw8aRar97ENZgGqVoV%2B/WDbNjh%2B3CwWQPfuZu09PKBCBUhNhcxMo1Dezz9ulktgIDz%2BOF4vPQvx8WaxnnrKrL2bG/j55T1H2dlmsQAOHjRrX64cVK6c1y9nzpjFCgkxa%2B/mlvfZd%2BaMcd9UeX2JWS7%2B/jBoEBXffQ0SE81iAYwZYx6jShXzGCVJIxPGVEyIiIiISNmkYsKYelBERERERIpFIxMiIiIiUjZpZMKYigkRERERKZtUTBhTD4qIiIiISLFoZEJEREREyiaNTBhTD4qIiIiISLFoZEJEREREyiaNTBhTMSEiIiIiZZOKCWPqQRERERERKRaNTIiIiIhI2aSRCWPqQRERERERKRaNTIiIiIhI2aSRCWPqQSlzwsPDWbNmzVW9zw0bNhAeHn5V71NEREQuwcXF%2Bn9ljEYm5G8nPDychIQEXAp5Qz/77LNXLY9169YRHh6Or68vvXr1olevXlftvkVEROTadPjwYWbMmMGPP/5IuXLl6N69O5MnTz5vv2bIkCF8%2B%2B23Tsuys7MZPXo0Y8aMYfDgwezYscOpXa1atXjnnXcszVfFhPwtTZ06lXvvvbfQdf/%2B97%2Bv%2BP3n5OTw3HPPERYWhq%2Bv7xW/PxERESmGUjiSMHbsWBo1akR0dDR//PEHDz30EFWrVuUf//iH03bLli1z%2Bv%2BpU6fo3r07Xbp0cSybOXMmffr0uaL5lr4eFLmKcnNzmTNnDp07d6Zp06bcfffdfP/99wBMmDCBxx9/3Gn7FStWcMcddwBw8OBBhg4dSqtWrWjVqhWTJk3i1KlTALRs2ZLTp08TERHBvHnzeOutt7jlllsccX799Vfuv/9%2BmjdvTqtWrZg2bRpnz54F4K233uKuu%2B5ynBoVFhbGxIkTycrKuhpdIiIiIiUkJiaG2NhYHnnkESpUqEBISAgPPvgga9euvWTb2bNn06VLF%2BrVq3cVMv2LRiakTFu5ciWbNm3ilVde4frrr2ft2rU8/PDDbNu2jW7dujF9%2BnRycnJwdXUFYPPmzXTv3h3IG/2oUaMGn332GampqQwdOpQFCxbw2GOPsXHjRjp16sTGjRupXbs2b731luM%2BMzMzGTJkCL169WLJkiUkJiYycuRIXn75Zf75z38CeUOcu3bt4r333uPw4cP06dPH6b6LIjExkaSkJKdlfqmp%2BPv5Fb/DPD2db02EhZnHqF/f%2BfZvolIls/Y%2BPs63RrKrmrWvXNn51pSHh1l7NzfnWxOBgWbtAwKcb02YPh4r%2BwWgXDmz9l5ezrcmTB/Tn5//jlsT/v5m7atUcb4Vc1dgZKLQ718/P/yL8Pzv3r2bGjVqUKnAF0GjRo3Yv38/qamp%2BFzgg/3AgQNs2LCB6Ohop%2BXvv/8%2Br7zyCkePHqVp06Y8/fTTBAUFFeNRXZiKCflbioyM5JlnnnFaVq5cOb7%2B%2BmunZevWrePBBx8kJCQEgMGDB7Ny5Uq2bdtGeHg4Z8%2Be5fvvv6dly5b88ccf7Nixg6effhqAJUuWYLPZ8PDwwNfXl1tvvZUdO3ZcMrdPP/2U9PR0xo4di4eHB0FBQQwaNIhXXnnFUUykpaUxYcIEypUrx4033ki9evWIi4u7rD5Yu3Yt8%2BbNc1o2ZvRoxo4bd1lxChUcbB6jCH1VZK%2B/bhzCgt2EvDgWBLJqrn7LllZE6WdFECgw7F4qmBTV%2Bc4ZuSy2IUOsiWMFq3ZSrehfgBtvtCaOFawoiAcNMo8BcBkHlq44d3fzGCU58n4FiolCv3/HjGHs2LGXbJuSkkLFihWdluUXFsnJyRcsJpYsWcLdd9/tdGp17dq18fb2ZtasWeTm5hIZGcmwYcN477338DA9MFOAign5W7rYnImCDh48yL/%2B9S%2BnwiM3N5ejR4/i5eVFhw4diI6OpmXLlmzdupUbb7yR2rVrA7Br1y6ioqLYu3cvWVlZ5OTk0Lhx40ve56FDhwgMDHR6IwcHB3PkyBFyc3MBqFKlitMHhre3NxkZGUV%2B/AD9%2B/c/7wpSfikp8MsvlxXHiadnXiFx4AD8eVpWsQ0YYNYe8kYkXn8dBg6E2FijUDnfmhc3rq6Qk2Mchk8%2BMWvv45NXSHzzDaSmmsUK/%2BNNswCVK%2BcVEps3Q0qKWSyAdu3M2ru55e3oJiVBdrZZrBUrzNoHBOQVEsuWQUKCWaxhw8zau7nlFRLJyeb9AnDsmFl7L6%2B8QuLXX%2BEyP/vOc/31Zu1dXfNexykp5m/w//3PrH2VKnmFxPvv5z1Xpu65x6y9u3vJFgKlVKHfv5dRYNvt9su6v5SUFDZu3MgHH3zgtHz69OlO/3/66adp1aoV33//PW3atLms%2B7gYFRNSpnl5eREZGUnXrl0LXX/HHXfwwgsv8MQTT/DRRx85TjM6efIkI0aM4N5772Xp0qX4%2BPgwe/Zsvvzyy0veZ2ZmZqHLbTab4%2B/CrkR1ufz9/c8fUv3xR0hPN47N2bPmcXbuNM8jX2ystfFK2MmT1sRJTbUg1vHjluRCSoo1sS7w/rls2dnmseLjrcklIcE8lhUFQH4cK2KdOWMeA/IKCdNYVvVNTo55rMREa3JJTrYuVll3BUYmCv3%2BLSJfX19SzjnwkpKSgs1mu%2BAFXbZs2UKtWrUIvMSplz4%2BPlSqVIkE04MX59AEbCnTAgMD2bt3r9OyQ4cOOf7u0KEDJ06cYMeOHXz11VeOYiIuLo60tDSGDh3qGEH4%2Beefi3yf8fHxTkVFXFwcNWvWtKSIEBERkWtT48aNOXr0KCdOnHAsi4mJoU6dOpQvX77QNlu2bHG6yAtAamoq06dPdyocTpw4wYkTJy5ZdFwu7blImTZgwABee%2B01fvjhB3Jycnj//fe58847OXLkCJA3cnHbbbcRFRVF3bp1HZOWrr/%2BelxcXNi5cydnzpxhxYoVHD9%2BnOPHj5OdnY3Xn5MGf//9d1LPOc%2Bkffv2uLm5MX/%2BfDIzM4mLi2PVqlX6HQoREZGrrZT9aF3Dhg0JDQ0lKiqK1NRU9u3bx/Llyx2nbnfr1o3vvvvOqc2ePXuoWbOm0zIfHx9%2B/PFHIiMjSUlJ4eTJk8yYMYN69eoRZsUFUApQMSF/S5GRkYSGhp7379xLvfbt25eBAwcyZswYbr75Zl555RXmzZvH9QXOsc1/4/bo0cOxLCAggEmTJvHEE0/QsWNHTp48yaxZs8jMzGTgwIFUrVqVrl27Mn78eGbPnu10n%2BXLl2fJkiV8%2B%2B23tGnThuHDhxMREcHIkSOvbKeIiIiIs1JWTADMmTOHxMREbrnlFu6//3569erFwIEDAdi/fz9nzjn1LykpiapVz7/y3vz587Hb7XTt2pXbbruNrKwslixZYvlZEJozIX87W7duLfK2Li4ujB8/nvHjx19wm65du553KhTA8OHDGT58uNOyzz//3PH3nDlznNYV/NGYJk2a8PoFrkDUp0%2Bf835gZvXq1Rd%2BECIiIvK3Ua1aNZYuXVrousL2R3bt2lXottdff/15V5W6ElRMiIiIiEjZpLmKxtSDIiIiIiJSLBqZEBEREZGySSMTxlRMiIiIiEjZpGLCmHpQRERERESKRSMTIiIiIlI2aWTCmHpQRERERESKRSMTIiIiIlI2aWTCmIoJERERESmbVEwYUw%2BKiIiIiEix2Ox2u72kkxCRqyQryzyGu7slcXJc3M1zAVxdISfHgjhuNrMAYWGwYwc0awY7d5rFysgwa2%2BzgYcHZGaC6Ud8drZZexcX8PaG9HTIzTWLZVU%2BFSrA6dPm%2BZi2d3WFihXh1CnjF/FDj1Uxah8YCFOnQmQkxMcbhQJg8e3rzQJUrgydOsGWLZCSYhZrwwaz9sHBeR0zdSocOGAU6vSC1UbtXVygfHlIS7Pm7VShnOGHp1UfwPmxSkK/ftbHfPNN62OWYhqZEBERERGRYtGcCREREREpmzRnwpiKCREREREpm1RMGFMPioiIiIhIsWhkQkRERETKJo1MGFMPioiIiIhIsWhkQkRERETKJo1MGFMxISIiIiJlk4oJY%2BpBEREREREpFo1MiIiIiEjZpJEJYyomRERERKRsUjFhTD0oIiIiIiLFopEJERERESmbNDJhTD0oUojPP/%2BcevXqMWPGjJJOhZycHJYvX17SaYiIiIicR8WESCHefPNNevTowaZNmzh79myJ5vLzzz/zyiuvlGgOIiIif0suLtb/K2PK3iMWuYTk5GS2bt3KuHHjqFKlCps3b3as27ZtGz179iQsLIx27drx4osvkpubC0C9evV466236Nu3L02aNKFXr17ExcU52sbGxvLAAw/QvHlzWrduTWRkJFlZWY71GzdupGvXroSFhTFgwAD27NnDTz/9xIABAzh%2B/DihoaF89dVXV68jRERE/u5UTBjTnAmRc2zcuJEGDRoQEhJCz549WbduHXfeeSdZWVlMnDiR%2BfPn06ZNGw4cOMCwYcMICwujc%2BfOACxfvpzZs2dTrVo1pk2bxqRJk9iwYQPp6ekMGzaMwYMHs3TpUhISEhg1ahT/%2Bc9/GDlyJLt27WL69OksXLiQm2%2B%2BmcWLFzNq1Ciio6OZOXMmUVFRfPHFF5f1OBITE0lKSnJa5lelCv5%2Bfpb11d9KWJhZ%2B/r1nW9N2GzWtDeNA%2BZfjAVzseJL1jRGfnsrcjHtXwtzCQw0a1%2BtmvOtscqVzdpXqOB8ayI42Kx99erOtwZK08tXxCoqJkTOsW7dOu69914AIiIimD9/PocOHaJy5cpkZGRQrlw5bDYbISEhfPTRR7gU%2BFSPiIigdu3aAAwbNoyIiAjKHztSAAAgAElEQVQSEhLYsWMHdrudhx56CIDAwECGDh3K4sWLGTlyJBs2bKB169a0bt0agKFDh1KrVi2jU6zWrl3LvHnznJaNGT2asePGFTumg7u7cQhX8yz%2BimVFsB07LAgCvP66NXGsYMHzZBkvr5LOwFn58iWdwV98fIxDTJ1qQR7AsGHWxIFO1oRp2dI8RieLchk92jiEVa86b2%2BLAlnxSWzFB3BOjnmM4lJlZkzFhEgBP/zwA7///jt33HEHkLfTf9NNN/HWW28xbtw4Ro8ezX333UeTJk245ZZb6NOnD9ULHK2qVauW4%2B8aNWoAkJCQQHx8PH/88QehoaGO9Xa7HQ8PDwDi4%2BMJCgpyrPP29qZHjx5Gj6V///6Eh4c7LfOrUgUKnFpVLO7u5jGAHBdrdnRdXa35HnJt0cwsQP36eYXEwIEQG2sWy/R0Npvtr%2BfJbjeLZdq5NlteIZGRYZ4LQHa2WXsXl7xCIi0N/jxFsdhMH4%2BLS14hkZpqnEvknIpG7atVyyskXnkFjh0zCgXA1DZbzAJUqJBXSHzzDZw%2BbRbr44/N2levnldIzJ8PR48ahUp7PNKovYtLXiGRnm7%2B8gUo72X4/rbqA1iuaSomRAp48803yc7OplOBI1lZWVkkJCQwZswYxowZQ79%2B/YiOjiY6OppXXnmFlStX0qRJEwDH/AnIKxYAbDYbnp6e3Hjjjbz77ruF3q/NZnNsbxV/f3/8/f2dF1pQBPxt7dxpTZzYWPNYVr0W7HbzWKZ7LPlH/ex2a/Z%2BrIiRH8c0lpW5GO6Qxcdbk8qxYxbFapBiQRDyCokUw1gHDliTy9GjxrFK08tX/qSRCWMqJkT%2BlJaWxvvvv8%2BMGTMcpxsBpKen07dvX7Zv306jRo0ICAhg0KBBDBo0iMcff5yNGzc6iomDBw862h05cgSAatWqERQURHx8PGlpaZT/8/SK5ORk3N3d8fHxITAw0GmydmZmJqtXr6ZPnz5X46GLiIiUTSomjKkHRf70/vvv4%2BnpSe/evQkODnb8q1%2B/PuHh4Sxbtow77riDn376Cbvdzh9//MH%2B/fudTk/auHEjBw4cIC0tjaVLl9K4cWP8/Pxo164dvr6%2BPP/886SmppKUlMT48eOZNWsWAH369OHrr7/m448/JisrixUrVrBq1Sp8fHzw8vLi9OnTJCQkkJGRUVLdIyIiInIeFRMif1q/fj09e/Z0zGMo6O677%2Babb75h0KBBTJgwgaZNm9K7d2%2BaNm3KoEGDHNv17duXyZMn06ZNG3777TeioqIAcHd3Z8GCBcTFxXHLLbfQq1cvQkJCmDJlCgANGjRg1qxZzJw5kxYtWrB161YWLlyIu7s7rVu3pmbNmnTu3JmtW7denc4QEREpC3RpWGM6zUnkT2%2B88cYF13Xo0IGYmBgAxowZc8HtQkJCWLduXaHr6tevz6uvvnrBtt26daNbt27nLff19eX999%2B/YDsRERGRkqJiQkRERETKpjI4kmA1FRMiIiIiUjapmDCmYkLEInv37i3pFERERESuKhUTIiIiIlI2aWTCmHpQRERERESKRSMTIiIiIlI2aWTCmIoJERERESmbVEwYUw%2BKiIiIiJQShw8fZsSIEbRq1YqOHTvy4osvkpube952c%2BfOpUGDBoSGhjr9O378OABnz57l//7v/2jfvj2tWrVi3LhxJCcnW56vigkRERERKZtK4S9gjx07loCAAKKjo1m%2BfDnR0dGsXLmy0G0jIiKIiYlx%2Ble1alUAXnrpJXbv3s3atWv58MMPsdvtPP7448b5nUvFhIiIiIhIKRATE0NsbCyPPPIIFSpUICQkhAcffJC1a9deVpzs7GzWrVvHqFGjqF69OpUrV2bChAls27aNhIQES3PWnAmRMiQLd%2BMY7hbFee8d4xBUqgTh4fDJJ3DypFms3hkZZgFstrzbr74Cu90slpeXWfuwMNixA1q3hp07jUKdTDF7LC4uUAE4ne1NIaP0l63Sd1vMAlSoAC1bwp49cPq0UagPszsZp9K2LXy5q6JpKowYYdbe2zvvtk8fSE83iwUwZP7dRu2DgmB6J5j%2BWScOHjTLpUcvs1wqV4ZOwJaOkaSkmOXi/4NZ%2B/LloVkz2LsX0tLMYgHcWvd48Ru7ucF110FKCmRnmycTEGAeoziuwJyJxMREkpKSnJb5%2Bfnh7%2B9/yba7d%2B%2BmRo0aVKpUybGsUaNG7N%2B/n9TUVHx8fJy237t3LwMGDOCXX36hevXqPP7447Rr146DBw9y%2BvRpGjVq5Ni2du3aeHl5sXv3bgIs7G8VEyIiIiJSNl2BYmLt2rXMmzfPadmYMWMYO3bsJdumpKRQsWJFp2X5hUVycrJTMVGtWjUCAwOZPHky/v7%2BrF27lpEjR/LOO%2B%2BQ8mfle26sihUrWj5vQsWEiIiIiIhF%2BvfvT3h4uNMyPz%2B/Ire3F3F0u1%2B/fvTr18/x/wcffJBNmzbxzjvv0L59%2B8uKZULFhIiIiIiUTVdgZMLf379IpzQVxtfX1zGqkC8lJQWbzYavr%2B8l29eoUYPExETHtikpKZQvX96x/uTJk1x33XXFyu1CNAFbRERERKQUaNy4MUePHuXEiROOZTExMdSpU8epKABYsGAB27dvd1q2b98%2BAgMDCQwMpFKlSuzevdux7pdffiEzM5PGjRtbmrOKCREREREpm0rZpWEbNmxIaGgoUVFRpKamsm/fPpYvX869994LQLdu3fjuu%2B%2BAvFGHGTNmEBcXx9mzZ1m2bBkHDx6kd%2B/euLq6cs8997Bo0SKOHj1KcnIy//73v%2BnSpYvj0rFW0WlOIiIiIlI2lcJfwJ4zZw5PPfUUt9xyCz4%2BPgwYMICBAwcCsH//fs6cOQPA5MmTgby5EikpKdSpU4cVK1ZQrVo1AMaNG0daWhoRERFkZ2fTsWNHpk%2Bfbnm%2BKiZEREREREqJatWqsXTp0kLX7d271/G3p6cnTzzxBE888USh23p4eDBt2jSmTZt2RfLMp2JCRERERMqmUjgyca1RD4qIiIiISLFoZEJEREREyiaNTBhTMSEiIiIiZZOKCWPqQRERERERKRYVEyIWCQ8PZ82aNSWdhoiIiBRVKfudiWuRTnMSKaKsrCwWLlzIpk2bSEhIwGaz0bhxY8aPH0/z5s1LOj0RERGRq07FhEgRPffcc3z//ffMmTOHOnXqkJ6ezurVqxkyZAibNm0q6fRERETkcpXBkQSrqQdFiuiLL76gR48e1KtXD1dXV3x8fHj44YeJjIzEw8PDadvc3Fzmz59Ply5daNKkCb1792b79u2O9fXq1eOtt96ib9%2B%2BNGnShF69ehEXF%2BdYHxsbywMPPEDz5s1p3bo1kZGRZGVlXbXHKiIiUiboNCdjGpkQKaJatWrx9ttv065dOxo0aOBYftddd5237Wuvvcabb77J4sWLqVWrFq%2B%2B%2BiqjRo0iOjqa6667DoDly5cze/ZsqlWrxrRp05g0aRIbNmwgPT2dYcOGMXjwYJYuXUpCQgKjRo3iP//5DyNHjixyvomJiSQlJTktq1LFDz8//2L2gLUqVTKP4ePjfGvEZrOmvWkcgLAws/b16zvfGjD9Xsxvb9n3a4UKZu3LlXO%2BNUkl26x9%2BfLOtya8vc3ae3k535oKCjJrX726862JypXN2ue/5ExfemD%2BXOc/z6bPt4ObwW6gq6vzrYlswzeTlCib3W63l3QSIteCI0eOMGnSJHbu3EmNGjW4%2Beab6dChA7fffjseHh6Eh4czfPhw7r33XiIiIujRowcjRoxwtG/bti2TJ0/m7rvvpl69ejz66KMMGzYMyBuJiIiI4NNPP2XHjh1ERkbyxRdfONpu2LCBxYsX88EHHxQ537lz5zJv3jynZaNHj2HcuLGGPSEiImKhhAQICCiZ%2B54/3/qYo0dbH7MU08iESBFdf/31vPHGG/z22298%2BeWXfPvtt0ydOpWXX36ZV1991WnbQ4cOUbt2badlQUFBHD582PH/WrVqOf6uUaMGAAkJCcTHx/PHH38QGhrqWG%2B32887lepS%2BvfvT3h4uNOyKlX8MD1byt0d4xgAn31mHsPHB1q2hG%2B%2BgdRUs1jh7TLNAthsf3WO6TGa1q3N2tevD6%2B/DgMHQmysUajTn%2Bwwau/iknc0Ni0NcnONQgFQYc83ZgHKlYPGjWHXLjhzxijUl9ktjdqXLw9Nm8KPP%2Bb1jwnTo%2B9eXnDDDRAXBxkZZrEA/vtfs/bVq8NDD8HixXD0qFmsW281a1%2Bhwl%2BfM6dPm8WqUsWsvbc3NGgAe/ZAerpZLIBmwX8Uv7Gra94LLyUFcnLMk5FrlooJkctUp04d6tSpw/33309SUhL9%2BvVj5cqVTttkZha%2BY2orcApMboE9q/wBQpvNhqenJzfeeCPvvvuuUZ7%2B/v74%2Bzuf0lSapl2cPGldrNRUC%2BJZNUhrt5vH2rnTmlxiY41jWVEA5MexJJbp3ly%2BM2eMY5226MyMtDTzh%2BXpaU0uGRnW7KQePGgeA/IKCdNYKSnW5HL6tHmsyzwmdEHp6eYFKGDN6UU5Odf2aUplcI6D1dSDIkVw7Ngxpk%2BfTuo5h7/9/PyoX78%2B6ed8%2BwYFBTlNqM7OzubAgQMEBgY6lh0s8A155MgRAKpVq0ZQUBDx8fGkFfimSE5OPu%2B%2BRURExJAmYBsre49YpBh8fX358ssvefTRR4mLiyM3N5f09HTee%2B89tm/fft7pRBEREbz%2B%2Buvs27ePzMxMFi1aRE5OjtN2Gzdu5MCBA6SlpbF06VIaN26Mn58f7dq1w9fXl%2Beff57U1FSSkpIYP348s2bNutoPW0REROSidJqTSBF4eHiwevVq5s6dy9ChQzlx4gQuLi40aNCAqKgobj3npNwhQ4aQnJzM8OHDOXXqFA0aNGDVqlVUrFjRsU3fvn2ZPHkyv/zyC7Vq1eLll18GwN3dnQULFhAZGcktt9yCj48PnTp1YsqUKVf1MYuIiPztlcGRBKupmBApooCAACIjIy%2B4fuvWrY6/3dzcmDJlykULgJCQENatW1fouvr16583qVtEREQspmLCmHpQRERERESKRSMTIiIiIlI2aWTCmIoJkRKwd%2B/ekk5BRERExJiKCREREREpmzQyYUzFhIiIiIiUTSomjKkHRURERESkWDQyISIiIiJlk0YmjKkHRURERESkWDQyISIiIiJlk0YmjKmYEBEREZGyScWEMRUTImWI%2B/GjZgHc3MDPD/eUJMjONgrV%2B9gGs1wAsqsC/Qj/4004ftww1v1m7fO/kHJyIDfXKNTJFLtxKhWA05/sME2FSpVtZgHCwmDHDip0aAY7d5rFAhgwwKx9SAi0bAlvvw2//24Uquukima5lCsHNKJtpd3gfsYoVHrDFkbtbX8%2BzTfcAHazlx8Ay55PMgvg5gZUYfr4ZOPPGmN/5tKpmQW5mH5OeXkBtWhWZT94Z5jFAnKqNjBq7wrkVL7OOI/8WHJtUjEhIiIiImWTRiaMqQdFRERERKRYNDIhIiIiImWTRiaMqZgQERERkbJJxYQx9aCIiIiIiBSLRiZEREREpGzSyIQx9aCIiIiIiBSLRiZEREREpGzSyIQxFRMiIiIiUjapmDCmHhQRERERkWLRyISIiIiIlE0amTCmHhQRERERkWLRyISIiIiIlE0amTBW6ntw6tSp/POf/yzpNJy89dZb3HLLLZbGXLBgAffdd5%2BlMeXytWrVio0bNxar7Ztvvkn79u0tzkhERESuGBcX6/%2BVMZY%2B4vDwcNasWXPe8jVr1hAeHl6smJGRkbzwwgumqV3S9u3biYmJsTzu119/Tb169QgNDSU0NJSbbrqJXr16sWjRIjIyMhzbjRo1ildfffWS8XJycli%2BfLnleVrp22%2B/dTze0NBQ6tWrR%2BPGjR3/nzp1qvF9XGzH3aQgEBERESlJhw8fZsSIEbRq1YqOHTvy4osvkpubW%2Bi2a9asoWvXroSFhREREUF0dLRj3WOPPUbDhg2d9smaN29ueb46zelPK1as4LbbbiM0NPSKxP/uu%2B/w9PTk1KlT7N69m5deeokPP/yQV199lfLlyxc5zs8//8wrr7zCP/7xjyuSpxVatGjhVJjVq1ePBQsW6Ki9iIiIlC6lcCRh7NixNGrUiOjoaP744w8eeughqlatet6%2B34cffkhUVBSLFy%2BmSZMmbNiwgQkTJvDBBx8QGBgIwMMPP8zYsWOvaL5XvQePHTvGww8/TKtWrbj55puZOHEiKSkpQN5R/LCwMFasWEGzZs3YuXMnjz32GBMnTgRgyJAhTtVVw4YNnUY8oqOjueuuu7jpppsIDw9n1apVjnWPPfYYM2fO5Nlnn6Vly5a0bt2apUuXAjBy5Ei2bdtGZGQkDzzwAAAxMTEMHDiQ5s2b07ZtW6ZNm0ZWVpbx469YsSJt2rRhxYoVpKam8p///AeAuXPncs899wCQnp7OlClTaNOmDWFhYQwYMIBdu3bx008/MWDAAI4fP05oaChfffUVdrudWbNm0aFDB8LCwujduzfffvut4/4GDx7MokWLePTRR2nWrBm33nqr01H7%2BPh4hgwZQlhYGB07dnTqs8OHDzNy5EhatWpFixYt%2BOc//0lqaqpxH%2BQ7cuSII37z5s2ZPHkyJ0%2BetCz%2BV199RWhoKKdOnXIsO3PmDE2bNmX79u1kZWUxffp0WrVqRfv27Vm/fr1T%2B3vvvZeoqCh69uzJww8/fNk57927l8GDB9O8eXNat27NjBkzyMzMdKyfO3eu4/W1evVqBg8ezIIFC1i/fj3t2rVzOgoRHx9P/fr1iY%2BPt6x/REREpHSJiYkhNjaWRx55hAoVKhASEsKDDz7I2rVrz9s2IyODSZMmcfPNN%2BPu7k6/fv0oX748P/zww1XN%2BaqPTIwaNYo6deqwZcsWMjIyGD9%2BPNOmTePll18GICsriwMHDvDll1/i6enp1HnLli1z/J2amkqfPn3o27cvALGxsYwfP56XX36ZDh068N133zFy5EiCg4Pp0KEDAO%2B99x6PPfYYX3zxBe%2B88w5PPfUUERERLFq0iPDwcIYPH869994LwMSJE7nrrrtYvXo1CQkJDBgwgDp16jB48GBL%2BqFcuXLcc889vP3224wbN85p3cqVKzl%2B/DibN2/Gw8ODpUuX8tRTT/H2228zc%2BZMoqKi%2BOKLLwDYsGEDGzZsYN26dfj5%2BbFw4ULGjRvH559/jqurKwCvvfYazzzzDM888wyLFi3i6aefpnv37ri7uzNmzBhatmzJ/Pnz%2Bf333xk0aBC1a9embdu2jBo1imbNmvHSSy9x5swZJk2axPPPP8/MmTONH7/dbufhhx%2BmYcOGbNmyhfT0dMaOHcvTTz9NVFSUcXzIO92patWqfPjhh/Tr1w%2BATz/9lEqVKtGqVSvWrFnDli1bWLNmDQEBATz33HOkpaU5xdi0aRNz5syhUaNGl5VzRkYGQ4cOpW/fvixdupSEhARGjBjBvHnzmDRpEh988AGvvPIKy5cvp379%2BkRGRhIbG0ubNm3o2rUrM2fO5KuvvqJt27YAfPTRR9x0002OIw1FkZiYSFJSktMyP5sNfz%2B/4nRnHjc351sTVauax6hc2fnWhOnRKZvtr1vDWKap5Le35IBbWJhZ%2B/r1nW9NhYSYtb/%2BeudbE%2BXKmbX38nK%2BNZD/8jNtbxrHwfQz4s/vL8dtSbIyF9Pn2sPD%2BVbMXYGRiUK/f/388Pf3v2Tb3bt3U6NGDSpVquRY1qhRI/bv309qaio%2BPj6O5REREU5tT506RVpaGgEBAY5lX331FVu2bOHAgQPUrl2b6dOn07hx4%2BI%2BtEJZXkxERkbyzDPPOC3Lzc0lICCAPXv2sHv3bhYvXoyPjw8%2BPj6MGDGC0aNHO47YZmVlMXDgQLwu8YabNm0aQUFBDB8%2BHID169fTpk0bOnfuDECbNm247bbbeP/99x3FRM2aNenduzcA3bt354knnuD3338v9MndsGEDHh4euLq6cv3119OiRQt27dpl1jnnqFWrFocOHTpv%2BalTp3B3d8fLyws3NzdGjRrFqFGjCo3Rs2dPOnXqRIUKFQDo0aMHc%2BfO5ciRI44dz7CwMG699VYA7rjjDubNm0diYiInT55k7969rFy5Em9vbxo0aMC8efMICAggJiaGX3/9lTVr1uDt7Y23tzdjx45l6NChPP3009gMv3F27drF3r17Wb58ueO1MHz4cCZMmEBWVhbu7u6XjJGQkFDoaWn5ryWbzcZdd93FO%2B%2B84ygmNm/eTI8ePXBxcXGMZN1www0AjBs3jv/%2B979OsW666SbHmy4mJuaiORe0bds2MjMzGT16NO7u7gQHBzNo0CBWrVrFpEmT%2BPTTT%2BnQoQPNmjUDYMqUKbzzzjsA%2BPj40KlTJ959911HMbF582Z69uxZ5P4FWLt2LfPmzXNaNmb0aMaeU7wWS5Uq5jH%2BfE4s0aWLdbFMWbBjWMGCNAAu4wzKC9uxw4IgwOuvWxPHKld42P%2By1K5tHML8VZfH09OiQF4WfEYAVKxoTRwrWJGLFZ%2BdADVqWBLGilLNihorJ8c8RrFdgWKi0O/fMWOKdLpRSkoKFc95reUXFsnJyU7FREF2u52pU6fStGlTWrZsCUBgYCAuLi6MHz%2Be8uXLM2/ePIYMGcKHH35IFatei1yBYmLq1KmOo/v51qxZw9KlSzl06BCVKlXCr8CR0aCgILKyskhISHAsu/4SR4zWrVvHN998w8aNGx07tYcOHaL2OR/IwcHB7CjwRVizZk3H397e3gBOk6AL%2BuqrrxxH67Ozs8nOzqZbt24Xzety5eTkOEYPCho4cCBDhw6lQ4cO3HrrrXTu3JlOnToVGiM9PZ1nnnmGTz/91Ol0m4Kn0xR83PlFWkZGBgcPHsTHx4fKBY7q5u%2B8btq0iZycHFq1anVezsnJyfj6%2BhbjEf/l0KFDVKlSxSlOcHAwmZmZHD9%2BnOrVq18yRkBAAJ9%2B%2Bul5ywvmHBERwZIlSzh27Bi%2Bvr5s27bNMdH92LFj3H777Y5t/fz8znuTFnwtXirncx9fUFCQU1EUHBzMkSNHgLyjFnXr1nWsq1SpEsHBwY7/9%2BrViwkTJjB9%2BnROnjzJrl27WLBgwSX7pKD%2B/fufd%2BEDP5sNzjlaclnc3PK%2BDJOTITu7%2BHEAtm0zaw95IxJdusDmzfDn6ZLFduedZu1ttrxCIiMD7HajUKezvY3au7jkFRJpaXCBOXtFVqFDM7MA9evnFRIDB0JsrFksgK5dzdpff31eITF3Lvz5fiw20yvweXnlFRL79uW9bgxk1G5k1N5myyskzp41fvkC4JWebBbA1TVv5/3UqRLe07Q4F9PPKQ%2BPvELi8GEo8D1fXDlBtYzau7qW/NNTGhX6/XsZZwXYL/NNmJWVxWOPPcZvv/3mdLr66NGjnbZ79NFHee%2B994iOjnYcZLXCVT3NKfMiL/yCR7rdLjI8um/fPp555hkWLlzotFN3odgF47oUsfrct28f48ePZ8qUKdxzzz14eXnx6KOPkm2683SOn3/%2BmVq1zn8j16xZk/fff5%2Bvv/6arVu38n//93%2B88847zJkz57xtZ8yYwd69e3nttdcIDg4mPj6eLuccpb3Q43Zxcbng1QE8PT0pV64cO3fuLMYju7SivhZM3XDDDTRu3JhNmzZRp04dqlWrRoMGDRw55JzzKXhufxQs9i4n54ttC3kfFOe%2Bzgs%2BT23btsXb25uPP/6YEydO0LZt28su4Pz9/c8fdTt61LwIgLwYpnHOKcCMpKSYxzPd685//ux241imqRSMYxzLqs%2BA2FhrYtWrZx4D8gqJ3383i3HmjCWpkJFhHMuKAiA/jiWxrPq%2BzMmxLpYpK3IxLBodMjOti1XWXYGRiUK/f4vI19fXMZc4X0pKCjabrdD9gIyMDEaNGkV6ejqvvfbaRUccXF1dqV69OomJicXK7UKu6gTswMBATp486XQUNy4uDk9PT6fzuy4kIyODCRMmMGTIkPOOmAcFBREXF%2Be0LC4u7rLOMc%2B3Z88ePDw8uP/%2B%2B/Hy8sJut7Nnz57LjnMxJ06c4PXXXy/01JW0tDRycnJo27YtU6dO5c033%2BTDDz8kOfn8Iz0//fQTd911FyEhIdhsNnbv3l3kHAIDA0lLS3N6UUVHR/PNN98QFBTEmTNnnCb8pqamFppDcQQFBZGcnMyJEyccy%2BLi4vD29r6s6r0oevXqxf/%2B9z/%2B97//cddddzmW%2B/v7c/ToUcf/jx49ypmLfKFfTs5BQUHEx8c7nf5U8PXo6%2BvrGKWAvFPbfi%2BwY%2BPq6krPnj353//%2BxwcffOCUt4iIiPw9NW7cmKNHjzrta8TExFCnTp3zrv5pt9uZOHEibm5urFixwqmQsNvtPPvss8QWGBHOzMzk4MGDxdo3vpirWkyEhoZSu3ZtoqKiOHPmDAkJCSxcuJAePXoU6Rz5Z555Bl9f30LnD9x111188cUXfPzxx2RnZ/PZZ5%2Bxbds2evXqVaTcPD09OXjwIKdPn6ZGjRpkZGSwZ88eTp48yYsvvoiHhweJiYmXPfR0rtzcXHbu3MmwYcOoU6cOgwYNOm%2BbcePG8fzzz5OamurYvnLlylSqVAkvLy9Onz5NQkICGRkZ1KxZk5iYGDIzM/nhhx/YtGkTQJGqzgYNGtCwYUNmz55NWloav/zyC08%2B%2BSQZGRnUrVuXsLAw/vWvf3HixAlOnTrFtGnTLPsBwaZNmxIcHExUVBTp6ekcO3aMRYsW0bNnz0JP/TLRvXt3YmNj%2Beijj7izwKks7du359133%2BXgwYOkpqby8ssv43GRSW2Xk/Ntt90GwKJFi8jMzGTfvn2sXr3aMWendevWfPzxx8TExJCens4LL7xAuXMmc/bq1YtPPvmEPXv2XPA0NxERETFQyn60Lv93IaKiokhNTWXfvn0sX77cMYWgW7dufPfddwC8%2B%2B67/Pbbb7z88st4njPhyWazcejQIWbMmEFCQgJpaWnMmjULd3d3x/xiq4X8k3gAACAASURBVFzVYsJms7FgwQISExO57bbbuOeee2jatCn/93//V6T2a9eu5fvvv6dp06ZOl4g9fPiwY8c3KiqKFi1a8MILLzBr1izHJJRLueeee3j99de57777CAsLY9CgQdx333306NGDGjVq8MQTT/DLL784LlN7uZo3b05oaChNmzblscceo2PHjixfvrzQndeZM2dy4MAB2rdvT4sWLXj11VeZP38%2BLi4utG7dmpo1a9K5c2e2bt3K5MmT2bdvHy1btuSll17iqaeeokuXLowaNapIoxSLFi3i8OHDtG3blpEjRzJq1CjH70FERUVht9vp1KkTXbp0IScnh%2Beee65Yj/9cLi4uLFy4kCNHjtChQwf69%2B9Ps2bNePLJJy2JX1CVKlVo37499erVo0aBSWtDhw7l1ltvpW/fvtxxxx00b978oqMil5Ozj48PixYt4osvvqBNmzaMGDGCu%2B%2B%2B23HBgN69exMREcHgwYMd912jRg2nU53q1atHcHAwHTt2dMzxEREREQuVsmICYM6cOSQmJnLLLbdw//3306tXLwYOHAjA/v37HWdRrF%2B/nsOHD9OyZUun/eL8Hwf%2B17/%2BRUhICH369KFt27bs2bOHlStXnnfw0pTNbnqoXeQaMGDAAPr37%2B8YGSgNMjMznYrJ9u3bM3HiREeO2dnZdOnSheeff77IRfElFTitq1jc3MDPL28St%2Bm5wxs2mLWHvMvL9usHb75pPmfi/vvN2ru4gLc3pKcbT1Q4mW12GSYXF6hQAU6fNp8zUamy4RymsLC8K0I1a2bNnIkBA8zah4TAs8/C44%2Bbz5mYNMmsfbly0KgR7N5tPGcivXELo/YWXj8AAO9Ugws9gLUXezBlZS6mn1NeXlCrFuzfb8mciZy6DYzaWzkBu8SuArx9u/Ux27SxPmYpVvp%2B9k/EQrm5ubz66qskJSXRvXv3kk7H4auvvqJVq1bExMSQk5PDm2%2B%2BSUpKCq1btwbyCol///vfBAQEWFdIiIiIiLNSODJxrbnqP1r3d/LBBx9cdA5BixYtnH5o7%2B/iP//5D7Nnz77g%2BoiICCIjI4sdf8mSJcydO/eC6/v06cOMGTMuGScnJ4ebbrqJkJAQ5syZc975hCWpdevWjBkzhvHjx5OcnExgYCAvv/wy1atXJz4%2Bnv9n787DazjbB45/k5CF2EJiTWhFrUFCBCUIWmui9r1ql9qCohprLW3xoiL29Wdp7Etqp9rSaimtiJASrTULEhJJJCc5vz8O53UIkjyj0jf357pyHZmZ584zY845c8%2BzTKtWrahatSqzZ89%2B01UVQgghhHghSSYUtGzZkpYtW77pavzj%2BvXrR79%2B/V5b/IEDBzJw4EDlOBYWFoSEhGhQo9fjRcfR0dExR9dbCCGE%2BJ%2BRC1sStCbJhBBCCCGEyJ0kmVAmR1AIIYQQQgiRLdIyIYQQQgghcidpmVAmyYQQQgghhMidJJlQJkdQCCGEEEIIkS3SMiGEEEIIIXInaZlQJkdQCCGEEEIIkS1mer1e/6YrIYT4h9y/r1be3BwKFID4eEhPV4sVF6dWHsDSEkqWhNu3ISVFLVbhwmrltTw2p0%2BrlS9QAOrUgV9/NdRHxYoVauXLlYNZs%2BDTT%2BGvv9RiAXzzjVp5V1c4cwbc3ODsWbVYOp1aeQALC0hLUw5zO9pCqXyePGBvDzEx2uxWYqJaeUtLcHSE69fV39qlSqmVNzMDa2tITgbVKyZLS7XyoNkpY4gVd1etIoULGz7LtahQ0aLqMbIjNFT7mFWrah8zB5NuTkIIIYQQIneSbk7K5AgKIYQQQgghskVaJoQQQgghRO4kLRPK5AgKIYQQQgghskVaJoQQQgghRO4kLRPKJJkQQgghhBC5kyQTyuQICiGEEEIIIbJFWiaEEEIIIUTuJC0TyuQICiGEEEIIIbJFWiaEEEIIIUTuJC0TyiSZEEIIIYQQuZMkE8rkCAohhBBCCCGyJVclE4GBgfTs2VM5zvjx4/Hz89OgRv/Vt29f5s%2Bfr2lMkTXh4eFUrFiRyMjIbJUfM2YMY8aM0bhWQgghhHhtzM21/8llcswe9%2B7dG39//wzX7dq1Czc3NxITE7Mcd/Xq1eh0OgB8fX1Zv369Uj2zauHChVSqVAkXFxeqVauGu7s7vXv3Zvfu3SbbrVq1ipEjR74y3vXr19m/f//rqq4mAgMDcXFxMe5zxYoVjb%2B7uLiwc%2BdO5b/xogt31YRACCGEEEJkXo5JJjp27Mi%2BfftITk5%2Bbt3OnTtp3bo1%2BfLly1LMe/fu8eWXX5KWlqZVNbOlevXqhISEcP78eYKDg%2BncuTNfffUVEydOzHKsgwcPcuDAgddQS%2B34%2BvoSEhJCSEgIK1euBOD06dPGZe3atXvDNRRCCCGEQFomNJBj9vj999/H3NycgwcPmiy/ffs2J0%2BepFOnTiQnJzNt2jQaN25MzZo16dWrF5cvXzZuW7FiRdasWUODBg0ICAjA09MTvV5P7dq12b59OwsXLqRz587G7Y8fP463tzc1a9bEx8eHn3/%2B2bhu9%2B7dtGrVCldXV7y8vNi4caMm%2B1m8eHHatGnDqlWr2L59OydPngSgV69ezJkzB4CrV6/Sp08fateujbu7O0OHDiU2NpaVK1cyZ84c9u/fj4uLC2lpady7d4/hw4dTr149ateuzYABA7h9%2B7bJMTl48CDdunWjZs2atG3blgsXLmTqGPz888906dIFV1dXGjZsyKJFizQ5Bk%2BcOnWKTp06GeMvWLAAvV6vWfyvv/6aTp06mSw7efIkNWvW5OHDh8TExNCvXz9cXV1p06YNISEhxu10Op3xfHr33XeNSVFW6nzgwAHatm1LjRo18PLyYsOGDcZ1iYmJDB8%2BnOrVq9OiRQt%2B/vlnKleuzOnTpxk3btxzrVTbtm3Dy8tL0%2BMjhBBC5HqSTCjLMbM5WVlZ0bZtW3bs2IG3t7dx%2Ba5du3B2dqZ69epMnz6dCxcuEBQURKFChfj6668ZOnQo%2B/btw8zMDIDDhw%2Bzc%2BdOihYtauxSdPr0aaysrFi4cKExblRUFMOGDWPGjBk0b96cPXv28PHHH3P06FHi4%2BMZN24cK1eupF69epw8eZK%2Bffvi5uZGpUqVNNnfd955h/r167N//37q1q1rsu7zzz/Hzc2NFStW8PDhQ8aNG8fixYuZMGECf/75J48ePWLevHkAzJ49m4cPH3LkyBH0ej0jR45k5syZJvu6YsUKvvjiC0qWLMnQoUOZN28ey5cvf%2BkxSE5OxtfXl8mTJ9O2bVsuX75M//79cXJyom3btsr7Hx0dTb9%2B/ZgwYQLt27cnPDycAQMGUKJECbp06aIcH6Bdu3YEBgby999/U7ZsWcDQstO0aVPy58/PhAkT0Ol0fP/99zx8%2BJDRo0c/F%2BPo0aPs2bOHIkWKZKnOoaGhjBo1ioULF9KwYUNOnTrF4MGDKVu2LA0aNGD27NlcvnyZQ4cOATBu3DjS09ON9R48eDAJCQnY2toa692mTRvjeZ4Z0dHRxMTEmCyzt7HBwd4%2B8wfxWU8%2BJLX4sLS0VI%2BRJ4/pqwrVfdLy2BQooFb%2BSStuFltzM1SunFr5UqVMX1W5uqqVf/IZrtFneU6h%2BhbQ8q0E6m/vvHlNX1Vk4WPzpeVV4%2BRIFhbqZVViPPGGe5AINTkmmQDo1KkT7du35/bt25QsWRKAHTt20L17d9LT09m%2BfTvz58%2BnePHiAIwcOZL169dz7tw5atSoAUDLli0pVqzYK//Wvn37cHR0pFWrVgC0b98eKysr0tPTKVOmDCdPnqRQoUIA1KtXj6JFixIaGqpZMgHw1ltvERER8dzyBw8eYG1tTZ48eShUqBCBgYGYv%2BACZerUqeh0OmMXsGbNmrFkyRKTbXx8fHj77bcB8PLyMt5lf9kxCA4OpkKFCsYuSRUrVqRr167s2rVLk2Riz549lC1blq5duwJQrVo1vL292bt3b6aTiW%2B//fa5Ll9P37l3cnKiZs2a7Nmzh6FDh6LX6zl8%2BDDTpk0jPT2dI0eOsGjRIgoWLEjBggXp2bMnv/32m0m8Vq1aYWdnl%2BU6b9u2jYYNG%2BLl5QVA/fr18fT0ZO/evTRo0IDvv/%2Bevn37Gs/ljz76yNgq5OHhQZEiRTh06BAffPABCQkJ/PTTT1ke3B0UFERAQIDJsqEff8yw4cOzFCdD%2BfOrx1C9YH6aSoKkNS2OTZ066jEAqlVTj6FVXYYN0ybOrFnaxNGotVkTGlyMafUWKFJEmzhaKVHiTdfgv6ys3nQN/kuL63cAChdWj6HFZ/ndu%2BoxsisXtiRoLUclE5UrV6Zy5crs3LmTIUOGcPbsWW7duoW3tzd3797l4cOH%2BPr6mtydTU9P5/bt28ZkolQm735du3aNMmXKmCxr3bq18d%2BbNm1i69atREdHo9frSUlJISUlRYO9/K%2B0tDQsMvhEGDp0KJ988gk7d%2B6kQYMGtGnThurVq2cY4%2B%2B//%2BaLL77g3LlzJCcnk56eTuFnPhye3k8bGxsePXoEvPwYXLt2jZCQEFxcXIzr9Ho9b731VvZ29hk3btwwJjhPODk5ceTIkUzHaN26tbFr2BPh4eEmyU67du1YvXo1Q4cO5Y8//iAlJYUGDRpw7949UlNTTfa/XAZ3YJ8%2Bn7JS5xs3buDs7PzctqGhoQDExMRQunRp47qnj7O5uTlt27Zlz549fPDBB3z//fe8/fbbVKhQ4WWH4zldunQxJjNP2NvYQHx8luKYMDc3XCw/fAiPW1KyLSFBrTwYbqPa20NMDDyeaCHbHrcCZZuWxyYsTK18vnyGROL8ecjGxBUmduxQK1%2BqlCGRWLgQbt1SiwWgOmasUiVDItG9O1y8qBbr1Cm18mC4KtTgrmzMPbWryzx5DIlEbKz6Wwkgg%2BGPWZI3ryGRiIyE1FS1WKqJlpmZIZF49AhUe5pq0dKi0SljiBUfp1aRAgUM3ynSspCr5ahkAgwDsdeuXcuQIUPYsWMHzZo1o0iRIsQ/vgD65ptvqPaSu20ZXZxnxNzc3Nit5Flbtmxh2bJlBAYG4u7ujoWFBY0aNcr6zrzChQsXqFmz5nPLGzduzLFjx/j%2B%2B%2B85cuQIPXv2ZOzYsc9Na5uens6gQYOoVasWBw4cwM7Oji1btjw3xeyLusa87BhYW1vTqFGj51o5tPKixCwr3Xgyo1WrVsyYMYOQkBAOHTpEy5YtyZMnj/HvPz04P6NjkeepNv%2Bs1PlV2%2Br1epPYz8Zo164dK1eu5M6dOxw6dChbrUEODg44ODiYLrx/X/1CFwwxVONomZzrdOrxtDguT%2BKoxlJJ%2BJ6WmKge66%2B/NKkKt25pE%2BvsWfUYYEgktIqVA2iRADyJo0Usrd7eqanqsbQaaqbXaxcrx9AiCUhL%2B3cnE9IyoSzHHcG2bdsSGRnJmTNnOHDggHEAbYECBShcuDCXLl0y2f7GjRvZ%2BjtlypTh6tWrJsvWr1/P9evXCQkJoXbt2tStWxcLCwtiYmKIjo7O3g69wE8//cSZM2dMWkOeiI2NJX/%2B/LRq1Yq5c%2BcydepUgoKCntvuzp073Lx5k169ehm74jw9uPpVXnYMnJycCA8PN%2Bk2FBMTo1nrjJOT03NdvK5evYqTk5Mm8Z8oWLAgTZo0Yd%2B%2Bfezfv984HqdYsWJYWFiYDFZ/ejC/ap0z2jYiIsK4rZ2dHbeeukv79OBvgPLly1O5cmV27drFDz/8oEnXMiGEEEI8QwZgK8txe1ygQAHef/99Zs2aRf78%2BalXr55xXdeuXVm8eDFXrlwhNTWVNWvW0LFjR5KSkjKMZW1tDRgu%2BJ59RkWbNm24ffs2mzdvJiUlhW%2B//Zb//Oc/5M%2Bfn9KlSxMREcH9%2B/e5efMm06dPp1SpUkRFRSnv36NHj9i3bx%2BjR4%2Bmb9%2B%2Bz7WyJCcn8/7777Nr1y50Oh3JycmEhoYaL0KtrKy4ffs2Dx48wM7Ojnz58vH777/z6NEj9uzZQ1hYGAkJCTx8%2BPCVdXnZMWjdujVxcXEEBgaSnJzM9evX6du3L2vXrlU%2BBmBoMbh69SpbtmwhNTWV33//nV27dr2WaWPbtWtHUFAQZmZmuD4evGlpaYmHhwfr1q0jISGBGzdu8M0332hWZ29vb3788Ue%2B//574yDvH3/8ER8fHwDq1q3Lpk2buHPnDlFRUaxbty7DegcGBlKtWjXj2AohhBBCiJwkxyUTYBiIfe7cOTp06GDS/cPX15eGDRvSvXt3PDw8OHToEMuXL8fGxibDOJUrV8bV1ZWOHTuyadMmk3XFihVj5cqVrFmzBnd3d5YtW8aiRYuws7OjW7dulC1blkaNGjFw4EB69uxJz549Wb16tcn0npl17tw54wPbPDw8WLVqFePGjWPcuHHPbWttbc2CBQtYs2YNtWvXpnHjxkRGRjJp0iTA0HJz9epVmjRpQnR0NFOmTGHZsmXUr1%2BfU6dOsXDhQkqUKMF77733ynq97BgUKVKEwMBAjhw5gru7Oz179qRJkyb07ds3y/ufEUdHRxYuXMjGjRupU6cO48ePZ9SoUa/lDrynpyeWlpa0adPGZPmsWbNITU2lYcOGDBo0iA8//FCzOteuXZupU6fy1Vdf4e7uzty5c5k/fz61atUCDE9Rf9JqMmjQIPr37w9gMtC%2BdevWJCcnS6uEEEII8bpIy4QyM71MXC/%2Bxz148IDGjRuza9cuHB0d33R1jFJSUrB8PH/iX3/9xfvvv8/Ro0eNA7OvXr1Kx44d%2BeGHH8ivxQxBYBgzocLc/L8D7lTHBcQpDPx7wtISSpaE27fVO1arzmqi5bE5fVqtfIEChlmYfv1VfczEihVq5cuVM8zA9Omn2oyZeEUL4iu5usKZM%2BDmpj5mQovBBRqNpr0drT4AW6u5DEB93L%2BlJTg6wvXr6m9t1VmJzczA2towqFz1ikmLGbE1HYAdpzCLkoWF4XMzLk6bChUtqh4jO1S/FzPyeDbQ3CL3pU8iV0lOTmbq1Kk0adIkRyUSX3/9NR06dCAmJoakpCSWLVvGO%2B%2B8Y5wS%2BcGDB0yZMoUePXpol0gIIYQQwpS0TCjLcbM5/Zt8/vnnbN68%2BYXrhwwZgq%2Bv7z9Yo3/G4MGDOXHixAvXf/7550pjHwYMGGB8MnhGZs2a9VyXpYz88ssvDBgwgDp16jB79uxs1%2Bd1GDBgAJGRkbRt25a0tDSqVq3K/PnzMTc3Z%2BfOnUyZMoWmTZv%2BT54/QgghRI6RCy/%2BtSbJhIKJEycyceLEN12Nf9zrmi72ieXLl2sSx8PDg3PnzmkSS2s2NjbMnDkzw3Xt2rV7LQPRhRBCCCG0JsmEEEIIIYTInaRlQpkcQSGEEEIIIUS2SMuEEEIIIYTInaRlQpkkE0IIIYQQIneSZEKZHEEhhBBCCCFyiJs3bzJw4EA8PDxo0qQJs2fPJv0Fzy9at24d77//Pm5ubnTr1o3z588b1z169IhJkybh6emJh4cHw4cPJzY2VvP6SjIhhBBCCCFypxz4nIlhw4ZRvHhxDh8%2BzOrVqzl8%2BDBr1659brujR4%2BycOFCvvrqK3766SeaNGnC4MGDSXz81Mh58%2BYRGhpKUFAQBw4cQK/X8%2BmnnyrX71mSTAghhBBCCJEDhISEcPHiRcaMGUOBAgUoV64cffr0ISgo6Lltg4KCaN%2B%2BPTVq1MDa2pr%2B/fsD8N1336HT6di6dSu%2Bvr6ULFmSwoULM3LkSI4dO0ZUVJSmdZYxE0LkIkmWhZTKm5mBNZCctwB6vVpdbL7U4O6IoyN8%2BimsWQPXr6vFmjFDrbyZmeFVr4cXNEdn1gFdU6XyBXRQH/hJV4d4nVIo3h9VUC1AvnyG15494fHdMiXr16vHADh1Sj1GHsWvUFdXOHMG3N3h7FmlUCXSFd%2BQjxUrpkkYzI7/qBYgf35wdMMx5gw8fKgWK7GwWnlra6hQAevrf0JyslqsnTvVypcoAQMGYLFqOURGqsUCbnyU/Wdl5c0LxYGoR4VJTVWuCmXUQ2TPaxgzER0dTUxMjMkye3t7HBwcXlk2NDSU0qVLU6jQf7%2Bvq1atytWrV0lISMDW1tZk21atWhl/Nzc3p3LlyoSEhFC5cmXi4%2BOpWrWqcX358uWxtrYmNDSU4sWLq%2ByiCUkmhBBCCCFErqTHTPOYQUFBBAQEmCwbOnQow4YNe2XZuLg4ChY0vYnzJLGIjY01SSbi4uJMko4n28bGxhIXFwfwXKyCBQtqPm5CkgkhhBBCCCE00qVLF7y8vEyW2dvbZ7q8PgtN/6/aNiuxskuSCSGEEEIIkSsp9krNkIODQ6a6NGXEzs7O2KrwRFxcHGZmZtjZ2ZksL1KkSIbbVqhQwbhtXFwc%2BfPnN66/f/8%2BRYsWzVbdXkQGYAshhBBCiFwpPV37HxXVqlXj9u3b3Lt3z7gsJCQEZ2dnk6TgybahoaHG39PS0rhw4QI1atTA0dGRQoUKmawPDw8nJSWFatWqqVXyGZJMCCGEEEIIkQNUqVIFFxcX5s6dS0JCAleuXGH16tV069YNgBYtWnD69GkAunXrxs6dO/n9999JSkpi8eLFWFpa0rhxYywsLOjcuTNLlizh9u3bxMbG8p///IfmzZtTTKuZFh6Tbk5CCCGEECJXeh3dnFR9/fXXTJw4kXfffRdbW1u6du1K9%2B7dAbh69arxORKenp6MGjWKkSNHcvfuXVxcXFi2bBnW1tYADB8%2BnIcPH%2BLj44NOp6NJkyZMmTJF8/pKMiGEEEIIIUQOUaJECZYvX57hukuXLpn83r17d2Oi8SxLS0smT57M5MmTNa/j0ySZEEIIIYQQuVJObJn4t5FkQgghhBBC5EqSTKiTAdhCCCGEEEKIbJGWCSGEEEIIkStJy4Q6aZkQQgghhBBCZEuOTSb8/f0ZO3bsm66Gie3bt/Puu%2B9qGjMwMJCePXtqGlNob/369TRv3vxNV0MIIYQQGsppD637N9Kkm5OXlxcDBgwwPlDjiU2bNrF8%2BXKOHj2a5ZjTp0/Xomqv9PPPP2Nra4uLi4umcX/55Rd69%2B6NpaUlABYWFpQrV44WLVrQp08f4xzAvr6%2B%2BPr6vjJeWloa69at46OPPtK0nlo6deoUffv2Nf6ekpJC3rx5MTMzA8DHx0f5/3XLli34%2B/sbjyuAlZUVFStWZMSIEdSpU0cpvhBCCCFyj9x48a%2B1XD9mYs2aNTRu3FjzZOKJ06dPY2VlxYMHDwgNDWXevHkcOHCA9evXP/dY9Je5cOECK1asyNHJhLu7OyEhIcbfK1asSGBgIJ6enpr%2BneLFi/PDDz8Yf09MTGTdunUMHDiQ4OBgypQpo%2BnfE0IIIYQQGfvHujlFRkYyZMgQPDw8qFWrFn5%2BfsTFxQGGu/iurq6sWbMGNzc3zp49y/jx4/Hz8wOgb9%2B%2BuLi4GH%2BqVKmCl5eXMfbhw4fx9vamZs2aeHl5sW7dOuO68ePH8/nnnzNr1izq1KlD3bp1jQ8CGTx4MMeOHWP69Ol8%2BOGHAISEhNC9e3dq165N/fr1mTx5Mqmpqcr7X7BgQerVq8eaNWtISEhg5cqVACxcuJDOnTsDkJSUxLhx46hXrx6urq507dqV8%2BfPc%2B7cObp27cqdO3dwcXHh5MmT6PV65syZQ6NGjXB1deWDDz7g1KlTxr/Xq1cvlixZwieffIKbmxsNGzZk165dxvXXr1%2Bnb9%2B%2BuLq60qRJE5NjdvPmTQYPHoyHhwfu7u6MHTuWhIQE5WPwxK1bt4zxa9euzejRo7l//3624%2BXLl4/BgwdjZ2fHiRMnjMs3bNhAixYtqF69Oi1btmTfvn3GdXfv3mX48OHUq1eP2rVrM2jQIKKioozrz549S9u2balZsyb9%2BvUjNjYWgIiICKpUqcLDhw8BQyJTtWpV5s2bZyw7d%2B5cRowYAcDOnTtp2bIlrq6uNG3alKCgIABOnjyJi4sLDx48MJZLTEykRo0a/Pzzz1y5coXevXtTq1Yt3N3dGT58uPH9IoQQQghtSDcndf9Yy4Svry/Ozs4cOXKE5ORkRowYweTJk1mwYAEAqamp/P333/z0009YWVkZL7oAVq1aZfx3QkIC7du3p2PHjgBcvHiRESNGsGDBAho1asTp06cZPHgwZcuWpVGjRgAEBwczfvx4Tpw4we7du5k4cSI%2BPj4sWbLkuS5afn5%2BeHt783//939ERUXRtWtXnJ2d6dWrlybHIV%2B%2BfHTu3JkdO3YwfPhwk3Vr167lzp07HDp0CEtLS5YvX87EiRPZsWMHn3/%2BOXPnzjVeLO/cuZOdO3eydetW7O3tWbx4McOHD%2Bf48eNYWFgAhovpmTNnMnPmTJYsWcK0adNo1aoVefPmZejQodSpU4dFixbx119/0aNHD8qXL0/9%2BvXx9fXFzc2NefPmkZiYyKhRo/jyyy/5/PPPlfdfr9czZMgQqlSpwpEjR0hKSmLYsGFMmzaNuXPnKsV9Ouk7dOgQ8%2BbNY%2BnSpVSvXp19%2B/YxZswYKlSogLOzM19%2B%2BSVJSUkcPXqUtLQ0RowYwaxZs5g/fz46nY5hw4bRvn17hg4dSmhoKMOHD8fa2pq3336bokWLcu7cOerVq8fZs2dxdHTkt99%2BM/7t06dP065dO/7%2B%2B28%2B/fRT1qxZQ506dThx4gQDBgzAzc0NDw8PihUrxoEDB%2BjUqRMAP/zwA4UKFcLDw4M%2Bffrg4eHB6tWriY%2BP55NPPmHp0qWMGzcu08ckOjqamJgYk2UFC9pjb%2B%2BQ7eP8uMea8VWJo6N6jOLFTV9VPH7fZJu5uemrggIF1Mo/afTMQuPni%2BXLp1b%2BcZdO4%2Bv/EldXtfKVKpm%2B/i9RPflsbExfVaiee1ZWpq8qSpRQK1%2B0qOmrorx5s182Tx7TVxUa3LMVb5BmycT06dOZOXOmybL09HSKFy9OWFgYoaGhLF26FFtbW2xtbRk4cCAff/wxKSkpgCGZ6N69u3EswYtMnjwZJycnBgwYAMC2bduoV68ezZo1A6BevXo0btyYvXv3GpOJBZDjRwAAIABJREFUMmXK8MEHHwDQqlUrJkyYwF9//YWDw/MXVTt37sTS0hILCwtKlSqFu7s758%2BfVzs4z3jrrbe4cePGc8sfPHhA3rx5sba2Jk%2BePC8dT9G2bVuaNm1KgcdXHa1bt2bhwoXcunULx8cXaa6urjRs2BCAli1bEhAQQHR0NPfv3%2BfSpUusXbsWGxsbKleuTEBAAMWLFyckJIQ///yTTZs2YWNjg42NDcOGDaNfv35MmzbNOP4hu86fP8%2BlS5dYvXq18VwYMGAAI0eOJDU1lbzZ%2BGSLj49n9erVJCQkGFustm7dio%2BPD7Vq1QLA29ub1atXc%2BDAAZydnZk%2BfTppaWnYPP6i8vLyYvXq1QD88ccfxMbGMnDgQCwtLXF1dcXLy4uffvoJAA8PD86cOUO9evU4deoUPj4%2BrFmzhpSUFPR6PSEhIXz55Zc4Ojpy8uRJChUqBECDBg0oXLgwoaGhVKhQAW9vb3bv3m1MJg4dOkTr1q0xNzcnPj4ea2trLCwsKFy4MEuXLsU8ixepQUFBBAQEmCz7%2BOOhDB8%2BLMvH%2BFlafKfy6acaBHnsqbE6b5ytrXKI%2BvU1qAdQo4YWUapqEQTKl9cmjlZUk0eAM2fUYwBs3KgcQov8HjS6UQDg5qZNnMqVtYmjBScn9RgVKqjHAGjfXpMwGtyG0SSvyeCS6B%2BTG1sStKZZMuHv7//CAdg3btygUKFC2NvbG9c5OTmRmppq0rWkVKlSL/0bW7du5ddff2XXrl3Gi9obN25Q/pkvqbJly3LmqQ/5p/vQP7l4TE5OzvBvnDx50ni3XqfTodPpaNGixUvrlVVpaWnG1oOnde/enX79%2BtGoUSMaNmxIs2bNaNq0aYYxkpKSmDlzJj/88INJF6EnyRmY7veTJC05OZlr165ha2tL4cKFjevrP756%2Bfbbb0lLS8PDw%2BO5OsfGxmJnZ5eNPf6vGzduUKRIEZM4ZcuWJSUlhTt37lCyZMlXxoiKijIZ45KSkoKHhwfr1q0znmM3btx4bqyGk5MTN2/eBODq1at88cUXnD9/nuTkZNLT0yn6%2BBMxKiqKwoULY/vUReFbb71lkkwcOHAAMLRCjBw5kuPHjxMaGopOp8PBwQGnx18669evZ9u2bdy5cwe9Xk9KSoqxBcXHx4dly5YRGRmJnZ0dx44dY/369QAMHTqUcePGsX37dho0aEDbtm2pVq1aFo40dOnSxaQ7IBhaJl5w6meKmZkhkXj0CPT67McBsJ43Sy0AGFok%2BvaFVavgqc%2BSbPn4Y7Xy5uaGRCIhQfnb6afzBZXK589vSCT%2B%2BAMe98jLtvqFQtUCWFsbEokrV1A6%2BZ7Q4i6%2BhQWkpanHcXdXK1%2BpkiGR6N4dLl5UCqX/TT2xMTNTf18bY51VrI%2BNjSGRCAuDpCS1WKpNfVZWhkTi2jXDh5%2BKY8fUyhctakgktm%2BHu3fVYgFR3gOyXTZPHkN17t4FnU65Km%2BMJBPq/pFuTk9f4D7r6TvdeV7SVnblyhVmzpzJ4sWLTS5EXxT76biZvaN75coVRowYwbhx4%2BjcuTPW1tZ88skn6DR%2Bl1y4cIG33nrrueVlypRh7969/PLLLxw9epRJkyaxe/duvv766%2Be2nTp1KpcuXWLDhg2ULVuW69evPzd16Yv229zcnPQXvHusrKzIly8fZ8%2BezcaevVpmz4WXeXoAdnp6Ol27dqVcuXLPJRgv%2Bhs6nY6BAwdSr1495s6di52dHd988w2BgYHGss/%2Bnz99vOrWrctXX33Fo0ePuHjxItWrV8fV1ZXffvuN1NRU6tWrB8A333zDqlWrWLx4MbVq1cLCwoIGDRoY47z99ttUq1aNb7/9FmdnZ0qUKEHlx3fhmjZtyrFjxzh27Bjfffcd3bt3Z/z48XTv3j1TxwjAwcHhuda3pCRtLhb0eg3iXL%2BuXpEnoqLU42lxcQmGbybFWPHx2lTl4UMNYuVN1KQuJCdDokaxcgqtPicvXtQuVk6hmsU%2BkZSkHkulL8/THj1ST4gjI7Wpy927msTSonuRTifdlHK7f2QAtqOjI/fv3%2BfOnTvGZREREVhZWVE8E32dk5OTGTlyJH379n3ujrmTkxMREREmyyIiIoxdfbIiLCwMS0tLevfujbW1NXq9nrCwsCzHeZl79%2B6xceNG2rZt%2B9y6hw8fkpaWRv369fH392fLli0cOHDAOPj3aefOncPb25ty5cphZmZGaGjm7x46Ojry8OFDoqOjjcsOHz7Mr7/%2BipOTE4mJiVx/6sIsISEhwzpkh5OTE7Gxsdy7d8%2B4LCIiAhsbG5OWq8wyNzdn2rRpbN%2B%2B3WQAupOTE1evXjXZNiIiAicnJ2JiYoiMjKR3797GxPTChQvG7RwcHIiPjzcOsga4fPmy8d%2BOjo4UKFCA7du388477xi7Qp05c4bffvvN2Mpz7tw56tSpQ506dbCwsCAqKsrkPQDQrl079u/fz/79%2B/H29jYuj42NxdbWljZt2jB37lwmTpzI5s2bs3x8hBBCCPFiMgBb3T%2BSTLi4uFC%2BfHnmzp1LYmIiUVFRLF68mNatW2eqj/zMmTOxs7PLcPyAt7c3J06c4LvvvkOn0/Hjjz9y7Ngx2rVrl6m6WVlZce3aNeLj4yldujTJycmEhYVx//59Zs%2BejaWlJdHR0egVb8Omp6dz9uxZ%2Bvfvj7OzMz169Hhum%2BHDh/Pll1%2BSkJBg3L5w4cIUKlQIa2tr4uPjiYqKIjk5mTJlyhASEkJKSgq///473377LYBJgvAilStXpkqVKsyfP5%2BHDx8SHh7OZ599RnJyMu%2B88w6urq7MmDGDe/fu8eDBAyZPnqzZAwRr1KhB2bJlmTt3LklJSURGRrJkyRLatm2bYdevzKhUqRK9evXC39%2BfR4%2BboL29vdm1axfnzp0jNTWVLVu28Ndff9GqVSuKFi2KjY0NZ8%2Be5dGjR%2BzcuZNLly4RHx9PYmIiNWvWJF%2B%2BfKxcuZKUlBR%2B/fVXk6lowdDVae3atdSuXRswjE/5/fffCQ0NpW7dugCULl2aiIgIHjx4wI0bN5gxYwalSpUy6drXqlUrLl68yMGDB2nTpg1gmNWpefPmBAcHo9PpSEpK4sKFC8auU0IIIYQQOcU/kkyYmZkRGBhIdHQ0jRs3pnPnztSoUYNJkyZlqnxQUBC//fYbNWrUMJki9ubNm8YL37lz5%2BLu7s5XX33FnDlzMv3wss6dO7Nx40Z69uyJq6srPXr0oGfPnrRu3ZrSpUszYcIEwsPDjdPUZlXt2rVxcXGhRo0ajB8/niZNmrB69WqTh6498fnnn/P333/j6emJu7s769evZ9GiRZibm1O3bl3KlClDs2bNOHr0KKNHj%2BbKlSvUqVOHefPmMXHiRJo3b46vr2%2BmWimWLFnCzZs3qV%2B/PoMHD8bX19c4xmDu3Lno9XqaNm1K8%2BbNSUtL44svvsjW/j/L3NycxYsXc%2BvWLRo1akSXLl1wc3Pjs88%2BU4o7bNgwUlNTjV3CfHx86N%2B/P2PGjMHDw4MtW7awevVqHB0dsbS0ZPLkyQQGBvLuu%2B/y%2B%2B%2B/s3DhQooVK0aLFi3Ily8fixYt4sCBA7i7u7NkyZLnnu/h4eHB1atXjQO87ezsKFCgAPb29saxFz169KBUqVJ4enoyePBgevfuTbdu3Vi2bBnffPMNAEWKFMHT05OKFStSunRpwDDj14IFC1ixYgW1a9emcePG3L17F39/f6VjJIQQQghT0jKhzkyvestdCKGka9eudOnSxTjj2OukOo7RzMwwnjY5WX3MhM3oVz/5/ZUcHQ2zQs2apT5mYsYMtfIWFlCwIDx4oDxm4sCvRZTKFyhgmBHqp5/Ux0y8b3fq1Ru9TL58ULUqhIZqM2ZCi1mCtBqArTonpqurYUYoNzflMRP6dPWvck0HYB//US1A/vyG43LmjPqYiacmG8kWa2vDLEx//qk%2BZmLnTrXyJUrAgAGwfLkmYyZufDQx22Xz5jXMgREVpc2YiTf1vFkth%2B89ocXM5/8muf4J2EK8Kenp6WzcuJGYmBhatWr1pqsjhBBCCJFlkkxkw759%2B146hsDd3d3kQXv/K1auXMn8%2BfNfuN7Hx4fp06dnO/6yZctYuHDhC9e3b9%2BeqVOnZjt%2BTpKWlkbNmjUpV64cX3/9NVaaPLhBCCGEEFmRG7slaU2SiWxo2bIlLVu2fNPV%2BMf169ePfv36vbb4AwcOZODAga8tfk5iYWFBSEjIm66GEEIIIYQSSSaEEEIIIUSuJC0T6iSZEEIIIYQQuZIkE%2Br%2BkalhhRBCCCGEEP97pGVCCCGEEELkStIyoU5aJoQQQgghhBDZIi0TQgghhBAiV5KWCXWSTAghhBBCiFxJkgl1Znq9Xv%2BmKyGE%2BIdo8XY3M9MmTmSkeow8ecDeHmJiQKdTCjVoSkml8o6O4O8P06fD9etKoVB93IqNDVSpAhcuQFKSWqwqVdTKm5mBtTUkJ2tz2sTFqZXX8JShRAm18qDd28nM3EwtgKsrnDkDbm5w9qxyfdJ06jtlYQFpacphaNBArfw778DatfDhhxAerhbr590xagHy5IEiRSA2Vv0EBtLs7JXKa/V/9CTWmxAWpn3MypW1j5mTScuEEEIIIYTIlaRlQp0MwBZCCCGEEEJki7RMCCGEEEKIXElaJtRJMiGEEEIIIXIlSSbUSTcnIYQQQgghRLZIy4QQQgghhMiVpGVCnbRMCCGEEEIIIbJFWiaEEEIIIUSuJC0T6iSZEEIIIYQQuZIkE%2Bqkm5MQQgghhBAiW6RlQgghhBBC5ErSMqFOWiaEEEIIIYQQ2SItE0IIIYQQIleSlgl1kkwIIYQQQohcSZIJddLNSfwjbt68iYuLC1evXgWgYsWK/PDDDwB4eXmxadOmV8bo1asXc%2BbM0axOfn5%2BjB8/XrN4QgghhBCvU1xcHCNHjqR%2B/fo0aNCAzz77jOTk5Bduf/DgQby9vXF1deX9999n8%2BbNxnULFy6kcuXKuLi4mPzcuXMnS3WSZEJo4kUJwaZNm/Dy8qJ06dKEhITw1ltvvYHavX7nz5%2Bnd%2B/e1KpVi4YNG7Jy5co3XSUhhBBCvEJ6uvY/r9PEiRNJSkoiODiYbdu2ceXKlRfeaD137hxjxoxh%2BPDhnDp1igkTJjBt2jROnz5t3MbHx4eQkBCTn2LFimWpTpJMCKEoLi6O/v37U6NGDY4fP86qVavYsGED%2B/bte9NVE0IIIcRL/JuSiTt37nD48GH8/Pyws7OjePHi%2BPr6sm3bNlJTU5/bPi4ujkGDBtGsWTPy5MlDo0aNeOedd0ySCS3ImAnxj7hx4wZNmzZl7969lC9f/oXb/fHHH8yYMYM///wTS0tLmjVrxsSJE7G2tgYgLS2NSZMmERwcjJWVFRMnTqRVq1YAhISEMGvWLMLDw7G0tKR58%2Bb4%2B/uTN29eADZv3sySJUu4f/8%2B3t7epD/zjl%2B/fj0bNmzg1q1blClTBj8/P5o1a/bKffv99995%2BPAhI0eOxMLCggoVKtCvXz%2B2bt1Ky5YtAVizZg3r16/n7t27lChRAj8/P9577z0Axo8fj42NDWlpaezZswc7Oztmz57N6dOnWb16NQCffPIJ7du3z9Ixj46OJiYmxmSZfbFiODg4ZCnOa5NHg4%2BfJzE0iOXoqFa%2BRAnTVxU2NmrlH79djK8qzMy0Ka8a5wnV/2oNT5mcxdVVrXylSqav/0PeeUetfNmypq9KVE88CwvTV5EjZfj9a2%2Bv/P0bFhaGhYUFFStWNC6rWrUqiYmJREREmCwH8PT0xNPT0/i7TqcjJiaG4sWLG5ddunSJrl27Eh4eTsmSJfn0009p0KBBlur1v/ZxKv7lxo4dS//%2B/enQoQN37tzB19eXoKAgPvzwQwCCg4OZOXMm/v7%2BBAQEMGXKFN577z3y5MmDn58f3t7e/N///R9RUVF07doVZ2dnevXqRUREBJMmTSIgIABPT092797N9OnTadGiBWDoUxgQEMCKFSuoVKkSR48eZeTIkRw8eJBSpUq9st5mz1wpFSpUiLCwMABOnTrF3Llz2bZtGxUqVGDHjh2MGTOGY8eOYWdnB8DevXv54osv%2BOyzzxg6dCijRo2iU6dOfP/996xYsYKZM2fSrl07zM0z35gYFBREQECAybKhH3/MsOHDMx3jhbS4MrS3V4/xRJEiyiH8/TWoB9C/vzZxtPD222%2B6Bv9lZaVNHC0SJNDklNGMJonWmTMaBAE2btQkjFaXulpcM69dqx4DYNo0LaJodOIVLKhJGC3%2Bn7T4P0pLU4%2BRXa%2BjJSHD79%2BhQxk2bJhS3Li4OGxtbU2uOQoVKgRAbGzsK8vPmTOHfPnyGW/ClihRAkdHR0aPHo2DgwNBQUEMHjyY3bt383YWvkAkmRCamT59OjNnzjRZlp6ebpIBv8qDBw/Ily8f5ubmODg4sHnzZpMLaDc3Nxo2bAhAixYtWLp0Kffu3cPBwYGdO3diaWmJhYUFpUqVwt3dnfPnzwNw%2BPBhqlSpYmxp6NixI2uf%2BobZunUrHTt2pFq1agC899571KpVi%2BDgYAYOHPjSOru6umJjY8OCBQsYMmQIMTExbNy4kfv37wNQq1YtTpw4QcHHH/5t2rTh008/JTw8nLp16wJQrlw5mjRpAsC7777LL7/8woABA7C0tKRJkyYsWLCAu3fvYp%2BFC/AuXbrg5eVlssy%2BWDHQ6zMdI0NmZuoxALI4wCtDefIYrgpjY0GnUwo1falaclOihCGRWLECIiOVQpHFRqjnWFsbEomICHjJuLxMUU1IzMwMicSjR9qcNvHxauU1PGXIYrfiDGn1djKr5aYWoFIlQyLRvTtcvKhcn7RT6smNhYU2F5l9%2B6qVL1vWkEhMmgR//60Wa%2B38V1/wvZSFhSGRePBAk4OTVlAtudHq/%2Bh/TYbfv5n8/t61axdjx47NcJ2fnx/6bHxg6PV65syZQ3BwMOvWrcPq8d2dTp060alTJ%2BN2ffr04dtvv2X37t2MHDky0/ElmRCa8ff3p1u3bibLNm3axPLlyzMdY9SoUUyYMIGVK1fSoEEDfHx8TLpFlSlTxvjvJ2%2BGlJQUAE6ePMmiRYv466%2B/0Ol06HQ6Y8tDVFSUSVkwXMA/ce3aNU6cOGGSYOj1epydnV9Z50KFCrFo0SK%2B/PJL1q9fT4UKFWjfvr0xkUlLS2PRokXs37%2Bfe/fuGcs9qTcY7g48vV92dnZYWloCGF8fPXr0yro8zcHB4fkmVS2uWrSieiX3bCzFeNeva1OVyEj1WElJ2tQlOVk9llanjF6vTSytThsNTpmc5exZbeJcvKhdrBwiPFybOH//rUEsrU66tLT/sRP4zXkdLRMZfv9mko%2BPDz4%2BPhmuO3HiBAkJCaSlpWHxuEkoLi4OgKJFi2ZYJj09nU8//ZRz586xadMmHF/Rp7d06dJER0dnqc6STIgcpVOnTjRr1oyjR49y5MgR2rVrx7x584wtCs92J3riypUrjBgxgnHjxtG5c2esra355JNP0D3%2BsE1JSTH%2B%2B4mnx0xYW1szevRo%2BmbzFlbt2rXZsmWL8fcDBw4YW2QWLVrEvn37WLJkCZUqVUKv11OlShWT8s92X8pKdyYhhBBCZM%2B/6TkTlStXRq/Xc/HiRapWrQoYxosWLFjwhbNlzpw5kz///JNNmzZRuHBhk3WBgYG4urpSr14947IrV64Yu0FlllyxiBwlNjaWIkWK0KFDBwIDAxk0aBBbt259ZbmwsDAsLS3p3bs31tbW6PV645gFMNwliHym78mVK1eM/3ZycuLSpUsm62/dupWp5sRHjx6xY8cOEhISjMtOnDiB6%2BMBkSEhITRt2pQqVapgbm5OaGjoK2MKIYQQQjzNzs6O999/n/nz53Pv3j0iIyNZtGgRHTt2JM/jwf0ffvghe/fuBeC3335j9%2B7dLFu27LlEAgytGlOnTiUiIoJHjx6xatUqrl27xgcffJClekkyIXKMyMhIvLy8OH78OOnp6cTHxxMeHo6Tk9Mry5YuXZrk5GTCwsK4f/8%2Bs2fPxtLSkujoaPR6PZ6enly4cIFjx46RkpLChg0biIqKMpbv0qULe/fu5dixY%2Bh0Ok6ePEmbNm34448/Xvm38%2BbNS0BAAIsXL0an03H8%2BHF2795tHDReunRpLl68SFJSEpcvX2bFihUUKFDA5O8LIYQQ4p/3b5oaFmDatGkUKFCApk2b4u3tTfXq1fHz8zOuv379unHM5rZt24iPj6dJkyYmD6V70gtj9OjReHp60qdPH9zd3QkODmbNmjUmXa8zQ7o5iRyjRIkSzJgxgxkzZnDr1i1sbW3x9PRkeCZmH3J1daVHjx707NkTGxsbhgwZwoQJExgyZAh%2Bfn7Mnz8ff39/pkyZwoMHD2jbti0tWrQwtjy8%2B%2B67jBs3jmnTpnHnzh3KlCnDlClTqFmz5iv/trm5OfPnz2fy5MmsX7%2BeEiVKMHv2bGMT5KBBg/Dz86Nu3bpUqFCBWbNmUbx4caZPn26czUkIIYQQ4lUKFCjAf/7znxeuP3r0qPHfM2fOfG5inKdZWVkxYcIEJkyYoFQnM312hoULIf6dNJk2RqPpZ1SnPALD1Dz29hATozwYcdCUkkrlHR0N08tOn64%2BAPsVE4i9ko0NVKkCFy6oD8B%2BZnhPlpmZGWaXSk7W5rR5PNYw2zQ8ZTR5pohmszmZK84v6%2BpqmF7WzU2TAdhpOvWd0mqmoCxOmf%2Bcd94xTC/74YfqA7B/3h3z6o1eRsvpyIA0O7VZ7LSczelNPTrjdTxf9vEjpnINaZkQQgghhBC50r9pAHZOJcmEEK9Qu3btl07Lun//fkqXLv0P1kgIIYQQImeQZEKIVzh9%2BvSbroIQQgghXgNpmVAnszkJIYQQQgghskVaJoQQQgghRK4kLRPqJJkQQgghhBC5kiQT6qSbkxBCCCGEECJbpGVCCCGEEELkStIyoU5aJoQQQgghhBDZIi0TQuQm9%2B6plbewgMKF4f599ceeXrumVh4gXz7D44wjIyExUSnU0vd%2BUqtL4cJAU/zrHYHKao9p7ruog1J5JyeYMgU2b1Y/zKu%2B1OCJvdZFsE7S5om9txLVnthraWl4TU6GlBS1upgd/1EtQP784OaG2dkz8PChUihNnjgNpJ06oxwHwCKPNk/ktnBXfyL3zx99pFYXJydgCmvfmgIWim%2BoxMlq5bU8gYFohbfkk6fJ37unyVubkiXVY2SHtEyok2RCCCGEEELkSpJMqJNuTkIIIYQQQohskZYJIYQQQgiRK0nLhDppmRBCCCGEEEJki7RMCCGEEEKIXElaJtRJMiGEEEIIIXIlSSbUSTcnIYQQQgghRLZIy4QQQgghhMiVpGVCnbRMCCGEEEIIIbJFWiaEEEIIIUSuJC0T6iSZEEIIIYQQuZIkE%2Bqkm5MQQgghhBAiW6RlQgghhBBC5ErSMqFOWiaEpm7evImLiwtXr14FoGLFivzwww8AeHl5sWnTplfG6NWrF3PmzNGsTn5%2BfowfP16zeEIIIYQQwkCSCZElL0oINm3ahJeXF6VLlyYkJIS33nrrDdTu9fPy8sLT05PExEST5b/88gteXl5vqFZCCCGEyI70dO1/chtJJoTIopSUFAIDA990NYQQQgihSJIJdTJmQmjqxo0bNG3alL1791K%2BfPkXbvfHH38wY8YM/vzzTywtLWnWrBkTJ07E2toagLS0NCZNmkRwcDBWVlZMnDiRVq1aARASEsKsWbMIDw/H0tKS5s2b4%2B/vT968eQHYvHkzS5Ys4f79%2B3h7e5P%2BzDt7/fr1bNiwgVu3blGmTBn8/Pxo1qxZpvdx2LBhzJkzhw4dOrywBSYyMpKpU6dy5swZdDodnp6eTJ48mcKFCwNw%2BvRpvvrqK/7880/y589Phw4dGDFiBObm5ixcuJALFy7g5ubGmjVrSElJwcfHB39//0zXESA6OpqYmBiTZfaWljjY22cpjgkLC9NXFfnyqcd4fL4YX1U8/r/JtgIFTF8VODmplS9Z0vRVSR7FrwktzxnA0lKt/OOPCeOrkvz51crb2Ji%2B/i9xdVUrX6mS6auKnPSGUj2Bn7wfVd%2BXz4R701XR6dRjiDdHkgnxRowdO5b%2B/fvToUMH7ty5g6%2BvL0FBQXz44YcABAcHM3PmTPz9/QkICGDKlCm899575MmTBz8/P7y9vfm///s/oqKi6Nq1K87OzvTq1YuIiAgmTZpEQEAAnp6e7N69m%2BnTp9OiRQsADh48SEBAACtWrKBSpUocPXqUkSNHcvDgQUqVKpWpujs7O9O5c2emT5/OypUrM9zG19cXZ2dnjhw5QnJyMiNGjGDy5MksWLCAO3fu0K9fP8aOHUunTp24fPkyAwYMwMHBgR49egBw5swZqlevznfffcdvv/1Gnz598Pb2pnr16pk%2BxkFBQQQEBJgsG/rxxwwbPjzTMV5Igwtm5Yv3p1WooF0sVXXqKIeY0lSDegCDBmkRpYgWQaBgQU3COGpUnRIlNAji6KZBEKByZeUQ2qRqmuV8cOaMNnE2btQmjha0eUNpQ%2BWm0NNhNIhRRIP35O3b6jGyKze2JGhNkgmRZdOnT2fmzJkmy9LT0ylevHimYzx48IB8%2BfJhbm6Og4MDmzdvxtz8v73u3NzcaNiwIQAtWrRg6dKl3Lt3DwcHB3bu3ImlpSUWFhaUKlUKd3d3zp8/D8Dhw4epUqWKsaWhY8eOrF271hh369atdOzYkWrVqgHw3nvvUatWLYKDgxk4cGCm6z9s2DBatGjBoUOHaN68ucm6sLAwQkNDWbp0Kba2ttja2jJw4EA%2B/vhjUlJSCA4OplSpUsbEoUqVKvj4%2BLBv3z7jMgsLCwYNGoS5uTn16tXDzs6OK1euZCmZ6NKly3PjOOwtLSEuLtMxnmNhYUgk4uMhLS37cQCuX1crD4YWiQoV4M8/ITlZLVZ0tFr5AgUMicSvvxqOj4IpP6plEyVLGq57li5V/5KeMiJWLYCFhSF438%2BYAAAgAElEQVSRePBA/ZwBrieoXbnkzWtIJCIjITVVrS6OMYoXzDY2hkQiLAySkpRCpdVQT2wsLDT5LzLEclesT6VKhkSie3e4eFEtlre3Wnkt31CqCUmePIZEIiZGk9v5MXmy39qSJ48hkYiNlZaF3E6SCZFl/v7%2BdOvWzWTZpk2bWL58eaZjjBo1igkTJrBy5UoaNGiAj4%2BPSbeoMmXKGP9tZWUFGMYqAJw8eZJFixbx119/odPp0Ol0xpaHqKgok7IA5cqVM/772rVrnDhxwiTB0Ov1ODs7Z7ruALa2towZM4ZZs2YZk54nbty4QaFChbB/6s6Rk5MTqampREVFcePGjee6gJUtW5Z9%2B/YZfy9VqpRJcmVjY0NyFi%2BWHRwccHBwMF149642VwtpaepxnhnEriQ5WT2eSpL1tPh45VjXrmlTldu3NYil1VVCWpomsR5/DChLTdUg1sOHmtSFpCTtYuUUZ89qE%2BfiRfVYNWtqUxct3lBancA6nSaxtHh363T/7mRCWibUyQBs8UZ06tSJY8eO0aNHDy5fvky7du04fPiwcb2ZmVmG5a5cucKIESP44IMP%2BPnnnwkJCaFNmzbG9SkpKeie%2BVR7esyEtbU1o0ePJiQkxPhz/vx5Jk6cmOV9aNeuHcWLF2fp0qUmy1Ne8gFvZmb2wvVP7/PTiYQQQgghXg8ZgK1OrljEGxEbG0uRIkXo0KEDgYGBDBo0iK1bt76yXFhYGJaWlvTu3Rtra2v0ej1hYWHG9Q4ODkRGRpqUuXLlivHfTk5OXLp0yWT9rVu30Ov12dqPSZMmsWbNGq4/1WXH0dGR%2B/fvc%2BfOHeOyiIgIrKysKF68OE5OTkRERJjEiYiIwNHRMVt1EEIIIYR4UySZEP%2B4yMhIvLy8OH78OOnp6cTHxxMeHo5TJmbcKF26NMnJyYSFhXH//n1mz56NpaUl0dHR6PV6PD09uXDhAseOHSMlJYUNGzYQFRVlLN%2BlSxf27t3LsWPH0Ol0nDx5kjZt2vDHH39ka18qV65Mu3btmD9/vnGZi4sL5cuXZ%2B7cuSQmJhIVFcXixYtp3bo1efPmpWXLlly/fp2goCB0Oh3nzp1jx44dfPDBB9mqgxBCCCGyR1om1EkyIf5xJUqUYMaMGcyYMQNXV1datGhB/vz5GZ6JWYZcXV3p0aMHPXv2pHXr1pQuXZoJEyYQHh6On58fNWrUwN/fnylTplC3bl3Cw8ON4ykA3n33XcaNG8e0adNwc3Nj2rRpTJkyhZoKfWpHjhxp0rXKzMyMwMBAoqOjady4MZ07d6ZGjRpMmjQJMCREAQEBBAUF4e7uzieffMKIESNo165dtusghBBCiKyTZEKdmT67/TuEEP8%2Bd%2B%2BqlbewMEzpGhenPgD78mW18mB4VoWLC4SEqA/AvnFDrXzhwtC0KRw5ojwAu%2B%2B3HZTKOznBlCmGH9Xxoqu%2BjHn1Ri%2Bj8ZQvVx6oTWZpaQmOjobJxFTHr5a/9aNagPz5wc3NMI2q4gDstPoNX73RK2g6m1OejMe9ZZqrq%2BG4uLmpD8D%2B6CO18lq%2BoSZPVitvaWmYXer2bU0GYN%2B2LJvtshpPLKXNc3GyYe5c7WOOHq19zJxMZnMSQgghhBC5Um5sSdCaJBNCPFa7dm0ePXr0wvX79%2B%2BndOnS/2CNhBBCCCFyNkkmhHjs9OnTb7oKQgghhPgHScuEOkkmhBBCCCFEriTJhDqZzUkIIYQQQgiRLZJMCCGEEEKIXOnfNjVsXFwcI0eOpH79%2BjRo0IDPPvuM5OTkDLfdvn07lSpVwsXFxeTn3Llzj/c9nXnz5tG0aVPc3d3p16%2BfyUN4M0uSCSGEEEIIIf4FJk6cSFJSEsHBwWzbto0rV64wZ86cF27v7u5OSEiIyU/16tUB2LBhA3v27GHZsmV89913lCtXjo8//pisPjVCkgkhhBBCCJEr/ZtaJu7cucPhw4fx8/PDzs6O4sWL4%2Bvry7Zt20hNTc1yvKCgIPr06UP58uWxtbXFz8%2BPK1eu8Mcff2QpjiQTQgghhBAiV/o3JRNhYWFYWFhQsWJF47KqVauSmJhIREREhmVu377NRx99hLu7O02bNmXXrl0AJCcnc/nyZapUqWLc1tbWlrJlyxISEpKleslsTkLkJtbWauXNH99/sLJS/8QsV06tPBgewQpQqpT6I1gDAtTKly1reAL2d9/B338rhWrdTu0J2IULG14bNlR%2BGHeOU6qUWnmzxw9mtreHLLbkPy%2BxsFr5J%2B/HAgUgb16lUA0aqFXlnXdg7Vro2xfCw9ViAfysxVOnAby9oWZNtVirV6uVd3U1PP169271p3EvXKhW/slncOHCmly12iqEeFKVfPlkRqRnRUdHExMTY7LM3t4eBwcHpbhxcXHY2tpiZvbfJ8wXKlQIgNjY2Oe2t7Ozo1y5cowaNQpnZ2cOHTrE2LFjcXBw4O2330av1xvLPx0vo1gvI8mEEEIIIYTIlV5HIhQUFETAMzeohg4dyrBhw15ZdteuXYwdOzbDdX5%2Bflkaz9C4cWMaN25s/L1169YcOnSI7du3M2bMGIAsj4/IiCQTQgghhBBCaKRLly54eXmZLLO3t89UWR8fH3x8fDJcd%2BLECRISEkhLS8PCwgIwtFYA/8/evcfleP8PHH/dnRRJihIpvs7DknMhh7Q5h80hZjbn0yTfHcyM5LSDhpXz7Mswk81hzGHFcjZ88ZWUTTYqU6RQOt7dvz9a98%2BtkK67u6z38/HoUV3Xfb2v932%2B3tfncGFra1uk%2BLVq1eLSpUtYW1tjZGSk3T5fSkpKkWPlk2JCCCGEEEKUSyXRMmFnZ6e4S1NhmjRpgkajITo6mqZNmwIQERGBlZUVdevWLXD7LVu2UKVKFXr16qVdFhMTQ%2B3atalQoQINGjQgMjKStm3bAnD//n1u3Lihne2pqGQAthBCCCGEKJdepAHYNjY2vPrqqyxdupS7d%2B9y69Ytli9fzuuvv47J32MIR44cyd69ewHIyspi3rx5REREkJ2dzZ49ezhy5AhDhw4FwMfHh2%2B%2B%2BYaYmBhSU1NZvHgxTZo0oXnz5s%2BVl7RMCCGEEEII8QIICAhgzpw5eHp6YmpqSp8%2BffDz89Ouj42N5d69ewC8%2BeabpKWl4evry%2B3bt3F0dGT58uU0a9YMgKFDh3L79m1GjBhBWloa7dq1KzDWoyikmBBCCCGEEOXSizYTVeXKlfniiy%2BeuP7QoUPav1UqFZMmTWLSpEmF3lalUjF16lSmTp2qKCfp5iSEEEIIIYQoFmmZEEIIIYQQ5dKL1jJRFkkxIYQQQgghyiUpJpSTbk5CCCGEEEKIYpGWCSGEEEIIUS5Jy4Ry0jIhhBBCCCGEKBYpJoRBxMfH07x5c/744w8AGjVqxJEjRwDo1q0bW7ZseWaMESNGsHjxYr3l5Ofnx4wZM/QWTwghhBAvlhfponVllRQTQi%2BeVBBs2bKFbt26UatWLSIiIgq93Ps/QXR0NCNHjqRVq1a4u7szbdo0bt%2B%2BXdppCSGEEOIppJhQTooJIRTKyspi1KhRtG3blpMnT7Jnzx6SkpLw9/cv7dSEEEIIIUqUFBPCIOLi4mjUqBExMTFPvd3//vc/Bg8ejKurK%2B3ateOjjz4iIyNDu16tVjN79mxatmyJm5sbe/fu1a6LiIhg2LBhtG7dGnd3d%2BbMmUN2drZ2fUhICN26daNVq1bMnTuX3MdOH2zatImePXvi4uJC7969CQsLK9J9S09Px8/Pj/Hjx2NmZoaNjQ1eXl78/vvvAGzfvh0vLy%2B2bdtGp06daNGiBbNnzyYnJweAoKAgJkyYQFBQEG3atKFjx46EhYWxfft2OnfuTJs2bVi5cmWRchFCCCFE0UnLhHIym5MoU95//33GjBnDa6%2B9xp07d5g0aRJbt25l5MiRAOzZs4eFCxcya9YsgoOD8ff355VXXsHExAQ/Pz/69evHxo0bSUhIYOjQodSvX58RI0Zw7do1Zs%2BeTXBwMB4eHvz444/Mnz%2BfHj16APDzzz8THBzMV199RePGjTl06BDTpk3j559/pmbNmk/NuUqVKgwaNEj7/7Vr19ixYwc9e/bULktISCAiIoKff/6ZmzdvMnLkSOrVq6e9X%2BfPn6dz584cP36cefPm4e/vj5eXFz///DP79u1j5syZDB48GFtb2yI/lomJiQW6WlW3tMTOzq7IMQpQqf7/t5HCcxEmevj4MTbW/a2Es7Oy7R0cdH8rYG2tbPvKlXV/K6L0edLnc8T/vwSVbq80DgDm5sq2r1BB97cCDRsq2z7/5a/0baDl5KRsez2%2Bn3B1VbZ948a6v5VQ%2Brmpz89glJ1Rzt%2B9HtIolwfg/yRSTAi9mT9/PgsXLtRZlpubi729fZFj3L9/n4oVK2JkZISdnR0hISEYPfJJ1bJlSzp16gRAjx49WL16NXfv3sXOzo6dO3diZmaGsbExNWvWpE2bNly6dAmAsLAwXnrpJbp37w7A66%2B/zoYNG7Rxv//%2Be15//XWaNWsGwCuvvEKrVq3Ys2cP48aNK1Lu8fHxvPrqq%2BTk5DB48GCmTp2qXZeZmcm0adOwsLCgXr169O7dm/DwcG0xYWpqio%2BPDwCdO3cmJCSEcePGUaFCBbp164ZarSY2Nva5iomtW7cSHByss2zK5Mm880hexab0IArAwkJ5jHxKj74B5s9XHgNg8mTFITz1kAZA27b6iFJVH0HAykovYfTwygP0cvwODRroIQjKD7yBRz7OFAkI0E8c8NdPmPHjlcfQV5fTb7/VTxx90MdnMFBJDzH08VH%2B4IHyGMUlhYxyUkwIvZk1a5b2gDjfli1bWLt2bZFjTJ8%2BnZkzZ7Ju3To6duyIt7c39erV0653dHTU/l3h76OBrKwsAE6dOsXy5cv5888/ycnJIScnR9vykJCQoLMtQJ06dbR/37hxg%2BPHj%2BsUGBqNhvr16xc59/xB5tevX2f27Nm8//77BAYGAnmtFzY2Ntrb1qxZk2PHjmn/r1GjhvZvMzMzAG0Rln8/MzMzi5wLwJAhQ%2BjWrZvOsuqWlpCe/lxxdKhUeV9iGRmg0RQ/DsDDh8q2h7yz3dbWkJICarWyWEuWKNvewSGvkFi%2BHP76S1Gog12VFTaVK%2BcVEqdPK/%2BS9myZrCyAsXFeIXH/vvLnCMiwUFbcqFR5hURmpvKXsHns78oCVKiQV0jcuJGXkAIj5ysrbJyd8wqJ2bPh%2BnVFoQDYUNdfWQAHh7xCYvVqxe8nfvxR2faNG%2BcVEsOGQXS0sljHjyvbXp%2BfwUBabvErASOjvEIiPf3FPiB/kXMvK6SYEGXKoEGD6N69O4cOHeLgwYP079%2BfJUuWaFsUVE/omxATE4Ovry8ffPABgwcPxtzcnPfee087LiErK0v7d75Hx0yYm5vz73//m1GjRinKX6VSUadOHfz8/Bg6dCgfffQRkDfW41EajUbnvhgV0k5c2LLnYWdnV7BLU1qask/O/Jw0GuWfwI89H4qo1crj6eMICvIOfBTGSknRTyoPHughlr6eJ308R%2Bjl%2BEkbR3GsR8ZzKZKZqTjWb7/pJ5Xr1/UUy/iGHoKQ9366oTDW%2BfP6ySU6WnkspZ%2Bb%2BvwM1kM6%2BTHkgLx8kwHYokxJTk6matWqvPbaa6xYsYLx48fz/fffP3O7qKgozMzMePPNNzE3N0ej0RAVFaVdb2dnx61bt3S2eXQwuJOTE1euXNFZf/PmTTRFONo4efIkr776qk5xkl8ImJqaApCamsrdu3d1Yj9P9y8hhBBC6J8MwFZOiglRZty6dYtu3bpx7NgxcnNzefDgAb/99htORehPXKtWLTIyMoiKiuLevXt8/vnnmJmZkZiYiEajwcPDg8uXLxMeHk5WVhabN28mISFBu/2QIUPYu3cv4eHh5OTkcOrUKfr06cP//ve/Z%2B67WbNmpKam8vnnn5Oens7du3cJCgqidevWVP57BKyZmRnLly8nIyODq1ev8tNPPxXogiSEEEII8aKRbk6izKhRowYLFixgwYIF3Lx5E0tLSzw8PHQGMj%2BJq6srw4cP54033sDCwoKJEycyc%2BZMJk6ciJ%2BfH0uXLmXWrFn4%2B/tz//59%2BvbtS48ePbQtDx06dOCDDz4gICCAO3fu4OjoiL%2B/Py1atHjmvitXrszXX3/N/Pnzad%2B%2BPRUrVqR9%2B/YsWLBAexsrKysaNmyIl5cXDx48oF%2B/fgwdOrT4D5YQQgghFCuPLQn6ptIUpR%2BHEKLYtm/fTmBgIMeVDrzTh7Q0Zdvrc8Rdaqqy7SFv2lJbW0hKUt4f/913lW3v7Jw3I9SsWYrHTPzQf6Oi7a2twdMTDh5UPmbiNQ%2BFV3I3MYGqVSE5WS9jJtItqyvaXp/jVy2uRigLYG6eNyPU778rHjPhNq65ou0bNsybEWrkSP2MmTjZRNn4M5yc8mZh8vdXPmbiP/9Rtr2rK5w7By1bKh8zofRzT8%2Bjnh/kFn8%2BJyMjqFRJ%2BVC8fHqZyroYRo/Wf8x16/QfsyyTbk5CCCGEEEKIYpFuTkI8Q%2BvWrZ86Lev%2B/fupVauWATMSQgghhD5INyflpJgQ4hnOnj2raPuBAwcycOBAPWUjhBBCCFF2SDEhhBBCCCHKJWmZUE6KCSGEEEIIUS5JMaGcDMAWQgghhBBCFIu0TAghhBBCiHJJWiaUk2JCCCGEEEKUS1JMKCfdnIQQQgghhBDFIi0TQgghhBCiXJKWCeWkmBCiHEnOqqRoe2NjsLKA%2B9kWqNXKcqn67RplAQDs7GD4cNi/HxITFYV6sGKjou2NjKASkPbhfMVfTnYXlG1f6e%2BnuWpVMDNTFos7d5Rtb26el0hKCmRkKEwGzGyqK44BYGqqhyA7dyrbvkYNaNAAwsPh1i1FoU7%2BWENZLiYmQFU2LE2GnBxlsQAezlG2ff4Ld/x4yMpSFisoSNn2Rn934jh%2BXPmRp6Wlsu1dXeHcOejQAc6fVxYLeHhLU%2BxtTUzyPmsyMvTzkqlcWXkMUTqkmBBCCCGEEOWStEwoJ8WEEEIIIYQol6SYUE4GYAshhBBCCCGKRVomhBBCCCFEuSQtE8pJy4QQQgghhBCiWKRlQgghhBBClEvSMqGcFBNCCCGEEKJckmJCOenmJIQQQgghhCgWaZkQQgghhBDlkrRMKCctE0IIIYQQQohikZYJIYQQQghRLknLhHJSTAghhBBCiHJJignlpJuTEEIIIYQQolhKvZiYNWsW77//fmmnoWP79u106NBBrzFXrFjBG2%2B8odeY5VlsbCzNmzcnNja2tFN5opEjRxIcHFzaaQghhBDiCXJz9f9TklJSUpg2bRru7u507NiRjz76iIyMjEJvO2vWLJo3b67z89JLL/Hhhx8CMGPGDF566SWd9a1bt37unJ6rm1O3bt0YO3YsPj4%2BOsu3bNnC2rVrOXTo0HMnMH/%2B/OfepjhOnjyJpaUlzZs312vcX3/9lTfffBMzMzMAjI2NqVOnDj169OCtt97C3NwcgEmTJjFp0qRnxlOr1XzzzTe8/fbbes1Tn86cOcOoUaO0/2dlZWFqaopKpQLA29tb8fO6bds2Zs2apX1cHzVv3jz69%2B9PRESEon08zZIlS1i9ejWmpqYAqFQqHBwceO211xg9ejTGxsbPjLFhw4Yi7y8iIoLU1FTc3NyKnbMQQggh/tk%2B/vhjsrKy2LNnD9nZ2fj6%2BrJ48WJmzZpV4Lbz58/XOR7Lycmhf//%2B9OjRQ7ts4sSJvPPOO4pyKjdjJtavX0%2BXLl30XkzkO3v2LBUqVOD%2B/ftERkayZMkSDhw4wKZNm6hUqVKR41y%2BfJmvvvqqTBcTbdq00TmQb9SoEStWrMDDw0Ov%2B7G3t%2BfIkSN6jfk8XF1d2bJlC5BX5EVERDBlyhSMjY0ZPXq0Xve1bds2qlatKsWEEEIIYUAv0piJO3fuEBYWxo4dO7CxsQHyTlb7%2BvrywQcfaE%2BAPsmGDRuoWbMmnTt31mteei8mbt26xdy5czl37hw5OTl4eHgwZ84crK2t%2BfXXX5kwYQK%2Bvr58%2BeWXrFu3jq1bt5KZmcmSJUsYNWoUZ86c0cZSq9XUqFFD2%2BIRFhbGl19%2ByY0bN7CxseGtt97izTffBPKaaipVqoSJiQk7duzAyMiI0aNHM3bsWCZMmEB4eDjHjh1j//79bNiwgYiICBYtWsRvv/2GmZkZXl5ezJo165lPxLNYWVnh5uaGi4sL3t7erFu3jqlTpxIUFMTRo0cJCQkhPT0df39/jhw5QkZGBo0aNWLWrFnk5ubi4%2BNDTk4OzZs3Z%2B3atbRr147AwEB2797N/fv3qVOnDjNnzqRNmzYAjBgxgg4dOhATE8PBgwepVKkS7777Lt7e3kBed6A5c%2BZw/vx5rK2tefvtt7WPWXx8PPPmzeP8%2BfPk5ubStWtXZs%2BejaWlpaLHIN/NmzcJCAjg/PnzqNVqOnfuzOzZs6lSpYri2NevX%2BeVV17h559/xtnZGQ8PD4YNG0ZISAhdu3bl448/JjIykk8%2B%2BYSoqChMTU3p168f7733HiYmz/%2ByNzY2pkWLFgwdOpTQ0FBtMXHgwAGCg4O5ceMGtra2jB49muHDhwPg4%2BND27Zt8fPzY8mSJVy7do1mzZqxYcMGsrOzGThwIB9%2B%2BCFz5swhJCQEIyMj9u/fz4EDB9i2bRtr164lISEBW1tb3nzzTd56663nyjkxMZHbt2/rLDM3r0716nbPff/zGRnp/lbErvh5aFWtqvtbAaX3SZ%2BPzXOcfyiUhYXub0X%2Bbl0ttvzWxUJaGV94NWoo297WVve3EsX4XNOR39pahFbXIlH6fOffH6X3C5S/Kf9udUelUh7L1VXZ9o0b6/5WSMnDq8%2BXTE6O8hjFVRLFRGHfv9WrV8dO4fdeVFQUxsbGNGrUSLusadOmPHz4kGvXruksf9z9%2B/dZtWoV3377rc7yU6dOcfDgQa5fv069evXw9/enWbNmz5WX3ouJSZMmUb9%2BfQ4ePEhGRga%2Bvr7MmTOHZcuWAZCdnc3169c5ceIEFSpUYOvWrdptv/76a%2B3fqampDBw4kNdffx2A6OhofH19WbZsGZ07d%2Bbs2bNMmDABZ2dnbYW1Z88eZsyYwfHjx/nxxx/5%2BOOP8fb2ZtWqVQW6aPn5%2BdGvXz82btxIQkICQ4cOpX79%2BowYMUIvj0PFihUZPHgwO3bsYOrUqTrrNmzYwJ07dwgNDcXMzIy1a9fy8ccfs2PHDubNm0dgYCDHjx8HYOfOnezcuZPvv/%2Be6tWrs3LlSqZOncqxY8e0XW02b97MwoULWbhwIatWrSIgIIBevXphamrKlClTaNu2LcuXL%2BfPP/9k%2BPDh1KtXD3d3dyZNmkTLli1ZsmQJDx8%2BZPr06Xz66afMmzdP8f3XaDRMnDiRl156iYMHD5Kens4777xDQEAAgYGBiuMX5qeffmLDhg04OjqSlpbG2LFjefvtt/n666/566%2B/mDhxItWqVWPs2LHF3odarcbo7y%2BTyMhIpk%2BfTlBQEJ06deLMmTPa12THjh0LbHv69GmaN29OeHg4p0%2BfZvTo0fTr14%2B5c%2Bfy22%2B/aQuPuLg4Fi5cSEhICA0aNODixYuMGTMGNze3p35QPG7r1q0FxmxMnjyFqVOVNWcC6KXe/Lvo0otevRSHUHj8rqWPA/iWLZXHAGjSRB9R6uojCNSqpZcwejrU1c8xs4LPEh0DB%2Bonjj5YWZV2BrqqVy/tDP6f0sIa4Nw55TEAHjsgLC49lLFYWyuPkZCgPEZZUtj375QpUxR3J0pJScHS0lLbrRzQnqBNTk5%2B6rabNm2iTZs2NGjQQLusdu3aGBkZ4evrS6VKlQgODmbUqFEcOHCAqs9xku65i4n58%2BezcOFCnWW5ubnY29sTFRVFZGQkq1evxtLSEktLS8aNG8fkyZPJysoC8oqJYcOGaccSPMmcOXNwcnLSHvj98MMPuLm50b17dwDc3Nzo0qULe/fu1RYTjo6ODBgwAIBevXoxc%2BZM/vzzz0IrwZ07d2JmZoaxsTE1a9akTZs2XLp06XkfjqeqW7cucXFxBZbfv38fU1NTzM3NMTExeep4ir59%2B%2BLp6UnlypUB6N27N0FBQdy8eZPatWsDed1xOnXqBEDPnj0JDg4mMTGRe/fuceXKFTZs2ICFhQVNmjQhODgYe3t7IiIi%2BP3339myZQsWFhZYWFjwzjvvMHr0aAICAnReqMVx6dIlrly5wn/%2B8x/ta2Hs2LFMmzaN7OxsxS1AhenSpYv2Mfnll18wMTHRvn6cnJwYNWoU69evL1Yxkd/Nadu2bUyYMAHIe0126tSJbt26AeDu7o6Hhwd79%2B4ttJgwMzNj7NixqFQqOnbsSJUqVYiJiaFp06Y6t0tNTUWj0VCxYkUAXn75ZU6dOqUtYopqyJAh2tzymZtX5/795wqjw8gor5BITVV%2BNsdq92ZlASCvRaJXL9i7F57xQfosaf2VFTdGRnmFRHq68sfmyhVl21tY5BUSUVF5%2BSjRsuofygKYmeUVEvHx8Pf3gBJqJ%2BXFjbExqNWKw2D89VplAWxt8wqJ7dshKUlZrL9PvBWbsXFeIXH/vn4enCcMCC0yE5O8QuL2beWnrZUe7apUeYVERgZoNMpiKZ3cpXHjvEJi2DCIjlYWC0gKLX5xY2yc99CmpOjnJVNaSqJlorDv3%2BpFLIx37dr1xImJ/Pz80BTjNahWq9m8eXOBk7mTJ0/W%2Bf%2B9995jz549hIWFMWjQoCLHf%2B5iYtasWU8cgB0XF0eVKlV0HjAnJyeys7NJeKTsrFmz5lP38f3333P69Gl27dqlPaiNi4ujXr16Ordzdnbm3CNVvqOjo/Zvi79PDz5phPupU6e0Z%2BtzcnLIycnRGZCiD2q1utCBusOGDWP06NF07tyZTp060b17dzw9PQuNkZ6ezsKFCzly5Aj37t3TLs965Ev50fudX6RlZGRw48YNLC0tsX7kg9Td3R3IO4uvVqtp165dgZyTk5O1ffGKK0gayTgAACAASURBVC4ujqpVq%2BrEcXZ2Jisrizt37uDg4PDMGAkJCYWOcdm8eXOhXaUefV3duHGDxMREne0fPUAvivPnz2u3NzIywtHRkTFjxmi7McXFxVG/fn2dbZycnIiMjCw0Xq1atXSKNHNzczIzMwvcrlGjRvTq1YsePXrQtm1bOnXqRP/%2B/XWex6Kws7MrUEgnJ%2BvnQz83Vw9xEhOVJ5IvOVlxPH19oehjNo%2B0NP3kkp6uh1gWCg8K82VlKT/ALGtu3dJPnKQk5bH01U9ErdZPLD0UjkBeLkpjKX1D5p/I0WiUxzp/Xtn2%2BaKj9RJLH0%2B1vl4y/ySFff8Wlbe3t7ar%2BuOOHz9OamqqzvFlSkoKALZP6S555swZsrKynjlTk7GxMQ4ODiQ%2B5/epXrs5ZT3lDf/oQdTT%2BqzHxMSwcOFCVq5cqXMg%2BqTYj8Yt6pnbmJgY7WCVwYMHY25uznvvvUeOnt8Nly9fpm7dgmfRHB0d2bt3L7/%2B%2BiuHDh1i9uzZ/Pjjj3z55ZcFbjt37lyuXLnC5s2bcXZ2JjY2Fi8vL53bPOl%2BGxkZkfuED74KFSpQsWJFzuvrg%2B0xRX0tPM3TBmBfv369wLJHX1fm5uY0btyYnTt3FmlfhXl0AHZhivKaLMrywm63cOFCxo4dy8GDB/npp59Ys2YN27Zto5aeuooIIYQQ4sUagN2kSRM0Gg3R0dHaXg0RERFYWVkVeryZ7%2BDBg7Rv317nOEmj0fDJJ58wYMAAGv89BicrK4sbN25oe3kUlV6vM1G7dm3u3bvHnTt3tMuuXbtGhQoVsLe3f%2Bb2GRkZTJs2jVGjRhU4Y%2B7k5MS1a9d0ll27du257zDkDWAxMzPjzTffxNzcHI1GQ1RU1HPHeZq7d%2B/y7bff0rdv3wLr0tLSUKvVuLu7M2vWLLZt28aBAwcK7e928eJF%2BvXrR506dVCpVE88612Y2rVrk5aWplNhhoWFcfr0aZycnHj48KHOdRpSU1Of2eeuqJycnEhOTubu3bvaZdeuXcPCwqLITX1K93/9%2BnXSH%2BnjcffuXdL0dcqXJ78mnZycFMVVq9Xcv3%2BfunXrMmbMGLZt20adOnUICwtTFFcIIYQQul6k60zY2Njw6quvsnTpUu7evcutW7dYvnw5r7/%2BurZQGDlyJHv37tXZLioqSqcXC%2BSduIyLi2Pu3LkkJCSQlpbG4sWLMTU11Q4pKCq9FhPNmzenXr16BAYG8vDhQxISEli5ciW9e/cuUh/5hQsXYmNjU%2Bj4gX79%2BnH8%2BHF%2B%2BeUXcnJyOHr0KOHh4fTv379IuVWoUIEbN27w4MEDatWqRUZGBlFRUdy7d4/PP/8cMzMzEhMTi9UX7VG5ubmcP3%2BeMWPGUL9%2BfW2XmEdNnTqVTz/9lNTUVO3tra2tqVKlCubm5jx48ICEhAQyMjJwdHQkIiKCrKwsLly4wE8//QRQpCaoJk2a8NJLL7F06VLS0tL47bfftBc3adiwIa6urixYsIC7d%2B9y//595syZo7cLCLq4uODs7ExgYCDp6encunWLVatW0bdv3yJdo0EpDw8PrKys%2BOyzz0hNTSUxMZGpU6eyZMkSve2jX79%2BHD16lMOHD5OTk8Phw4c5evToE5snn8bc3Jy4uDju3bvH7t27GTp0KH/%2B%2BSeQNyNXQkICzs7OestdCCGEEC%2BegIAAKleujKenJ/369ePll1/Gz89Puz42NlanWzzA7du3qVatWoFYCxYsoE6dOgwcOBB3d3eioqLYsGHDc3UJBz13c1KpVKxYsYJ58%2BbRpUsXLCws6N69O%2B%2B%2B%2B26Rtt%2B6dSumpqa4uLjoLN%2B/f7/2wDcwMJDp06fj6OjI4sWLadu2bZFiDx48mKVLl3LixAl27drF8OHDeeONN7CwsGDixInMnDmTiRMn4ufnV6zrJTzaD61mzZr06dOHsWPHPvGia7Nnz8bDwwOVSkWDBg1Yvnw5RkZGtG/fHkdHR7p3786nn37Kv//9b95//33atm2Li4sLn332GZA3a9amTZuemdeqVat4//33cXd3x9bWlkmTJmnvX2BgIAEBAXh6emJmZoabmxuffPLJc9/3whgZGbFy5UrmzZtH586dsbCwwMvLq8ivBaXMzMxYuXIl8%2BfPx93dHSsrKzw9PfW6/9atWzN37lw%2B%2B%2Bwz7YD4pUuX0qpVq%2BeONXDgQGbPns2xY8c4efIkMTExvPHGGzx48IBq1aoxZMgQunTporfchRBCCPFidXMCqFy5Ml988cUT1xd2AekDBw4Ueltra2sWLVqkOCeVRumpeCHEC0NpLzZ9TvhSdb0eWons7PKmmN28WfEA7Adj/J59o6cwMsq7PkRamvIvpwsXlG1fqVLe9LLnzikfgN2pmsIuoObmULcu/PGHXgZgqxsqn%2B9Wb7M5LVQ4jXaNGnnTy65dq3wA9t%2BzzBWbiUne7GjJyfoZTfvwobLtzczAwQH%2B%2Bkv5AOxCzsg%2BF31O1aZ0Xm1X17w3dsuWehmAnXCr%2BIeAJiZ5E5IlJennJVOE3vAloiSuFXvypP5jlmXl5grYQgghhBBCPOpFa5koi6SYeIp9%2B/Y9dQxBmzZtdC6090%2Bxbt06li5d%2BsT13t7ezJ8/v9jx16xZQ1BQ0BPXDxw4kLlz5xY7/rPMmTOH7du3P3H9O%2B%2B8w7hx40ps/0IIIYQoG6SYUE6Kiafo2bMnPXv2LO00DG706NGMHj26xOKPGzeuVA/W586dW6LFihBCCCFEeSHFhBBCCCGEKJekZUI5vU4NK4QQQgghhCg/pGVCCCGEEEKUS9IyoZwUE0IIIYQQolySYkI56eYkhBBCCCGEKBZpmRBCCCGEEOWStEwoJy0TQgghhBBCiGJRaTSa4l9LXQjxYomPV7a9qSnY2UFiImRnK4tlZ6ds%2B0dzUpoLgJEezq0YG4NarTzOnTvKtjcxAVtbSEqCnBxFodTV7JXlgv4eFgDjlCTlyVhbQ0qK4qTi0m0VbW9qCvb2kJCg/CXs4KBse9Dv85SYqGx7ExOoXh1u31b8EsbSUtn2RkZQqRKkpSk/i/3wobLt9fjWBsC%2Bhqr4G7u6wrlz0LIlnD%2BvPJlSOhxt2lT/MSMj9R%2BzLJNuTkIIIYQQolySbk7KSTcnIYQQQgghRLFIy4QQQgghhCiXpGVCOSkmhBBCCCFEuSTFhHLSzUkIIYQQQghRLNIyIYQQQgghyiVpmVBOWiaEEEIIIYQQxSItE0IIIYQQolySlgnlpJgQQgghhBDlkhQTykk3JyGEEEIIIUSxSMuEEEIIIYQol6RlQjlpmRBCCCGEEEIUi7RMCCGEEEKIcklaJpSTYkKUKdeuXWP58uWcPHmStLQ0bG1t6datG1OmTMHa2rq009MaNWoUZ86cAUCtVpObm4upqal2/f79%2B6lVq1ZppSeEEEKIIpBiQjkpJkSZERUVxfDhw/Hx8eHHH3%2BkatWq/PbbbyxcuBAfHx927NiBubl5aacJwNdff639OygoiKNHjxISElKKGQkhhBBCGJ6MmRBlRkBAAB07duS9996jWrVqGBsb06RJE1auXEmLFi1ITEzk1q1bTJw4kXbt2tGqVSv8/PxISUnhwYMHNGvWjNOnT%2BvE7NevH2vWrAHg5MmTDBkyBFdXVzp16sTy5cu1twsKCmL8%2BPFMmzaNli1bKr4vZ8%2BepVmzZiQnJ2uXZWRk4OrqyrFjx5gxYwYffvghAQEBtGzZkvbt2/Ptt9/q3DYgIIAuXbrQokULRowYwdWrVxXnJYQQQoj/l5ur/5/yRlomRJmQlJTEuXPn2LhxY4F1lpaWLFq0CICBAwdSv359Dh48SEZGBr6%2BvsyZM4dly5bRoUMHwsLCaNu2LQCxsbFcuXKF5cuXc%2BvWLSZNmsScOXPo27cvV69eZcyYMTg5OdG3b18ALly4gK%2BvL4GBgYrvT6tWrbC3t2f//v34%2BPgAcOzYMSpVqoSbmxt79uxh//79fPjhh5w6dYojR44wZcoUWrZsSePGjVm8eDGXL19m69atVKlShS%2B//JIpU6awb98%2BVCpVkXJITEzk9u3bOsuqA3bVqxf/jpmY6P4WJUPp42tsrPv7n0TpfdLjY/NIz8Zi%2BSe/nZTeJ30%2BNkYKT5vmb680DpTBt7ara/G3bdxY97cS588rjyFKzT/wI0y8iGJjYwGoW7fuE28TFRVFZGQkq1evxtLSEktLS8aNG8fkyZPJysqiZ8%2BeBAUFMXPmTABCQ0N5%2BeWXqV27Nl999RUNGjSgf//%2BADRq1IihQ4eya9cubTFhbGyMj49PkQ/Wn0alUuHt7c3u3bu1xcTPP/9Mr169MP77W6BmzZoMHjwYgO7du9OkSRN%2B%2BeUXGjZsyPbt21m6dCn29vYATJs2jU2bNnHx4kVcXFyKlMPWrVsJDg7WWTZl8mTemTpV8f3DxkZ5DH1SelSnT/r4lre1VR4DQA/jjPR1zKK3gx99jZ2qXFlxCHs9pAH6e7r1QV/Pk5JzFo%2BqWlU/cfTBwkJ5jEqVlMcA/b0NOHdOeYxHWtWLTQ/fu8VVHlsS9E2KCVEm5B/A5z7lXR0XF0eVKlWo/si3lJOTE9nZ2SQkJODp6cmsWbOIjo6mcePGhIaG0rt3bwBu3LhBREQEzZs3126r0Wh0ipcaNWropZDI179/f1auXEl8fDx2dnaEh4ezbt067frHC6eaNWuSmJhIUlISaWlpTJo0SSef3Nxc/vrrryIXE0OGDKFbt246y6oDJCYW%2Bz5hYpJXSNy9Czk5xY8D%2BjtKMDWF7GzlcfRx2tHYGNRq5XFSUpTnYW2dF0dhPmpr5Ue6%2BnpYAIwf6OGxqVwZHjxQnFRCprIjOhOTvEIiKUn526laNWXbg36fp7t3lW1vYpL3EZGcrPyxqVhR2fZGRnmFRHq68gPPjAxl2%2BvxrQ2ArZeCbr2NG%2BcVEsOGQXS08mRKiRQTykkxIcoEJycnAH7//Xft2fjHZWVlPXF7lUpF5cqV6dixI2FhYdja2nLx4kWWLl0KgLm5OZ07d2bVqlVPjGGi574GTk5OuLi48NNPP9G0aVNsbGx0ihn1Y98EGo0GlUqlHWT%2B3Xff0axZs2Lv387ODjs7O92F8fH6OfDOydFPHFE4pUdP%2BdRq/cUqK/R1tKtWK46lr7fAP/HtpK%2BXXU6O8lj6OljUR3/4MvfW1kf3ouho6aZUzskAbFEmVK1albZt2/Kf//ynwLr09HQGDhxI9erVuXfvHnfu3NGuu3btGhUqVNAWID169OCXX34hLCyMFi1aaJc7OTnx22%2B/odFotNvevn37qQWKPvTv35/9%2B/ezb98%2BbXeqfPldu/LdvHmTGjVqULlyZaytrbly5YrO%2Bri4uBLNVQghhChvZAC2clJMiDLjo48%2B4sKFC0yfPp1bt26Rm5tLVFQUY8aMwdzcHFdXV%2BrVq0dgYCAPHz4kISGBlStX0rt3b%2B01Hjw9Pbl69So//vgjvXr10sbu3bs3KSkprFixgoyMDGJjYxk1ahQbNmwo0fvUq1cvrl69WmgxER8fz86dO8nOziY0NJTo6Gi6dOkCwNChQ1m5ciUxMTFkZ2ezfv16Xn/9ddLT00s0XyGEEEKI5yHdnESZ0bhxY0JCQggKCmLAgAE8fPiQGjVq0KdPH8aOHYupqSkrVqxg3rx5dOnSBQsLC7p37867776rjVG5cmXc3Nw4cuSIzuDjqlWrsmLFCj777DNWrVqFjY0N3t7ejBo1qkTvk5WVFV26dCEhIUHblSufh4cH58%2BfZ968eZiamuLv70/Dhg0BmDRpEvfv32fYsGFkZ2fTpEkT1q5di4U%2BRgAKIYQQAiifLQn6ptI82u9DCKF3b7zxBt7e3gwaNEi7bMaMGWRmZrJkyRLDJhMfr2x7U1Ows8sbxK20k/fj4zmU5PRPG4D9SFe%2BYtHjyF51NeVzFul1AHZKkvJk9DSCNS5d2eB0U1Owt4eEBOUvYQcHZduDfp8nJfM8QN5LuHp1uH1b%2BdgAS0tl2xsZ5c3ClJam/MDz4UNl2%2Btz0D6AfQ0Fk464uubNBtWypX7GTJTS4ageJnYr4MED/ccsy6RlQogSotFo2LJlC/Hx8QW6OAkhhBBC/BNIMSFEIVq3bk1mZuYT1%2B/fv59atWo9NYaLiwu1a9dm2bJl2hmahBBCCFF2SDcn5aSYEKIQZ8%2BeVRzj4sWLT1z3ySefKI4vhBBCCFHapJgQQgghhBDlkrRMKCfFhBBCCCGEKJekmFBOrjMhhBBCCCGEKBYpJoQQQgghRLn0Il4BOyIiAi8vLwYPHvzM237zzTe8%2BuqrtGzZEh8fHy5duqRdl5mZyezZs/Hw8KBdu3ZMnTqV5OTk585HigkhhBBCCCFeAD/%2B%2BCPvvPMOzs7Oz7ztoUOHCAoK4rPPPuPEiRN07dqVCRMm8PDvC54sWbKEyMhItm7dyoEDB9BoNHz44YfPnZMUE0IIIYQQolx60VomMjMz2bp1Ky4uLs%2B87datWxk4cCAuLi6Ym5szZswYAH755RdycnL4/vvvmTRpEg4ODlhbWzNt2jTCw8NJSEh4rpxkALYQQgghhCiXSuLgPzExkdu3b%2Bssq169OnZ2dopjDxo0qMi3jYyMpFevXtr/jYyMaNKkCRERETRp0oQHDx7QtGlT7fp69ephbm5OZGQk9vb2Rd6PFBNClCfPuNDesyQmJrI1KIghQ4ZgpzCWPiQmJrJ169a8fPTwIV1mcnmOD/En5pL/PCmMZaxo6xJ4jmxtleeT/9gozMdR0dZ5uQQF5T02jo7/oNcv4OCgPJ/8x8bBofQfm6%2B/1s9jU7my8lzyHxd7ez08LhqNoly2BgUxZP/%2BUv/8VULBQ/BEQUFbCQ4O1lk2ZcoU3nnnHf3v7ClSUlKoUqWKzrIqVaqQnJxMSkoKAFZWVjrrraysnnvchHRzEkIU2e3btwkODi5wxqW0lKV8JJeynwuUrXwklycrS/lILmU/l7JmyJAhbN%2B%2BXednyJAhRdp2165dNGrUqNCf7du3P3cummdUS89aXxTSMiGEEEIIIYSe2NnZFbu1xtvbG29vb73kUbVqVW0LRL6UlBQaNGiAjY2N9v9KlSpp19%2B7dw/b52wBlpYJIYQQQggh/mGaNWtGZGSk9n%2B1Ws3ly5dxcXGhdu3aVKlSRWf9b7/9RlZWFs2aNXuu/UgxIYQQQgghxD9Ajx49OHv2LAA%2BPj7s3LmTCxcukJ6ezsqVKzEzM6NLly4YGxszePBgVq1axV9//UVycjJffPEFXl5eVKtW7bn2aezv7%2B9fAvdFCPEPValSJdq2bavTLFqaylI%2BkkvZzwXKVj6Sy5OVpXwkl7KfS3nx6quv8tlnn3H69Glu3brF6tWrWblyJd7e3lhZWTFv3jx69uyJs7Mzzs7OWFpasmDBAr788kuysrIIDAzUztTUunVrfv/9dwICAli/fj0NGzZk3rx5VKhQ4blyUmn0MfJCCCGEEEIIUe5INychhBBCCCFEsUgxIYQQQgghhCgWKSaEEEIIIYQQxSLFhBBCCCGEEKJYpJgQQgghhBBCFIsUE0IIIYQQQohikWJCCCGEEEIIUSxSTAghhBBCCCGKRYoJIYQQQgghRLFIMSGEEEIIIYQoFikmhBCiGE6fPl3o8szMTPbt22fgbIQQwnAuXrzIzz//rP0/MzOzFLMRpU2KCSGEKIaxY8cWuvzevXvMmDHDwNmAWq1mzZo19OrVizZt2gCQlpZGQEBAqXzRx8XFGXyfhenWrRtLly4lJiamtFMp05KSkrh582aBH1F2/P7779q///rrLzZu3Mjhw4cNmkNMTAw9e/ZkxIgRTJ8%2BHYD4%2BHi6du3K5cuXDZqLKDtUGo1GU9pJCCHKlhEjRqBSqYp022%2B%2B%2BaaEs9GlVqsJDQ0lJiam0IPk/C%2B4krJ%2B/Xq%2B%2BuorkpKSsLW1LbA%2BNTWVWrVq8dNPP5VoHo9bsGABp0%2BfZtSoUXz88cdcvHiRlJQUfH19qVu3Lv7%2B/gbNp0mTJrRo0YK%2BffvSs2dPqlatatD95/vhhx8IDQ3lxIkT1KtXj759%2B9KnTx/s7OxKJZ979%2B6xZs0a3nvvPQC%2B%2B%2B47QkJCcHJyYtasWVSrVs2g%2BezYsYNFixbx4MEDneUajQaVSkVUVJRB8wGIiIh44vt7yJAhBsvj7t27xMXFFZpHfsFuKBs2bGDFihX8%2Buuv3Lt3j969e%2BPs7ExCQgI%2BPj6MHj3aIHmMHDmS5s2bM3XqVFq3bs3FixeBvM/FgwcPsnHjRoPkIcoWKSaEEAUEBgZq/05PT2fXrl20atWKunXrkpuby9WrV7l48SLDhg3Dz8/PoLlNnz6dsLAwGjVqhLm5uc46lUpV4sVNbm4ukZGR%2BPj4MG/evALrK1SogJubm8EPnt3d3dm2bRu1atXCxcWF//3vfwDcvn2bAQMGcOzYMYPmExsbS2hoKD///DOXL1/G3d2dfv364enpSYUKFQyaC%2BS10oSHhxMWFsbRo0dp1qwZ/fr1o2fPnlhYWBgsD19fXzIyMli9ejWRkZEMHz6cWbNmcfnyZZKSkli2bJnBcgHo0KEDb7zxBl27dsXMzKzA%2Bn/9618GzWfBggVs3LgRGxubQt/fBw8eNEgeq1atIigoCLVaXWBdaRRZnp6eLF26lObNm7N%2B/Xr27t1LSEgIsbGxjBo1itDQUIPk4erqyq%2B//oqZmZnO50xOTg7t27fn7NmzBslDlDEaIYR4Cj8/P83hw4cLLA8NDdX4%2BvoaPB9XV1fN77//bvD9Pu5///ufJjc3V5Oenq5dlpKSosnJySmVfFq3bq3d98svv6xd/uDBA02LFi1KJad8CQkJmm%2B//VYzYsQITZs2bTQffPCB5syZM6WSS1JSkmbdunWaVq1aaZo2bapp3bq1ZsmSJZrMzEyD7L9t27aae/fuaTQajWbRokWa6dOnazQajebhw4ea9u3bGySHR7m5uWmys7MNvt8nadOmjebEiROlnYamTZs2mh07dmju37%2BvyczMLPBjaC4uLtq/R40apVm3bp32f0O%2Bv7t06aK5ffu2RqPR/ZyJiYnRtGvXzmB5iLJFxkwIIZ4qPDwcd3f3Ass7d%2B5s8P66ANbW1jg6Ohp8v48zMzPD09OTQ4cOaZf98MMPdO/enejoaIPn07RpU77%2B%2BmudZenp6SxevJhmzZoZPJ9HWVtbY2tri62tLdnZ2Vy5coX3338fHx8f4uPjS3z/Dx8%2BZOfOnYwePRoPDw/27dvHtGnTOHr0KCEhIVy4cIFZs2aVeB6Q103PysoKgOPHj%2BPp6QnkvZ4yMjIMksOj3nrrLdauXUtOTo7B910YMzMzWrduXdppULFiRXr16kXlypUxMzMr8GNo9vb2xMTEcPPmTc6cOaN93Vy/fh1LS0uD5dG1a1emTp3KsWPH0Gg0REVFsWPHDiZMmEDv3r0NlocoW6SbkxDiqXr06MGIESMYPny4zvKtW7eybt06nRk9DGHnzp1ERkYyffp0g3ZPeZyPjw8dO3Zk9OjR2u4YWVlZrF%2B/nvDwcL799luD5hMdHc2YMWOAvMG0DRo0IDY2FhsbG1asWEGjRo0Mmg/AyZMn2b17N6GhoVhYWNC3b1%2B8vb1p2LAharWawMBALly4UKKPlZ%2BfH%2BHh4VhbW2v3X69ePZ3b3L9/n86dO3P%2B/PkSyyPfyJEjad%2B%2BPebm5qxYsYLDhw9TsWJF9u7dy9q1a9mxY0eJ5/Co8%2BfPM336dJKTk7GxsSkwVspQ3YryffXVV6jVasaPH2/Q/T5ux44dXLp0iXHjxmFvb1%2BquQBs27aNhQsXAtCrVy8WLFjAgwcPGDp0KJ6eniU%2BVixfZmYmn3/%2BOTt27CAtLQ3IO1kwZMgQJk%2BeXCqFlih9UkwIIZ7q0KFDTJ8%2BnUqVKuHg4IBarSYhIYEHDx7wxRdf4OXlZdB8vL29iY%2BP5%2BHDh1StWrXAwY%2Bhxga0bNmSM2fOYGxsrLM8JyeHNm3aGOTA9HEZGRn88ssvxMbGYm5ujpOTEx07dsTExMTguXTq1InU1FQ8PT3p378/7u7uGBnpNobn5OTQsmVL7SDOkjBjxgy8vb1p3779UycV2LlzJ/379y%2BxPPJdu3aNuXPn8uDBA6ZMmUK3bt1ISUnBy8uLL7/8Ejc3txLP4VFeXl7UrVuXDh06FDqWZejQoQbNZ9KkSZw7dw4TExNq1qxZ4DXz3XffGSSPsLAw/P39SUpKKnR9aQxMj4%2BPJzU1VXtiIDc3l927d%2BPt7W3wXDQaDUlJSZibmxu0ZUSUTYb/hhFCvFC6devG0aNHOXr0KAkJCWRlZWFnZ4e7u3upnLF76623DL7PwtjZ2XHu3LkCs7ocP34cGxubUsnJ3NycVq1aUbt2bVQqFTVq1CiVQgLyWgReffVVKlWq9MTbmJiYsHfv3hLNY/DgwQCFFndGRkbY29vj4OBgkEIC8gbEb9iwQWeZtbU1R44cKZWWtqSkJH766acyc0b5pZde4qWXXirtNJg7dy7dunXDw8OjVCYMeFzPnj0LXL/GyMiIrl274ubmxsmTJw2Wy5UrVwgPDycxMVH7OdO1a9cCLX6i/JCWCSFEob744osi3c5Qzetlza5du/D398fd3R1HR0dyc3P5448/OH36NIsXL%2BaVV14xaD5//PEH06dPJzo6mvyPdZVKRbNmzQgMDMTJycmg%2BaxateqJ64yMjLCzs8PNza3EC9KXXnpJ%2B3g8%2Brjk/69SqWjQoAHLli2jbt26JZoLQIsWLbC2tqZPnz707du3VLqfPWr27Nl4enrSuXPnUs2jrGnbti0nTpwotWI836lTpzh16hRfffVVode2iY2N5dChQ5w7d84g%2BWzZsoWAgACaNGlCzZo10Wg03Lx5kytXrjB37lwGDRpkkDxE2SLFhBCiUCNGjHjmbQwxFevjsrKy%2BPLLL9m7dy9//fUXKpUKR0dHBgwYwPjx4wt0iyhJly5dYteuXcTGxqJSqahduzYDBw6kcePGBssh38CB3oGCxAAAIABJREFUA6lXrx5jxoyhVq1aaDQa4uPjWbt2LdeuXTN4X/w333yT6Oho0tPTcXZ2RqVScf36dW13uaSkJO7du8eyZctK9ED21KlTrF69mrfeeovmzZtjZGTExYsX2bhxI2PHjsXOzo41a9Zw69atAgPYS0J6ejqHDx8mNDSUw4cP4%2BDgQN%2B%2Bfenbty8ODg4lvv/HBQQEsG/fPpydnXFwcCjw/nl0mmhD2bJlC3v37iU%2BPh6VSoWTkxMDBgygX79%2BBsshODgYOzs7bctWabl06RLbtm0jJCQEV1fXAuvNzc0ZOHAgffr0MUg%2B7u7uBAQE0L17d53loaGhzJkzhxMnThgkD1G2SDEhhHih%2BPv7c%2BbMGYYPH46zszOQd1XWjRs30r9/fyZPnlzKGZYOFxcXTp06VaCrTFpaGu7u7tr54A1l06ZNREVF8cEHH2hnL7p//z6ff/45rVu3xtvbm23btrF582Z27txZYnn06tWLDRs2UL16dZ3lCQkJTJw4ke3bt5Oenk6XLl349ddfSyyPwmRnZ3Py5En27dunvXbKoEGD6Nmzp8G6HX344YdPXb9o0SKD5JFv6dKlfP/993h7e%2Bu8v3fu3Imfn5/BxnBMnDiRc%2BfOUaFCBWrUqFFqYzfy%2Bfv7G/zCk4Vp1aoVv/76a4EWG7nORPkmYyaEEC%2BUAwcOEBISQu3atbXLOnToQMeOHRk/frzBigm1Ws26devYuXMnt2/f5syZM6SlpREYGMgHH3xg8H7WDRs25NatWwW66iQlJdGwYUOD5gJ53ZzCwsJ0LjxmZWXFzJkz6d27N97e3rz22mslfrD6pKlnTUxMiImJAfIKLkO2aOW7e/cuMTEx2qs9W1tb89133xEcHExQUJBBWrie9vj/8ssvJb7/x23fvp21a9fSpEkTneW9e/fmgw8%2BMFgx0bRpU5o2bWqQfRWFv78/UVFRxMTEFDqF8Ouvv26QPPr378/OnTsL7G/v3r307dvXIDmIskeKCSHECyUnJ6fQfvaOjo6kpKQYLI9PPvmE06dPM378eD7%2B%2BGMg70xzTEwMixYtMvhZxGHDhuHr68uAAQOoU6cOarWa2NhYdu3axeuvv64zy1XHjh1LPJ/MzEyuXLmCi4uLzvKYmBiSk5MBuHz5MtbW1iWaR%2BfOnXn77bfx8fHB0dERExMTbt68yZYtW3B1dSUrK4tRo0Zp5%2B0vaampqezfv5/du3dz9uxZmjVrhre3N2vWrNE%2BFps2beL999/nxx9/NEhOycnJ/P7772RlZWmXJSQkMH/%2BfIPPSpaamkqDBg0KLG/atCmJiYkGy2PKlCkG21dRfPrpp/znP/%2BhSpUqhV4Z3FDFRHp6OosWLWLDhg3UrVsXtVrNjRs3iI%2BPp3Pnzvz73//W3rY0usiJ0iHdnIQQL5SRI0fSokULpkyZgqmpKZBXYCxfvpzTp0%2BzefNmg%2BTh7u7Otm3bqFWrFi4uLtpuRLdv32bAgAEGm6I2X1HPYqtUKoNMaxkUFMTXX3%2BNh4cHtWrVwtTUlPj4eMLDw/H29uaDDz6gXbt2TJ06lbfffrvE8khPT%2Bezzz7jyJEj3L59m9zcXGxtbWnfvj0zZsygatWqBAUFMW7cOIO0Jr388stUq1aNvn370r9//0IHfWs0GlxcXEp0ytx8oaGhvPvuu2RmZqJSqbSD1K2srOjfvz8zZ84s8RweNWTIEAYMGFCgBWLbtm1s2bKF7du3GySP3NxcNm/erJ21CKBGjRp069aNoUOHPnWa4ZLQtm1blixZQocOHQy638c9q1vcowzdRU6UHikmhBAvlKtXrzJq1CgyMjK0V8KOi4vDxMSE1atX07x5c4Pk0aZNG06dOoWxsbFOMZGamkqnTp1K5ToTZc2%2Bffu0B/EajQYbGxvat2/PgAEDMDIy4tdff6Vdu3alnaZBPe0%2Bh4SEaAf85ubmGqTrVe/evRk7diy9evWidevWXLhwgUuXLvHVV1/h6%2Btr8Ok%2Bz549q51EIH/f165dIzY2lqCgIDw8PAySx%2BzZszl06BB9%2BvTRmbXop59%2BwsvLizlz5hgkj3wdOnQgPDxcewKltJw4cQI3NzeDF1OibJNiQgjxwsnKyuLIkSPExcWRlZWFk5MTHh4eVKxY0WA5vPXWW3To0IGxY8dqi4n09HQ%2B/fRT7YBwQ7t79y6HDx/WjhNwdnama9eu5f6iUidOnNCZHah27doMGDCAFi1alEo%2BMTExREZGFuhWtHbtWi5cuGDQXFxdXbWF76NF8dWrV5k5cyYhISEGzQfgzp077Nmzh/j4eO37u2fPntSsWdNgObi6uvLDDz/wr3/9S2f51atXGTRokMFPFqxZswaVSlXo9LCG5OrqiqWlJb1796Zfv35l4pogovRJMSGEEMUQHR3NmDFjgLxBzg0aNCA2NhYbGxtWrFhh8OsHnDx5ksmTJ2NhYaEdnH7jxg3UajUbN240%2BCDs7OxsgoOD2bdvHzdv3gTQHsSPHTvWYGc2N2/ezGeffYaHh4f2WhvXrl3j6NGjLFu2zGBjJfKFhITg7%2B%2BPtbU1ycnJ2NrakpSURM2aNfHx8dG%2BpgzFy8uLVatWUa9ePbp06cKqVato3LgxGRkZuLm5lVoLW3Z2tvaiaPb29gWuNF/SPDw8CAsLKzCrVmZmJl5eXhw5csSg%2BUydOpXTp09ToUKFQq8MbqjunZmZmRw9epTQ0FDCw8OxtbWlb9%2B%2B9OnTR2dSDFG%2BSDEhhCjzPD09OXjwIPDswcOGHKuQkZHBL7/8QmxsLObm5jg5OdGxY8dSudDVwIED6du3r874A7VazcqVKzlz5kyBqy6XtICAAE6cOMHw4cN1DuI3bdrEoEGDmDBhgkHy6N69O/PmzcPNzU1n%2BeHDh1myZEmJTktbGC8vL%2BbOnYu7uzsvv/wyFy9eJCEhgYULFzJy5Ehatmxp0HzWr1/PkiVLOH78OEFBQRw4cABPT0%2Bio6NRq9UGnwL13r17%2BPv7ExYWRk5ODgAVKlSgT58%2BfPzxxwabJe37778nIiICPz8/7cD4lJQUgoODtVP4GtLSpUufun7atGkGyuT/5eTkcPr0afbt28f%2B/fupV68egwcPpk%2BfPmXmiurCMKSYEEKUebt27cLb2xvgmRdfGzBggCFS0kpMTNSeQa1Rowa2trYG3X%2B%2BFi1acPbs2QKFTFZWFh07duT06dMGzcfd3Z2tW7cWOFsZExPDhAkTCA0NNUgeLVq04L///W%2BBM9tqtZq2bdvy3//%2B1yB55Hu0W1GLFi04f/48KpWKuLg4Jk6cyO7duw2aD%2BSNU2jdujU5OTkEBwcTERFBrVq1mDBhgkG7FgFMnz6dxMRExo4dq3OdiVWrVtGiRQs%2B%2Bugjg%2BTxyiuvkJCQQFZWFlZWVuTm5pKamoqpqan2uin5DD3ZQlnx119/sWfPHvbt20dMTAxdunTh7t27xMfHExQUVKam1hUlS6aGFUKUefmFBMDNmzcLvZZEWloaS5cuNVgx8ccffzB9%2BnSio6O1M%2BCoVCqaNWtGYGCg9my8odjZ2fHnn39Sv359neWxsbGlMmYiOzu70Cl8nZycDDqFr5OTE8eOHStwle0TJ06UyhWnHRwctAfvtra2/Pe//6V169ZUqVKFuLg4g%2BcD0Lp1ayDv2hulcYb7UUePHuXAgQPY2Nhol9WpU4dmzZoxdOhQgxUTEydONMh%2Bnmb58uXaz7ply5Y99ba%2Bvr6GSIl79%2B5ppzY%2Bf/48Li4uDBkyhF69elG5cmUAvv32W2bMmFEqhbEoHVJMCCFeCCkpKdy9e5fVq1fTu3dvHm9U/fPPP9m6davBDjb%2B/e9/U79%2BfT755BNq1aqFRqMhPj6etWvX4uvr%2B8wWFH3z9vZm3LhxDB8%2BXDto9Nq1a2zevJkePXoYNBeAJk2asHr1aiZNmqRtFVCr1axevbpAwVOSJk%2BezOTJk%2BnUqZPO7EBHjhwhICDAYHnkGzduHCNHjuTUqVMMHDiQiRMn0rZtW2JiYgzexenAgQNs27aN6OhoUlJSMDExwd7eHldXV9566y2DXDTvccbGxgWu4g55U9U%2BfPjQYHk8flIiOTkZU1NTgxbmZ86c0f79tKuzG3JmpQ4dOuDg4EC/fv1YtGhRoeMkhg0bxqeffmqwnETpk25OQogXwg8//MAnn3xCampqgUIi3yuvvMKXX35pkHxcXFw4depUgQOftLQ03N3dtbPiGIpGo2HTpk388MMPOrNc9evXj9GjRxt8AOuVK1cYPXo02dnZ2u4qN27cAGD16tUFLmZXkiIjI9m%2Bfbv2calduzbe3t60atXKYDk86saNG9qWq%2B%2B%2B%2B07brWj48OFUqVLFIDl8/fXXbN68mdGjR2NlZcWWLVvo378/VapU4dixY%2BzevZvFixcbfID6pEmTqFKlCu%2B99562deLu3bssXryYxMREvvrqqxLd//%2B1d99RUdzd/8Dfiwqo2BMbRY2xRSOgCIpYgsYoCMZeiCZqLESxYIuCXeyYPBbU2EjQxBYVC6BC1NhQsaJBMUoSUAQrNmBpvz/87v5YFxHNzmdm4/t1Ts5ZZuY498HNk7lzP/d%2B7t27h5kzZyI%2BPh7u7u7w8fHB%2BPHjERYWBuBFFScwMBCVK1eWNA6lSU5O1qmqEeXHZIKIjEZOTg6cnJwQGhqqd87c3Fxov0LPnj2xcOFCvU3H/vnnH4wbNw7btm0TFotSZWZm4tChQzojPtu0afNOj6odO3YsnJ2d0bx5c1mn37Rp0wbr1q3TVon%2B%2BecfTJw4UdtwfezYMQQEBCA8PFxoXCkpKfD29kZcXJy2N%2BHx48eoXbs2goKCJF8%2BOHbsWNy7dw8dOnTAzp070bhxYyQkJGDs2LFQqVRYvnw5ypYtK8vuznKOOM4/NpjoZUwmiOg/ITc3F/369RM2fWbnzp3YsGEDunbtipo1ayInJweJiYkIDQ1Fjx49ULNmTe21r5tA9ba2bNlS5Gt79%2B4tSQxKNHHixCJfu3DhQgkj0bdkyRLExMQgNjYWVapUQYsWLdCiRQs0b95cp09Aag4ODjh%2B/Lh2OlJGRgZcXFwQExMD4EXi7uDgINto2Li4OG0Sam1tLWwzShcXF%2BzZswcVKlTAzZs30blzZ0RERGiTmAcPHsDDwwPHjx8XEo%2BG3COONZPHiArCngkiMipPnz7FihUrcPnyZWRlZWmP37t3D5mZmcLimDx5MgAUuDZ4zpw52s8qlQpxcXGSxLB69epCz6tUKuTl5UGlUglJJtq0aVPk9duHDx%2BWLA4lvyPz9fUF8GLK1qVLlxATE4OdO3ciICAAlStXFtZr07BhQwQFBWHs2LHIy8vDypUrtb02z549w6pVq/SqbiI1aNAADRo0EH7fjIwMbeXsgw8%2BgImJiU41pFSpUkJ7NzQ2bNiAVatWvXLEsdTJBHe8psIwmSAiozJ9%2BnQkJCSgVatWWLduHYYMGYI//vgDWVlZQpceXL16Vdi9XuW3334DACQlJcHU1FS7jjslJQU//fQTMjIy4OrqipYtWwqJZ%2BTIkULu8zqLFi2SO4TXMjExQYkSJVCiRAntTP709HRh958wYQKGDBmC4OBgqFQqmJubIygoCABw8OBBhIeHv3aCkCG5urq%2B9oFVpVIhMjJS0jgaNWqEdevWYejQoTAxMcGBAwd0zi9btgyNGjWSNIaC3Lt3D46OjnrHXVxctAmqlNRqNfr06fPa60TvS0LKwGVORGRUmjdvjoiICJQvX16n9P7jjz/i8ePH8PHxERZLVFSU9o1gbGwsQkNDUbNmTfTr109vh1qpxMTE4Ouvv8acOXPQuXNnqNVqeHh4ICsrC3Xr1sWpU6ewZMkSfPLJJ0LieVlOTg7u3bsH4MX4WjnecG7btk1vrXm3bt3g5uYmPJbvv/8eZ8%2BexfXr11GnTh3Y2dnB3t4e9vb2qFChgtBY0tLScOHCBQAv9r/Q9Cio1WqUKFFC5%2B9K04ArlcL2QUlMTMT333%2BPnJwcnDhxQrIYgBcvCQYPHoyxY8eiR48eOuc6duyIp0%2BfYsOGDahTp46kcbzM09MT48aN0xtxfPToUSxYsAB79%2B6V9P4NGzYs0rhcpbxQILGYTBCRUXFycsKJEydQrFgxNG3aFEePHkWpUqWQnp4OV1dXnDx5UkgcixcvxoEDB3DgwAHcuXMHbm5u%2BOyzz/DXX3/Bzs4OkyZNEhLHl19%2BiRYtWmh3lN63bx/8/f0RGRmJSpUqYe/evdi8eTM2btwoJB6NJ0%2BeYObMmTh48CDUajWAF03ynp6e8PPzE7ZD7tKlS7F582Z4enrqrDXfvXs3xo8fj169egmJQ6NRo0baKVuOjo74%2BOOPUaJECaExvA05GnDVajVWrVqFDRs2oFu3bhg9erTehnFS3TcjI0PvXtHR0WjUqJHOAAGpkyyN/fv3Y9y4ca8ccdytWzdJ788GbCoMkwkiMipff/01qlatimnTpuGrr76Cg4MDBg4ciAsXLuDbb78tdB67IbVq1QqbNm2CjY0NgoKCEBMTg/Xr1%2BP%2B/fvo3r27pD0B%2BTVp0gTHjh1DqVKlALxoPs7IyNCOyM3IyEDLli2F7/Q8btw43L59W28n49WrV8PBwUHbcyK1Nm3aICgoSG833gsXLmDKlCnakZ%2BipKen4/z58zhz5gxiYmJw7do11K1bF02bNkXTpk3RunVrofEUlegG3MjISMydOxfVqlXD1KlTZdnzoihEPmTLOeKYDdhUGPZMEJFRmTZtGqZOnQrgRTPrsGHDsGbNGpiYmAhZO6zx9OlT7Zvu48ePw93dHQBQqVIloTs85%2BXl6ex1ERMTg0GDBml/NjMzQ25urrB4NI4ePYrw8HCdcb21a9dG48aN0bdvX2HJxJMnT1CvXj29440aNUJKSoqQGPIrWbIknJ2d4ezsDODFW/AdO3bgxx9/xA8//CBZs/6/JWp52t9//43Zs2fj2rVrGD9%2BPLp06SLkvm9L5PvYhg0b6iXFojRr1uyNrhdVsSFlYDJBREbFxsYGP/74I4AX4y0PHTqEmzdvolq1aqhSpYqwOKytrREdHY1SpUrh0qVL%2BO677wC86J0Qud9FlSpVcOPGDXz44Ye4evUqkpOTdSa%2B/PXXX8LX4gMvGow11ZL8ypcvL3QazocffoidO3eiZ8%2BeOsdDQ0O1FROR0tPTcfHiRZw/fx7nz5/HxYsXYWFhAScnJ%2B1StXdRRkYGVqxYgU2bNqF37974/vvvjWI/EqmTrLy8PAQHB%2BPAgQMoVqwY3Nzc0K9fP0nvWZB169a90fUdO3bksqh3CJMJIjIq7u7ucHNzQ6dOnfDBBx%2BgbNmyQjZtepmvry%2B8vb2hVqvh7e2NypUrIy0tDcOGDcMXX3whLA43NzdMnDgR7u7u2LlzJ%2Bzs7LRrqp89e4bFixdLts9FYezs7DBnzhxMmDAB5cuXBwA8evQIgYGBQqfhTJgwAV9//TVCQkJ01ponJCRg2bJlwuLQcHBwwHvvvQdHR0d06NAB06ZNg5WVlfA4lOazzz5DVlYWJkyYgA8//PCVFZo3fUNu7NasWYOQkBD07t0bubm5CAoKQk5ODvr37y93aIXiCvp3C3smiMioBAcHIzIyEufOnUPdunW1iYUcuwlnZ2cjMzMTpUuXBgCcPHkSa9euRUxMjLC3ctnZ2Zg3bx5OnDiBWrVqYerUqdrlBTNnzsSJEycQEhKiHRsrSnJyMoYPH47r169rKyMPHz5EjRo1sHLlSp1N/aR29%2B5d7NmzR7vW3MbGRrbvTEJCgqz7N7wtqXsDXF1dX3uNSqVCVFSUZDG8Dal/L506dcL8%2BfNha2sL4EWvz9SpU7Fnzx7J7mkIbNh%2BtzCZICKj9ODBA0RFReHgwYOIjo5GvXr14O7ujq%2B%2B%2BkpoHLdv38aOHTuwc%2BdO3L17F66urujWrZsiGmlTUlJQsWJFWacFXb58Wech3tbWVuh42A0bNmDgwIHC7leQDh06FPl/8/79%2ByWO5u0o7eFQKWvypf692NnZ4fz589rvT25uLpo2bSrbzuRFpbTvC0mLy5yIyChVrFgRPXv2RM%2BePXH%2B/HkEBgZiwYIFQpIJtVqNyMhIbNu2DadPn4atrS1SU1Oxbds2RU2dEdlD8rIRI0ZgxYoVaNSokSybfGmsW7cOXbt21S61ksOXX36p/fzw4UNs3boVrq6uqFmzJvLy8nD9%2BnX8/vvvGDx4sGwxvo6oUb5F9a6sydfsYK9hYmIiy0AFosIwmSAio5OXl4ezZ88iMjISUVFRuH//Ptq2bStkx97Zs2dj7969KF%2B%2BPDw8PDBr1ixYW1vD3t5eu9yJgH/%2B%2BQdxcXFo0KCBrHEMGTIEY8aMgbu7O6pVq4ZixYrpnM/frC4VLy8v7eevv/4aS5cu1evzOX36NFavXi1LFeXOnTvYu3cv7ty5A39/fwDApUuX0LhxY%2B01Z86cER5XYZSyqEJpSRaRHLjMiYiMyuTJk3HkyBGkp6ejbdu26NSpE9q0aQMzMzMh969fvz7c3d0xevRo7WhY4MUOwrt375ZlHb4SLVmyBPv27YOdnR0sLS31HuJHjx4tJI7CKkUqlUr4KFZ7e3ucPn1ab%2BmZWq2Gk5OT8OUrUVFRGDt2LJo0aYKzZ88iNjYWycnJ6Ny5M2bNmqUdeaw0IpbRFCXJklqDBg3QpEkTnWPnzp3TO7Zp0yZhMRUFlzm9W1iZICKjkpGRgenTp6Nt27bCEoj81q5di%2B3bt8PDwwMNGjRAly5d0KlTJ%2BFxKF1MTAyqVKmC5ORkJCcn65wT2TNx5coVYfcqCmtra6xYsQJDhw7Vjs5NT0/HunXrYGlpKTye77//HkuWLEH79u21D8nVqlXDihUrMGfOHMUmE1J7Ocny9/dHcnIyBg4cKDTJGjZsmN4xY5hoxYrNu4WVCSIyOhkZGTh06BDu3LmjXRZy584dVK1aVVgMDx8%2BRGhoKH799VckJCQgNzcXM2fORNeuXVG8ON/TyOlNNqMT3Vdy/vx5jBo1Cg8fPkT58uWRk5ODJ0%2BeoFSpUli%2BfDkcHR2FxmNnZ4dz587BxMRE521yTk4OmjZtigsXLgiNp6ikfvPt4eGB0aNHa5Msze7P0dHRmDNnDvbu3SvZvf%2BNNWvWYMiQIZL9%2BQ4ODjhz5ozQFwKkfEwmiMionDt3Dt7e3ihbtiySk5Nx%2BfJl3Lp1C507d0ZQUJCQNfAvu3DhArZt24bw8HCYm5vD09MT3377rfA45PYm4yo9PDwki6N%2B/fpFftiRY8fpnJwcXLhwASkpKVCr1ahSpQrs7Ox0djIXxc3NDYGBgWjQoIHOA/qRI0cwa9YsxY1i1RAxRYlJlr6xY8eiefPm6N27t2T3IOPD12dEZFTmzZuHUaNGwcvLS7ssw9LSEgEBAVi8eDF%2B/fVX4THZ2dnBzs4Ofn5%2B2LdvnywxKEFAQIDOz8%2BfP4darYaFhQVyc3Px/PlzmJmZ4f3335c0mcif1MTGxuLXX3%2BFl5cXatWqhdzcXFy/fh2bN2%2BWbWRssWLF0LRpU51jGRkZaNu2LQ4fPiw0ln79%2BmHw4MHo0aMHcnJyEBwcjGvXriEsLAwTJ04UGouSVK9eHdeuXdMbIHDs2DGhO9y/KanfD6enp%2BP777/HsmXLULVqVb0q7ObNmyW9PykTkwkiMirXr1/XvhXL//a5Y8eO8PPzkyssAECpUqW042rfRdHR0drPO3bswIULFzBq1Ci89957AF7sDbB06VLJ13zXqVNH%2B9nX1xdr167VWc7UsGFDODg4wNvbG5999pmksbwsNTUV8%2BfPx%2BXLl6FWq7XHHz9%2BrP09ifTFF1%2BgcuXK%2BPXXX2FtbY3Q0FBYW1tj5cqVcHZ2Fh5PUUm9Jt9Ykyyplx/JPeqZlInLnIjIqHz66adYv349rK2tdUr6Fy9exMiRI3H06FGZIyQAaNu2LcLDw/WW7jx9%2BhTu7u44cuSIkDiaNGmC33//HRYWFnpxtGrVSvj0pOHDhyMrKwvt2rXD3LlzMXXqVFy5cgXx8fFYvny5LAmF0ihhihIAHDhwAL/%2B%2Biv%2B%2BecfmJubw9raGn369FF0ksUpSiQHViaIyKh4eHhgyJAhGDhwIHJzcxEZGYmrV69i06ZN6Nevn9zh0f9JT09HamoqatSooXP84cOHyMjIEBaHvb09fHx8MHjwYFhaWiInJwfJycn48ccfYWtrKywOjXPnzuHw4cMoVaoUFixYoK2yhYaGIigoCNOmTZM8hsmTJxf52nnz5kkYiT6lTFECXuxc3qFDB2H3Mxb79u1DaGgoUlNTsWvXLqjVaoSEhGDQoEFszH5HMZkgIqPi4%2BODMmXKICQkBCqVClOmTIG1tTV8fX3Ro0cPucOj/9OpUycMGDAAnp6esLKyQnZ2Nu7cuYM9e/YIfUBbsGABZs%2BeDW9vb2RlZQF40bPg5OSEBQsWCItDo3jx4tp15mZmZkhLS0O5cuXQqVMnBAQECEkmMjMztZ9zc3Nx5MgRWFlZaXtKbty4gZSUFFlGHss5qlbJSZZSBAUFYcuWLejduzdWrVoF4MUSvV27duHJkycYM2aMzBGSHLjMiYiIDC47OxtbtmxBVFQU7ty5A7VajcqVK6N169YYNGiQsDn06enpKFmyJPLy8vDgwQOo1WpUqlRJtjn4o0aNglqtxvfff49vvvkGVatWxYABA3D%2B/HmsWLECx44dExrPrFmz0LBhQ3Tv3l3n%2BKZNm/Dnn39i%2BvTpQuORc4qSr6%2Bv9vPrkqyXhw0oRf4xtlJo06YN1q5dizp16uj8/SQmJmLAgAE4dOiQZPcm5WJlgogUb8uWLUW%2BliMLleHs2bPw8vKCl5eXrHE4Ozujffv28PDwgIuLC0xMTGSNZ8aMGViwYAGKFSuGSZMmYejQodixYwdKliwp/MEdeDH5asqUKXrHe/bsCWdnZ%2BExyTlFacmSJdrPs2bNgr%2B//yuTLKXq1q2bpH/%2BkydPdAYcaFSuXBkPHjyQ9N4x/oZMAAAgAElEQVSkXKxMEJHiubq6Fuk6lUql2Ln47xo7OzuUL18enTt3hoeHB%2BrVqydLHNHR0Th48CAiIyORlZWFTp06wdPTU5Z%2BiYLk5OTg7t27qFixoizVEldXV/j7%2B%2Bv9O3bkyBHMmDFD%2BJvmjRs3IigoCD169MD69esxfvx4nSlKopLTZs2a4eTJk3qjT9VqNZydnRETEyMkDo3c3Fxs2LABu3btwt27dxEdHY3nz5/ju%2B%2B%2Bw4QJE4R9d/r06YN%2B/fpp/x3SVCZWrlyJgwcPYseOHULiIGVhMkFERmHfvn2vXS/t5%2Ben2OUH75r09HQcOXIEBw8exJEjR1CtWjV4eHjAw8MD1apVkyWmCxcuIDIyEhERETAxMYGHhwe6desGS0tLoXHcuHEDBw4cwK1bt6BSqWBtbQ13d3fhcQDA1q1bMX36dHz00UewsrLSNqjHxcXBz89PlsqSEqYoKS3Jmj9/Po4fP46BAwdixowZuHTpEh4%2BfIhRo0ahTp06QnptAODkyZMYMWIE6tati0uXLqFNmzaIj49HWloagoKChO/gTsrAZIKIjMLLIw9btGiBkydPFnoNKUNWVhZOnjyJ8PBwREZGol69eujZsyc6deok/G38lStXEBYWhi1btsDU1BRqtRqffPIJ/P39Ua5cOcnvHxERgfHjx6Nu3bqwsbEBANy8eRN///03NmzYgCZNmkgew8tu3ryJqKgo7Y7cmt4W0aNYlURpSVbLli2xefNmvZHYqamp6N69u9CR2CkpKdizZw8SExNhbm4OGxsbuLu7o3z58sJiIGVhMkFERuHlxsKCGg2lbj6kt5OSkoKwsDCEh4fj6tWraN26Ne7fv4979%2B5h2bJlqF%2B/vqT3T0xMxO7du7Fnzx4kJyfjk08%2BQZcuXdC6dWs8e/YMs2bNwpMnT7B69WpJ4wAAd3d3DB8%2BXG8H8G3btuHXX39V1A7Cvr6%2BOn0EUlHqFCUlJVnNmjXDqVOn9BrTnz17hpYtW0ramJ7fmjVr4O7ujurVqwu5HxkHNmATkVF4eX55QfPMOeNcOZ4%2BfYqIiAjs2bMHMTExaNSoEbp06YIffvhB%2BwZz48aNmDhxInbv3i1ZHL169cLly5dha2uLgQMHws3NDWXKlNGeL1u2LObMmQMnJyfJYsgvKSkJbm5uese7du0qy6janJwcbN68WW9H7tTUVMTHxwuJQamjaj/44AN88MEHesdFJVn5NWjQAMHBwRg0aJD2WGZmJpYsWYKGDRsKi2PPnj1YsmQJGjduDHd3d3Tq1Anvv/%2B%2BsPuTMjGZICIig3N2dsZ7770HDw8PzJgxA7Vq1dK7xsvLCwsXLpQ0jlatWiEwMBDW1tavvMbc3Bxr1qyRNA6N6tWrIzY2FnZ2djrH//jjD8mnFRVk9uzZOHToEBwcHBAREQF3d3fExcXB1NQUK1euFBKDEqcoKSHJyk8z%2Beunn36CWq1Gt27d8Pfff6Ns2bLC/p4AYPfu3UhMTNT2Hi1cuBD29vbo1KkTOnbsiIoVKwqLhZSDy5yIyCi83A9RUH8EeyaU49SpU0V625%2BbmyvJuNaUlJQiXVelShWD37swmzZtwv/%2B9z94enqidu3aAF4sp9m9ezcGDx6MoUOHCo3HxcUF27dvR9WqVbXLBPPy8rB48WJYWVmhb9%2B%2BQuNRyhQlTZN1QUmWn5%2BfLL0tz58/x6FDh3Qa09u0aYMSJUoIj0Xj3r17iIyMxO7du3Hp0iVcvnxZtlhIPqxMEJFRyMnJwdatW6F5//Hyz5pjJK89e/YU%2BPllmp4BqfZ9aNOmTaHL3vLy8qBSqRAXFyfJ/V/Fy8sLlStXxq%2B//oro6Gio1WpYW1vDz88Pnp6eQmMBXiyVqVq1KoAXO4Or1WqYmppi6NCh8PDwEJ5MlClTBr///rveFKWTJ0/qLE%2BTWmRkpDbJOnjwIBYuXKhNsq5duyY8mZgzZw78/f0l3QH8TV27dg2RkZE4dOgQ/vzzT7i4uMgdEsmElQkiMgpF3Wvit99%2BkzgSKkzz5s11fk5LS9ObkqRSqfQmcRna9evXtZ/z8vLQvXv3AmfgF7QB17vEy8sLLVq0wLBhw9C7d290794dXl5euHr1Kr744gvh%2BykoZYpSs2bNcObMGQCAvb09Tp06BVNTU6SlpcHDwwO///67kDg02rVrh59%2B%2BkmW8cH5xcTEIDIyUruzfcuWLdGxY0e0b98eFhYWssZG8mEyQUREklHKhC0lLIG7ffs2jh8/DuDFw2H%2B9eVqtRrLli3DuHHjhMYUGxsLX19fhIaG4sSJExgzZgxMTU2RmZkJLy%2BvAnfHlpoSpigpLclavXo1wsLC0LZtW1SvXh3FihXTOd%2BjRw8hcXz88cdMIEgPkwkiIpKMEh7ilRDHmTNnMHToUFSoUAHZ2dl49uwZQkJC8NFHHyEmJgZ%2Bfn7Izs6WfQf3mzdvIi4uDpaWlnpN4nITOUVJaUlWmzZtXnlOpVLh8OHDQuJ4%2BvQpLCwskJWVhdTUVKhUKlSpUkUvuaF3C5MJIiKSjNwP8UqJo3///nBxccGwYcMAAEuXLsW5c%2BdQs2ZNbN%2B%2BHQMGDICPjw9KliwpPLbc3Fw8ePBAZ2qRhuj9BF43RenUqVNC49FQcpIl0uPHjzF9%2BnRERkYiOzsbeXl5MDc3R%2BfOnTF16lSYmZnJHSLJgA3YREREErt27ZrO%2BNlBgwYhKCgIGRkZ2L59u%2BQb971KWFgYZs6cicePH%2Bscl6tBXQmjajXyJ1nm5uawt7cH8GK5mogkKykpCVZWVgBebLxYmMJGHxvSjBkzcPfuXSxfvhw1atQAANy4cQOrVq3C4sWL4efnJyQOUhZWJoiISDJyVQQmTpyo8/PevXvRuXNnveuk3udCo6DfgxL6SVq2bIkePXqgY8eOBb5VLmjTNikpZVStEpKs/N%2BZ%2BvXrQ6VS6Uyv0/wsMulr1qwZ9u/fr7efREpKCvr06YNDhw4JiYOUhZUJIiIymJdHsmZmZqJt27Z610m9xvvl92Tu7u56x%2BSmhB3bMzIy4OPjo7evg1yUMqo2ICAAffr0eWWSJcLevXu1nw8cOCBLDC8rVqxYgUvxypYti%2BfPn8sQESmBMv7fg4iI/hNGjhwpdwgAgEWLFr3R9WFhYXBzc5MoGuXq3LkzTp8%2BDWdnZ7lDAQDUrVsXy5cvx7Bhw1CrVi1s27YNXl5eSE5OFvqwqoQky9raGrt374anpydsbGy0x2NjY/Hxxx/LElOTJk0wa9YsTJgwQVudePDgARYvXixbTCQ/LnMiIiLZzJw5E9OnT5c7DMmXY9WvX1/7xl3jzp07esdETOXJPxEpKysLYWFhaNKkCaysrPSqJb6%2BvpLHk59SpihNnz4dn332mexJVkHfSzmHCaSkpMDb2xtxcXEoW7YsgBd7ydSuXRsrV67USXro3cFkgoiIZCP3lCUNqfsXtm3bVqTrevbsKVkMGv379y/SdSqVCj/99JPE0RRO5BQlJSZZBX0v5eq1UavVSElJgbW1Na5evYqkpCTtPiD29vYcD/sO4zInIiKSjVLeZ0ndv/CmSYKUFZuQkBBJ/lxDkWuK0vnz53V%2BtrGxwb1793Dv3j2d4yJ7XQq6lxy9NmlpaejXrx9sbW0xd%2B5c1K9fXzuBrFu3bihdujTWrVsHU1NT4bGR/JhMEBGRbJTQhKxEO3bsELL8S61WY8uWLdpqRVRUFLZv346aNWti5MiRKF26tOQx5CfnFCWlJ1lyWr58OSpWrAh/f3%2B9c5s2bcKQIUOwZs0ajBgxQoboSG4mcgdAREREukRVbGbNmoU9e/YAeLGkyNfXFw0bNsStW7cQEBAgJIb8NFOUduzYgX379mn/CQsLw759%2B4TFoVardZKLqKgoeHt7Y8GCBXj27JmwOJTi0KFD8PPzQ6lSpfTOlSxZEn5%2BfjrTp%2BjdwsoEERGRwoiq2ERFRWmTidDQULi4uGDkyJF48uQJOnXqJCSG/JQwRQl4kWTFx8ejf//%2B2iRryJAhiI%2BPR0BAAObOnSskjoJGKxd0TOrG/fv376NevXqvPF%2B/fn3cuXNH0hhIuZhMEBHRO08pvRuiZWZm4r333gMAHD9%2BHF5eXgAACwsLWd7AK2VUrVKSrNmzZwu7V2FKlSqFhw8f6m1Wp5Gamlrg/hP0bmAyQUREslHKQ3zz5s3lDkEWderUwY4dO2Bubo4///wTrq6uAIATJ06gWrVqQmLIP0WpVKlSmDx5suxTlJSSZCmlcb9FixYIDg5%2B5e9/4cKF7%2By/Q8RkgoiIJJSTk6OdhlO5cmW9h0MRm9xt27YNYWFhuHXrFlQqFaytrdGtWzedTep%2B%2BOEHyeNQoilTpmDixIl4%2BvQp/Pz8UK5cOTx69AgjR44UtpRHiVOUlJBkvQ2pGvdHjBiBHj16IDExEV5eXqhVqxZycnLw559/Yv369bh48SK2bt1q8PuSceA%2BE0REZHBPnjzBzJkzcfDgQajVagCAubk5PD094efnJ2yE5NKlS7F582adXYRv3ryJ3bt3Y/z48ejVq5eQON6UXHsJaKSkpKBKlSqy3V9uFy9e1CZZY8aMQc%2BePfHo0SN88sknmDt3riz9JEUh5fcmLi4Os2fPxrlz57SJXV5eHhwdHeHn51doTwX9tzGZICIigxs3bhxu376NIUOGoEaNGgCAGzduYPXq1XBwcMDkyZOFxNGmTRsEBQWhYcOGOscvXLiAKVOmICwsTEgcL3tdxeaHH37A0KFDJbn3li1binxt7969JYnhVZQ2qvZlSk%2ByRGwC%2BeDBAyQmJkKlUsHGxgbly5eX9H6kfEwmiIjI4BwdHREeHo5KlSrpHL9z5w769u2LQ4cOCYmjSZMmOH36tN50oOzsbDg5OeHs2bNC4tBQQsVGs2TndVQqFaKioiSORpe/vz/i4%2BOxdetW3Lx5E127dtVOUbKwsJB06ZWSk6yiUsqO8vRuYc8EEREZnImJSYEz6cuXL4/nz58Li%2BPDDz/Ezp079RpZQ0NDtRUTkWbMmIHbt2/ju%2B%2B%2B06vYBAYGCqnY/Pbbb5Lf423JOUVp9erVRbpOpVIpNpkgkgOTCSIiMjg7OzvMmTMHEyZM0C6DePToEQIDA9GoUSNhcUyYMAFff/01QkJCULt2bQAveiYSEhKwbNkyYXFoHD16VK9iU7t2bTRu3Bh9%2B/YVkkwcO3asSNepVCq0bNlS4mh0yTlFSclJFpGSMZkgIiKDmz59OoYPHw5nZ2dUqFABAPDw4UPUqFEDK1euFBZHs2bNEBkZiT179iApKQlqtRru7u7o1KkTrK2thcWhoYSKzddff12k61QqFeLi4iSORpecU5SUnGQVFVeukxzYM0FERJK5fPmy9iHexsYGtra2QkZ87t69G56enpLf500NHz4clSpVKrBic/v2baxbt07mCOUl5xSl%2BvXrF%2Bk6OZIsDTkb94lehckEEREZzLRp0zBr1iy5w1BsI2pycjKGDx%2BO69evF1ixqVmzpuQx/P3339p%2BjYSEhFdep1KphMRTFEqfoiQ1JTTuE70KkwkiIjIYpTzEy71Pw%2BvIVbEBdH83hb2NF/UGXilTlJScZCll1DJRQZhMEBGRwSjlId7W1hYxMTGvXUMu6o2uUio2AHD79m1Ur14dAHDr1q1Cr7W0tJQ8HqWMqlVakpWfUkYtExWEDdhERGQwubm5OH78%2BGsf4l1cXCSNIzMzE40bN37tdaIeCkNDQxWTTFSvXh2urq5FqoSI2GdCKVOUIiIitJ9F76/xOkpo3Cd6FSYTRERkMNnZ2Rg8eHCh14h4s1u8eHGsX79e0nu8CaUtAsjfpJuXl4eAgAD4%2B/vLEotSpigpLcnKTymjlokKwmVORERkMErpmVBKHBqNGjXC6tWrZa/YvIqcvy8lTVHavHmz9nNhSVafPn0kjeNlSmjcJ3oVJhNERGQwSnmIV0rvhkZRHpjlHDmqlL83pVHa70XOxn2iV%2BEyJyIiMhilvJ/y9vZ%2Bo%2Bul3pfCzMxMUQ%2BlSqLkKUpye7lxv1GjRlzWRIrDygQRERlMTEwMHBwciny9UqYcSf0GWmlvuF8mZ3xKnqIk99%2Bb3PcnKgpWJoiIyGDeJJEAlDPlSOr3akp7b/fy3g45OTnYunWrXpxS7uugoeQpSnJT2veGqCBMJoiISDZKeViSet35m06Wkrpis3r1ap2fK1eujFWrVukcU6lUQpIJJU1RUlKSBShn1DJRYbjMiYiIZKOUZRxKiUNDafFITSlTlIqygZ7Um%2Bflp/TGfSKAlQkiIiLFedfe872cJMyfP1/4%2BFVAORvoabBxn4yBidwBEBERkS6O%2ByQiY8FkgoiI3nnvWiWAjAO/l2QMmEwQEZFslPKw1LJlS7lDINLzNo37RKKxZ4KIiCT18OFDVKhQocBzY8aMkfz%2B%2B/btw65du3D37l3s2rULarUaISEhGDRokHY50cqVKyWPg15NaVOUlMJYRy3Tu4XJBBERGdyzZ8%2BwYMEC7N69G9nZ2bh8%2BTIePXqESZMmYd68eahYsSIAYNCgQZLGERQUhC1btqB3797a0aePHz/Grl278OTJEyHJzNtQSsVGFCWNqjVm79r3hpSBo2GJiMjgJk2ahNTUVIwYMQKDBg3CpUuX8OzZM0ybNg15eXlYsmSJkDjatGmDtWvXok6dOjrjVhMTEzFgwAAcOnRISBwFKaxis379eskTLfrveddGCpMysGeCiIgM7vDhwwgMDISDg4N2KVHp0qUxffp0nDx5UlgcT548QZ06dfSOV65cGQ8ePBAWh4YmobKzs0OrVq0AAI8ePcKwYcN04mEiQUTGgskEEREZnEqlgoWFhd7xnJwcZGZmCoujbt262L17t97x9evXo3bt2sLi0Jg1axYSExOxdu1amJi8%2BE9wiRIlYGFhgTlz5giPh4jo32LPBBERGZy9vT0WLlyI8ePHa4/dunULAQEBcHR0FBbH6NGjMWLECPz888/IysqCt7c34uPjkZaWhqCgIGFxaBw%2BfBjh4eGoWLGiXsXms88%2BEx4PEdG/xWSCiIgMburUqfjmm2/g4OCA7OxsNG3aFM%2BfP4ednZ2wfgkAaNGiBcLCwrBv3z7Uq1cP5ubmcHFxgbu7O8qXLy8sDg2lVGzov4ltsCQHJhNERGRw1atXx65duxAbG4vExESYmZnBxsamwP4FqZUqVQrdunXTNjsnJSXJ9tCllIoNGS%2B5Ry0TvYw9E0REJImTJ0%2BidOnScHNzQ7t27ZCWloZjx44JjSE6Ohqurq46Td9HjhzBp59%2BilOnTgmNBXhRsYmJiYGDgwMyMzPRtGlTtG/fHg8fPsT06dOFx0PGgY37pGQcDUtERAYXEhKC//3vf1i2bBlatGgBAIiKisKUKVPg4%2BODL774QkgcXbt2xYABA9C1a1ed4/v27cPatWuxc%2BdOIXG8TAkVGzIeShm1TFQQJhNERGRwrq6uCAoKQv369XWOx8fHw9vbG1FRUULisLe3R0xMDIoVK6ZzPCsrC46Ojjh//ryQOPI7efIkqlSpgg8%2B%2BAAAEBMTg4yMDLi4uAiPhYyDk5OTtnE//14Sjx8/xmeffSZ03DLRy7jMiYiIDO7hw4fah%2BX8rKyshO7vYGNjg4MHD%2Bod37VrF6pXry4sDo2QkBD4%2BPggJSVFeywtLQ3jxo3Dxo0bhcdDxoGN%2B6RkrEwQEZHBDR48GHXq1MGIESNQpkwZAMC9e/fw/fffIykpCcHBwULiOHbsGHx8fFCjRg1YWVkhNzcXCQkJSE5Oxtq1a%2BHg4CAkDg2lVGzIuHh7e8PS0hLjx4%2BHk5MTLl68qG3cz83NxapVq%2BQOkd5hTCaIiMjgEhMT4ePjg/j4eFhYWCA3NxfPnj1DgwYNsGrVKlSuXFlYLCkpKQgPD0diYiJUKhWsra3RuXNnVKpUSVgMGvb29jh16hRMTU11jj9//hwtW7aUZdkVKd/t27fxzTff4M8//0R2djZKly6tM2q5WrVqcodI7zAmE0REJJk//vgDiYmJMDExgbW1td4b%2BXeNUio2ZJzYuE9KxGSCiIgkkZOTg9TUVGRkZOidq1WrlpAYrl69iu%2B%2B%2Bw43btwoMA7Ro2qVVLEh48LGfVIqJhNERGRwu3fvxuzZs/H06VOd43l5eVCpVIiLixMSh4eHB6pUqQJXV1eULFlS7/zLI2NFYcWG3oRSRi0TFYTJBBERGVybNm3Qs2dPdOrUCebm5nrnLS0thcRhb2%2BP6OhomJmZCblfUSihYkPGhY37pGTF5Q6AiIj%2Be54%2BfQpvb2%2B9/R1Ea9CgAe7cuYMaNWrIGoeGUio2ZFyUMmqZqCBMJoiIyODatWuHU6dOwdnZWdY4Bg4ciEmTJqFLly6wtLSEiYnu9kqi15sHBgbiyy%2B/fGXFhqggTZo0wZIlSwps3Le1tZU5OnrXcZkTEREZ3OrVq/Hzzz/D3t4eVlZWeg/xvr6%2BQuIorBdBjkpA06ZNcfr0adkrNmRc2LhPSsZkgoiIDK5///6vPKdSqfDTTz8JjEY5Jk6ciM8//1z2ig0ZJzbukxIxmSAiIqHi4uLQoEEDWWNIT0/Hp59%2BKnw0rFIqNmR82LhPSsWeCSIikkReXh5u374NtVqtPZaSkoJvvvkG586dExLDnTt3MHfuXFy%2BfFknjmfPnsmyNOTYsWOwsbHB/fv3cf/%2BfZ1zKpVKeDxkHNi4T0rGygQRERlcTEwMRo0ahYcPHwL4/w89ANC%2BfXssW7ZMSBxDhw7V3nPWrFmYMWMGrly5gri4OKxYsQKVKlUSEkdRKKFiQ8qklFHLRAVhMkFERAbXrVs3tGvXDm5ubvD09ERYWBguX76MsLAwTJ06VVhVwNHREUeOHEHJkiVha2uLixcvAgD27t2LmJgYzJgxQ0gc%2BSmhYkPGhY37pGRc5kRERAaXkJCAb775BiqVCiqVCtbW1rC2tka1atUwadIkbNiwQUgcxYsX1/YlmJmZ4dGjRyhfvjw6dOiAmTNnCk8mXlexISqIUkYtExWEyQQRERlcuXLlcPfuXVSuXBlly5ZFYmIirK2t0bBhQ1y4cEFYHA4ODhg5ciSWLl2Kjz/%2BGPPnz8cXX3yBCxcuyLIr9ty5c%2BHl5fXKig1RQWrXro3JkyezcZ8UickEEREZXOfOndG9e3eEh4ejVatW8PHxgaenJ2JjY2FlZSUsjpkzZ2LRokUoXrw4vv32WwwbNgy7du1CqVKlMHPmTGFxaCilYkPGhY37pGTsmSAiIkns2rULXbp0wbNnzzBz5kzExsbC0tISEyZMEDYfP/8yIs3P9%2B7dQ8WKFXH37l1UrVpVSBwabdu2xdatW1G5cmW4uLjgl19%2BgbW1NbKysuDo6Ijz588LjYeMHxv3SW5MJoiIyOBOnz4NR0dHveOZmZn47bff0KlTJyFx5G%2B6zu/JkydwdXXFmTNnhMShsXjxYoSGhiI8PBwBAQGIi4vTVmz%2B/PNP7NmzR2g8ZDzYuE9KxWSCiIgM7lUP8ampqfj0008LPGdI%2B/fvx/79%2BxEREVFg4nL79m0kJCQgOjpa0jgKooSKDRkXpYxaJioIeyaIiMhggoODsXbtWqjVari4uOidf/r0qZCeiY8%2B%2BghJSUmIiIiAqamp3vl69eph3LhxksfxstOnT%2BPzzz8HAFhYWGDRokUA/n/FhskEFYSN%2B6RkrEwQEZHB5Obm4sqVK%2Bjbty9mz56td97MzAwtWrRAhQoVhMSzfv16DBo0SMi9ikLuig0ZJ3t7e5w7dw4qlQqNGzfGpUuXAAAXLlzA//73Pzbuk6xYmSAiIoMxMTHBxx9/jJ9//hmNGzfWOZeWloZy5coJjadbt26YP38%2Bvv32WwDApk2bsGXLFtSsWRP%2B/v7CNs9TSsWGjJNSRi0TFYTJBBERGVyxYsXQrVs37NixAwAwevRo7N%2B/HxUqVEBQUBDs7e2FxDFt2jRkZ2cDAGJjY7Fo0SLMmDEDly9fxpw5c7B06VIhcQwYMABNmzZF3759C1xepanYEBVEKaOWiQrCZU5ERGRwffv2RatWrfDNN98gMjISM2bMwNatW3Hu3Dn8/PPP%2BPnnn4XE4eTkhMjISJQpUwbz5s3D/fv3sXjxYmRkZMDV1RUnTpwQEofGpUuXFFGxIePDxn1SKpPXX0JERPRm4uPjMXToUABAVFQU3NzcUL16dbi7u%2BP69evC4sjNzYWFhQUA4Pjx42jXrh0AoESJEkhPTxcWh4amYqMxevRoODk5oUWLFtxjgl5J07ivUqm0jfsREREICgpCQkKC3OHRO47JBBERGZyZmRmysrKQk5ODo0eP4pNPPgEAPH/%2BHLm5ucLiaNSoEVasWIE1a9YgNTUVbdu2BQCEhYWhVq1awuLQmDNnDtq3bw8AiIyMxNmzZ/Hbb7/Bz89PO9mJ6GVDhgwp8HhaWpq2H4hILuyZICIig2vdujVGjRqF4sWLw8LCAs2bN0dWVha%2B%2B%2B47NGnSRFgc06dPx%2BzZs/H48WMsWrQIJUuWxKNHj4T2S%2BQXHx%2BPkJAQALoVm2rVqmHmzJnC4yFlY%2BM%2BGQP2TBARkcFlZGQgODgYT548Qb9%2B/WBpaYnnz5/Dx8cHAQEBqFq1qqzxZWZmwszMTPh9nZ2dERUVBVNTU7Rp0waLFi1CixYt8OzZM7Ru3Rpnz54VHhMpl9JGLRMVhMkEERH9pyxduhSjRo0CACxZsqTQa319fUWEpPXtt9/i/v37KF68OBISEhAeHo7s7GwsWLAAf//9N9asWSM0HjIObNwnJeMyJyIiMoj%2B/ftrl/D07t0bKpXqlddu3rxZsjjyb/xWWFNzYfFJZcaMGdqKjb%2B/P1QqFbKyspCQkICAgADh8ZBxUMqoZaKCMJkgIiKDcHZ21n5u1aqVbHGsW7dO%2B1mT3CiFubk5hg8frnOsVKlSOjETvexVjfvnzp3DokWLhI1aJioIlzkREdF/1unTpxEREYGkpCSYmJjggw8%2BgIeHBxo0aCAsBqVUbEd0k0QAABIzSURBVMh4NW3aFKdOnULx4sUxefJklClTBlOmTEFeXh4cHR1x5swZuUOkdxgrE0REZFDp6elYt25dgQ/xXl5eKF5czH96pk2bhh07dsDZ2Rm1atVCTk4OLly4gA0bNmDw4MEYP368kDiUUrEh46UZtaxSqXD06FHtGGHRo5aJCsLKBBERGUxmZib69u2Lp0%2BfolevXtqH%2BCtXruCXX35B3bp1sX79epiamkoax549e7BgwQJs2LABderU0Tl34sQJjB8/HpMnT4aHh4ekcRAZAhv3ScmYTBARkcGsXr0av/32G4KDg1GyZEmdcw8ePMCQIUPg6uqKESNGSBrHV199hV69esHNza3A8%2BHh4QgODsaWLVskjSM/pVRsyPgofdQyvduYTBARkcF069YNkyZNgpOTU4Hnz58/j8mTJyMiIkLSOJycnLB37168//77BZ5Xq9VwcnIqdNqTISmlYkNEZGh8DUJERAaTkJCAhg0bvvK8nZ0dkpOTJY8jPT39lYkEAJiamgpdax4cHIwSJUogNDRUp2LToUMHfPnllxgyZAjWrFkjecWGjAcb98lYMJkgIiKDyc3NhYWFxSvPi9rbQY49JAqzf/9%2BTJo0SW/pFwBUrFgR/v7%2BmDx5MpMJ0mLjPhkLJhNERGRQWVlZkHsFrVqtRp8%2BfQq9JisrS1A0yqnYkPHw9vbWfh45cqSMkRAVjskEEREZTGZmJho3bix3GEV6w%2B/i4iIgkheUUrEh48PGfVI6fgOJiMhgfvrpJ7lDAPDmb3JXrVqltzO1oSmhYkPGJTMzE15eXgU27q9YsQIHDx5k4z7JjtOciIhINoMHD8a6devkDgO2tra4ePGiZH9%2B/fr1i1R9iIuLkywGMj5KGbVMVBhWJoiISDYxMTFyhwAAklcMlFKxIePCxn0yBkwmiIjonSd1z4Kjo%2BMbXa%2BUig3Ji437ZAxM5A6AiIiIdCmlYkPyYuM%2BGQNWJoiIiIgUio37pHRMJoiIiIgUSCmjlokKw2SCiIjeeXzzS0rExn0yBkwmiIhINkp5iO/Vq5fcIRDpYeM%2BGQMmE0REJJuFCxca/M8cN25cka8NDAwEAPj7%2Bxs8DiLR2LhPcmAyQUREBuHi4lLka48dOwYA6Nixo8Hj%2BC/sBqyUig0R0eswmSAiIoN4k4qAlObNm1ek637%2B%2BWeJI3l7UlRsiIikoMrj6w8iIhLI19cXS5YsEXa/%2BPh4XLlyBWq1WnssJSUFGzZswPnz5yW//9tUbIjehq2tLS5evCh3GPSOYWWCiIgMLicnB5s3b8bly5d1HuJTU1MRHx8vLI5ffvkFs2fPRqVKlXDv3j1UqVIFqampsLS0xOjRo4XEoJSKDRGRFJhMEBGRwc2ePRuHDh2Cg4MDIiIi4O7ujri4OJiammLlypXC4li3bh3Wr1%2BP5s2bo3Hjxjh8%2BDDu3r2LgIAANGrUSEgMXbt2LdJ1vr6%2BRb6WiEgpmEwQEZHBRUZGYvv27ahatSoOHjyIhQsXIi8vD4sXL8a1a9fQpEkTIXHcv38fzZs3BwCYmJggLy8P77//PiZMmIDhw4djz549QuLQUErFhv6buHKd5GAidwBERPTfk5mZiapVqwIAihUrBrVaDZVKhaFDhwqtTFSvXh3R0dEAgPfff187OrNMmTJISkoSFofG7Nmz8cMPP0CtViMiIgLFihVDfHw8nj9/LvT3Qv9NbNwnObAyQUREBle3bl0sX74cw4YNQ61atbBt2zZ4eXkhOTkZz58/FxbHsGHDMHjwYERHR6N79%2B7w9vaGg4MDbt68iaZNmwqLQ0MpFRtSPqWMWiZ6HU5zIiIig4uNjYWvry9CQ0Nx4sQJjBkzBqampsjMzES/fv3g5%2BcnLJakpCRYWVkBALZt24bY2FhYWVmhb9%2B%2BKFOmjLA4AKBZs2Y4c%2BYMAMDe3h6nTp2Cqakp0tLS4OHhgd9//11oPKRcO3fuLPK17LUhOTGZICIiyd28eRNxcXGwtLSEnZ2d0Hs/fvwYOTk5qFChAgAgMTERFhYW2p9F8vLyQosWLTBs2DD07t0b3bt3h5eXF65evYovvviCOxjTGxM9apnoZUwmiIjI4AYOHIgNGzboHX/69Cn69%2B//Rm9d/43o6GiMHDkSs2bNgpubGwBg06ZN%2BO6777BixQo4OTkJiUNDSRUbMh6va9w/deqUjNHRu47JBBERGcyVK1cQGxuLOXPmYNq0aXrTZf755x/8/PPPQjaLA14s/xgwYIDeMpB9%2B/Zh7dq1wpKaV5GzYkPGY8aMGa8ctezn58deG5IVkwkiIjKY6OhoBAcH4/Dhw6hevbreeXNzc/Tq1QtfffWVkHjs7e0RExODYsWK6RzPysqCo6OjsKRGQykVGzIuLi4u2sb9xo0b49KlS9rGfU3/D5FcOM2JiIgMpnnz5mjevDm8vb0VMerUxsYGBw8e1Jtys2vXrgKTHaloKjZnzpzB1q1bC6zY/PXXX8LiIeNS0KhlU1NTDB06FB4eHkwmSFZMJoiIyOBWrlyJe/fu4a%2B//kJGRobe%2BTcZe/lvTJgwAT4%2BPli1ahWsrKyQm5uLhIQEJCcnY%2B3atUJiAIAnT57g8OHDyM7OxqpVq/TOm5ubY/To0cLiIeOilFHLRAXhMiciIjK4devWYcmSJcjJydE7p1KpEBcXJyyWlJQUhIeHIzExESqVCtbW1ujcuTMqVaokLAYNpVRsyLiwcZ%2BUjMkEEREZXIsWLTBhwgS4ubnB3Nxc7nAURQkVGzJubNwnJWEyQUREBufk5IQTJ07oNT6L0L9/f4SEhAAAevfuDZVK9cprN2/eLCosAMqq2JDxYOM%2BKRl7JoiIyOC6du2KvXv3okuXLsLv7ezsrP3cqlUr4fcvzNq1azF79mxWbKhI2LhPxoDJBBERGVx2djbmz5%2BPjRs3wsrKCiYmJjrnAwMDJbu3t7e39rOVlRU%2B//xzvWvS09OFVyUAIDc3F126dJGlYkPGh437ZAy4zImIiAxu8uTJhZ6fN2%2BepPfPzc1FdnY2mjVrhpiYGL03ujdv3kSvXr1w6dIlSeN42fz589GgQQNZKjZkvNi4T0rGZIKIiP5zgoODsWDBgkKvsbOzwy%2B//CIoohfmzJmDffv2wcrKSnjFhowbG/dJqbjMiYiIJPH7778jPDwcSUlJUKlUsLGxweeffw4HBwfJ7/3VV1/B09MTrVu3xvr16/XOm5ubo0GDBpLH8bJnz56hbdu2wu9Lxo2N%2B6RkrEwQEZHBhYSEIDAwEG3btkWNGjUAvFhadPjwYSxZsgSffvqpkDhWrFiBESNGCLkXkVQ4apmUjMkEEREZXLt27RAQEIDmzZvrHD969CgWL16M0NBQIXG4uLhg9%2B7dqFixopD7FYWcFRsyTnKOWiZ6HSYTRERkcPb29oiJidF7%2BMnJyYGjoyPOnj0rJI7g4GBERUXBzc0N1atX14tH9FpzpVRsyLiwcZ%2BUjD0TRERkcDY2Njhy5AhcXV11jh87dgzVq1cXFsf8%2BfMBAGfOnNE7J8da8%2BDgYKxateqVFRsmE1QQOUctE70OkwkiIjI4Hx8fjBo1Cs7OzqhduzaAF2/gjx8/jjlz5giL4%2BrVq8LuVRQPHjxAs2bN9I47OzsjKSlJhojIGLBxn5SMyQQRERlc%2B/btsX37duzYsQN///031Go1bGxssHHjRtjZ2QmP59KlS7hz5w46dOgAAMjMzISZmZnwOJRSsSHjIvW%2BLET/BnsmiIhIqOTkZFSrVk3IvW7cuIGRI0fi9u3byMnJweXLl3Hr1i307NkTa9euxUcffSQkDo3IyEiMGTPmlRWbgnbrJgLYuE/KZfL6S4iIiN5M%2B/btsW7dugLPdezYUVgcs2bNQrt27XDmzBntOnNLS0sMHTpUlre9mopNzZo18ffff%2BP69euwtLTExo0bmUjQK4WEhGDUqFFIT09HkyZNYG9vj7S0NAwcOBAHDx6UOzx6x7EyQUREBtewYUNYWVmhRo0aWLhwIcqXL68917hxY1y6dElIHPb29jh16hRMTU1ha2uLixcvAnjR0Nq8eXPExMQIiaMoRFZsyLgoZdQyUUFYmSAiIoMrXrw4duzYgTJlyqBLly46D%2B0qlUpYHOXLl8fjx4/1jv/zzz8oXlx826BSKjZkXNi4T0rGZIKIiCRRunRpBAYGwsfHB8OGDUNQUBDy8vIgsiD%2BySefYNSoUTh27Bjy8vIQFxeHnTt3Yvjw4XB3dxcWh0ZycjK2bt2KoUOH4tGjRzrnuFCAXkXTuP8yNu6TEnCZExERGdzLS5lu3LiBMWPG4L333kNMTAxiY2MlvX9GRgbMzc2RmZmJRYsWYefOnXj27BmAF9WK3r17Y8SIETA1NZU0jpfZ2trixIkTmDZtGmJiYhAYGKhtoM2/DIsoPzbuk5IxmSAiIoMbPHiw3nKezMxMBAQEYOvWrZLv/9C0aVN07twZPXv2RKNGjZCXl4f79%2B/D3NwcFhYWkt67MPkThu3bt2PevHkYPHgwvL29YWtrK6yXhIzP1atXsWPHDiQlJWlHLXt6esoyapkoPyYTRET0nxMaGorQ0FBER0ejTp066NmzJzw8PFCuXDlZ45K7YkP/PWzcJ7kxmSAiIoP49ttvMX/%2BfADAuHHjCr02MDBQREi4c%2BcOdu3ahV27diE5ORnt27dHjx490KJFCyH3f5ncFRsyTu3bt0ffvn0xePBgvXNcHkdyYwM2EREZRIkSJbSfTU1NC/1HlKpVq2L48OGIiIhAcHAwSpcujTFjxqB9%2B/ZYtWqVsDg0CprkZGZmhlmzZjGRoFdi4z4pGSsTRERkcBcvXoStra3cYRTo/PnzmDdvHmJjYxEXFyf5/ZRYsSHjwsZ9UjJWJoiIyOAGDx4MtVotdxhaKSkpWL16NTp27IhBgwahVq1aCAkJEXJvJVZsyPgoYdQyUUFYmSAiIoP78ccf8c8//6Bfv36oXr06ihUrpnNexIOzWq3GgQMHsHPnTkRHR6N%2B/fro0aMHPDw8ZJvopOSKDSkXG/dJyZhMEBGRwdnb2yM7OxvZ2dkFnpd6edHUqVMREREBlUoFDw8P9OzZE/Xr15f0nkXh4OCAEydOsApBb4SN%2B6RkTCaIiMjgTp8%2BXeh5R0dHSe//5ZdfokePHvjss88U9eCuhIoNEZEhMZkgIiLJZGVlITU1FSqVClWqVNF7eH7XyF2xIePBxn0yFsXlDoCIiP57Hj9%2BjOnTpyMyMlL74GxmZobOnTtj6tSpMDMzkzlCeaxevVruEMhIvNy4T6RUrEwQEZHB%2Bfr6IjU1FUOGDEGNGjUAvGgaXbVqFezs7ODn5ydzhPJixYbeBBv3ScmYTBARkcE1a9YM%2B/fvR8WKFXWOp6SkoE%2BfPjh06JBMkcmLFRt6G2zcJyXjPhNERGRwxYoVQ8mSJfWOly1bFs%2BfP5chImWYMWMG7t69i%2BXLlyM8PBzh4eEIDAzEtWvXsHjxYrnDI4Xy8fHBggULcOPGDaSnp0OtVuv8QyQnViaIiMjgvvnmG5QrVw4TJkzQVicePHiAxYsXIzU1FWvXrpU5QnmwYkNvg437pGRswCYiIoObPn06vL294ezsjHLlygEA0tLSULt2bQQFBckcnXxYsaG3wcZ9UjJWJoiISDJXr15FUlIS1Go1bGxs0KhRI7lDkhUrNvRvsHGflIjJBBERGYyrqytUKtVrr4uKihIQjfKkpKTA29sbf/zxR4EVG83kK6L82LhPSsZkgoiIDGbz5s3az3l5eQgICIC/v7/edX369BEZluKwYkNvgqOWScmYTBARkWRsbW1x8eJFucOQHSs29G%2BwcZ%2BUjA3YREREEhs6dKj2c2EVG6KCsHGflIyVCSIikgwrEwXj74XeBBv3SclYmSAiIiJSMI5aJiVjMkFERAazZcsWnZ9zcnKwdetWvFwE7927t8iwiIxalSpVsGPHDjbukyJxmRMRERmMq6vra69RqVTvfKMxlzlRUbBxn4wBKxNERGQwv/32m9whKBIrNvQ22LhPxoCVCSIiIomxYkOGwIoWKRErE0RERBJjxYaI/qtM5A6AiIiIiIiME5MJIiIiIiJ6K1zmRERERKRAbNwnY8AGbCIiIiIFYuM%2BGQMmE0RERERE9FbYM0FERERERG%2BFyQQREREREb0VJhNERERERPRWmEwQEREREdFbYTJBRERERERvhckEERERERG9FSYTRERERET0VphMEBERERHRW/l/kBnmdeulKBIAAAAASUVORK5CYII%3D" class="center-img">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAxMAAALICAYAAAAE1K0IAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzs3Xl8Dff%2Bx/HXSWRBCCGxB6WNLUiLxNIi6lJKaKmtWtdW%2B9qW9ufWeqlWLrUXLUppbrVFaW8rVLWKq0UbS1BRuyRISCKR7fz%2BiJzmSBCZQW7zfj4eHidm5vs5n/lmTmY%2B852ZY7FarVZERERERETukcPDTkBERERERP43qZgQEREREZE8UTEhIiIiIiJ5omJCRERERETyRMWEiIiIiIjkiYoJERERERHJExUTIiIiIiKSJyomREREREQkT1RMiIiIiIhInqiYEBERERGRPCn0sBMQEZG/tt9%2B%2B43PP/%2Bc/fv3c/bsWRITE3FxccHLy4s6derQrl07AgMDsVgsDztVERG5Rxar1Wp92EmIiMhfj9Vq5Z///CerVq0CwN3dncceeww3Nzfi4%2BM5cuQI8fHxALRs2ZK5c%2Bfi7Oz8MFMWEZF7pGJCRETui48//pgpU6bg6urK5MmT6dChA46Ojrb5SUlJfPLJJ8ycOZP09HQGDhzI2LFjH2LGIiJyr3TPhIiI3Bdr164F4KWXXqJTp052hQSAq6srffr0YcCAAQCsXr2apKSkB56niIjknYoJERG5L06fPg2Aj4/PHZfr06cPy5YtY8uWLbi6ugIwfvx4fHx8mDlzJgkJCcyYMYPAwEB8fX0JCAhg2LBhHDt27LYxd%2B3axbBhw2jWrBl16tQhICCAPn368OWXX962TWxsLLNnz6ZTp074%2BflRu3ZtmjRpwqBBg9izZ0%2BObQIDA/Hx8WHHjh1s2bKFdu3a4evryy%2B//ALAvHnz8PHxYfz48SQnJzNv3jzatGlD3bp1ad68OVOnTuX69esAHD9%2BnBEjRtC0aVN8fX3p0KEDGzduzPF9ExMTWbp0KS%2B88AINGjSgVq1a%2BPv78/e//50tW7bk2KZ37974%2BPiwdu1aEhISmD17ti2Xhg0bMmDAAA4ePHjb/hERyYmKCRERuS88PDwAOHDgwF2Xe/LJJyldunS2eQkJCfTp04e1a9dSsmRJGjZsSFpaGlu2bKFr1645HvzOnj2bPn36sGXLFtzc3PD398fDw4Ndu3bx6quvMmrUKNLS0uzaREdH89xzz7F48WIiIiKoVasWAQEBODs789133/Hyyy/z2Wef3XYdIiIiGD16NE5OTgQEBNiKoqxee%2B01PvzwQypUqMCjjz5KVFQUq1evZty4cRw7dowePXoQHh5OjRo18PT05NixY7z22mts377dLk5iYiI9e/Zk1qxZHDp0iGrVqtG4cWPc3d356aefGDZsGPPmzbttrsnJyfTp04eVK1dSrlw5ateuTVpaGjt27KB37962IlBEJDf0NCcREbkvAgIC%2BOKLL1i9ejXOzs706dMHLy%2Bve4qxefNmypcvT2hoqK1tXFwcffv25bfffmPy5Ml8%2BumntuW3bNnC4sWLcXd3Z/78%2BTRq1Mg2b%2B/evYwaNYqvv/6a%2BvXr06dPH9u8JUuWcO7cOapWrcrHH39MqVKlAEhLS2PmzJmsXLmSGTNm0LZtW4oWLZotz1WrVjFs2DAGDRqU43rs3LkTDw8PtmzZYiuaQkNDGTp0KFu2bOH333%2BnT58%2BDB06FIvFQnp6OiNGjGDLli2sXr2aFi1a2GKFhIRw%2BPBhPDw8WLt2LVWqVLHNW7FiBTNmzGDRokV06tSJSpUqZctlxYoVlC9fnm3bttkKvujoaDp37kx0dDQff/wxb7zxxl1%2BMyIiGTQyISIi98WoUaMoV64cVquVDz74gBYtWvDiiy8yZ84cfvrpJxITE%2B8aIz4%2BnkmTJtkVIcWKFWP8%2BPFAxmNn//jjD9u8RYsWATBu3Di7QgKgYcOGvP766wCsXLnSbp6Xlxft27dn2LBhtkICwNHRkdGjR%2BPg4EBcXNxtR1kybyC/naioKCZPnmw3%2BvL000/j6emJ1WqlUKFCtkICwMHBgW7dugFw%2BPBhu1hFixalQ4cODBo0yK6QAHj55Zfx9PQkLS2NXbt25ZhLdHQ0//rXv2yFBICnpycdOnQA4Ndff73teoiI3EojEyIicl%2BULVuWTz/9lHfffZdNmzaRlpbG3r172bt3L4sWLcLJyQl/f386duxI%2B/btKVQo%2By7Jy8uLJ554Itt0Pz8/ihYtSkJCAgcOHKBKlSpERkZy6NAhAFq3bp1jTk8//TQWi4Xz589z8uRJqlatCmC7CTwnhQsXplSpUkRHRxMdHZ3jMgEBATg43P78nKenJ/Xr1882vUKFCkRHR9OyZcts37NRoUIFAK5evWo3vWvXrnTt2jXH97FYLFSsWPGOuTZq1IgyZcpkm%2B7t7Q1ATEzMbddDRORWKiZEROS%2B8fT05J133uH1119ny5Yt7Nq1i59//pnLly%2BTkpLCjz/%2ByI8//sjSpUtZuHCh7YA202OPPZZjXAcHBypUqMCxY8c4e/YsgN0N2ZkjFzkpVKgQKSkp/PHHH7ZiAjLuJfj%2B%2B%2B85ePAgUVFRXLt2jcynp8fFxQEZIxA5yengPKuKFSvmOD3zezUyC4ec5qWmpmabl56ezs6dO9m/fz9RUVHExsbacjt58uQdc83p0icAFxcXAFJSUu60KiIidlRMiIjIfVe6dGl69OhBjx49gIwblvfs2cOGDRvYv38/x48fZ%2BDAgXz55Zc4OTnZ2rm7u982ZrFixYCMm7TB/gz%2B1q1b75pTZoEAEB4eztChQ22Fyb26U56A3TrlZX5W58%2BfZ%2BjQodkuf8otfTGgiJhJxYSIiDxwjzzyCI888gg9evRg5cqVTJ8%2BnZMnT/LVV18RFBRkW%2B5Olw5lnnnPXCbzMiEnJyfCwsKyXTZ0O0lJSQwePJjz589TpUoVhgwZQuPGjSlZsqTtID8wMJBz587dNkZu38sMI0aM4PDhw5QuXZrhw4fz1FNPUbp0aVuR0Lt3b/773/8%2BsHxEpGBTMSEiIg/Vyy%2B/zOrVqzl9%2BjTHjx%2B3mxcfH3/bdpnzMkcoSpQoAWRcphMTE2N3g/Gd7Nixg/Pnz2OxWFiyZAmVK1fOtkx%2B%2BTK9Q4cOERYWBsCsWbNo3LhxtmXyS64iUjDoaU4iImK6LVu28NZbb7FkyZJcLZ95mVDmdfuZfv/99xyXT0tLs40UZN4DkPX%2BiluLkjvJfBqUt7d3joXE6dOnuXz5cq7j3U%2BnTp0CsH2fxa0SEhLu%2BGV%2BIiJmUzEhIiKmCwsLIyQkhCVLlnD%2B/Pk7LhsZGUl4eDhAticenTlzhiNHjmRr88svv9i%2BOdrPzw/IuNm7du3aAPz73//O8b1OnjxJUFCQ7RGy8OfIxo0bN3Jss2DBAtvPt37Z3YPm5uYGZFzildON0h988IFtZCKnG7dFRMymYkJEREz397//HU9PT%2BLi4nj55ZfZs2dPtmWsViu7d%2B%2Bmf//%2BpKSkUL9%2BfZo2bWq3TLFixZgwYYLdyEBsbCxvv/02kPFI1vLly9vmZX5p3KZNm1ixYoXtaUyQcVZ/xIgRhIeHc%2BHCBdv0GjVqAHDx4kVCQ0Nt0xMTE5k2bRphYWG2giUiIiLPfWKGxx57DEdHR9LS0lizZo1tempqKu%2B//z4hISG0bNkS%2BPOpTiIi95PumRAREdOVLFmSDz/8kOHDh/PHH3/w0ksv4enpySOPPEKRIkWIjY3l7Nmztu9CeOKJJ5g/f362G64DAwM5c%2BYMrVq1wtfXFycnJ3799Vfi4%2BNxc3Pjrbfeslv%2Bb3/7GwMHDmTJkiXMmDGDjz76iEceeYSYmBiOHDlCWloatWrV4tVXX7W18fPzo1mzZvz4448MHz6cevXq4eLiwsGDBylUqBDLly9n8%2BbN7N%2B/n48%2B%2BoijR48yePBgGjZseP878hZly5alS5cuhISEMGPGDDZu3EiJEiU4cuQICQkJzJ8/n7Nnz/Ldd9/xzTff0Lt3b3r27MkzzzzzwHMVkYJBxYSIiNwXjz32GF9%2B%2BSWbN29m27ZthIeH89tvv5GcnIyrqytlypShcePGtGvXjhYtWuT4RCQHBwc%2B/PBDFi5cyLfffsv58%2BcpWrQobdq0YfTo0XbfE5Fp7NixNGnShNWrV3PgwAF27dqFi4sLderUoV27dvTs2TPb41HnzJnDrFmz2Lp1KwcPHsTLy4s2bdowaNAgvL29KVu2LIcPH2bfvn0cP378gT696VYTJkzA3d2dTZs2cezYMUqVKoW/vz%2BvvPIKNWvWJCkpiT179rBjxw6OHz9uNzojImI2i1V/ZUREJJ8ZP348X3zxBZ07d7Zd0iQiIvmP7pkQEREREZE8UTEhIiIiIiJ5omJCRERERETyRMWEiIiIiEg%2B8sMPP9CkSRNGjx59x%2BXS09OZPXs2rVq1omHDhvTr148zZ87Y5sfGxjJq1CiaNGlCs2bN%2BL//%2Bz/bd9GYRTdgi4iIiIjkE0uXLmXdunV4eHhQtmxZZs%2BefdtlV61axfLly1m6dCllypRh9uzZ7N27lw0bNmCxWBg%2BfDjJycnMmDGDlJQURo4cSZ06dZgwYYJp%2BWpkQkREREQkn3BxcWHdunVUrlz5rsuGhITQp08fqlWrhpubG6NHj%2BbEiRP8%2BuuvXLp0idDQUEaPHo2HhwdlypRhyJAhfPbZZ6SkpJiWr75nQkRERETEJFFRUbYv5Mzk6emJl5dXrtq/9NJLuVouKSmJ33//nVq1atmmubm5UblyZcLCwoiLi8PR0REfHx/b/Nq1a3P9%2BnUiIiLsphuhYkKkIDH6RVtVq8Lx4/Doo3DypLFYBw4Yaw/g7Aw1akB4OCQnG4tVo4Y5%2BRjNA7A6uxjPxSSWK5eNBXBwgBIlIDYW0tONJ1SkiLH2Fgu4uMCNG2D0Kt/UVOO5FC0KCQmGc0lxLWYsF6BQIeOrlMnp%2BlVjASwWKFYM4uKM/55u%2BYLCPOVi0jZjdS1sLBeTWc6fy3tjR0coUwYiIyEtzXgyFSoYj5EX9%2BELKEPmzmX%2B/Pl204YNG8bw4cNNfZ%2BrV69itVpxd3e3m%2B7u7k5MTAwlSpTAzc3N7ks2M5eNiYkxLQ8VEyKSeyVKZOxASpR42JlkcHTM2BE4Oj7sTDLyyPyXD25Fyydp2PdLfpFfcsmH20y%2Bkc/6Jj91Tn7pEhwcMpJxcDCnmPgL6datG4GBgXbTPD0979v73en25wdxa7SKCREREREpmBzMv33Yy8sr15c0GVGiRAkcHByIjY21mx4bG0upUqXw8PAgPj6etLQ0HG%2BedMtctlSpUqbloRuwRURERET%2Bx7i4uPDoo49y6NAh27Rr165x%2BvRp6tatS82aNbFarYSHh9vmh4WFUbx4capWrWpaHiomRERERKRgcnAw/999FBkZSdu2bW3fJdGjRw8%2B%2BugjTpw4QXx8PLNmzaJmzZr4%2Bvri4eFBmzZtmDNnDleuXOHixYssWLCALl26UKiQeRcn6TInERERESmY7vPBf174%2BvoCkHrziQihoaFAxqhCSkoKJ0%2BeJPnmwz66d%2B9OdHQ0vXv3JiEhAX9/f7ubv6dMmcLEiRNp1aoVTk5OPPvss3f9Irx7pWJCRERERCSfCAsLu%2B28ihUrcvToUdv/LRYLI0aMYMSIETkuX6xYMf71r3%2BZnmNWKiZEREREpGDKhyMT/2vUgyIiIiIikicamRARERGRgkkjE4apmBARERGRgknFhGHqQRERERERyRMVEyL/A2bNmkXv3r0fdhoiIiJ/Lf9j3zORHxW8NRbJp3r37s2sWbMedhoiIiIiuaZ7JkRERESkYCqAIwlmUzEhkg9t27aNmTNnEhUVRfPmzSlduvTDTklEROSvR8WEYepBkXzm2rVrjB49mhdffJE9e/bQuXNn1q9f/7DTEhEREclGIxMi%2BcyPP/5IkSJF6NWrFw4ODjRv3pwGDRqQkJBwT3GioqKIjo62m%2BZZtSpeJUrkPbkaNexfjShc2HgMFxf7VyMsFnPaG42T3zg6mtPeaJxM%2Ben3ZPSMZmb7v%2BKZ0fzUN/lpm8lvnJzy3rZQIftXI1JSjMfIq7/i5%2B8BUzEhks9cvHiRcuXK4ZDlD1yVKlU4dOjQPcUJCQlh/vz5dtOGjRzJ8JEjjSe5Zo3xGGaqXPlhZ/AnIzvnm8w6ZDHl2MdI8ZlVsWLmxDGLGQWoWUworI1vdTfjmBbIpN930aLmxDGDCdtMvvpsA3h5GY/h4WE8xrlzxmPIQ6NiQiSfSU5OJi0tzW5aenr6Pcfp1q0bgYGBdtM8O3SAlSvznlyNGhmFRM%2BeEB6e9zgAn3xirD1k7NwrV4ZTp%2BDGDWOxqlQx1t5iyTgSS0kBq9VQKKuTs7FcbqZjMI2MOFdjjQVwdMwoJOLi4JbtOk9cXY21t1gytpsbN4x3kNH1cXDIKCQSEyEPn/GsUpyNH3Rnbr5mcEqKMxbAwSGjkEhIMNw3hiskE7cZq4vB7RfzPtsAluiovDcuVCijkLhyBVJTzUnoYdDIhGEqJkTyGS8vLyIjI7FarVhunn46ceJEnuJ43XrW6eRJM1LMKCT27zcWIzHRnFwgYydvNJ5Ze2er1bxY%2BYEZBUBmHDNi5affk9GD3KxxzIqVX%2BSnvslP20x%2BY0b1mJr6cC9TMkrFhGHqQZF8pkmTJsTHx/PJJ5%2BQnJxMaGgov/7668NOS0RERCQbFRMi%2BUzZsmUJDg7mww8/pFGjRmzcuJGePXs%2B7LRERET%2BevQN2IbpMieRfGLVqlW2n9u0aUObNm0eYjYiIiIid6diQkREREQKpgI4kmA2FRMiIiIiUjCpmDBMPSgiIiIiInmikQkRERERKZg0MmGYelBERERERPJEIxMiIiIiUjBpZMIwFRMiIiIiUjCpmDBMPSgiIiIiInmikQkRERERKZg0MmGYelBERERERPJEIxMiBcmBA8baFy6c8frJJ5CYaCxW/frG2gP4%2BcG%2BfdC9O%2BzfbyhUWqrVcDqOQJqjs%2BE4mzYaa%2B/uDi1awPffw9WrxmIFXTGYTKlS0LEj7NgBly8biwXQrp2x9oUKgasrJCRAaqqxWG%2B9Zax9pUowYQLMng1nzhgK5TR3rrFcLBbAGSdrMliNfxY4dsxY%2ByJFoHZtOH0arl83Fsvb21h7E7cZy/TpxnIpVw6GDMGyaCFcuGAsFsCUKcZjeHoaj/EwaWTCMBUTIiIiIlIwqZgwTD0oIiIiIiJ5opEJERERESmYNDJhmIoJERERESmYVEwYph4UEREREZE80ciEiIiIiBRMGpkwTD0oIiIiIiJ5opEJERERESmYNDJhmIoJERERESmYVEwYph4UEREREZE80ciEiIiIiBRMGpkwTD0oIiIiIiJ5opEJERERESmYNDJhmHpQCpzAwEDWrl37QN9z/fr1BAYGPtD3FBERkbtwcDD/XwGjkQn5ywkMDCQyMhKHHD7QM2bMeGB5rFu3jsDAQDw8POjUqROdOnV6YO8tIiIi/5vOnTvH5MmT%2BfXXXylSpAjt2rVj7Nix2Y5r%2Bvbty969e%2B2mpaamMnToUIYNG0bv3r3Zt2%2BfXbuqVauyceNGU/NVMSF/SRMmTKBHjx45zvvXv/51398/LS2Nt99%2BGz8/Pzw8PO77%2B4mIiEge5MORhOHDh1O7dm1CQ0O5fPkyr7zyCqVLl%2Bbvf/%2B73XIffvih3f%2BvXbtGu3btaN26tW3a1KlTee655%2B5rvvmvB0UeoPT0dObOncvTTz9NvXr1eP755/nll18AGDVqFG%2B88Ybd8itWrOCZZ54B4PTp0/Tr1w9/f3/8/f0ZM2YM165dA6BRo0bExcURFBTE/Pnz%2Bfzzz2natKktzvHjx3nppZdo0KAB/v7%2BTJw4kRs3bgDw%2Beef07FjR9ulUX5%2BfowePZqUlJQH0SUiIiLykISFhREeHs6rr75KsWLFqFKlCn369CEkJOSubefMmUPr1q3x8fF5AJn%2BSSMTUqCtXLmSzZs3s2zZMsqXL09ISAiDBw9m%2B/bttG3blkmTJpGWloajoyMAW7ZsoV27dkDG6EeFChX44YcfiI%2BPp1%2B/fixcuJDx48ezYcMGWrVqxYYNG6hWrRqff/657T2Tk5Pp27cvnTp1YsmSJURFRTFo0CDee%2B89Xn/9dSBjiPPgwYNs2rSJc%2BfO8dxzz9m9d25ERUURHR1tN80zPh4vT8%2B8d5iLi/2rEX5%2BxmPUqGH/%2Bhfh7m6svZub/ashllLG2meujNGVylTI4G7r5mfZ9mpEpUrG2pcta/9qhMViTnujcTIVKWKsvaur/asR%2BWmbKVfOWPvSpe1fxbj7MDKR4/7X0xMvL6%2B7tj106BAVKlTAPcvfzNq1a3Py5Eni4%2BNxu80f9lOnTrF%2B/XpCQ0Ptpn/11VcsW7aMCxcuUK9ePaZMmYK3t3ce1ur2VEzIX9K0adOYPn263bQiRYqwZ88eu2nr1q2jT58%2BVKlSBYDevXuzcuVKtm/fTmBgIDdu3OCXX36hUaNGXL58mX379jFlyhQAlixZgsViwdnZGQ8PD5588kn27dt319x27NhBYmIiw4cPx9nZGW9vb3r16sWyZctsxURCQgKjRo2iSJEiPProo/j4%2BBAREXFPfRASEsL8%2BfPtpg0bOpThI0bcU5wcVa5sPEYu%2BirX1qwxHMKEw4SMOCYEatHCeAyABg3MiNLRjCDQvLk5ccxSooTxGBMmGI8B0L%2B/OXHM4ORkTpzatc2JU62aOXHMYMY2M2SI8RgAL7xgThwzmFGAWq3GY%2BTVfSgmctz/DhvG8OHD79o2NjaW4sWL203LLCxiYmJuW0wsWbKE559/3u7S6mrVqlG4cGFmzZpFeno606ZNo3///mzatAlnZ%2Bd7Xa3bUjEhf0l3umciq9OnT/PPf/7TrvBIT0/nwoULuLq60rx5c0JDQ2nUqBHbtm3j0UcfpdrNndvBgwcJDg7m6NGjpKSkkJaWRp06de76nmfPnqVSpUp2H%2BTKlStz/vx50tPTAShZsqTdH4zChQuTlJSU6/UH6NatW7YnSHnGxsKxY/cUx46LS0YhceoU3LwsK8%2B6dzfWHjJGJNasgZ49ITzcUKi0vcaLG0dHSEszHIYffjDW3s0to5D4%2BWeIjzcWq8U1gzfqubtnFBLffw9XrxqLBZDlcsE8cXTMOCiMjTX%2By1q0yFj7smUzColly%2BDiRWOxbp6IyDOLJaOQSEkx58Du%2BHFj7V1dMwqJEyfgHv/2ZWN05MfMbSYXl6rcUenSGYXEv/8Nly4ZiwUweLCx9hbLwy0E8qkc97/3cFWA9R77NDY2lg0bNvD111/bTZ80aZLd/6dMmYK/vz%2B//PILjRs3vqf3uBMVE1Kgubq6Mm3aNNq0aZPj/GeeeYZ33nmHN998k2%2B//dZ2mdHVq1cZOHAgPXr0YOnSpbi5uTFnzhx%2B%2Bumnu75ncnJyjtMtWc7u5PQkqnvl5eWVfUj1118hMdFwbG7cMB5n/37jeWQKDzc33kNmxjE3ZBQShmNduWxKLly9CpdNiJWaajwGZBwUGo115ow5uVy8aDyWWQd0Vqs5sa5fNx4DMgoJo7Hy0zZz4YI5uVy6ZF6sgu4%2BjEzkuP/NJQ8PD2JjY%2B2mxcbGYrFYbvtAl61bt1K1alUq3eXSSzc3N9zd3YmMjMxTbrejG7ClQKtUqRJHjx61m3b27Fnbz82bN%2BfKlSvs27eP3bt324qJiIgIEhIS6Nevn20E4fDhw7l%2BzzNnztgVFREREVSsWNGUIkJERET%2BN9WpU4cLFy5w5coV27SwsDCqV69O0aJFc2yzdetWu4e8AMTHxzNp0iS7wuHKlStcuXLlrkXHvdKRixRo3bt35%2BOPP%2BbAgQOkpaXx1Vdf8eyzz3L%2B/HkgY%2BSiRYsWBAcH89hjj9luWipfvjwODg7s37%2Bf69evs2LFCi5dusSlS5dITU3F9eZNg3/88Qfxt1xn8tRTT1GoUCEWLFhAcnIyERERfPTRR/oeChERkQctn31pXa1atfD19SU4OJj4%2BHhOnDjB8uXLbZdut23blp9//tmuzZEjR6hYsaLdNDc3N3799VemTZtGbGwsV69eZfLkyfj4%2BOBnxgNQslAxIX9J06ZNw9fXN9u/Wx/12qVLF3r27MmwYcN44oknWLZsGfPnz6d8%2BfK2ZTI/uO3bt7dNK1OmDGPGjOHNN9%2BkZcuWXL16lVmzZpGcnEzPnj0pXbo0bdq0YeTIkcyZM8fuPYsWLcqSJUvYu3cvjRs3ZsCAAQQFBTFo0KD72ykiIiJiL58VEwBz584lKiqKpk2b8tJLL9GpUyd69uwJwMmTJ7l%2By6V/0dHRlM7hCV8LFizAarXSpk0bWrRoQUpKCkuWLDH9KgjdMyF/Odu2bcv1sg4ODowcOZKRI0fedpk2bdpkuxQKYMCAAQwYMMBu2o8//mj7ee7cuXbzsn5pTN26dVlzmycQPffcc9m%2BYGbVqlW3XwkRERH5yyhbtixLly7NcV5OxyMHDx7Mcdny5ctne6rU/aBiQkREREQKJt2raJh6UERERERE8kQjEyIiIiJSMGlkwjAVEyIiIiJSMKmYMEw9KCIiIiIieaKRCREREREpmDQyYZh6UERERERE8kQjEyIiIiJSMGlkwjAVEyIiIiJSMKmYMEw9KCIiIiIieWKxWq3Wh52EiDwgN24Ya2%2BxgLMzJCeDwT8daYVcjOVyk6MjpKWZEKeQxVgAPz/Ytw8efxz27zcWKz7eWHsHByhcGBITIT3deCwjLBZwdYWkJMPbDADXrxtr7%2BgIJUpAbKw5G04%2ByWX%2B2lKG2nt6QrduEBIC0dGGQgEwzOl9YwFKl4bnn4fPPoNLl4zF%2BuknY%2B0rV4YpU%2BCtt%2BDUKUOhrs5daai9gwMUKwZxccY/2gDuxQ1%2BJi0Wcz7XmbEehq5dzY/56afmx8zHNDIhIiIiIiJ5onsmRERERKRg0j0ThqmYEBEREZGCScWEYepBERERERHJE41MiIiIiEjBpJEJw9SDIiIiIiKSJxqZEBEREZGCSSMThqmYEBEREZGCScWEYepBERERERHJE41MiIiIiEjBpJEJw1RMiIiIiEjBpGLCMPWgiIiIiIjkiUYmK3N%2BAAAgAElEQVQmRERERKRg0siEYepBkRz8%2BOOP%2BPj4MHny5IedCmlpaSxfvvxhpyEiIiKSjYoJkRx8%2BumntG/fns2bN3Pjxo2Hmsvhw4dZtmzZQ81BRETkL8nBwfx/BUzBW2ORu4iJiWHbtm2MGDGCkiVLsmXLFtu87du306FDB/z8/GjWrBnvvvsu6enpAPj4%2BPD555/TpUsX6tatS6dOnYiIiLC1DQ8P5%2BWXX6ZBgwYEBAQwbdo0UlJSbPM3bNhAmzZt8PPzo3v37hw5coTffvuN7t27c%2BnSJXx9fdm9e/eD6wgREZG/OhUThumeCZFbbNiwgZo1a1KlShU6dOjAunXrePbZZ0lJSWH06NEsWLCAxo0bc%2BrUKfr374%2Bfnx9PP/00AMuXL2fOnDmULVuWiRMnMmbMGNavX09iYiL9%2B/end%2B/eLF26lMjISIYMGcIHH3zAoEGDOHjwIJMmTWLRokU88cQTvP/%2B%2BwwZMoTQ0FCmTp1KcHAwO3fuvKf1iIqKIjo62m6ap7s7Xl5eee8ci8X%2B9a/Ez89Y%2Bxo17F%2BNMLozyvp7MiuWGbmYwdHRnPZG45jBxFw8PY21L1nS/tWwQqWNtS9Rwv7ViMqVjbUvV87%2B1QCjH8fM9gXweFXyMRUTIrdYt24dPXr0ACAoKIgFCxZw9uxZSpQoQVJSEkWKFMFisVClShW%2B/fZbHLL8VQ8KCqJatWoA9O/fn6CgICIjI9m3bx9Wq5VXXnkFgEqVKtGvXz/ef/99Bg0axPr16wkICCAgIACAfv36UbVqVUOXWIWEhDB//ny7acOGDmX4iBF5jmnj5GQ4hJmHcqYcF%2B7bZ0IQYM0ac%2BKYwdX1YWfwJxcXc%2BKYtU7FipkTxwwm5NKtmwl5AH/7mzlx4HlzwrRqZTzG8yblMniw4RBmbXVFi5oUCBOKfDNOFFitxmPklSozw1RMiGRx4MAB/vjjD5555hkg46C/fv36fP7554wYMYKhQ4fy4osvUrduXZo2bcpzzz1HuSxnq6pWrWr7uUKFCgBERkZy5swZLl%2B%2BjK%2Bvr22%2B1WrF2dkZgDNnzuDt7W2bV7hwYdq3b29oXbp160ZgYKDdNE93d0hOzntQiyWjkEhJMfzHP83R2VD7TI6OkJZmQpyGjxsLUKNGRiHRsyeEhxuLdY%2BjUNlYLBkH3UlJxnfSZoxMuLjAjRvmHDAkJRlr7%2BiYcfAeF2fOhpNPcgn5xtgZ/JIlMwqJb7%2BFmBhDoQDoVugzYwFKlMgoJLZuhdhYY7F%2B/dVY%2B3LlMgqJRYvgwgVDoeJem2KovYNDRiGRkAA3r7A1pJibCX8fHmYhIPmCigmRLD799FNSU1NpleVsWEpKCpGRkQwbNoxhw4bRtWtXQkNDCQ0NZdmyZaxcuZK6desC2O6fgIxiAcBiseDi4sKjjz7Kl19%2BmeP7WiwW2/Jm8fLyyn5Jk1kHdFbrX28Hsn%2B/OXHCw43HMnqUkHmmzWo1L5ZRZm0zZhUAaWkPv5jIZEIut1zRmGcxMSbFcrpkQhAyColLBmOdOmVOLhcuGI5lRgGQGcesWAWeRiYMUzEhclNCQgJfffUVkydPtl1uBJCYmEiXLl3YtWsXtWvXpkyZMvTq1YtevXrxxhtvsGHDBlsxcfr0aVu78%2BfPA1C2bFm8vb05c%2BYMCQkJFL05Ph0TE4OTkxNubm5UqlTJ7mbt5ORkVq1axXPPPfcgVl1ERKRgUjFhmHpQ5KavvvoKFxcXOnfuTOXKlW3/atSoQWBgIB9%2B%2BCHPPPMMv/32G1arlcuXL3Py5Em7y5M2bNjAqVOnSEhIYOnSpdSpUwdPT0%2BaNWuGh4cHM2fOJD4%2BnujoaEaOHMmsWbMAeO6559izZw/fffcdKSkprFixgo8%2B%2Bgg3NzdcXV2Ji4sjMjKSJKOXeIiIiIiYSMWEyE2fffYZHTp0sN3HkNXzzz/Pf//7X3r16sWoUaOoV68enTt3pl69evTq1cu2XJcuXRg7diyNGzfm999/Jzg4GAAnJycWLlxIREQETZs2pVOnTlSpUoVx48YBULNmTWbNmsXUqVNp2LAh27ZtY9GiRTg5OREQEEDFihV5%2Bumn2bZt24PpDBERkYJAj4Y1TJc5idz0ySef3HZe8%2BbNCQsLA2DYsGG3Xa5KlSqsW7cux3k1atRg9erVt23btm1b2rZtm226h4cHX3311W3biYiIiDwsKiZEREREpGAqgCMJZlMxISIiIiIFk4oJw1RMiJjk6NGjDzsFERERkQdKxYSIiIiIFEwamTBMPSgiIiIiInmikQkRERERKZg0MmGYigkRERERKZhUTBimHhQRERERySfOnTvHwIED8ff3p2XLlrz77rukp6dnW27evHnUrFkTX19fu3%2BXLl0C4MaNG7z11ls89dRT%2BPv7M2LECGJiYkzPV8WEiIiIiBRM%2BfAbsIcPH06ZMmUIDQ1l%2BfLlhIaGsnLlyhyXDQoKIiwszO5f6dKlAZg9ezaHDh0iJCSEb775BqvVyhtvvGE4v1upmBARERERyQfCwsIIDw/n1VdfpVixYlSpUoU%2BffoQEhJyT3FSU1NZt24dQ4YMoVy5cpQoUYJRo0axfft2IiMjTc1Z90yIFCBWZxfDMSyA1cnZcJxNGw2HwN0dWrSAH36Aq1eNxQqKjzcWIPNs1M6dkMNw9D1xczPW3s8P9u2Dpk1h/35jsZKSjLW3WDJeHRzAajUWC2DbNmPtS5SA1q1h716IjTUU6pviXQ21L1YMmjSBnw6XIC7OUCgGDzbWPlOXLubEGffmK4bal0%2BCkcB7Z5/n/HljubTobiyX4sWhKbCzzRSuXTOWi9tvxtoXLQqPPw7Hj0NCgrFYAE9Wv5j3xoUKgacnXLoEqanGkylXzniMvLgP90xERUURHR1tN83T0xMvL6%2B7tj106BAVKlTA3d3dNq127dqcPHmS%2BPh43G7ZPxw9epTu3btz7NgxypUrxxtvvEGzZs04ffo0cXFx1K5d27ZstWrVcHV15dChQ5QpU8bgWv5JxYSIiIiIFEz3oZgICQlh/vz5dtOGDRvG8OHD79o2NjaW4sWL203LLCxiYmLsiomyZctSqVIlxo4di5eXFyEhIQwaNIiNGzcSe/Nkya2xihcvbvp9EyomRERERERM0q1bNwIDA%2B2meXp65rq9NZejuF27dqVr1z9HSPv06cPmzZvZuHEjTz311D3FMkLFhIiIiIgUTPdhZMLLyytXlzTlxMPDwzaqkCk2NhaLxYKHh8dd21eoUIGoqCjbsrGxsRQtWtQ2/%2BrVq5QqVSpPud2ObsAWEREREckH6tSpw4ULF7hy5YptWlhYGNWrV7crCgAWLlzIrl277KadOHGCSpUqUalSJdzd3Tl06JBt3rFjx0hOTqZOnTqm5qxiQkREREQKpnz2aNhatWrh6%2BtLcHAw8fHxnDhxguXLl9OjRw8A2rZty88//wxkjDpMnjyZiIgIbty4wYcffsjp06fp3Lkzjo6OvPDCCyxevJgLFy4QExPDv/71L1q3bm17dKxZdJmTiIiIiBRM%2BfAbsOfOncs//vEPmjZtipubG927d6dnz54AnDx5kuvXrwMwduxYIONeidjYWKpXr86KFSsoW7YsACNGjCAhIYGgoCBSU1Np2bIlkyZNMj1fFRMiIiIiIvlE2bJlWbp0aY7zjh49avvZxcWFN998kzfffDPHZZ2dnZk4cSITJ068L3lmUjEhIiIiIgVTPhyZ%2BF%2BjHhQRERERkTzRyISIiIiIFEwamTBMxYSIiIiIFEwqJgxTD4qIiIiISJ6omBAxSWBgIGvXrn3YaYiIiEhu5bPvmfhfpMucRHIpJSWFRYsWsXnzZiIjI7FYLNSpU4eRI0fSoEGDh52eiIiIyAOnYkIkl95%2B%2B21%2B%2BeUX5s6dS/Xq1UlMTGTVqlX07duXzZs3P%2Bz0RERE5F4VwJEEs6kHRXJp586dtG/fHh8fHxwdHXFzc2Pw4MFMmzYNZ2dnu2XT09NZsGABrVu3pm7dunTu3Jldu3bZ5vv4%2BPD555/TpUsX6tatS6dOnYiIiLDNDw8P5%2BWXX6ZBgwYEBAQwbdo0UlJSHti6ioiIFAi6zMkwjUyI5FLVqlX54osvaNasGTVr1rRN79ixY7ZlP/74Yz799FPef/99qlatyurVqxkyZAihoaGUKlUKgOXLlzNnzhzKli3LxIkTGTNmDOvXrycxMZH%2B/fvTu3dvli5dSmRkJEOGDOGDDz5g0KBBuc43KiqK6Ohou2mlS3vi5eWVxx4wl7u78RhubvavhhjdAVgsf74ajeXnZ6x9jRr2r0ZkrpfR9kbjZCpRwlj7YsXsX42EKmqsfdGi9q9/JeXLG2vv6Wn/akTx4sbam/l7KlzYnPZG49gUMnAYmNnWSIxMqanGY8hDY7FardaHnYTI/4Lz588zZswY9u/fT4UKFXjiiSdo3rw5f/vb33B2diYwMJABAwbQo0cPgoKCaN%2B%2BPQMHDrS1b9KkCWPHjuX555/Hx8eH1157jf79%2BwMZIxFBQUHs2LGDffv2MW3aNHbu3Glru379et5//32%2B/vrrXOc7b9485s%2Bfbzdt6NBhjBgx3GBPiIiImOjCBShX7uG894IF5sccOtT8mPmYRiZEcql8%2BfJ88skn/P777/z000/s3buXCRMm8N5777F69Wq7Zc%2BePUu1atXspnl7e3Pu3Dnb/6tWrWr7uUKFCgBERkZy5swZLl%2B%2BjK%2Bvr22%2B1WrNdinV3XTr1o3AwEC7aaVLe2L09IHFguEYAN9/bzyGmxs0aAA//wzx8cZitfBPNBbAYgFXV0hKMt5BTZsaa1%2BjBqxZAz17Qni4sVi7dxtrb7GAkxOkpOSPDadYMQgIyFivuDhDoX4q2tpQ%2B6JFoV49%2BPVXSEgwFAp/f2PtARwdIS3NeByAW85j3DNPz4zNd80auGWA9Z4ZfT5G0aJQvz4cOGD892TGyETNmnDkCCQa/JMF8HglA51bqBCULAkxMRpZKOBUTIjco%2BrVq1O9enVeeukloqOj6dq1KytXrrRbJjk5Oce2liyXeqSnp9t%2BzhwgtFgsuLi48Oijj/Lll18aytPLyyvbJU35aRzy6lXzYsXHmxAvy%2B8jTzIvbbJajcfav99Y%2B0zh4cZjmbXRWK3mxIqNNR4DMgoJg7HiDP6aMyUkGK5r8p3z582JEx1tPNa1a%2BbkkpBgPJbRPw2ZEhONFzaAOUVAaur/djFRAO9xMJt6UCQXLl68yKRJk4i/5fS3p6cnNWrUIPGWU0Te3t52N1SnpqZy6tQpKlWqZJt2%2BvRp28/nb%2B4ty5Yti7e3N2fOnCEhy54iJiYm23uLiIiIQboB27CCt8YieeDh4cFPP/3Ea6%2B9RkREBOnp6SQmJrJp0yZ27dqV7XKioKAg1qxZw4kTJ0hOTmbx4sWkpaXZLbdhwwZOnTpFQkICS5cupU6dOnh6etKsWTM8PDyYOXMm8fHxREdHM3LkSGbNmvWgV1tERETkjnSZk0guODs7s2rVKubNm0e/fv24cuUKDg4O1KxZk%2BDgYJ588km75fv27UtMTAwDBgzg2rVr1KxZk48%2B%2BojiWR4r0qVLF8aOHcuxY8eoWrUq7733HgBOTk4sXLiQadOm0bRpU9zc3GjVqhXjxo17oOssIiLyl1cARxLMpmJCJJfKlCnDtGnTbjt/27Zttp8LFSrEuHHj7lgAVKlShXXr1uU4r0aNGtlu6hYRERGTqZgwTD0oIiIiIiJ5opEJERERESmYNDJhmIoJkYfg6NGjDzsFEREREcNUTIiIiIhIwaSRCcNUTIiIiIhIwaRiwjD1oIiIiIiI5IlGJkRERESkYNLIhGHqQRERERERyRONTIiIiIhIwaSRCcNUTIiIiIhIwaRiwjAVEyIFiOXKZWMBHB2hRAksV2MhLc1QqKArG43lAmApBXSkxbWNYHTdHLobzMXy56vRnVNSkjm57N4NVquxWK6uxtr7%2BcG%2BfRAQAPv3G4sF8OyzxtpXqwatW8PmzXDihKFQbeY8biwXZ2egEk0qnYHkZEOh4hOrGWrv4ABFisCNG5CebigUADMnJRoLYLEArox8Jcn4Npyaaqy9gwNQlKb1E4x3TkSEsfaFCwOP8bjbMXA02MdAnFu9PLd1cICiQEIRT1O2mWLGQ8hDomJCRERERAomjUwYph4UEREREZE80ciEiIiIiBRMGpkwTMWEiIiIiBRMKiYMUw%2BKiIiIiEieaGRCRERERAomjUwYph4UEREREZE80ciEiIiIiBRMGpkwTMWEiIiIiBRMKiYMUw%2BKiIiIiEieaGRCRERERAomjUwYph4UEREREZE80ciEiIiIiBRMGpkwLN/34IQJE3j99dcfdhp2Pv/8c5o2bWpqzIULF/Liiy%2BaGlPunb%2B/Pxs2bMhT208//ZSnnnrK5IxERETkvnFwMP9fAWPqGgcGBrJ27dps09euXUtgYGCeYk6bNo133nnHaGp3tWvXLsLCwkyPu2fPHnx8fPD19cXX15f69evTqVMnFi9eTFJSkm25IUOGsHr16rvGS0tLY/ny5abnaaa9e/fa1tfX1xcfHx/q1Klj%2B/%2BECRMMv8edDtyNFAQiIiIiD9O5c%2BcYOHAg/v7%2BtGzZknfffZf09PQcl127di1t2rTBz8%2BPoKAgQkNDbfPGjx9PrVq17I7JGjRoYHq%2BuszpphUrVtCiRQt8fX3vS/yff/4ZFxcXrl27xqFDh5g9ezbffPMNq1evpmjRormOc/jwYZYtW8bf//73%2B5KnGRo2bGhXmPn4%2BLBw4UKdtRcREZH8JR%2BOJAwfPpzatWsTGhrK5cuXeeWVVyhdunS2Y79vvvmG4OBg3n//ferWrcv69esZNWoUX3/9NZUqVQJg8ODBDB8%2B/L7m%2B8B78OLFiwwePBh/f3%2BeeOIJRo8eTWxsLJBxFt/Pz48VK1bw%2BOOPs3//fsaPH8/o0aMB6Nu3r111VatWLbsRj9DQUDp27Ej9%2BvUJDAzko48%2Bss0bP348U6dOZcaMGTRq1IiAgACWLl0KwKBBg9i%2BfTvTpk3j5ZdfBiAsLIyePXvSoEEDmjRpwsSJE0lJSTG8/sWLF6dx48asWLGC%2BPh4PvjgAwDmzZvHCy%2B8AEBiYiLjxo2jcePG%2BPn50b17dw4ePMhvv/1G9%2B7duXTpEr6%2BvuzevRur1cqsWbNo3rw5fn5%2BdO7cmb1799rer3fv3ixevJjXXnuNxx9/nCeffNLurP2ZM2fo27cvfn5%2BtGzZ0q7Pzp07x6BBg/D396dhw4a8/vrrxMfHG%2B6DTOfPn7fFb9CgAWPHjuXq1aumxd%2B9eze%2Bvr5cu3bNNu369evUq1ePXbt2kZKSwqRJk/D39%2Bepp57is88%2Bs2vfo0cPgoOD6dChA4MHD77nnI8ePUrv3r1p0KABAQEBTJ48meTkZNv8efPm2bavVatW0bt3bxYuXMhnn31Gs2bN7M5CnDlzhho1anDmzBnT%2BkdERETyl7CwMMLDw3n11VcpVqwYVapUoU%2BfPoSEhGRbNikpiTFjxvDEE0/g5ORE165dKVq0KAcOHHigOT/wkYkhQ4ZQvXp1tm7dSlJSEiNHjmTixIm89957AKSkpHDq1Cl%2B%2BuknXFxc7Drvww8/tP0cHx/Pc889R5cuXQAIDw9n5MiRvPfeezRv3pyff/6ZQYMGUblyZZo3bw7Apk2bGD9%2BPDt37mTjxo384x//ICgoiMWLFxMYGMiAAQPo0aMHAKNHj6Zjx46sWrWKyMhIunfvTvXq1endu7cp/VCkSBFeeOEFvvjiC0aMGGE3b%2BXKlVy6dIktW7bg7OzM0qVL%2Bcc//sEXX3zB1KlTCQ4OZufOnQCsX7%2Be9evXs27dOjw9PVm0aBEjRozgxx9/xNHREYCPP/6Y6dOnM336dBYvXsyUKVNo164dTk5ODBs2jEaNGrFgwQL%2B%2BOMPevXqRbVq1WjSpAlDhgzh8ccfZ/bs2Vy/fp0xY8Ywc%2BZMpk6danj9rVYrgwcPplatWmzdupXExESGDx/OlClTCA4ONhwfMi53Kl26NN988w1du3YFYMeOHbi7u%2BPv78/atWvZunUra9eupUyZMrz99tskJCTYxdi8eTNz586ldu3a95RzUlIS/fr1o0uXLixdupTIyEgGDhzI/PnzGTNmDF9//TXLli1j%2BfLl1KhRg2nTphEeHk7jxo1p06YNU6dOZffu3TRp0gSAb7/9lvr169vONORGVFQU0dHRdtM8nZ3x8vTMS3dmuLlN2V6NKFXKeAx3d/tXIywWc9objZPfcvHzM9a%2BRg37V6OqVTPWvmJF%2B1cjnJ2NtXdysn81wOjJ1aybjCknavPTNmx0hTLbm9ExhQsba%2B/iYv9qkJFVMrNbbnMFz4NxH0Ymctz/enri5eV117aHDh2iQoUKuGfZr9WuXZuTJ08SHx%2BPm5ubbXpQUJBd22vXrpGQkECZMmVs03bv3s3WrVs5deoU1apVY9KkSdSpUyevq5Yj04uJadOmMX36dLtp6enplClThiNHjnDo0CHef/993NzccHNzY%2BDAgQwdOtR2xjYlJYWePXvi6up6x/eZOHEi3t7eDBgwAIDPPvuMxo0b8/TTTwPQuHFjWrRowVdffWUrJipWrEjnzp0BaNeuHW%2B%2B%2BSZ//PFHjr/c9evX4%2BzsjKOjI%2BXLl6dhw4YcPHjQWOfcomrVqpw9ezbb9GvXruHk5ISrqyuFChViyJAhDBkyJMcYHTp0oFWrVhQrVgyA9u3bM2/ePM6fP2878PTz8%2BPJJ58E4JlnnmH%2B/PlERUVx9epVjh49ysqVKylcuDA1a9Zk/vz5lClThrCwMI4fP87atWspXLgwhQsXZvjw4fTr148pU6ZgMfgH/uDBgxw9epTly5fbtoUBAwYwatQoUlJScMrFzjUyMjLHy9IytyWLxULHjh3ZuHGjrZjYsmUL7du3x8HBwTaS9cgjjwAwYsQI/v3vf9vFql%2B/vu1DFxYWdsecs9q%2BfTvJyckMHToUJycnKleuTK9evfjoo48YM2YMO3bsoHnz5jz%2B%2BOMAjBs3jo0bNwLg5uZGq1at%2BPLLL23FxJYtW%2BjQoUOu%2BxcgJCSE%2BfPn200bNnQow28pXvPk5vZmSMeOxmNkuvkZzxdM2smbwoSDVPbtMx4DYM0ac%2BKY5dVXH3YGfypb1nCIIiakAcaPdf905314ruWnz5MZnfPYY8ZjAFSubEqY3F9kfXtmdEtcnPEYeXYfiokc97/DhuXqcqPY2FiKFy9uNy2zsIiJibErJrKyWq1MmDCBevXq0ahRIwAqVaqEg4MDI0eOpGjRosyfP5%2B%2BffvyzTffULJkybysWo5MLyYmTJhgO7ufae3atSxdupSzZ8/i7u6OZ5Yzo97e3qSkpBAZGWmbVr58%2BTu%2Bx7p16/jvf//Lhg0bbAe1Z8%2BepdotZ6wqV67Mviw7wopZzkQVvrn1Z70JOqvdu3fbztanpqaSmppK27Zt75jXvUpLS7ONHmTVs2dP%2BvXrR/PmzXnyySd5%2BumnadWqVY4xEhMTmT59Ojt27LC73Cbr5TRZ1zuzSEtKSuL06dO4ublRokQJ2/zMg9fNmzeTlpaGv79/tpxjYmLw8PDIwxr/6ezZs5QsWdIuTuXKlUlOTubSpUuUK1furjHKlCnDjh07sk3PmnNQUBBLlizh4sWLeHh4sH37dtuN7hcvXuRvf/ubbVlPT89sH9Ks2%2BLdcr51/by9ve2KosqVK3P%2B/Hkg46zFY1l2Ku7u7lTOsnPo1KkTo0aNYtKkSVy9epWDBw%2BycOHCu/ZJVt26dcv24ANPZ2e4eVlhnjg6ZhQScXGQlpb3OAA5/O7umbt7RiHx/fdg9BK5LNtCnlgsGQc%2BN26A1WoslhmnmZ2cICXFeC4BAcba16iRUUj07Anh4cZiARi996pixYxCYtYsyOFkzj0ZO9ZYeyenjELi4sWM35UB10vlftQyJxZLxkFhYqLxTQagiEPO%2B9Z7Ssisz5PRv1UODn92jtFT6OfOGWvv4pJRSJw6ldE3BiVUyHtxY2a3/NXkuP%2B9h6sCrPe4zaekpDB%2B/Hh%2B//13u8vVhw4darfca6%2B9xqZNmwgNDbWdZDXDA73MKesB7q2ynukuVOj2aZ04cYLp06ezaNEiu4O628XOGtchlzvoEydOMHLkSMaNG8cLL7yAq6srr732Gqmpqblqn1uHDx%2BmatWq2aZXrFiRr776ij179rBt2zbeeustNm7cyNy5c7MtO3nyZI4ePcrHH39M5cqVOXPmDK1bt7Zb5nbr7eDgcNunA7i4uFCkSBH279%2BfhzW7u9xuC0Y98sgj1KlTh82bN1O9enXKli1LzZo1bTmk3bKTubU/shZ795LznZaFjD8Ut27nWX9PTZo0oXDhwnz33XdcuXKFJk2a3HMB5%2BXllX3U7fJl4ztWyIhhNM7ly8bzyHT1qvF4ZhxBZcYxGis/5WLW34DwcHNiVahgPAZkFBInThiLcZfPea6lpBiOZfRgLvPPj9Vq0oGhJR9tw2Yd6aanG4%2BVmGhOLjdumBLLjK4xo1seqvswMpHj/jeXPDw8bPcSZ4qNjcViseR4HJCUlMSQIUNITEzk448/vuOIg6OjI%2BXKlSMqKipPud3OA70Bu1KlSly9etXuLG5ERAQuLi5213fdTlJSEqNGjaJv377Zzph7e3sTERFhNy0iIuKerjHPdOTIEZydnXnppZdwdXXFarVy5MiRe45zJ1euXGHNmjU5XrqSkJBAWloaTZo0YcKECXz66ad88803xMTEZFv2t99%2Bo2PHjlSpUgWLxcKhQ4dynUOlSpVISEiw26hCQ0P573//i7e3N9evX7e74Tc%2BPj7HHPLC29ubmJgYrly5YpsWERFB4cKF76l6z41OnTrxn//8h//85z90zHJpjZeXFxcuXLD9/8KFC1y/ft2UnL29vTlz5ozd5U9Zt0cPDw/bKAVkXNr2xx9/2P7v6OhIhw4d%2BM9//sPXX39tl7eIiGgVDjQAACAASURBVIj8NdWpU4cLFy7YHWuEhYVRvXr1bE//tFqtjB49mkKFCrFixQq7QsJqtTJjxgzCs4wIJycnc/r06TwdG9/JAy0mfH19qVatGsHBwVy/fp3IyEgWLVpE%2B/btc3WN/PTp0/Hw8Mjx/oGOHTuyc%2BdOvvvuO1JTU/nhhx/Yvn07nTp1ylVuLi4unD59mri4OCpUqEBSUhJHjhzh6tWrvPvuuzg7OxMVFXXPQ0%2B3Sk9PZ//%2B/fTv35/q1avTq1evbMuMGDGCmTNnEh8fb1u%2BRIkSuLu74%2BrqSlxcHJGRkSQlJVGxYkXCwsJITk7mwIEDbN68GSBXVWfNmjWpVasWc%2BbMISEhgWPHjvF///d/JCUl8dhjj%2BHn58c///lPrly5wrVr15g4caJpXyBYr149KleuTHBwMImJiVy8eJHFixfToUOHHC/9MqJdu3aEh4fz7bff8uyzz9qmP/XUU3z55ZecPn2a%2BPh43nvvPZzvcEPlveTcokULABYvXkxycjInTpxg1apVtnt2AgIC%2BO677wgLCyMxMZF33nmHIkXsr3ru1KkT33//PUeOHLntZW4iIiJiQD770rrM74UIDg4mPj6eEydOsHz5ctstBG3btuXnn38G4Msvv%2BT333/nvffew%2BWW%2B4ssFgtnz55l8uTJREZGkpCQwKxZs3BycrLdX2yWB1pMWCwWFi5cSFRUFC1atOCFF16gXr16vPXWW7lqHxISwi%2B//EK9evXsHhF77tw524FvcHAwDRs25J133mHWrFm2m1Du5oUXXmDNmjW8%2BOKL%2BPn50atXL1588UXat29PhQoVePPNNzl27JjtMbX3qkGDBvj6%2BlKvXj3Gjx9Py5YtWb58eY4Hr1OnTuXUqVM89dRTNGzYkNWrV7NgwQIcHBwICAigYsWKPP3002zbto2xY8dy4sQJGjVqxOzZs/nHP/5B69atGTJkSK5GKRYvXsy5c%2Bdo0qQJgwYNYsiQIbbvgwgODsZqtdKqVStat25NWloab7/9dp7W/1YODg4s%2Bn/27juuqvp/4PgL0MsQB6hgImiJOVFBEXGgouYGc%2Beqn1tykZZmOHNUai7EneNrGqk5IrdmfrPsq2mJOCjRcjFMkCX7/v64cr9eJ/A5Jn15Px8PHhfOOZ83n3PuOu/zGWf5cm7evEmLFi3o3bs3Hh4efPDBB5rEf5CdnR0%2BPj5Ur14dpwe6SAwePJjmzZvTo0cPOnToQMOGDZ/aKpKfOtva2rJixQqOHz%2BOt7c3w4YNo3v37sYJA15//XX8/f0ZMGCA8X87OTmZdHWqXr06lStXplWrVsYxPkIIIYTQUCFLJgCWLFlCbGwsTZs2ZeDAgXTt2pW%2BffsCcOXKFWMviu3bt3Pjxg0aNWpkcl6ce3Pg2bNnU6VKFbp160aTJk24cOECGzZseOTipSozveqldiH%2BAfr06UPv3r2NLQOFQUZGhkky6ePjQ2BgoLGOWVlZtG3blo8//jjPSfEzqY4rsLCAMmUMg7hVx0zcn71KSdmyhlmhdu9W37c%2BfdTKm5mBlRWkpRWOAdg6naEfvmpdnjGz3jO5uxtmhPLw0GbMxAOtiwVStSosWgTjxqmPmVi0SK28TgfOznDtmvKYiWRHtSlzzc3BxgZSU7Xp/25rodifX8v3k%2Bp4R3NzKFECUlLUD85D3bHzzdraMCNUZKQmYyaSXqlX4LJaHhbQZpLAAvnxR%2B1jentrH7MQK3y3/RNCQzk5OWzatIm4uDg6duz4oqtjdOLECby8vAgPDyc7O5utW7eSkJBA4/sz52RlZfHpp5/i6OioXSIhhBBCCFOFsGXin%2BZvv2nd/5K9e/c%2BdQyBp6enyY32/lesXbuWRU%2B5Iufv78%2BsWbMKHH/VqlUsXbr0ieu7devGjBkznhknOzub%2BvXrU6VKFZYsWfJIf8IXqXHjxowaNYqxY8cSHx%2BPs7Mzixcv5qWXXuLatWt07NiR2rVrM2/evBddVSGEEEKIJ5JkQkGHDh3o0KHDi67G327w4MEMHjz4ucUfNmwYw4YNU45jYWFBeHi4BjV6Pp50HJ2dnQt1vYUQQoj/GUWwJUFrkkwIIYQQQoiiSZIJZXIEhRBCCCGEEAUiLRNCCCGEEKJokpYJZZJMCCGEEEKIokmSCWVyBIUQQgghhBAFIi0TQgghhBCiaJKWCWVyBIUQQgghhBAFIi0TQhQlNjZq5c3MDI9WVqDXq8XS4o7kxe5/hDVtCllZarFSU9XKW1gYjktaGmRnq8U6ckStfJky0LYtfPcdJCSoxercWa181aqGRx8fcHJSiwUQFqZW3t3d8HjsGJw5oxQqfdvXSuXNzEAHZDg6K7%2BdstPUyuf%2B/%2BxsyMlRiwVw6Ya1UnlLS6hSBa5GW5GerlaXKlXUyhufp%2BIllJ8nyzp11ALkyn1fKSqZmqRQ2hwoQQlSAA1eNJTUIEYBSMuEMkkmhBBCCCFE0STJhDI5gkIIIYQQQogCkZYJIYQQQghRNEnLhDI5gkIIIYQQQogCkZYJIYQQQghRNEnLhDJJJoQQQgghRNEkyYQyOYJCCCGEEEKIApGWCSGEEEIIUTRJy4QyOYJCCCGEEEKIApGWCSGEEEIIUTRJy4QySSaEEEIIIUTRJMmEMjmCQgghhBBCiAIpUslESEgI/fv3V44zadIkAgMDNajRfw0aNIhFixZpGlPkT2RkJNWrVyc6OrpA5SdMmMCECRM0rpUQQgghnhtzc%2B1/iphCs8cDBw4kKCjoset27dqFh4cHqamp%2BY67bt06srKyAAgICGDTpk1K9cyvpUuXUqNGDdzc3KhTpw6enp4MHDiQ3bt3m2z32WefMW7cuGfGu3btGvv27Xte1dVESEgIbm5uxn2uXr268W83Nzd27typ/D%2BedOKumhAIIYQQQoi8KzTJRI8ePdi7dy9paWmPrNu5cyedOnXCxsYmXzHv3LnDxx9/THZ2tlbVLJC6desSHh7OuXPnCAsLo1evXnzyySdMmTIl37EOHDjA/v37n0MttRMQEEB4eDjh4eGsXbsWgFOnThmXde3a9QXXUAghhBACaZnQQKHZ43bt2mFubs6BAwdMlt%2B6dYsTJ07Qs2dP0tLSmDlzJi1btqR%2B/foMGDCA33//3bht9erVWb9%2BPc2aNSM4OBgfHx/0ej0NGzbkq6%2B%2BYunSpfTq1cu4/ffff4%2Bfnx/169fH39%2BfH3/80bhu9%2B7ddOzYEXd3d3x9fdm8ebMm%2B%2Bno6Ejnzp357LPP%2BOqrrzhx4gQAAwYMYP78%2BQBcuXKFt956i4YNG%2BLp6cmoUaOIj49n7dq1zJ8/n3379uHm5kZ2djZ37txhzJgxeHt707BhQ4YOHcqtW7dMjsmBAwd44403qF%2B/Pl26dOH8%2BfN5OgY//vgjvXv3xt3dnebNm7Ns2TJNjkGukydP0rNnT2P8xYsXo9frNYu/ZMkSevbsabLsxIkT1K9fn5SUFOLi4hg8eDDu7u507tyZ8PBw43ZZWVnG11PTpk2NSVF%2B6rx//366dOlCvXr18PX15fPPPzeuS01NZcyYMdStW5f27dvz448/UrNmTU6dOsXEiRMfaaXavn07vr6%2Bmh4fIYQQosiTZEJZoZnNydLSki5durBjxw78/PyMy3ft2oWrqyt169Zl1qxZnD9/ntDQUEqXLs2SJUsYNWoUe/fuxczMDIBDhw6xc%2BdOypYta%2BxSdOrUKSwtLVm6dKkxbkxMDKNHj2b27Nm0bduWr7/%2BmrfffpsjR46QlJTExIkTWbt2Ld7e3pw4cYJBgwbh4eFBjRo1NNnfV199lSZNmrBv3z4aN25ssu7DDz/Ew8ODNWvWkJKSwsSJE1m%2BfDmTJ0/mt99%2BIz09nYULFwIwb948UlJSOHz4MHq9nnHjxjFnzhyTfV2zZg0fffQRL730EqNGjWLhwoWsXr36qccgLS2NgIAApk2bRpcuXfj9998ZMmQILi4udOnSRXn/Y2NjGTx4MJMnT6Zbt25ERkYydOhQKlSoQO/evZXjA3Tt2pWQkBD%2B%2BOMPKleuDBhadlq3bk2JEiWYPHkyWVlZfPfdd6SkpDB%2B/PhHYhw5coSvv/4aOzu7fNU5IiKCd955h6VLl9K8eXNOnjzJiBEjqFy5Ms2aNWPevHn8/vvvHDx4EICJEyeSk5NjrPeIESNITk7G1tbWWO/OnTsbX%2Bd5ERsbS1xcnMmy8qVK4VC%2BfN4P4sNy/38%2B6vFExTT4%2BLGwMH3UIlZhqEuZMmrlS5Y0fVRRtapa%2BUqVTB9Vuburlc/9DNfgs1z1baDl20n1/CW3vFbnQZaWauV1OtNHFYXpeSp0VJ5wLV8097//xD9ToUkmAHr27Em3bt24desWL730EgA7duygb9%2B%2B5OTk8NVXX7Fo0SIcHR0BGDduHJs2beLs2bPUq1cPgA4dOlCuXLln/q%2B9e/fi7OxMx44dAejWrRuWlpbk5ORQqVIlTpw4QenSpQHw9vambNmyREREaJZMALz88stERUU9sjwxMRErKyuKFStG6dKlCQkJwfwJb9YZM2aQlZVl7ALWpk0bVqxYYbKNv78/r7zyCgC%2Bvr7Gq%2BxPOwZhYWFUq1bN2CWpevXq9OnTh127dmmSTHz99ddUrlyZPn36AFCnTh38/PzYs2dPnpOJb7755pEuXw9euXdxcaF%2B/fp8/fXXjBo1Cr1ez6FDh5g5cyY5OTkcPnyYZcuWUapUKUqVKkX//v35%2BeefTeJ17NgRe3v7fNd5%2B/btNG/eHF9fXwCaNGmCj48Pe/bsoVmzZnz33XcMGjTI%2BFr%2Bv//7P2OrkJeXF3Z2dhw8eJDXX3%2Bd5ORkfvjhh3wP7g4NDSU4ONhk2ai332b0mDH5ivNYqmcKAFZW6jFyqZ58a0mLE/i2bdVjADx0oaJAtKpLYZucQIPWZg3OcwEoXlw9hhYn3QAlSmgTR4u3AUDFitrE0YIWzxNocLEBtLloAdo84dbW6jGSktRjFFQRbEnQWqFKJmrWrEnNmjXZuXMnI0eO5MyZM9y8eRM/Pz/%2B%2BusvUlJSCAgIMLk6m5OTw61bt4zJRMU8fvL8%2BeefVHroSlmnTp2Mv2/ZsoVt27YRGxuLXq8nIyODjIwMDfbyv7Kzs7F4zAfCqFGjePfdd9m5cyfNmjWjc%2BfO1K1b97Ex/vjjDz766CPOnj1LWloaOTk5lHnoxOrB/bS2tiY9PR14%2BjH4888/CQ8Px83NzbhOr9fz8ssvF2xnH3L9%2BnVjgpPLxcWFw4cP5zlGp06djF3DckVGRpokO127dmXdunWMGjWKX3/9lYyMDJo1a8adO3fIzMw02f8qVao88j8efD3lp87Xr1/H1dX1kW0jIiIAiIuLw8nJybjuweNsbm5Oly5d%2BPrrr3n99df57rvveOWVV6hWrdrTDscjevfubUxmcpUvVQoeMy4pz8zMDIlEejqodrlKSVErD4Yv1DJlICEBVMdGadEyUbKk4UtRtS4nT6qVL1nSkEicOKH%2BJf3NN2rlK1UyJBLz58P162qxAI4dUytfo4YhkejbFy5eVAqVceK0UnkzM8MJamam%2Btvp/sd6gZmbG84rU1K0uUj8119q5XU6QyJx8yaofvWqJiRaPk86Cw3GcFpYqH/G5FL5PjA3NyQS9%2B5Jy0IRV6iSCTAMxN6wYQMjR45kx44dtGnTBjs7O5LufyF%2B8cUX1KlT54nlH3dy/jjm5ubGbiUP27p1K6tWrSIkJARPT08sLCxo0aJF/nfmGc6fP0/9%2BvUfWd6yZUuOHj3Kd999x%2BHDh%2Bnfvz/vvffeI9Pa5uTkMHz4cBo0aMD%2B/fuxt7dn69atj0wx%2B6SuMU87BlZWVrRo0eKRVg6tPCkxy083nrzo2LEjs2fPJjw8nIMHD9KhQweKFStm/P8PDs5/3LEo9kBXnPzU%2BVnb6vV6k9gPx%2BjatStr167l9u3bHDx4sECtQQ4ODjg4OJguvHdP/dsQDDFU49yfZU0T2dnaxlORna3%2BRZ%2BQoE1dkpLUY12%2BrE1drl/XJtaZM%2BoxwJBIKMbSagiTFm8nrc7lcnK0iaWa3OTKyFCPVZiep0JHiydbqxfNiyItE8oK3RHs0qUL0dHRnD59mv379xsH0JYsWZIyZcpw6dIlk%2B2vF/BKV6VKlbhy5YrJsk2bNnHt2jXCw8Np2LAhjRs3xsLCgri4OGJjYwu2Q0/www8/cPr0aZPWkFzx8fGUKFGCjh07smDBAmbMmEFoaOgj292%2BfZsbN24wYMAAY1ecBwdXP8vTjoGLiwuRkZEm3Ybi4uI0a51xcXF5pIvXlStXcHFx0SR%2BrlKlStGqVSv27t3Lvn37jONxypUrh4WFhclg9QcH86vW%2BXHbRkVFGbe1t7fn5s2bxnUPDv4GqFq1KjVr1mTXrl0cO3ZMk65lQgghhHiIDMBWVuj2uGTJkrRr1465c%2BdSokQJvL29jev69OnD8uXLuXz5MpmZmaxfv54ePXpw7969x8ayut8n%2B8qVK4/co6Jz587cunWLL7/8koyMDL755hs%2B/fRTSpQogZOTE1FRUdy9e5cbN24wa9YsKlasSExMjPL%2Bpaens3fvXsaPH8%2BgQYMeaWVJS0ujXbt27Nq1i6ysLNLS0oiIiDCehFpaWnLr1i0SExOxt7fHxsaGX375hfT0dL7%2B%2BmsuXLhAcnIyKXnoQvK0Y9CpUycSEhIICQkhLS2Na9euMWjQIDZs2KB8DMDQYnDlyhW2bt1KZmYmv/zyC7t27Xou08Z27dqV0NBQzMzMcL8/eFOn0%2BHl5cXGjRtJTk7m%2BvXrfPHFF5rV2c/Pj3//%2B9989913xkHe//73v/H39wegcePGbNmyhdu3bxMTE8PGjRsfW%2B%2BQkBDq1KljHFshhBBCCFGYFLpkAgwDsc%2BePUv37t1Nun8EBATQvHlz%2Bvbti5eXFwcPHmT16tVYP2HwT82aNXF3d6dHjx5s2bLFZF25cuVYu3Yt69evx9PTk1WrVrFs2TLs7e154403qFy5Mi1atGDYsGH079%2Bf/v37s27dOpPpPfPq7Nmzxhu2eXl58dlnnzFx4kQmTpz4yLZWVlYsXryY9evX07BhQ1q2bEl0dDRTp04FDC03V65coVWrVsTGxjJ9%2BnRWrVpFkyZNOHnyJEuXLqVChQq89tprz6zX046BnZ0dISEhHD58GE9PT/r370%2BrVq0YNGhQvvf/cZydnVm6dCmbN2%2BmUaNGTJo0iXfeeee5XIH38fFBp9PRuXNnk%2BVz584lMzOT5s2bM3z4cN58803N6tywYUNmzJjBJ598gqenJwsWLGDRokU0aNAAMNxFPbfVZPjw4QwZMgTAZKB9p06dSEtLk1YJIYQQ4nmRlgllZnqZuF78j0tMTKRly5bs2rULZ2fnF10do4yMDHT3p2C5evUq7dq148iRI8aB2VeuXKFHjx4cO3aMElpNsfKEVrw8MzMzzMKUlqbeeTgxUa08GKaXLVvWMNpTdcyE6lS1Wg4GP3JErXyZMoZZmA4eVB8z8ZhWs3ypWhUWLYJx47QZMxEWplbe3R1OnwYPD%2BUxE%2Blpau8BMzPDQOOMDPW3k8o4WjCc/%2BTOH6BF9/foaLXylpZQpQpcvao%2BZuIxc2vki5bPk2WxQjYA%2B6FeG/mi9ah9raYAy6%2B7d7WPeX820KKi6KVPokhJS0tjxowZtGrVqlAlEkuWLKF79%2B7ExcVx7949Vq1axauvvmqcEjkxMZHp06fTr18/7RIJIYQQQpiSlgllhW42p3%2BSDz/8kC%2B//PKJ60eOHElAQMDfWKO/x4gRIzh%2B/PgT13/44YdKYx%2BGDh1qvDP448ydO/eRLkuP89NPPzF06FAaNWrEvHnzClyf52Ho0KFER0fTpUsXsrOzqV27NosWLcLc3JydO3cyffp0Wrdu/T/5%2BhFCCCEKjSJ48q81SSYUTJkyhSlTprzoavztntd0sblWr16tSRwvLy/Onj2rSSytWVtbM2fOnMeu69q163MZiC6EEEIIoTVJJoQQQgghRNEkLRPK5AgKIYQQQgghCkRaJoQQQgghRNEkLRPKJJkQQgghhBBFkyQTyuQICiGEEEIIUUjcuHGDYcOG4eXlRatWrZg3bx45T7iXx8aNG2nXrh0eHh688cYbnDt3zrguPT2dqVOn4uPjg5eXF2PGjCE%2BPl7z%2BkoyIYQQQgghiqZCeJ%2BJ0aNH4%2BjoyKFDh1i3bh2HDh1iw4YNj2x35MgRli5dyieffMIPP/xAq1atGDFiBKn3b0a4cOFCIiIiCA0NZf/%2B/ej1et5//33l%2Bj1MkgkhhBBCCCEKgfDwcC5evMiECRMoWbIkVapU4a233iI0NPSRbUNDQ%2BnWrRv16tXDysqKIUOGAPDtt9%2BSlZXFtm3bCAgI4KWXXqJMmTKMGzeOo0ePEhMTo2mdZcyEEEVJVpZa%2BdwrLtnZ8IQm1zybOlWtPICzMwQFwfLlcO2aWqwn3PfjRdhfqqdS%2BZIloAnwQ4m2JCk%2BTe0WeagF0OkMj%2BPHQ0aGWiwgfdvXSuXNzEAHZJw4jV6vVhdLKzO1AO7ucPo0usYecOaMUqjUO2o7Y3Z/V/R69bc2QPWb36oFsLWFKp5UiTsJyclqsdLs1cpbW8Orr6K7Ggn37qnFOnZMrXz58tCnD2zdCnFxarGA%2BP6jC1zWwgJKAYnZJcjOVq4KduohCuY5jJmIjY0l7qHnp3z58jg4ODyzbEREBE5OTpQuXdq4rHbt2ly5coXk5GRsbW1Ntu3YsaPxb3Nzc2rWrEl4eDg1a9YkKSmJ2rVrG9dXrVoVKysrIiIicHR0VNlFE5JMCCGEEEKIIkmP4kWBxwgNDSU4ONhk2ahRoxg9%2BtnJW0JCAqVKlTJZlptYxMfHmyQTCQkJJklH7rbx8fEkJCQAPBKrVKlSmo%2BbkGRCCCGEEEIIjfTu3RtfX1%2BTZeXLl89zeX0%2Bmk2ftW1%2BYhWUJBNCCCGEEKJI0qJb38McHBzy1KXpcezt7Y2tCrkSEhIwMzPD3t60y56dnd1jt61WrZpx24SEBEqUKGFcf/fuXcqWLVuguj2JDMAWQgghhBBFUk6O9j8q6tSpw61bt7hz545xWXh4OK6uriZJQe62ERERxr%2Bzs7M5f/489erVw9nZmdKlS5usj4yMJCMjgzp16qhV8iGSTAghhBBCCFEI1KpVCzc3NxYsWEBycjKXL19m3bp1vPHGGwC0b9%2BeU6dOAfDGG2%2Bwc%2BdOfvnlF%2B7du8fy5cvR6XS0bNkSCwsLevXqxYoVK7h16xbx8fF8%2BumntG3blnLlymlaZ%2BnmJIQQQgghiqTn0c1J1ZIlS5gyZQpNmzbF1taWPn360LdvXwCuXLlivI%2BEj48P77zzDuPGjeOvv/7Czc2NVatWYWVlBcCYMWNISUnB39%2BfrKwsWrVqxfTp0zWvryQTQgghhBBCFBIVKlRg9erVj1136dIlk7/79u1rTDQeptPpmDZtGtOmTdO8jg%2BSZEIIIYQQQhRJhbFl4p9GkgkhhBBCCFEkSTKhTgZgCyGEEEIIIQpEWiaEEEIIIUSRJC0T6qRlQgghhBBCCFEghTaZCAoK4r333nvR1TDx1Vdf0bRpU01jhoSE0L9/f01jCu1t2rSJtm3bvuhqCCGEEEJDhe2mdf9EmnRz8vX1ZejQocYbauTasmULq1ev5siRI/mOOWvWLC2q9kw//vgjtra2uLm5aRr3p59%2BYuDAgeh0OgAsLCyoUqUK7du356233jLOARwQEEBAQMAz42VnZ7Nx40b%2B7//%2BT9N6aunkyZMMGjTI%2BHdGRgbFixfHzMwMAH9/f%2BXndevWrQQFBRmPK4ClpSXVq1dn7NixNGrUSCm%2BEEIIIYqOonjyr7UiP2Zi/fr1tGzZUvNkItepU6ewtLQkMTGRiIgIFi5cyP79%2B9m0adMjt0V/mvPnz7NmzZpCnUx4enoSHh5u/Lt69eqEhITg4%2BOj6f9xdHTk2LFjxr9TU1PZuHEjw4YNIywsjEqVKmn6/4QQQgghxOP9bd2coqOjGTlyJF5eXjRo0IDAwEASEhIAw1V8d3d31q9fj4eHB2fOnGHSpEkEBgYCMGjQINzc3Iw/tWrVwtfX1xj70KFD%2BPn5Ub9%2BfXx9fdm4caNx3aRJk/jwww%2BZO3cujRo1onHjxsYbgYwYMYKjR48ya9Ys3nzzTQDCw8Pp27cvDRs2pEmTJkybNo3MzEzl/S9VqhTe3t6sX7%2Be5ORk1q5dC8DSpUvp1asXAPfu3WPixIl4e3vj7u5Onz59OHfuHGfPnqVPnz7cvn0bNzc3Tpw4gV6vZ/78%2BbRo0QJ3d3def/11Tp48afx/AwYMYMWKFbz77rt4eHjQvHlzdu3aZVx/7do1Bg0ahLu7O61atTI5Zjdu3GDEiBF4eXnh6enJe%2B%2B9R3JysvIxyHXz5k1j/IYNGzJ%2B/Hju3r1b4Hg2NjaMGDECe3t7jh8/blz%2B%2Beef0759e%2BrWrUuHDh3Yu3evcd1ff/3FmDFj8Pb2pmHDhgwfPpyYmBjj%2BjNnztClSxfq16/P4MGDiY%2BPByAqKopatWqRkpICGBKZ2rVrs3DhQmPZBQsWMHbsWAB27txJhw4dcHd3p3Xr1oSGhgJw4sQJ3NzcSExMNJZLTU2lXr16/Pjjj1y%2BfJmBAwfSoEEDPD09GTNmjPH9IoQQQghtSDcndX9by0RAQACurq4cPnyYtLQ0xo4dy7Rp01i8eDEAmZmZ/PHHH/zwww9YWloaT7oAPvvsM%2BPvycnJdOvWjR49egBw8eJFxo4dy%2BLFi2nRogWnTp1ixIgRVK5cmRYtWgAQFhbGpEmTOH78OLt372bKlCn4%2B/uzYsWKR7poBQYG4ufnx7/%2B9S9iYmLo06cPrq6uDBgwQJPjYGNjQ69evdixYwdjxowxWbdhwwZu377NwYMH0el0rF69milTprBjxw4%2B/PBDFixYYDxZ3rlzJzt37mTbtm2UL1%2Be5cuXM2bMGL7//nssLCwAw8n0nDlzmDNnDitWrGDmzJl0WEmCBQAAIABJREFU7NiR4sWLM2rUKBo1asSyZcu4evUq/fr1o2rVqjRp0oSAgAA8PDxYuHAhqampvPPOO3z88cd8%2BOGHyvuv1%2BsZOXIktWrV4vDhw9y7d4/Ro0czc%2BZMFixYoBT3waTv4MGDLFy4kJUrV1K3bl327t3LhAkTqFatGq6urnz88cfcu3ePI0eOkJ2dzdixY5k7dy6LFi0iKyuL0aNH061bN0aNGkVERARjxozBysqKV155hbJly3L27Fm8vb05c%2BYMzs7O/Pzzz8b/ferUKbp27coff/zB%2B%2B%2B/z/r162nUqBHHjx9n6NCheHh44OXlRbly5di/fz89e/YE4NixY5QuXRovLy/eeustvLy8WLduHUlJSbz77rusXLmSiRMn5vmYxMbGEhcXZ7KsvI0NDg4OBT7OmJubPqpwdlaPUaGC6aOK%2B%2B8b5fKqcYCSJdXK5zZ65qPx88ke6FJYIMWLmz4qut9rUrm8ahwA3N3VyteoYfqoQPVlp%2BVbGwBbW7XyNjamjyqsrdXKW1qaPqooX16tvJ2d6aMildeNlq%2BZ7Gz1GOLF0SyZmDVrFnPmzDFZlpOTg6OjIxcuXCAiIoKVK1dia2uLra0tw4YN4%2B233yYjIwMwJBN9%2B/Y1jiV4kmnTpuHi4sLQoUMB2L59O97e3rRp0wYAb29vWrZsyZ49e4zJRKVKlXj99dcB6NixI5MnT%2Bbq1auPPanauXMnOp0OCwsLKlasiKenJ%2BfOnVM7OA95%2BeWXuX79%2BiPLExMTKV68OFZWVhQrVuyp4ym6dOlC69atKXn/rKNTp04sXbqUmzdv4nz/JM3d3Z3mzZsD0KFDB4KDg4mNjeXu3btcunSJDRs2YG1tTc2aNQkODsbR0ZHw8HB%2B%2B%2B03tmzZgrW1NdbW1owePZrBgwczc%2BZM4/iHgjp37hyXLl1i3bp1xtfC0KFDGTduHJmZmRQvwAlHUlIS69atIzk52dhitW3bNvz9/WnQoAEAfn5%2BrFu3jv379%2BPq6sqsWbPIzs7G%2Bv6XjK%2BvL%2BvWrQPg119/JT4%2BnmHDhqHT6XB3d8fX15cffvgBAC8vL06fPo23tzcnT57E39%2Bf9evXk5GRgV6vJzw8nI8//hhnZ2dOnDhB6dKlAWjWrBllypQhIiKCatWq4efnx%2B7du43JxMGDB%2BnUqRPm5uYkJSVhZWWFhYUFZcqUYeXKlZjn8xM7NDSU4OBgk2Wj3n6b0Q8lsQWi%2BuUMEBSkHiPXkCHaxVKlmgkATZpoUA%2BgXj0tomiQ9IE2CR%2BgmNoYaZLbnD6tQRBg82blEKU0qAao5wBGnp7axKldW5s4WqhcWT3Gq6%2BqxwBo106TMFq8brR4zdxv/H8himJLgtY0SyaCgoKeOAD7%2BvXrlC5dmvIPZOQuLi5kZmaadC2pWLHiU//Htm3b%2BM9//sOuXbuMJ7XXr1%2BnatWqJttVrlyZ0w98yD/Yhz735DEtLe2x/%2BPEiRPGq/VZWVlkZWXRvn37p9Yrv7Kzs42tBw/q27cvgwcPpkWLFjRv3pw2bdrQunXrx8a4d%2B8ec%2BbM4dixYyZdhHKTMzDd79wkLS0tjT///BNbW1vKlCljXN/k/tnLN998Q3Z2Nl5eXo/UOT4%2BHnt7%2BwLs8X9dv34dOzs7kziVK1cmIyOD27dv89JLLz0zRkxMjMkYl4yMDLy8vNi4caPxNXb9%2BvVHxmq4uLhw48YNAK5cucJHH33EuXPnSEtLIycnh7JlyxrjlylTBtsHPiFffvllk2Ri//79gKEVYty4cXz//fdERESQlZWFg4MDLi4ugGEWqO3bt3P79m30ej0ZGRnGFhR/f39WrVpFdHQ09vb2HD16lE2bNgEwatQoJk6cyFdffUWzZs3o0qULderUyceRht69e5t0BwRDywT3u2gViLm5IZG4d0/9E/iBrmEFVqGCIZFYswaio9VijRqlVt7CwpBIJCUpX2b74XyZZ2/0FCVKGBKJX39Ve7oBmjhfUwtQvLjheYqOBg26jGY4qiU3ZmaGKmVmgl6vVhddYw%2B1ADVqGBKJvn3h4kWlUIlH1RIbc3PDSWFysjYnV6UunXz2Rk9jY2NIJCIiIDVVLdb9CzoFZmlpSCT%2B%2BAPS09ViqSagdnaGRGL/fk3OwBM79ilwWa1fMy/KP7nuhcXf0s3pwRPchz14pbtYsSdX5/Lly8yZM4fly5ebnIg%2BKfaDcfN6Rffy5cuMHTuWiRMn0qtXL6ysrHj33XfJysrKU/m8On/%2BPC%2B//PIjyytVqsSePXv46aefOHLkCFOnTmX37t0sWbLkkW1nzJjBpUuX%2BPzzz6lcuTLXrl17ZOrSJ%2B23ubk5OU9491haWmJjY8OZM2cKsGfPltfXwtM8OAA7JyeHPn36UKVKlUcSjCf9j6ysLIYNG4a3tzcLFizA3t6eL774gpCQEGPZh5/zB49X48aN%2BeSTT0hPT%2BfixYvUrVsXd3d3fv75ZzIzM/H29gbgiy%2B%2B4LPPPmP58uU0aNAACwsLmjVrZozzyiuvUKdOHb755htcXV2pUKECNWvWBKB169YcPXqUo0eP8u2339K3b18mTZpE375983SMABwcHB5tfUtK0uaTU4uOodcUT1IfFB2tHk%2BrdvbsbOVYSUnaVCUlRYNYT3nP5ktmpiaxVBOAB%2BMox9Lqc/LiReVYWr18c3I0iqXVOLvUVPVYql31cqWnGy6kqHio62mBxcdrEkuL51qz14z4x/pbBmA7Oztz9%2B5dbt%2B%2BbVwWFRWFpaUljo6OzyyflpbGuHHjGDRo0CNXzF1cXIiKijJZFhUVZezqkx8XLlxAp9MxcOBArKys0Ov1XLhwId9xnubOnTts3ryZLl26PLIuJSWF7OxsmjRpQlBQEFu3bmX//v3Gwb8POnv2LH5%2BflSpUgUzMzMiIiLyXAdnZ2dSUlKIjY01Ljt06BD/%2Bc9/cHFxITU1lWsPnJglJyc/tg4F4eLiQnx8PHfu3DEui4qKwtra2qTlKq/Mzc2ZOXMmX331lckAdBcXF65cuWKybVRUFC4uLsTFxREdHc3AgQONien58%2BeN2zk4OJCUlGQcZA3w%2B%2B%2B/G393dnamZMmSfPXVV7z66qvGrlCnT5/m559/NrbynD17lkaNGtGoUSMsLCyIiYkxeQ8AdO3alX379rFv3z78/PyMy%2BPj47G1taVz584sWLCAKVOm8OWXX%2Bb7%2BAghhBDiyWQAtrq/JZlwc3OjatWqLFiwgNTUVGJiYli%2BfDmdOnXKUx/5OXPmYG9v/9jxA35%2Bfhw/fpxvv/2WrKws/v3vf3P06FG6du2ap7pZWlry559/kpSUhJOTE2lpaVy4cIG7d%2B8yb948dDodsbGx6BUvYeXk5HDmzBmGDBmCq6sr/fr1e2SbMWPG8PHHH5OcnGzcvkyZMpQuXRorKyuSkpKIiYkhLS2NSpUqER4eTkZGBr/88gvffPMNgEmC8CQ1a9akVq1aLFq0iJSUFCIjI/nggw9IS0vj1Vdfxd3dndmzZ3Pnzh0SExOZNm2aZjcQrFevHpUrV2bBggXcu3eP6OhoVqxYQZcuXR7b9SsvatSowYABAwgKCiL9fhO0n58fu3bt4uzZs2RmZrJ161auXr1Kx44dKVu2LNbW1pw5c4b09HR27tzJpUuXSEpKIjU1lfr162NjY8PatWvJyMjgP//5j8lUtGDo6rRhwwYaNmwIGMan/PLLL0RERNC4cWMAnJyciIqKIjExkevXrzN79mwqVqxo0rWvY8eOXLx4kQMHDtC5c2fAMKtT27ZtCQsLIysri3v37nH%2B/Hlj1ykhhBBCiMLib0kmzMzMCAkJITY2lpYtW9KrVy/q1avH1KlT81Q%2BNDSUn3/%2BmXr16plMEXvjxg3jie%2BCBQvw9PTkk08%2BYf78%2BXm%2BeVmvXr3YvHkz/fv3x93dnX79%2BtG/f386deqEk5MTkydPJjIy0jhNbX41bNgQNzc36tWrx6RJk2jVqhXr1q0zuelarg8//JA//vgDHx8fPD092bRpE8uWLcPc3JzGjRtTqVIl2rRpw5EjRxg/fjyXL1%2BmUaNGLFy4kClTptC2bVsCAgLy1EqxYsUKbty4QZMmTRgxYgQBAQHGMQYLFixAr9fTunVr2rZtS3Z2Nh999FGB9v9h5ubmLF%2B%2BnJs3b9KiRQt69%2B6Nh4cHH3zwgVLc0aNHk5mZaewS5u/vz5AhQ5gwYQJeXl5s3bqVdevW4ezsjE6nY9q0aYSEhNC0aVN%2B%2BeUXli5dSrly5Wjfvj02NjYsW7aM/fv34%2BnpyYoVKx65v4eXlxdXrlwxDvC2t7enZMmSlC9f3jj2ol%2B/flSsWBEfHx9GjBjBwIEDeeONN1i1ahVffPEFAHZ2dvj4%2BFC9enWcnJwAw4xfixcvZs2aNTRs2JCWLVvy119/EaTlgGUhhBBCSMuEBsz0qpfchRBK%2BvTpQ%2B/evY0zjj1Xqh3ozc0No3tTUtQ/MSdMUCsPhullg4Jg1iz1MRMPzUaXbxYWUKYMJCQodyDef6qsUvmSJQ0zQv3wg/pT3s71sloAnc7wPF27psmYifRKVZ%2B90VOYmRmqlJGhPmbC0kpxfll3d8OAXA8P5TET8XfUdsbCAkqVgsREbfq/2/3yrVoAW1vDjFAnT6qPmVCcOARra8MsTJGR6mMmHmrlzrfy5aFPH/jiC03GTMT3H13gspq/ZrSZ7TbftBy%2Bl0uLmc//SYr8HbCFeFFycnLYvHkzcXFxdOzY8UVXRwghhBAi3ySZKIC9e/c%2BdQyBp6enyY32/lesXbuWRYsWPXG9v78/s2bNKnD8VatWsXTp0ieu79atGzNmzChw/MIkOzub%2BvXrU6VKFZYsWYKlFjdDEkIIIUS%2BFMVuSVqTZKIAOnToQIcOHV50Nf52gwcPZvDgwc8t/rBhwxg2bNhzi1%2BYWFhYEB4e/qKrIYQQQgihRJIJIYQQQghRJEnLhDpJJoQQQgghRJEkyYS6v2VqWCGEEEIIIcT/HmmZEEIIIYQQRZK0TKiTlgkhhBBCCCFEgUjLhBBCCCGEKJKkZUKdJBNCCCGEEKJIkmRCnZler9e/6EoIIf4emZnqMYoX1yhOTrp6EDMz0OkgIwMUP8qCV6vdOLB8eejdG0JDIS5OKRQjR6qVB7CwgOxs9Tj37qmVNzcHGxtITdXmS1t1n8zNoWRJSEpSr49qeQsLKFUKEhPV98vO3kwtgLs7nD4NHh5w5oxaLODy72rvR50OnJ3h2jXD21uFn59a%2BZo1Yds26NEDLlxQixVxSvENZWYGVlaQlqb8mQegt7JWro5WZ5Fmii/hglJ9Th%2BnZk3tYxZm0jIhhBBCCCGKJGmZUCcDsIUQQgghhBAFIi0TQgghhBCiSJKWCXWSTAghhBBCiCJJkgl10s1JCCGEEEIIUSDSMiGEEEIIIYokaZlQJy0TQgghhBBCiAKRlgkhhBBCCFEkScuEOkkmhBBCCCFEkSTJhDrp5iSEEEIIIYQoEGmZEEIIIYQQRZK0TKiTlgkhhBBCCCFEgUjLhBBCCCGEKJKkZUKdJBNCCCGEEKJIkmRCnXRzEn%2BLGzdu4ObmxpUrVwCoXr06x44dA8DX15ctW7Y8M8aAAQOYP3%2B%2BZnUKDAxk0qRJmsUTQgghhHieEhISGDduHE2aNKFZs2Z88MEHpKWlPXH7AwcO4Ofnh7u7O%2B3atePLL780rlu6dCk1a9bEzc3N5Of27dv5qpMkE0ITT0oItmzZgq%2BvL05OToSHh/Pyyy%2B/gNo9f%2BfOnWPgwIE0aNCA5s2bs3bt2hddJSGEEEI8Q06O9j/P05QpU7h37x5hYWFs376dy5cvP/FC69mzZ5kwYQJjxozh5MmTTJ48mZkzZ3Lq1CnjNv7%2B/oSHh5v8lCtXLl91kmRCCEUJCQkMGTKEevXq8f333/PZZ5/x%2Beefs3fv3hddNSGEEEI8xT8pmbh9%2BzaHDh0iMDAQe3t7HB0dCQgIYPv27WRmZj6yfUJCAsOHD6dNmzYUK1aMFi1a8Oqrr5okE1qQMRPib3H9%2BnVat27Nnj17qFq16hO3%2B/XXX5k9eza//fYbOp2ONm3aMGXKFKysrADIzs5m6tSphIWFYWlpyZQpU%2BjYsSMA4eHhzJ07l8jISHQ6HW3btiUoKIjixYsD8OWXX7JixQru3r2Ln58fOQ%2B94zdt2sTnn3/OzZs3qVSpEoGBgbRp0%2BaZ%2B/bLL7%2BQkpLCuHHjsLCwoFq1agwePJht27bRoUMHANavX8%2BmTZv466%2B/qFChAoGBgbz22msATJo0CWtra7Kzs/n666%2Bxt7dn3rx5nDp1inXr1gHw7rvv0q1bt3wd89jYWOLi4kyW2dmVp3x5h3zFeW7MzLSLoUGs8uXVytvZmT7%2BrzBXvOT04FOkGgtAr1crn1sHLeqi%2BrLTsi64u6uVr1HD9FGRTqdW/v7HtvFRRc2aauVzG9Q1aVhXfdFo%2BJknnp/Hff%2BWL18eBwe1798LFy5gYWFB9erVjctq165NamoqUVFRJssBfHx88PHxMf6dlZVFXFwcjo6OxmWXLl2iT58%2BREZG8tJLL/H%2B%2B%2B/TrFmzfNVLkglRqLz33nsMGTKE7t27c/v2bQICAggNDeXNN98EICwsjDlz5hAUFERwcDDTp0/ntddeo1ixYgQGBuLn58e//vUvYmJi6NOnD66urgwYMICoqCimTp1KcHAwPj4%2B7N69m1mzZtG%2BfXvA0KcwODiYNWvWUKNGDY4cOcK4ceM4cOAAFStWfGa9zR76YC9dujQXLlwA4OTJkyxYsIDt27dTrVo1duzYwYQJEzh69Cj29vYA7Nmzh48%2B%2BogPPviAUaNG8c4779CzZ0%2B%2B%2B%2B471qxZw5w5c%2BjatSvm%2BTjrCA0NJTg42GTZ22%2BPYsyY0XmO8SRafMGD4tnGgzSoUO/eGtQDuJ8jFgoWFuoxbGzUYwBYW2sTRyslSrzoGvyXra0GQU6f1iAIsHmzJmGcNYkCFSqox9i2TT0GwLx5WkSx0iIIWFpqEkaLlESLvEb1IoGK59GS8Ljv31GjRjF6tNr3b0JCAra2tibnHKVLlwYgPj7%2BmeXnz5%2BPjY2N8SJshQoVcHZ2Zvz48Tg4OBAaGsqIESPYvXs3r7zySp7rJcmE0MysWbOYM2eOybKcnByTDPhZEhMTsbGxwdzcHAcHB7788kuTE2gPDw%2BaN28OQPv27Vm5ciV37tzBwcGBnTt3otPpsLCwoGLFinh6enLu3DkADh06RK1atYwtDT169GDDhg3GuNu2baNHjx7UqVMHgNdee40GDRoQFhbGsGHDnlpnd3d3rK2tWbx4MSNHjiQuLo7Nmzdz9%2B5dABo0aMDx48cpVaoUAJ07d%2Bb9998nMjKSxo0bA1ClShVatWoFQNOmTfnpp58YOnQoOp2OVq1asXjxYv766y/K5%2BPyee/evfH19TVZZmdXnse0hOZL8eIoxwAors9QD2Jm9t8KKX4bhe5QS27s7AyJxIEDkIfP9Kfq0UOtPBgSiexs9Tjp6WrlzcwMicS9e9qcMKjuk7m5IZFISVE/idCilcTWFpKT1etSqqWHWoAaNQyJRN%2B%2BcPGiWizg2i615KZ4cUMiER2t/nkTGKhW/uWXDYnEu%2B/C/TlECmzbpicPlM0TMzNDIpGerskbSm%2BpltyYmb3YRKCwetz3b16/v3ft2sV777332HWBgYHoC3DA9Xo98%2BfPJywsjI0bN2J5Pxnt2bMnPXv2NG731ltv8c0337B7927GjRuX5/iSTAjNBAUF8cYbb5gs27JlC6tXr85zjHfeeYfJkyezdu1amjVrhr%2B/v0m3qEqVKhl/z30zZGQYTkpPnDjBsmXLuHr1KllZWWRlZRlbHmJiYkzKguEEPteff/7J8ePHTRIMvV6Pq6vrM%2BtcunRpli1bxscff8ymTZuoVq0a3bp1MyYy2dnZLFu2jH379nHnzh1judx6g%2BHqwIP7ZW9vj%2B5%2BP4Hcx/R8ntU5ODg80qSqRRKgGS2/gfR65XgPtUgXWHy8drEKA9WT3NxrAXq9NlcAtbqKqEXfZi3ropz4nTmjSV24eFGTWBkaXCsAw2eWaqz7jcTKrlzRIJZWn3safOYJg%2BfRMvG479%2B88vf3x9/f/7Hrjh8/TnJyMtnZ2Vjcb3pOSEgAoGzZso8tk5OTw/vvv8/Zs2fZsmULzs5Pbzd0cnIiNjY2X3WWZEIUKj179qRNmzYcOXKEw4cP07VrVxYuXGhsUXi4O1Guy5cvM3bsWCZOnEivXr2wsrLi3XffJSsrCzCcuOf%2BnuvBMRNWVlaMHz%2BeQYMGFajeDRs2ZOvWrca/9%2B/fb2yRWbZsGXv37mXFihXUqFEDvV5PrVq1TMo/3H0pP92ZhBBCCFEw/6T7TNSsWRO9Xs/FixepXbs2YBgvWqpUqSfOljlnzhx%2B%2B%2B03tmzZQpkyZUzWhYSE4O7ujre3t3HZ5cuXjd2g8krOWEShEh8fj52dHd27dyckJIThw4ezLQ8dXi9cuIBOp2PgwIFYWVmh1%2BuNYxbAcJUgOjrapMzly5eNv7u4uHDp0iWT9Tdv3sxTc2J6ejo7duwgOTnZuOz48eO43x8QGR4eTuvWralVqxbm5uZEREQ8M6YQQgghxIPs7e1p164dixYt4s6dO0RHR7Ns2TJ69OhBsWKG9oE333yTPXv2APDzzz%2Bze/duVq1a9UgiAYZWjRkzZhAVFUV6ejqfffYZf/75J6%2B//nq%2B6iXJhCg0oqOj8fX15fvvvycnJ4ekpCQiIyNxcXF5ZlknJyfS0tK4cOECd%2B/eZd68eeh0OmJjY9Hr9fj4%2BHD%2B/HmOHj1KRkYGn3/%2BOTExMcbyvXv3Zs%2BePRw9epSsrCxOnDhB586d%2BfXXX5/5v4sXL05wcDDLly8nKyuL77//nt27dxsHjTs5OXHx4kXu3bvH77//zpo1ayhZsqTJ/xdCCCHE3%2B%2BfNDUswMyZMylZsiStW7fGz8%2BPunXrEvjAwKBr164Zx2xu376dpKQkWrVqZXJTutxeGOPHj8fHx4e33noLT09PwsLCWL9%2BvUnX67yQbk6i0KhQoQKzZ89m9uzZ3Lx5E1tbW3x8fBgzZswzy7q7u9OvXz/69%2B%2BPtbU1I0eOZPLkyYwcOZLAwEAWLVpEUFAQ06dPJzExkS5dutC%2BfXtjy0PTpk2ZOHEiM2fO5Pbt21SqVInp06dTv379Z/5vc3NzFi1axLRp09i0aRMVKlRg3rx5xibI4cOHExgYSOPGjalWrRpz587F0dGRWbNmGWdzEkIIIYR4lpIlS/Lpp58%2Bcf2RI0eMv8%2BZM%2BeRiXEeZGlpyeTJk5k8ebJSncz0BRkWLoT4R9JkFiatZnPKUZwmCAxTieh0hhGaih9lwavVplosX94wvWxoqPoA7JEj1cqDdrM53bunVt7c3DC9bGqqNlfstJjNqWRJSEp68QOwLSygVClITFTfLzt7xfk53d0N08t6eGgyAPvy72rvR50OnJ3h2jX1Adh%2Bfmrla9Y0TC/bo4f6AOyIU4pvKDMzsLKCtDRtZnOyUpuzWcvZnF7UrTOex/1l799iqsiQlgkhhBBCCFEk/ZMGYBdWkkwI8QwNGzZ86rSs%2B/btw8nJ6W%2BskRBCCCFE4SDJhBDPcOrUqRddBSGEEEI8B9IyoU5mcxJCCCGEEEIUiLRMCCGEEEKIIklaJtRJMiGEEEIIIYokSSbUSTcnIYQQQgghRIFIy4QQQgghhCiSpGVCnbRMCCGEEEIIIQpEWiaEKEKKp95VC2BuDsVLUjxNg9sHR0aqlQfDrZVr14bffjPcYlnBqOKn1epSrBzQnd7FtkPx20qhJk4erlS%2BYkUYOxaCg%2BHmTaVQfDxdgzv2YoWNeRqYqd8q99INtTv2Wloa7oD911/wlNvH5En1m9%2BqBbC1BU9PSl06CcnJSqE0ueM0cG3XaeU7TgNUddXmjtzO/up35I5QvaW8szPwPtsazAWHa2qx7nygVr54ccMdsBMTITNTLRYQbV7weyQVKwbly8Pt25CVpVwVXnpJPUZBSMuEOkkmhBBCCCFEkSTJhDrp5iSEEEIIIYQoEGmZEEIIIYQQRZK0TKiTlgkhhBBCCCFEgUjLhBBCCCGEKJKkZUKdJBNCCCGEEKJIkmRCnXRzEkIIIYQQQhSItEwIIYQQQogiSVom1EnLhBBCCCGEEKJApGVCCCGEEEIUSdIyoU6SCSGEEEIIUSRJMqFOujkJIYQQQgghCkRaJoQQQgghRJEkLRPqpGVCaOrGjRu4ublx5coVAKpXr86xY8cA8PX1ZcuWLc%2BMMWDAAObPn69ZnQIDA5k0aZJm8YQQQgghhIEkEyJfnpQQbNmyBV9fX5ycnAgPD%2Bfll19%2BAbV7/nx9ffHx8SE1NdVk%2BU8//YSvr%2B8LqpUQQgghCiInR/ufokaSCSHyKSMjg5CQkBddDSGEEEIokmRCnYyZEJq6fv06rVu3Zs%2BePVStWvWJ2/3666/Mnj2b3377DZ1OR5s2bZgyZQpWVlYAZGdnM3XqVMLCwrC0tGTKlCl07NgRgPDwcObOnUtkZCQ6nY62bdsSFBRE8eLFAfjyyy9ZsWIFd%2B/exc/Pj5yH3tmbNm3i888/5%2BbNm1SqVInAwEDatGmT530cPXo08%2BfPp3v37k9sgYmOjmbGjBmcPn2arKwsfHx8mDZtGmXKlAHg1KlTfPLJJ/z222%2BUKFGC7t27M3bsWMzNzVm6dCnnz5/Hw8OD9evXk5GRgb%2B/P0FBQXmuI0BsbCxxcXEmy8pbW%2BNQvny%2B4pgwNzd9VGFjox7j/uvF%2BKiiXDm18vefW%2BOjgoppauVzn2KVp9rIzEyb8qpx7rO0VCuv05k%2BKrG1VSuf%2Bx7Q4L2guj/3Pz6Nj8rc3dXK16hh%2BqjC2VmtvKOj6aMK1QNcrJjpo6JiCm9LLauSlaUeQ7w4kkyIF%2BK9995jyJAhdO/endu3bxMQEEBoaChvvvkmAGFhYcyZM4egoCCCg4OZPn06r732GsWKFSMwMBA/Pz9iOkDtAAAgAElEQVT%2B9a9/ERMTQ58%2BfXB1dWXAgAFERUUxdepUgoOD8fHxYffu3cyaNYv27dsDcODAAYKDg1mzZg01atTgyJEjjBs3jgMHDlCxYsU81d3V1ZVevXoxa9Ys1q5d%2B9htAgICcHV15fDhw6SlpTF27FimTZvG4sWLuX37NoMHD%2Ba9996jZ8%2Be/P777wwdOhQHBwf69esHwOnTp6lbty7ffvstP//8M2%2B99RZ%2Bfn7UrVs3z8c4NDSU4OBgk2Wj3n6b0WPG5DnGE5UooR6jdm31GLmekrjmmVb1ad1aOcRYDaoB0LevFlE0SNRAPQu4r0oVTcKQx7f701Xx1CAImrz2FE%2BXjSpU0CjQ6dPaxNm8WZs4Whg06EXX4L/s7TUJo8X1Bjs79Ri3bqnHKKii2JKgNUkmRL7NmjWLOXPmmCzLycnBMR9XbRITE7GxscHc3BwHBwe%2B/PJLzB%2B42u3h4UHz5s0BaN%2B%2BPStXruTOnTs4ODiwc%2BdOdDodFhYWVKxYEU9PT86dOwfAoUOHqFWrlrGloUePHmzYsMEYd9u2bfTo0YM6deoA8Nprr9GgQQPCwsIYNmxYnus/evRo2rdvz8GDB2nbtq3JugsXLhAREcHKlSuxtbXF1taWYcOG8fbbb5ORkUFYWBgVK1Y0Jg61atXC39%2BfvXv3GpdZWFgwfPhwzM3N8fb2xt7ensuXL%2Bcrmejdu/cj4zjKW1tDUlKeYzzC3NyQSKSkqH8C//mnWnkwtEhUrQqXL0Oa4uX8ixfVypcpY0gkDh%2BGhASlUIuvd1cqX768IZHYvBkeapzKt7HDFY%2BrmZkhkUhPB71eLRZwNVotudHpDInEzZuQkaFWlypxJ9UC2NgYEomICHhoHFZ%2BXaugltgUL25IJKKjITNTKRQAzv4eagFq1DC8gPv2VX9v9uypVt7R0ZBIfPYZxMSoxRo8WK18sWKGROLOHU0u58eZOShVxc4O4uOlZaGok2RC5FtQUBBvvPGGybItW7awevXqPMd45513mDx5MmvXrqVZs2b4%2B/ubdIuqVKmS8XfL%2B1c0M%2B5/8584cYJly5Zx9epVsrKyyMrKMrY8xMTEmJQFqPLApcw///yT48ePmyQYer0eV1fXPNcdwNbWlgkTJjB37lxj0pPr%2BvXrlC5dmvIP9DFxcXEhMzOTmJgYrl%2B//kgXsMqVK7N3717j3xUrVjRJrqytrUnL58myg4MDDg4PfVHcvavNZRgtOoYqnjyZSEtTj3f7tjZ1SUhQjnXzpjZViYvTIJYGCYAxjgax0tM1qAuGREI5VnKyJnUhNVU5lmpilCszU6NYZ85oEARDIqEaq3FjbeoSEwPXrqnF0CJTA8PZuwaxsjTosZqV9c9OJqRlQp0MwBYvRM%2BePTl69Cj9%2BvXj999/p2vXrhw6dMi43uwJ/asvX77M2LFjef311/nxxx8JDw%2Bnc%2BfOxvUZGRlkPfSp9uCYCSsrK8aPH094eLjx59y5c0yZMiXf%2B9C1a1ccHR1ZuXKlyfKMp3wTm5mZPXH9g/tsrsWYBCGEEEI8lQzAVidnLOKFiI%2BPx87Oju7duxMSEsLw4cPZtm3bM8tduHABnU7HwIEDsbKyQq/Xc%2BHCBeN6BwcHoqOjTcpcvnzZ%2BLuLiwuXLl0yWX/z5k30BbxiOnXqVNavX8%2B1B65WOTs7c/fuXW4/cHU6KioKS0tLHB0dcXFxISoqyiROVFQUzqqDBIUQQggh/maSTIi/XXR0NL6%2Bvnz//ffk5OSQlJREZGQkLi4uzyzr5OREWloaFy5c4O7du8ybNw%2BdTkdsbCx6vR4fHx/Onz/P0aNHycjI4PPPPyfmgT6uvXv3Zs%2BePRw9epSsrCxOnDhB586d%2BfXXXwu0LzVr1qRr164sWrTIuMzNzY2qVauyYMECUlNTiYmJYfny5XTq1InixYvToUMHrl27RmhoKFlZWZw9e5YdO3bw%2BuuvF6gOQgghhCgYaZlQJ8mE%2BNtVqFCB2bNnM3v2bNzd3Wnfvj0lSpRgTB5mGXJ3d6dfv37079%2BfTp064eTkxOTJk4mMjCQwMJB69eoRFBTE9OnTady4MZGRkcbxFABNmzZl4sSJzJw5Ew8PD2bOnMn06dOpX79%2Bgfdn3LhxJl2rzMzMCAkJITY2lpYtW9KrVy/q1avH1KlTAUNCFBwcTGhoKJ6enrz77ruMHTuWrl27FrgOQgghhMg/SSbUmekL2r9DCPHPc/euWnlzcyhZ0jAjlOonZmSkWnnQdDYc5aksy5WD7t1h%2B3blAdgTo4Yrla9YEcaOhcWL1Qdgfzz9nloAMzPDrFtpaZoMwL70p7VSeUtLw/SyV6%2BqD8CufvNbtQC2tuDpCSdPKg/AvuzSSqm8Tme4HcO1a9oMwK7qqnhfEXd3w3vSw0N9APbIkWrlnZ3h/fdh7lz1AdgffKBWvnhxcHCA2FhNBmDfMncqcNlixQwzx8XFaTMA%2B6WX1GMUxIIF2sccP177mIWZzOYkhBBCCCGKpKLYkqA1SSaEuK9hw4akP%2BVS5b59%2B3ByKvhVHCGEEEKI/zWSTAhx36lTp150FYQQQgjxN5KWif9n787DoirbB45/hx1FRFDcQV/3LURFBRVRtNxRXzc0s1zKLRXfsjIXxK1MUgOXNHs1NQPNpcwlcF9Tf2oigiaWIirggiyyDvP7g5jXEVTkDAPG/bkuroFz5rnnnuHMzLnP8zznKCfFhBBCCCGEKJWkmFBOzuYkhBBCCCGEKBQpJoQQQgghRKn0qp0aNiEhgSlTpuDm5kb79u359NNPSUtLy/e%2B27Zto2HDhjRr1kzn5%2BLFi38/92yWLFmCp6cnLi4ujBo1SucivAUlxYQQQgghhBCvgJkzZ5KamsquXbv48ccfiYqKYvHixc%2B8v4uLC2FhYTo/r732GgCbNm3i559/ZvXq1Rw8eJBatWoxYcIEXvaqEVJMCCGEEEKIUulV6pm4d%2B8eoaGh%2BPj4YGtrS%2BXKlRk/fjw//vgjmYW47khQUBBvv/02derUwcrKCh8fH6Kiovj9999fKo4UE0IIIYQQolR6lYqJiIgIjI2NadCggXZZkyZNePz4MdevX8%2B3zZ07d3jnnXdwcXHB09OTnTt3ApCWlsa1a9do3Lix9r5WVlY4OjoSFhb2UnnJ2ZyEKE3MzJS1V/19VVtTU%2BVXM3ZwUNYeci7BClClivJLsAYGKmvv6JhzBezff4cbNxSF8hii7ArY1tY5t61aQWKiolDKX1ejv49ZqdV6%2BZatVUtZ%2B9xNuFo1PVyQO81WWXvLv6/mXb684vdmnz7KUmnUCLZuBR8fiIhQFgsgXB9XnQYYOBDatlUWa%2BVKZe2dnXOugL1li/KrcS9Zoqx97gZsY6OXK8qXU3AR7dy3dtmyckakp8XFxREfH6%2BzrFKlStjb2yuKm5CQgJWVFSrV/64wX758eQAePnyY5/62trbUqlWLqVOnUrduXUJCQpg2bRr29vb861//QqPRaNs/GS%2B/WM8jxYQQQgghhCiViqIQCgoKIvCpA1QTJ07k/ffff2HbnTt3Mm3atHzX%2Bfj4vNR8Bg8PDzw8PLR/9%2BzZk5CQELZt28YHH3wA8NLzI/IjxYQQQgghhBB6MnjwYDp37qyzrFKlSgVq6%2BXlhZeXV77rjh8/TnJyMmq1GmNjYyCntwLAzs6uQPGrV6/OpUuXsLGxwcjISNs%2BV0JCQoFj5ZJiQgghhBBClEpF0TNhb2%2BveEhTfho1aoRGoyEyMpImTZoAEBYWhrW1NbVr185z/82bN1O%2BfHl69OihXRYVFUXNmjUxNzenXr16hIeH07p1awASExO5efOm9mxPBSUTsIUQQgghRKn0Kk3AtrW15Y033mDp0qU8ePCAu3fvsnz5cgYMGIDJ33MIR4wYwe7duwHIyMhg7ty5hIWFkZmZya5duzhy5AhDhgwBwNvbm%2B%2B%2B%2B46oqCiSk5NZvHgxjRo1olmzZi%2BVl/RMCCGEEEII8Qrw8/Nj9uzZeHp6YmpqSq9evfDx8dGuj46O5tGjRwC89dZbpKSkMHnyZOLj46lRowbLly%2BnadOmAAwZMoT4%2BHiGDx9OSkoKbdq0yTPXoyCkmBBCCCGEEKXSq3YmqnLlyvHll18%2Bc/2BAwe0v6tUKsaPH8/48ePzva9KpWLSpElMmjRJUU4yzEkIIYQQQghRKNIzIYQQQgghSqVXrWeiJJJiQgghhBBClEpSTCgnw5yEEEIIIYQQhSI9E0IIIYQQolSSngnlpGdCCCGEEEIIUShSTAiDiImJoVmzZvz5558ANGjQgCNHjgDQuXNnNm/e/MIYw4cPZ/HixXrLycfHh48//lhv8YQQQgjxanmVLlpXUkkxIfTiWQXB5s2b6dy5M9WrVycsLCzfy73/E0RGRjJixAhatmyJm5sbU6ZMIT4%2BvrjTEkIIIcRzSDGhnBQTQiiUkZHByJEjad26NSdPnmTXrl3cv38fX1/f4k5NCCGEEKJISTEhDOLWrVs0aNCAqKio597v999/Z9CgQTg7O9OmTRs%2B/fRT0tLStOvVajWzZs2iRYsWuLq6snv3bu26sLAwhg4dSqtWrXBzc2P27NlkZmZq1wcHB9O5c2datmzJnDlzyH7q8MHGjRvp3r07Tk5O9OzZk9DQ0AI9t9TUVHx8fHjvvfcwMzPD1taWrl278scffwCwbds2unbtypYtW%2BjQoQPNmzdn1qxZZGVlARAQEMDYsWMJCAjAxcWF9u3bExoayrZt2%2BjYsSMuLi6sXLmyQLkIIYQQouCkZ0I5OZuTKFGmTZvG6NGj%2Bfe//829e/cYP348QUFBjBgxAoBdu3axYMECZsyYQWBgIL6%2Bvrz%2B%2BuuYmJjg4%2BNDnz592LBhA7GxsQwZMoS6desyfPhwrl%2B/zqxZswgMDMTd3Z2ffvqJefPm0a1bNwB%2B/fVXAgMD%2Beabb2jYsCEHDhxgypQp/Prrr1SrVu25OZcvX56BAwdq/75%2B/Trbt2%2Bne/fu2mWxsbGEhYXx66%2B/cvv2bUaMGEGdOnW0z%2Bv8%2BfN07NiR48ePM3fuXHx9fenatSu//vore/bsYfr06QwaNAg7O7sCv5ZxcXF5hlpVsrbGvlKlAsfIQ6XSvVXCRA8fP8bGurdKODoqa1%2B1qu6tAtbWytqXLat7q4iRwmNOue2Vxvmb0k1Pn5swlpbK2pub694q0KiRsva5I1D1NhK1Zk1l7StX1r1VwtlZWfuGDXVvlShRG7Cyt%2BWTqSh9e5fGHfB/EikmhN7MmzePBQsW6CzLzs6m8kt8GSQmJlKmTBmMjIywt7cnODgYoyc%2BpVq0aEGHDh0A6NatG19//TUPHjzA3t6eHTt2YGZmhrGxMdWqVcPFxYVLly4BEBoaSuPGjenSpQsAAwYMYP369dq4W7duZcCAATRt2hSA119/nZYtW7Jr1y7efffdAuUeExPDG2%2B8QVZWFoMGDWLSpEnadenp6UyZMgVLS0vq1KlDz549OXTokLaYMDU1xdvbG4COHTsSHBzMu%2B%2B%2Bi7m5OZ07d0atVhMdHf1SxURQUBCBgYE6yyZOmMD7T%2BRVaHrY%2BcHCQnmMXDY2ymP4%2BSmPATBunOIQ7fSQBkDz5vqIoo%2BKBOU73n8z00sUMDXVQ5D69fUQBOWFLLB1qx7yAL74Qj9x4BP9hBk5UnmMT/SUy/ff6yeOPuhlA4YyenhD6eOtnZysPEZhSSGjnBQTQm9mzJih3SHOtXnzZtasWVPgGFOnTmX69OmsXbuW9u3b4%2BXlRZ06dbTra9Soof3d/O8d2oyMDABOnTrF8uXL%2Beuvv8jKyiIrK0vb8xAbG6vTFqBWrVra32/evMnx48d1CgyNRkPdunULnHvuJPMbN24wa9Yspk2bhr%2B/P5DTe2Fra6u9b7Vq1Th27Jj27ypVqmh/NzPL%2BXTPLcJyn2d6enqBcwEYPHgwnTt31llWydoanhg29tJUqpxCIj0dNJrCxwFISVHWHnJ6JGxsICEB1GplsZYtU9a%2BatWcQmLlSrhzR1Go428oK2zKls0pJC5cUP4yt2uuMICRUc7eRmqqXr61M0yVFTcqVc5%2BWGam8k3Y7K%2BrygKYm%2BcUEjdu5LynFBgwXVlhU7t2TiHx4Yfw90n3FNnacqGyAJUr5xQS334LsbHKYm3Zoqx9w4Y5hcTQoRAZqSzWqVPK2utzAwYeZxW%2BmlCp/vfW1kMqxUaKCeWkmBAlysCBA%2BnSpQsHDhxg//799O3blyVLlmh7FFTP6NqNiopi8uTJfPTRRwwaNAgLCws%2B/PBD7byEjIwM7e%2B5npwzYWFhwX/%2B8x9GKjwKplKpqFWrFj4%2BPgwZMoRPP/0UyJnr8SSNRqPzXIzy6SPOb9nLsLe3x97eXnehvj71NRrlcZ76fyiiViuPd%2BOGfnK5c0dxrMRE/aSSkqKHWPr6ptXTYGJ97bToYxMmNVUvuZCerjhWRIR%2BUvnzTz3Fso/WQxByColohbHOn9dPLpGRymOVqA1Y2Vsy9ytKo5Ed8tJOJmCLEuXhw4dUqFCBf//736xYsYL33nuPrQXov4%2BIiMDMzIy33noLCwsLNBoNEU98I9rb23P37l2dNk9OBndwcODKlSs662/fvo2mAB/WJ0%2Be5I033tApTnILAdO/u6KTk5N58OCBTuyXGf4lhBBCCP2TCdjKSTEhSoy7d%2B/SuXNnjh07RnZ2NklJSVy9ehUHB4cXtq1evTppaWlERETw6NEjvvjiC8zMzIiLi0Oj0eDu7s7ly5c5dOgQGRkZbNq0idgnus4HDx7M7t27OXToEFlZWZw6dYpevXrx%2B%2B%2B/v/CxmzZtSnJyMl988QWpqak8ePCAgIAAWrVqRbly5YCcoUvLly8nLS2Na9eu8csvv%2BQZgiSEEEII8aqRYU6ixKhSpQrz589n/vz53L59GysrK9zd3XUmMj%2BLs7Mzw4YN480338TS0pJx48Yxffp0xo0bh4%2BPD0uXLmXGjBn4%2BvqSmJhI79696datm7bnoV27dnz00Uf4%2Bflx7949atSoga%2BvL80LMIO1XLlyfPvtt8ybN4%2B2bdtSpkwZ2rZty/z587X3sba2pn79%2BnTt2pWkpCT69OnDkCFDCv9iCSGEEEKx0tiToG8qTUHGcQghCm3btm34%2B/tz/Pjx4k5F%2BRhvlSrnLExpacrH6%2BpjYoCJCdjZwf37yudMTJumrL2jY84ZoWbNUjxnYs%2BQ9S%2B%2B03NYW0O7dnD8uPKXuXv7JGUBjIxyZoSnpOjlWzvdrJyi9ioVmJlBRobyTdg88sU9l89laZlzRqirVxW/N5sMdVLUvlGjnDNCDRignzkT4R3HKwtQs2bOWZgWLlQ%2BZ0LpdXqcneHcOWjRQvmcCSUnwAD9bsBAcmbhz8xnZARlysDjx/rZIbeyUh6jMEaN0n/MtWv1H7Mkk2FOQgghhBBCiEKRYU5CvECrVq2ee1rWvXv3Ur16dQNmJIQQQgh9kGFOykkxIcQLnD17VlH7/v37079/fz1lI4QQQghRckgxIYQQQgghSiXpmVBOigkhhBBCCFEqSTGhnEzAFkIIIYQQQhSK9EwIIYQQQohSSXomlJNiQgghhBBClEpSTCgnw5yEEEIIIYQQhSI9E0IIIYQQolSSngnlpJgQohTRWFgqjqECNOYWyuMsWKA4BlWrwvjxEBQEd%2B4oCvXoq/WK2hsZQTkg6UM/xV9OVheVtbe0/N%2Bt4i/K69eVJ1O/PsTEQGqqwmTAvGlTxTHAGDNjtfIwR44oa1%2BpUs5rc%2B4cxMcrChV%2Btr6yXFQqwIKtG9NAo1EWC%2BDBp8ram5rm3I4aBZmZymItWaKsvUqVc3vqlPLXxkLhZ6ezc8720rYtnD%2BvLBagTij888l9KdRq2SEv7aSYEEIIIYQQpZIUQspJMSGEEEIIIUolKSaUkwnYQgghhBBCiEKRngkhhBBCCFEqSc%2BEctIzIYQQQgghhCgU6ZkQQgghhBClkvRMKCfFhBBCCCGEKJWkmFBOhjkJIYQQQgghCkV6JoQQQgghRKkkPRPKSc%2BEEEIIIYQQolCkZ0IIIYQQQpRK0jOhnBQTQgghhBCiVJJiQjkZ5iSEEEIIIYQolGIvJmbMmMG0adOKOw0d27Zto127dnqNuWLFCt588029xizNoqOjadasGdHR0cWdyjONGDGCwMDA4k5DCCGEEM%2BQna3/n6KUkJDAlClTcHNzo3379nz66aekpaXle98ZM2bQrFkznZ/GjRvzySefAPDxxx/TuHFjnfWtWrV66ZxeaphT586dGTNmDN7e3jrLN2/ezJo1azhw4MBLJzBv3ryXblMYJ0%2BexMrKimbNmuk17m%2B//cZbb72FmZkZAMbGxtSqVYtu3brx9ttvY2FhAcD48eMZP378C%2BOp1Wq%2B%2B%2B473nnnHb3mqU9nzpxh5MiR2r8zMjIwNTVFpVIB4OXlpfj/umXLFmbMmKF9XZ80d%2B5c%2BvbtS1hYmKLHeJ4lS5bw9ddfY2pqCoBKpaJq1ar8%2B9//ZtSoURgbG78wxvr16wv8eGFhYSQnJ%2BPq6lronIUQQgjxzzZz5kwyMjLYtWsXmZmZTJ48mcWLFzNjxow89503b57O/lhWVhZ9%2B/alW7du2mXjxo3j/fffV5RTqZkzsW7dOjw8PPReTOQ6e/Ys5ubmJCYmEh4ezpIlS9i3bx8bN26kbNmyBY5z%2BfJlvvnmmxJdTLi4uOjsyDdo0IAVK1bg7u6u18epXLkyR44c0WvMl%2BHs7MzmzZuBnCIvLCyMiRMnYmxszKhRo/T6WFu2bKFChQpSTAghhBAG9CrNmbh37x6hoaFs374dW1tbIOdg9eTJk/noo4%2B0B0CfZf369VSrVo2OHTvqNS%2B9FxN3795lzpw5nDt3jqysLNzd3Zk9ezY2Njb89ttvjB07lsmTJ/PVV1%2Bxdu1agoKCSE9PZ8mSJYwcOZIzZ85oY6nVaqpUqaLt8QgNDeWrr77i5s2b2Nra8vbbb/PWW28BOV01ZcuWxcTEhO3bt2NkZMSoUaMYM2YMY8eO5dChQxw7doy9e/eyfv16wsLCWLhwIVevXsXMzIyuXbsyY8aMF/4jXsTa2hpXV1ecnJzw8vJi7dq1TJo0iYCAAI4ePUpwcDCpqan4%2Bvpy5MgR0tLSaNCgATNmzCA7Oxtvb2%2BysrJo1qwZa9asoU2bNvj7%2B/Pzzz%2BTmJhIrVq1mD59Oi4uLgAMHz6cdu3aERUVxf79%2BylbtiwffPABXl5eQM5woNmzZ3P%2B/HlsbGx45513tK9ZTEwMc%2BfO5fz582RnZ9OpUydmzZqFlZWVotcg1%2B3bt/Hz8%2BP8%2BfOo1Wo6duzIrFmzKF%2B%2BvOLYN27c4PXXX%2BfXX3/F0dERd3d3hg4dSnBwMJ06dWLmzJmEh4fz2WefERERgampKX369OHDDz/ExOTlN3tjY2OaN2/OkCFDCAkJ0RYT%2B/btIzAwkJs3b2JnZ8eoUaMYNmwYAN7e3rRu3RofHx%2BWLFnC9evXadq0KevXryczM5P%2B/fvzySefMHv2bIKDgzEyMmLv3r3s27ePLVu2sGbNGmJjY7Gzs%2BOtt97i7bfffqmc4%2BLiiI%2BP11lWsWIl7O3tX/r5F4mqVZXHqFhR91YBI4WDPnPbK40D8BLHH/Jlaal7q5dghWVurnv7T1KpkrL2FSro3irxd8%2Bw4vZK4%2BRS%2BF1K7ud0IT6v8yhJr42zs7L2DRvq3iqk5PNKn595xblDXxSPnd/3b6VKyr9/IyIiMDY2pkGDBtplTZo04fHjx1y/fl1n%2BdMSExNZtWoV33//vc7yU6dOsX//fm7cuEGdOnXw9fWladOmL5WX3ouJ8ePHU7duXfbv309aWhqTJ09m9uzZLFu2DIDMzExu3LjBiRMnMDc3JygoSNv222%2B/1f6enJxM//79GTBgAACRkZFMnjyZZcuW0bFjR86ePcvYsWNxdHTUVli7du3i448/5vjx4/z000/MnDkTLy8vVq1alWeIlo%2BPD3369GHDhg3ExsYyZMgQ6taty/Dhw/XyOpQpU4ZBgwaxfft2Jk2apLNu/fr13Lt3j5CQEMzMzFizZg0zZ85k%2B/btzJ07F39/f44fPw7Ajh072LFjB1u3bqVSpUqsXLmSSZMmcezYMe1Qm02bNrFgwQIWLFjAqlWr8PPzo0ePHpiamjJx4kRat27N8uXL%2Beuvvxg2bBh16tTBzc2N8ePH06JFC5YsWcLjx4%2BZOnUqn3/%2BOXPnzlX8/DUaDePGjaNx48bs37%2Bf1NRU3n//ffz8/PD391ccPz%2B//PIL69evp0aNGqSkpDBmzBjeeecdvv32W%2B7cucO4ceOoWLEiY8aMKfRjqNVqjP7%2B5AwPD2fq1KkEBATQoUMHzpw5o90m27dvn6ft6dOnadasGYcOHeL06dOMGjWKPn36MGfOHK5evaotPG7dusWCBQsIDg6mXr16XLx4kdGjR%2BPq6vrcD4qnBQUF5ZmzMWHCRCZNUtadCXra3yjAsL8CGzRIcYhyekgDlBcCAC1aKI8B0KiRPqLU10cQcHTUTxx9KcBQxRcaMkR5DIA33tBPHH3QV9H39xBfxf4%2B%2BloiKC2QAM6dUx4D4KkdwsLSx%2BeePj7zHj1SHqMkye/7d%2BLEiYqHEyUkJGBlZaUdVg5oD9A%2BfPjwuW03btyIi4sL9erV0y6rWbMmRkZGTJ48mbJlyxIYGMjIkSPZt28fFV7iIMdLFxPz5s1jwYIFOsuys7OpXLkyERERhIeH8/XXX2NlZYWVlRXvvvsuEyZMICMjA8gpJoYOHaqdS/Ass2fPxsHBQbvj9%2BOPP%2BLq6kqXLl0AcHV1xcPDg927d2uLiRo1atCvXz8AevTowfTp0/nrr7/yrQR37NiBmZkZxsbGVKtWDRcXFy5duvSyL8dz1a5dm1u3buVZnpiYiKmpKRYWFpiYmDx3PkXv3r3x9PSkXLmct3zPnj0JCAjg9u3b1KxZE8gZjtOhQwcAunfvTmBgIHFxcTx69IgrV66wfv16LC0tadSoEQ0gyRIAACAASURBVIGBgVSuXJmwsDD%2B%2BOMPNm/ejKWlJZaWlrz//vuMGjUKPz8/nQ21MC5dusSVK1f473//q90WxowZw5QpU8jMzFTcA5QfDw8P7Wty8OBBTExMtNuPg4MDI0eOZN26dYUqJnKHOW3ZsoWxY8cCOdtkhw4d6Ny5MwBubm64u7uze/fufIsJMzMzxowZg0qlon379pQvX56oqCiaNGmic7/k5GQ0Gg1lypQB4LXXXuPUqVPaIqagBg8erM0tV8WKldBoXipMHioVimMAqFauUB6kYsWcQiI4GO7dUxQqabiy4sbIKOdLNSVF%2BZGuP/5Q1t7SMqeQiIiA1FRlsVpYXVUWwNw8p5C4cQPS05XFAqhTR3kMY2NQq5XH2bJFWfsKFXIKiX374AU7Ai/Ut6%2By9ipVzv8qPV0/b/DERGXtTUxyCokHDyArS1ksGxtl7VWqnEIiM1P5a9O2rbL2DRvmFBJDh0JkpLJYQNLhwhc3%2BvzMK05FkXt%2B37%2BVCtiTuXPnzmeemMjHxwdNIbZBtVrNpk2b8hzMnTBhgs7fH374Ibt27SI0NJSBAwcWOP5LFxMzZsx45gTsW7duUb58eZ0XzMHBgczMTGJjY7XLqlWr9tzH2Lp1K6dPn2bnzp3andpbt25R56kvEUdHR849UeXXqFFD%2B7vl313zz5rhfurUKe3R%2BqysLLKysnQmpOiDWq3Od6Lu0KFDGTVqFB07dqRDhw506dIFT0/PfGOkpqayYMECjhw5wqMnSvfc4gx0n3dukZaWlsbNmzexsrLC5okPUjc3NyDnKL5araZNmzZ5cn748KF2LF5h3bp1iwoVKujEcXR0JCMjg3v37lG1AENcYmNj853jsmnTpnyHSj25Xd28eZO4uDid9k/uoBfE%2BfPnte2NjIyoUaMGo0eP1g5junXrFnXr1tVp4%2BDgQHh4eL7xqlevrlOkWVhYkJ7PzlWDBg3o0aMH3bp1o3Xr1nTo0IG%2Bffvq/B8Lwt7ePk8hrY99BL25c0d/se7dUxxPX18o%2BjibR0qKfnJJTdVDLGOF1Uiu9HTllU1J89QwhkJ7%2BFB5LH29uTUa/cTKzFQeA3IKCaWxStJrc/68fnKJjNRLLH187hniDEavmvy%2BfwvKy8tLO1T9acePHyc5OVln/zIhIQEAOzu7Z8Y8c%2BYMGRkZLzxTk7GxMVWrViUuLu6lctbrMKcnd3Cf9uRO1PPGrEdFRbFgwQJWrlypsyP6rNhPxi3okduoqCjtZJVBgwZhYWHBhx9%2BSJbSox9PuXz5MrVr186zvEaNGuzevZvffvuNAwcOMGvWLH766Se%2B%2BuqrPPedM2cOV65cYdOmTTg6OhIdHU3Xrl117vOs521kZET2M97h5ubmlClThvP6%2BmB7SkG3hed53gTsGzdu5Fn25HZlYWFBw4YN2bFjR4EeKz9PTsDOT0G2yYIsz%2B9%2BCxYsYMyYMezfv59ffvmF1atXs2XLFqpXr16gGEIIIYR4sVepEGrUqBEajYbIyEjtqIawsDCsra3z3d/MtX//ftq2bauzn6TRaPjss8/o168fDf%2Beg5ORkcHNmze1ozwKSq/XmahZsyaPHj3i3hPDDa5fv465uTmVK1d%2BYfu0tDSmTJnCyJEj8xwxd3Bw4Pr16zrLrl%2B//tJPGHImsJiZmfHWW29hYWGBRqMhIiLipeM8z4MHD/j%2B%2B%2B/p3bt3nnUpKSmo1Wrc3NyYMWMGW7ZsYd%2B%2BffmOd7t48SJ9%2BvShVq1aqFSqZx71zk/NmjVJSUnRqTBDQ0M5ffo0Dg4OPH78WOc6DcnJyS8cc1dQDg4OPHz4kAcPHmiXXb9%2BHUtLywJ39Sl9/Bs3bpD6xJHQBw8ekKKvQ748e5t0cHBQFFetVpOYmEjt2rUZPXo0W7ZsoVatWoSGhiqKK4QQQghdr9J1JmxtbXnjjTdYunQpDx484O7duyxfvpwBAwZoC4URI0awe/dunXYRERE6o1gg58DlrVu3mDNnDrGxsaSkpLB48WJMTU21UwoKSq/FRLNmzahTpw7%2B/v48fvyY2NhYVq5cSc%2BePQs0Rn7BggXY2trmO3%2BgT58%2BHD9%2BnIMHD5KVlcXRo0c5dOgQfQs4TtTc3JybN2%2BSlJRE9erVSUtLIyIigkePHvHFF19gZmZGXFxcocaiPSk7O5vz588zevRo6tatqx0S86RJkybx%2Beefk5ycrL2/jY0N5cuXx8LCgqSkJGJjY0lLS6NGjRqEhYWRkZHBhQsX%2BOWXXwAK1AXVqFEjGjduzNKlS0lJSeHq1avai5vUr18fZ2dn5s%2Bfz4MHD0hMTGT27Nl6u4Cgk5MTjo6O%2BPv7k5qayt27d1m1ahW9e/cu0DUalHJ3d8fa2ppFixaRnJxMXFwckyZNYsmSJXp7jD59%2BnD06FEOHz5MVlYWhw8f5ujRo8/snnweCwsLbt26xaNHj/j5558ZMmQIf/31F5BzRq7Y2FgcS9oEViGEEEIYlJ%2BfH%2BXKlcPT05M%2Bffrw2muv4ePjo10fHR2tMyweID4%2Bnor5nPVw/vz51KpVi/79%2B%2BPm5kZERATr169/qSHhoOdhTiqVihUrVjB37lw8PDywtLSkS5cufPDBBwVqHxQUhKmpKU5OTjrL9%2B7dq93x9ff3Z%2BrUqdSoUYPFixfTunXrAsUeNGgQS5cu5cSJE%2BzcuZNhw4bx5ptvYmlpybhx45g%2BfTrjxo3Dx8enUNdLeHIcWrVq1ejVqxdjxox55kXXZs2ahbu7OyqVinr16rF8%2BXKMjIxo27YtNWrUoEuXLnz%2B%2Bef85z//Ydq0abRu3RonJycWLVoE5Jw1a%2BPGjS/Ma9WqVUybNg03Nzfs7OwYP3689vn5%2B/vj5%2BeHp6cnZmZmuLq68tlnn730c8%2BPkZERK1euZO7cuXTs2BFLS0u6du1a4G1BKTMzM1auXMm8efNwc3PD2toaT09PvT5%2Bq1atmDNnDosWLdJOiF%2B6dCktW7Z86Vj9%2B/dn1qxZHDt2jJMnTxIVFcWbb75JUlISFStWZPDgwXh4eOgtdyGEEEK8WsOcAMqVK8eXX375zPX5XUB63759%2Bd7XxsaGhQsXKs5JpVF6KF4I8crQy1mY9HU2p1kzlQepWjXnFLMrViiegP3oA2WnRDYygnLlIClJ%2BZfTxYvK2pctm3N62XPnlE/A7mD9u7IAlpZQvz5cvaqfCdgvef7zfOnrbE4rFJ6RrFKlnNPL/vCD8gnYo0cra69S5ZzONS1NP2/wJ4a4FoqpKdjbQ1yc8gnYSq9Do1KBmRlkZCh/bZSeMtfZOeeN3aKFXiZgP0oo/PPR52cegB4uQVUoRXGt2JMn9R%2BzJCs1V8AWQgghhBDiSa9az0RJJMXEc%2BzZs%2Be5cwhcXFx0LrT3T7F27VqWLl36zPVeXl7Mmzev0PFXr15NQEDAM9f379%2BfOXPmFDr%2Bi8yePZtt27Y9c/3777/Pu%2B%2B%2BW2SPL4QQQoiSQYoJ5aSYeI7u3bvTvXv34k7D4EaNGsWoUaOKLP67775brDvrc%2BbMKdJiRQghhBCitJBiQgghhBBClErSM6GcXk8NK4QQQgghhCg9pGdCCCGEEEKUStIzoZwUE0IIIYQQolSSYkI5GeYkhBBCCCGEKBTpmRBCCCGEEKWS9EwoJz0TQgghhBBCiEJRaTRKrw0vhHhlxMQoa29qCvb2EBcHmZnKYlWrpqx9LpUKSsrHmL5yuXtXWXsTE6hUCeLjIStLUagkq6qK2hsZQdmykJKinyOA5UhSFkCPCT3MKqeovbExWFtDYiKo1YpCYWOjrD3o961UgjZhyin7N2FkBGXKwOPHyrdhpf9nI6Oc55OUpJ/3U3kbVeEbOzvDuXPQogWcP688mWL6HG/SRP8xw8P1H7Mkk2FOQgghhBCiVJJhTsrJMCchhBBCCCFEoUjPhBBCCCGEKJWkZ0I5KSaEEEIIIUSpJMWEcjLMSQghhBBCCFEo0jMhhBBCCCFKJemZUE56JoQQQgghhBCFIj0TQgghhBCiVJKeCeWkmBBCCCGEEKWSFBPKyTAnIYQQQgghRKFIz4QQQgghhCiVpGdCOemZEEIIIYQQQhSK9EwIIYQQQohSSXomlJNiQpQo169fZ/ny5Zw8eZKUlBTs7Ozo3LkzEydOxMbGprjT0xo5ciRnzpwBQK1Wk52djampqXb93r17qV69enGlJ4QQQogCkGJCOSkmRIkRERHBsGHD8Pb25qeffqJChQpcvXqVBQsW4O3tzfbt27GwsCjuNAH49ttvtb8HBARw9OhRgoODizEjIYQQQgjDkzkTosTw8/Ojffv2fPjhh1SsWBFjY2MaNWrEypUrad68OXFxcdy9e5dx48bRpk0bWrZsiY%2BPDwkJCSQlJdG0aVNOnz6tE7NPnz6sXr0agJMnTzJ48GCcnZ3p0KEDy5cv194vICCA9957jylTptCiRQvFz%2BXs2bM0bdqUhw8fapelpaXh7OzMsWPH%2BPjjj/nkk0/w8/OjRYsWtG3blu%2B//17nvn5%2Bfnh4eNC8eXOGDx/OtWvXFOclhBBCiP/Jztb/T2kjPROiRLh//z7nzp1jw4YNedZZWVmxcOFCAPr370/dunXZv38/aWlpTJ48mdmzZ7Ns2TLatWtHaGgorVu3BiA6OporV66wfPly7t69y/jx45k9eza9e/fm2rVrjB49GgcHB3r37g3AhQsXmDx5Mv7%2B/oqfT8uWLalcuTJ79%2B7F29sbgGPHjlG2bFlcXV3ZtWsXe/fu5ZNPPuHUqVMcOXKEiRMn0qJFCxo2bMjixYu5fPkyQUFBlC9fnq%2B%2B%2BoqJEyeyZ88eVCpVgXKIi4sjPj5eZ1klwL5SpcI/MRMT3VtRNJS%2Bvnr8PxkpPOSU215pnCciKmyuv4SMjUtMKiVOCdqEFb%2B%2BuR%2B5KpXyWBqNsvZ632acnQvftmFD3Vslzp9XHkMUG9kjECVCdHQ0ALVr137mfSIiIggPD%2Bfrr7/GysoKKysr3n33XSZMmEBGRgbdu3cnICCA6dOnAxASEsJrr71GzZo1%2Beabb6hXrx59%2B/YFoEGDBgwZMoSdO3dqiwljY2O8vb0LvLP%2BPCqVCi8vL37%2B%2BWdtMfHrr7/So0cPjP/eA6lWrRqDBg0CoEuXLjRq1IiDBw9Sv359tm3bxtKlS6lcuTIAU6ZMYePGjVy8eBEnJ6cC5RAUFERgYKDOsokTJvD%2BpEmKnx%2B2tspj6JMe/md6o49clBR8T6pQQXGIsnpIA8DSUk%2BB9JWRHhKy1kMaAFZWegqkB/p6K5WgTVhv9LcNK1dWX2/Mc%2BeUx3iiV73QivEzvDT2JOibFBOiRMjdgc9%2Bzrv61q1blC9fnkpPfEs5ODiQmZlJbGwsnp6ezJgxg8jISBo2bEhISAg9e/YE4ObNm4SFhdGsWTNtW41Go1O8VKlSRS%2BFRK6%2BffuycuVKYmJisLe359ChQ6xdu1a7/unCqVq1asTFxXH//n1SUlIYP368Tj7Z2dncuXOnwMXE4MGD6dy5s86ySgBxcYV%2BTpiY5BQSDx5AVlbh44D%2B9jZUKuWH%2B/RFX7ncu6esvYlJzl7Yw4eK/08pZZT9n4yMcnbCUlP186VdlhRlAfSYUKJa2R6dkVFOIZGcrPy1KVdOWXvQ71upBG3Cine8Var/bTJKXx%2B1Wll7I6Oc55OSop/3U7mOCob1NmyYU0gMHQqRkcqTKSZSTCgnxYQoERwcHAD4448/tEfjn5aRkfHM9iqVinLlytG%2BfXtCQ0Oxs7Pj4sWLLF26FAALCws6duzIqlWrnhnDRM9DdxwcHHBycuKXX36hSZMm2Nra6hQz6qe%2BVTQaDSqVSjvJ/IcffqBp06aFfnx7e3vs7e11F8bEQGZmoWNqZWXpJ47In9K9pyfjKIylry9a/Y0lLjkJKd0xfDIVfcUqKUrQJqx4u8sdUqTRKI9V4t5P%2BhheFBkpw5RKuX/gSE3xKqpQoQKtW7fmv//9b551qamp9O/fn0qVKvHo0SPuPXHI6/r165ibm2sLkG7dunHw4EFCQ0Np3ry5drmDgwNXr15F88Rhpfj4%2BOcWKPrQt29f9u7dy549e7TDqXLlDu3Kdfv2bapUqUK5cuWwsbHhypUrOutv3bpVpLkKIYQQpY1MwFZOiglRYnz66adcuHCBqVOncvfuXbKzs4mIiGD06NFYWFjg7OxMnTp18Pf35/Hjx8TGxrJy5Up69uypvcaDp6cn165d46effqJHjx7a2D179iQhIYEVK1aQlpZGdHQ0I0eOZP369UX6nHr06MG1a9fyLSZiYmLYsWMHmZmZhISEEBkZiYeHBwBDhgxh5cqVREVFkZmZybp16xgwYACpqalFmq8QQgghxMuQYU6ixGjYsCHBwcEEBATQr18/Hj9%2BTJUqVejVqxdjxozB1NSUFStWMHfuXDw8PLC0tKRLly588MEH2hjlypXD1dWVI0eO6Ew%2BrlChAitWrGDRokWsWrUKW1tbvLy8GDlyZJE%2BJ2trazw8PIiNjdUO5crl7u7O%2BfPnmTt3Lqampvj6%2BlK/fn0Axo8fT2JiIkOHDiUzM5NGjRqxZs0aLEvSDEAhhBDiFVcaexL0TaXRlJSZi0L8M7355pt4eXkxcOBA7bKPP/6Y9PR0lixZYthkYmKUtTc1BXv7nEncSudMVKumrH2uf%2BIE7Lt3lbU3McmZ4B4fr3jAeZJVVUXt9T5hlCRlAfSY0MMsZbOejY3B2hoSE5XPmbCxUdYe9PtWKkGbsOLJ6UZGUKYMPH6sfBvWxwTscuUgKUk/76fyNgpOOuLsnHM2qBYt9DNnopg%2Bx/Vx8oKnJSn8mHrVSM%2BEEEVEo9GwefNmYmJi8gxxEkIIIYT4J5BiQoh8tGrVivT09Geu37t3L9WrV39uDCcnJ2rWrMmyZcu0Z2gSQgghRMkhw5yUk2JCiHycPXtWcYyLFy8%2Bc91nn32mOL4QQgghRHGTYkIIIYQQQpRK0jOhnBQTQgghhBCiVJJiQjm5zoQQQgghhBCiUKSYEEIIIYQQpdKreAXssLAwunbtyqBBg1543%2B%2B%2B%2B4433niDFi1a4O3tzaVLl7Tr0tPTmTVrFu7u7rRp04ZJkybx8OHDl85HigkhhBBCCCFeAT/99BPvv/8%2Bjo6OL7zvgQMHCAgIYNGiRZw4cYJOnToxduxYHj9%2BDMCSJUsIDw8nKCiIffv2odFo%2BOSTT146JykmhBBCCCFEqfSq9Uykp6cTFBSEk5PTC%2B8bFBRE//79cXJywsLCgtGjRwNw8OBBsrKy2Lp1K%2BPHj6dq1arY2NgwZcoUDh06RGxs7EvlJBOwhRBCCCFEqVQUO/9xcXHEx8frLKtUqRL29vaKYw8cOLDA9w0PD6dHjx7av42MjGjUqBFhYWE0atSIpKQkmjRpol1fp04dLCwsCA8Pp3LlygV%2BHCkmhChNXnChvReJi4sjKCCAwYMHY68wlj7ExcURFBSUk48ePqRLTC5VqyrPJff/pDBWOUWtc3L59lt9/o%2BUZRQXF0fQt9/qJZ8Kilrn5LJ%2B/T9w%2B0XxJkxcXBwBATn5VK1a/K/N2rUl5/%2B0bp0ec9FoFOUSFBDA4L17i/11UULBS/BMAQFBBAYG6iybOHEi77//vv4f7DkSEhIoX768zrLy5cvz8OFDEhISALC2ttZZb21t/dLzJmSYkxCiwOLj4wkMDMxzxKW4lKR8JJeSnwuUrHwkl2crSflILiU/l5Jm8ODBbNu2Tedn8ODBBWq7c%2BdOGjRokO/Ptm3bXjoXzQuqpRetLwjpmRBCCCGEEEJP7O3tC91b4%2BXlhZeXl17yqFChgrYHIldCQgL16tXD1tZW%2B3fZsmW16x89eoSdnd1LPY70TAghhBBCCPEP07RpU8LDw7V/q9VqLl%2B%2BjJOTEzVr1qR8%2BfI6669evUpGRgZNmzZ9qceRYkIIIYQQQoh/gG7dunH27FkAvL292bFjBxcuXCA1NZWVK1diZmaGh4cHxsbGDBo0iFWrVnHnzh0ePnzIl19%2BSdeuXalYseJLPaaxr6%2BvbxE8FyHEP1TZsmVp3bq1TrdocSpJ%2BUguJT8XKFn5SC7PVpLykVxKfi6lxRtvvMGiRYs4ffo0d%2B/e5euvv2blypV4eXlhbW3N3Llz6d69O46Ojjg6OmJlZcX8%2BfP56quvyMjIwN/fX3umplatWvHHH3/g5%2BfHunXrqF%2B/PnPnzsXc3PylclJp9DHzQgghhBBCCFHqyDAnIYQQQgghRKFIMSGEEEIIIYQoFCkmhBBCCCGEEIUixYQQQgghhBCiUKSYEEIIIYQQQhSKFBNCCCGEEEKIQpFiQgghhBBCCFEoUkwIIYQQQgghCkWKCSGEEEIIIUShSDEhhBBCCCGEKBQpJoQQohBOnz6d7/L09HT27Nlj4GyEEMJwLl68yK%2B//qr9Oz09vRizEcVNigkhhCiEMWPG5Lv80aNHfPzxxwbOBtRqNatXr6ZHjx64uLgAkJKSgp%2BfX7F80d%2B6dcvgj5mfzp07s3TpUqKiooo7lRLt/v373L59O8%2BPKDn%2B%2BOMP7e937txhw4YNHD582KA5REVF0b17d4YPH87UqVMBiImJoVOnTly%2BfNmguYiSQ6XRaDTFnYQQomQZPnw4KpWqQPf97rvvijgbXWq1mpCQEKKiovLdSc79gisq69at45tvvuH%2B/fvY2dnlWZ%2BcnEz16tX55ZdfijSPp82fP5/Tp08zcuRIZs6cycWLF0lISGDy5MnUrl0bX19fg%2BbTqFEjmjdvTu/evenevTsVKlQw6OPn%2BvHHHwkJCeHEiRPUqVOH3r1706tXL%2Bzt7Ysln0ePHrF69Wo%2B/PBDAH744QeCg4NxcHBgxowZVKxY0aD5bN%2B%2BnYULF5KUlKSzXKPRoFKpiIiIMGg%2BAGFhYc98fw8ePNhgeTx48IBbt27lm0duwW4o69evZ8WKFfz22288evSInj174ujoSGxsLN7e3owaNcogeYwYMYJmzZoxadIkWrVqxcWLF4Gcz8X9%2B/ezYcMGg%2BQhShYpJoQQefj7%2B2t/T01NZefOnbRs2ZLatWuTnZ3NtWvXuHjxIkOHDsXHx8eguU2dOpXQ0FAaNGiAhYWFzjqVSlXkxU12djbh4eF4e3szd%2B7cPOvNzc1xdXU1%2BM6zm5sbW7ZsoXr16jg5OfH7778DEB8fT79%2B/Th27JhB84mOjiYkJIRff/2Vy5cv4%2BbmRp8%2BffD09MTc3NyguUBOL82hQ4cIDQ3l6NGjNG3alD59%2BtC9e3csLS0NlsfkyZNJS0vj66%2B/Jjw8nGHDhjFjxgwuX77M/fv3WbZsmcFyAWjXrh1vvvkmnTp1wszMLM/6f/3rXwbNZ/78%2BWzYsAFbW9t839/79%2B83SB6rVq0iICAAtVqdZ11xFFmenp4sXbqUZs2asW7dOnbv3k1wcDDR0dGMHDmSkJAQg%2BTh7OzMb7/9hpmZmc7nTFZWFm3btuXs2bMGyUOUMBohhHgOHx8fzeHDh/MsDwkJ0UyePNng%2BTg7O2v%2B%2BOMPgz/u037//XdNdna2JjU1VbssISFBk5WVVSz5tGrVSvvYr732mnZ5UlKSpnnz5sWSU67Y2FjN999/rxk%2BfLjGxcVF89FHH2nOnDlTLLncv39fs3btWk3Lli01TZo00bRq1UqzZMkSTXp6ukEev3Xr1ppHjx5pNBqNZuHChZqpU6dqNBqN5vHjx5q2bdsaJIcnubq6ajIzMw3%2BuM/i4uKiOXHiRHGnoXFxcdFs375dk5iYqElPT8/zY2hOTk7a30eOHKlZu3at9m9Dvr89PDw08fHxGo1G93MmKipK06ZNG4PlIUoWmTMhhHiuQ4cO4ebmlmd5x44dDT5eF8DGxoYaNWoY/HGfZmZmhqenJwcOHNAu%2B/HHH%2BnSpQuRkZEGz6dJkyZ8%2B%2B23OstSU1NZvHgxTZs2NXg%2BT7KxscHOzg47OzsyMzO5cuUK06ZNw9vbm5iYmCJ//MePH7Njxw5GjRqFu7s7e/bsYcqUKRw9epTg4GAuXLjAjBkzijwPyBmmZ21tDcDx48fx9PQEcrantLQ0g%2BTwpLfffps1a9aQlZVl8MfOj5mZGa1atSruNChTpgw9evSgXLlymJmZ5fkxtMqVKxMVFcXt27c5c%2BaMdru5ceMGVlZWBsujU6dOTJo0iWPHjqHRaIiIiGD79u2MHTuWnj17GiwPUbLIMCchxHN169aN4cOHM2zYMJ3lQUFBrF27VueMHoawY8cOwsPDmTp1qkGHpzzN29ub9u3bM2rUKO1wjIyMDNatW8ehQ4f4/vvvDZpPZGQko0ePBnIm09arV4/o6GhsbW1ZsWIFDRo0MGg%2BACdPnuTnn38mJCQES0tLevfujZeXF/Xr10etVuPv78%2BFCxeK9LXy8fHh0KFD2NjYaB%2B/Tp06OvdJTEykY8eOnD9/vsjyyDVixAjatm2LhYUFK1as4PDhw5QpU4bdu3ezZs0atm/fXuQ5POn8%2BfNMnTqVhw8fYmtrm2eulKGGFeX65ptvUKvVvPfeewZ93Kdt376dS5cu8e6771K5cuVizQVgy5YtLFiwAIAePXowf/58kpKSGDJkCJ6enkU%2BVyxXeno6X3zxBdu3byclJQXIOVgwePBgJkyYUCyFlih%2BUkwIIZ7rwIEDTJ06lbJly1K1alXUajWxsbEkJSXx5Zdf0rVrV4Pm4%2BXlRUxMDI8fP6ZChQp5dn4MNTegRYsWnDlzBmNjY53lWVlZuLi4GGTH9GlpaWkcPHiQ6OhoLCwscHBwoH379piYmBg8lw4dOpCcnIynpyd9%2B/bFzc0NIyPdzvCsrCxatGihncRZFD7%2B%2BGO8vLxo27btc08qsGPHDvr27VtkeeS6fv06c%2BbMISkpiYkTJ9K5c2cSEhLo2rUrX331Fa6urkWew5O6du1K7dq1adeuXb5zWYYMGWLQfMaPH8%2B5c%2BcwMTGhWrVqebaZH374wSB5hIaG4uvry/379/NdXxwT02NiYkhOTtYeGMjOzubnn3/G39C04QAAIABJREFUy8vL4LloNBru37%2BPhYWFQXtGRMlk%2BG8YIcQrpXPnzhw9epSjR48SGxtLRkYG9vb2uLm5FcsRu7ffftvgj5kfe3t7zp07l%2BesLsePH8fW1rZYcrKwsKBly5bUrFkTlUpFlSpViqWQgJwegTfeeIOyZcs%2B8z4mJibs3r27SPMYNGgQQL7FnZGREZUrV6Zq1aoGKSQgZ0L8%2BvXrdZbZ2Nhw5MiRYulpu3//Pr/88kuJOaLcuHFjGjduXNxpMGfOHDp37oy7u3uxnDDgad27d89z/RojIyM6deqEq6srJ0%2BeNFguV65c4dChQ8TFxWk/Zzp16pSnx0%2BUHtIzIYTI15dfflmg%2Bxmqe72k2blzJ76%2Bvri5uVGjRg2ys7P5888/OX36NIsXL%2Bb11183aD5//vknU6dOJTIyktyPdZVKRdOmTfH398fBwcGg%2BaxateqZ64yMjLC3t8fV1bXIC9LGjRtrX48nX5fcv1UqFfXq1WPZsmXUrl27SHMBaN68OTY2NvTq1YvevXsXy/CzJ82aNQtPT086duxYrHmUNK1bt%2BbEiRPFVoznOnXqFKdOneKbb77J99o20dHRHDhwgHPnzhkkn82bN%2BPn50ejRo2oVq0aGo2G27dvc%2BXKFebMmcPAgQMNkocoWaSYEELka/jw4S%2B8jyFOxfq0jIwMvvrqK3bv3s2dO3dQqVTUqFGDfv368d577%2BUZFlGULl26xM6dO4mOjkalUlGzZk369%2B9Pw4YNDZZDrv79%2B1OnTh1Gjx5N9erV0Wg0xMTEsGbNGq5fv27wsfhvvfUWkZGRpKam4ujoiEql4saNG9rhcvfv3%2BfRo0csW7asSHdkT506xddff83bb79Ns2bNMDIy4uLFi2zYsIExY8Zgb2/P6tWruXv3bp4J7EUhNTWVw4cPExISwuHDh6latSq9e/emd%2B/eVK1atcgf/2l%2Bfn7s2bMHR0dHqlatmuf98%2BRpog1l8%2BbN7N69m5iYGFQqFQ4ODvTr148%2BffoYLIfAwEDs7e21PVvF5dKlS2zZsoXg4GCcnZ3zrLewsKB///706tXLIPm4ubnh5%2BdHly5ddJaHhIQwe/ZsTpw4YZA8RMkixYQQ4pXi6%2BvLmTNnGDZsGI6OjkDOVVk3bNhA3759mTBhQjFnWDycnJw4depUnqEyKSkpuLm5ac8HbygbN24kIiKCjz76SHv2osTERL744gtatWqFl5cXW7ZsYdOmTezYsaPI8ujRowfr16%2BnUqVKOstjY2MZN24c27ZtIzU1FQ8PD3777bciyyM/mZmZnDx5kj179mivnTJw4EC6d%2B9usGFHn3zyyXPXL1y40CB55Fq6dClbt27Fy8tL5/29Y8cOfHx8DDaHY9y4cZw7dw5zc3OqVKlSbHM3cvn6%2Bhr8wpP5admyJb/99lueHhu5zkTpJnMmhBCvlH379hEcHEzNmjW1y9q1a0f79u157733DFZMqNVq1q5dy44dO4iPj%2BfMmTOkpKTg7%2B/PRx99ZPBx1vXr1%2Bfu3bt5hurcv3%2Bf%2BvXrGzQXyBnmFBoaqnPhMWtra6ZPn07Pnj3x8vLi3//%2Bd5HvrD7r1LMmJiZERUUBOQWXIXu0cj148ICoqCjt1Z5tbGz44YcfCAwMJCAgwCA9XM97/Q8ePFjkj/%2B0bdu2sWbNGho1aqSzvGfPnnz00UcGKyaaNGlCkyZNDPJYBeHr60tERARRUVH5nkJ4wIABBsmjb9%2B%2B7NixI8/j7d69m969exskB1HySDEhhHilZGVl5TvOvkaNGiQkJBgsj88%2B%2B4zTp0/z3nvvMXPmTCDnSHNUVBQLFy40%2BFHEoUOHMnnyZPr160etWrVQq9VER0ezc%2BdOBgwYoHOWq/bt2xd5Punp6Vy5cgUnJyed5VFRUTx8%2BBCAy5cvY2NjU6R5dOzYkXfeeQdvb29q1KiBiYkJt2/fZvPmzTg7O5ORkcHIkSO15%2B0vasnJyezdu5eff/6Zs2fP0rRpU7y8vFi9erX2tdi4cSPTpk3jp59%2BMkhODx8%2B5I8//iAjI0O7LDY2lnnz5hn8rGTJycnUq1cvz/ImTZoQFxdnsDwmTpxosMcqiM8//5z//ve/lC9fPt8rgxuqmEhNTWXhwoWsX7%2Be2rVro1aruXnzJjExMXTs2JH//Oc/2vsWxxA5UTxkmJMQ4pUyYsQImjdvzsSJEzE1NQVyCozly5dz%2BvRpNm3aZJA83Nzc2LJlC9WrV8fJyUk7jCg%2BPp5%2B/foZ7BS1uQp6FFulUhnktJYBAQF8%2B%2B23uLu7U716dUxNTYmJieHQoUN4eXnx0Ucf0aZNGyZNmsQ777xTZHmkpqayaNEijhw5Qnx8PNnZ2djZ2dG2bVs%2B/vhjKlSoQEBAAO%2B%2B%2B65BepNee%2B01KlasSO/evenbt2%2B%2Bk741Gg1OTk5FesrcXCEhIXzwwQekp6ejUqm0k9Stra3p27cv06dPL/IcnjR48GD69euXpwdiy5YtbN68mW3bthkkj%2BzsbDZt2qQ9axFAlSpV6Ny5M0OGDHnuaYaLQuvWrVmyZAnt2rUz6OM%2B7UXD4p5k6CFyovhIMSGEeKVcu3aNkSNHkpaWpr0S9q1btzAxMeHrr7%2BmWbNmBsnDxcWFU6dOYWxsrFNMJCcn06FDh2K5zkRJs2fPHu1OvEajwdbWlrZt29KvXz%2BMjIz47bffaNOmTXGnaVDPe87BwcHaCb/Z2dkGGXrVs2dPxowZQ48ePWjVqhUXLlzg0qVLfPPNN0yePNngp/s8e/as9iQCuY99/fp1oqOjCQgIwN3d3SB5zJo1iwMHDtCrVy%2Bdsxb98ssvdO3aldmzZxskj1zt2rXj0KFD2gMoxeXEiRO4uroavJgSJZsUE0KIV05GRgZHjhzh1q1bZGRk4ODggLu7O2XKlDFYDm%2B//Tbt2rVjzJgx2mIiNTWVzz//XDsh3NAePHjA4cOHtfMEHB0d6dSpU6m/qNSJEyd0zg5Us2ZN%2BvXrR/PmzYsln6ioKMLDw/MMK1qzZg0XLlwwaC7Ozs7awvfJovjatWtMnz6d4OBgg%2BYDcO/ePXbt2kVMTIz2/d29e3eqVatmsBycnZ358ccf%2Bde//qWz/Nq1awwcONDgBwtWr16NSqXK9/SwhuTs7IyVlRU9e/akT58%2BJeKaIKL4STEhhBCFEBkZyejRo4GcSc716tUjOjoaW1tbVqxYYfDrB5w8eZIJEyZgaWmpnZx%2B8%2BZN1Go1GzZsMPgk7MzMTAIDA9mzZw%2B3b98G0O7EjxkzxmBHNjdt2sSiRYtwd3fXXmvj%2BvXrHD16lGXLlhlsrkSu4OBgfH19sbGx4eHDh9jZ2XH//n2qVauGt7e3dpsylK5du7Jq1Srq1KmDh4cHq1atomHDhqSlpeHq6lpsPWyZmZnai6JVrlw5z5Xmi5q7uzuhoaF5zqqVnp5O165dOXLkiEHzmTRpEqdPn8bc3DzfK4Mbanhneno6R48eJSQkhEOHDmFnZ0fv3r3p1auXzkkxROkixYQQosTz9PRk//79wIsnDxtyrkJaWhoHDx4kOjoaCwsLHBwcaN%2B%2BfbFc6Kp///707t1bZ/6BWq1m5cqVnDlzJs9Vl4uan58fJ06cYNiwYTo78Rs3bmTgwIGMHTvWIHl06dKFuXPn4urqqrP88OHDLFmypEhPS5ufrl27MmfOHNzc3Hjttde4ePEisbGxLFiwgBEjRtCiRQuD5rNu3TqWLFnC8ePHCQgIYN%2B%2BfXh6ehIZGYlarTb4KVAfPXqEr68voaGhZGVlAWBubk6vXr2YOXOmwc6StnXrVsLCwvDx8dFOjE9ISCAwMFB7Cl9DWrp06XPXT5kyxUCZ/E9WVhanT59mz5497N27lzp16jBo0CB69epVYq6oLgxDigkhRIm3c%2BdOvLy8AF548bV%2B/foZIiWtuLg47RHUKlWqYGdnZ9DHz9W8eXPOnj2bp5DJyMigffv2nD592qD5uLm5ERQUlOdoZVRUFGPHjiUkJMQgeTRv3pz/%2B7//y3NkW61W07p1a/7v//7PIHnkenJYUfPmzTl//jwqlYpbt24xbtw4fv75Z4PmAznzFFq1akVWVhaBgYGEhYVRvXp1xo4da9ChRQBTp04lLi6OMWPG6FxnYtWqVTRv3pxPP/3UIHm8/vrrxMbGkpGRgbW1NdnZ2SQnJ2Nqaqq9bkouQ59soaS4c%2BcOu3btYs%2BePURFReHh4cGDBw%2BIiYkhICCgRJ1aVxQtOTWsEKLEyy0kAG7fvp3vtSRSUlJYunSpwYqJP//8k6lTpxIZGak9A45KpaJp06b4%2B/trj8Ybir29PX/99Rd169bVWR4dHV0scyYyMzPzPYWvg4ODQU/h6/D/7d17XI/3/z/wxztUyHlz6oAZY0xFimQsZpQy50Njo4kQ5jjLMXLONodoTm1hzuRUqGFOIcdYZLQpUo45Ve9Ovz98e/96907i875e1/Wex/122%2B327rqum%2Bv5yXv7XM/r%2BXo%2BX1ZWOHbsmM4u2ydOnJBlx%2BkaNWpoHt6rVKmCs2fPws7ODhUqVEBiYqLweADAzs4OwMu9N%2BR4w53f0aNHsX//flSuXFlzrHbt2mjcuDH69OkjLJnw9vYWcp%2BiLFu2TPPfup9//rnIa0eNGiUiJKSmpmpGG58/fx7W1tbo3bs3XFxcUK5cOQDAhg0b8P3338uSGJM8mEwQkUF4/PgxHj58iKCgILi6uqJgUfWff/7Bpk2bhD1sjB07Fh9%2B%2BCHmzp0Lc3Nz5Obm4vbt21i5ciVGjRr12gqKvnXp0gVeXl7w8PDQNI3evHkT69evR8eOHYXGAgANGzZEUFAQhg0bpqkKZGdnIygoSCfhkdLw4cMxfPhwtG7dWms60J9//gk/Pz9hceTx8vLC119/jaioKHTr1g3e3t6wt7fHjRs3hC9x2r9/P7Zs2YKrV6/i8ePHKFmyJKpVqwZbW1t88803QjbNK6hEiRI6u7gDL0fVvnjxQlgcBV9KPHr0CKVKlRKamJ85c0bzuajd2UVOVmrVqhVq1KgBd3d3zJkzp9A%2BiX79%2BmHevHnCYiL5cZkTERmEbdu2Ye7cuXj27JlOIpGnQ4cOWLx4sZB4rK2tERUVpfPg8/z5czg6Omqm4oiSm5uLdevWYdu2bVpTrtzd3eHp6Sm8gfXatWvw9PREZmamZrnKrVu3AABBQUE6m9lJ6cqVK9i%2Bfbvm92JpaYkuXbqgWbNmwmLI79atW5rK1caNGzXLijw8PFChQgUhMaxZswbr16%2BHp6cnypcvj99//x1ffvklKlSogGPHjmH37t1YuHCh8Ab1YcOGoUKFChg/frymOvHw4UMsXLgQKSkpWLVqlaT3v3//PmbMmIG4uDi4urrCx8cH48aNw759%2BwC8rOIEBASgatWqksahNElJSVpVNaL8mEwQkcHIzs6Gg4MDQkNDdc6ZmpoK7Vfo2bMn5s%2Bfr7Pp2K1btzB27Fhs2bJFWCxKlZGRgUOHDmmN%2BGzTps07Par2u%2B%2B%2Bg6OjI1q0aCHr9Js2bdpg9erVmirRrVu3MGHCBE3D9bFjx%2BDv74%2BwsDChcSUnJ8Pb2xuxsbGa3oQnT56gbt26CAwMlHz54HfffYf79%2B%2BjQ4cO2LFjB5o0aYL4%2BHh89913UKlUWLp0KcqXLy/L7s5yjjjOPzaYqCAmE0T0n5CTk4N%2B/foJmz6zY8cOrF27Fl27dkXt2rWRnZ2NhIQEhIaGokePHqhdu7bm2tdNoHpbmzZtKva1vXv3liQGJZowYUKxr50/f76EkehatGgRoqOjERMTg2rVqqFly5Zo2bIlWrRoodUnIDU7OzscP35cMx0pPT0dTk5OiI6OBvAycbezs5NtNGxsbKwmCbW0tBS2GaWTkxN2796NSpUq4ebNm%2BjcuTPCw8M1SczDhw/h5uaG48ePC4knj9wjjvMmjxEVhj0TRGRQnj17hmXLluHy5cvIzMzUHL9//z4yMjKExTFp0iQAKHRt8KxZszSfVSoVYmNjJYkhKCioyPMqlQq5ublQqVRCkok2bdoUe/324cOHJYtDye/IxowZA%2BDllK1Lly4hOjoaO3bsgL%2B/P6pWrSqs16ZRo0YIDAzEd999h9zcXCxfvlzTa/P8%2BXOsWLFCp%2BomUsOGDdGwYUPh901PT9dUzj744AMYGRlpVUPKlCkjtHcjz9q1a7FixYpXjjiWOpngjtdUFCYTRGRQpk2bhvj4eLRu3RqrV6/G4MGD8ddffyEzM1Po0oOrV68Ku9er/PHHHwCAxMREGBsba9ZxJycn47fffkN6ejqcnZ3RqlUrIfGMGDFCyH1eZ8GCBXKH8FpGRkYoVaoUSpUqpZnJn5aWJuz%2B48ePx%2BDBgxEcHAyVSgVTU1MEBgYCAA4ePIiwsLDXThDSJ2dn59c%2BsKpUKkREREgaR%2BPGjbF69Wp4eXnByMgIBw4c0Dq/ZMkSNG7cWNIYCnP//n3Y29vrHHdyctIkqFJSq9Xo06fPa68TvS8JKQOXORGRQWnRogXCw8NRsWJFrdL7r7/%2BiidPnsDHx0dYLJGRkZo3gjExMQgNDUXt2rXRr18/nR1qpRIdHY1vv/0Ws2bNQufOnaFWq%2BHm5obMzEzUr18fp06dwqJFi/DZZ58Jiaeg7Oxs3L9/H8DL8bVyvOHcsmWLzlrzbt26wcXFRXgsP/30E86ePYvr16%2BjXr16sLGxga2tLWxtbVGpUiWhsaSmpuLChQsAXu5/kdejoFarUapUKa2/q7wGXKkUtQ9KQkICfvrpJ2RnZ%2BPEiROSxQC8fEng6emJ7777Dj169NA617FjRzx79gxr165FvXr1JI2jIHd3d4wdO1ZnxPHRo0cxb9487NmzR9L7N2rUqFjjcpXyQoHEYjJBRAbFwcEBJ06cQIkSJdCsWTMcPXoUZcqUQVpaGpydnXHy5EkhcSxcuBAHDhzAgQMHcPfuXbi4uOCLL77AP//8AxsbG0ycOFFIHF9//TVatmyp2VF67969mDx5MiIiIlClShXs2bMHGzduxLp164TEk%2Bfp06eYMWMGDh48CLVaDeBlk7y7uzt8fX2F7ZC7ePFibNy4Ee7u7lprzXft2oVx48ahV69eQuLI07hxY82ULXt7e3zyyScoVaqU0BjehhwNuGq1GitWrMDatWvRrVs3jBo1SmfDOKnum56ernOvqKgoNG7cWGuAgNRJVp79%2B/dj7Nixrxxx3K1bN0nvzwZsKgqTCSIyKN9%2B%2By2qV6%2BOqVOn4ptvvoGdnR0GDhyICxcu4Pvvvy9yHrs%2BtW7dGuvXr4eVlRUCAwMRHR2NNWvW4MGDB%2BjevbukPQH5NW3aFMeOHUOZMmUAvGw%2BTk9P14zITU9PR6tWrYTv9Dx27FjcuXNHZyfjoKAg2NnZaXpOpNamTRsEBgbq7MZ74cIF/PDDD5qRn6KkpaXh/PnzOHPmDKKjo3Ht2jXUr18fzZo1Q7NmzfDpp58Kjae4RDfgRkREYPbs2ahRowamTJkiy54XxSHyIVvOEcdswKaisGeCiAzK1KlTMWXKFAAvm1mHDBmClStXwsjISMja4TzPnj3TvOk%2Bfvw4XF1dAQBVqlQRusNzbm6u1l4X0dHRGDRokOZnExMT5OTkCIsnz9GjRxEWFqY1rrdu3bpo0qQJ%2BvbtKyyZePr0KT766COd440bN0ZycrKQGPIrXbo0HB0d4ejoCODlW/Dt27fj119/xS%2B//CJZs/7/StTytH///RczZ87EtWvXMG7cOHTp0kXIfd%2BWyPexjRo10kmKRWnevPkbXS%2BqYkPKwGSCiAyKlZUVfv31VwAvx1seOnQIN2/eRI0aNVCtWjVhcVhaWiIqKgplypTBpUuX8OOPPwJ42Tshcr%2BLatWq4caNG/jwww9x9epVJCUlaU18%2Beeff4SvxQdeNhjnVUvyq1ixotBpOB9%2B%2BCF27NiBnj17ah0PDQ3VVExESktLw8WLF3H%2B/HmcP38eFy9ehJmZGRwcHDRL1d5F6enpWLZsGdavX4/evXvjp59%2BMoj9SKROsnJzcxEcHIwDBw6gRIkScHFxQb9%2B/SS9Z2FWr179Rtd37NiRy6LeIUwmiMiguLq6wsXFBZ06dcIHH3yA8uXLC9m0qaAxY8bA29sbarUa3t7eqFq1KlJTUzFkyBB89dVXwuJwcXHBhAkT4Orqih07dsDGxkazpvr58%2BdYuHChZPtcFMXGxgazZs3C%2BPHjUbFiRQDA48ePERAQIHQazvjx4/Htt98iJCREa615fHw8lixZIiyOPHZ2dnjvvfdgb2%2BPDh06YOrUqbCwsBAeh9J88cUXyMzMxPjx4/Hhhx%2B%2BskLzpm/IDd3KlSsREhKC3r17IycnB4GBgcjOzkb//v3lDq1IXEH/bmHPBBEZlODgYERERODcuXOoX7%2B%2BJrGQYzfhrKwsZGRkoGzZsgCAkydPYtWqVYiOjhb2Vi4rKwtz5szBiRMnUKdOHUyZMkWzvGDGjBk4ceIEQkJCNGNjRUlKSsLQoUNx/fp1TWXk0aNHqFWrFpYvX661qZ/U7t27h927d2vWmltZWcn2nYmPj5d1/4a3JXVvgLOz82uvUalUiIyMlCyGtyH176VTp06YO3curK2tAbzs9ZkyZQp2794t2T31gQ3b7xYmE0RkkB4%2BfIjIyEgcPHgQUVFR%2BOijj%2BDq6opvvvlGaBx37tzB9u3bsWPHDty7dw/Ozs7o1q2bIhppk5OTUblyZVmnBV2%2BfFnrId7a2lroeNi1a9di4MCBwu5XmA4dOhT7f/P%2B/fsljubtKO3hUClr8qX%2BvdjY2OD8%2BfOa709OTg6aNWsm287kxaW07wtJi8uciMggVa5cGT179kTPnj1x/vx5BAQEYN68eUKSCbVajYiICGzZsgWnT5%2BGtbU1UlJSsGXLFkVNnRHZQ1LQ8OHDsWzZMjRu3FiWTb7yrF69Gl27dtUstZLD119/rfn86NEjbN68Gc7OzqhduzZyc3Nx/fp1/Pnnn/D09JQtxtcRNcq3uN6VNfl5O9jnMTIykmWgAlFRmEwQkcHJzc3F2bNnERERgcjISDx48ABt27YVsmPvzJkzsWfPHlSsWBFubm7w8/ODpaUlbG1tNcudCLh16xZiY2PRsGFDWeMYPHgwRo8eDVdXV9SoUQMlSpTQOp%2B/WV0qHh4ems/ffvstFi9erNPnc/r0aQQFBclSRbl79y727NmDu3fvYvLkyQCAS5cuoUmTJpprzpw5IzyuoihlUYXSkiwiOXCZExEZlEmTJuHIkSNIS0tD27Zt0alTJ7Rp0wYmJiZC7t%2BgQQO4urpi1KhRmtGwwMsdhHft2iXLOnwlWrRoEfbu3QsbGxuYm5vrPMSPGjVKSBxFVYpUKpXwUay2trY4ffq0ztIztVoNBwcH4ctXIiMj8d1336Fp06Y4e/YsYmJikJSUhM6dO8PPz08z8lhpRCyjKU6SJbWGDRuiadOmWsfOnTunc2z9%2BvXCYioOLnN6t7AyQUQGJT09HdOmTUPbtm2FJRD5rVq1Clu3boWbmxsaNmyILl26oFOnTsLjULro6GhUq1YNSUlJSEpK0jonsmfiypUrwu5VHJaWlli2bBm8vLw0o3PT0tKwevVqmJubC4/np59%2BwqJFi9C%2BfXvNQ3KNGjWwbNkyzJo1S7HJhNQKJlmTJ09GUlISBg4cKDTJGjJkiM4xQ5hoxYrNu4WVCSIyOOnp6Th06BDu3r2rWRZy9%2B5dVK9eXVgMjx49QmhoKLZt24b4%2BHjk5ORgxowZ6Nq1K0qW5HsaOb3JZnSi%2B0rOnz%2BPkSNH4tGjR6hYsSKys7Px9OlTlClTBkuXLoW9vb3QeGxsbHDu3DkYGRlpvU3Ozs5Gs2bNcOHCBaHxFJfUb77d3NwwatQoTZKVt/tzVFQUZs2ahT179kh27//FypUrMXjwYMn%2BfDs7O5w5c0boCwFSPiYTRGRQzp07B29vb5QvXx5JSUm4fPkybt%2B%2Bjc6dOyMwMFDIGviCLly4gC1btiAsLAympqZwd3fH999/LzwOub3JuEo3NzfJ4mjQoEGxH3bk2HE6OzsbFy5cQHJyMtRqNapVqwYbGxutncxFcXFxQUBAABo2bKj1gH7kyBH4%2BfkpbhRrHhFTlJhk6fruu%2B/QokUL9O7dW7J7kOHh6zMiMihz5szByJEj4eHhoVmWYW5uDn9/fyxcuBDbtm0THpONjQ1sbGzg6%2BuLvXv3yhKDEvj7%2B2v9/OLFC6jVapiZmSEnJwcvXryAiYkJ3n//fUmTifxJTUxMDLZt2wYPDw/UqVMHOTk5uH79OjZu3CjbyNgSJUqgWbNmWsfS09PRtm1bHD58WGgs/fr1g6enJ3r06IHs7GwEBwfj2rVr2LdvHyZMmCA0FiWpWbMmrl27pjNA4NixY0J3uH9TUr8fTktLw08//YQlS5agevXqOlXYjRs3Snp/UiYmE0RkUK5fv655K5b/7XPHjh3h6%2BsrV1gAgDJlymjG1b6LoqKiNJ%2B3b9%2BOCxcuYOTIkXjvvfcAvNwbYPHixZKv%2Ba5Xr57m85gxY7Bq1Sqt5UyNGjWCnZ0dvL298cUXX0gaS0EpKSmYO3cuLl%2B%2BDLVarTn%2B5MkTze9JpK%2B%2B%2BgpVq1bFtm3bYGlpidDQUFhaWmL58uVwdHQUHk9xSb0m31CTLKmXH8k96pmUicuciMigfP7551izZg0sLS21SvoXL17EiBEjcPToUZkjJABo27YtwsLCdJbuPHv2DK6urjhy5IiQOJo2bYo///wTZmZmOnG0bt1a%2BPSkoUOHIjMzE%2B3atcPs2bMxZcoUXLlyBXFxcVi6dKksCYXSKGGKEgAcOHAA27Ztw61bt2BqagpLS0v06dNH0UkWpyiRHFiZICKD4ubmhsGDB2PgwIHIyclBREQErl69ivXr16Nfv35yh0f/Jy0tDSkpKahVq5bW8UePHiE9PV1YHLa2tvDx8YGnpyfMzc2RnZ2NpKQk/Prrr7C2thYWR55z587h8OHDKFOmDObNm6epsoVjrSAiAAAgAElEQVSGhiIwMBBTp06VPIZJkyYV%2B9o5c%2BZIGIkupUxRAl7uXN6hQwdh9zMUe/fuRWhoKFJSUrBz506o1WqEhIRg0KBBbMx%2BRzGZICKD4uPjg3LlyiEkJAQqlQo//PADLC0tMWbMGPTo0UPu8Oj/dOrUCQMGDIC7uzssLCyQlZWFu3fvYvfu3UIf0ObNm4eZM2fC29sbmZmZAF72LDg4OGDevHnC4shTsmRJzTpzExMTpKamokKFCujUqRP8/f2FJBMZGRmazzk5OThy5AgsLCw0PSU3btxAcnKyLCOP5RxVq%2BQkSykCAwOxadMm9O7dGytWrADwconezp078fTpU4wePVrmCEkOXOZERER6l5WVhU2bNiEyMhJ3796FWq1G1apV8emnn2LQoEHC5tCnpaWhdOnSyM3NxcOHD6FWq1GlShXZ5uCPHDkSarUaP/30E4YNG4bq1atjwIABOH/%2BPJYtW4Zjx44JjcfPzw%2BNGjVC9%2B7dtY6vX78ef//9N6ZNmyY0HjmnKI0ZM0bz%2BXVJVsFhA0qRf4ytFNq0aYNVq1ahXr16Wn8/CQkJGDBgAA4dOiTZvUm5WJkgIsXbtGlTsa/lyEJlOHv2LDw8PODh4SFrHI6Ojmjfvj3c3Nzg5OQEIyMjWeOZPn065s2bhxIlSmDixInw8vLC9u3bUbp0aeEP7sDLyVc//PCDzvGePXvC0dFReExyTlFatGiR5rOfnx8mT578yiRLqbp16ybpn//06VOtAQd5qlatiocPH0p6b1IuViaISPGcnZ2LdZ1KpVLsXPx3jY2NDSpWrIjOnTvDzc0NH330kSxxREVF4eDBg4iIiEBmZiY6deoEd3d3WfolCpOdnY179%2B6hcuXKslRLnJ2dMXnyZJ1/x44cOYLp06cLf9O8bt06BAYGokePHlizZg3GjRunNUVJVHLavHlznDx5Umf0qVqthqOjI6Kjo4XEkScnJwdr167Fzp07ce/ePURFReHFixf48ccfMX78eGHfnT59%2BqBfv36af4fyKhPLly/HwYMHsX37diFxkLIwmSAig7B3797Xrpf29fVV7PKDd01aWhqOHDmCgwcP4siRI6hRowbc3Nzg5uaGGjVqyBLThQsXEBERgfDwcBgZGcHNzQ3dunWDubm50Dhu3LiBAwcO4Pbt21CpVLC0tISrq6vwOABg8%2BbNmDZtGj7%2B%2BGNYWFhoGtRjY2Ph6%2BsrS2VJCVOUlJZkzZ07F8ePH8fAgQMxffp0XLp0CY8ePcLIkSNRr149Ib02AHDy5EkMHz4c9evXx6VLl9CmTRvExcUhNTUVgYGBwndwJ2VgMkFEBqHgyMOWLVvi5MmTRV5DypCZmYmTJ08iLCwMERER%2BOijj9CzZ0906tRJ%2BNv4K1euYN%2B%2Bfdi0aROMjY2hVqvx2WefYfLkyahQoYLk9w8PD8e4ceNQv359WFlZAQBu3ryJf//9F2vXrkXTpk0lj6GgmzdvIjIyUrMjd15vi%2BhRrEqitCSrVatW2Lhxo85I7JSUFHTv3l3oSOzk5GTs3r0bCQkJMDU1hZWVFVxdXVGxYkVhMZCyMJkgIoNQsLGwsEZDqZsP6e0kJydj3759CAsLw9WrV/Hpp5/iwYMHuH//PpYsWYIGDRpIev%2BEhATs2rULu3fvRlJSEj777DN06dIFn376KZ4/fw4/Pz88ffoUQUFBksYBAK6urhg6dKjODuBbtmzBtm3bFLWD8JgxY7T6CKSi1ClKSkqymjdvjlOnTuk0pj9//hytWrWStDE9v5UrV8LV1RU1a9YUcj8yDGzAJiKDUHB%2BeWHzzDnjXDmePXuG8PBw7N69G9HR0WjcuDG6dOmCX375RfMGc926dZgwYQJ27dolWRy9evXC5cuXYW1tjYEDB8LFxQXlypXTnC9fvjxmzZoFBwcHyWLILzExES4uLjrHu3btKsuo2uzsbGzcuFFnR%2B6UlBTExcUJiUGpo2o/%2BOADfPDBBzrHRSVZ%2BTVs2BDBwcEYNGiQ5lhGRgYWLVqERo0aCYtj9%2B7dWLRoEZo0aQJXV1d06tQJ77//vrD7kzIxmSAiIr1zdHTEe%2B%2B9Bzc3N0yfPh116tTRucbDwwPz58%2BXNI7WrVsjICAAlpaWr7zG1NQUK1eulDSOPDVr1kRMTAxsbGy0jv/111%2BSTysqzMyZM3Ho0CHY2dkhPDwcrq6uiI2NhbGxMZYvXy4kBiVOUVJCkpVf3uSv3377DWq1Gt26dcO///6L8uXLC/t7AoBdu3YhISFB03s0f/582NraolOnTujYsSMqV64sLBZSDi5zIiKDULAforD%2BCPZMKMepU6eK9bY/JydHknGtycnJxbquWrVqer93UdavX4%2Bff/4Z7u7uqFu3LoCXy2l27doFT09PeHl5CY3HyckJW7duRfXq1TXLBHNzc7Fw4UJYWFigb9%2B%2BQuNRyhSlvCbrwpIsX19fWXpbXrx4gUOHDmk1prdp0walSpUSHkue%2B/fvIyIiArt27cKlS5dw%2BfJl2WIh%2BbAyQUQGITs7G5s3b0be%2B4%2BCP%2BcdI3nt3r270M8F5fUMSLXvQ5s2bYpc9pabmwuVSoXY2FhJ7v8qHh4eqFq1KrZt24aoqCio1WpYWlrC19cX7u7uQmMBXi6VqV69OoCXO4Or1WoYGxvDy8sLbm5uwpOJcuXK4c8//9SZonTy5Emt5WlSi4iI0CRZBw8exPz58zVJ1rVr14QnE7NmzcLkyZMl3QH8TV27dg0RERE4dOgQ/v77bzg5OckdEsmElQkiMgjF3Wvijz/%2BkDgSKkqLFi20fk5NTdWZkqRSqXQmcenb9evXNZ9zc3PRvXv3QmfgF7YB17vEw8MDLVu2xJAhQ9C7d290794dHh4euHr1Kr766ivh%2BykoZYpS8%2BbNcebMGQCAra0tTp06BWNjY6SmpsLNzQ1//vmnkDjytGvXDr/99pss44Pzi46ORkREhGZn%2B1atWqFjx45o3749zMzMZI2N5MNkgoiIJKOUCVtKWAJ3584dHD9%2BHMDLh8P868vVajWWLFmCsWPHCo0pJiYGY8aMQWhoKE6cOIHRo0fD2NgYGRkZ8PDwKHR3bKkpYYqS0pKsoKAg7Nu3D23btkXNmjVRokQJrfM9evQQEscnn3zCBIJ0MJkgIiLJKOEhXglxnDlzBl5eXqhUqRKysrLw/PlzhISE4OOPP0Z0dDR8fX2RlZUl%2Bw7uN2/eRGxsLMzNzXWaxOUmcoqS0pKsNm3avPKcSqXC4cOHhcTx7NkzmJmZITMzEykpKVCpVKhWrZpOckPvFiYTREQkGbkf4pUSR//%2B/eHk5IQhQ4YAABYvXoxz586hdu3a2Lp1KwYMGAAfHx%2BULl1aeGw5OTl4%2BPCh1tSiPKL3E3jdFKVTp04JjSePkpMskZ48eYJp06YhIiICWVlZyM3NhampKTp37owpU6bAxMRE7hBJBmzAJiIikti1a9e0xs8OGjQIgYGBSE9Px9atWyXfuO9V9u3bhxkzZuDJkydax%2BVqUFfCqNo8%2BZMsU1NT2NraAni5XE1EkpWYmAgLCwsALzdeLEpRo4/1afr06bh37x6WLl2KWrVqAQBu3LiBFStWYOHChfD19RUSBykLKxNERCQZuSoCEyZM0Pp5z5496Ny5s851Uu9zkaew34MS%2BklatWqFHj16oGPHjoW%2BVS5s0zYpKWVUrRKSrPzfmQYNGkClUmlNr8v7WWTS17x5c%2Bzfv19nP4nk5GT06dMHhw4dEhIHKQsrE0REpDcFR7JmZGSgbdu2OtdJvca74HsyV1dXnWNyU8KO7enp6fDx8dHZ10EuShlV6%2B/vjz59%2BrwyyRJhz549ms8HDhyQJYaCSpQoUehSvPLly%2BPFixcyRERKoIz/ehAR0X/CiBEj5A4BALBgwYI3un7fvn1wcXGRKBrl6ty5M06fPg1HR0e5QwEA1K9fH0uXLsWQIUNQp04dbNmyBR4eHkhKShL6sKqEJMvS0hK7du2Cu7s7rKysNMdjYmLwySefyBJT06ZN4efnh/Hjx2uqEw8fPsTChQtli4nkx2VOREQkmxkzZmDatGlyhyH5cqwGDRpo3rjnuXv3rs4xEVN58k9EyszMxL59%2B9C0aVNYWFjoVEvGjBkjeTz5KWWK0rRp0/DFF1/InmQV9r2Uc5hAcnIyvL29ERsbi/LlywN4uZdM3bp1sXz5cq2kh94dTCaIiEg2ck9ZyiN1/8KWLVuKdV3Pnj0liyFP//79i3WdSqXCb7/9JnE0RRM5RUmJSVZh30u5em3UajWSk5NhaWmJq1evIjExUbMPiK2tLcfDvsO4zImIiGSjlPdZUvcvvGmSIGXFJiQkRJI/V1/kmqJ0/vx5rZ%2BtrKxw//593L9/X%2Bu4yF6Xwu4lR69Namoq%2BvXrB2tra8yePRsNGjTQTCDr1q0bypYti9WrV8PY2Fh4bCQ/JhNERCQbJTQhK9H27duFLP9Sq9XYtGmTploRGRmJrVu3onbt2hgxYgTKli0reQz5yTlFSelJlpyWLl2KypUrY/LkyTrn1q9fj8GDB2PlypUYPny4DNGR3IzkDoCIiIi0iarY%2BPn5Yffu3QBeLikaM2YMGjVqhNu3b8Pf319IDPnlTVHavn079u7dq/ln37592Lt3r7A41Gq1VnIRGRkJb29vzJs3D8%2BfPxcWh1IcOnQIvr6%2BKFOmjM650qVLw9fXV2v6FL1bWJkgIiJSGFEVm8jISE0yERoaCicnJ4wYMQJPnz5Fp06dhMSQnxKmKAEvk6y4uDj0799fk2QNHjwYcXFx8Pf3x%2BzZs4XEUdho5cKOSd24/%2BDBA3z00UevPN%2BgQQPcvXtX0hhIuZhMEBHRO08pvRuiZWRk4L333gMAHD9%2BHB4eHgAAMzMzWd7AK2VUrVKSrJkzZwq7V1HKlCmDR48e6WxWlyclJaXQ/Sfo3cBkgoiIZKOUh/gWLVrIHYIs6tWrh%2B3bt8PU1BR///03nJ2dAQAnTpxAjRo1hMSQf4pSmTJlMGnSJNmnKCklyVJK437Lli0RHBz8yt///Pnz39l/h4jJBBERSSg7O1szDadq1ao6D4ciNrnbsmUL9u3bh9u3b0OlUsHS0hLdunXT2qTul19%2BkTwOJfrhhx8wYcIEPHv2DL6%2BvqhQoQIeP36MESNGCFvKo8QpSkpIst6GVI37w4cPR48ePZCQkAAPDw/UqVMH2dnZ%2BPvvv7FmzRpcvHgRmzdv1vt9yTBwnwkiItK7p0%2BfYsaMGTh48CDUajUAwNTUFO7u7vD19RU2QnLx4sXYuHGj1i7CN2/exK5duzBu3Dj06tVLSBxvSq69BPIkJyejWrVqst1fbhcvXtQkWaNHj0bPnj3x%2BPFjfPbZZ5g9e7Ys/STFIeX3JjY2FjNnzsS5c%2Bc0iV1ubi7s7e3h6%2BtbZE8F/bcxmSAiIr0bO3Ys7ty5g8GDB6NWrVoAgBs3biAoKAh2dnaYNGmSkDjatGmDwMBANGrUSOv4hQsX8MMPP2Dfvn1C4ijodRWbX375BV5eXpLce9OmTcW%2Btnfv3pLE8CpKG1VbkNKTLBGbQD58%2BBAJCQlQqVSwsrJCxYoVJb0fKR%2BTCSIi0jt7e3uEhYWhSpUqWsfv3r2Lvn374tChQ0LiaNq0KU6fPq0zHSgrKwsODg44e/askDjyKKFik7dk53VUKhUiIyMljkbb5MmTERcXh82bN%2BPmzZvo2rWrZoqSmZmZpEuvlJxkFZdSdpSndwt7JoiISO%2BMjIwKnUlfsWJFvHjxQlgcH374IXbs2KHTyBoaGqqpmIg0ffp03LlzBz/%2B%2BKNOxSYgIEBIxeaPP/6Q/B5vS84pSkFBQcW6TqVSKTaZIJIDkwkiItI7GxsbzJo1C%2BPHj9csg3j8%2BDECAgLQuHFjYXGMHz8e3377LUJCQlC3bl0AL3sm4uPjsWTJEmFx5Dl69KhOxaZu3bpo0qQJ%2BvbtKySZOHbsWLGuU6lUaNWqlcTRaJNzipKSkywiJWMyQUREejdt2jQMHToUjo6OqFSpEgDg0aNHqFWrFpYvXy4sjubNmyMiIgK7d%2B9GYmIi1Go1XF1d0alTJ1haWgqLI48SKjbffvttsa5TqVSIjY2VOBptck5RUnKSVVxcuU5yYM8EERFJ5vLly5qHeCsrK1hbWwsZ8blr1y64u7tLfp83NXToUFSpUqXQis2dO3ewevVqmSOUl5xTlBo0aFCs6%2BRIsvLI2bhP9CpMJoiISG%2BmTp0KPz8/ucNQbCNqUlIShg4diuvXrxdasaldu7bkMfz777%2Bafo34%2BPhXXqdSqYTEUxxKn6IkNSU07hO9CpMJIiLSG6U8xMu9T8PryFWxAbR/N0W9jRf1Bl4pU5SUnGQpZdQyUWGYTBARkd4o5SHe2toa0dHRr11DLuqNrlIqNgBw584d1KxZEwBw%2B/btIq81NzeXPB6ljKpVWpKVn1JGLRMVhg3YRESkNzk5OTh%2B/PhrH%2BKdnJwkjSMjIwNNmjR57XWiHgpDQ0MVk0zUrFkTzs7OxaqEiNhnQilTlMLDwzWfRe%2Bv8TpKaNwnehUmE0REpDdZWVnw9PQs8hoRb3ZLliyJNWvWSHqPN6G0RQD5m3Rzc3Ph7%2B%2BPyZMnyxKLUqYoKS3Jyk8po5aJCsNlTkREpDdK6ZlQShx5GjdujKCgINkrNq8i5%2B9LSVOUNm7cqPlcVJLVp08fSeMoSAmN%2B0SvwmSCiIj0RikP8Urp3chTnAdmOUeOKuXvTWmU9nuRs3Gf6FW4zImIiPRGKe%2BnvL293%2Bh6qfelMDExUdRDqZIoeYqS3Ao27jdu3JjLmkhxWJkgIiK9iY6Ohp2dXbGvV8qUI6nfQCvtDXdBcsan5ClKcv%2B9yX1/ouJgZYKIiPTmTRIJQDlTjqR%2Br6a093YF93bIzs7G5s2bdeKUcl%2BHPEqeoiQ3pX1viArDZIKIiGSjlIclqdedv%2BlkKakrNkFBQVo/V61aFStWrNA6plKphCQTSpqipKQkC1DOqGWionCZExERyUYpyziUEkcepcUjNaVMUSrOBnpSb56Xn9Ib94kAViaIiIgU5117z1cwSZg7d67w8auAcjbQy8PGfTIERnIHQERERNo47pOIDAWTCSIieue9a5UAMgz8XpIhYDJBRESyUcrDUqtWreQOgUjH2zTuE4nGngkiIpLUo0ePUKlSpULPjR49WvL77927Fzt37sS9e/ewc%2BdOqNVqhISEYNCgQZrlRMuXL5c8Dno1pU1RUgpDHbVM7xYmE0REpHfPnz/HvHnzsGvXLmRlZeHy5ct4/PgxJk6ciDlz5qBy5coAgEGDBkkaR2BgIDZt2oTevXtrRp8%2BefIEO3fuxNOnT4UkM29DKRUbUZQ0qtaQvWvfG1IGjoYlIiK9mzhxIlJSUjB8%2BHAMGjQIly5dwvPnzzF16lTk5uZi0aJFQuJo06YNVq1ahXr16mmNW01ISMCAAQNw6NAhIXEUpqiKzZo1ayRPtOi/510bKUzKwJ4JIiLSu8OHDyMgIAB2dnaapURly5bFtGnTcPLkSWFxPH36FPXq1dM5XrVqVTx8%2BFBYHHnyEiobGxu0bt0aAPD48WMMGTJEKx4mEkRkKJhMEBGR3qlUKpiZmekcz87ORkZGhrA46tevj127dukcX7NmDerWrSssjjx%2Bfn5ISEjAqlWrYGT08v%2BCS5UqBTMzM8yaNUt4PERE/yv2TBARkd7Z2tpi/vz5GDdunObY7du34e/vD3t7e2FxjBo1CsOHD8eGDRuQmZkJb29vxMXFITU1FYGBgcLiyHP48GGEhYWhcuXKOhWbL774Qng8RET/KyYTRESkd1OmTMGwYcNgZ2eHrKwsNGvWDC9evICNjY2wfgkAaNmyJfbt24e9e/fio48%2BgqmpKZycnODq6oqKFSsKiyOPUio29N/ENliSA5MJIiLSu5o1a2Lnzp2IiYlBQkICTExMYGVlVWj/gtTKlCmDbt26aZqdExMTZXvoUkrFhgyX3KOWiQpizwQREUni5MmTKFu2LFxcXNCuXTukpqbi2LFjQmOIioqCs7OzVtP3kSNH8Pnnn%2BPUqVNCYwFeVmyio6NhZ2eHjIwMNGvWDO3bt8ejR48wbdo04fGQYWDjPikZR8MSEZHehYSE4Oeff8aSJUvQsmVLAEBkZCR%2B%2BOEH%2BPj44KuvvhISR9euXTFgwAB07dpV6/jevXuxatUq7NixQ0gcBSmhYkOGQymjlokKw2SCiIj0ztnZGYGBgWjQoIHW8bi4OHh7eyMyMlJIHLa2toiOjkaJEiW0jmdmZsLe3h7nz58XEkd%2BJ0%2BeRLVq1fDBBx8AAKKjo5Geng4nJyfhsZBhcHBw0DTu599L4smTJ/jiiy%2BEjlsmKojLnIiISO8ePXqkeVjOz8LCQuj%2BDlZWVjh48KDO8Z07d6JmzZrC4sgTEhICHx8fJCcna46lpqZi7NixWLdunfB4yDCwcZ%2BUjJUJIiLSO09PT9SrVw/Dhw9HuXLlAAD379/HTz/9hMTERAQHBwuJ49ixY/Dx8UGtWrVgYWGBnJwcxMfHIykpCatWrYKdnZ2QOPIopWJDhsXb2xvm5uYYN24cHBwccPHiRU3jfk5ODlasWCF3iPQOYzJBRER6l5CQAB8fH8TFxcHMzAw5OTl4/vw5GjZsiBUrVqBq1arCYklOTkZYWBgSEhKgUqlgaWmJzp07o0qVKsJiyGNra4tTp07B2NhY6/iLFy/QqlUrWZZdkfLduXMHw4YNw99//42srCyULVtWa9RyjRo15A6R3mFMJoiISDJ//fUXEhISYGRkBEtLS5038u8apVRsyDCxcZ%2BUiMkEERFJIjs7GykpKUhPT9c5V6dOHSExXL16FT/%2B%2BCNu3LhRaByiR9UqqWJDhoWN%2B6RUTCaIiEjvdu3ahZkzZ%2BLZs2dax3Nzc6FSqRAbGyskDjc3N1SrVg3Ozs4oXbq0zvmCI2NFYcWG3oRSRi0TFYbJBBER6V2bNm3Qs2dPdOrUCaampjrnzc3NhcRha2uLqKgomJiYCLlfcSihYkOGhY37pGQl5Q6AiIj%2Be549ewZvb2%2Bd/R1Ea9iwIe7evYtatWrJGkcepVRsyLAoZdQyUWGYTBARkd61a9cOp06dgqOjo6xxDBw4EBMnTkSXLl1gbm4OIyPt7ZVErzcPCAjA119//cqKDVFhmjZtikWLFhXauG9tbS1zdPSu4zInIiLSu6CgIGzYsAG2trawsLDQeYgfM2aMkDiK6kWQoxLQrFkznD59WvaKDRkWNu6TkjGZICIivevfv/8rz6lUKvz2228Co1GOCRMm4Msvv5S9YkOGiY37pERMJoiISKjY2Fg0bNhQ1hjS0tLw%2BeefCx8Nq5SKDRkeNu6TUrFngoiIJJGbm4s7d%2B5ArVZrjiUnJ2PYsGE4d%2B6ckBju3r2L2bNn4/Lly1pxPH/%2BXJalIceOHYOVlRUePHiABw8eaJ1TqVTC4yHDwMZ9UjJWJoiISO%2Bio6MxcuRIPHr0CMD/f%2BgBgPbt22PJkiVC4vDy8tLc08/PD9OnT8eVK1cQGxuLZcuWoUqVKkLiKA4lVGxImZQyapmoMEwmiIhI77p164Z27drBxcUF7u7u2LdvHy5fvox9%2B/ZhypQpwqoC9vb2OHLkCEqXLg1ra2tcvHgRALBnzx5ER0dj%2BvTpQuLITwkVGzIsbNwnJeMyJyIi0rv4%2BHgMGzYMKpUKKpUKlpaWsLS0RI0aNTBx4kSsXbtWSBwlS5bU9CWYmJjg8ePHqFixIjp06IAZM2YITyZeV7EhKoxSRi0TFYbJBBER6V2FChVw7949VK1aFeXLl0dCQgIsLS3RqFEjXLhwQVgcdnZ2GDFiBBYvXoxPPvkEc%2BfOxVdffYULFy7Isiv27Nmz4eHh8cqKDVFh6tati0mTJrFxnxSJyQQREeld586d0b17d4SFhaF169bw8fGBu7s7YmJiYGFhISyOGTNmYMGCBShZsiS%2B//57DBkyBDt37kSZMmUwY8YMYXHkUUrFhgwLG/dJydgzQUREkti5cye6dOmC58%2BfY8aMGYiJiYG5uTnGjx8vbD5%2B/mVEeT/fv38flStXxr1791C9enUhceRp27YtNm/ejKpVq8LJyQm///47LC0tkZmZCXt7e5w/f15oPGT42LhPcmMyQUREenf69GnY29vrHM/IyMAff/yBTp06CYkjf9N1fk%2BfPoWzszPOnDkjJI48CxcuRGhoKMLCwuDv74/Y2FhNxebvv//G7t27hcZDhoON%2B6RUTCaIiEjvXvUQn5KSgs8//7zQc/q0f/9%2B7N%2B/H%2BHh4YUmLnfu3EF8fDyioqIkjaMwSqjYkGFRyqhlosKwZ4KIiPQmODgYq1atglqthpOTk875Z8%2BeCemZ%2BPjjj5GYmIjw8HAYGxvrnP/oo48wduxYyeMo6PTp0/jyyy8BAGZmZliwYAGA/1%2BxYTJBhWHjPikZKxNERKQ3OTk5uHLlCvr27YuZM2fqnDcxMUHLli1RqVIlIfGsWbMGgwYNEnKv4pC7YkOGydbWFufOnYNKpUKTJk1w6dIlAMCFCxfw888/s3GfZMXKBBER6Y2RkRE%2B%2BeQTbNiwAU2aNNE6l5qaigoVKgiNp1u3bpg7dy6%2B//57AMD69euxadMm1K5dG5MnTxa2eZ5SKjZkmJQyapmoMEwmiIhI70qUKIFu3bph%2B/btAIBRo0Zh//79qFSpEgIDA2FrayskjqlTpyIrKwsAEBMTgwULFmD69Om4fPkyZs2ahcWLFwuJY8CAAWjWrBn69u1b6PKqvIoNUWGUMmqZqDBc5kRERHrXt29ftG7dGsOGDUNERASmT5%2BOzZs349y5c9iwYQM2bNggJA4HBwdERESgXLlymDNnDh48eICFCxciPT0dzs7OOHHihJA48ly6dEkRFRsyPGzcJ6Uyev0lREREbyYuLg5eXl4AgMjISLi4uKBmzZpwdXXF9evXhcWRk5MDMzMzAMDx48fRrl07AECpUqWQlg9ew7AAABRwSURBVJYmLI48eRWbPKNGjYKDgwNatmzJPSbolfIa91UqlaZxPzw8HIGBgYiPj5c7PHrHMZkgIiK9MzExQWZmJrKzs3H06FF89tlnAIAXL14gJydHWByNGzfGsmXLsHLlSqSkpKBt27YAgH379qFOnTrC4sgza9YstG/fHgAQERGBs2fP4o8//oCvr69mshNRQYMHDy70eGpqqqYfiEgu7JkgIiK9%2B/TTTzFy5EiULFkSZmZmaNGiBTIzM/Hjjz%2BiadOmwuKYNm0aZs6ciSdPnmDBggUoXbo0Hj9%2BLLRfIr%2B4uDiEhIQA0K7Y1KhRAzNmzBAeDykbG/fJELBngoiI9C49PR3BwcF4%2BvQp%2BvXrB3Nzc7x48QI%2BPj7w9/dH9erVZY0vIyMDJiYmwu/r6OiIyMhIGBsbo02bNliwYAFatmyJ58%2Bf49NPP8XZs2eFx0TKpbRRy0SFYTJBRET/KYsXL8bIkSMBAIsWLSry2jFjxogISeP777/HgwcPULJkScTHxyMsLAxZWVmYN28e/v33X6xcuVJoPGQY2LhPSsZlTkREpBf9%2B/fXLOHp3bs3VCrVK6/duHGjZHHk3/itqKbmouKTyvTp0zUVm8mTJ0OlUiEzMxPx8fHw9/cXHg8ZBqWMWiYqDJMJIiLSC0dHR83n1q1byxbH6tWrNZ/zkhulMDU1xdChQ7WOlSlTRitmooJe1bh/7tw5LFiwQNioZaLCcJkTERH9Z50%2BfRrh4eFITEyEkZERPvjgA7i5uaFhw4bCYlBKxYYMV7NmzXDq1CmULFkSkyZNQrly5fDDDz8gNzcX9vb2OHPmjNwh0juMlQkiItKrtLQ0rF69utCHeA8PD5QsKeb/eqZOnYrt27fD0dERderUQXZ2Ni5cuIC1a9fC09MT48aNExKHUio2ZLjyRi2rVCocPXpUM0ZY9KhlosKwMkFERHqTkZGBvn374tmzZ%2BjVq5fmIf7KlSv4/fffUb9%2BfaxZswbGxsaSxrF7927MmzcPa9euRb169bTOnThxAuPGjcOkSZPg5uYmaRxE%2BsDGfVIyJhNERKQ3QUFB%2BOOPPxAcHIzSpUtrnXv48CEGDx4MZ2dnDB8%2BXNI4vvnmG/Tq1QsuLi6Fng8LC0NwcDA2bdokaRz5KaViQ4ZH6aOW6d3GZIKIiPSmW7dumDhxIhwcHAo9f/78eUyaNAnh4eGSxuHg4IA9e/bg/fffL/S8Wq2Gg4NDkdOe9EkpFRsiIn3jaxAiItKb%2BPh4NGrU6JXnbWxskJSUJHkcaWlpr0wkAMDY2FjoWvPg4GCUKlUKoaGhWhWbDh064Ouvv8bgwYOxcuVKySs2ZDjYuE%2BGgskEERHpTU5ODszMzF55XtTeDnLsIVGU/fv3Y%2BLEiTpLvwCgcuXKmDx5MiZNmsRkgjTYuE%2BGgskEERHpVWZmJuReQatWq9GnT58ir8nMzBQUjXIqNmQ4vL29NZ9HjBghYyRERWMyQUREepORkYEmTZrIHUax3vA7OTkJiOQlpVRsyPCwcZ%2BUjt9AIiLSm99%2B%2B03uEAC8%2BZvcFStW6OxMrW9KqNiQYcnIyICHh0ehjfvLli3DwYMH2bhPsuM0JyIiko2npydWr14tdxiwtrbGxYsXJfvzGzRoUKzqQ2xsrGQxkOFRyqhloqKwMkFERLKJjo6WOwQAkLxioJSKDRkWNu6TIWAyQURE7zypexbs7e3f6HqlVGxIXmzcJ0NgJHcAREREpE0pFRuSFxv3yRCwMkFERESkUGzcJ6VjMkFERESkQEoZtUxUFCYTRET0zuObX1IiNu6TIWAyQUREslHKQ3yvXr3kDoFIBxv3yRAwmSAiItnMnz9f73/m2LFji31tQEAAAGDy5Ml6j4NINDbukxyYTBARkV44OTkV%2B9pjx44BADp27Kj3OP4LuwErpWJDRPQ6TCaIiEgv3qQiIKU5c%2BYU67oNGzZIHMnbk6JiQ0QkBVUuX38QEZFAY8aMwaJFi4TdLy4uDleuXIFardYcS05Oxtq1a3H%2B/HnJ7/82FRuit2FtbY2LFy/KHQa9Y1iZICIivcvOzsbGjRtx%2BfJlrYf4lJQUxMXFCYvj999/x8yZM1GlShXcv38f1apVQ0pKCszNzTFq1CghMSilYkNEJAUmE0REpHczZ87EoUOHYGdnh/DwcLi6uiI2NhbGxsZYvny5sDhWr16NNWvWoEWLFmjSpAkOHz6Me/fuwd/fH40bNxYSQ9euXYt13ZgxY4p9LRGRUjCZICIivYuIiMDWrVtRvXp1HDx4EPPnz0dubi4WLlyIa9euoWnTpkLiePDgAVq0aAEAMDIyQm5uLt5//32MHz8eQ4cOxe7du4XEkUcpFRv6b%2BLKdZKDkdwBEBHRf09GRgaqV68OAChRogTUajVUKhW8vLyEViZq1qyJqKgoAMD777%2BvGZ1Zrlw5JCYmCosjz8yZM/HLL79ArVYjPDwcJUqUQFxcHF68eCH090L/TWzcJzmwMkFERHpXv359LF26FEOGDEGdOnWwZcsWeHh4ICkpCS9evBAWx5AhQ%2BDp6YmoqCh0794d3t7esLOzw82bN9GsWTNhceRRSsWGlE8po5aJXofTnIiISO9iYmIwZswYhIaG4sSJExg9ejSMjY2RkZGBfv36wdfXV1gsiYmJsLCwAABs2bIFMTExsLCwQN%2B%2BfVGuXDlhcQBA8%2BbNcebMGQCAra0tTp06BWNjY6SmpsLNzQ1//vmn0HhIuXbs2FHsa9lrQ3JiMkFERJK7efMmYmNjYW5uDhsbG6H3fvLkCbKzs1GpUiUAQEJCAszMzDQ/i%2BTh4YGWLVtiyJAh6N27N7p37w4PDw9cvXoVX331FXcwpjcmetQyUUFMJoiISO8GDhyItWvX6hx/9uwZ%2Bvfv/0ZvXf8XUVFRGDFiBPz8/ODi4gIAWL9%2BPX788UcsW7YMDg4OQuLIo6SKDRmO1zXunzp1Ssbo6F3HZIKIiPTmypUriImJwaxZszB16lSd6TK3bt3Chg0bhGwWB7xc/jFgwACdZSB79%2B7FqlWrhCU1ryJnxYYMx/Tp0185atnX15e9NiQrJhNERKQ3UVFRCA4OxuHDh1GzZk2d86ampujVqxe%2B%2BeYbIfHY2toiOjoaJUqU0DqemZkJe3t7YUlNHqVUbMiwODk5aRr3mzRpgkuXLmka9/P6f4jkwmlORESkNy1atECLFi3g7e2tiFGnVlZWOHjwoM6Um507dxaa7Eglr2Jz5swZbN68udCKzT///CMsHjIshY1aNjY2hpeXF9zc3JhMkKyYTBARkd4tX74c9%2B/fxz///IP09HSd828y9vJ/MX78ePj4%2BGDFihWwsLBATk4O4uPjkZSUhFWrVgmJAQCePn2Kw4cPIysrCytWrNA5b2pqilGjRgmLhwyLUkYtExWGy5yIiEjvVq9ejUWLFiE7O1vnnEqlQmxsrLBYkpOTERYWhoSEBKhUKlhaWqJz586oUqWKsBjyKKViQ4aFjfukZEwmiIhI71q2bInx48fDxcUFpqamcoejKEqo2JBhY%2BM%2BKQmTCSIi0jsHBwecOHFCp/FZhP79%2ByMkJAQA0Lt3b6hUqldeu3HjRlFhAVBWxYYMBxv3ScnYM0FERHrXtWtX7NmzB126dBF%2Bb0dHR83n1q1bC79/UVatWoWZM2eyYkPFwsZ9MgRMJoiISO%2BysrIwd%2B5crFu3DhYWFjAyMtI6HxAQINm9vb29NZ8tLCzw5Zdf6lyTlpYmvCoBADk5OejSpYssFRsyPGzcJ0PAZU5ERKR3kyZNKvL8nDlzJL1/Tk4OsrKy0Lx5c0RHR%2Bu80b158yZ69eqFS5cuSRpHQXPnzkXDhg1lqdiQ4WLjPikZkwkiIvrPCQ4Oxrx584q8xsbGBr///rugiF6aNWsW9u7dCwsLC%2BEVGzJsbNwnpeIyJyIiksSff/6JsLAwJCYmQqVSwcrKCl9%2B%2BSXs7Owkv/c333wDd3d3fPrpp1izZo3OeVNTUzRs2FDyOAp6/vw52rZtK/y%2BZNjYuE9KxsoEERHpXUhICAICAtC2bVvUqlULwMulRYcPH8aiRYvw%2BeefC4lj2bJlGD58uJB7EUmFo5ZJyZhMEBGR3rVr1w7%2B/v5o0aKF1vGjR49i4cKFCA0NFRKHk5MTdu3ahcqVKwu5X3HIWbEhwyTnqGWi12EyQUREemdra4vo6Gidh5/s7GzY29vj7NmzQuIIDg5GZGQkXFxcULNmTZ14RK81V0rFhgwLG/dJydgzQUREemdlZYUjR47A2dlZ6/ixY8dQs2ZNYXHMnTsXAHDmzBmdc3KsNQ8ODsaKFSteWbFhMkGFkXPUMtHrMJkgIiK98/HxwciRI%2BHo6Ii6desCePkG/vjx45g1a5awOK5evSrsXsXx8OFDNG/eXOe4o6MjEhMTZYiIDAEb90nJmEwQEZHetW/fHlu3bsX27dvx77//Qq1Ww8rKCuvWrYONjY3weC5duoS7d%2B%2BiQ4cOAICMjAyYmJgIj0MpFRsyLFLvy0L0v2DPBBERCZWUlIQaNWoIudeNGzcwYsQI3LlzB9nZ2bh8%2BTJu376Nnj17YtWqVfj444%2BFxJEnIiICo0ePfmXFprDduokANu6Tchm9/hIiIqI30759e6xevbrQcx07dhQWh5%2BfH9q1a4czZ85o1pmbm5vDy8tLlre9eRWb2rVr499//8X169dhbm6OdevWMZGgVwoJCcHIkSORlpaGpk2bwtbWFqmpqRg4cCAOHjwod3j0jmNlgoiI9K5Ro0awsLBArVq1MH/%2BfFSsWFFzrkmTJrh06ZKQOGxtbXHq1CkYGxvD2toaFy9eBPCyobVFixaIjo4WEkdxiKzYkGFRyqhlosKwMkFERHpXsmRJbN%2B%2BHeXKlUOXLl20HtpVKpWwOCpWrIgnT57oHL916xZKlhTfNqiUig0ZFjbuk5IxmSAiIkmULVsWAQEB8PHxwZAhQxAYGIjc3FyILIh/9tlnGDlyJI4dO4bc3FzExsZix44dGDp0KFxdXYXFkScpKQmbN2%2BGl5cXHj9%2BrHWOCwXoVfIa9wti4z4pAZc5ERGR3hVcynTjxg2MHj0a7733HqKjoxETEyPp/dPT02FqaoqMjAwsWLAAO3bswPPnzwG8rFb07t0bw4cPh7GxsaRxFGRtbY0TJ05g6tSpiI6ORkBAgKaBNv8yLKL82LhPSsZkgoiI9M7T01NnOU9GRgb8/f2xefNmyfd/aNasGTp37oyePXuicePGyM3NxYMHD2BqagozMzNJ712U/AnD1q1bMWfOHHh6esLb2xvW1tbCeknI8Fy9ehXbt29HYmKiZtSyu7u7LKOWifJjMkFERP85oaGhCA0NRVRUFOrVq4eePXvCzc0NFSpUkDUuuSs29N/Dxn2SG5MJIiLSi%2B%2B//x5z584FAIwdO7bIawMCAkSEhLt372Lnzp3YuXMnkpKS0L59e/To0QMtW7YUcv%2BC5K7YkGFq3749%2BvbtC09PT51zXB5HcmMDNhER6UWpUqU0n42NjYv8R5Tq1atj6NChCA8PR3BwMMqWLYvRo0ejffv2WLFihbA48hQ2ycnExAR%2Bfn5MJOiV2LhPSsbKBBER6d3FixdhbW0tdxiFOn/%2BPObMmYOYmBjExsZKfj8lVmzIsLBxn5SMlQkiItI7T09PqNVqucPQSE5ORlBQEDp27IhBgwahTp06CAkJEXJvJVZsyPAoYdQyUWFYmSAiIr379ddfcevWLfTr1w81a9ZEiRIltM6LeHBWq9U4cOAAduzYgaioKDRo0AA9evSAm5ubbBOdlFyxIeVi4z4pGZMJIiLSO1tbW2RlZSErK6vQ81IvL5oyZQrCw8OhUqng5uaGnj17okGDBpLeszjs7Oxw4sQJViHojbBxn5SMyQQREend6dOnizxvb28v6f2//vpr9OjRA1988YWiHtyVULEhItInJhNERCSZzMxMpKSkQKVSoVq1ajoPz%2B8auSs2ZDjYuE%2BGoqTcARAR0X/PkydPMG3aNERERGgenE1MTNC5c2dMmTIFJiYmMkcoj6CgILlDIANRsHGfSKlYmSAiIr0bM2YMUlJSMHjwYNSqVQvAy6bRFStWwMbGBr6%2BvjJHKC9WbOhNsHGflIzJBBER6V3z5s2xf/9%2BVK5cWet4cnIy%2BvTpg0OHDskUmbxYsaG3wcZ9UjLuM0FERHpXokQJlC5dWud4%2BfLl8eLFCxkiUobp06fj3r17WLp0KcLCwhAWFoaAgABcu3YNCxculDs8UigfHx/MmzcPN27cQFpaGtRqtdY/RHJiZYKIiPRu2LBhqFChAsaPH6%2BpTjx8%2BBALFy5ESkoKVq1aJXOE8mDFht4GG/dJydiATUREejdt2jR4e3vD0dERFSpUAACkpqaibt26CAwMlDk6%2BbBiQ2%2BDjfukZKxMEBGRZK5evYrExESo1WpYWVmhcePGcockK1Zs6H/Bxn1SIiYTRESkN87OzlCpVK%2B9LjIyUkA0ypOcnAxvb2/89ddfhVZs8iZfEeXHxn1SMiYTRESkNxs3btR8zs3Nhb%2B/PyZPnqxzXZ8%2BfUSGpTis2NCb4KhlUjImE0REJBlra2tcvHhR7jBkx4oN/S/YuE9KxgZsIiIiiXl5eWk%2BF1WxISoMG/dJyViZICIiybAyUTj%2BXuhNsHGflIyVCSIiIiIF46hlUjImE0REpDebNm3S%2Bjk7OxubN29GwSJ47969RYZFZNCqVauG7du3s3GfFInLnIiISG%2BcnZ1fe41KpXrnG425zImKg437ZAhYmSAiIr35448/5A5BkVixobfBxn0yBKxMEBERSYwVG9IHVrRIiViZICIikhgrNkT0X2UkdwBERERERGSYmEwQEREREdFb4TInIiIiIgVi4z4ZAjZgExERESkQG/fJEDCZICIiIiKit8KeCSIiIiIieitMJoiIiIiI6K0wmSAiIiIiorfCZIKIiIiIiN4KkwkiIiIiInorTCaIiIiIiOitMJkgIiIiIqK3wmSCiIiIiIjeyv8D44eQtp1ynMsAAAAASUVORK5CYII%3D" class="center-img">
</div>
    <div class="row headerrow highlight">
        <h1>Sample</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
        <table border="1" class="dataframe sample">
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
</div>
</div>



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
