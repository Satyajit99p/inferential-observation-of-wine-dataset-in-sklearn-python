from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb

wine=datasets.load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)

df['type']=wine.target
df=df[(df['type'] == 0) | (df['type'] == 1)]
df['type'].replace({0 : 'Pinot Noir', 1 : 'Merlot'}, inplace=True)
 # classification based on two wine types: Pinot Noir is rendered as 1 and Merlot is rendered as 0

print("no of datapoints ",df.shape[0])    # no of datapoints  130
print("no of features ", df.shape[1])     # no of features  14

with pd.option_context('display.max_columns', 14):
 print(df.describe())
 #provides numerical analysis of complete datalike mean,frequency,std etc.
 
 print(df.info()) # provides datatype info of cloumns and memory usage.
 
print("features are ",df.columns)
 # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
 #       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
 #       'proanthocyanins', 'color_intensity', 'hue',
 #       'od280/od315_of_diluted_wines', 'proline', 'type']
 
 # type:
 # 0= Pinot Noir type of wine
 # 1= Merlot type of wine
 
print(df.type.value_counts())
 # Merlot        71
 # Pinot Noir    59
 # Thus we observe that there are 71 instances of Merlot wine and 59 instances of Pinot Noir wine
 
print("Pinot Noir type ",(51/130*100),"%")   # Pinot Noir type  39.23076923076923 %
print("Merlot type ", (79/130*100),"%")      # Merlot type  60.76923076923077 %
 # There is appx equal distribution of dataset among the two types.
 
print(df.head())

sb.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')
           # Thus we conclude that data has no missing values

# We take univariate data analysis taking into consideration the 'color_intensity' 
# of wine types in order to differentiate between class 1 and class 0.

sb.FacetGrid(df,hue='type',height=4) \
    .map(sb.distplot, 'color_intensity')        #with color_intensity and displot as grid format.
 # as the progressive lines for both the wine types overlap, color_intensity is not a suitable parameter for comparison.

sb.boxplot(x='hue',y='type',data=df)            # to remove ambiguity, weuse the parameter hue in boxplot format.
 # evidently the wine types overlap each other in the box plot and thus hue is not a suffecient parameter.

sb.violinplot(x='color_intensity',y='type',data=df)  # using color intensity and violin plot format.
 # there is still overlappring which does not add up to concrete conclusion.

sb.catplot(x='total_phenols',y='type',data=df)  #using total phenols and catplot format.
 # insufficient overlapping leading to flawed observation.

#Bivariate analysis using hue and color intensity

sb.FacetGrid(df,hue='type',height=5)\
    .map(plot.scatter,'hue','color_intensity')

# accurate classification is not achieved
# thus it can be concluded that from the given datapoints and parameters we cannot accurately judge the type of wine.

plot.show()








