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
#print("no of datapoints ",df.shape[0])
#print("no of features ", df.shape[1])
print("features are ",df.columns)
 # 0= Pinot Noir class of wine
 # 1= Merlot class of wine
#print(df.type.value_counts())
#print("Pinot Noir type ",(51/130*100),"%")
#print("Merlot type ", (79/130*100),"%")
print(df.head())

#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
#       'proanthocyanins', 'color_intensity', 'hue',
#       'od280/od315_of_diluted_wines', 'proline', 'type']

#sb.FacetGrid(df,hue='type',height=4) \
#    .map(sb.distplot, 'color_intensity')        #with color_intensity
# with hue
#    .map(sb.distplot,'hue')

#sb.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis'
           #data has no missing values
#sb.boxplot(x='hue',y='type',data=df)

#as the bopx plots are overlapping hue of wine cannot be used to classify

# using color intensity
#sb.violinplot(x='color_intensity',y='type',data=df)

#using total phenols
#sb.catplot(x='total_phenols',y='type',data=df)

#Bivariate analysis using hue and color intensity
#accurate classification is not achieved
#sb.FacetGrid(df,hue='type',height=5)\
#    .map(plot.scatter,'hue','color_intensity')


plot.show()




#print(df.describe())







#print(df.columns.values)

#   ['alcohol' 'malic_acid' 'ash' 'alcalinity_of_ash' 'magnesium'
#   'total_phenols' 'flavanoids' 'nonflavanoid_phenols' 'proanthocyanins'
#   'color_intensity' 'hue' 'od280/od315_of_diluted_wines' 'proline']

#print(df.info())






