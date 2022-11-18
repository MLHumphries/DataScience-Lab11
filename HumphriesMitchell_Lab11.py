# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:29:49 2022

@author: hummitl
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#load dataset from scikit learn
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#extract data into a 2D array for processing
x = df.values
print("X shape:", x.shape)

#PCA requires scaled data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
xScaled = scaler.transform(x)
print("X scaled shape::", xScaled.shape)

#Do PCA for 30 features
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(xScaled)
xPca = pca.transform(xScaled)
print("XPCA shape:", xPca.shape)

#Analyze how much each transformed feature explains the variablility of data
xvr = pca.explained_variance_ratio_
print("PCA variance rations:")
for ix, v in enumerate(xvr[:10]):
    print(" ", ix, ":", v)
    
#Use cumulative sum to see how much of the data's variance is explained as we add more PCA features
csum = np.cumsum(xvr)
plt.plot(csum)
plt.xlabel("PCA Components")
plt.ylabel("Explained Variance")  
plt.title("PCA for Iris Dataset")
plt.show()  

#Plot top 2 P?CA features
sns.scatterplot(x=xPca[:, 0], y=xPca[:, 1], hue=iris.target)