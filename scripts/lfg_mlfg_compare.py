# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:06:57 2024

@author: jarre
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2
df = pd.read_csv("moutput.csv")
df = df.drop(["seed"],axis="columns")

#x = SelectKBest(chi2,k=23).fit_transform(df.drop(["gtype"],axis="columns"),df["gtype"])
slk = SelectKBest(chi2,k=23)
x = slk.fit_transform(df.drop(["gtype"],axis="columns"),df["gtype"])
names = slk.get_feature_names_out()
y = df["gtype"]
mdd = pd.DataFrame(x)
mdd.columns=names
mdd["gtype"] = y
df = mdd

class1_data = df[df['gtype'] == 'LFG']
class2_data = df[df['gtype'] == 'MLFG']

unique_parameters_class1 = class1_data['parameter'].unique()
unique_parameters_class2 = class2_data['parameter'].unique()

num_params_class1 = len(unique_parameters_class1)
num_params_class2 = len(unique_parameters_class2)

colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"orchid","LFG":"maroon","MLFG":"green","PMLCG":"saddlebrown"}
for k in range(int(num_params_class1/3)+1):
   for l in range(int(num_params_class2/3)+1):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, param_class1 in enumerate(unique_parameters_class1[3*k:3*k+3]):
        for j, param_class2 in enumerate(unique_parameters_class2[3*l:3*l+3]):
            class1_subset = class1_data[class1_data['parameter'] == param_class1]
            class2_subset = class2_data[class2_data['parameter'] == param_class2]
            combined_subset = pd.concat([class1_subset, class2_subset])
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_subset.drop(columns=['gtype', 'parameter']))
            
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['gtype'] = combined_subset['gtype'].values
            
            sns.scatterplot(x='PC1', y='PC2', hue='gtype',palette = colormaps, data=pca_df, ax=axes[i, j])
            axes[i, j].set_title(f'LFG_param={param_class1}, MLFG_param={param_class2}')
            axes[i, j].set_xlabel(f'Principal Component 1 {pca.explained_variance_ratio_[0]:.2f}')
            axes[i, j].set_ylabel(f'Principal Component 2 {pca.explained_variance_ratio_[1]:.2f}')
    
    plt.tight_layout()
    plt.show()
