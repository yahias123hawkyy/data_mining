

# PART 1 import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import explore
import data_visulaize 
import preprocess

import data_crorreleation
import pca_implementation

import feature_selection

import model



df = pd.read_csv('Dataset_Study4.csv')
df.head() ## first 5 rows
df.shape ## 426 rows, 20 columns
df.describe() ## summary statistics


explore.exploreData(df)



data_visulaize.showDatawithDiagrams(df,plt)



preprocess.precprocessDataforAnalysis(plt,df)



data_crorreleation.dataCorreleation(plt,sns,df)



x,y,x_scaled,x_pca= pca_implementation.createPCA(df,plt,sns)




selected_features_RFC,selected_features_DTC,selected_features_SFSF,selected_features_SFSB,x_rfe=feature_selection.featureSelection(x,y,x_scaled)



model.trainModel(y,x_pca,selected_features_RFC,selected_features_DTC,selected_features_SFSF,selected_features_SFSB,x_rfe,x)








