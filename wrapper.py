

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



dataFrame = pd.read_csv('F:\datamining\data_mining\Dataset_Study2.csv')


# print(dataFrame.head())
# print(dataFrame.shape)


# print("done with data import")


# explore.exploreData(dataFrame)

# print("done with data explore")


# data_visulaize.showDatawithDiagrams(dataFrame,plt)

# print("done with data showing")


# preprocess.precprocessDataforAnalysis(plt,dataFrame)
# preprocess.showboxPlotwithoutOutliers(dataFrame)

# print("done with data preprocessing")


# data_crorreleation.dataCorreleation(plt,sns,dataFrame)

# print("done with data cprreleation")



xComponent,yComponent,x_standardized,x_pca = pca_implementation.createPCA(dataFrame,plt,sns)

print("done with PCA")



featuresRFC,featuresDTC,featuresSFS,xComponentRFE = feature_selection.featureSelectionTechniquesWrapper(xComponent,yComponent,x_standardized)

print("done with feature extract")


model.bulkEvaluate(yComponent,x_pca,featuresRFC,featuresDTC,featuresSFS,xComponentRFE)

print("done with training")







