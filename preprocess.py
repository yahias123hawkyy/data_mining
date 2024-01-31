


# def find_outliers(dataFrame, featureName):
#     q1 = dataFrame[featureName].quantile(0.25)
#     q3 = dataFrame[featureName].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - (1.5 * iqr)
#     upper_bound = q3 + (1.5 * iqr)
#     outliers = dataFrame.loc[(dataFrame[featureName] < lower_bound) | (dataFrame[featureName] > upper_bound)]
#     return outliers



# def find_outliers(dataFrame, featureName):
#     q1 = dataFrame[featureName].quantile(0.25)
#     q3 = dataFrame[featureName].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     outliers = dataFrame.loc[~dataFrame[featureName].between(lower_bound, upper_bound)]
#     return outliers


from matplotlib import pyplot as plt
import pandas as pd


def find_outliers(dataFrame):
    outliers = pd.DataFrame()
    for column in dataFrame.columns:
        q1 = dataFrame[column].quantile(0.25)
        q3 = dataFrame[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers[column] = dataFrame.loc[(dataFrame[column] < lower_bound) | (dataFrame[column] > upper_bound)][column]
    return outliers

# Assuming 'dataFrame' is your DataFrame
def showboxPlotwithoutOutliers(dataFrame):
            outliers = find_outliers(dataFrame)

            for column in outliers.columns:
                median = dataFrame[column].median()
                dataFrame.loc[outliers.index, column] = median
                
            plt.figure(figsize=(10,6))
            dataFrame.boxplot()
            plt.title('Boxplot of Data without Outliers')
            plt.xticks(rotation=45)
            plt.show()



# def precprocessDataforAnalysis(plt,df):
    
#             preprocessOfeachFeature('HRV_VLF',plt,df)


#             preprocessOfeachFeature('HRV_LF',plt,df)



#             preprocessOfeachFeature('HRV_HF',plt,df)
            


#             preprocessOfeachFeature('HRV_TP',plt,df)
            
            

# def preprocessOfeachFeature(featureName: str,plt,dataFrame):
    
    
#             plt.figure(figsize=(10,8))
#             dataFrame.boxplot(column=featureName,grid=False,vert=False)
#             plt.show()
            
            
#             dataFrame[featureName].hist()
#             plt.show()
            
            
#             outliers = find_outliers(dataFrame, featureName) 
#             outliers.shape 
#             dataFrame[featureName].loc[outliers.index] = dataFrame[featureName].median() 
#             plt.figure(figsize=(10,8)) 
            
#             dataFrame.boxplot(column=featureName,grid=False,vert=False) 
#             plt.show()
            
#             dataFrame[featureName].hist()
#             plt.show()
    