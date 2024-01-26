def find_outliers(df, arg):
    q1 = df[arg].quantile(0.25)
    q3 = df[arg].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df.loc[(df[arg] < lower_bound) | (df[arg] > upper_bound)]
    return outliers

# HRV_VLF -> Heart Rate Variability Very Low Frequency

def precprocessDataforAnalysis(plt,df):
    
            plt.figure(figsize=(10,8))
            df.boxplot(column='HRV_VLF',grid=False,vert=False)
            plt.show()
            # histogram for HRV_VLF
            df['HRV_VLF'].hist()
            plt.show()
            # remove outliers from HRV_VLF
            outliers = find_outliers(df, 'HRV_VLF') 
            outliers.shape ## 15 rows, 20 columns
            df['HRV_VLF'].loc[outliers.index] = df['HRV_VLF'].median() ## replace outliers with median value
            plt.figure(figsize=(10,8)) ## boxplot after removing outliers
            df.boxplot(column='HRV_VLF',grid=False,vert=False) 
            plt.show()
            # histogram for HRV_VLF after removing outliers
            df['HRV_VLF'].hist()
            plt.show()


            # HRV_LF -> Heart Rate Variability Low Frequency
            plt.figure(figsize=(10,8))
            df.boxplot(column='HRV_LF',grid=False,vert=False)
            plt.show()
            # histogram for HRV_LF
            df['HRV_LF'].hist()
            plt.show()
            # remove outliers from HRV_LF
            outliers = find_outliers(df, 'HRV_LF')
            outliers.shape ## 15 rows, 20 columns
            df['HRV_LF'].loc[outliers.index] = df['HRV_LF'].median() ## replace outliers with median value
            plt.figure(figsize=(10,8)) ## boxplot after removing outliers
            df.boxplot(column='HRV_LF',grid=False,vert=False)
            plt.show()


            # HRV_HF -> Heart Rate Variability High Frequency   
            plt.figure(figsize=(10,8))
            df.boxplot(column='HRV_HF',grid=False,vert=False)
            plt.show()
            df['HRV_HF'].hist()
            plt.show()
            outliers = find_outliers(df, 'HRV_HF')
            outliers.shape ## 15 rows, 20 columns
            df['HRV_HF'].loc[outliers.index] = df['HRV_HF'].median() ## replace outliers with median value


            # HRV_TP -> Heart Rate Variability Total Power
            plt.figure(figsize=(10,8))
            df.boxplot(column='HRV_TP',grid=False,vert=False)
            plt.show()
            df['HRV_TP'].hist()
            plt.show()
            outliers = find_outliers(df, 'HRV_TP')
            outliers.shape ## 15 rows, 20 columns
            df['HRV_TP'].loc[outliers.index] = df['HRV_TP'].median() ## replace outliers with median value
            df.boxplot(figsize=(20,12)) ## boxplot for all columns
            plt.show()
            df.hist(figsize = (12,15))
            plt.show()
