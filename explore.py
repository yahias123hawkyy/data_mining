
def exploreData(df):
    
    df.dtypes ## all data types are correct
    df.isnull().sum() ## no missing values
    df.info() ## get info about data
    df.duplicated().sum() ## no duplicated values
