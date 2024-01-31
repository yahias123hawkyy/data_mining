
def exploreData(dataFrame):
    
    print(dataFrame.isnull().sum())
    print(dataFrame.info() )
    
    print(dataFrame.dtypes)
   
    print(dataFrame.duplicated().sum()) 
