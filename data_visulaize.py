

def showDatawithDiagrams(dataFrame,plt):
    
     dataFrame.boxplot(figsize=(30,10))
     plt.show()


     dataFrame.hist(figsize=(30,20)) 
     plt.subplots_adjust(hspace=0.5, wspace=0.5) 

     plt.show()