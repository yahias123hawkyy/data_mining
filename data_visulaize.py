def showDatawithDiagrams(df,plt):
    
     df.boxplot(figsize=(20,10)) ## boxplot for all columns
     plt.show()

     # histogram -> representation of the distribution of numerical data.
     # It is an estimate of the probability distribution of a continuous variable.
     # To construct a histogram, the first step is to "bin" the range of values
     # -- that is, divide the entire range of values into a series of intervals --
     # and then count how many values fall into each interval.
     # The bins are usually specified as consecutive, non-overlapping intervals of a variable.
     # The bins (intervals) must be adjacent, and are often (but are not required to be) of equal size.

     df.hist(figsize=(20,10)) ## histogram for all columns
     plt.show()