
def dataCorreleation(plt,sns,df):

                df_clean = df.drop('Label', axis=1) ## drop label column
                # correlation matrix
                correlation_matrix = df_clean.corr(numeric_only=True)
                # Heatmap -> graphical representation of data where the individual values contained in a matrix are represented as colors.
                # Heatmaps are perfect for exploring the correlation of features in a dataset.
                # the diagonal of the matrix is always 1, because the correlation between a variable and itself is always 1.
                plt.figure(figsize=(17,15))
                sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True,vmin=-1, vmax=1, linewidths=0.5, square=True)
                plt.title('Correlation Matrix of mtcars variables')

                # Pairplot -> a plot where each feature is compared to each other feature in the dataset.
                # Pairplots are just elaborations on this, showing all variables paired with all the other variables.
                # They are a great method to identify trends for follow-up analysis and, fortunately, are easily implemented in Python!
                # Pairplots are extremely useful when you want to see how different features of your dataset correlate with one another.
                # In a pairplot, the diagonal would be a histogram of each feature.
                # The upper triangle of the pairplot is mirrored across the diagonal.
                # The lower triangle of the pairplot is also mirrored across the diagonal.
                plt.figure(figsize=(5, 5))  # Set the figure size
                # Pairwise feature correlations
                sns.pairplot(df, height=3)  # Adjust height as needed
                plt.show()

                # condition plot 
                baseline = df[df['Label'] == -1]
                graditude = df[df['Label'] == 210]
                natural = df[df['Label'] == 110]
                graph_name = "HRV_TP"
                plt.figure(figsize=(10, 6))
                plt.hist(baseline[graph_name], bins=30, alpha=0.5, label='baseline')
                plt.hist(graditude[graph_name], bins=30, alpha=0.5, label='graditude')
                plt.hist(natural[graph_name], bins=30, alpha=0.5, label='natural')
                plt.xlabel('future values')
                plt.ylabel('frequency')
                plt.title('Histogram of ' + graph_name + ' by condition')
                plt.legend()
                plt.show()

                # boxplot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df[df['Label'].isin([-1, 310, 109])], x='Label', y='HRV_TP')
                plt.xlabel('condition')
                plt.ylabel('future values')
                plt.title('Boxplot of ' + graph_name + ' by condition')
                plt.show()

                # violin plot -> a method of plotting numeric data.
                # It is similar to a box plot, with the addition of a rotated kernel density plot on each side.
                # Violin plots are similar to box plots, except that they also show the probability density of the data at different values.
                # Violin plots are useful for comparing distributions.
                sns.violinplot(x='Label', y='HRV_TP', data=df)
                plt.show()