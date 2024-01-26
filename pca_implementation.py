


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler


def createPCA(df,plt,sns):
            x = df.drop('Label', axis=1) ## drop label column
            y = df['Label'] ## label column

            # standardize the features
            x_scaled = StandardScaler().fit_transform(x)

            # perform PCA
            x_pca = PCA().fit_transform(x_scaled)

            # dataframe for PCA
            pca_df = pd.DataFrame(data=x_pca, columns=[f'PCA{i}' for i in range(1, x.shape[1] + 1)])

            # concatenate labels to dataframe
            pca_df_labels = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)

            # ratio of variance explained by each principal component
            explained_variance = PCA().fit(x_scaled).explained_variance_ratio_
            print("explained variance ratio:", explained_variance)

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

            # plot bar
            axes[0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
            axes[0].set_xlabel('Principal Components')
            axes[0].set_ylabel('Explained Variance Ratio')
            axes[0].set_title('Explained Variance Ratio by Principal Components (Bar Plot)')

            # plot line
            axes[1].plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-', color='b')
            axes[1].set_xlabel('Principal Components')
            axes[1].set_ylabel('Explained Variance Ratio')
            axes[1].set_title('Explained Variance Ratio by Principal Components (Line Plot)')

            plt.tight_layout()
            plt.show()

            # elbow method -> basically it's a method to find the optimal number of clusters in k-means clustering.
            # we take number which is the point after which the distortion/inertia start decreasing in a linear fashion.

            pca = PCA(n_components=5)
            x_pca = pca.fit_transform(x_scaled)

            labels = {
                str(i): f'PC{i + 1} ({var:.1f}%)'
                for i, var in enumerate(pca.explained_variance_ratio_ * 100)
            }

            # Create a DataFrame with the principal components and the target variable
            pca_df = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
            pca_df['Label'] = y

            plt.figure(figsize=(10, 8))
            scatter_label = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df['Label'], cmap='viridis')
            plt.title('PCA of Dataset - First two principal components')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(*scatter_label.legend_elements(), title="Label Class")
            plt.grid(True)
            plt.show()

            # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=df['Label'], cmap='viridis')
            ax.set_title(f'PCA of Dataset - First three principal components')
            ax.set_xlabel('Principal Component 1')        
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.legend(*scatter.legend_elements(), title="Labels")
            plt.show()

            # Create DataFrame with 5 PC components
            df_pca = pd.DataFrame(PCA(n_components=5).fit_transform(x_scaled), columns=[f'PC{i+1}' for i in range(5)])
            df_pca['Label'] = df['Label'].values

            # Scatter Pairplot
            sns.set_theme(style="ticks")
            sns.pairplot(df_pca, hue="Label", diag_kind='hist', palette='Dark2')
            plt.show()
            
            
            return x, y, x_scaled,x_pca
