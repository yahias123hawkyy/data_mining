


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler


def createPCA(dataFrame,plt,sns):
    
            x = dataFrame.drop('Label', axis=1)  
            y = dataFrame['Label']
            # Standardize features
            x_standardized = StandardScaler().fit_transform(x)

            print(x_standardized[:,0])
            print(x_standardized[:,1])

            x_pca = PCA().fit_transform(x_standardized)
            pca_dataFrame = pd.DataFrame(data=x_pca, columns=[f'PCA{idx}' for idx in range(1, x_pca.shape[1] + 1)])



            # Calculate explained variance ratio by each principal component
            pcaComponentVariances = PCA().fit(x_standardized).explained_variance_ratio_
            print("Explained variance ratios of all components: \n", pcaComponentVariances)

            # Plot explained variance ratios by principal components
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            axes[0].bar(range(1, len(pcaComponentVariances) + 1), pcaComponentVariances, alpha=0.7)
            axes[0].set_xlabel('Principal Components')
            axes[0].set_ylabel('Explained Variance Ratio')
            axes[0].set_title('Explained Variance Ratio by Principal Components (Bar Plot)')
            axes[1].plot(range(1, len(pcaComponentVariances) + 1), pcaComponentVariances, marker='o', linestyle='-', color='b')
            axes[1].set_xlabel('Principal Components')
            axes[1].set_ylabel('Explained Variance Ratio')
            axes[1].set_title('Explained Variance Ratio by Principal Components (Line Plot)')
            plt.tight_layout()
            plt.show()

            # Perform PCA with 5 principal components
            pca = PCA(n_components=5)
            x_pca = pca.fit_transform(x_standardized)

            # Create DataFrame with first 5 principal components and target variable
            pca_dataFrame = pd.DataFrame(data=x_pca, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Principal Component 4', 'Principal Component 5'])
            pca_dataFrame['Label'] = y

            # Plot PCA of first two principal components
            print(x_pca[:, 0])
            plt.figure(figsize=(10, 8))
            scatter_label = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=dataFrame['Label'], cmap='viridis')
            plt.title('PCA of Dataset - First two principal components')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(*scatter_label.legend_elements(), title="Label Class")
            plt.grid(True)
            plt.show()

            # Plot PCA of first three principal components in 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=dataFrame['Label'], cmap='viridis')
            ax.set_title(f'PCA of Dataset - First three principal components')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.legend(*scatter.legend_elements(), title="Labels")
            plt.show()

            # Create DataFrame with 5 PC components
          
            return x, y, x_standardized,x_pca
