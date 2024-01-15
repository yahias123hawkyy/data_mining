# PART 0 setup environment
# basic environment setup byt creating one in project itself.
# TO START: source .venv/bin/activate --> provides virtual environment

# PART 1 import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# imoports for PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# imports for 3D plot
from mpl_toolkits.mplot3d import Axes3D
# imports for decision tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# imports for sfs
from sklearn.feature_selection import SequentialFeatureSelector as SFS
# imports for logistic regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# imports for model development
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score




# PART 2 import dataset
df = pd.read_csv('Dataset_Study3.csv')
df.head() ## first 5 rows
df.shape ## 426 rows, 20 columns
df.describe() ## summary statistics


# PART 3 data exploration and EDA(Exploratory Data Analysis)
# checking for missing values, duplicated, and data types.
df.dtypes ## all data types are correct
df.isnull().sum() ## no missing values
df.info() ## get info about data

df.duplicated().sum() ## no duplicated values

# PART 4 data visualization
# Boxplot -> representation of statical data based on the minimum, first quartile, median, third quartile, and maximum.
# baically there are two whisker points representing max and min vlues. In the box itself, there are 50% of data points.
# Before Q1 there are 25% and after Q3 there are 25% of data points. Inside of box there are also mean and median.
# Ourliers -> data points that are far from other data points.

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

# PART 5 data preprocessing
# function to remove outliers from dataset
def find_outliers(df, arg):
    q1 = df[arg].quantile(0.25)
    q3 = df[arg].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df.loc[(df[arg] < lower_bound) | (df[arg] > upper_bound)]
    return outliers

# HRV_VLF -> Heart Rate Variability Very Low Frequency
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


# PART 6 data correlation
# correlation matrix -> a table showing correlation coefficients between variables.
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
graditude = df[df['Label'] == 310]
natural = df[df['Label'] == 109]
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

# PART 7 PCA (Principal Component Analysis)
# PCA is a technique for reducing the number of dimensions in a dataset whilst retaining the majority of information.
# In simple words, pca decreases columns and rows in dataset.
# step 1, covariance --> it's calculation of how much two variables change together.
# step 2, eigenvalues and eigenvectors --> eigenvectors are vectors that do not change direction in transformation.
# step 3, principal component --> it's a linear combination of the original features.
# step 4 ranking principal components --> it's a ranking of the principal components by eigenvalues.
# step 5 number of principal components --> it's a selection of the number of principal components.
# step 6 principal component analysis --> it's a projection of the original data into the space of principal components.

# separate features and labels
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

# PART 8 Future Selection
# decision tree -> a decision support tool that uses a tree-like model of decisions and their possible consequences.
# It is one way to display an algorithm that only contains conditional control statements.
# Decision trees where the target variable is categorical are called classification trees.
# Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# Decision trees are commonly used in operations research and operations management.
# decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute

# Random Forest Classifier for feature importance
model = RandomForestClassifier()
model.fit(x, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, x.columns[indices[f]], importances[indices[f]]))

selected_features_RFC = x[['HRV_DFA_alpha1', 'HRV_VLF', 'HRV_SD1SD2', 'HRV_DFA_alpha2', 'HRV_ApEn']]
selected_features_RFC.head()

# Decision Tree Classifier for feature importance
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(x, y)

# Extracting feature importances
dtree_importances = dtree.feature_importances_
indices = np.argsort(dtree_importances)[::-1]
for f in range(x.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, x.columns[indices[f]], dtree_importances[indices[f]]))

# Selecting features
selected_features_DTC = x[['HRV_DFA_alpha2', 'HRV_LF', 'HRV_HTI', 'HRV_LFHF', 'HRV_Prc20NN']]
selected_features_DTC.head()

# sfs -> Sequential Forward Selection
# Forward Selection using SFS from sklearn
sfs_forward = SFS(estimator=model,
                  n_features_to_select=5,
                  direction='forward',
                  scoring='accuracy',
                  cv=5)
sfs_forward = sfs_forward.fit(x, y)

x.columns[sfs_forward.get_support()==True]

selected_features_SFSF = x[['HRV_MeanNN', 'HRV_pNN50', 'HRV_LFHF', 'HRV_SD1', 'HRV_ApEn']]
selected_features_SFSF.head()

# Backward Elimination using SFS from sklearn
sfs_backward = SFS(estimator=model,
                   n_features_to_select=5,
                   direction='backward',
                   scoring='accuracy',
                   cv=5)
sfs_backward = sfs_backward.fit(x, y)

x.columns[sfs_backward.get_support()==True]

selected_features_SFSB = x[['HRV_MeanNN', 'HRV_RMSSD', 'HRV_Prc80NN', 'HRV_HF', 'HRV_TP']]
selected_features_SFSB.head()

# logistic regression -> a statistical model that in its basic form uses a logistic function to model a binary dependent variable,
# although many more complex extensions exist.

model = LogisticRegression(solver='lbfgs',max_iter=6000)

# Initializing RFE model
rfe = RFE(estimator = model,n_features_to_select= 5)  # selecting 5 features
fit = rfe.fit(x_scaled, y)

# summarize the selection of the attributes
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
x.columns[fit.get_support()==True]

x_rfe = x[x.columns[fit.get_support()==True]]
x_rfe.head()


# PART 9 Model Development (supervised learning)-> a type of machine learning algorithm that uses a known dataset (called the training dataset)
# to make predictions. The training dataset includes input data and response values. The response values can be categories such as
# “spam” or “not spam,” or they can be numbers that correspond to different pricing options for a flight.
# Assuming X is your feature set, y is your target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=45)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# function to build a model and evaluate it
def evaluate_model(X, y, split): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall


# evaluation 
evaluate_model(x_pca,y,0.3)

evaluate_model(selected_features_RFC,y,0.3)
evaluate_model(selected_features_DTC,y,0.3) 
evaluate_model(selected_features_SFSF,y,0.3)  
evaluate_model(selected_features_SFSB,y,0.3)
evaluate_model(x_rfe,y,0.3)
