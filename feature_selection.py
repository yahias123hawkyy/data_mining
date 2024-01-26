
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
import numpy as np



def featureSelection(x,y,x_scaled):
    

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
                
                
                return selected_features_RFC,selected_features_DTC,selected_features_SFSF,selected_features_SFSB, x_rfe