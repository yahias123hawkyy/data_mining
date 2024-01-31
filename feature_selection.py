
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE

import numpy as np




def featureSelectionTechniquesWrapper(xComoponent,yComponent,x_standardized):
    
   featuresRFC,model= featureSelectionwithrfc(xComoponent,yComponent)
    
    
   featuresDTC= featureSelectionwithDTC(xComoponent,yComponent)
    
    # I used here the model we already created using the random classifier model
   featuresSFS= featureSelectionwithSFS(model,xComoponent,yComponent)
    
    
   xComponentRFE= featureSelectionwithRFE(x_standardized,yComponent,xComoponent)
   
   
   return featuresRFC,featuresDTC,featuresSFS,xComponentRFE
    
    
    
    
    
    
    
    


            
            
            
def featureSelectionwithrfc(xComoponent,yComponent):
    
                    model = RandomForestClassifier()
                    model.fit(xComoponent, yComponent)
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    print("\n \n")

                    print("Feature ranking with random forest classifier:")
                    print("\n \n")

                    for f in range(xComoponent.shape[1]):
                        print("%d. Feature %s (%f)" % (f + 1, xComoponent.columns[indices[f]], importances[indices[f]]))

                    featuresRFC = xComoponent[['HRV_ApEn', 'HRV_SampEn', 'HRV_DFA_alpha2', 'HRV_SD1SD2', 'HRV_Prc20NN']]
                    featuresRFC.head()
                    
                    
                    return featuresRFC,model
                    
                    
                    
                    
                    
def featureSelectionwithDTC(xComoponent,yComponent):
                
                # print("before")
                # print(xComoponent["HRV_MeanNN"])
    
                dtree = DecisionTreeClassifier(random_state=0)
                dtree.fit(xComoponent,yComponent)

                # Extracting feature importances
                dtree_importances = dtree.feature_importances_
                indices = np.argsort(dtree_importances)[::-1]
                
                print("\n \n")
                print("Feature ranking with Decision tree classifier:")
                print("\n \n")

                for f in range(xComoponent.shape[1]):
                    print("%d. Feature %s (%f)" % (f + 1, xComoponent.columns[indices[f]], dtree_importances[indices[f]]))



                # Selecting features
                featuresDTC =  xComoponent[['HRV_ApEn', 'HRV_SampEn', 'HRV_DFA_alpha2', 'HRV_Prc20NN', 'HRV_LFHF']]
                featuresDTC.head()

                # print("after")
                # print(xComoponent["HRV_MeanNN"])
                
                return featuresDTC
                
                

def featureSelectionwithSFS(model,xComponent,yComponent):
    
    
                sfs_forward = SFS(estimator=model,
                                n_features_to_select=5,
                                direction='forward',
                                scoring='accuracy',
                                cv=5)
                sfs_forward = sfs_forward.fit(xComponent, yComponent)

                xComponent.columns[sfs_forward.get_support()==True]

                selected_features = xComponent.columns[sfs_forward.get_support()]

# Print the selected features
                print("Selected Features:")
                for feature in selected_features:
                    print(feature)

                featuresSFS = xComponent[['HRV_MeanNN', 'HRV_SDNN', 'HRV_DFA_alpha2', 'HRV_ApEn', 'HRV_SampEn']]

                featuresSFS.head()
                
                return featuresSFS
    
    
    
def featureSelectionwithRFE(x_standardized,yComponent,xComponent):
    
                model = LogisticRegression(solver='lbfgs',max_iter=6000)

    
                rfe = RFE(estimator = model,n_features_to_select= 5) 
                fit = rfe.fit(x_standardized, yComponent)


                print("Feature Ranking: %s" % (fit.ranking_))
                print("Number of Features: %s" % (fit.n_features_))
                print("Selected Features: %s" % (fit.support_))
                
                xComponent.columns[fit.get_support()==True]

                xComponentRFE = xComponent[xComponent.columns[fit.get_support()==True]]
                xComponentRFE.head()
                
                
                return xComponentRFE
                
                