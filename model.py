



from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC






def evaluate_model(X, y, split): 
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

                #  # Initialize and train Support Vector Machine classifier
                # model = SVC(kernel='rbf', random_state=42)
                # model.fit(X_train, y_train)
                
                # # Make predictions
                # y_pred = model.predict(X_test)
                
                # # Calculate evaluation metrics
                # accuracyScore = accuracy_score(y_test, y_pred)
                # precisionScore = precision_score(y_test, y_pred, average='macro')
                # recallScore = recall_score(y_test, y_pred, average='macro')
                
                # # Compute confusion matrix
                # cm = confusion_matrix(y_test, y_pred)

                # # Plot confusion matrix
                # plt.figure(figsize=(8, 6))
                # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                # plt.title('Confusion Matrix')
                # plt.xlabel('Predicted Labels')
                # plt.ylabel('True Labels')
                # plt.show()
    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            accuracyScore = accuracy_score(y_test, y_pred)
            
            precisionScore = precision_score(y_test, y_pred, average='macro')
            recallScore = recall_score(y_test, y_pred, average='macro')

            return accuracyScore, precisionScore, recallScore
            
            
            
        #     cm = confusion_matrix(y_test, y_pred)
    
        #     # Plot confusion matrix
        #     plt.figure(figsize=(8, 6))
        #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        #     plt.title('Confusion Matrix')
        #     plt.xlabel('Predicted Labels')
        #     plt.ylabel('True Labels')
        #     plt.show()
            
            
        #     return accuracyScore, precisionScore, recallScore

        
        
def bulkEvaluate(y,x_pca,selected_features_RFC,selected_features_DTC,selected_features_SFSF,x_rfe):     
        
        print(evaluate_model(x_pca,y,0.3))
        print(evaluate_model(selected_features_RFC,y,0.3))
        print(evaluate_model(selected_features_DTC,y,0.3) )
        print(evaluate_model(selected_features_SFSF,y,0.3)  )
        print(evaluate_model(x_rfe,y,0.3))