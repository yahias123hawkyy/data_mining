
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split




def evaluate_model(X, y, split): 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            
            return accuracy, precision, recall



def trainModel(y,x_pca,selected_features_RFC,selected_features_DTC,selected_features_SFSF,selected_features_SFSB,x_rfe,x):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=45)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

       
        # evaluation 
        evaluate_model(x_pca,y,0.3)

        evaluate_model(selected_features_RFC,y,0.3)
        evaluate_model(selected_features_DTC,y,0.3) 
        evaluate_model(selected_features_SFSF,y,0.3)  
        evaluate_model(selected_features_SFSB,y,0.3)
        evaluate_model(x_rfe,y,0.3)