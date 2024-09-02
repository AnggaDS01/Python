from exeption import CustomExeption
import sys
import os
import dill

from sklearn.metrics import ( 
    accuracy_score, 
    precision_score, 
    recall_score, 
    classification_report, 
    confusion_matrix
)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomExeption(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)
        
        # Prepare evaluation report
        evaluation_report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': confusion,  
        }
        
        return model, evaluation_report
    
    except Exception as e:
        raise CustomExeption(e, sys)