from exeption import CustomExeption
import sys
import os
import dill

from sklearn.model_selection import GridSearchCV # , RandomizedSearchCV
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

def evaluate_model(X_train, y_train, X_test, y_test, model, hyper_params):
    try:
        # Train the model
        # Inisialisasi GridSearchCV
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=hyper_params, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        # Make predictions
        y_pred = best_model.predict(X_test)
        
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
            'best_parameters': grid_search.best_params_,
            'best_cross_validastion_score': round(grid_search.best_score_, 4)
        }
        
        return best_model, evaluation_report
    
    except Exception as e:
        raise CustomExeption(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomExeption(e, sys)