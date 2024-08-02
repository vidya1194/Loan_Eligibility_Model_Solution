from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from src.utils.utils import setup_logging
import pickle


def train_logistic_regression(X,y):
    
    logger = setup_logging()
    
    try:
    
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Set up KFold cross-validation
        kfold = KFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=kfold)
        
        logger.info(f'Logistic Regression Cross-Validation Accuracy scores: {scores}')
        logger.info(f'Logistic Regression Mean accuracy: {scores.mean()}')
        logger.info(f'Logistic Regression Standard deviation: {scores.std()}')
        
        return model
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def train_random_forest(X, y):
    
    logger = setup_logging()
    
    try:    
        model = RandomForestClassifier()
        model.fit(X, y)
        
        # Set up KFold cross-validation
        kfold = KFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=kfold)
        
        logger.info(f'Random Forest Cross-Validation Accuracy scores:{scores}')
        logger.info(f'Random Forest Mean accuracy:{scores.mean()}')
        logger.info(f'Random Forest Standard deviation:{scores.std()}')
        
        return model
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def save_model(model, filename):
    logger = setup_logging()
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
