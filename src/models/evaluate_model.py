from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils.utils import setup_logging

def evaluate_model(model, X_test, y_test):
    logger = setup_logging()

    try:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        return accuracy, conf_matrix, class_report
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")



