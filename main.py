import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_data
from src.preprocess_partition.data_preprocess import preprocess_data
from src.preprocess_partition.data_partition import data_partitioning
from src.models.train_model import train_logistic_regression, train_random_forest, save_model
from src.models.evaluate_model import evaluate_model
from src.models.predict_model import load_model, make_prediction
from src.utils.utils import setup_logging

def main():
    logger = setup_logging()
    logger.info('Starting Loan Eligibility Prediction Model')

    try:
        
        # Load data
        df = load_data('data/credit.csv')
        logger.info("Data loaded")

        # Preprocess data
        df = preprocess_data(df)
        logger.info('Data Preprocessed')

        # Define features and target
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        # Split the data
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = data_partitioning(X,y)
        logger.info('Data Partitioned')
        
        # Training models with feature names
        logistic_regression_model = train_logistic_regression(pd.DataFrame(X_train_scaled, columns=X.columns), y_train)
        random_forest_model = train_random_forest(pd.DataFrame(X_train, columns=X.columns), y_train)

        # Evaluating models with feature names
        lr_accuracy, lr_conf_matrix, lr_class_report = evaluate_model(logistic_regression_model, pd.DataFrame(X_test_scaled, columns=X.columns), y_test)
        rf_accuracy, rf_conf_matrix, rf_class_report = evaluate_model(random_forest_model, pd.DataFrame(X_test, columns=X.columns), y_test)

        
        logger.info(f'Logistic Regression Model Accuracy: {lr_accuracy}')
        logger.info(f'Logistic Regression Confusion Matrix: {lr_conf_matrix}')
        logger.info(f'Random Forest Model Accuracy: {rf_accuracy}')
        logger.info(f'Random Forest Confusion Matrix: {rf_conf_matrix}')
        
        # Save the best model
        save_model(random_forest_model, 'models/best_model.pkl')
        logger.info('Best model saved')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
