from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.utils.utils import setup_logging

def data_partitioning(X, y):
        
    logger = setup_logging()
    
    try:
        # Standard Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123,stratify=y)
        
        # Min-Max Scaling (optional, depending on your use case)
        minmax_scaler = MinMaxScaler()
        X_train_scaled = minmax_scaler.fit_transform(X_train)
        X_test_scaled = minmax_scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")