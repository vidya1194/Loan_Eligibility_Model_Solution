import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from src.utils.utils import setup_logging

def preprocess_data(df):
    logger = setup_logging()
    
    try:
        #Drop LoanID
        df = df.drop('Loan_ID', axis=1)
            
        # Handling missing values
        imputer = SimpleImputer(strategy='mean')
        df['LoanAmount'] = imputer.fit_transform(df[['LoanAmount']])
        df['Loan_Amount_Term'] = imputer.fit_transform(df[['Loan_Amount_Term']])
        df['Credit_History'] = imputer.fit_transform(df[['Credit_History']])
        
        # Fill missing categorical values with mode
        for column in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
        
        # Encode categorical variables
        
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype=int)
        
        return df
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
