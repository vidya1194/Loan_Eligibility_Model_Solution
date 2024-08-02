import pickle
from src.utils.utils import setup_logging

def load_model(filename):
    logger = setup_logging()
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def make_prediction(model, input_data):
    logger = setup_logging()
    try:
        return model.predict(input_data)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
