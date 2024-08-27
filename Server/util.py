import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

import numpy as np

def pricePredict(location, total_sqft, bath, bedrooms):
    if __model is None or __data_columns is None:
        raise Exception("Model and data columns are not loaded.")
    
    try:
        x = np.zeros(len(__data_columns))

        x[__data_columns.index('total_sqft')] = total_sqft
        x[__data_columns.index('bath')] = bath
        x[__data_columns.index('bedrooms')] = bedrooms

        loc_col_name = f'{location}'
        if loc_col_name in __data_columns:
            x[__data_columns.index(loc_col_name)] = 1
        else:
            raise Exception(f"Location column '{loc_col_name}' is missing from data columns.")
        
        prediction = round(__model.predict([x])[0], 2)
        return prediction
    except Exception as e:
        print(f"Error in pricePredict: {e}")
        raise


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("Server/artificats/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]  

    global __model
    if __model is None:
        with open('Server/artificats/housesModel', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(pricePredict('1st Phase JP Nagar',1000, 3, 3))