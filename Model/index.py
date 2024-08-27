import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from loop import optimise, feature_columns
import pickle
import json

data = pd.read_csv('Python\Data\Bengaluru_House_Data.csv')

values = ['location', 'size', 'total_sqft', 'bath', 'balcony', 'price']
data = data[values]

data = data.fillna({
    'bath' : data.bath.median(),
    'balcony' : data.balcony.median()
})
data = data.dropna()

data['bedrooms'] = data['size'].apply(lambda x: x.split(' ')[0])
data['bedrooms'] = data['bedrooms'].astype('int')
data.drop('size', axis=1, inplace=True)

def convert(x):
    tokens = x.split('-') or x.split(' ')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert)
data = data.dropna()

numeric = ['total_sqft', 'bath', 'balcony', 'price', 'bedrooms']
data[numeric] = data[numeric].astype('int')

location = data['location'].value_counts()
location = pd.DataFrame(location)
other = location[location['count']<10].index
data['location'] = data['location'].apply(lambda x: 'other' if x in other else x)

data['price_per_sqft'] = data['price']*100000/data['total_sqft']

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

data = remove_pps_outliers(data)

bathBeds = data['bath'] > data['bedrooms']
data = data[~bathBeds]
data.drop('price_per_sqft', axis=1, inplace=True)

nominal = ['location']

encodedData = pd.get_dummies(data[nominal])
encodedData = pd.DataFrame(encodedData, index=data.index)
data = pd.concat((data, encodedData), axis=1)
data.drop(['location','location_other'], axis=1, inplace=True)
data.columns = data.columns.astype('str')

x = data.drop('price', axis=1)
y = data['price']

# def bestModel(x, y):
#     algos = {
#         'linear_reg': {
#             'model': LinearRegression(),
#             'params': {
#                 # 'normalize': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [0.001, 1, 2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'ridge': {
#             'model': Ridge(),
#             'params': {
#                 'alpha': [0.01, 0.1, 1, 10]
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion': ['mse', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         }
#     }

#     scores = []
#     for algo_name, config in algos.items():
#         gs = GridSearchCV(config['model'], config['params'], cv=5, return_train_score=False)
#         gs.fit(x, y)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })

#     return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# print(bestModel(x,y))

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1,random_state=1)

model = LinearRegression()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)

def pricePredict(feature_columns, location, sqft, bath, bedrooms):

    x = np.zeros(len(feature_columns))
    
    x[feature_columns.index('total_sqft')] = sqft
    x[feature_columns.index('bath')] = bath
    x[feature_columns.index('bedrooms')] = bedrooms
    
    loc_cols = [col for col in feature_columns if col.startswith('location_')]
    loc_col_name = f'location_{location}'
    if loc_col_name in loc_cols:
        x[feature_columns.index(loc_col_name)] = 1

    prediction = model.predict([x])[0]
    return prediction
