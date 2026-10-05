import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# # Function for comparing different approaches
# def score_dataset(X_train, X_valid, y_train, y_valid):
#     """
#     Builds a random forest regressor, fits it to the training data, 
#     makes predictions and evaluates the mean absolute error between 
#     the predictions and the actual values.

#     Parameters: 
#     X_train: the training data X-values (independent variable)
#     X_valid: the validation data X-values
#     y_train: the training data y-values (dependent variable)
#     y_valid: the validation data y-values

#     Returns: 
#     mean_absolute_error(): the mean absolute error for the model
#     """
#     model = RandomForestRegressor(n_estimators=10, random_state=0)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_valid)
#     return mean_absolute_error(y_valid, preds)

# Read the data
X_full = pd.read_csv('melb_data.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# # Obtain target and predictors
# y = X_full.SalePrice
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
#             'BedroomAbvGr', 'TotRmsAbvGrd']
# X = X_full[features].copy()
# X_test = X_test_full[features].copy()

# # Break off validation set from training data
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, 
#                                                       test_size=0.2,
#                                                       random_state=0)

# # Get names of columns with missing values
# cols_with_missing = [col for col in X_train.columns
#                      if X_train[col].isnull().any()]

# print(cols_with_missing)

# # Drop columns in training and validation data
# reduced_X_train = X_train.drop(cols_with_missing, axis=1)
# reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# print("MAE from Approach 1 (Drop columns with missing values):")
# print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))