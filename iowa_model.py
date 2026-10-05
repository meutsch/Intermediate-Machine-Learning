import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Function for comparing different models
def score_model(model, X_t, X_v, y_t, y_v):
    """
    Fits a model to given training data, makes predictions, and 
    evalutes the mean absolute error between those predictions and 
    the actual values.

    Parameters:
    model: any ML model
    X_t: the training data X-values (independent variable)
    X_v: the validation data X-values
    y_t: the training data y-values (dependent variable)
    y_v: the validation data y-values

    Returns:
    mean_absolute_error(): the mean absolute error for the model
    """
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
            'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

# # Print first few rows of data
# print(X_train.head())

""" Comparing Model Parameters """

# # Define models with different parameters to determine best one
# model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
# model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', 
#                                 random_state=0)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, 
#                                 random_state=0)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
# models = [model_1, model_2, model_3, model_4, model_5]

# # Print the mean absolute error of the models above
# for i in range(0, len(models)):
#     mae = score_model(models[i], X_train, X_valid, y_train, y_valid)
#     print("Model %d MAE: %d" % (i+1, mae))

""" Building a Simple Model """

# Build a new random forest regressor w/o specifying most parameters for now
# default: n_estimators=100, criterion='squared_error', max_depth=None
# min_samples_split=2, random_state=None
my_model = RandomForestRegressor(random_state=1)

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

# Calculate and print MAE of new model
print("My Model: ", score_model(my_model, X_train, X_valid, y_train, y_valid))