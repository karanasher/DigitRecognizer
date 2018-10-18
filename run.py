'''
All the necessary imports.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import itertools
from utils import show_countplot, plot_confusion_matrix

'''
Housekeeping tasks.
'''
nr.seed(1000)
train = pd.read_csv('input/train.csv')   # Read the train set.
test = pd.read_csv('input/test.csv')     # Read the test set.

'''
Data exploration and preparation.
    Dimensions
    Head
    Unique labels
    Label distribution
    Scaling and normalization
    Label, feature separation
    PCA (maybe) to reduce dimensions from 784 to fewer
'''
# Understand some basic properties.
# Dimensions.
# print(train.shape)  
# print(test.shape)
# print(train.head())
# print(train['label'].unique())    # Unique values.

# Label (digit) distribution.
# show_countplot(train, 'label', 'Digit distribution')

# Label, feature separation (train set.)
# scikit-learn requires the input in the form of a numpy array.
y_train = np.array(train['label'])

X_train = train.drop(labels = ['label'], axis = 1) 

# scikit-learn requires the input in the form of a numpy array.
X_train = X_train.values    
X_test = test.values

# Feature normalization. This will normalize it to [0, 1]. It is easier for 
# an algorithm to converge when the range is [0, 1] instead of [0, 255].
scaler = preprocessing.Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
Modelling.
    Train, validation separation
    Logistic
    Neural network
    Confusion matrix (target is accuracy)
    Error/accuracy plots
'''
# Split the train and the validation set for the fitting.
X_train, X_val, y_train, y_val = ms.train_test_split(X_train, y_train, test_size=0.2)

# Perform logistic regression using different values of 
# the regularization parameter. This will return some 
# useful parameters and model with the least misclassification error.
def regularized_logistic_regression(X_train, y_train, X_val, y_val, lambda_inv=[0.1, 1, 10, 100]):
    logistic_mod_list = []
    misclassification_errors = []

    for C in lambda_inv:
        logistic_mod_list.append(linear_model.LogisticRegression(C=C)) 
        logistic_mod = logistic_mod_list[-1]
    
        # Fit the model. By default, the one-vs-all method is used.
        logistic_mod.fit(X_train, y_train)

        # Perform the prediction on validation set.
        val_probabilities = logistic_mod.predict_proba(X_val)

        # Since one-vs-all method is used, the index of the max element 
        # (predicted probability) will be the predicted score.
        val_scores = val_probabilities.argmax(axis = 1)

        # Compute the confusion matrix.
        conf_mtx = sklm.confusion_matrix(y_val, val_scores) 

        misclassification_errors.append(1 - sklm.accuracy_score(y_val, val_scores))
        print('C = %s | Misclassification error = %0.4f' % (C, misclassification_errors[-1]))

    # Compute the index where you have the error is minimum.
    min_index = misclassification_errors.index(min(misclassification_errors))

    # Compute useful properties to return.
    least_misclassification_error = misclassification_errors[min_index]
    opt_lambda_inv = lambda_inv[min_index]
    logistic_mod = logistic_mod_list[min_index]

    return logistic_mod, least_misclassification_error, opt_lambda_inv, conf_mtx

# Fetch the best logistic regression model.
logistic_mod, least_misclassification_error, opt_lambda_inv, conf_mtx = \
regularized_logistic_regression(X_train, y_train, X_val, y_val)

# The order is which the probabilities are arranged is -> [0 1 2 3 4 5 6 7 8 9].
# print(logistic_mod.classes_)

# Plot the confusion matrix.
# plot_confusion_matrix(conf_mtx, classes = logistic_mod.classes_)   

'''
Final prediction on test set.
'''
# Perform the prediction on test set.
test_probabilities = logistic_mod.predict_proba(X_test) 

# Since one-vs-all method is used, the index of the max element 
# (predicted probability) will be the predicted score.
test_scores = test_probabilities.argmax(axis = 1)

# Generate and save the output DataFrame.
submission_df = pd.DataFrame()
submission_df['ImageId'] = range(test_scores.shape[0])
submission_df['ImageId'] = submission_df['ImageId'] + 1
submission_df['Label'] = test_scores
submission_df.to_csv('output/submission.csv', index=False)