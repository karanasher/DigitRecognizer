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
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import sklearn.metrics as sklm
import itertools
from utils import show_countplot, plot_confusion_matrix, plot_misclassification_errors

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
'''
# Understand some basic properties.
# Dimensions.
print(train.shape)  
print(test.shape)
print(train.head())
print(train['label'].unique())    # Unique values.

# Label (digit) distribution.
show_countplot(train, 'label', 'Digit distribution')

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
    Neural network
    Confusion matrix (target is accuracy)
    Error/accuracy plots
'''
# Split the train and the validation set for the fitting.
X_train, X_val, y_train, y_val = ms.train_test_split(X_train, y_train, test_size=0.2)  

# Tuning model hyperparameters.
param_grid = {"alpha": [0.001, 0.003, 0.01]}
nn_clf = MLPClassifier(hidden_layer_sizes=(200,200,200), max_iter=200)
nn_clf = ms.GridSearchCV(estimator=nn_clf, param_grid=param_grid, cv=4, scoring='accuracy')
nn_clf.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
print('Optimum hyperparameters are,')
print(nn_clf.best_params_)

# Fit the neural net model. Used GridSearchCV to find the optimum alpha.
nn_mod = MLPClassifier(hidden_layer_sizes=(500,500,500), alpha=nn_clf.best_params_['alpha'], max_iter=400)
nn_mod.fit(X_train, y_train)
val_scores = nn_mod.predict(X_val)

# Compute confusion matrix.
conf_mtx = sklm.confusion_matrix(y_val, val_scores)

# Plot the confusion matrix.
plot_confusion_matrix(conf_mtx, classes = nn_mod.classes_) 

'''
Final prediction on test set.
'''
# Perform the prediction on test set.
test_scores = nn_mod.predict(X_test)

# Generate and save the output DataFrame.
submission_df = pd.DataFrame()
submission_df['ImageId'] = range(test_scores.shape[0])
submission_df['ImageId'] = submission_df['ImageId'] + 1
submission_df['Label'] = test_scores
submission_df.to_csv('output/submission.csv', index=False)