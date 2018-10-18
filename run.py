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
print(train.shape)  
print(test.shape)
print(train.head())
print(train['label'].unique())    # Unique values.

# Label (digit) distribution.
def show_countplot(data, column, title):
    sns.set(style="darkgrid")
    count_plot = sns.countplot(x = column, data = data, order = data[column].value_counts().index)
    plt.title(title)
    plt.show()

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
    Logistic
    Neural network
    Confusion matrix (target is accuracy)
    Error/accuracy plots
'''
# Split the train and the validation set for the fitting.
X_train, X_val, y_train, y_val = ms.train_test_split(X_train, y_train, test_size = 0.2)

# Fit the model. By default, the one-vs-all method is used.
logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(X_train, y_train)

# The order is which the probabilities are arranged is -> [0 1 2 3 4 5 6 7 8 9].
# print(logistic_mod.classes_)

# Perform the prediction on validation set.
val_probabilities = logistic_mod.predict_proba(X_val)

# Since one-vs-all method is used, the index of the max element 
# (predicted probability) will be the predicted score.
val_scores = val_probabilities.argmax(axis = 1)

# Look at confusion matrix.
# Reference. https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute the confusion matrix.
conf_mtx = sklm.confusion_matrix(y_val, val_scores) 
print('Accuracy  %0.2f' % sklm.accuracy_score(y_val, val_scores))

# Plot the confusion matrix.
plot_confusion_matrix(conf_mtx, classes = logistic_mod.classes_)   

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