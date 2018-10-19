'''
This file includes all helper functions.
'''

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

# Label (digit) distribution.
def show_countplot(data, column, title):
    sns.set(style="darkgrid")
    count_plot = sns.countplot(x = column, data = data, order = data[column].value_counts().index)
    plt.title(title)
    plt.show()

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

# Plots misclassification_errors (Y-axis) vs. lambda_inv (X-axis.) 
# This is to visually see the effect of different regularization parameters.
def plot_misclassification_errors(lambda_inv, misclassification_errors):
    plt.plot(lambda_inv, misclassification_errors)
    plt.xlabel('C (lambda_inv)')
    plt.ylabel('Misclassification error')
    plt.title('Validation set error plot')
    plt.xticks(lambda_inv)
    plt.show()