# Computer Assignment 3: Decision Tree Algorithms

## Executive Summary
This is a machine learning exercise that involves building a decision tree algorithm for a dataset containing salaries of people along with seven demographic variables. The dataset contains 48,842 rows and 7 columns. The objective is to classify the data into two categories of income based on the demographic variables. Before building the model, the data quality of the dataset is analyzed for missing values, outliers, and NaNs, and the necessary data cleansing and transformation is performed. The decision tree classifier model is built using the DecisionTreeClassifier algorithm from scikit learn. The decision tree's performance is evaluated by calculating the confusion matrix, accuracy, precision, recall, and F1 score. The decision tree's hyper-parameters are varied, and the performance is compared to determine the best-performing tree with respect to accuracy.

## Packages Installed and Imported

### Packages installed:

* pandas
* numpy
* matplotlib
* scikit-learn

### Packages imported:

* pandas as pd
* numpy as np
* matplotlib.pyplot as plt
* from sklearn.model_selection import train_test_split
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## Observations of Best Decision Tree

Based on the output, the accuracy of the decision tree model is 0.84196, which means that the model correctly predicted the income level of approximately 84% of the test set. The classification report shows that the model has a high precision (0.87) for predicting the '0' class (income level <=50K), which means that the model correctly identified a high percentage of individuals with income level <=50K out of all individuals predicted to have that income level. However, the precision for the '1' class (income level >50K) is lower at 0.71, indicating that the model did not correctly identify as many individuals with income level >50K.

The recall for the '0' class is high at 0.93, which means that the model correctly identified a high percentage of individuals with income level <=50K out of all individuals who actually had that income level. However, the recall for the '1' class is lower at 0.57, indicating that the model did not correctly identify as many individuals with income level >50K out of all individuals who actually had that income level.

The F1-score for the '0' class is high at 0.90, indicating that the model has a good balance between precision and recall for predicting the '0' class. This indicates a high level of precision and recall, suggesting that the model is performing well in both minimizing false positives and false negatives. Furthermore, it implies that the model has a high level of accuracy in identifying positive instances while also minimizing the number of false positives and negatives. However, the F1-score for the '1' class is lower at 0.63, indicating that the model did not have a good balance between precision and recall for predicting the '1' class.

## Insights from the Decision Tree

The msr_bin_encoded was used as the root note. From it _capital_gl_ and education were chosen as the next nodes. The Gini impurity indicates that both nodes are impure reflected in 0.124 and 0.495 for the first two nodes that branch off: the capital_gl and education_bn nodes.
