# -*- coding: utf-8 -*-
"""Final_Diabetes Classification_team15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HJ80fPqdcOQWL5TdPhQDW1uW_lqRBNnK

# Diabetes Prediction using dataset from the National Institute of Diabetes and Digestive and Kidney Diseases

Pima Indians Diabetes Dataset
Context:

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset

## About the Dataset
Pregnancies :- Number of times a woman has been pregnant

Glucose :- Plasma Glucose concentration of 2 hours in an oral glucose
tolerance test

BloodPressure :- Diastollic Blood Pressure (mm hg)

SkinThickness :- Triceps skin fold thickness(mm)

Insulin :- 2 hour serum insulin(mu U/ml)

BMI :- Body Mass Index ((weight in kg/height in m)^2)

Age :- Age(years)

DiabetesPedigreeFunction :-scores likelihood of diabetes based on family history

Outcome :- 0(doesn't have diabetes) or 1 (has diabetes)

----------------------

## Mounting the Drive
"""

#from google.colab import drive
#drive.mount('/content/drive')

"""Introduction
Early diagnosis of diabetes is important to prevent the onset of complications. In this project, I will analyze the survey data on health indicators that may be associated with diabetes.

There are two main aims of this project:

* to find out the indicators that are most related to diabetes
* to build a model to predict diabetes

## Importing Libraries
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from scipy import stats

"""## Loading the Dataset"""

# google.colab import drive
# Mount Google Drive
#drive.mount('/content/drive')
# Provide the path to your CSV file in Google Drive
#csv_path = '/content/drive/MyDrive/Tanya sbe/Stat Project/dataset/diabetes.csv'
csv_path = './diabetes.csv'

# Read the CSV file
dframe = pd.read_csv(csv_path)
print(dframe)

"""We can observe that all the features are quantitative except for the outcome is
(Yes, the person has diabetes or No, the person hasn't diabetes) categorical
but here in the dataset it is mapped to 1/0

## Variables Classification
"""

quantitative_vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_vars = ['Outcome']

"""# Data Cleaning

## Checking for Null values
"""

# Null value count per column": The isnull() function is applied to the DataFrame,
# resulting in a Boolean mask where True represents a missing value and False represents a non-missing value.
dframe.isnull().sum()

# Displaying DataFrame information
dframe.info()

"""## It is not applicable for the values of Glucose | Blood Pressure | Skin Thickness | Insulin | | BMI to be zero. Thus, if any zero values are found, they should be corrected to NAN."""

# Replacing zero values with NaN in selected columns
replace_value = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in replace_value:
    dframe[col].replace({0: np.nan}, inplace=True)

# Updated DataFrame with replaced values
print(dframe)

"""## To check the count of Null (NAN) Values


"""

# Null value count per column": The isnull() function is applied to the DataFrame,
# resulting in a Boolean mask where True represents a missing value and False represents a non-missing value.
dframe.isnull().sum()

"""<font color='Red'>Null Valus Found</font>

## Impute the Null Values with mean for each feature
"""

# Filling missing values with column means
dframe['BloodPressure'] = dframe['BloodPressure'].fillna(dframe['BloodPressure'].mean())
dframe['SkinThickness']=dframe['SkinThickness'].fillna(dframe['SkinThickness'].mean())
dframe['Glucose']=dframe['Glucose'].fillna(dframe['Glucose'].mean())
dframe['Insulin']=dframe['Insulin'].fillna(dframe['Insulin'].mean())
dframe['BMI']=dframe['BMI'].fillna(dframe['BMI'].mean())
# Updated DataFrame with filled values
print(dframe)

"""The code applies the fillna() method to each column individually. The fillna() method replaces missing values in a column with the specified value, which in this case is the mean value of the corresponding column.

The .mean() method calculates the mean of each column using the mean() function from the NumPy library (np).

## To check the count of Null (NAN) Values
"""

# Creating a new DataFrame to store missing value information
dframe_new = pd.DataFrame()
# Extracting column names
dframe_new['Column'] = dframe.columns
# Calculating missing value counts for each column
dframe_new['Missing_Count'] = [ dframe[col].isnull().sum() for col in dframe.columns ]
# Updated DataFrame with missing value information
print(dframe_new)

"""<font color='green'>Null Values imputed</font>

--------------------------

## Displotting dataframe
"""

# Function to create subplots of distribution plots
def create_displot_subplots(dataframe):
    # Determine the number of features (columns) in the DataFrame
    num_features = len(dataframe.columns)
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(30, 5))
    # Iterate over each column and plot distribution using seaborn
    for i, column in enumerate(dataframe.columns):
        sns.histplot(dataframe[column], ax=axes[i], kde=True)
        axes[i].set_title(column)

    # Adjust the layout and display the subplots
    plt.tight_layout()
    plt.show()

# Call the function to generate distribution subplots for the DataFrame
create_displot_subplots(dframe)

"""* "The <font color='pink'>[GLucose , BMI]</font> distribution exhibits a near-normal shape, with a bell curve and relatively symmetrical data points. However, there are minor deviations from strict normality, such as slightly skewed tails or outliers that cause slight deviations from a perfectly normal distribution."

* The <font color='pink'>[Blood Pressure] </font> distribution exhibits a near-normal shape, but with a prominent peak in the middle indicating a value of unusually high frequency. This central spike adds a distinct feature to an otherwise relatively symmetric distribution.

* The <font color='pink'>[SkinThickness , Insulin ]</font> distribution displays a somewhat normal shape, with a modest resemblance to a bell curve. However, there is a prominent peak in the middle and skewed tails, indicating high departures from a strictly normal distribution.

* The <font color='pink'>[Pregnancies , Age , DiabetesPedigreeFunction ]</font> distribution of the feature is heavily right-skewed, with a majority of the data concentrated towards the lower values and a long tail extending towards the higher values. The distribution lacks a significant portion on the left side, indicating a strong rightward skewness and a presence of extreme values in the upper range.

-----------------------

## Boxplotting dataframe
"""

# Function to create subplots of boxplots
def create_boxplot_subplots(dataframe):
    # Determine the number of features (columns) in the DataFrame
    num_features = len(dataframe.columns)

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(20, 4))

    # Iterate over each column and plot boxplot using matplotlib
    for i, column in enumerate(dataframe.columns):
        axes[i].boxplot(dataframe[column], vert=False)
        axes[i].set_title(column)

    # Adjust the layout and display the subplots
    plt.tight_layout()
    plt.show()

# Call the function to generate boxplot subplots for the DataFrame
create_boxplot_subplots(dframe)

"""## Implementing shapiro_test, remove_outliers by z-score & IQR"""

# Import the necessary libraries
from scipy import stats

# Perform Shapiro-Wilk normality test
def shapiro_test(dataframe, column_name, alpha=0.05):
    data = dataframe[column_name]
    _, p_value = stats.shapiro(data)
    # If p-value is less than alpha (significance level), reject the null hypothesis of normality
    if p_value < alpha:
        return False
    # Failed to reject the null hypothesis of normality
    else:
        return True

# Remove outliers using z-score method
def remove_outliers_zscore(dataframe, column_name, threshold=3):
    data = dataframe[column_name]
    # Calculate z-scores for each data point
    z_scores = np.abs((data - data.mean()) / data.std())
     # Keep data points with z-scores within the specified threshold
    filtered_data = dataframe[z_scores <= threshold]
     # Calculate the number of outliers removed
    outliers_removed = len(data) - len(filtered_data)
    return filtered_data, outliers_removed

# Remove outliers using IQR method
def remove_outliers_iqr(dataframe, column_name, k=2):
    data = dataframe[column_name]
    # Calculate the first quartile
    q1 = data.quantile(0.25)
     # Calculate the third quartile
    q3 = data.quantile(0.75)
    # Calculate the interquartile range
    iqr = q3 - q1
    # Calculate the lower bound for outlier detection
    lower_bound = q1 - k * iqr
     # Calculate the upper bound for outlier detection
    upper_bound = q3 + k * iqr
    # Keep data points within the bounds
    filtered_data = dataframe[(data >= lower_bound) & (data <= upper_bound)]
    # Calculate the number of outliers removed
    outliers_removed = len(data) - len(filtered_data)

    return filtered_data, outliers_removed

"""------------------------------

## Removing Outliers Section
In general if we have outliers we can handle them by:
* if the distribution is Normal ----> use Z-score method
* if ths distribution isn't Normal ------> use Inter_Quartile Method

-------------

### Remove outliers from Pregnancies

Lets conduct shapiro_test to see if the Pregnancies distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

print(shapiro_test(dframe , 'Pregnancies'))

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'Pregnancies')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['Pregnancies'])

"""### Remove outliers from BloodPressure

Lets conduct shapiro_test to see if the BloodPressure distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

print(shapiro_test(dframe , 'BloodPressure'))

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'BloodPressure')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['BloodPressure'])

"""### Remove outliers from SkinThickness

Lets conduct shapiro_test to see if the SkinThickness distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

print(shapiro_test(dframe , 'SkinThickness'))

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'SkinThickness')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['SkinThickness'])

"""### Remove outliers from DiabetesPedigreeFunction

Lets conduct shapiro_test to see if the DiabetesPedigreeFunction distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

shapiro_test(dframe , 'DiabetesPedigreeFunction')

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'DiabetesPedigreeFunction')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['DiabetesPedigreeFunction'])

"""### Remove outliers from Age

Lets conduct shapiro_test to see if the Age distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

shapiro_test(dframe , 'Age')

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'Age')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['Age'])

"""### Remove outliers from BMI

Lets conduct shapiro_test to see if the BMI distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

shapiro_test(dframe , 'BMI')

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'BMI')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['BMI'])

"""### Remove outliers from Insulin

Lets conduct shapiro_test to see if the Insulin distribution is normal
we do this test to choose the appropiate method for removing our outliers
"""

shapiro_test(dframe , 'Insulin')

"""* use interquartile"""

dframe, outliers_removed = remove_outliers_iqr(dframe , 'Insulin')
print(f"Number of Outliers removed: {outliers_removed}")
sns.displot(dframe['Insulin'])

"""---------------------------

## Check For Duplicates:
"""

dframe.duplicated().sum()

"""<font color='green'>No Duplicates found</font>

-----------------------

----------------------------------------

## Calculating the Descriptive Statstics
"""

print(dframe.describe())

# Calculate median
median_values = dframe.median()
# Calculate variance
variance_values = dframe.var()
print("\nMedian for each Feature:")
print(median_values)
print("\nVariance for each Feature:")
print(variance_values)

"""---------------------------------

# Check the Association between the variables & EDA:
"""

# Set the figure size for the heatmap
plt.figure(figsize=(15, 10))
# Create the correlation heatmap
sns.heatmap(dframe.corr(), cbar=True, fmt='.4f', annot=True)
# Set the title of the heatmap
plt.title('Correlation')

"""---------------

---------------------------

## Visualizing the data
"""

sns.pairplot(hue='Outcome',data=dframe,kind="scatter")

"""The resulting pair plot will show scatter plots for each pair of variables in the DataFrame (dframe), with different colors representing different outcomes based on the 'Outcome' column.

orange--> 1 ---> have diabetes

blue ---> 0 ---> don't have diabetes

## Feature vs. Target Visualization
"""

for col in quantitative_vars:
    sns.displot(data=dframe, x=col, col="Outcome", kde= True,color='r')

"""This code snippet loops through each quantitative variable specified in the quantitative_vars list. Within each iteration, a distribution plot is generated using the sns.displot() function.

* data=dframe specifies the DataFrame from which the data is taken.
* x=col sets the variable to be plotted on the x-axis, where col represents the current variable in the iteration.
* col="Outcome" groups the plots based on the unique values in the "Outcome" column, creating separate plots for each value.

By utilizing the kde=True parameter, a kernel density estimate line is added to each plot, providing a smooth representation of the distribution.

--------------------

-----------------------

## Data After Cleaning
"""

print(dframe)

print(dframe.describe())

# Calculate median
median_values = dframe.median()
# Calculate variance
variance_values = dframe.var()
print("\nMedian for each Feature:")
print(median_values)
print("\nVariance for each Feature:")
print(variance_values)

"""---------------------

## Standardize the dataset
"""

def standardize_features(data, columns):
    """
    Standardizes the specified columns in the given DataFrame.

    Parameters:
    - data (DataFrame): The input DataFrame.
    - columns (list): A list of column names to be standardized.

    Returns:
    - DataFrame: The standardized DataFrame.
    """
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    return data
# Standardize the quantitative variables in the DataFrame
dframe = standardize_features(dframe, quantitative_vars)
# Display the standardized DataFrame
print(dframe)

print(dframe.describe())

# Calculate median
median_values = dframe.median()
# Calculate variance
variance_values = dframe.var()
print("\nMedian for each Feature:")
print(median_values)
print("\nVariance for each Feature:")
print(variance_values)

"""<font color="pink"> We can observe that our mean is not exactly equal to zero after standardizing the features using the standardize_features() function is likely due to floating-point precision limitations. When performing arithmetic operations and calculations involving decimal numbers, there can be some small rounding errors but in general all means are close to zero </font>

-----------------

# SPLITTING THE DATASET INTO TRAINING / TEST SET

## Before Splitting
"""

x = dframe.iloc[:,0:-1]
print(x)

y = dframe.iloc[:,-1]
print(y)

"""----------------

## After Splitting
"""

x_train, x_test , y_train , y_test= train_test_split(x,y, test_size=0.2, random_state=41)
print(x_train.head())

print(y_train.head())

"""----------------------

# calculate accuracy from Guassian NB Built in classifier
"""

classifier = GaussianNB()
classifier.fit(x_train, y_train)
ypredict=classifier.predict(x_test)

print("THE PREDICTED SCORE FOR DIABETES IN FEMALE PATIENT USING NAIVE BAYES MODEL IS  :{}%".format(accuracy_score(ypredict,y_test)*100))

"""-------------------

# Implement Naive Bayes from scratch

### To ensure that the assumptions of Naive Bayes are met, we first identify and remove any dependent features by detecting relationships between them. This step is crucial for the proper application of Naive Bayes algorithm.

## 1.Plotting correlation heatmap Revisited
"""

#imorting figure from matplotlib
plt.figure(figsize=(15,10))
#including a heatmap it's data values are correlation between each feature
sns.heatmap(dframe.corr(), cbar =True , fmt = '.4f' , annot = True)
plt.title('Correlation')

"""### From the figure above we can conclude that the relation between each feature is very low as the highest correlation value observed is 0.5 So, no feature will be excluded

##Check the distribution of our features
"""

#adding 8 subplots for each feature
fig, axes = plt.subplots(1, 8, figsize=(18, 8), sharey=True)
#plotting histogram for each feature using seaborn
sns.histplot(dframe, ax=axes[0], x="Pregnancies", kde=True, color='r')
sns.histplot(dframe, ax=axes[1], x="Glucose", kde=True, color='b')
sns.histplot(dframe, ax=axes[2], x="BloodPressure", kde=True, color='g')
sns.histplot(dframe, ax=axes[3], x="SkinThickness", kde=True, color='r')
sns.histplot(dframe, ax=axes[4], x="Insulin", kde=True, color='b')
sns.histplot(dframe, ax=axes[5], x="BMI", kde=True, color='g')
sns.histplot(dframe, ax=axes[6], x="DiabetesPedigreeFunction", kde=True, color='r')
sns.histplot(dframe, ax=axes[7], x="Age", kde=True, color='b')

"""# $P(Y|X) = \frac{{P(X|Y) \cdot P(Y)}}{{P(X)}}$

where P(Y|X) is named posterior for each value of x where x is the features

& P(X|Y) is named likelihood for each value of x

& P(Y) is named prior

& P(X) is named evidence

![(37) The Math Behind Bayesian Classifiers Clearly Explained! - YouTube - Google Chrome 6_19_2023 7_18_30 PM](https://github.com/Youssef-Ashraf71/Diabetes-Classifier/assets/83988379/cfabdc91-a900-4e2a-954a-9a9a1cb58394)

#Calculate P(Y=y) for all possible y where y is the possible outcomes
"""

#creating a function to calculate the prior probability
def calculate_prior(df, Y):
    """
    Calculate the prior probability of each outcome in the target variable.

    Parameters:
    - df: DataFrame containing the dataset.
    - Y: Name of the target variable column.

    Returns:
    - prior: List containing the prior probabilities of each outcome.
    """
    #sorted list of the outcomes 0 & 1
    classes = sorted(list(df[Y].unique()))
    #list to add the prob of outcomes in it
    prior = []
    for i in classes:
      prior.append(len(df[df[Y]==i])/len(df)) # calculating P = number of specific outcome / total number then add it to prior list
    return prior

"""### Verifying the 'calculate_prior' Function"""

non_dia ,dia = calculate_prior(dframe,Y = "Outcome")
print(f"Diabetic : {dia*100} || Non diabetic : { non_dia*100}")
# Count the occurrences of each category in the feature
counts = dframe['Outcome'].value_counts()
# Get the labels and values for the pie chart
labels = counts.index
values = counts.values
# Create the pie chart
#plt.pie(values, labels=labels, autopct='%1.1f%%')
# Add a title to the pie chart
#plt.title('Distribution of Having Diabetes')
# Display the pie chart
#plt.show()

"""#Approach 1: Calculate P(X=x|Y=y) using Gaussian dist."""

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    """
    Calculate the likelihood probability of a feature value given a specific label using Gaussian distribution.

    Parameters:
    - df: DataFrame containing the dataset.
    - feat_name: Name of the feature.
    - feat_val: Value of the feature for which to calculate the likelihood.
    - Y: Name of the target variable column.
    - label: Value of the target variable label(Outcome).

    Returns:
    - p_x_given_y: The likelihood probability of the feature value given the label.
    """
    feat = list(df.columns)
    # decreasing our sample space to be the chosen label only in order to make the probability conditional
    df = df[df[Y]==label]
    # calculating mean and standard deviation of the feature
    mean, std = df[feat_name].mean(), df[feat_name].std()
    # # Calculate the probability Density Function using the Gaussian distribution formula
    p_x_given_y = (1/ (np.sqrt(2*np.pi)*std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2)))
    return p_x_given_y

"""##Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum"""

def naive_bayes_gaussian(df, X, Y):
  """
  Naive Bayes classifier implementation using Gaussian distribution for continuous features.

  Parameters:
  - df: DataFrame containing the training dataset.
  - X: Input features array for prediction.
  - Y: Name of the target variable column.

  Returns:
  - Y_pred: Predicted labels based on the Naive Bayes classifier.
  """
  # Extracting feature names from the DataFrame columns
  features = list(df.columns)[:-1]

  # calculate the prior probabilities
  prior = calculate_prior(df, Y)

  Y_pred = []
  # loop over every data sample
  for x in X:
    # calculate the likelihood probabilities
    labels = sorted(list(df[Y].unique()))
    likelihood = [1]*len(labels)
    for j in range(len(labels)): #iterating over 0 & 1
      for i in range(len(features)): #calculating conditional prob of each feature
        likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j]) #using independency assumptions

    # calculate posterior prob. (numerator only)
    post_prob = [1]*len(labels)
    for j in range(len(labels)):
      post_prob[j] = likelihood[j] * prior[j]
    # taking the larger between the two values of posterior probability
    Y_pred.append(np.argmax(post_prob))

  return np.array(Y_pred)

"""#  MODEL EVALUATION"""

from sklearn.model_selection import train_test_split
#splitting our data set to train and test with 0.8 to 0.2 another time
# because 'the naive_bayes_gaussian' fn take the whole train data as parameter
train, test = train_test_split(dframe, test_size=.2, random_state = 41)

# Using the 'naive_bayes_gaussian' function to predict the target variable ('Outcome') using the test data
y_pred = naive_bayes_gaussian(train, X=x_test.values , Y="Outcome")

"""* Using SKLEARN"""

# Printing the confusion matrix, which shows the predicted and actual values of the target variable
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
confusion_mat = metrics.confusion_matrix(y_test,y_pred)
print(confusion_mat)

# Displaying the predicted score for diabetes in female patients using the Naive Bayes model
print("THE PREDICTED SCORE FOR DIABETES IN FEMALE PATIENT USING NAIVE BAYES MODEL IS  :{}%".format(accuracy_score(y_test, y_pred)*100))
print(classification_report(y_test, y_pred))

"""## Evaluation Metrics

"""

from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(y_test, y_pred))

ylabel = ["Actual [Non-Diab]","Actual [Diab]"]
xlabel = ["Pred [Non-Diab]","Pred [Diab]"]
plt.figure(figsize=(15,6))
sns.heatmap(confusion_mat, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)

"""<font color='Pink'>A confusion matrix</font> is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.

So here:

<font color = 'Pink'>
TN = 68
</font>
||
<font color = 'Pink'>
TP = 29
</font>
||
<font color = 'Pink'>
FN = 10
</font>
||
<font color = 'Pink'>
FP = 12
</font>


This is how I interpret the confusion matrix: 0 = No Diabetes; 1 = Diabetes

TP: Our model predicted 29 women as diabetic and in actual they were diabetic (Model was correct here)

TN: Our model predicted 68 women as non-diabetic and in actual they were non-diabetic (Model was correct here)

FP: Our model predicted 12 women as diabetic and in actual they were non-diabetic (Model was wrong here - "Type 1 error")

FN: Our model predicted 10 women as non-diabetic and in actual they were diabetic (Model was wrong here - "Type 2 error")

------------------

Accuracy: Overall, how often is the classifier correct?

$Accuracy = \frac{{TP + TN}}{{\text{{total}}}} = \frac{{29 + 68}}{{29 + 68 + 10 + 12}}$

--------------------

Precision: Precsion tells us about when model predicts yes, how often is it correct.

TP/predicted yes

$Precision = \frac{TP}{{TP + FP}}= \frac{29}{{29 + 12}}=\frac{29}{{41}} $

So when our model predict 1 and actual it is 1 then it's precision is X%. It should be high as possible.

-------------------
Recall: When the actual value is positive, how often is the prediction correct?

TP/actual yes

$Recall =\frac{{TP}}{{{TP + FN}}}=\frac{29}{{29 + 10}}=\frac{29}{{39}} $

When it's actually yes, how often does model predict yes?

Recall is also known as “sensitivity” and “true positive rate” (TPR).

----------------
Specificity (True Negative Rate):

Specificity measures the proportion of correctly predicted negative samples out of all actual negative samples.

------------

The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.

The formula for the F1 score is: 2 (precision recall) / (precision + recall)

"""

TP = confusion_mat[1, 1]
TN = confusion_mat[0, 0]
FP = confusion_mat[0, 1]
FN = confusion_mat[1, 0]

# Accuracy
print(f"Accuracy of Naive Bayes Classifier is =  {(TP+TN)/(TP+TN+FP+FN)*100} %")

# Precision
Precision = TP / ( TP + FP)
print ('Precision: ', Precision*100,'%')

# Recall
Recall = TP / ( TP + FN )
print ('Recall: ',Recall)

#f1-score
f1 = f1_score(y_test, y_pred)
print('F1-Score: ',f1)

"""both recall and precision are high:

* True Positives: The number of correctly identified positive cases is high. our model is effectively capturing a large portion of the positive instances.
* False Positives: The number of falsely identified positive cases is relatively low. our model is making fewer incorrect positive predictions.
* True Negatives: The number of correctly identified negative cases is high. our model is effectively capturing a large portion of the negative instances.
* False Negatives: The number of falsely identified negative cases is relatively low. our model is making fewer incorrect negative predictions.

In summary, having high recall and high precision implies that our model is performing well in identifying both positive and negative cases accurately, leading to a smaller number of false predictions and a larger number of correct predictions.
"""

dframe['Outcome'].value_counts()

"""We observe that number of people who do not have diabetes is more than people who do which indicates that our data is little imbalanced

However, when dealing with imbalanced datasets, it's important to consider additional evaluation metrics such as:

* Specificity (True Negative Rate):
Specificity measures the proportion of correctly predicted negative samples out of all actual negative samples.

* Balanced Accuracy:
Balanced accuracy takes into account the imbalance in class distribution by calculating the average of sensitivity and specificity.
"""

# Specificity (True Negative Rate)
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
print ('specificity: ',specificity_score(y_test,y_pred))

# Balanced Accuracy:
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print ('balanced_accuracy: ',balanced_accuracy)

"""-----------

## Comparing results to the case of using the NB classifier from standard Python packages
"""

classifier_NB = GaussianNB()
classifier_NB.fit(x_train, y_train)
y_predict_SKLEARN=classifier_NB.predict(x_test)

# Accuracy using Sklearn
print("THE PREDICTED SCORE FOR DIABETES IN FEMALE PATIENT USING NAIVE BAYES MODEL IS  :{}%".format(accuracy_score(y_predict_SKLEARN,y_test)*100))

# Accuracy using Naive Bayes Function we built
print("THE PREDICTED SCORE FOR DIABETES IN FEMALE PATIENT USING NAIVE BAYES MODEL IS  :{}%".format(accuracy_score(y_pred,y_test)*100))

"""When comparing the results of our custom Naive Bayes classifier implementation to the case of using the NB classifier from standard Python packages, it is noteworthy that both approaches achieved <font color="Red"> the same accuracy</font>.

 This indicates that our custom implementation is performing on par with the established NB classifier available in Python packages. It demonstrates the effectiveness of our implementation in accurately classifying the data. This result provides confidence in the reliability and consistency of our custom implementation, validating its ability to produce comparable results to the established solutions in the Python ecosystem.
"""