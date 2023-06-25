# 💊 Diabetes Prediction using Naive Bayes Classifier

This project aims to predict the presence of diabetes in individuals using the Naive Bayes Classifier algorithm. The dataset used for this project contains various features related to diabetes, such as glucose level, blood pressure, BMI, and age. By training a Naive Bayes Classifier on this dataset, we can predict whether an individual has diabetes or not based on their feature values.

## 📁 Project Structure

The project is organized into the following files:

1. `Final_Diabetes_Classification`: This Jupyter Notebook contains the main code for data preprocessing, model training, evaluation, and result analysis. It provides a step-by-step guide to the project implementation.

2. `diabetes.csv`: This is the dataset file in CSV format that contains the information about the individuals and their corresponding diabetes status.

3. `README.md`: This file provides an overview of the project, its purpose, and the files included.

## 📚 Prerequisites

To run the project successfully, the following libraries need to be installed:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- scipy

These libraries can be installed using the following command:

```
pip install pandas numpy seaborn matplotlib sklearn scipy

```

## 📝 Getting Started

1. Clone the repository to your local machine or download the project files.

2. Open the `diabetes_prediction.ipynb` notebook using Jupyter Notebook or any compatible Python development environment.

3. Execute the cells in the notebook sequentially to perform data preprocessing, model training, evaluation, and result analysis.

4. Follow the instructions and comments provided in the notebook for a detailed understanding of each step.

5. Explore the results and analysis presented in the notebook to gain insights into the model's performance and the factors influencing diabetes prediction.

## 📊 Dataset

The dataset used in this project, `diabetes.csv` from [kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) , contains information about various features related to diabetes in individuals. The features include:

- Pregnancies: Pregnancies: Number of times a woman has been pregnant.
- Glucose: Plasma Glucose concentration measured after a 2-hour oral glucose tolerance test.
- BloodPressure: Diastolic Blood Pressure in (mmHg)
- SkinThickness: Triceps skin fold thickness in (mm)
- Insulin: 2-hour serum insulin measured in (mu U/ml)
- BMI: Body Mass Index calculated as weight in kilograms divided by height in meters squared.
- DiabetesPedigreeFunction: A score that estimates the likelihood of diabetes based on family history
- Age: Age in years
- Outcome: Presence or absence of diabetes (0 = No, 1 = Yes)

## 💻 Naive Bayes Classifier

The Naive Bayes classifier is a popular algorithm for supervised learning, particularly in cases where the features are independent of each other. It is based on the Bayes' theorem and assumes that all features are conditionally independent given the class variable.

In our project, we have employed the Naive Bayes classifier to predict the target variable, which is related to the presence or absence of a certain condition (e.g., diabetes). The Naive Bayes classifier calculates the probability of each class given the observed feature values and makes predictions based on the highest probability.

The steps involved in implementing the Naive Bayes classifier are as follows:

1. **Data Preprocessing:** We perform necessary data preprocessing steps, such as handling Null values, removing Outliers, and standardizing the quantitative features.

2. **Splitting the Dataset:** We split the dataset into training(80%) and testing(20%) sets. The training set is used to train the Naive Bayes classifier, while the testing set is used to evaluate its performance.

3. **Training the Classifier:** We fit the Naive Bayes classifier to the training data, estimating the parameters needed to calculate the class probabilities and feature likelihoods.

4. **Making Predictions:** Once the classifier is trained, we can use it to make predictions on new unseen data by calculating the posterior probabilities for each class and selecting the class with the highest probability as the predicted class.

5. **Evaluating Performance:** We assess the performance of the Naive Bayes classifier using various metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into the classifier's ability to correctly classify instances from the testing set.
   We observe that number of people who do not have diabetes is more than people who do which indicates that our data is little imbalanced
   However, when dealing with imbalanced datasets, it's important to consider additional evaluation metrics such as:

- Specificity (True Negative Rate)
- Balanced Accuracy

  The Naive Bayes classifier is known for its simplicity, efficiency, and ability to handle high-dimensional data. However, it assumes independence between features, which may not hold true in some real-world scenarios. Despite this assumption, Naive Bayes classifiers have been successfully applied in various domains, including text classification, spam filtering, and medical diagnosis.

## 📊 Plots & EDA

- [Click Me](https://github.com/Youssef-Ashraf71/Diabetes-Classifier/tree/main/plots)

## 🧾🔎 Results and Discussion

The project evaluates the performance of the Naive Bayes Classifier in predicting diabetes presence. The evaluation includes metrics such as accuracy, precision, recall, Specificity, balanced_accuracy, and F1-score. The results and their analysis are presented in the `Final_Diabetes_Classification` notebook.


<p align="center">
  <img src="https://github.com/Youssef-Ashraf71/Diabetes-Classifier/assets/83988379/3921b296-15d6-4f6c-8a1f-c9040a2b11b2" alt="Image" />
</p>

## 📈 Conclusion

In conclusion, this project demonstrates the application of the Naive Bayes Classifier algorithm for diabetes prediction. By training the model on the provided dataset, we can predict the likelihood of diabetes based on the individual's features. The project provides valuable insights into the model's performance and highlights the factors influencing the prediction.
