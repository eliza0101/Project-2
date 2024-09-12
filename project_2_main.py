# Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# Load dataset
transaction_cleaned = pd.read_csv('transaction_cleaned.csv')

# Identify non-numeric columns
non_numeric_cols = transaction_cleaned.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Check datatypes
transaction_cleaned.dtypes

# Checking what columns there are
columns = transaction_cleaned.columns.tolist()
print(columns)

# Wrapping the columns for display
pd.set_option('display.max_columns', None)
print(transaction_cleaned)


# Create Exploratory Data Analysis

# EDA 1 - Missing Values Analysis
# Attempt to identify columns with missing data to decide on removal.
missing_percentage = transaction_cleaned.isnull().mean() * 100
plt.figure(figsize=(120, 90))
sns.barplot(x=missing_percentage.index, y=missing_percentage)
plt.xticks(rotation=90)
plt.title('Missing Values Percentage by Column')
plt.ylabel('Percentage of Missing Values')
plt.xlabel('Column Names')
plt.show()
missing_percentage

# Sort by percentage of missing values
# Attempt to identify what is best percentage for removal
sorted_df_desc = missing_percentage.sort_values(ascending=False)
sorted_df_desc

# Filter columns with missing percentage greater than 7
missing_percentage_filtered = missing_percentage[missing_percentage > 7]

# Plot the filtered missing percentages
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_percentage_filtered.index, y=missing_percentage_filtered)
plt.xticks(rotation=90)
plt.title('Missing Values Percentage by Column (Filtered)')
plt.ylabel('Percentage of Missing Values')
plt.xlabel('Column Names')
plt.show()

# Identify columns to drop
columns_to_drop = missing_percentage_filtered.index
columns_to_drop

# Drop columns with more than 7% missing values
transaction_cleaned = transaction_cleaned.drop(columns=columns_to_drop)
transaction_cleaned

# EDA 2 - Numerical Data Analysis
# Attempt to understand the distribution and detect outliers in transaction amounts.
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(transaction_cleaned['TransactionAmt'], bins=30, color='skyblue', kde=True)
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x=transaction_cleaned['TransactionAmt'])
plt.title('Box Plot of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.show()

# Finding - Most transactions are under $200, but does not show which amount are prone to fraud

# Filter fraudulent transactions
transaction_cleaned_fraud = transaction_cleaned[transaction_cleaned['isFraud']==1]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(transaction_cleaned_fraud['TransactionAmt'], bins=30, color='skyblue', kde=True)
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x=transaction_cleaned_fraud['TransactionAmt'])
plt.title('Box Plot of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.show()

# Filter non-fraudulent transactions
transaction_cleaned_notfraud = transaction_cleaned[transaction_cleaned['isFraud']==0]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(transaction_cleaned_notfraud['TransactionAmt'], bins=30, color='skyblue', kde=True)
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x=transaction_cleaned_notfraud['TransactionAmt'])
plt.title('Box Plot of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.show()

# EDA 3 - Visualize Class Distribution Non-Fraudulent
plt.figure(figsize=(6, 4))
sns.countplot(x='DeviceType_mobile', data=transaction_cleaned_notfraud)
plt.title('Distribution of DeviceType_mobile')
plt.xlabel('DeviceType_mobile')
plt.ylabel('Frequency')
plt.show()

# Finding - More fraud with transactions through non-mobile devices

# EDA 3 - Visualize Class Distribution Fraudulent
plt.figure(figsize=(6, 4))
sns.countplot(x='DeviceType_mobile', data=transaction_cleaned_fraud)
plt.title('Distribution of DeviceType_mobile')
plt.xlabel('DeviceType_mobile')
plt.ylabel('Frequency')
plt.show()

# Finding - More fraud with transactions through non-mobile devices

# EDA 4 - Categorical Data Analysis
# Attempt to Examine the distribution of fraud instances.
def plot_categorical_distribution(column):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=column, data=transaction_cleaned)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

plot_categorical_distribution('isFraud')

# EDA 5 - Correlation Analysis
# Attempt to identify relationships between numerical features, to detect strong correlations that could inform feature selection.
corr_matrix = transaction_cleaned.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Model Development

# Handle NaN values and replace with mean
df_mean = transaction_cleaned.fillna(transaction_cleaned.mean())
df_mean

df_mean.info()

df_med = transaction_cleaned.fillna(transaction_cleaned.median())
df_med

#Review data 
df_mean.dtypes

X = df_mean.drop(columns=('isFraud'))
X

# Define target(y)
y = df_mean['isFraud']
y

# Apply train_test_split to data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# Display X_train
X_train

df_mean.info()

nan_data = X_train.isnull().sum()
print(nan_data)

#Display X_test
X_test

# Check the number of fraud vs. not fraud('isFraud')
# using value_counts
df_mean['isFraud'].value_counts()

# Calculate value counts with percentages
value_counts = transaction_cleaned['isFraud'].value_counts(normalize=True) * 100
# Display the percentages
print(value_counts)

# Scale the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
# Create a `LogisticRegression` function to lr_model 
lr_model = LogisticRegression(max_iter=1000, random_state=15)
# Fit train data to LogisticRegression model
lr_model.fit(X_train_scaled, y_train)

# Print the logistic regression Train and Test
print(f"Logistic Regression Training data :{lr_model.score(X_train_scaled, y_train)}")
print(f"Logistic Regression Testing data :{lr_model.score(X_test_scaled, y_test)}")

# Create Prediction based on the Logistic Regression model fitted
predictions = lr_model.predict(X_train_scaled)
# Convert and display predictions vs actual data to a DataFrame
fraud_results_df = pd.DataFrame({"Prediction": predictions, "Actual": y_train})
fraud_results_df

# Predictions applied to testing data
test_predictions = lr_model.predict(X_test_scaled)

# Convert and display predictions vs actual data of test data to DF
fraud_test_result = pd.DataFrame({ "Test Predictions":test_predictions, "Actual": y_test})
fraud_test_result

# Logistic Regression Classfication Report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, test_predictions))

# Calculate the model accuracy score using y_test and testing 
accuracy_score= accuracy_score(y_test, test_predictions)
print(f"Your accuracy score for the model is: {accuracy_score}")

# Train Random Forest Classifier
forest = RandomForestClassifier(n_estimators=100, random_state=15)
forest.fit(X_train_scaled, y_train)

# Evaluate the model
print(f'Training Score: {forest.score(X_train_scaled, y_train)}')
print(f'Testing Score: {forest.score(X_test_scaled, y_test)}')

# Randomforest prediction on X_test
y_pred_forest = forest.predict(X_test_scaled)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_forest))

# Accuracy of RandomForest Classfier
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred_forest)
print(f"Your accuracy score for the model is: {acc_score:.2f}")

# Get the feature importance array
feature_importances = forest.feature_importances_

# List the top 10 most important features
importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
importances_sorted[:50]

# Extract features and importances
top_features = importances_sorted[:50]
features, importances = zip(*top_features)  # Unpack tuples into two lists

# Create plot
fig, ax = plt.subplots(figsize=(20, 15))  # Set figure size
ax.barh(y=features, width=importances, color='skyblue')  # Create horizontal bar chart
plt.xticks(rotation=45)

# Adding labels and title
ax.set_xlabel('Importance')
ax.set_title('Top Feature Importances')

plt.show()

# Confusion Matrix on Logistic Regression
conf_matrix_log = confusion_matrix(y_test, test_predictions)
sns.heatmap(conf_matrix_log, annot=True, fmt='d', cmap='summer')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(conf_matrix_log)

# Confusion Matrix on Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_forest)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='summer')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(conf_matrix)

# ROC Curve for Logistic Regression
fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# ROC Curve for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, forest.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()


# Model Optimization and Reporting

# Hyperparameter tuning
# Attempt to optimize the Random Forest model for better performance
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=15), 
                           param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score:", grid_search.best_score_)

# Sample DataFrames for multiple lines
results_list = [
    pd.DataFrame({
        'param_n_estimators': [10, 20, 30, 40, 50],
        'mean_test_score': np.random.rand(5)
    }) for _ in range(2
    )
]

colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))

for i, results in enumerate(results_list):
    plt.plot(results['param_n_estimators'], results['mean_test_score'], color=colors[i], label=f'Model {i+1}', marker='o')

plt.xlabel('Number of Estimators')
plt.ylabel('Mean ROC AUC Score')
plt.title('Hyperparameter Tuning Results')
plt.legend()
plt.show()

# Line plot of ROC AUC scores
results = pd.DataFrame(grid_search.cv_results_)
plt.plot(results['param_n_estimators'], results['mean_test_score'])
plt.xlabel('Number of Estimators')
plt.ylabel('Mean ROC AUC Score')
plt.title('Hyperparameter Tuning Results')
plt.show()

print(results)

# Check to see if len(y_pred_forest) and len(predictions) matches
print(len(y_test)) 
print(len(y_pred_forest))
print(len(test_predictions))

# random forest auc roc
roc_auc_RF = roc_auc_score(y_test, y_pred_forest)
print(roc_auc_RF)

# LR ROC AUC
roc_auc_LR = roc_auc_score(y_test, test_predictions)
print(roc_auc_LR)

# Summary chart
# To compare model performance visually
summary = {
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [
        classification_report(y_test, y_pred_forest, output_dict=True)['accuracy'],
        classification_report(y_test, test_predictions, output_dict=True)['accuracy']
    ],
    'ROC AUC': [
        roc_auc_score(y_test, y_pred_forest),
        roc_auc_score(y_test, test_predictions)
    ]
}
summary_df = pd.DataFrame(summary)
summary_df.plot(kind='bar', x='Model', figsize=(8, 6))
plt.title('Model Performance Summary')
plt.xticks(rotation=0)
plt.show()

# Finding - Random Forest is the better model to use due to better ROC AUC score, it may be better in detecting fraud