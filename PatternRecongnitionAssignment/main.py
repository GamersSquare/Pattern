import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Preprocess the Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_data = pd.read_csv(url, header=None)
adult_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                      'income']

adult_data.replace('?', pd.NA, inplace=True)  # Replace '?' with NA
adult_data.dropna(inplace=True)  # Drop rows with missing values
adult_data['income'] = adult_data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)  # Convert income to binary

# Encode categorical variables
adult_data = pd.get_dummies(adult_data, columns=['workclass', 'education', 'marital-status', 'occupation',
                                                 'relationship', 'race', 'sex', 'native-country'])

# Separate features and target variable
X = adult_data.drop('income', axis=1)  # features
y = adult_data['income']  # target

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state to insure the outputs are the same at every function call

# Train the Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the Model
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy, "\n")

conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='g', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Compute Sensitivity (True Positive Rate)
sensitivity = tp / (tp + fn)

# Compute Specificity (True Negative Rate)
specificity = tn / (tn + fp)

print("Sensitivity (True Positive Rate):", sensitivity)
print("Specificity (True Negative Rate):", specificity, "\n")

# Compute the posterior probability of making over 50K a year
posterior_probs = nb_classifier.predict_proba(X_test)
positive_class_prob = posterior_probs[:, 1]  # Probability of the positive class


# Assuming the positive class corresponds to making over 50K a year
mean_positive_prob = positive_class_prob.mean()

print("Mean posterior probability of making over 50K a year:", mean_positive_prob, "\n")

print(classification_report(y_test, y_pred))
