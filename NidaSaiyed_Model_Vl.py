pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time
import sys

# Concatenate features and target into a DataFrame
data = pd.concat([X, y], axis=1)

# Split the dataset into features (X) and target variable (y)
X = data.drop('Diagnosis', axis=1)  # Features
y = data['Diagnosis']                # Target variable

# Split the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16, color='blue')
plt.show()

# Print accuracy and classification report
print("\n\033[1m\033[91mML model for Breast Cancer Wisconsin (Diagnostic)\033[0m")
print("\n\033[1mAccuracy of the Random Forest Classifier:\033[0m", round(accuracy, 4))
print("\n\033[1mClassification Report:\033[0m")
df_classification_report = pd.DataFrame(classification_rep).transpose()
df_classification_report.style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black'), ('font-weight', 'bold')]}])
print(df_classification_report)

# Save the trained model to a file
joblib.dump(rf_classifier, 'breast_cancer_classifier.pkl')

# Blinking "Model saved successfully!" message
blink_count = 10
for _ in range(blink_count):
    sys.stdout.write("\r\033[1m\033[92mModel saved successfully!\033[0m" if _ % 2 == 0 else "\r\033[1m\033[92m                                      \033[0m")
    sys.stdout.flush()
    time.sleep(0.5)
