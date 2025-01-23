import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the datasets
bank_full_path = 'd:\\bank-full.csv'  # Replace with the correct path
bank_sample_path = 'd:\\bank.csv'    # Replace with the correct path

# Load the datasets into dataframes
bank_full_df = pd.read_csv(bank_full_path, sep=';')
bank_sample_df = pd.read_csv(bank_sample_path, sep=';')

# Function to preprocess data
def preprocess_data(data):
    # Encode categorical variables
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data[col] = label_encoders[col].fit_transform(data[col])
    
    # Normalize numerical columns
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data

# Preprocess both datasets
bank_full_df = preprocess_data(bank_full_df)
bank_sample_df = preprocess_data(bank_sample_df)

# Split the larger dataset into training and testing sets
X_full = bank_full_df.drop(columns=['y'])
y_full = bank_full_df['y']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier on the full dataset
dt_classifier_full = DecisionTreeClassifier(random_state=42)
dt_classifier_full.fit(X_train_full, y_train_full)

# Evaluate the model on the full dataset
y_pred_full = dt_classifier_full.predict(X_test_full)
print("Performance on Full Dataset:")
print(f"Accuracy: {accuracy_score(y_test_full, y_pred_full):.2f}\n")
print("Classification Report:")
print(classification_report(y_test_full, y_pred_full))

# Use the smaller dataset for cross-verification
X_sample = bank_sample_df.drop(columns=['y'])
y_sample = bank_sample_df['y']
y_pred_sample = dt_classifier_full.predict(X_sample)

print("\nPerformance on Sample Dataset:")
print(f"Accuracy: {accuracy_score(y_sample, y_pred_sample):.2f}\n")
print("Classification Report:")
print(classification_report(y_sample, y_pred_sample))

# Visualize the confusion matrix for the full dataset
ConfusionMatrixDisplay.from_estimator(dt_classifier_full, X_test_full, y_test_full, cmap='Blues')
plt.title("Confusion Matrix - Full Dataset")
plt.show()

# Visualize the confusion matrix for the sample dataset
ConfusionMatrixDisplay.from_estimator(dt_classifier_full, X_sample, y_sample, cmap='Blues')
plt.title("Confusion Matrix - Sample Dataset")
plt.show()

# Visualize the decision tree (limited to depth=3 for simplicity)
plt.figure(figsize=(26, 10))
plot_tree(
    dt_classifier_full,
    feature_names=X_full.columns,
    class_names=["No Purchase", "Purchase"],  # Class names for clarity
    filled=True,
    max_depth=3,
    impurity=False  # Disable impurity display for clarity
)
plt.title("Decision Tree Visualization (Depth=3)")
plt.show()

# Output predictions for each leaf node
tree = dt_classifier_full.tree_
print("\nLeaf Node Predictions:")
for i in range(tree.node_count):
    if tree.children_left[i] == tree.children_right[i]:  # Check if it's a leaf node
        predicted_class = "Purchase" if tree.value[i][0][1] > tree.value[i][0][0] else "No Purchase"
        print(f"Leaf Node {i}: Prediction = {predicted_class}")

# Visualize the feature importance
feature_importances = dt_classifier_full.feature_importances_
feature_names = X_full.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()