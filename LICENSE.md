import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_data.csv' with your dataset)
data = pd.read_csv('your_data.csv')

# Select relevant features and target variable (safety outcome)
features = ['SocioEconomicStatus', 'Age', 'Gender']
X = data[features]
y = data['SafetyOutcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a logistic regression model (you can choose a different model)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
