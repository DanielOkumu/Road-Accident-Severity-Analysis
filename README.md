# Road-Accident-Severity-Analysis
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Select relevant features and target variable
X = data[['Weather_Condition', 'Road_Conditions', 'Visibility', 'Time_of_Day']]
y = data['Severity']

# Handle missing values
X.fillna(X.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'accident_severity_model.pkl')
# Load the model
loaded_model = joblib.load('accident_severity_model.pkl')

# Example of hypothetical independent variables
new_data = pd.DataFrame({
    'Weather_Condition': ['Clear'],
    'Road_Conditions': ['Dry'],
    'Visibility': [10],
    'Time_of_Day': ['Morning']
})

# Predict accident severity
prediction = loaded_model.predict(new_data)
print(prediction)
