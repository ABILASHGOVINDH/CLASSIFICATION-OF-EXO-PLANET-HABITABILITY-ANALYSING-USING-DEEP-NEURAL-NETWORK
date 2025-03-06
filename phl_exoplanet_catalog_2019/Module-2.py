# Module 2: Conservatively Habitable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')


file_path = os.path.join('data', 'conservatively_habitable_2.csv')
data = pd.read_csv(file_path)


X = data.drop(columns=['P_HABITABLE'])
y = data['P_HABITABLE']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='logistic', max_iter=30, random_state=42)
mlp.fit(X_train_preprocessed, y_train)

# Evaluate model
y_train_pred = mlp.predict(X_train_preprocessed)
y_val_pred = mlp.predict(X_val_preprocessed)

training_accuracy = accuracy_score(y_train, y_train_pred)
validation_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training accuracy: {training_accuracy * 100:.2f}%")
print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")

# Predict on new data
input_file_path = os.path.join('data', 'full_input_data_con_hab.csv')
input_data = pd.read_csv(input_file_path)
input_data_preprocessed = preprocessor.transform(input_data)

predictions = mlp.predict(input_data_preprocessed)
predicted_df = pd.DataFrame({
    'P_NAME': input_data['P_NAME'],
    'P_HABITABLE': predictions,
    'Prediction': ['CONSERVATIVELY HABITABLE' if x == 1 else 'NON HABITABLE' for x in predictions]
})

print(predicted_df)

# Confusion matrix
confusion = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix:")
print(confusion)