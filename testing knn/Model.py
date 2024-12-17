import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('new_data_with_price_classes.csv')

# Filter the dataset
df = df[df['rent'] == 'no']
df = df[df['region'] == '6th of october']
df = df.drop(columns=['ad_id', 'rent', 'region', 'city'])

# Encode categorical columns
categorical_columns = ['type', 'furnished', 'level']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Separate features (X) and target (y)
X = df.drop(columns=['price_class'])  # Assuming 'price_class' is the target column
y = df['price_class']

# Encode the target variable if it is still categorical
y = le.fit_transform(y.astype(str))  # Convert price_class to numeric labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune 'n_neighbors'
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
