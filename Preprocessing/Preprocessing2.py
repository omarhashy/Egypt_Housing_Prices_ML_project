import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


df = pd.read_csv('data.csv')

df = df[df['rent'] == 'no']
df = df[df['region'] == '6th of october']
df = df.drop(columns=['ad_id', 'rent' , 'region' , 'city'])
# Use LabelEncoder to encode categorical columns
categorical_columns = ['type', 'furnished', 'level',]

# Initialize the label encoder
le = LabelEncoder()

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Display the dataset after encoding
df.head()
