import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib


df = pd.read_csv('data.csv')


df = df[df['rent'] == 'no']
df = df[df['region'] == '6th of october']
df = df.drop(columns=['ad_id', 'rent', 'region', 'city'])


categorical_columns = ['type', 'furnished', 'level']


le = LabelEncoder()


for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))


df = df.apply(pd.to_numeric, errors='coerce').dropna()


scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


X = df.drop(columns=['price'])
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=42)


svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)


y_pred = svr_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


joblib.dump(svr_model, 'svr_model.pkl')


