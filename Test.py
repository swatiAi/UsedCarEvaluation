import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")

# Importing data
df = pd.read_csv('C:\\Users\\usman\\Documents\\UsedCarEvaluation\\carlistings.csv')

# Dropping the 'ID' column
df.drop(['ListingID'], axis=1, inplace=True)

# Renaming columns
df.columns = ['Car', 'Location', 'Year', 'ODO', 'Fuel', 'Transmission', 'Owner', 'Mileage', 'Engine', 'Power', 'Seats', 'CP', 'SP']

# Dropping rows where 'Owner' is 'Second'
df = df[df['Owner'] != 'Second']
df.drop('Owner', axis=1, inplace=True)

# Handling missing values in 'CP'
df['CP'].fillna(0, inplace=True)

# Converting 'CP' values
for index, row in df.iterrows():
    if type(row['CP']) == str and 'Lakh' in row['CP']:
        converted_value = float(row['CP'].replace(' Lakh', ''))
        df.at[index, 'CP'] = converted_value
    if type(row['CP']) == str and 'Cr' in row['CP']:
        converted_value = float(row['CP'].replace(' Cr', '')) * 100
        df.at[index, 'CP'] = converted_value

df['CP'] = df['CP'].astype(float)

# Dropping rows with missing values
df = df.dropna()

# Stripping units and converting to floats
df['Engine'] = df['Engine'].str.replace(' CC', '').astype(float)
df['Power'] = df['Power'].str.replace(' bhp', '').astype(float)

# Converting mileage based on fuel type
condition = (df['Mileage'].str.contains('km/kg'))
for index, row in df[condition].iterrows():
    if row['Fuel'] == 'Diesel':
        converted_value = float(row['Mileage'].replace(' km/kg', '')) / 0.832
        df.at[index, 'Mileage'] = converted_value
    if row['Fuel'] == 'Petrol':
        converted_value = float(row['Mileage'].replace(' km/kg', '')) / 0.740
        df.at[index, 'Mileage'] = converted_value

# Removing mileage units
for index, row in df.iterrows():
    if type(row['Mileage']) == str and ' kmpl' in row['Mileage']:
        df.at[index, 'Mileage'] = row['Mileage'].replace(' kmpl', '')

df['Mileage'] = df['Mileage'].astype(float)

# Calculating Age from Year and dropping Year
df['Age'] = 2024 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Reordering columns
df = df[['Car', 'Location', 'Fuel', 'Transmission', 'ODO', 'Age', 'Mileage', 'Engine', 'Power', 'Seats', 'CP', 'SP']]

# Converting 'Seats' to float
df['Seats'] = df['Seats'].astype('float64')

# Splitting data into training and testing sets
df_1 = df[df['CP'] == 0]
df_1.drop('CP', axis=1, inplace=True)
df_1 = df_1.reset_index(drop=True)
df_2 = df[df['CP'] != 0]
df_2 = df_2.reset_index(drop=True)

# Saving test and training data
df_1.to_csv('test.csv', index=False)
df_2.to_csv('train.csv', index=False)

df = df_2

# Correlation and feature selection
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, 4:].corr(), cmap='flare', annot=True)
plt.title("Correlation Between Numerical Variables", size=15)
plt.show()

# Define numerical and categorical columns
num_cols = ['ODO', 'Age', 'Mileage', 'Engine', 'Power', 'Seats', 'CP']
cat_cols = ["Location", "Fuel", "Transmission"]

# Selecting relevant columns based on correlation
relevant_cols = cat_cols + num_cols
df = df[relevant_cols + ['SP']]

# Preparing feature matrix and target variable
X = df.drop('SP', axis=1)
y = df['SP']

# Encoding categorical variables
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Standardizing numerical features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=2)

# Training XGBoost model
xgb_params = {
    'subsample': 1.0,
    'reg_lambda': 0,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'max_depth': 4,
    'learning_rate': 0.2,
    'gamma': 1,
    'colsample_bytree': 0.9
}
xgb = XGBRegressor(**xgb_params)
xgb.fit(X_train, y_train)

# Saving the trained model
joblib.dump(xgb, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Processing final test data for prediction
final_test = pd.read_csv("test.csv")
static_test = final_test[["Car", "ODO", "Age", "Mileage", "Engine", "Seats"]]
predict_y = final_test[["Location", "Fuel", "Transmission", "Power", "SP"]]

# Encoding and normalizing test data
predict_y = pd.get_dummies(predict_y, columns=cat_cols, drop_first=True)
predict_y[num_cols] = scaler.transform(predict_y[num_cols])

# Predicting cost prices
y_pred = xgb.predict(predict_y)
df_pred = pd.DataFrame({'CP': y_pred})
df_pred = ((100 * (df_pred['CP'].round(2))).astype(int)) / 100
result = pd.concat([final_test, df_pred], axis=1)
result.to_csv('result.csv', index=False)
