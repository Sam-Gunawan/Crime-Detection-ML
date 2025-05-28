import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

# Download latest version
path = kagglehub.dataset_download("ishajangir/crime-data")
print("Path to dataset files:", path)

file = path + '\\Crime_Data_from_2020_to_Present.csv'
df = pd.read_csv(file)

df_copy= df.copy() #ensure the original isnot modified
# Dropping unused column
df_copy= df_copy.drop(columns=['DR_NO'])
df_copy= df_copy.drop(columns=['Part 1-2'])
df_copy= df_copy.drop(columns=['Crm Cd'])
df_copy= df_copy.drop(columns=['Crm Cd Desc'])
df_copy= df_copy.drop(columns=['Mocodes'])
df_copy= df_copy.drop(columns=['Vict Age'])
df_copy= df_copy.drop(columns=['Vict Sex'])
df_copy= df_copy.drop(columns=['Vict Descent'])
df_copy= df_copy.drop(columns=['Weapon Used Cd'])
df_copy= df_copy.drop(columns=['Weapon Desc'])
df_copy= df_copy.drop(columns=['Status'])
df_copy= df_copy.drop(columns=['Status Desc'])
df_copy= df_copy.drop(columns=['Crm Cd 1'])
df_copy= df_copy.drop(columns=['Crm Cd 2'])
df_copy= df_copy.drop(columns=['Crm Cd 3'])
df_copy= df_copy.drop(columns=['Crm Cd 4'])
df_copy= df_copy.drop(columns=['LOCATION'])
df_copy= df_copy.drop(columns=['Cross Street'])
df_copy= df_copy.drop(columns=['Rpt Dist No'])
df_copy= df_copy.drop(columns=['Date Rptd'])
df_copy= df_copy.drop(columns=['Premis Cd'])
df_copy= df_copy.drop(columns=['Premis Desc'])

## Convert the Time and Date into proper format
# Convert DATE OCC to datetime properly
df_copy['DATE OCC'] = pd.to_datetime(df_copy['DATE OCC'], format='%m/%d/%Y %H:%M')

#  Make sure TIME OCC is a 4-digit string
df_copy['TIME OCC'] = df_copy['TIME OCC'].apply(lambda x: str(int(x)).zfill(4))

# Now create 'datetime' column manually by combining
df_copy['datetime'] = pd.to_datetime(
    df_copy['DATE OCC'].dt.strftime('%Y-%m-%d') + ' ' +
    df_copy['TIME OCC'].str[:2] + ':' +
    df_copy['TIME OCC'].str[2:]
)

# Drop old columns
df_copy = df_copy.drop(columns=['DATE OCC', 'TIME OCC'])

## Calculate the LAT and LON mean of each area
df_copy['LAT'] = df_copy['AREA NAME'].map(df_copy.groupby('AREA NAME')['LAT'].mean())
df_copy['LON'] = df_copy['AREA NAME'].map(df_copy.groupby('AREA NAME')['LON'].mean())

# Sort the Data based on AREA (number)
df_copy= df_copy.sort_values(by=['AREA'])

## Split Date and Area
# Create the area mapping dictionary
split_area_mapping = df_copy.groupby('AREA').agg({'AREA NAME': 'first', 'LAT': 'mean', 'LON': 'mean'}).to_dict('index')

# Create the area DataFrame
df_area = pd.DataFrame.from_dict(split_area_mapping, orient='index').reset_index().rename(columns={'index': 'AREA'})

# Save the area DataFrame to a CSV file
df_area.to_csv('./dataset/area_reference.csv', index=False)

# Create the time DataFrame
df_time = df_copy[['AREA', 'datetime']]

print(df_area)
print(df_time)

# === Add Weekly Features ===
df_copy['iso_year'] = df_copy['datetime'].dt.isocalendar().year
df_copy['iso_week'] = df_copy['datetime'].dt.isocalendar().week

# Group and Count Crimes by AREA + Week
grouped_week = df_copy.groupby(
    ['AREA', 'LAT', 'LON', 'iso_year', 'iso_week']
).size().reset_index(name='crime_count')

# Save the grouped DataFrame to a CSV file
grouped_week.to_csv('./dataset/grouped_week.csv', index=False)

# Generate crime density data
crime_density = grouped_week.groupby(['AREA', 'LAT', 'LON'])['crime_count'].sum().reset_index()
crime_density.to_csv('./dataset/crime_density_by_area.csv', index=False)

# Features and Target
X = grouped_week[['AREA', 'LAT', 'LON', 'iso_year', 'iso_week']]
y = grouped_week['crime_count']

# Filter to only use 2020–2022 data for training (less accurate)
# df_filtered = grouped_week[grouped_week['iso_year'].isin([2020, 2021, 2022, 2023])]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

# Train and Evaluate Models
models = {
    'XGBoost': XGBRegressor(),
    'Random Forest': RandomForestRegressor(),
    'KNN': KNeighborsRegressor()
}

# Accuracy-like score: % of predictions within ±N crimes
def accuracy_within_range(y_true, y_pred, tolerance=3):
    # Default tolerance is ±3
    return (abs(y_true - y_pred) <= tolerance).mean()

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_within_range(y_test, preds, tolerance=20)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name} accuracy (within ±20 crimes): {acc:.2%}")
    print(f"{name} Evaluation (Weekly + Area-Based)")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# Save the models and test scores
joblib.dump(results, './models/crime_model_results.pkl')
joblib.dump(results['XGBoost']['model'], './models/xgboost_model.pkl')
joblib.dump(results['Random Forest']['model'], './models/random_forest_model.pkl')
joblib.dump(results['KNN']['model'], './models/knn_model.pkl')

### THIS IS FOR MODELS TRAINED ON DAILY CRIMES (ONLY 55% ACCURATE). BEST NOT TO USE THIS WITHOUT ADVANCED PREPROCESSING DUE TO THE SPORADIC NATURE OF CRIME
# # Make sure datetime is a datetime object (you've already done this)
# df_time['date'] = df_time['datetime'].dt.date  # Extract date only (no time)

# # Count number of crimes per AREA per DAY
# df_daily = df_time.groupby(['AREA', 'date']).size().reset_index(name='crime_count')

# # === Add Time Features ===
# df_daily['dayofweek'] = pd.to_datetime(df_daily['date']).dt.dayofweek  # 0 = Monday
# df_daily['month'] = pd.to_datetime(df_daily['date']).dt.month
# df_daily['year'] = pd.to_datetime(df_daily['date']).dt.year

# # === Add area features (LAT/LON) ===
# df_daily = df_daily.merge(df_area, on='AREA', how='left')  # df_area contains AREA NAME, LAT, LON

# # === Features and Target ===
# X = df_daily[['AREA', 'LAT', 'LON', 'dayofweek', 'month', 'year']]
# y = df_daily['crime_count']

# # === Train-Test Split ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Initialize Models ===
# models = {
#     'XGBoost': XGBRegressor(),
#     'Random Forest': RandomForestRegressor(),
#     'KNN': KNeighborsRegressor()
# }

# # Accuracy-like score: % of predictions within ±N crimes
# def accuracy_within_range(y_true, y_pred, tolerance=3):
#     # Default tolerance is ±3
#     return (abs(y_true - y_pred) <= tolerance).mean()

# # === Train and Evaluate ===
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     acc = accuracy_within_range(y_test, preds, tolerance=5)
#     print(f"\n{name} accuracy (within ±5 crimes): {acc:.2%}")
#     print(f"{name} Evaluation (Area + Time Based)")
#     print("MAE:", mean_absolute_error(y_test, preds))
#     print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
#     print("R²:", r2_score(y_test, preds))
###