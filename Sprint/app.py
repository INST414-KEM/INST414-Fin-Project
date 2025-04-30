import pandas as pd

# Load dataset
spotify_df = pd.read_csv('top_spotify_daily.csv')

# Clean artist/title columns
spotify_df['artist'] = spotify_df['artist'].str.lower().str.strip()
spotify_df['title'] = spotify_df['title'].str.lower().str.strip()

# Drop duplicates
spotify_df.drop_duplicates(subset=['artist', 'title', 'date'], inplace=True)

# Check for missing values
print(spotify_df.isnull().sum())

# Optional: convert types
spotify_df['streams'] = pd.to_numeric(spotify_df['streams'], errors='coerce')
spotify_df.dropna(subset=['streams'], inplace=True)

# Save cleaned version (optional)
# spotify_df.to_csv('cleaned_spotify.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Features and target
features = ['danceability', 'energy', 'tempo', 'acousticness']
target = 'streams'

X = spotify_df[features]
y = spotify_df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression RMSE: {rmse}')
print(f'R^2 Score: {r2}')
