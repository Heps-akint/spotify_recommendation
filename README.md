# Genre Prediction and Analysis of Music Genres and Playlists

## Description

This project focuses on the analysis of music genres and playlists using provided datasets. It demonstrates a comprehensive approach to data science, including data cleaning, exploratory data analysis (EDA), feature engineering, and the development of a genre prediction system using machine learning techniques. The project aims to showcase a range of data science skills essential for a data scientist role.

## Table of Contents

- [Installation](#installation)
- [Data Preparation and Cleaning](#data-preparation-and-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Genre Popularity Analysis](#genre-popularity-analysis)
- [Playlist Pattern Analysis](#playlist-pattern-analysis)
- [Genre Prediction System](#genre-prediction-system)
- [Results](#results)
- [Conclusion and Recommendations](#conclusion-and-recommendations)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Heps-akint/spotify_recommendation.git
   cd spotify_recommendation
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the provided datasets (`genres_v2.csv` and `playlists.csv`) are in the project directory.

## Data Preparation and Cleaning

The data preparation process involves loading the datasets, handling missing values, and normalizing numerical features.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the datasets
genres_df = pd.read_csv('genres_v2.csv')
playlists_df = pd.read_csv('playlists.csv')

# Drop columns with a large number of missing values and rows with any missing values
genres_df.drop(columns=['song_name', 'Unnamed: 0', 'title'], inplace=True)
genres_df.dropna(inplace=True)
playlists_df.dropna(inplace=True)

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = genres_df.select_dtypes(include=['float64', 'int64']).columns
genres_df[numeric_cols] = scaler.fit_transform(genres_df[numeric_cols])
```

## Exploratory Data Analysis (EDA)

Perform EDA to understand the structure and relationships within the data, including the distribution of genres and key numerical features.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of genres
plt.figure(figsize=(12, 6))
sns.countplot(data=genres_df, x='genre', order=genres_df['genre'].value_counts().index)
plt.title('Distribution of Genres')
plt.xticks(rotation=90)
plt.show()

# Distribution of danceability
plt.figure(figsize=(12, 6))
sns.histplot(genres_df['danceability'], bins=50, kde=True)
plt.title('Distribution of Danceability')
plt.xlabel('Danceability')
plt.show()

# Correlation heatmap for numerical features
numeric_cols = genres_df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(genres_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Numerical Features in Genres Dataset')
plt.show()
```

## Genre Popularity Analysis

Analyse the popularity of different genres by counting their occurrences in the dataset.

```python
# Group by genre and count occurrences
genre_popularity = genres_df['genre'].value_counts()

# Plot genre popularity
plt.figure(figsize=(12, 6))
genre_popularity.plot(kind='bar')
plt.title('Genre Popularity')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```

## Playlist Pattern Analysis

Investigate genre transitions within playlists.

```python
# Display genre transitions in playlists
playlist_transitions = playlists_df.groupby('Playlist')['Genre'].apply(lambda x: ' -> '.join(x))

# Display a sample of playlist transitions
print("\nSample Playlist Transitions:")
print(playlist_transitions.head())
```

## Genre Prediction System

Develop a genre prediction system using a Random Forest classifier with hyperparameter tuning.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Encode the genres
le = LabelEncoder()
genres_df['genre_encoded'] = le.fit_transform(genres_df['genre'])

# Split the data into training and testing sets
X = genres_df.drop(columns=['genre', 'genre_encoded', 'type', 'id', 'uri', 'track_href', 'analysis_url'])
y = genres_df['genre_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the Random Forest Classifier with the best parameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Make predictions
y_pred = best_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nGenre Prediction System Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

## Results

The genre prediction system achieved an accuracy of 68.16%. The classification report provides detailed performance metrics for each genre.

```
Genre Prediction System Accuracy: 68.16%

Classification Report:
                 precision    recall  f1-score   support

      Dark Trap       0.56      0.46      0.50       970
            Emo       0.74      0.73      0.73       341
         Hiphop       0.47      0.43      0.45       621
            Pop       0.40      0.06      0.11        98
            Rap       0.62      0.30      0.41       341
            RnB       0.43      0.37      0.40       396
     Trap Metal       0.45      0.27      0.33       384
Underground Rap       0.40      0.60      0.48      1192
            dnb       0.96      0.99      0.97       599
      hardstyle       0.88      0.93      0.90       619
      psytrance       0.94      0.92      0.93       598
      techhouse       0.86      0.91      0.89       568
         techno       0.86      0.85      0.86       590
         trance       0.81      0.89      0.85       562
           trap       0.86      0.85      0.85       582
```

## Conclusion and Recommendations

### Summary

This project successfully demonstrates a comprehensive approach to data analysis and machine learning. Key steps included:

- Data cleaning and preprocessing.
- Exploratory data analysis to understand the dataset's characteristics.
- Genre popularity analysis.
- Playlist pattern analysis.
- Development and tuning of a genre prediction system.

### Recommendations

To further improve the genre prediction system and analysis, consider the following:

1. **Additional Features**: Incorporate more features that may impact genre classification, such as lyrics or artist information.
2. **Advanced Models**: Experiment with more advanced machine learning models like XGBoost, Gradient Boosting, or deep learning approaches.
3. **Cross-Validation**: Implement cross-validation to ensure the model's robustness and avoid overfitting.
4. **Larger Playlist Dataset**: Collect a larger playlist dataset to better understand transitions and user preferences.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
