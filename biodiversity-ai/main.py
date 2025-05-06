import pandas as pd
species_df = pd.read_csv('species.csv')
parks_df = pd.read_csv('parks.csv')
# Check for common key
print("Species Columns:", species_df.columns)
print("Parks Columns:", parks_df.columns)

# Merge (adjust column names if needed)
merged_df = pd.merge(species_df, parks_df, on='Park Name', how='inner')
print("Merged dataset shape:", merged_df.shape)
print(merged_df.head())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
species_df.dropna(subset=['Category', 'Conservation Status', 'Park Name'], inplace=True)
species_df['Conservation Status'].fillna('Not Listed', inplace=True)
species_df = species_df.loc[:, ~species_df.columns.str.contains('^Unnamed')]
label_encoders = {}
for col in ['Category', 'Conservation Status']:
    le = LabelEncoder()
    merged_df[col + '_Encoded'] = le.fit_transform(merged_df[col])
    label_encoders[col] = le
merged_df['Vulnerability'] = merged_df['Conservation Status_Encoded'].apply(lambda x: 1 if x > 1 else 0)
features = ['Category_Encoded', 'Conservation Status_Encoded']
X = merged_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x='Category_Encoded', y='Conservation Status_Encoded', hue='Cluster', data=merged_df, palette='Set2')
plt.title('2D Clustering of Species Conservation Risk')
plt.xlabel('Category')
plt.ylabel('Conservation Status')
plt.show()
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(merged_df['Category_Encoded'], merged_df['Conservation Status_Encoded'], merged_df['Vulnerability'],
           c=merged_df['Cluster'], cmap='Set1')
ax.set_xlabel('Category')
ax.set_ylabel('Conservation Status')
ax.set_zlabel('Vulnerability')
plt.title('3D View of Species Clusters')
plt.show()
X_model = merged_df[['Category_Encoded', 'Conservation Status_Encoded']]
y = merged_df['Vulnerability']
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the first tree in the Random Forest
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], filled=True, feature_names=X.columns, class_names=['Not Vulnerable', 'Vulnerable'], rounded=True)
plt.title('First Decision Tree in Random Forest')
plt.show()
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

import matplotlib.pyplot as plt

# Random Forest Feature Importance Plot
rf_feature_importance = rf.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(['Category_Encoded', 'Conservation Status_Encoded'], rf_feature_importance, color='skyblue')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
print(f"Decision Tree Accuracy: {dt_acc:.2f}")
print(f"Random Forest Accuracy: {rf_acc:.2f}")
park_vulnerability = merged_df.groupby('Park Name')['Vulnerability'].mean().sort_values(ascending=False)
print("\nTop Parks by Average Species Vulnerability:\n")
print(park_vulnerability.head(5))
print("\nSuggested Protection Strategies:")
for park in park_vulnerability.head(3).index:
    print(f"- {park}: Increase monitoring and protection of high-risk species.")


