import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import statsmodels.api as sm

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('Indian_Kids_Screen_Time.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print("\nFeature names:")
print(df.columns.tolist())

# Display data types and missing values
print("\nData types and missing values:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data cleaning
def clean_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Handle missing values if any
    df_clean = df_clean.dropna()
    
    # Convert categorical variables to appropriate types
    df_clean['Gender'] = df_clean['Gender'].astype('category')
    df_clean['Primary_Device'] = df_clean['Primary_Device'].astype('category')
    df_clean['Urban_or_Rural'] = df_clean['Urban_or_Rural'].astype('category')
    df_clean['Exceeded_Recommended_Limit'] = df_clean['Exceeded_Recommended_Limit'].astype('bool')
    
    # Extract health impacts into separate columns
    health_impacts = df_clean['Health_Impacts'].str.get_dummies(sep=', ')
    
    # Merge health impact columns with main dataframe
    df_clean = pd.concat([df_clean, health_impacts], axis=1)
    
    return df_clean

# Clean the data
df_clean = clean_data(df)

# Display the cleaned data structure
print("\nCleaned data structure:")
print(df_clean.info())

# 1. Screen Time Pattern Analysis
print("\n1. SCREEN TIME PATTERN ANALYSIS")

# Average daily screen time by age and gender
plt.figure()
sns.boxplot(x='Age', y='Avg_Daily_Screen_Time_hr', hue='Gender', data=df_clean)
plt.title('Average Daily Screen Time by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Daily Screen Time (hours)')
plt.savefig('screen_time_by_age_gender.png')

# Device preference by age
age_bins = [5, 10, 13, 16, 19]
age_labels = ['5-10', '11-13', '14-16', '17-19']
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels, right=False)

device_counts = df_clean.groupby(['Age_Group', 'Primary_Device']).size().reset_index(name='Count')
plt.figure()
sns.barplot(x='Age_Group', y='Count', hue='Primary_Device', data=device_counts)
plt.title('Device Preference by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig('device_preference.png')

# Screen time distribution
plt.figure()
sns.histplot(df_clean['Avg_Daily_Screen_Time_hr'], kde=True)
plt.axvline(x=2, color='red', linestyle='--', label='Recommended Limit (2 hours)')
plt.title('Distribution of Daily Screen Time')
plt.xlabel('Daily Screen Time (hours)')
plt.ylabel('Count')
plt.legend()
plt.savefig('screen_time_distribution.png')

# 2. Correlation Study: Screen Time vs. Health Outcomes
print("\n2. CORRELATION STUDY: SCREEN TIME VS. HEALTH OUTCOMES")

# Extract health impact columns
health_columns = [col for col in df_clean.columns if col in ['Poor Sleep', 'Eye Strain', 'Anxiety', 'Obesity', 'None']]

# Screen time vs health issues
plt.figure()
health_data = []
health_labels = []

for health_issue in health_columns:
    if health_issue in df_clean.columns:
        health_data.append(df_clean[df_clean[health_issue] == 1]['Avg_Daily_Screen_Time_hr'])
        health_labels.append(health_issue)

plt.boxplot(health_data, labels=health_labels)
plt.title('Screen Time vs Health Issues')
plt.xlabel('Health Issue')
plt.ylabel('Daily Screen Time (hours)')
plt.savefig('screen_time_health.png')

# Correlation analysis
print("\nCorrelation between screen time and health issues:")
for health_issue in health_columns:
    if health_issue in df_clean.columns and health_issue != 'None':
        correlation = df_clean['Avg_Daily_Screen_Time_hr'].corr(df_clean[health_issue])
        print(f"{health_issue}: {correlation:.4f}")

# 3. Urban vs. Rural Screentime Disparity
print("\n3. URBAN VS. RURAL SCREENTIME DISPARITY")

# Urban vs rural screen time comparison
plt.figure()
sns.boxplot(x='Urban_or_Rural', y='Avg_Daily_Screen_Time_hr', data=df_clean)
plt.title('Urban vs Rural Screen Time Comparison')
plt.xlabel('Location')
plt.ylabel('Daily Screen Time (hours)')
plt.savefig('urban_rural_comparison.png')

# Statistical test for urban vs rural difference
urban_screen_time = df_clean[df_clean['Urban_or_Rural'] == 'Urban']['Avg_Daily_Screen_Time_hr']
rural_screen_time = df_clean[df_clean['Urban_or_Rural'] == 'Rural']['Avg_Daily_Screen_Time_hr']

print("\nUrban vs Rural Screen Time Statistics:")
print(f"Urban mean: {urban_screen_time.mean():.2f} hours")
print(f"Rural mean: {rural_screen_time.mean():.2f} hours")
print(f"Difference: {urban_screen_time.mean() - rural_screen_time.mean():.2f} hours")

# T-test for significance
from scipy import stats
t_stat, p_value = stats.ttest_ind(urban_screen_time, rural_screen_time, equal_var=False)
print(f"T-test p-value: {p_value:.4f} (< 0.05 indicates significant difference)")

# Device usage by location
device_location = df_clean.groupby(['Urban_or_Rural', 'Primary_Device']).size().reset_index(name='Count')
plt.figure()
sns.barplot(x='Urban_or_Rural', y='Count', hue='Primary_Device', data=device_location)
plt.title('Device Usage: Urban vs Rural')
plt.xlabel('Location')
plt.ylabel('Count')
plt.savefig('device_urban_rural.png')

# 4. Educational vs. Recreational Screen Time Classification
print("\n4. EDUCATIONAL VS. RECREATIONAL SCREEN TIME CLASSIFICATION")

# Calculate educational and recreational time
df_clean['Educational_Time'] = df_clean['Avg_Daily_Screen_Time_hr'] * df_clean['Educational_to_Recreational_Ratio']
df_clean['Recreational_Time'] = df_clean['Avg_Daily_Screen_Time_hr'] - df_clean['Educational_Time']

# Educational vs recreational time statistics
print("\nEducational vs Recreational Screen Time:")
print(f"Average Educational Time: {df_clean['Educational_Time'].mean():.2f} hours")
print(f"Average Recreational Time: {df_clean['Recreational_Time'].mean():.2f} hours")
print(f"Average Ratio: {df_clean['Educational_to_Recreational_Ratio'].mean():.2f}")

# Plot educational vs recreational time
plt.figure()
edu_rec_data = [df_clean['Educational_Time'], df_clean['Recreational_Time']]
plt.boxplot(edu_rec_data, labels=['Educational', 'Recreational'])
plt.title('Educational vs Recreational Screen Time')
plt.ylabel('Hours per Day')
plt.savefig('edu_rec_time.png')

# Educational to recreational ratio by age
plt.figure()
sns.scatterplot(x='Age', y='Educational_to_Recreational_Ratio', hue='Gender', data=df_clean)
plt.title('Educational to Recreational Ratio by Age')
plt.xlabel('Age')
plt.ylabel('Educational to Recreational Ratio')
plt.savefig('edu_rec_ratio_by_age.png')

# 5. Time Series Analysis (simulated since we don't have actual time series data)
print("\n5. TIME SERIES ANALYSIS (SIMULATED)")

# Create a simulated time series for demonstration
# In a real project, you would use actual time series data
months = pd.date_range(start='2020-01-01', end='2025-01-01', freq='ME')
n_months = len(months)

# Simulate pre-COVID, COVID, and post-COVID periods with different trends
pre_covid = np.linspace(2, 3, 14)  # Jan 2020 - Feb 2021
covid_peak = np.linspace(3, 5, 12)  # Mar 2021 - Feb 2022
post_covid = np.linspace(5, 4, n_months - 14 - 12)  # Mar 2022 - Jan 2025

# Combine the periods and add some random variation
trend = np.concatenate([pre_covid, covid_peak, post_covid])
random_var = np.random.normal(0, 0.3, n_months)
screen_time_trend = trend + random_var

# Create a time series dataframe
ts_df = pd.DataFrame({
    'Date': months,
    'Avg_Screen_Time': screen_time_trend
})

# Plot the time series
plt.figure(figsize=(14, 8))
sns.lineplot(x='Date', y='Avg_Screen_Time', data=ts_df)
plt.axvspan('2021-03-01', '2022-02-28', alpha=0.2, color='red', label='COVID-19 Peak')
plt.title('Simulated Screen Time Trend (2020-2025)')
plt.xlabel('Date')
plt.ylabel('Average Daily Screen Time (hours)')
plt.legend()
plt.savefig('screen_time_trend.png')

# 6. Clustering & Segmentation
print("\n6. CLUSTERING & SEGMENTATION")

# Select features for clustering
features = ['Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
X = df_clean[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure()
plt.plot(k_range, inertia, 'o-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.savefig('elbow_curve.png')

# Apply K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Label the clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = ['Low Usage', 'Moderate Usage', 'Heavy Usage']

# Sort clusters by screen time
sorted_indices = np.argsort(cluster_centers[:, 0])
cluster_map = {i: cluster_labels[j] for i, j in enumerate(sorted_indices)}
df_clean['User_Segment'] = df_clean['Cluster'].map(cluster_map)

# Plot the clusters
plt.figure()
sns.scatterplot(x='Avg_Daily_Screen_Time_hr', y='Educational_to_Recreational_Ratio', 
                hue='User_Segment', data=df_clean)
plt.title('User Segments by Screen Time and Educational Ratio')
plt.xlabel('Daily Screen Time (hours)')
plt.ylabel('Educational to Recreational Ratio')
plt.savefig('user_segments.png')

# Cluster statistics
print("\nUser Segment Statistics:")
for segment in df_clean['User_Segment'].unique():
    segment_data = df_clean[df_clean['User_Segment'] == segment]
    print(f"\n{segment}:")
    print(f"Count: {len(segment_data)}")
    print(f"Average Screen Time: {segment_data['Avg_Daily_Screen_Time_hr'].mean():.2f} hours")
    print(f"Average Educational Ratio: {segment_data['Educational_to_Recreational_Ratio'].mean():.2f}")
    print(f"Average Age: {segment_data['Age'].mean():.2f} years")

# 7. Predictive Modeling: At-Risk Children
print("\n7. PREDICTIVE MODELING: AT-RISK CHILDREN")

# Define features and target
features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
X = df_clean[features]
y = df_clean['Exceeded_Recommended_Limit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure()
sns.barplot(x='Feature', y='Importance', data=feature_importance)
plt.title('Feature Importance for Predicting Excessive Screen Time')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.savefig('feature_importance.png')

# Add risk predictions to dataframe
# Handle the case where there's only one class in the dataset
try:
    # If there are multiple classes, use the second column (probability of positive class)
    risk_scores = model.predict_proba(df_clean[features])[:, 1]
except IndexError:
    # If there's only one class, use the first column
    risk_scores = model.predict_proba(df_clean[features])[:, 0]
    
df_clean['Risk_Score'] = risk_scores

# Create risk categories based on the score
# If all values are the same, create equal categories
if df_clean['Risk_Score'].nunique() <= 1:
    df_clean['Risk_Category'] = 'Medium Risk'  # All are the same risk
else:
    df_clean['Risk_Category'] = pd.qcut(df_clean['Risk_Score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

# Plot risk score by age
plt.figure()
sns.scatterplot(x='Age', y='Risk_Score', hue='Risk_Category', data=df_clean)
plt.title('Risk Score by Age')
plt.xlabel('Age')
plt.ylabel('Risk Score (0-1)')
plt.savefig('risk_by_age.png')

# 8. Save processed data for dashboard
df_clean.to_csv('processed_data.csv', index=False)
print("\nProcessed data saved to 'processed_data.csv'")

print("\nAnalysis complete. All visualizations saved as PNG files.")