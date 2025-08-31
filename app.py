import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
df = pd.read_csv('Indian_Kids_Screen_Time.csv')

# Data cleaning
def clean_data(df):
    # Handle missing values if any
    df = df.dropna()
    
    # Convert categorical variables to appropriate types
    df['Gender'] = df['Gender'].astype('category')
    df['Primary_Device'] = df['Primary_Device'].astype('category')
    df['Urban_or_Rural'] = df['Urban_or_Rural'].astype('category')
    df['Exceeded_Recommended_Limit'] = df['Exceeded_Recommended_Limit'].astype('bool')
    
    # Extract health impacts into separate columns
    health_impacts = df['Health_Impacts'].str.get_dummies(sep=', ')
    
    # Merge health impact columns with main dataframe
    df = pd.concat([df, health_impacts], axis=1)
    
    return df

# Clean the data
df = clean_data(df)

# Perform clustering for user segmentation
def perform_clustering(df):
    # Select features for clustering
    features = ['Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
    X = df[features]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (using 3 for simplicity)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Label the clusters
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_labels = ['Low Usage', 'Moderate Usage', 'Heavy Usage']
    
    # Sort clusters by screen time
    sorted_indices = np.argsort(cluster_centers[:, 0])
    cluster_map = {i: cluster_labels[j] for i, j in enumerate(sorted_indices)}
    df['User_Segment'] = df['Cluster'].map(cluster_map)
    
    return df

# Apply clustering
df = perform_clustering(df)

# Build predictive model for at-risk children
def build_risk_model(df):
    # Define features and target
    features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
    X = df[features]
    y = df['Exceeded_Recommended_Limit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Add risk predictions to dataframe
    df['Risk_Score'] = model.predict_proba(df[features])[:, 1]
    df['Risk_Category'] = pd.qcut(df['Risk_Score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    return df, model, accuracy, report

# Apply risk model
df, risk_model, model_accuracy, model_report = build_risk_model(df)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Indian Kids Screen Time Analysis 2025", className="text-center mb-4"),
            html.P("Interactive dashboard analyzing screen time patterns, health impacts, and risk factors among Indian children", 
                   className="text-center mb-5")
        ])
    ]),
    
    dbc.Tabs([
        # Tab 1: Screen Time Patterns
        dbc.Tab(label="Screen Time Patterns", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Average Daily Screen Time by Age and Gender", className="mt-3"),
                    dcc.Graph(id="screen-time-by-age-gender"),
                ], width=6),
                dbc.Col([
                    html.H4("Device Preference by Age Group", className="mt-3"),
                    dcc.Graph(id="device-preference"),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Screen Time Distribution", className="mt-3"),
                    dcc.Graph(id="screen-time-distribution"),
                ], width=12),
            ])
        ]),
        
        # Tab 2: Health Impacts
        dbc.Tab(label="Health Impacts", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Screen Time vs Health Issues", className="mt-3"),
                    dcc.Graph(id="screen-time-health"),
                ], width=6),
                dbc.Col([
                    html.H4("Health Issues by Age Group", className="mt-3"),
                    dcc.Graph(id="health-by-age"),
                ], width=6),
            ])
        ]),
        
        # Tab 3: Urban vs Rural Analysis
        dbc.Tab(label="Urban vs Rural", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Urban vs Rural Screen Time", className="mt-3"),
                    dcc.Graph(id="urban-rural-comparison"),
                ], width=6),
                dbc.Col([
                    html.H4("Device Usage: Urban vs Rural", className="mt-3"),
                    dcc.Graph(id="device-urban-rural"),
                ], width=6),
            ])
        ]),
        
        # Tab 4: Educational vs Recreational
        dbc.Tab(label="Educational vs Recreational", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Educational vs Recreational Screen Time", className="mt-3"),
                    dcc.Graph(id="edu-rec-ratio"),
                ], width=12),
            ])
        ]),
        
        # Tab 5: User Segmentation
        dbc.Tab(label="User Segmentation", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("User Segments by Screen Time and Educational Ratio", className="mt-3"),
                    dcc.Graph(id="user-segments"),
                ], width=12),
            ])
        ]),
        
        # Tab 6: Risk Prediction
        dbc.Tab(label="Risk Prediction", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Risk Factors for Excessive Screen Time", className="mt-3"),
                    dcc.Graph(id="risk-factors"),
                ], width=6),
                dbc.Col([
                    html.H4("Risk Distribution by Age", className="mt-3"),
                    dcc.Graph(id="risk-by-age"),
                ], width=6),
            ])
        ]),
    ]),
    
    html.Footer(
        html.P("© 2025 Indian Kids Screen Time Analysis Project", className="text-center mt-5")
    )
], fluid=True)

# Callbacks for interactive visualizations
@app.callback(
    Output("screen-time-by-age-gender", "figure"),
    Input("screen-time-by-age-gender", "id")
)
def update_screen_time_by_age_gender(_):
    fig = px.box(df, x="Age", y="Avg_Daily_Screen_Time_hr", color="Gender",
                title="Average Daily Screen Time by Age and Gender",
                labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"})
    return fig

@app.callback(
    Output("device-preference", "figure"),
    Input("device-preference", "id")
)
def update_device_preference(_):
    age_bins = [5, 10, 13, 16, 19]
    age_labels = ['5-10', '11-13', '14-16', '17-19']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    device_counts = df.groupby(['Age_Group', 'Primary_Device']).size().reset_index(name='Count')
    fig = px.bar(device_counts, x="Age_Group", y="Count", color="Primary_Device", barmode="group",
                title="Device Preference by Age Group")
    return fig

@app.callback(
    Output("screen-time-distribution", "figure"),
    Input("screen-time-distribution", "id")
)
def update_screen_time_distribution(_):
    fig = px.histogram(df, x="Avg_Daily_Screen_Time_hr", color="Exceeded_Recommended_Limit",
                     nbins=20, marginal="box",
                     title="Distribution of Daily Screen Time",
                     labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"})
    
    # Add a vertical line for recommended limit (assuming 2 hours for children)
    fig.add_vline(x=2, line_dash="dash", line_color="red",
                annotation_text="Recommended Limit", annotation_position="top right")
    
    return fig

@app.callback(
    Output("screen-time-health", "figure"),
    Input("screen-time-health", "id")
)
def update_screen_time_health(_):
    # Extract unique health issues
    health_columns = [col for col in df.columns if col in ['Poor Sleep', 'Eye Strain', 'Anxiety', 'Obesity', 'None']]
    
    # Create a new dataframe for plotting
    plot_data = []
    for health_issue in health_columns:
        if health_issue in df.columns:
            temp_df = df[df[health_issue] == 1].copy()
            temp_df['Health_Issue'] = health_issue
            plot_data.append(temp_df)
    
    if plot_data:
        plot_df = pd.concat(plot_data)
        fig = px.box(plot_df, x="Health_Issue", y="Avg_Daily_Screen_Time_hr",
                    title="Screen Time vs Health Issues",
                    labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"})
        return fig
    else:
        # Fallback if health columns not found
        fig = px.box(df, x="Health_Impacts", y="Avg_Daily_Screen_Time_hr",
                    title="Screen Time vs Health Issues",
                    labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"})
        return fig

@app.callback(
    Output("health-by-age", "figure"),
    Input("health-by-age", "id")
)
def update_health_by_age(_):
    # Similar approach as above but grouped by age
    age_bins = [5, 10, 13, 16, 19]
    age_labels = ['5-10', '11-13', '14-16', '17-19']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    health_columns = [col for col in df.columns if col in ['Poor Sleep', 'Eye Strain', 'Anxiety', 'Obesity', 'None']]
    
    # Create a new dataframe for plotting
    plot_data = []
    for health_issue in health_columns:
        if health_issue in df.columns:
            health_by_age = df.groupby('Age_Group')[health_issue].mean().reset_index()
            health_by_age['Health_Issue'] = health_issue
            health_by_age['Percentage'] = health_by_age[health_issue] * 100
            plot_data.append(health_by_age)
    
    if plot_data:
        plot_df = pd.concat(plot_data)
        fig = px.bar(plot_df, x="Age_Group", y="Percentage", color="Health_Issue", barmode="group",
                    title="Health Issues by Age Group",
                    labels={"Percentage": "Percentage of Children (%)"})
        return fig
    else:
        # Fallback
        fig = go.Figure()
        fig.update_layout(title="Health Issues by Age Group - Data Not Available")
        return fig

@app.callback(
    Output("urban-rural-comparison", "figure"),
    Input("urban-rural-comparison", "id")
)
def update_urban_rural_comparison(_):
    fig = px.box(df, x="Urban_or_Rural", y="Avg_Daily_Screen_Time_hr", color="Gender",
                title="Urban vs Rural Screen Time Comparison",
                labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"})
    return fig

@app.callback(
    Output("device-urban-rural", "figure"),
    Input("device-urban-rural", "id")
)
def update_device_urban_rural(_):
    device_location = df.groupby(['Urban_or_Rural', 'Primary_Device']).size().reset_index(name='Count')
    fig = px.bar(device_location, x="Urban_or_Rural", y="Count", color="Primary_Device", barmode="group",
                title="Device Usage: Urban vs Rural")
    return fig

@app.callback(
    Output("edu-rec-ratio", "figure"),
    Input("edu-rec-ratio", "id")
)
def update_edu_rec_ratio(_):
    # Calculate average educational and recreational time
    df['Educational_Time'] = df['Avg_Daily_Screen_Time_hr'] * df['Educational_to_Recreational_Ratio']
    df['Recreational_Time'] = df['Avg_Daily_Screen_Time_hr'] - df['Educational_Time']
    
    # Create a new dataframe for plotting
    edu_rec_df = pd.melt(df, 
                        id_vars=['Age', 'Gender', 'Urban_or_Rural'], 
                        value_vars=['Educational_Time', 'Recreational_Time'],
                        var_name='Screen_Time_Type', 
                        value_name='Hours')
    
    fig = px.box(edu_rec_df, x="Age", y="Hours", color="Screen_Time_Type",
                title="Educational vs Recreational Screen Time by Age",
                labels={"Hours": "Hours per Day"})
    return fig

@app.callback(
    Output("user-segments", "figure"),
    Input("user-segments", "id")
)
def update_user_segments(_):
    fig = px.scatter(df, x="Avg_Daily_Screen_Time_hr", y="Educational_to_Recreational_Ratio",
                   color="User_Segment", hover_data=['Age', 'Gender'],
                   title="User Segments by Screen Time and Educational Ratio",
                   labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)",
                          "Educational_to_Recreational_Ratio": "Educational to Recreational Ratio"})
    return fig

@app.callback(
    Output("risk-factors", "figure"),
    Input("risk-factors", "id")
)
def update_risk_factors(_):
    # Feature importance from the model
    feature_importance = pd.DataFrame({
        'Feature': risk_model.feature_names_in_,
        'Importance': risk_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Feature', y='Importance',
                title=f"Risk Factors for Excessive Screen Time (Model Accuracy: {model_accuracy:.2f})",
                labels={"Importance": "Feature Importance"})
    return fig

@app.callback(
    Output("risk-by-age", "figure"),
    Input("risk-by-age", "id")
)
def update_risk_by_age(_):
    fig = px.scatter(df, x="Age", y="Risk_Score", color="Risk_Category",
                   title="Risk Score by Age",
                   labels={"Risk_Score": "Risk Score (0-1)"})
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)