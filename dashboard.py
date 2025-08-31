import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime

# Load the processed data
try:
    df = pd.read_csv('processed_data.csv')
    data_loaded = True
except FileNotFoundError:
    print("Warning: processed_data.csv not found. Run data_analysis.py first.")
    # Load the original data as fallback
    try:
        df = pd.read_csv('Indian_Kids_Screen_Time.csv')
        # Add minimal processing for dashboard to work
        df['Educational_Time'] = df['Avg_Daily_Screen_Time_hr'] * df['Educational_to_Recreational_Ratio']
        df['Recreational_Time'] = df['Avg_Daily_Screen_Time_hr'] - df['Educational_Time']
        df['Risk_Category'] = 'Unknown'  # Placeholder
        df['User_Segment'] = 'Unknown'   # Placeholder
        data_loaded = True
    except FileNotFoundError:
        print("Error: No data file found. Please ensure the dataset is available.")
        df = pd.DataFrame()  # Empty dataframe
        data_loaded = False

# Custom CSS for enhanced styling
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap",
]

# Initialize the Dash app with custom theme
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server  # For deployment

# Define app_css variable for custom styling
app_css = '''
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;
    }
    .app-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem;
        color: white;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .app-title {
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .app-subtitle {
        font-weight: 300;
        opacity: 0.9;
    }
    .section-title {
        color: #333;
        font-weight: 600;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .stat-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .stat-card h3 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stat-card p {
        font-size: 0.9rem;
        margin-bottom: 0;
        opacity: 0.8;
    }
    .stat-card.primary {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    .stat-card.secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .stat-card.success {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .stat-card.warning {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
    }
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .custom-tabs {
        margin-top: 1rem;
    }
    .custom-tab {
        background-color: #f8f9fa;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
    }
    .custom-tab--selected {
        background-color: white;
        border-top: 3px solid #6a11cb;
        color: #6a11cb;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
    }
'''

# Custom CSS for the dashboard
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Indian Kids Screen Time Analysis 2025</title>
        {%favicon%}
        {%css%}
        <style>
''' + app_css + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Get current date for the dashboard
current_date = datetime.now().strftime("%B %d, %Y")


# App header with logo and title
current_year = datetime.now().year
app_header = html.Div(
    className="dashboard-header",
    children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(
                            [html.I(className="fas fa-mobile-alt mr-2"), " Indian Kids Screen Time Analysis 2025"],
                            className="display-4 font-weight-bold mb-2"
                        ),
                        html.P(
                            "Interactive dashboard analyzing screen time patterns, health impacts, and risk factors for Indian children.",
                            className="lead mb-0"
                        ),
                        html.Div([
                            html.Span("Last updated: ", className="text-light"),
                            html.Span(f"{current_year}-{datetime.now().strftime('%m-%d')}", className="font-weight-bold")
                        ], className="mt-2")
                    ])
                ], width=12)
            ])
        ])
    ]
)

# Tab content functions
def create_overview_tab():
    if not data_loaded:
        return html.Div("No data available. Please run data_analysis.py first.", className="p-5 text-center")
    
    # Calculate key statistics
    total_children = len(df)
    age_min = df['Age'].min()
    age_max = df['Age'].max()
    avg_screen_time = df['Avg_Daily_Screen_Time_hr'].mean()
    exceeding_limit_count = df['Exceeded_Recommended_Limit'].sum()
    exceeding_limit_pct = df['Exceeded_Recommended_Limit'].mean() * 100
    
    # Age distribution with improved styling
    age_fig = px.histogram(
        df, x="Age", color="Gender", 
        title="Age Distribution by Gender",
        labels={"Age": "Age (years)", "count": "Number of Children"},
        color_discrete_map={"Male": "#2575fc", "Female": "#f093fb"},
        template="plotly_white"
    )
    age_fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        legend_font_size=12,
        hoverlabel=dict(font_size=14),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Screen time distribution with improved styling
    screen_time_fig = px.histogram(
        df, x="Avg_Daily_Screen_Time_hr", 
        title="Screen Time Distribution",
        labels={"Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)", "count": "Number of Children"},
        color_discrete_sequence=["#6a11cb"],
        template="plotly_white"
    )
    screen_time_fig.add_vline(
        x=2, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Recommended Limit",
        annotation_position="top right",
        annotation_font_size=14,
        annotation_font_color="red"
    )
    screen_time_fig.update_layout(
        title_font_size=20,
        hoverlabel=dict(font_size=14),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Device preference with improved styling
    device_fig = px.pie(
        df, names="Primary_Device", 
        title="Device Preference Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma,
        template="plotly_white"
    )
    device_fig.update_traces(textposition='inside', textinfo='percent+label')
    device_fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        legend_font_size=12,
        hoverlabel=dict(font_size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Urban vs Rural distribution with improved styling
    location_fig = px.pie(
        df, names="Urban_or_Rural", 
        title="Urban vs Rural Distribution",
        hole=0.4,
        color_discrete_sequence=["#43e97b", "#f6d365"],
        template="plotly_white"
    )
    location_fig.update_traces(textposition='inside', textinfo='percent+label')
    location_fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        legend_font_size=12,
        hoverlabel=dict(font_size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Dataset Overview", className="section-title text-center")
            ], width=12)
        ]),
        
        # Key statistics in colorful cards
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(f"{total_children:,}"),
                    html.P("Total Children")
                ], className="stat-card primary")
            ], width=12, lg=3, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H3(f"{age_min}-{age_max}"),
                    html.P("Age Range (years)")
                ], className="stat-card secondary")
            ], width=12, lg=3, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H3(f"{avg_screen_time:.1f}"),
                    html.P("Avg. Daily Screen Time (hours)")
                ], className="stat-card success")
            ], width=12, lg=3, className="mb-4"),
            
            dbc.Col([
                html.Div([
                    html.H3(f"{exceeding_limit_pct:.1f}%"),
                    html.P("Exceeding Recommended Limit")
                ], className="stat-card warning")
            ], width=12, lg=3, className="mb-4")
        ]),
        
        # Charts in card containers
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=age_fig)
                ], className="chart-container")
            ], width=12, lg=6),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=screen_time_fig)
                ], className="chart-container")
            ], width=12, lg=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=device_fig)
                ], className="chart-container")
            ], width=12, lg=6),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=location_fig)
                ], className="chart-container")
            ], width=12, lg=6)
        ])
    ])

def create_patterns_tab():
    if not data_loaded:
        return html.Div("No data available. Please run data_analysis.py first.")
    
    # Screen time by age and gender
    age_gender_fig = px.box(
        df, x="Age", y="Avg_Daily_Screen_Time_hr", color="Gender",
        title="Screen Time by Age and Gender",
        labels={"Age": "Age (years)", "Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"}
    )
    
    # Create age groups
    if 'Age_Group' not in df.columns:
        age_bins = [5, 10, 13, 16, 19]
        age_labels = ['5-10', '11-13', '14-16', '17-19']
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Device preference by age group
    device_age_df = df.groupby(['Age_Group', 'Primary_Device']).size().reset_index(name='Count')
    device_age_fig = px.bar(
        device_age_df, x="Age_Group", y="Count", color="Primary_Device",
        title="Device Preference by Age Group",
        labels={"Age_Group": "Age Group", "Count": "Number of Children"}
    )
    
    # Screen time by location
    location_fig = px.box(
        df, x="Urban_or_Rural", y="Avg_Daily_Screen_Time_hr",
        title="Screen Time by Location",
        labels={"Urban_or_Rural": "Location", "Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)"}
    )
    
    # Device preference by location
    device_loc_df = df.groupby(['Urban_or_Rural', 'Primary_Device']).size().reset_index(name='Count')
    device_loc_fig = px.bar(
        device_loc_df, x="Urban_or_Rural", y="Count", color="Primary_Device",
        title="Device Preference by Location",
        labels={"Urban_or_Rural": "Location", "Count": "Number of Children"}
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=age_gender_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=device_age_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=location_fig)], width=6),
            dbc.Col([dcc.Graph(figure=device_loc_fig)], width=6)
        ])
    ])

def create_health_tab():
    if not data_loaded:
        return html.Div("No data available. Please run data_analysis.py first.")
    
    # Extract health impact columns if they exist
    health_columns = [col for col in df.columns if col in ['Poor Sleep', 'Eye Strain', 'Anxiety', 'Obesity', 'None']]
    
    if not health_columns:
        # If health columns don't exist, create a placeholder visualization
        health_fig = go.Figure()
        health_fig.add_annotation(
            text="Health impact data not available. Run data_analysis.py first.",
            showarrow=False,
            font=dict(size=20)
        )
    else:
        # Create health impact data for visualization
        health_data = []
        for health_issue in health_columns:
            if health_issue != 'None':
                avg_time = df[df[health_issue] == 1]['Avg_Daily_Screen_Time_hr'].mean()
                health_data.append({
                    'Health_Issue': health_issue,
                    'Avg_Screen_Time': avg_time,
                    'Count': df[health_issue].sum()
                })
        
        health_df = pd.DataFrame(health_data)
        
        # Health issues by screen time
        health_fig = px.bar(
            health_df, x="Health_Issue", y="Avg_Screen_Time",
            title="Average Screen Time by Health Issue",
            labels={"Health_Issue": "Health Issue", "Avg_Screen_Time": "Average Daily Screen Time (hours)"}
        )
        
        # Health issues prevalence
        health_count_fig = px.bar(
            health_df, x="Health_Issue", y="Count",
            title="Prevalence of Health Issues",
            labels={"Health_Issue": "Health Issue", "Count": "Number of Children"}
        )
    
    # Screen time vs health scatter plot
    scatter_fig = px.scatter(
        df, x="Avg_Daily_Screen_Time_hr", y="Educational_to_Recreational_Ratio", 
        color="Exceeded_Recommended_Limit",
        title="Screen Time vs Educational Ratio (Color by Exceeded Limit)",
        labels={
            "Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)", 
            "Educational_to_Recreational_Ratio": "Educational to Recreational Ratio",
            "Exceeded_Recommended_Limit": "Exceeded Limit"
        }
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Health Impact Analysis", className="text-center mb-4"),
                html.P("This section explores the relationship between screen time and various health impacts reported in children.")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=health_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=health_count_fig if 'health_count_fig' in locals() else health_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=scatter_fig)], width=12)
        ])
    ])

def create_educational_tab():
    if not data_loaded:
        return html.Div("No data available. Please run data_analysis.py first.")
    
    # Educational vs Recreational time
    edu_rec_df = pd.DataFrame({
        'Type': ['Educational'] * len(df) + ['Recreational'] * len(df),
        'Hours': df['Educational_Time'].tolist() + df['Recreational_Time'].tolist(),
        'Age': df['Age'].tolist() * 2,
        'Gender': df['Gender'].tolist() * 2
    })
    
    edu_rec_fig = px.box(
        edu_rec_df, x="Type", y="Hours",
        title="Educational vs Recreational Screen Time",
        labels={"Type": "Screen Time Type", "Hours": "Hours per Day"}
    )
    
    # Educational ratio by age
    ratio_age_fig = px.scatter(
        df, x="Age", y="Educational_to_Recreational_Ratio", color="Gender",
        title="Educational to Recreational Ratio by Age",
        labels={"Age": "Age (years)", "Educational_to_Recreational_Ratio": "Educational to Recreational Ratio"}
    )
    
    # Educational time by device
    edu_device_fig = px.box(
        df, x="Primary_Device", y="Educational_Time",
        title="Educational Screen Time by Device",
        labels={"Primary_Device": "Device", "Educational_Time": "Educational Screen Time (hours)"}
    )
    
    # Recreational time by device
    rec_device_fig = px.box(
        df, x="Primary_Device", y="Recreational_Time",
        title="Recreational Screen Time by Device",
        labels={"Primary_Device": "Device", "Recreational_Time": "Recreational Screen Time (hours)"}
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Educational vs Recreational Screen Time", className="text-center mb-4"),
                html.Div([
                    html.P(f"Average Educational Time: {df['Educational_Time'].mean():.2f} hours"),
                    html.P(f"Average Recreational Time: {df['Recreational_Time'].mean():.2f} hours"),
                    html.P(f"Average Educational to Recreational Ratio: {df['Educational_to_Recreational_Ratio'].mean():.2f}")
                ])
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=edu_rec_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=ratio_age_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=edu_device_fig)], width=6),
            dbc.Col([dcc.Graph(figure=rec_device_fig)], width=6)
        ])
    ])

def create_segments_tab():
    if not data_loaded or 'User_Segment' not in df.columns or df['User_Segment'].eq('Unknown').all():
        return html.Div("User segment data not available. Please run data_analysis.py first.")
    
    # User segments scatter plot
    segments_fig = px.scatter(
        df, x="Avg_Daily_Screen_Time_hr", y="Educational_to_Recreational_Ratio", color="User_Segment",
        title="User Segments by Screen Time and Educational Ratio",
        labels={
            "Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)", 
            "Educational_to_Recreational_Ratio": "Educational to Recreational Ratio",
            "User_Segment": "User Segment"
        }
    )
    
    # Segment distribution
    segment_dist_fig = px.histogram(
        df, x="User_Segment", color="User_Segment",
        title="Distribution of User Segments",
        labels={"User_Segment": "User Segment", "count": "Number of Children"}
    )
    
    # Segment by age
    segment_age_fig = px.box(
        df, x="User_Segment", y="Age", color="User_Segment",
        title="Age Distribution by User Segment",
        labels={"User_Segment": "User Segment", "Age": "Age (years)"}
    )
    
    # Segment by location
    segment_loc_df = df.groupby(['User_Segment', 'Urban_or_Rural']).size().reset_index(name='Count')
    segment_loc_fig = px.bar(
        segment_loc_df, x="User_Segment", y="Count", color="Urban_or_Rural",
        title="User Segments by Location",
        labels={"User_Segment": "User Segment", "Count": "Number of Children", "Urban_or_Rural": "Location"}
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("User Segmentation Analysis", className="text-center mb-4"),
                html.P("This section presents the clustering analysis of children based on their screen time patterns.")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=segments_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=segment_dist_fig)], width=6),
            dbc.Col([dcc.Graph(figure=segment_age_fig)], width=6)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=segment_loc_fig)], width=12)
        ])
    ])

def create_risk_tab():
    if not data_loaded or 'Risk_Category' not in df.columns or df['Risk_Category'].eq('Unknown').all():
        return html.Div("Risk prediction data not available. Please run data_analysis.py first.")
    
    # Risk score by age
    if 'Risk_Score' in df.columns:
        risk_age_fig = px.scatter(
            df, x="Age", y="Risk_Score", color="Risk_Category",
            title="Risk Score by Age",
            labels={"Age": "Age (years)", "Risk_Score": "Risk Score (0-1)", "Risk_Category": "Risk Category"}
        )
    else:
        # Fallback if Risk_Score is not available
        risk_age_fig = px.scatter(
            df, x="Age", y="Avg_Daily_Screen_Time_hr", color="Risk_Category",
            title="Screen Time by Age (Colored by Risk Category)",
            labels={"Age": "Age (years)", "Avg_Daily_Screen_Time_hr": "Daily Screen Time (hours)", "Risk_Category": "Risk Category"}
        )
    
    # Risk distribution
    risk_dist_fig = px.histogram(
        df, x="Risk_Category", color="Risk_Category",
        title="Distribution of Risk Categories",
        labels={"Risk_Category": "Risk Category", "count": "Number of Children"}
    )
    
    # Risk by location
    risk_loc_df = df.groupby(['Risk_Category', 'Urban_or_Rural']).size().reset_index(name='Count')
    risk_loc_fig = px.bar(
        risk_loc_df, x="Risk_Category", y="Count", color="Urban_or_Rural",
        title="Risk Categories by Location",
        labels={"Risk_Category": "Risk Category", "Count": "Number of Children", "Urban_or_Rural": "Location"}
    )
    
    # Risk by device
    risk_device_df = df.groupby(['Risk_Category', 'Primary_Device']).size().reset_index(name='Count')
    risk_device_fig = px.bar(
        risk_device_df, x="Risk_Category", y="Count", color="Primary_Device",
        title="Risk Categories by Device",
        labels={"Risk_Category": "Risk Category", "Count": "Number of Children", "Primary_Device": "Device"}
    )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Risk Prediction Analysis", className="text-center mb-4"),
                html.P("This section presents the predictive modeling results for identifying children at risk of excessive screen time.")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=risk_age_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=risk_dist_fig)], width=12)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=risk_loc_fig)], width=6),
            dbc.Col([dcc.Graph(figure=risk_device_fig)], width=6)
        ])
    ])

# App layout
app.layout = html.Div([
    # Header section with gradient background
    html.Div([
        html.H1("Indian Kids Screen Time Analysis 2025", className="app-title"),
        html.P("Interactive Dashboard for Screen Time Analysis and Risk Prediction", className="app-subtitle"),
        html.P(f"Last updated: {current_date}", className="mt-2 small")
    ], className="app-header text-center mb-4"),
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="overview", className="custom-tab", active_label_class_name="custom-tab--selected"),
            dbc.Tab(label="Screen Time Patterns", tab_id="patterns", className="custom-tab", active_label_class_name="custom-tab--selected"),
            dbc.Tab(label="Health Impacts", tab_id="health", className="custom-tab", active_label_class_name="custom-tab--selected"),
            dbc.Tab(label="Educational vs Recreational", tab_id="educational", className="custom-tab", active_label_class_name="custom-tab--selected"),
            dbc.Tab(label="User Segments", tab_id="segments", className="custom-tab", active_label_class_name="custom-tab--selected"),
            dbc.Tab(label="Risk Prediction", tab_id="risk", className="custom-tab", active_label_class_name="custom-tab--selected")
        ], id="tabs", active_tab="overview", className="custom-tabs"),
        html.Div(id="tab-content", className="p-4")
    ]),
    html.Div([
        html.P("© 2025 Indian Kids Screen Time Analysis Project"),
        html.P("Data is simulated for educational purposes", className="small")
    ], className="footer")
])

# Callback to render tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "overview":
        return create_overview_tab()
    elif active_tab == "patterns":
        return create_patterns_tab()
    elif active_tab == "health":
        return create_health_tab()
    elif active_tab == "educational":
        return create_educational_tab()
    elif active_tab == "segments":
        return create_segments_tab()
    elif active_tab == "risk":
        return create_risk_tab()
    return "No content available"

# Run the app
if __name__ == '__main__':
    print("Starting dashboard server...")
    app.run_server(debug=True)