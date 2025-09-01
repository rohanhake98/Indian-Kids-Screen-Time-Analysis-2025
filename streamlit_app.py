import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Indian Kids Screen Time Analysis 2025",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 60px;
        color: #6a11cb;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 36px;
        color: #2575fc;
        font-weight: 500;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .stat-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Indian Kids Screen Time Analysis 2025</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Insights into digital habits of Indian children</p>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        try:
            # Load and process the original data if processed data is not available
            df = pd.read_csv('Indian_Kids_Screen_Time.csv')
        except FileNotFoundError:
            try:
                # Load sample data if neither processed nor original data is available
                df = pd.read_csv('sample_data.csv')
                
                # Basic data cleaning
                df = df.dropna()
                df['Gender'] = df['Gender'].astype('category')
                df['Primary_Device'] = df['Primary_Device'].astype('category')
                df['Urban_or_Rural'] = df['Urban_or_Rural'].astype('category')
                df['Exceeded_Recommended_Limit'] = df['Exceeded_Recommended_Limit'].astype('bool')
                
                # Extract health impacts into separate columns
                health_impacts = df['Health_Impacts'].str.get_dummies(sep=', ')
                df = pd.concat([df, health_impacts], axis=1)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return pd.DataFrame()
            
        # Calculate educational and recreational time
        df['Educational_Time'] = df['Avg_Daily_Screen_Time_hr'] * df['Educational_to_Recreational_Ratio']
        df['Recreational_Time'] = df['Avg_Daily_Screen_Time_hr'] - df['Educational_Time']
        
        # Perform clustering for user segmentation
        features = ['Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Label the clusters
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_labels = ['Low Usage', 'Moderate Usage', 'Heavy Usage']
        sorted_indices = np.argsort(cluster_centers[:, 0])
        cluster_map = {i: cluster_labels[j] for i, j in enumerate(sorted_indices)}
        df['User_Segment'] = df['Cluster'].map(cluster_map)
        
        # Build risk model
        features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
        X = df[features]
        y = df['Exceeded_Recommended_Limit']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        df['Risk_Score'] = model.predict_proba(df[features])[:, 1]
        df['Risk_Category'] = pd.qcut(df['Risk_Score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        # Create age groups
        age_bins = [5, 10, 13, 16, 19]
        age_labels = ['5-10', '11-13', '14-16', '17-19']
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        
    except FileNotFoundError:
        st.error("Error: No data file found. Please ensure the dataset is available.")
        return pd.DataFrame()
    
    return df

# Load the data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Screen Time Analysis", "User Segments", "Risk Analysis", "Health Impacts", "Recommendations"]
)

# Dashboard page
if page == "Dashboard":
    st.markdown("## Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='stat-card'>"
                    f"<h3>{df['Avg_Daily_Screen_Time_hr'].mean():.1f} hrs</h3>"
                    "<p>Average Daily Screen Time</p>"
                    "</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-card'>"
                    f"<h3>{(df['Exceeded_Recommended_Limit'].mean() * 100):.1f}%</h3>"
                    "<p>Exceed Recommended Limit</p>"
                    "</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-card'>"
                    f"<h3>{df['Educational_to_Recreational_Ratio'].mean():.2f}</h3>"
                    "<p>Avg Edu/Rec Ratio</p>"
                    "</div>", unsafe_allow_html=True)
    
    with col4:
        high_risk = (df['Risk_Category'] == 'High Risk').mean() * 100
        st.markdown("<div class='stat-card'>"
                    f"<h3>{high_risk:.1f}%</h3>"
                    "<p>High Risk Children</p>"
                    "</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='section-title'>Screen Time Overview</p>", unsafe_allow_html=True)
    
    # Screen time by age and gender
    fig = px.box(df, x='Age', y='Avg_Daily_Screen_Time_hr', color='Gender',
                title='Screen Time by Age and Gender')
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Recommended Limit")
    st.plotly_chart(fig, use_container_width=True)
    
    # Device preference
    col1, col2 = st.columns(2)
    
    with col1:
        device_counts = df.groupby('Primary_Device').size().reset_index(name='Count')
        fig = px.pie(device_counts, values='Count', names='Primary_Device', 
                    title='Device Preference Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        urban_rural = df.groupby(['Urban_or_Rural', 'Primary_Device']).size().reset_index(name='Count')
        fig = px.bar(urban_rural, x='Urban_or_Rural', y='Count', color='Primary_Device',
                    title='Device Preference by Location')
        st.plotly_chart(fig, use_container_width=True)

# Screen Time Analysis page
elif page == "Screen Time Analysis":
    st.markdown("<p class='section-title'>Screen Time Analysis</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Screen time distribution
        fig = px.histogram(df, x='Avg_Daily_Screen_Time_hr', nbins=20,
                        title='Distribution of Daily Screen Time')
        fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Recommended Limit")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Educational vs Recreational time
        edu_rec_df = pd.DataFrame({
            'Time Type': ['Educational', 'Recreational'],
            'Hours': [df['Educational_Time'].mean(), df['Recreational_Time'].mean()]
        })
        fig = px.pie(edu_rec_df, values='Hours', names='Time Type',
                    title='Average Educational vs Recreational Screen Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational to Recreational ratio by age
    fig = px.box(df, x='Age_Group', y='Educational_to_Recreational_Ratio', color='Gender',
                title='Educational to Recreational Ratio by Age Group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Urban vs Rural comparison
    urban_rural_df = df.groupby('Urban_or_Rural')['Avg_Daily_Screen_Time_hr'].mean().reset_index()
    fig = px.bar(urban_rural_df, x='Urban_or_Rural', y='Avg_Daily_Screen_Time_hr',
                title='Average Screen Time: Urban vs Rural')
    st.plotly_chart(fig, use_container_width=True)

# User Segments page
elif page == "User Segments":
    st.markdown("<p class='section-title'>User Segments</p>", unsafe_allow_html=True)
    
    # User segments distribution
    segment_counts = df.groupby('User_Segment').size().reset_index(name='Count')
    fig = px.pie(segment_counts, values='Count', names='User_Segment',
                title='Distribution of User Segments')
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics
    segment_stats = df.groupby('User_Segment')[
        ['Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio', 'Age']
    ].mean().reset_index()
    
    fig = px.bar(segment_stats, x='User_Segment', y=['Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio', 'Age'],
                title='Characteristics of User Segments',
                barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment by age group
    segment_age = df.groupby(['Age_Group', 'User_Segment']).size().reset_index(name='Count')
    fig = px.bar(segment_age, x='Age_Group', y='Count', color='User_Segment',
                title='User Segments by Age Group')
    st.plotly_chart(fig, use_container_width=True)

# Risk Analysis page
elif page == "Risk Analysis":
    st.markdown("<p class='section-title'>Risk Analysis</p>", unsafe_allow_html=True)
    
    # Risk category distribution
    risk_counts = df.groupby('Risk_Category').size().reset_index(name='Count')
    fig = px.pie(risk_counts, values='Count', names='Risk_Category',
                title='Distribution of Risk Categories')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk by age group
    risk_age = df.groupby(['Age_Group', 'Risk_Category']).size().reset_index(name='Count')
    fig = px.bar(risk_age, x='Age_Group', y='Count', color='Risk_Category',
                title='Risk Categories by Age Group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.markdown("### Key Risk Factors")
    
    # Feature importance from the risk model
    if 'Risk_Score' in df.columns:
        features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
        X = df[features]
        y = df['Exceeded_Recommended_Limit']
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title='Feature Importance for Risk Prediction')
        st.plotly_chart(fig, use_container_width=True)

# Health Impacts page
elif page == "Health Impacts":
    st.markdown("<p class='section-title'>Health Impacts</p>", unsafe_allow_html=True)
    
    # Get health impact columns
    health_columns = [col for col in df.columns if col not in [
        'Age', 'Gender', 'Primary_Device', 'Urban_or_Rural', 'Avg_Daily_Screen_Time_hr',
        'Educational_to_Recreational_Ratio', 'Health_Impacts', 'Exceeded_Recommended_Limit',
        'Cluster', 'User_Segment', 'Risk_Score', 'Risk_Category', 'Age_Group',
        'Educational_Time', 'Recreational_Time'
    ]]
    
    if health_columns:
        # Health impacts prevalence
        health_data = df[health_columns].sum().reset_index()
        health_data.columns = ['Health_Impact', 'Count']
        health_data = health_data.sort_values('Count', ascending=False)
        
        fig = px.bar(health_data, x='Health_Impact', y='Count',
                    title='Prevalence of Health Impacts')
        st.plotly_chart(fig, use_container_width=True)
        
        # Health impacts by screen time
        st.markdown("### Health Impacts by Screen Time")
        
        # Create screen time bins
        df['Screen_Time_Category'] = pd.cut(
            df['Avg_Daily_Screen_Time_hr'],
            bins=[0, 2, 4, 6, 10],
            labels=['0-2 hrs', '2-4 hrs', '4-6 hrs', '6+ hrs']
        )
        
        # Select top 5 health impacts for analysis
        top_impacts = health_data.head(5)['Health_Impact'].tolist()
        
        for impact in top_impacts:
            impact_by_time = df.groupby('Screen_Time_Category')[impact].mean().reset_index()
            fig = px.bar(impact_by_time, x='Screen_Time_Category', y=impact,
                        title=f'{impact} by Screen Time Category')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No health impact data available.")

# Recommendations page
elif page == "Recommendations":
    st.markdown("<p class='section-title'>Recommendations</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Screen Time Guidelines
    
    Based on our analysis, we recommend the following guidelines for healthy screen time habits:
    
    1. **Limit daily screen time to 2 hours** for recreational purposes
    2. **Maintain an educational to recreational ratio of at least 1:1**
    3. **Take regular breaks** - 10 minutes break for every 50 minutes of screen time
    4. **No screen time at least 1 hour before bedtime**
    5. **Encourage outdoor activities** and physical exercise
    
    ### For Parents
    
    1. **Set clear boundaries** for screen time
    2. **Monitor content** children are consuming
    3. **Be a role model** by demonstrating healthy digital habits
    4. **Create screen-free zones** in the home, especially bedrooms
    5. **Use parental controls** and screen time management apps
    
    ### For Educators
    
    1. **Integrate technology meaningfully** into curriculum
    2. **Teach digital literacy** and responsible technology use
    3. **Encourage balance** between digital and traditional learning methods
    4. **Promote critical thinking** about media consumption
    5. **Communicate with parents** about healthy technology use
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>© 2025 Indian Kids Screen Time Analysis | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)