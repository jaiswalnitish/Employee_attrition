import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
from io import StringIO
from io import BytesIO

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Prediction")

#feature names from your model's booster
MODEL_FEATURES = [
    'Job_Level',
    'Years_at_Company',
    'Years_in_Current_Role',
    'Years_Since_Last_Promotion',
    'Work_Life_Balance',
    'Job_Satisfaction',
    'Performance_Rating',
    'Overtime',
    'Work_Environment_Satisfaction',
    'Relationship_with_Manager',
    'Job_Involvement',
    'Distance_From_Home',
    'Gender_Male',
    'Marital_Status_Married',
    'Marital_Status_Single',
    'Department_HR',
    'Department_IT',
    'Department_Marketing',
    'Department_Sales',
    'Job_Role_Assistant',
    'Job_Role_Executive',
    'Job_Role_Manager'
]

#load test data FIRST
@st.cache_data
def load_test_data():
    try:
        try:
            test_data = pd.read_csv('X_test.csv')[MODEL_FEATURES]
            return test_data
        except FileNotFoundError:
            st.warning("File not found at local path, loading from GitHub as automated")
            #github
            github_url = "https://raw.githubusercontent.com/jaiswalnitish/Employee_attrition/refs/heads/main/X_test.csv"
            #download the file
            response = requests.get(github_url)
            response.raise_for_status()  # Check for errors
            
            # Read into DataFrame
            test_data = pd.read_csv(StringIO(response.text))
            test_data = test_data[MODEL_FEATURES]
            return test_data

    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None
X_test = load_test_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Employee Analysis", "Department Analysis"])

@st.cache_resource
def load_model():
    """Load the trained model and validate feature names"""
    try:
        # Try local load first
        try:
            with open('xgboost_attrition_model.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            # Fallback to GitHub
            st.warning("Model file not found locally, loading from GitHub")
            github_url = "https://raw.githubusercontent.com/jaiswalnitish/Employee_attrition/refs/heads/main/xgboost_attrition_model.pkl"
            response = requests.get(github_url)
            response.raise_for_status()
            model = pickle.load(BytesIO(response.content))
        
        # Validate model type
        if not isinstance(model, xgb.XGBClassifier):
            st.error("Loaded object is not an XGBoost classifier!")
            return None, None
        
        # Get actual feature names from model
        actual_features = model.get_booster().feature_names
        
        # Validate against expected features
        if set(actual_features) != set(MODEL_FEATURES):
            st.warning(f"""
                Feature mismatch! 
                Model features: {actual_features}
                Expected features: {MODEL_FEATURES}
                Using model's features.
            """)
        
        return model, actual_features  # Always return tuple (model, features)
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Load the model and get features
model, model_features = load_model()

# Initialize SHAP explainer
@st.cache_resource
def load_explainer():
    if model is not None:
        explainer = shap.TreeExplainer(model)
        return explainer
    return None

explainer = load_explainer()

with tab1:
    if model is not None:
        with st.form("attrition_form"):
            st.markdown("Please fill in the following details to predict attrition risk:")
            # Demographic Info
            st.subheader("Employee Demographic Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.radio("Gender:", ["Male", "Female"])
            with col2:
                marital_status = st.selectbox("Marital Status:", ["Single", "Married", "Divorced"])
            with col3:
                department = st.selectbox("Department:", ["HR", "IT", "Marketing", "Sales", "Finance"])
                job_role = st.selectbox("Job Role:", ["Analyst", "Assistant", "Executive", "Manager"])
            
            st.header("Employee Details")
            # Work Experience
            col4, col5 = st.columns(2)
            with col4:
                job_level = st.selectbox("Job Level:", [1, 2, 3, 4, 5])
                years_at_company = st.number_input("Years at Company:", min_value=0, max_value=40, value=3)
                years_in_current_role = st.number_input("Years in Current Role:", min_value=0, max_value=40, value=2)
                

            with col5:
                years_since_last_promotion = st.number_input("Years Since Last Promotion:", min_value=0, max_value=40, value=2)
                overtime = st.selectbox("Works Overtime:", ["No", "Yes"])
                distance_from_home = st.number_input("Distance From Home (miles):", min_value=0, max_value=200, value=10)
                
            
            # Satisfaction Ratings (1-5 scale)
            st.subheader("Job Satisfaction Ratings (1-5)")
            col6, col7 = st.columns(2)
            with col6:
                work_life_balance = st.slider("Work-Life Balance:", 1, 4, 3)
                job_satisfaction = st.slider("Job Satisfaction:", 1, 5, 3)
                performance_rating = st.slider("Performance Rating:", 1, 4, 3)

            with col7:
                work_env_satisfaction = st.slider("Work Environment Satisfaction:", 1, 4, 3)
                relationship_manager = st.slider("Relationship with Manager:", 1, 4, 3)
                job_involvement = st.slider("Job Involvement:", 1, 4, 3)
            
            
            
            submitted = st.form_submit_button("Predict Attrition Risk")

        if submitted:
            # Prepare input data with EXACT feature names and order
            input_data = {
                'Job_Level': job_level,
                'Years_at_Company': years_at_company,
                'Years_in_Current_Role': years_in_current_role,
                'Years_Since_Last_Promotion': years_since_last_promotion,
                'Work_Life_Balance': work_life_balance,
                'Job_Satisfaction': job_satisfaction,
                'Performance_Rating': performance_rating,
                'Overtime': 1 if overtime == "Yes" else 0,
                'Work_Environment_Satisfaction': work_env_satisfaction,
                'Relationship_with_Manager': relationship_manager,
                'Job_Involvement': job_involvement,
                'Distance_From_Home': distance_from_home,
                'Gender_Male': 1 if gender == "Male" else 0,
                'Marital_Status_Married': 1 if marital_status == "Married" else 0,
                'Marital_Status_Single': 1 if marital_status == "Single" else 0,
                'Department_HR': 1 if department == "HR" else 0,
                'Department_IT': 1 if department == "IT" else 0,
                'Department_Marketing': 1 if department == "Marketing" else 0,
                'Department_Sales': 1 if department == "Sales" else 0,
                'Job_Role_Assistant': 1 if job_role == "Assistant" else 0,
                'Job_Role_Executive': 1 if job_role == "Executive" else 0,
                'Job_Role_Manager': 1 if job_role == "Manager" else 0
            }
            
                # Store in session state
            st.session_state['input_data'] = input_data

            # Create DataFrame with EXACTLY the features and order the model expects
            input_df = pd.DataFrame([input_data])[model_features]
            
            try:
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                
                # Display results
                st.header("Prediction Results")
                
                leave_prob = prediction_proba[0][1] * 100
                stay_prob = 100 - leave_prob

                # Store in session state for tab2
                st.session_state['leave_prob'] = leave_prob
                st.session_state['stay_prob'] = stay_prob

                if leave_prob >= 50:
                    st.error(f"🚨 Employee at Attrition Risk: {leave_prob:.1f}% probability")
                else:
                    st.success(f"✅ Employee at Staying-back Likelihood: {stay_prob:.1f}% probability")
                
                # Probability visualization
                proba_df = pd.DataFrame({
                    'Outcome': ['Stay', 'Leave'],
                    'Probability': [stay_prob, leave_prob]
                })
                st.bar_chart(proba_df.set_index('Outcome'))
                
                # Store the input for SHAP analysis
                st.session_state['shap_input'] = input_df
                st.session_state['shap_prediction'] = prediction_proba
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                if st.checkbox("Show debug info"):
                    st.write("Input features provided:", input_df.columns.tolist())
                    st.write("Model expects:", model.get_booster().feature_names)
                    st.write("Input data sample:", input_df)
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

with tab2:
    st.header("SHAP Value Analysis")

    if 'shap_input' not in st.session_state or explainer is None:
        st.warning("Please make a prediction first to see analysis")
    else:
        # Check if required session data exists
        if 'input_data' not in st.session_state or 'leave_prob' not in st.session_state:
            st.error("No prediction data available")
            st.stop()
            
        # Retrieve all stored data
        input_df = st.session_state['shap_input']
        input_data = st.session_state['input_data']
        leave_prob = st.session_state['leave_prob']  # Get the stored probability
        shap_values = explainer.shap_values(input_df)
        

        # 1. Get the ACTIVE categorical features from user input
        active_features = []

        # Marital Status
        if input_data['Marital_Status_Married'] == 1:
            active_features.append('Marital_Status_Married')
        elif input_data['Marital_Status_Single'] == 1:
            active_features.append('Marital_Status_Single')
        # Note: Divorced would be when both are 0

        # Department 
        for dept in ['HR', 'IT', 'Marketing', 'Sales']:
            if input_data[f'Department_{dept}'] == 1:
                active_features.append(f'Department_{dept}')
                break

        # Job Role
        for role in ['Assistant', 'Executive', 'Manager']:
            if input_data[f'Job_Role_{role}'] == 1:
                active_features.append(f'Job_Role_{role}')
                break

        # Gender (only show if Male)
        if input_data['Gender_Male'] == 1:
            active_features.append('Gender_Male')

        # 2. Filter the SHAP display to only show:
        # - The active categorical features
        # - All continuous/numerical features
        display_features = [
            f for f in input_df.columns 
            if (f in active_features) or  # Show active categories
            (not any(f.startswith(prefix) for prefix in ['Department_', 'Job_Role_', 'Marital_Status_', 'Gender_']))  # Exclude other categories
        ]

        # 3. Apply filtering to the SHAP visualization
        filtered_shap_values = shap_values[0][[model_features.index(f) for f in display_features]]
        filtered_features = display_features

        # Now use filtered_shap_values and filtered_features in your plots:
        fig = px.bar(
            x=filtered_shap_values,
            y=filtered_features,
            color=filtered_shap_values > 0,
            color_discrete_map={True: '#ff0051', False: "#008bfb"},
            labels={'x': 'SHAP Value', 'y': 'Feature'},
            title="How Features Affect This Employee's Attrition Risk",
            height=500
        )

        # Update legend names
        fig.update_traces(
            selector={'name': 'True'},
            name='Increases Risk',
            legendgroup='Increases Risk',
            hovertemplate='Feature: %{y}<br>Impact: +%{x:.2f}<extra></extra>'
        )
        fig.update_traces(
            selector={'name': 'False'},
            name='Decreases Risk',
            legendgroup='Decreases Risk',
            hovertemplate='Feature: %{y}<br>Impact: -%{x:.2f}<extra></extra>'
        )

        fig.update_layout(
            hovermode="y",
            showlegend=True,legend_title_text="Impact",
            title_x=0.3, legend_traceorder="reversed",
            xaxis_title="SHAP Value (Positive → Increases Attrition)",
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=False)
        
        # 2. Add interpretation text
        st.markdown("""
            **How to read this plot:**
            - <span style='color:red'>**→Red bars**</span>: Features that increase attrition risk when values are high
            - <span style='color:blue'>**←Blue bars**</span>: Features that decrease attrition risk when values are high
            - **Bar length**: Strength of the effect
            """, unsafe_allow_html=True)
        
        
    #add line spacing
        st.markdown("<br>", unsafe_allow_html=True)


        # Force plot for individual prediction
        st.subheader("Prediction Explanation")

        # Convert to probability space
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        base_prob = sigmoid(explainer.expected_value)
        shap_values_prob = explainer.shap_values(input_df) * model.predict_proba(input_df)[:,1] * (1 - model.predict_proba(input_df)[:,1])

        # Create filtered force plot in probability space
        def probability_force_plot():
            shap.initjs()
            plot = shap.force_plot(
                base_prob,
                shap_values_prob[0][[model_features.index(f) for f in display_features]],  # Use filtered features
                input_df[display_features].iloc[0].values,
                feature_names=display_features,
                link='logit',
                matplotlib=False
            )
            return f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"

        html = probability_force_plot()
        st.components.v1.html(html, height=200)



        
        # Generate plain English insights
        st.subheader("🔍 Key Takeaways")
        shap_df = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_values[0],
            'Impact': shap_values[0]  # Raw SHAP values (not absolute)
        })
        if leave_prob > 50:
            # High-risk case (red alert)
            top_3 = shap_df.nlargest(3, 'Impact')  # Top 5 positive SHAP values
            st.error("🚨 **High attrition risk !** ")
            st.write("**Top factors increasing risk:**")
  
            for _, row in top_3.iterrows():
                if 'Gender' not in row['Feature']:
                    st.write(f"- **{row['Feature']}** ➔ Increases risk by **{row['Impact']:.2f}**")
        else:
            # Low-risk case (green success)
            top_3 = shap_df.nsmallest(3, 'Impact')  # Top 5 negative SHAP values
            st.success("✅ **Low attrition risk !** ")
            st.write("**Top protective factors:**") 
            for _, row in top_3.iterrows():
                if 'Gender' not in row['Feature']:
                    st.write(f"- **{row['Feature']}** ➔ Decreases risk by **{abs(row['Impact']):.2f}**")
            



# Add some styling
st.markdown("""
<style>
    .stNumberInput, .stRadio > div, .stSelectbox > div {
        margin-bottom: 1rem;
    }
    .st-bb {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

with tab3:
    st.header("Department-specific features contribution Insights")
    
    if explainer is not None and 'X_test' in globals():

        
        # Calculate SHAP values for entire test set
        global_shap_values = explainer.shap_values(X_test[model_features])
        
        # Handle different SHAP output formats
        if hasattr(global_shap_values, 'values'):
            global_shap_array = global_shap_values.values
        else:
            global_shap_array = global_shap_values
        
        if isinstance(global_shap_array, list):
            global_shap_array = global_shap_array[1]  # For binary classification
        
        # Calculate mean SHAP values
        mean_shap = pd.DataFrame(global_shap_array, columns=X_test.columns).mean()


       
        # =====================================
        # 1. DEPARTMENT-SPECIFIC ANALYSIS
        # =====================================

        
        departments = ['HR', 'IT', 'Marketing', 'Sales', 'Finance']
        selected_dept = st.selectbox(
            "Select Department:",
            departments,
            index=0
        )
        
                # Handle department filtering (consistent with tab1 logic)
        if selected_dept == "Finance":
            # Finance employees have all other department columns = 0
            dept_mask = (X_test['Department_HR'] == 0) & \
                       (X_test['Department_IT'] == 0) & \
                       (X_test['Department_Marketing'] == 0) & \
                       (X_test['Department_Sales'] == 0)
            dept_col = None  # No specific department column for Finance
        else:
            dept_col = f'Department_{selected_dept}'
            dept_mask = X_test[dept_col] == 1
            
        if sum(dept_mask) > 10:  # Minimum sample threshold

            # Get representative employee (median)
            dept_probas = model.predict_proba(X_test[dept_mask][model_features])[:,1]
            median_idx = np.argsort(dept_probas)[len(dept_probas)//2]
            employee_data = X_test[dept_mask].iloc[median_idx]
            employee_shap = global_shap_array[dept_mask][median_idx]
            
            # Filter out other department features
            non_dept_features = [f for f in model_features if not f.startswith('Department_')]
            #Note: Finance department is represented when all other department features are 0
            if selected_dept != "Finance":
                filtered_features = non_dept_features + [dept_col]
                filtered_shap = np.array([employee_shap[model_features.index(f)] for f in filtered_features])
            else:
                # For Finance, we don't include any department features
                filtered_features = non_dept_features
                filtered_shap = np.array([employee_shap[model_features.index(f)] for f in filtered_features])
                
            # Create waterfall plot
            st.markdown(f"**{selected_dept} Department Feature Impact**")
            
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=filtered_shap,
                    base_values=explainer.expected_value,
                    data=None,
                    feature_names=filtered_features
                ),
                max_display=15,
                show=False
            )
            
            plt.title(f"{selected_dept} Department", pad=20)
            st.pyplot(fig)
    
        else:
            st.warning(f"Department feature {dept_col} not found in model")
        
        

    else:
        st.warning("Required data not available - make sure X_test is loaded and explainer is initialized")