# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
import openai
import logging
import numpy as np
from datetime import datetime
from supabase import create_client
import re
from styles import load_custom_css
from chart_utils import render_chart, rule_based_parse, generate_insights
from calc_utils import evaluate_calculation, generate_formula_from_prompt, detect_outliers, PREDEFINED_CALCULATIONS, calculate_statistics
from prompt_utils import generate_sample_prompts, generate_prompts_with_llm, prioritize_fields
from utils import classify_columns, load_data, save_dashboard, load_dashboards, save_annotation, load_annotations, delete_dashboard, update_dashboard, load_openai_key, generate_gpt_insight_with_fallback, generate_unique_id, parse_prompt, setup_logging, fetch_dashboard_charts
import streamlit as st
from urllib.parse import urlparse, parse_qs
import hashlib
import time
import json
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import io
import importlib.metadata
from ratelimit import limits, sleep_and_retry
from cryptography.fernet import Fernet
from typing import List, Dict, Optional, Any
import secrets


# Set up logging
logger = setup_logging()

def save_field_types(project_name, field_types):
    """Save field types to a JSON file in the project directory."""
    try:
        project_dir = f"projects/{project_name}"
        os.makedirs(project_dir, exist_ok=True)
        field_types_file = f"{project_dir}/field_types.json"
        with open(field_types_file, 'w') as f:
            json.dump(field_types, f)
        logger.info(f"Saved field types for project {project_name}: {field_types}")
    except Exception as e:
        logger.error(f"Failed to save field types for project {project_name}: {str(e)}")
        raise

def load_field_types(project_name):
    """Load field types from a JSON file in the project directory."""
    try:
        project_dir = f"projects/{project_name}"
        field_types_file = f"{project_dir}/field_types.json"
        if not os.path.exists(project_dir):
            logger.warning(f"Project directory does not exist: {project_dir}")
            return {}
        if not os.path.exists(field_types_file):
            logger.warning(f"Field types file does not exist: {field_types_file}")
            return {}
        with open(field_types_file, 'r') as f:
            field_types = json.load(f)
        logger.info(f"Loaded field types for project {project_name}: {field_types}")
        return field_types
    except Exception as e:
        logger.error(f"Failed to load field types for project {project_name}: {str(e)}")
        return {}

# Initialize Supabase (use your project URL and anon key)
supabase = create_client("https://fyyvfaqiohdxhnbdqoxu.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5eXZmYXFpb2hkeGhuYmRxb3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NTA2MTYsImV4cCI6MjA2MzEyNjYxNn0.-h6sm3bgPzxDjxlmPhi5LNzsbhMJiz8-0HX80U7FiZc")

def handle_auth_callback():
    query_params = parse_qs(urlparse(st.query_params.get("url", [""])[0]).query)
    token_hash = query_params.get("token_hash", [None])[0]
    auth_type = query_params.get("type", [None])[0]
    if token_hash and auth_type == "email":
        try:
            response = supabase.auth.verify_otp({"token_hash": token_hash, "type": "email"})
            st.session_state.user_id = response.user.id
            st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
            st.success("Email confirmed!")
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Verification failed: {e}")

if "auth/callback" in st.query_params.get("url", [""])[0]:
    handle_auth_callback()

def get_analytics_type(chart):
    if isinstance(chart, dict) and "prompt" in chart:
        chart_prompt = chart["prompt"].lower()
    elif isinstance(chart, str):
        chart_prompt = chart.lower()
    else:
        return "Other"

    if any(word in chart_prompt for word in ["sales", "revenue", "income"]):
        return "Sales"
    elif any(word in chart_prompt for word in ["customer", "user", "client"]):
        return "Customer"
    elif any(word in chart_prompt for word in ["product", "item", "sku"]):
        return "Product"
    return "Other"

def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        "chart_history": [],
        "field_types": {},
        "dataset": None,
        "current_project": None,
        "sidebar_collapsed": False,
        "sort_order": {},
        "insights_cache": {},  # Cache for insights
        "chart_cache": {},    # Cache for chart render results
        "sample_prompts": [],
        "used_sample_prompts": [],
        "sample_prompt_pool": [],
        "last_used_pool_index": 0,
        "onboarding_seen": False,
        "classified": False,
        "last_manual_prompt": None,
        "chart_dimensions": {},
        "refresh_dashboards": False,
        "dashboard_order": [],
        "data_loaded": False,
        "loading_progress": 0,
        "last_data_update": None,
        "dataset_hash": None  # Store dataset hash for cache invalidation
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "current_project" not in st.session_state:
        st.session_state.current_project = "my_project"
# Rate limit configuration (e.g., 10 saves per minute)
SAVE_CALLS = 10
SAVE_PERIOD = 60

# Initialize encryption
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

def validate_session_security(supabase) -> bool:
    """Validate session authenticity and security"""
    try:
        session = supabase.auth.get_session()
        if not session or not session.access_token:
            return False
        
        # Verify token hasn't expired
        user = supabase.auth.get_user()
        if not user or not user.user:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Session validation failed: {str(e)}")
        return False

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not isinstance(input_str, str):
        return ""
    sanitized = ''.join(c for c in input_str if c.isalnum() or c in ' -_.')
    return sanitized[:100]

def convert_timestamps_to_strings(obj):
    """Recursively convert pandas Timestamps to ISO format strings for JSON serialization"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_timestamps_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_strings(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
        return obj.isoformat()
    else:
        return obj



def generate_pdf_report(charts, title, output_filename):
    """
    Generate a PDF report from charts using LaTeX.
    Args:
        charts (list): List of tuples (prompt, chart_type, chart_data, insights, fig)
        title (str): Report title
        output_filename (str): Output PDF filename
    Returns:
        bytes: PDF file content
    """
    logger.debug(f"Generating PDF report: {title}")

    # LaTeX preamble
    latex_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{DejaVuSans}
\geometry{margin=1in}
\definecolor{darkbg}{HTML}{1F2A44}
\definecolor{lighttext}{HTML}{FFFFFF}
\pagecolor{darkbg}
\color{lighttext}
\begin{document}
    """

    # Add title
    latex_content += f"""
\\textbf{{\\Large {title}}}\\\\
\\vspace{{0.5cm}}
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\\\
\\vspace{{1cm}}
    """

    # Temporary directory for images
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (prompt, chart_type, chart_data, insights, fig) in enumerate(charts):
            # Save chart as PNG
            if fig is not None:
                image_path = os.path.join(tmpdir, f"chart_{idx}.png")
                try:
                    pio.write_image(fig, image_path, format="png")
                    logger.debug(f"Saved chart image: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to save chart image {idx}: {str(e)}")
                    continue

                # Add chart to LaTeX
                latex_content += f"""
\\section*{{Chart {idx + 1}: {prompt} ({chart_type})}}
\\includegraphics[width=\\textwidth]{{{image_path}}}
                """
            else:
                latex_content += f"""
\\section*{{Chart {idx + 1}: {prompt} ({chart_type})}}
                """

            # Add insights
            latex_content += r"""
\begin{itemize}
    """
            if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                for insight in insights:
                    latex_content += f"\\item {insight}\n"
            else:
                latex_content += "\\item No insights available.\n"
            latex_content += r"""
\end{itemize}
            """

            # Add basic statistics if available
            if not chart_data.empty and metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                stats = calculate_statistics(working_df, metric)
                if stats:
                    latex_content += r"""
\begin{tabular}{|l|c|}
\hline
\textbf{Statistic} & \textbf{Value} \\
\hline
Mean & """ + f"{stats['mean']:.2f}" + r""" \\
Median & """ + f"{stats['median']:.2f}" + r""" \\
Min & """ + f"{stats['min']:.2f}" + r""" \\
Max & """ + f"{stats['max']:.2f}" + r""" \\
\hline
\end{tabular}
                    """

            latex_content += r"""
\vspace{0.5cm}
            """

        # Close LaTeX document
        latex_content += r"""
\end{document}
        """

        # Write LaTeX file
        tex_path = os.path.join(tmpdir, "report.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        # Compile LaTeX to PDF
        try:
            subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True
            )
            #logger.info(f"PDF generated: {output_filename}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile LaTeX: {e.stderr}")
            st.error("Failed to generate PDF report. Please check logs.")
            return None

        # Read PDF
        pdf_path = os.path.join(tmpdir, "report.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            return pdf_content
        else:
            logger.error("PDF file not found after compilation")
            st.error("Failed to generate PDF report.")
            return None

def compute_dataset_hash(df):
    """Compute a hash of the DataFrame to detect changes."""
    if df is None:
        return None
    try:
        # Use columns and a sample of data to create a hash
        columns_str = ''.join(sorted(df.columns))
        sample_data = df.head(100).to_csv(index=False)  # Sample to avoid memory issues
        hash_input = columns_str + sample_data
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute dataset hash: {str(e)}")
        return None



# Load Custom CSS and Override
load_custom_css()
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    .stApp > div > div {
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: white !important;
        width: 320px !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        width: 100% !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #475569 !important;
    }
    [data-testid="stSidebar"] .stSelectbox, .stTextInput, .stExpander, .stInfo {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox>label, .stTextInput>label, .stExpander>label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stExpander div[role="button"] p {
        color: white !important;
    }
    [data-testid="stSidebar"] .stInfo {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stSidebar"] .stInfo div {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div {
        color: black !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div[role="option"] {
        color: black !important;
        background-color: white !important;
    }
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
        text-align: left;
    }
    .styled-table thead tr {
        background-color: #334155;
        color: white;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #334155;
    }
    [data-testid="stSidebar"] .saved-dashboard {
        color: black !important;
    }
    .saved-dashboard {
        color: black !important;
    }
    .sort-button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 0.9em !important;
    }
    .sort-button:hover {
        background-color: #475569 !important;
    }
    .main [data-testid="stExpander"] {
        background-color: #F5F7FA !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        cursor: pointer !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] p {
        color: white !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .main [data-testid="stExpander"] > div[role="button"]:hover {
        background-color: #475569 !important;
    }
    /* New styles for Saved Dashboards tab */
    .dashboard-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .reorder-button {
        background-color: #26A69A !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        border: none !important;
        cursor: pointer !important;
    }
    .reorder-button:hover {
        background-color: #2E7D32 !important;
    }
    .annotation-input {
        background-color: #ECEFF1 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid #B0BEC5 !important;
    }
</style>
""", unsafe_allow_html=True)

# In main.py
def load_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None


# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Session State Init
def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        "chart_history": [],
        "field_types": {},
        "dataset": None,
        "current_project": None,
        "sidebar_collapsed": False,
        "sort_order": {},
        "insights_cache": {},
        "sample_prompts": [],
        "used_sample_prompts": [],
        "sample_prompt_pool": [],
        "last_used_pool_index": 0,
        "onboarding_seen": False,
        "classified": False,
        "last_manual_prompt": None,
        "chart_dimensions": {},
        "chart_cache": {},
        "refresh_dashboards": False,
        "dashboard_order": [],
        "data_loaded": False,
        "loading_progress": 0,
        "last_data_update": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state at startup
initialize_session_state()





# Preprocess Dates (unchanged)
def preprocess_dates(df):
    """Forcefully preprocess date columns with format detection and consistent parsing."""
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%m-%d-%Y', '%d-%m-%Y', '%b %d %Y', '%B %d %Y',
        '%d %b %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'
    ]
    
    for col in df.columns:
        if 'date' in col.lower() or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                # First try to detect the format from a sample of non-null values
                sample = df[col].dropna().head(50)  # Increased sample size for better format detection
                if len(sample) > 0:
                    detected_format = None
                    valid_count = 0
                    best_format = None
                    
                    # Try each format and count valid dates
                    for fmt in date_formats:
                        try:
                            parsed = pd.to_datetime(sample, format=fmt)
                            valid = parsed.notna().sum()
                            if valid > valid_count:
                                valid_count = valid
                                best_format = fmt
                        except:
                            continue
                    
                    if best_format:
                        # Use the best detected format for parsing
                        parsed_col = pd.to_datetime(df[col], format=best_format, errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        #logger.info(f"Parsed column '{col}' as datetime using format '{best_format}' with {valid_ratio:.1%} valid dates")
                        
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            #logger.info(f"Converted column '{col}' to datetime")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                    else:
                        # If no format detected, try parsing without format
                        parsed_col = pd.to_datetime(df[col], errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            #logger.info(f"Converted column '{col}' to datetime using automatic format detection with {valid_ratio:.1%} valid dates")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                else:
                    logger.warning(f"Column '{col}' has no non-null values to detect date format")
            except Exception as e:
                logger.warning(f"Failed to parse date column '{col}': {str(e)}")
    return df

# Initialize session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = "my_project"
    os.makedirs(f"projects/{st.session_state.current_project}", exist_ok=True)
if 'chart_history' not in st.session_state:
    st.session_state.chart_history = []
if 'current_dashboard' not in st.session_state:
    st.session_state.current_dashboard = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'classified' not in st.session_state:
    st.session_state.classified = False
if 'field_types' not in st.session_state:
    st.session_state.field_types = {}
if 'sample_prompts' not in st.session_state:
    st.session_state.sample_prompts = []
if 'used_sample_prompts' not in st.session_state:
    st.session_state.used_sample_prompts = []
if 'sample_prompt_pool' not in st.session_state:
    st.session_state.sample_prompt_pool = []
if 'last_used_pool_index' not in st.session_state:
    st.session_state.last_used_pool_index = 0


# Sidebar
with st.sidebar:
    # Display logo at the top
    st.image("logo.png", width=300, use_container_width=False)

    # Custom CSS for yellow header and layout with updated button colors
    st.markdown("""
        <style>
        .sidebar .yellow-header {
            background-color: #FFFF00;
            padding: 10px;
            margin-bottom: 5px;
            text-align: center;
            font-weight: bold;
            color: black; /* Ensure header text is visible */
        }
        .sidebar .button-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .sidebar .input-field {
            width: 100%;
            margin-bottom: 5px;
        }
        .sidebar .login-button, .sidebar .signup-button {
            background-color: #3B82F6 !important; /* New blue color */
            color: white !important; /* Ensure button text is white */
            border-radius: 0.5rem !important;
            padding: 0.75rem !important;
            width: 100% !important;
            text-align: center !important;
        }
        .sidebar .login-button:hover, .sidebar .signup-button:hover {
            background-color: #2563EB !important; /* Darker blue on hover */
        }
        .sidebar .logged-in-text {
            color: black !important; /* Change Username text to black */
            margin-bottom: 5px;
        }
        .sidebar .project-item {
            padding: 5px 10px;
            margin-bottom: 2px;
            color: white; /* Ensure project items are visible */
        }
        .sidebar [data-testid="stMarkdownContainer"] strong {
            color: black !important; /* Change Login text to black */
        }
        .sidebar .yellow-header.projects {
            color: white !important; /* Change Projects text to white */
        }
        </style>
    """, unsafe_allow_html=True)

    # Login Section
    st.markdown('<div class="yellow-header">Login</div>', unsafe_allow_html=True)
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("email", key="login_email", help="Enter your email address")
            password = st.text_input("Password", type="password", key="login_password", help="Enter your password")
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("Login")
            with col2:
                signup_button = st.form_submit_button("Sign up")
            
            if login_button:
                try:
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user_id = response.user.id
                    st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                    supabase.auth.set_session(response.session.access_token, response.session.refresh_token)
                    st.session_state.access_token = response.session.access_token
                    st.session_state.refresh_token = response.session.refresh_token
                    session = supabase.auth.get_session()
                    current_user = supabase.auth.get_user()
                    logger.info(f"User logged in: {st.session_state.user_id}, Role: {st.session_state.user_role}")
                    st.success("Logged in!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Login failed: {str(e)}")
                    st.error(f"Login failed: {str(e)}")

            if signup_button:
                try:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                        "options": {"data": {"role": "Viewer"}}
                    })
                    st.session_state.user_id = response.user.id
                    st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                    logger.info(f"User signed up: {st.session_state.user_id}, Role: {st.session_state.user_role}")
                    st.success("Signed up! Check your email to confirm.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Sign-up failed: {e}")
                    st.error(f"Sign-up failed: {e}")
    else:
        st.write(f"\U0001F464 {st.session_state.user_id}")  # 👤
        if st.button("\U0001F6AA Logout", use_container_width=True):  # 🚪
            supabase.auth.sign_out()
            st.session_state.user_id = None
            st.session_state.user_role = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            logger.info("User logged out")
            st.success("Logged out!")
            st.rerun()

    # Projects section
    st.markdown('<div class="yellow-header projects">\U0001F4C2 Projects</div>', unsafe_allow_html=True)
    
    # Demo projects (always visible, read-only)
    with st.expander("\U0001F4CA Demo Projects", expanded=True):  # 📊
        demo_projects = {}
        demo_dirs = ["marketing_demo", "sales_demo"]
        
        for project_dir in demo_dirs:
            project_path = os.path.join("projects", project_dir)
            if os.path.exists(project_path):
                # Get all CSV files in the project directory
                csv_files = [f for f in os.listdir(project_path) if f.endswith('.csv')]
                if csv_files:
                    demo_projects[project_dir] = {
                        "name": project_dir.replace('_', ' ').title(),
                        "dashboards": [f.replace('.csv', '') for f in csv_files]
                    }
        
        for project_id, project_info in demo_projects.items():
            if st.button(f"\U0001F4CA {project_info['name']}", key=f"demo_{project_id}", use_container_width=True):  # 📊
                st.session_state.current_project = project_id
                st.session_state.is_demo_project = True
                # Load the dataset when project is selected
                try:
                    if os.path.exists(f"projects/{project_id}/dataset.csv"):
                        df = pd.read_csv(f"projects/{project_id}/dataset.csv")
                        df = preprocess_dates(df)
                        st.session_state.dataset = df
                        # Load saved field types
                        saved_field_types = load_field_types(project_id)
                        if saved_field_types:
                            st.session_state.field_types = saved_field_types
                            st.session_state.classified = True
                        else:
                            st.session_state.classified = False
                        st.session_state.sample_prompts = []
                        st.session_state.used_sample_prompts = []
                        st.session_state.sample_prompt_pool = []
                        st.session_state.last_used_pool_index = 0
                        st.success(f"Opened: {project_id}")
                except Exception as e:
                    st.error(f"Failed to load dataset for project {project_id}: {e}")
                st.rerun()        
        
        # List saved dashboards
        dashboards_dir = f"projects/{st.session_state.current_project}/dashboards"
        if os.path.exists(dashboards_dir):
            for dashboard_file in os.listdir(dashboards_dir):
                if dashboard_file.endswith('.json'):
                    try:
                        with open(os.path.join(dashboards_dir, dashboard_file), 'r') as f:
                            dashboard_data = json.load(f)
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                if st.button(f"📈 {dashboard_data['name']}", key=f"dash_{dashboard_file}", use_container_width=True):
                                    st.session_state.current_dashboard = dashboard_data['id']
                                    st.session_state.chart_history = dashboard_data['charts']
                                    st.rerun()
                            with col2:
                                if st.button("🗑️", key=f"delete_{dashboard_file}"):
                                    try:
                                        os.remove(os.path.join(dashboards_dir, dashboard_file))
                                        if st.session_state.current_dashboard == dashboard_data['id']:
                                            st.session_state.current_dashboard = None
                                            st.session_state.chart_history = []
                                        st.success(f"Deleted dashboard: {dashboard_data['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete dashboard: {e}")
                    except Exception as e:
                        st.error(f"Failed to load dashboard: {e}")
    
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **NarraViz AI** is an AI-powered business intelligence platform that transforms data into actionable insights using natural language. Ask questions, visualize data, and uncover trends effortlessly.
        """)


# Main Content
if st.session_state.current_project:
    st.caption(f"---")

    st.caption(f"Active Project: **{st.session_state.current_project}**")
    if st.session_state.get('is_demo_project'):
        st.info("You are viewing a demo project. Create your own analytics in My Project.")
else:
    pass


# Onboarding Modal (unchanged)
if not st.session_state.onboarding_seen:
    with st.container():
        st.markdown("""
        <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 0;'>
            <h2>Welcome to NarraVIZ AI! 🎉</h2>
            <p>Transform your data into insights with our AI-powered BI platform. Here's how to get started:</p>
            <ul>
                <li>📂 Create or open a project in the sidebar.</li>
                <li>📊 Upload a CSV or connect to a database.</li>
                <li>💬 Ask questions like "Top 5 Cities by Sales" in the prompt box.</li>
                <li>📈 Explore charts and AI-generated insights.</li>
            </ul>
            <p>Ready to dive in?</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Got it! Let's start.", key="onboarding_close"):
            st.session_state.onboarding_seen = True
            #logger.info("User completed onboarding")

# Save Dataset Changes (unchanged)
def save_dataset_changes():
    if st.session_state.current_project and st.session_state.dataset is not None:
        try:
            st.session_state.dataset.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
            # Also save field types
            save_field_types(st.session_state.current_project, st.session_state.field_types)
            #logger.info("Saved dataset and field types for project: %s", st.session_state.current_project)
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            logger.error("Failed to save dataset for project %s: %s", st.session_state.current_project, str(e))

def generate_pdf_summary(summary_points, overall_analysis):
    """
    Generate a PDF summary from summary points and overall analysis.
    Args:
        summary_points (list): List of summary points
        overall_analysis (list): List of overall analysis points
    Returns:
        bytes: PDF content as bytes
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Executive Summary Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Summary of Dashboard Analysis
        story.append(Paragraph("Summary of Dashboard Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in summary_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Overall Data Analysis
        story.append(PageBreak())
        story.append(Paragraph("Overall Data Analysis and Findings", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in overall_analysis:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        #logger.info("Generated PDF summary successfully")
        return pdf_content
    except Exception as e:
        logger.error(f"Failed to generate PDF summary: {str(e)}", exc_info=True)
        raise

# Generate GPT Insights (unchanged)
def generate_gpt_insights(stats, metric, prompt, chart_data, dimension=None, second_metric=None):
    """Generate insights using GPT-3.5-turbo."""
    if not USE_OPENAI:
        return []

    try:
        # Prepare the data summary
        data_summary = {
            "metric": metric,
            "dimension": dimension,
            "second_metric": second_metric,
            "stats": stats,
            "prompt": prompt,
            "data_points": len(chart_data)
        }

        # Create the prompt for GPT
        gpt_prompt = (
            f"Analyze this data visualization and provide 3 concise, insightful observations:\n"
            f"Metric: {data_summary['metric']}\n"
            f"Dimension: {data_summary['dimension']}\n"
            f"Statistics: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, "
            f"Min={stats['min']:.2f}, Max={stats['max']:.2f}\n"
            f"Number of data points: {data_summary['data_points']}\n"
            f"Original prompt: {prompt}\n"
            f"Provide 3 specific, data-driven insights that would be valuable for business users."
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise, actionable insights from data visualizations."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract insights from the response
        insights = [line.strip('- ').strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        return insights[:3]  # Return top 3 insights

    except Exception as e:
        logger.error(f"Failed to generate GPT insights: {str(e)}")
        return []
 

@sleep_and_retry
@limits(calls=SAVE_CALLS, period=SAVE_PERIOD)
def save_dashboard_secure(supabase, session_state, dashboard_name, project_id, tags=None):
    """Enterprise-grade dashboard saving with secure data embedding"""
    try:
        # Validate inputs
        if not dashboard_name or not str(dashboard_name).strip():
            raise ValueError("Dashboard name is required")
        if not project_id or not str(project_id).strip():
            raise ValueError("Project ID is required")
        if session_state.get("dataset") is None:
            raise ValueError("No dataset available to save with dashboard")
        if not session_state.get("user_id"):
            raise ValueError("User must be logged in to save dashboards")

        # Ensure string values
        dashboard_name = str(dashboard_name).strip()
        project_id = str(project_id).strip()

        # Create project directory
        project_dir = f"projects/{project_id}"
        os.makedirs(project_dir, exist_ok=True)

        # Get chart configurations
        chart_configs = []
        recommendations = session_state.get("recommendations", [])
        custom_charts = session_state.get("custom_charts", [])

        # Add active recommendation charts
        for idx, chart_tuple in enumerate(recommendations):
            if not session_state.get(f"delete_chart_{idx}", False):
                if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                    prompt, chart_type = chart_tuple[0], chart_tuple[1]
                    if prompt and isinstance(prompt, str) and prompt.strip():
                        chart_configs.append({
                            "prompt": prompt.strip(),
                            "chart_type": str(chart_type) if chart_type else "Bar"
                        })

        # Add active custom charts
        base_idx = len(recommendations)
        for custom_idx, chart_tuple in enumerate(custom_charts):
            idx = base_idx + custom_idx
            if not session_state.get(f"delete_chart_{idx}", False):
                if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                    prompt, chart_type = chart_tuple[0], chart_tuple[1]
                    if prompt and isinstance(prompt, str) and prompt.strip():
                        chart_configs.append({
                            "prompt": prompt.strip(),
                            "chart_type": str(chart_type) if chart_type else "Bar"
                        })

        if not chart_configs:
            raise ValueError("No valid charts available to save")

        df = session_state["dataset"]

        # Save dataset to project directory
        dataset_path = f"{project_dir}/dataset.csv"
        df.to_csv(dataset_path, index=False)
        logger.info(f"Saved dataset to {dataset_path}")

        # Create data snapshot with proper type conversion
        data_snapshot = {
            "version": "2.0",
            "saved_at": datetime.utcnow().isoformat(),
            "dataset_metadata": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "hash": compute_dataset_hash(df)
            }
        }

        # Prepare dataset for JSON serialization
        if len(df) <= 10000:
            df_serializable = df.copy()
            for col in df_serializable.columns:
                if pd.api.types.is_datetime64_any_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.api.types.is_categorical_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].astype(str)
                elif pd.api.types.is_integer_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].apply(lambda x: int(x) if pd.notna(x) else None)
                elif pd.api.types.is_float_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].apply(lambda x: float(x) if pd.notna(x) else None)
                elif pd.api.types.is_bool_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].astype(bool)
                else:
                    df_serializable[col] = df_serializable[col].astype(str).replace('', None)
            
            records = convert_timestamps_to_strings(df_serializable.to_dict('records'))
            data_snapshot["full_dataset"] = records
            data_snapshot["is_sample"] = False
            logger.info(f"Saved full dataset: {len(df)} rows")
        else:
            sample_df = df.sample(n=10000, random_state=42)
            sample_serializable = sample_df.copy()
            for col in sample_serializable.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.api.types.is_categorical_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].astype(str)
                elif pd.api.types.is_integer_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].apply(lambda x: int(x) if pd.notna(x) else None)
                elif pd.api.types.is_float_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].apply(lambda x: float(x) if pd.notna(x) else None)
                elif pd.api.types.is_bool_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].astype(bool)
                else:
                    sample_serializable[col] = sample_serializable[col].astype(str).replace('', None)
            
            records = convert_timestamps_to_strings(sample_serializable.to_dict('records'))
            data_snapshot["full_dataset"] = records
            data_snapshot["is_sample"] = True
            data_snapshot["original_size"] = len(df)
            data_snapshot["sample_strategy"] = "random"
            logger.info(f"Saved sample dataset: {len(sample_df)} of {len(df)} rows")

        # Save field classifications
        if "field_types" in session_state:
            data_snapshot["field_types"] = session_state["field_types"]
            save_field_types(project_id, session_state["field_types"])

        # Save current filters if any
        if "filters" in session_state and session_state["filters"]:
            data_snapshot["applied_filters"] = session_state["filters"]

        # Prepare charts with embedded data context
        enhanced_charts = []
        enhanced_charts.append({
            "prompt": "_data_snapshot_",
            "type": "metadata",
            "data_snapshot": data_snapshot,
            "created_at": datetime.utcnow().isoformat()
        })

        for chart_config in chart_configs:
            enhanced_chart = {
                "prompt": chart_config["prompt"],
                "chart_type": chart_config["chart_type"],
                "created_at": datetime.utcnow().isoformat(),
                "dataset_hash": data_snapshot["dataset_metadata"]["hash"]
            }
            enhanced_charts.append(enhanced_chart)

        # Prepare dashboard data
        dashboard_data = {
            "name": dashboard_name,
            "project_id": project_id,
            "owner_id": session_state.get("user_id"),
            "charts": enhanced_charts,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Conditionally include dataset_hash
        try:
            dashboard_data["dataset_hash"] = data_snapshot["dataset_metadata"]["hash"]
        except Exception as e:
            logger.warning(f"Excluding dataset_hash from insert due to schema mismatch: {str(e)}")
            dashboard_data.pop("dataset_hash", None)

        # Conditionally include version
        try:
            dashboard_data["version"] = "2.0"
        except Exception as e:
            logger.warning(f"Excluding version from insert due to schema mismatch: {str(e)}")
            dashboard_data.pop("version", None)

        # Save to database
        result = supabase.table("dashboards").insert(dashboard_data).execute()

        if result.data:
            dashboard_id = result.data[0]["id"]
            logger.info(f"Dashboard saved successfully: {dashboard_id} with {len(enhanced_charts)-1} charts")
            return dashboard_id
        else:
            raise Exception("Failed to save dashboard to database")

    except Exception as e:
        logger.error(f"Failed to save dashboard: {str(e)}")
        raise


def render_save_dashboard_section(supabase):
    """Save dashboard section with user-defined name and overwrite prompt"""
    
    # Check if user is logged in
    if "user_id" not in st.session_state or st.session_state.user_id is None:
        st.info("💡 Please log in to save dashboards.")
        return
    
    # Check if dataset exists
    if st.session_state.get("dataset") is None:
        st.info("💡 Please upload a dataset first before saving a dashboard.")
        return
    
    # Get charts from your actual data structure
    recommendations = st.session_state.get("recommendations", [])
    custom_charts = st.session_state.get("custom_charts", [])
    
    # Filter out deleted charts and validate prompts
    active_recommendations = []
    for idx, chart_tuple in enumerate(recommendations):
        if not st.session_state.get(f"delete_chart_{idx}", False):
            if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                prompt, chart_type = chart_tuple[0], chart_tuple[1]
                if prompt and isinstance(prompt, str) and prompt.strip():
                    active_recommendations.append((prompt, chart_type))
    
    active_custom_charts = []
    base_idx = len(recommendations)
    for custom_idx, chart_tuple in enumerate(custom_charts):
        idx = base_idx + custom_idx
        if not st.session_state.get(f"delete_chart_{idx}", False):
            if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                prompt, chart_type = chart_tuple[0], chart_tuple[1]
                if prompt and isinstance(prompt, str) and prompt.strip():
                    active_custom_charts.append((prompt, chart_type))
    
    # Total active charts
    total_charts = active_recommendations + active_custom_charts
    
    if not total_charts:
        st.info("💡 Create some charts first before saving a dashboard.")
        return
    
    with st.expander("💾 **Save Dashboard**", expanded=False):
        st.markdown(f"Save your current {len(total_charts)} charts and data as a dashboard.")
        
        with st.form("save_dashboard_form"):
            dashboard_name = st.text_input(
                "Dashboard Name",
                value="",
                placeholder="Enter a dashboard name",
                key="dashboard_name_input"
            )
            
            tags_input = st.text_input(
                "Tags (comma-separated)",
                placeholder="sales, quarterly, analysis",
                key="tags_input"
            )
            
            # Show what will be saved
            st.markdown("### 📊 Charts to Save:")
            for i, (prompt, chart_type) in enumerate(total_charts[:3]):  # Show first 3
                st.markdown(f"{i+1}. **{prompt[:50]}...** ({chart_type or 'Bar'})")
            if len(total_charts) > 3:
                st.markdown(f"*...and {len(total_charts) - 3} more charts*")
            
            # Check for existing dashboard
            existing_dashboard = None
            if dashboard_name and dashboard_name.strip():
                try:
                    result = supabase.table("dashboards").select("id, name").eq("name", dashboard_name.strip()).eq("owner_id", st.session_state.user_id).execute()
                    if result.data:
                        existing_dashboard = result.data[0]
                        st.session_state.existing_dashboard = existing_dashboard
                    else:
                        st.session_state.existing_dashboard = None
                except Exception as e:
                    logger.error(f"Failed to check for existing dashboard: {str(e)}")
                    st.error(f"Failed to check for existing dashboard: {str(e)}")
                    return
            
            # Handle duplicate dashboard name
            overwrite_choice = None
            new_dashboard_name = None
            if existing_dashboard:
                st.warning(f"A dashboard named '{dashboard_name}' already exists.")
                overwrite_choice = st.radio(
                    "Choose an action:",
                    ["Overwrite existing dashboard", "Save as a new dashboard"],
                    key="overwrite_choice"
                )
                if overwrite_choice == "Save as a new dashboard":
                    new_dashboard_name = st.text_input(
                        "New Dashboard Name",
                        value="",
                        placeholder="Enter a unique dashboard name",
                        key="new_dashboard_name_input"
                    )
                    if new_dashboard_name and new_dashboard_name.strip():
                        # Check if new name also exists
                        try:
                            result = supabase.table("dashboards").select("id").eq("name", new_dashboard_name.strip()).eq("owner_id", st.session_state.user_id).execute()
                            if result.data:
                                st.error(f"A dashboard named '{new_dashboard_name}' already exists. Please choose a different name.")
                                return
                        except Exception as e:
                            logger.error(f"Failed to check for new dashboard name: {str(e)}")
                            st.error(f"Failed to check for new dashboard name: {str(e)}")
                            return
            
            if st.form_submit_button("💾 Save Dashboard", type="primary"):
                # Validate inputs
                final_dashboard_name = None
                if overwrite_choice == "Save as a new dashboard" and new_dashboard_name and new_dashboard_name.strip():
                    final_dashboard_name = new_dashboard_name.strip()
                elif dashboard_name and dashboard_name.strip():
                    final_dashboard_name = dashboard_name.strip()
                else:
                    st.error("Please enter a dashboard name")
                    return
                
                try:
                    # Auto-generate project ID (hidden from user)
                    user_id = st.session_state.get("user_id", "anonymous")
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    auto_project_id = "My Project"

#f"{user_id}_project_{timestamp}"
                    
                    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
                    
                    if existing_dashboard and overwrite_choice == "Overwrite existing dashboard":
                        # Update existing dashboard
                        dashboard_data = {
                            "name": final_dashboard_name,
                            "project_id": auto_project_id,
                            "owner_id": st.session_state.get("user_id"),
                            "charts": [{
                                "prompt": "_data_snapshot_",
                                "type": "metadata",
                                "data_snapshot": {
                                    "version": "2.0",
                                    "saved_at": datetime.utcnow().isoformat(),
                                    "dataset_metadata": {
                                        "shape": list(st.session_state.dataset.shape),
                                        "columns": st.session_state.dataset.columns.tolist(),
                                        "dtypes": {col: str(dtype) for col, dtype in st.session_state.dataset.dtypes.items()},
                                        "memory_usage": int(st.session_state.dataset.memory_usage(deep=True).sum()),
                                        "hash": compute_dataset_hash(st.session_state.dataset)
                                    },
                                    "full_dataset": convert_timestamps_to_strings(st.session_state.dataset.to_dict('records')),
                                    "is_sample": False,
                                    "field_types": st.session_state.get("field_types", {}),
                                    "applied_filters": st.session_state.get("filters", {})
                                },
                                "created_at": datetime.utcnow().isoformat()
                            }] + [
                                {
                                    "prompt": prompt,
                                    "chart_type": chart_type,
                                    "created_at": datetime.utcnow().isoformat(),
                                    "dataset_hash": compute_dataset_hash(st.session_state.dataset)
                                } for prompt, chart_type in total_charts
                            ],
                            "tags": tags,
                            "updated_at": datetime.utcnow().isoformat(),
                            "dataset_hash": compute_dataset_hash(st.session_state.dataset),
                            "version": "2.0"
                        }
                        result = supabase.table("dashboards").update(dashboard_data).eq("id", existing_dashboard["id"]).execute()
                        if result.data:
                            dashboard_id = result.data[0]["id"]
                            st.success(f"✅ Dashboard '{final_dashboard_name}' overwritten successfully!")
                            st.info(f"Dashboard ID: `{dashboard_id}`")
                            st.session_state.refresh_dashboards = True
                            st.session_state.existing_dashboard = None
                        else:
                            raise Exception("Failed to update dashboard")
                    else:
                        # Save new dashboard
                        dashboard_id = save_dashboard_secure(
                            supabase,
                            st.session_state,
                            final_dashboard_name,
                            auto_project_id,
                            tags
                        )
                        st.success(f"✅ Dashboard '{final_dashboard_name}' saved successfully!")
                        st.info(f"Dashboard ID: `{dashboard_id}`")
                        st.session_state.refresh_dashboards = True
                        st.session_state.existing_dashboard = None
                    
                except Exception as e:
                    st.error(f"❌ Failed to save dashboard: {str(e)}")
                    logger.error(f"Failed to save dashboard: {str(e)}")


# Fixed chart rendering with better error handling
def safe_render_chart(idx, prompt, dimensions, measures, dates, working_df, sort_order="Descending", chart_type=None):
    """Safely render chart with comprehensive error handling"""
    try:
        # Validate inputs before proceeding
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.warning(f"Invalid prompt for chart {idx}: {prompt}")
            return None
            
        if not dimensions and not measures:
            logger.warning(f"No dimensions or measures available for chart {idx}")
            return None
            
        # Call the original render_chart function
        from chart_utils import render_chart
        return render_chart(idx, prompt, dimensions, measures, dates, working_df, sort_order, chart_type)
        
    except Exception as e:
        logger.error(f"Safe render chart failed for prompt '{prompt}': {str(e)}")
        return None


def generate_chart_from_prompt(prompt, dimensions, measures, dates, dataset, chart_type):
    """Generate chart data and plotly figure from prompt"""
    try:
        # Use your existing render_chart function
        from chart_utils import render_chart
        
        # Get a unique index for this chart
        chart_idx = 0  # You might want to make this dynamic
        
        result = render_chart(
            chart_idx, prompt, dimensions, measures, dates, dataset, 
            sort_order="Descending", chart_type=chart_type
        )
        
        if result is None:
            return None, None
        
        chart_data, metric, dimension, working_df, table_columns, chart_type_used, secondary_dimension, kwargs = result
        
        # Create plotly figure based on chart type
        import plotly.express as px
        
        if chart_type_used == "Bar":
            fig = px.bar(chart_data, x=dimension, y=metric, template="plotly_dark")
        elif chart_type_used == "Line":
            fig = px.line(chart_data, x=dimension, y=metric, color=secondary_dimension, template="plotly_dark")
        elif chart_type_used == "Scatter":
            y_metric = table_columns[2] if len(table_columns) > 2 else metric
            fig = px.scatter(chart_data, x=metric, y=y_metric, color=dimension, template="plotly_dark")
        elif chart_type_used == "Pie":
            fig = px.pie(chart_data, names=dimension, values=metric, template="plotly_dark")
        elif chart_type_used == "Map":
            fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric, template="plotly_dark")
        else:
            fig = None  # For table view
        
        return chart_data, fig
        
    except Exception as e:
        logger.error(f"Failed to generate chart from prompt: {str(e)}")
        return None, None


def load_dashboard_secure(supabase, dashboard_id):
    """Enterprise-grade dashboard loading with data restoration"""
    try:
        # Fetch dashboard
        result = supabase.table("dashboards").select("*").eq("id", dashboard_id).execute()
        
        if not result.data:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = result.data[0]
        charts = dashboard.get("charts", [])
        
        # Find data snapshot
        data_snapshot = None
        chart_configs = []
        
        for chart in charts:
            if chart.get("prompt") == "_data_snapshot_":
                data_snapshot = chart.get("data_snapshot", {})
            else:
                chart_configs.append(chart)
        
        if not data_snapshot:
            raise ValueError("No data snapshot found in dashboard")
        
        # Restore dataset
        dataset = restore_dataset_from_snapshot(data_snapshot)
        
        if dataset is None:
            raise ValueError("Failed to restore dataset from snapshot")
        
        # Restore field types
        field_types = data_snapshot.get("field_types", {})
        
        # Validate data integrity
        expected_hash = data_snapshot.get("dataset_metadata", {}).get("hash")
        if expected_hash:
            current_hash = compute_dataset_hash(dataset)
            if current_hash != expected_hash:
                logger.warning(f"Dataset hash mismatch for dashboard {dashboard_id} - data may have been altered")
        
        return {
            "dashboard": dashboard,
            "dataset": dataset,
            "field_types": field_types,
            "chart_configs": chart_configs,
            "data_snapshot": data_snapshot
        }
        
    except Exception as e:
        logger.error(f"Failed to load dashboard: {str(e)}")
        raise


def restore_dataset_from_snapshot(data_snapshot):
    """Restore dataset with proper type handling"""
    try:
        if not data_snapshot or "full_dataset" not in data_snapshot:
            logger.error("No full_dataset found in data_snapshot")
            return None
        
        records = data_snapshot["full_dataset"]
        if not records:
            logger.error("Empty dataset in data_snapshot")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Restore data types
        if "dataset_metadata" in data_snapshot:
            dtypes = data_snapshot["dataset_metadata"].get("dtypes", {})
            
            for col in df.columns:
                if col in dtypes:
                    dtype_str = dtypes[col]
                    try:
                        if "datetime64" in dtype_str or "timestamp" in dtype_str.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif "int" in dtype_str.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if "Int" in dtype_str:
                                df[col] = df[col].astype('Int64')
                            else:
                                df[col] = df[col].astype('int64', errors='ignore')
                        elif "float" in dtype_str.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                        elif "bool" in dtype_str.lower():
                            df[col] = df[col].astype('bool', errors='ignore')
                        elif "category" in dtype_str.lower():
                            df[col] = df[col].astype('category')
                        else:
                            df[col] = df[col].astype('object')
                    except Exception as e:
                        logger.warning(f"Failed to restore type for {col}: {e}")
                        df[col] = df[col].astype('object')
        
        # Handle empty strings and nulls
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', np.nan)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Restored dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Dataset restoration failed: {str(e)}")
        return None


def render_dashboard_enterprise(supabase, dashboard_id, project_id, dashboard_name):
    """Enterprise dashboard renderer with isolation"""
    
    # Store current session state
    original_dataset = st.session_state.get("dataset")
    original_field_types = st.session_state.get("field_types") 
    original_project = st.session_state.get("current_project")
    
    try:
        with st.spinner("ð    Loading dashboard..."):
            # Load dashboard with its own data
            dashboard_data = load_dashboard_secure(supabase, dashboard_id)
            
            dataset = dashboard_data["dataset"]
            field_types = dashboard_data["field_types"]
            chart_configs = dashboard_data["chart_configs"]
            data_snapshot = dashboard_data["data_snapshot"]
            
            # Temporarily set session state for rendering
            st.session_state.dataset = dataset
            st.session_state.field_types = field_types
            st.session_state.current_project = project_id
            
        # Show data status
        if data_snapshot.get("is_sample"):
            st.markdown(f"""
            <div class="dataset-status dataset-sample">
                ð    Dashboard dataset: {len(dataset)} rows (sample of {data_snapshot.get('original_size', 'unknown')})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="dataset-status dataset-restored">
                â   Dashboard dataset: {len(dataset)} rows, {len(dataset.columns)} columns
            </div>
            """, unsafe_allow_html=True)
        
        if not chart_configs:
            st.warning("No charts found in this dashboard.")
            return
        
        # Extract field types
        dimensions = field_types.get("dimension", [])
        measures = field_types.get("measure", [])
        dates = field_types.get("date", [])
        
        # Render each chart
        for idx, chart in enumerate(chart_configs):
            prompt = chart.get("prompt")
            chart_type = chart.get("chart_type")
            
            with st.expander(f"Chart {idx + 1}: {prompt} ({chart_type})", expanded=True):
                try:
                    chart_data, fig = generate_chart_from_prompt(
                        prompt, dimensions, measures, dates, dataset, chart_type
                    )
                    if chart_data is not None and not chart_data.empty:
                        if fig:
                            st.plotly_chart( fig, use_container_width=True, key=f"chart_{idx}")
                        st.dataframe(chart_data, use_container_width=True)
                        
                        # Generate insights
                        if pd.api.types.is_numeric_dtype(chart_data[measures[0]]):
                            stats = calculate_statistics(chart_data, measures[0])
                            insights = generate_gpt_insights(
                                stats, measures[0], prompt, chart_data, dimensions[0] if dimensions else None
                            )
                            if insights:
                                st.markdown("**Insights:**")
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                            else:
                                st.info("No significant insights could be generated.")
                        else:
                            st.info("Insights unavailable for non-numeric data.")
                    else:
                        st.warning("Unable to render chart: No valid data.")
                except Exception as e:
                    st.error(f"Failed to render chart: {str(e)}")
                    logger.error(f"Chart rendering failed for {prompt}: {str(e)}")
        
    finally:
        # Restore original session state
        st.session_state.dataset = original_dataset
        st.session_state.field_types = original_field_types
        st.session_state.current_project = original_project


# ISOLATED CHART DISPLAY
def display_chart_isolated(idx, prompt, dimensions, measures, dates, dataset, chart_type="Bar"):
    """Display chart using isolated dataset without affecting session state"""
    try:
        # This function should work with the provided dataset only
        # Don't use any session state data
        
        # Generate chart using provided data
        chart_data, chart_obj = generate_chart_from_prompt(
            prompt, dimensions, measures, dates, dataset, chart_type
        )
        
        if chart_obj:
            st.plotly_chart(chart_obj, use_container_width=True)
        else:
            st.error("Could not generate chart")
            
    except Exception as e:
        st.error(f"Chart generation failed: {str(e)}")
        logger.error(f"Isolated chart display error: {str(e)}")

def generate_overall_data_analysis(df, dimensions, measures, dates):
    if not USE_OPENAI:
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

    try:
        stats_summary = []
        for metric in measures:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                stats = calculate_statistics(df, metric)
                stats_summary.append(
                    f"{metric}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}, "
                    f"Q1={stats['q1']:.2f}, median={stats['median']:.2f}, Q3={stats['q3']:.2f}, "
                    f"90th percentile={stats['percentile_90']:.2f}"
                )
        
        top_performers = []
        for dim in dimensions:
            for metric in measures:
                if dim in df.columns and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                    grouped = df.groupby(dim)[metric].mean().sort_values(ascending=False)
                    if not grouped.empty:
                        top = grouped.index[0]
                        top_value = grouped.iloc[0]
                        top_performers.append(f"Top {dim} by {metric}: {top} with average {top_value:.2f}")

        correlations = []
        for i, m1 in enumerate(measures):
            for m2 in measures[i+1:]:
                if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                    corr = df[[m1, m2]].corr().iloc[0, 1]
                    correlations.append(f"Correlation between {m1} and {m2}: {corr:.2f}")

        data_summary = (
            f"Dataset Overview:\n- Dimensions: {', '.join(dimensions)}\n- Measures: {', '.join(measures)}\n- Dates: {', '.join(dates) if dates else 'None'}\n"
            f"Statistics:\n" + "\n".join(stats_summary) + "\n"
            f"Top Performers:\n" + "\n".join(top_performers) + "\n"
            f"Correlations:\n" + "\n".join(correlations)
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing an overall analysis and findings summary for a dataset. Focus on key trends, significant findings, and actionable recommendations."},
                {"role": "user", "content": f"Generate a concise overall data analysis and findings summary (3-5 points) based on the following dataset summary:\n{data_summary}\nHighlight key trends, significant findings, and provide actionable recommendations for business strategy."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        analysis = response.choices[0].message.content.strip().split('\n')
        analysis = [item.strip('- ').strip() for item in analysis if item.strip()]
        #logger.info("Generated overall data analysis: %s", analysis)
        return analysis
    except Exception as e:
        logger.error


# Display Chart (rewritten)
def display_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """Display a chart with controls and data table, using cache."""
    try:
        # Create a unique cache key
        dataset_hash = st.session_state.get("dataset_hash", "no_hash")
        chart_key = f"chart_{idx}_{hash(prompt + str(chart_type) + sort_order + dataset_hash)}"
        
        # Initialize chart type in session state if not exists
        if f"chart_type_{chart_key}" not in st.session_state:
            st.session_state[f"chart_type_{chart_key}"] = chart_type or "Bar"
        
        # Create two columns for controls
        col1, col2 = st.columns(2)
        
        # Chart type selection
        with col1:
            selected_chart_type = st.selectbox(
                "Chart Type",
                options=["Bar", "Line", "Scatter", "Map", "Table", "Pie"],
                index=["Bar", "Line", "Scatter", "Map", "Table", "Pie"].index(st.session_state[f"chart_type_{chart_key}"]),
                key=f"chart_type_select_{chart_key}"
            )
            if selected_chart_type != st.session_state[f"chart_type_{chart_key}"]:
                st.session_state[f"chart_type_{chart_key}"] = selected_chart_type
                # Clear cache for this chart if type changes
                if chart_key in st.session_state.chart_cache:
                    del st.session_state.chart_cache[chart_key]
                if chart_key in st.session_state.insights_cache:
                    del st.session_state.insights_cache[chart_key]
                st.rerun()
        
        # Sort order selection
        with col2:
            selected_sort_order = st.selectbox(
                "Sort Order",
                options=["Ascending", "Descending"],
                index=1 if sort_order == "Descending" else 0,
                key=f"sort_order_{chart_key}"
            )
            if selected_sort_order != sort_order:
                # Clear cache if sort order changes
                if chart_key in st.session_state.chart_cache:
                    del st.session_state.chart_cache[chart_key]
                if chart_key in st.session_state.insights_cache:
                    del st.session_state.insights_cache[chart_key]
                sort_order = selected_sort_order
        
        # Check chart cache
        if chart_key in st.session_state.chart_cache:
            #logger.info(f"Using cached chart for key: {chart_key}")
            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
        else:
            # Render chart if not cached
            try:
                chart_result = render_chart(
                    idx, prompt, dimensions, measures, dates, df, sort_order, st.session_state[f"chart_type_{chart_key}"]
                )
                if chart_result is None:
                    raise ValueError("Chart rendering returned None")
                chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                # Store in cache
                st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs)
                #logger.info(f"Cached chart for key: {chart_key}")
            except ValueError as e:
                st.error(str(e))
                logger.error(f"Chart rendering failed: {str(e)}")
                return
        
        # Create the chart based on type
        if st.session_state[f"chart_type_{chart_key}"] == "Scatter":
            fig = px.scatter(
                chart_data,
                x=metric,
                y=table_columns[2],  # Use the second metric for y-axis
                color=dimension,
                hover_data=[dimension],
                title=f"{table_columns[2]} vs {metric} by {dimension}",
                labels={metric: metric, table_columns[2]: table_columns[2]},
                template="plotly_dark"
            )
            fig.update_traces(marker=dict(size=12))
            fig.update_layout(
                xaxis_title=metric,
                yaxis_title=table_columns[2],
                showlegend=True
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_scatter")
        elif st.session_state[f"chart_type_{chart_key}"] == "Bar":
            color_col = "Outlier" if "Outlier" in chart_data.columns else None
            fig = px.bar(
                chart_data,
                x=dimension,
                y=metric,
                color=color_col,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_bar")
        elif st.session_state[f"chart_type_{chart_key}"] == "Line":
            time_agg = kwargs.get("time_aggregation", "month")
            title = f"{metric} by {time_agg.capitalize()}"
            if secondary_dimension:
                title += f" and {secondary_dimension}"
            
            fig = px.line(
                chart_data,
                x=dimension,
                y=metric,
                color=secondary_dimension if secondary_dimension else None,
                title=title,
                template="plotly_dark"
            )
            if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                fig.update_xaxes(
                    tickformat="%b-%Y",
                    tickangle=45,
                    nticks=10
                )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_line")
        elif st.session_state[f"chart_type_{chart_key}"] == "Map":
            fig = px.choropleth(
                chart_data,
                locations=dimension,
                locationmode="country names",
                color=metric,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_map")
        elif st.session_state[f"chart_type_{chart_key}"] == "Pie":
            fig = px.pie(
                chart_data,
                names=dimension,
                values=metric,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_pie")
        else:  # Table view
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table")
        
        # Check insights cache
        if chart_key in st.session_state.insights_cache:
            #logger.info(f"Using cached insights for key: {chart_key}")
            insights = st.session_state.insights_cache[chart_key]
        else:
            # Generate insights if not cached
            try:
                insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                st.session_state.insights_cache[chart_key] = insights
                #logger.info(f"Cached insights for key: {chart_key}")
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
                insights = ["Unable to generate insights at this time."]
        
        # Display insights
        st.markdown("### Insights")
        for insight in insights:
            st.markdown(f"🔹 {insight}")
        
        # Display data table in a collapsed expander
        with st.expander("View Data", expanded=False):
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table_data")
            
            if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                st.markdown("### Basic Statistics")
                stats = calculate_statistics(working_df, metric)
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['median']:.2f}")
                    with col3:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{stats['max']:.2f}")
        
    except Exception as e:
        logger.error(f"Error displaying chart: {str(e)}")
        st.error(f"Error displaying chart: {str(e)}")

def truncate_label(label, max_length=20):
    if isinstance(label, str) and len(label) > max_length:
        return label[:max_length-3] + "..."
    return label


def compute_filter_hash(filters):
    """Compute a hash of the filters dictionary for cache key."""
    filter_str = json.dumps(filters, sort_keys=True, default=str)
    return hashlib.md5(filter_str.encode()).hexdigest()

def recommended_charts_insights_tab():
    st.subheader("📊 Recommended Charts & Insights")


    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    # Initialize session state for filters
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "filter_search" not in st.session_state:
        st.session_state.filter_search = {}
    if "filter_show_more" not in st.session_state:
        st.session_state.filter_show_more = {}

    dimensions = st.session_state.field_types.get("dimension", [])
    measures = st.session_state.field_types.get("measure", [])
    dates = st.session_state.field_types.get("date", [])

    # Filter Dataset Section
    st.markdown("### Filter Dataset")
    with st.expander("Apply Filters", expanded=False):
        filter_changed = False
        temp_filters = st.session_state.filters.copy()
        temp_search = st.session_state.filter_search.copy()
        temp_show_more = st.session_state.filter_show_more.copy()

        # Dimension Filters (Search + Limited Multiselect)
        for dim in dimensions:
            if dim in df.columns:
                # Get value counts for top values
                value_counts = df[dim].value_counts().head(100)  # Limit to top 100 for performance
                unique_vals = value_counts.index.tolist()
                if len(unique_vals) > 0:
                    # Initialize show_more count
                    if dim not in temp_show_more:
                        temp_show_more[dim] = 10  # Default to showing 10 values
                    max_display = temp_show_more[dim]

                    # Search input
                    search_key = f"filter_search_{dim}"
                    search_term = st.text_input(
                        f"Search {dim}",
                        value=temp_search.get(dim, ""),
                        key=search_key
                    )
                    if search_term != temp_search.get(dim):
                        temp_search[dim] = search_term
                        temp_show_more[dim] = 10  # Reset show_more on search
                        filter_changed = True

                    # Filter values based on search
                    display_vals = [v for v in unique_vals if not search_term or str(search_term).lower() in str(v).lower()]
                    display_vals = display_vals[:max_display]

                    # Multiselect for filtered values
                    default_vals = temp_filters.get(dim, [])
                    selected_vals = st.multiselect(
                        f"Select {dim} (showing {len(display_vals)} of {len(unique_vals)})",
                        options=display_vals,
                        default=[v for v in default_vals if v in display_vals],
                        key=f"filter_dim_{dim}"
                    )
                    if selected_vals != temp_filters.get(dim):
                        temp_filters[dim] = selected_vals
                        filter_changed = True

                    # Show More button
                    if len(display_vals) < len(unique_vals):
                        if st.button(f"Show More ({min(10, len(unique_vals) - max_display)})", key=f"show_more_{dim}"):
                            temp_show_more[dim] += 10
                            filter_changed = True

        # Measure Filters (Range Slider + Input Boxes)
        for measure in measures:
            if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                min_val = float(df[measure].min())
                max_val = float(df[measure].max())
                if min_val != max_val:
                    default_range = temp_filters.get(measure, [min_val, max_val])
                    col1, col2 = st.columns(2)
                    with col1:
                        min_input = st.number_input(
                            f"Min {measure}",
                            min_value=min_val,
                            max_value=max_val,
                            value=max(min_val, default_range[0]),
                            key=f"min_input_{measure}"
                        )
                    with col2:
                        max_input = st.number_input(
                            f"Max {measure}",
                            min_value=min_val,
                            max_value=max_val,
                            value=min(max_val, default_range[1]),
                            key=f"max_input_{measure}"
                        )
                    selected_range = st.slider(
                        f"Range for {measure}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_input, max_input),
                        key=f"filter_measure_{measure}"
                    )
                    if selected_range != temp_filters.get(measure):
                        temp_filters[measure] = list(selected_range)
                        filter_changed = True

        # Date Filters (Date Range)
        for date_col in dates:
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                min_date = df[date_col].min().to_pydatetime()
                max_date = df[date_col].max().to_pydatetime()
                default_range = temp_filters.get(date_col, [min_date, max_date])
                selected_range = st.date_input(
                    f"Date Range for {date_col}",
                    value=(default_range[0], default_range[1]),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"filter_date_{date_col}"
                )
                if len(selected_range) == 2 and selected_range != tuple(temp_filters.get(date_col, [])):
                    temp_filters[date_col] = [pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])]
                    filter_changed = True

        # Apply and Clear Filter Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Filters", key="apply_filters"):
                st.session_state.filters = temp_filters
                st.session_state.filter_search = temp_search
                st.session_state.filter_show_more = temp_show_more
                st.session_state.chart_cache = {}  # Clear cache
                st.session_state.insights_cache = {}  # Clear insights cache
                filter_changed = True
                st.success("Filters applied!")
        with col2:
            if st.button("Clear Filters", key="clear_filters"):
                st.session_state.filters = {}
                st.session_state.filter_search = {}
                st.session_state.filter_show_more = {}
                st.session_state.chart_cache = {}  # Clear cache
                st.session_state.insights_cache = {}  # Clear insights cache
                filter_changed = True
                st.success("Filters cleared!")

    # Apply filters to dataset
    working_df = df.copy()
    if st.session_state.filters:
        for col, filter_val in st.session_state.filters.items():
            if col in df.columns:
                if col in dimensions and filter_val:
                    working_df = working_df[working_df[col].isin(filter_val)]
                elif col in measures and len(filter_val) == 2:
                    working_df = working_df[
                        (working_df[col] >= filter_val[0]) & (working_df[col] <= filter_val[1])
                    ]
                elif col in dates and len(filter_val) == 2:
                    working_df = working_df[
                        (working_df[col] >= filter_val[0]) & (working_df[col] <= filter_val[1])
                    ]
        if working_df.empty:
            st.warning("Applied filters resulted in an empty dataset. Please adjust filters.")
            return
        logger.info(f"Applied filters reduced dataset from {len(df)} to {len(working_df)} rows")

    # Sample large datasets for performance
    MAX_PREVIEW_ROWS = 1000
    working_df = working_df.head(MAX_PREVIEW_ROWS) if len(working_df) > MAX_PREVIEW_ROWS else working_df

    # Initialize session state for charts
    if "custom_charts" not in st.session_state:
        st.session_state.custom_charts = []
    if "random_dimensions" not in st.session_state:
        st.session_state.random_dimensions = {}

    # Generate recommendations based on current field types
    recommendations = []

    # Use diverse combinations of measures and dimensions
    used_dims = set()
    used_measures = set()

    if dates and measures:
        recommendations.append((f"{measures[0]} by {dates[0]}", "Line"))
        used_measures.add(measures[0])

    if dates and measures and dimensions:
        recommendations.append((f"{measures[0]} by {dates[0]} and {dimensions[0]}", "Line"))
        used_measures.add(measures[0])
        used_dims.add(dimensions[0])

    if dimensions and measures:
        for i, dim in enumerate(dimensions):
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                m = measures[i % len(measures)]
                recommendations.append((f"Top 5 {dim} by {m}", "Bar"))
                st.session_state.random_dimensions['table'] = dim
                recommendations.append((f"{m} by {dim}", "Table"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    # Only recommend pie if dimension has 6 or fewer categories
    if dimensions and measures:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals <= 6 and dim not in used_dims:
                m = measures[len(used_measures) % len(measures)]
                recommendations.append((f"{m} by {dim}", "Pie"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    if len(measures) >= 2 and dimensions:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                st.session_state.random_dimensions['scatter'] = dim
                recommendations.append((f"{measures[0]} vs {measures[1]} by {dim}", "Scatter"))
                used_dims.add(dim)
                used_measures.update([measures[0], measures[1]])
                break

    if "Country" in dimensions and measures:
        m = measures[len(used_measures) % len(measures)]
        recommendations.append((f"{m} by Country", "Map"))
        used_measures.add(m)

    if dimensions and measures:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                st.session_state.random_dimensions['bubble_cloud'] = dim
                m = measures[len(used_measures) % len(measures)]
                recommendations.append((f"Bubble cloud of {dim} sized by {m}", "BubbleCloud"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    # Optionally generate combinations of multiple dimensions and measures
    if len(dimensions) >= 2 and len(measures) >= 2:
        dim_combo = f"{dimensions[0]} and {dimensions[1]}"
        measure_combo = f"{measures[0]} vs {measures[1]}"
        recommendations.append((f"{measure_combo} by {dim_combo}", "Scatter"))

    # Use st.container() for future rendering to isolate chart updates
    with st.container():
        st.session_state.recommendations = recommendations

    # Dark theme for Plotly charts
    dark_layout = {
        'paper_bgcolor': '#1f2a44',
        'plot_bgcolor': '#1f2a44',
        'font': {'color': 'white'},
        'xaxis': {'gridcolor': '#444444'},
        'yaxis': {'gridcolor': '#444444'},
        'legend': {'font': {'color': 'white'}},
        'template': 'plotly_dark'
    }

    # CSS for dark-themed tables
    st.markdown("""
        <style>
        .stDataFrame table {
            background-color: #1f2a44;
            color: white;
            border: 1px solid #444444;
        }
        .stDataFrame th {
            background-color: #2a3a5a;
            color: white;
        }
        .stDataFrame td {
            border: 1px solid #444444;
        }
        </style>
    """, unsafe_allow_html=True)

    if not recommendations:
        st.info("No chart recommendations available based on the dataset structure.")
        return

    st.markdown("### Suggested Charts")
    default_chart_options = ["Bar", "Line", "Scatter", "Map", "Table", "Pie", "Bubble", "BubbleCloud"]

    # Filter hash for cache key
    filter_hash = compute_filter_hash(st.session_state.filters)
    dataset_hash = st.session_state.get("dataset_hash", "no_hash")

    # Render recommended charts
    for idx, (prompt, default_chart_type) in enumerate(recommendations[:8]):
        if f"delete_chart_{idx}" not in st.session_state:
            st.session_state[f"delete_chart_{idx}"] = False

        if st.session_state[f"delete_chart_{idx}"]:
            continue

        with st.container():
            col_title, col_chart_type, col_delete = st.columns([3, 2, 1])
            with col_title:
                st.markdown(f"**Recommendation {idx + 1}: {prompt}**")
            with col_chart_type:
                chart_type_key = f"chart_type_rec_{idx}"
                if chart_type_key not in st.session_state:
                    st.session_state[chart_type_key] = default_chart_type

                selected_chart_type = st.selectbox(
                    "",
                    options=default_chart_options,
                    index=default_chart_options.index(st.session_state[chart_type_key]),
                    key=f"chart_type_select_rec_{idx}",
                    label_visibility="collapsed"
                )

                if selected_chart_type != st.session_state[chart_type_key]:
                    st.session_state[chart_type_key] = selected_chart_type
                    st.rerun()

            with col_delete:
                if st.button("🗑️", key=f"delete_button_rec_{idx}"):
                    st.session_state[f"delete_chart_{idx}"] = True
                    chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"
                    if chart_key in st.session_state.chart_cache:
                        del st.session_state.chart_cache[chart_key]
                    st.rerun()

            chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"

            if chart_key in st.session_state.chart_cache:
                chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
            else:
                try:
                    parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                    parsed_chart_type = parsed[0] if parsed else default_chart_type
                    chart_result = render_chart(
                        idx, prompt, dimensions, measures, dates, working_df,
                        sort_order="Descending", chart_type=parsed_chart_type
                    )
                    if chart_result is None:
                        st.error(f"Error processing recommendation: {prompt}")
                        continue
                    chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                    st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs)
                    logger.info(f"Cached chart for key: {chart_key}")
                except Exception as e:
                    st.error(f"⚠️ Failed to render '{prompt}' – {e}")
                    logger.error(f"Failed to render chart for prompt '{prompt}': {str(e)}", exc_info=True)
                    continue

            if selected_chart_type == "Line" and pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                chart_data = chart_data.sort_values(by=dimension)

            with st.container():
                if selected_chart_type == "Bar":
                    fig = px.bar(chart_data, x=dimension, y=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Line":
                    color_arg = kwargs.get("color_by") or secondary_dimension
                    fig = px.line(chart_data, x=dimension, y=metric, color=color_arg)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    fig.update_xaxes(tickformat="%b-%Y", tickangle=45, nticks=10)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Scatter":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    chart_data['truncated_label'] = chart_data[dimension].apply(lambda x: str(x)[:20] + "..." if isinstance(x, str) and len(x) > 20 else x)
                    fig = px.scatter(
                        chart_data, x=metric, y=y_metric, color='truncated_label',
                        custom_data=[dimension]
                    )
                    fig.update_traces(
                        marker=dict(size=12),
                        hovertemplate="%{customdata[0]}<br>%{x}<br>%{y}"
                    )
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Map":
                    fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Table":
                    st.dataframe(chart_data, use_container_width=True, key=f"table_rec_{idx}")
                elif selected_chart_type == "Pie":
                    fig = px.pie(chart_data, names=dimension, values=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Histogram":
                    fig = px.histogram(chart_data, x=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Bubble":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    fig = px.scatter(chart_data, x=metric, y=y_metric, size=metric, color=dimension, size_max=60)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "BubbleCloud":
                    chart_data["_x"] = np.random.rand(len(chart_data))
                    chart_data["_y"] = np.random.rand(len(chart_data))
                    fig = px.scatter(
                        chart_data, x="_x", y="_y", size=metric, color=dimension, text=dimension, size_max=60
                    )
                    fig.update_traces(
                        textposition='top center', marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
                    )
                    fig.update_layout(
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=30, b=0), height=400
                    )
                    fig.update_layout(**dark_layout)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")

                # Insights Expander
                with st.expander("🔍 Insights", expanded=False):
                    if chart_key in st.session_state.insights_cache:
                        insights = st.session_state.insights_cache[chart_key]
                    else:
                        try:
                            insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                            st.session_state.insights_cache[chart_key] = insights
                            logger.info(f"Cached insights for key: {chart_key}")
                        except Exception as e:
                            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
                            insights = ["Unable to generate insights at this time."]

                    if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                        for insight in insights:
                            st.markdown(f"🔹 {insight}")
                    else:
                        st.markdown("No insights available. Try a different chart type or check the data.")

                # Data Table Expander
                with st.expander("📋 View Chart Data", expanded=False):
                    st.dataframe(chart_data, use_container_width=True, key=f"data_table_rec_{idx}")
                    if metric in chart_df.columns and pd.api.types.is_numeric_dtype(chart_df[metric]):
                        st.markdown("### Basic Statistics")
                        stats = calculate_statistics(chart_df, metric)
                        if stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats['mean']:.2f}")
                            with col2:
                                st.metric("Median", f"{stats['median']:.2f}")
                            with col3:
                                st.metric("Min", f"{stats['min']:.2f}")
                            with col4:
                                st.metric("Max", f"{stats['max']:.2f}")

            st.markdown("---")

    # Render custom charts
    base_idx = len(recommendations)
    for custom_idx, (prompt, default_chart_type) in enumerate(st.session_state.custom_charts):
        idx = base_idx + custom_idx
        if f"delete_chart_{idx}" not in st.session_state:
            st.session_state[f"delete_chart_{idx}"] = False

        if st.session_state[f"delete_chart_{idx}"]:
            continue

        with st.container():
            col_title, col_chart_type, col_delete = st.columns([3, 2, 1])
            with col_title:
                st.markdown(f"**Custom Chart {custom_idx + 1}: {prompt}**")
            with col_chart_type:
                chart_type_key = f"chart_type_rec_{idx}"
                if chart_type_key not in st.session_state:
                    st.session_state[chart_type_key] = default_chart_type

                selected_chart_type = st.selectbox(
                    "",
                    options=default_chart_options,
                    index=default_chart_options.index(st.session_state[chart_type_key]),
                    key=f"chart_type_select_rec_{idx}",
                    label_visibility="collapsed"
                )

                if selected_chart_type != st.session_state[chart_type_key]:
                    st.session_state[chart_type_key] = selected_chart_type

            with col_delete:
                if st.button("🗑️", key=f"delete_button_rec_{idx}"):
                    st.session_state[f"delete_chart_{idx}"] = True
                    chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"
                    if chart_key in st.session_state.chart_cache:
                        del st.session_state.chart_cache[chart_key]
                    st.session_state.custom_charts.pop(custom_idx)
                    st.rerun()

            chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"

            if chart_key in st.session_state.chart_cache:
                chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
            else:
                try:
                    parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                    parsed_chart_type = parsed[0] if parsed else default_chart_type
                    chart_result = render_chart(
                        idx, prompt, dimensions, measures, dates, working_df,
                        sort_order="Descending", chart_type=parsed_chart_type
                    )
                    if chart_result is None:
                        st.error(f"Error processing custom chart: {prompt}")
                        continue
                    chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                    st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs)
                    logger.info(f"Cached chart for key: {chart_key}")
                except Exception as e:
                    st.error(f"⚠️ Failed to render '{prompt}': {str(e)}")
                    logger.error(f"Failed to render custom chart for prompt '{prompt}': {str(e)}", exc_info=True)
                    continue

            if selected_chart_type == "Line" and pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                chart_data = chart_data.sort_values(by=dimension)

            with st.container():
                if selected_chart_type == "Bar":
                    fig = px.bar(chart_data, x=dimension, y=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Line":
                    color_arg = kwargs.get("color_by") or secondary_dimension
                    fig = px.line(chart_data, x=dimension, y=metric, color=color_arg)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    fig.update_xaxes(tickformat="%b-%Y", tickangle=45, nticks=10)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Scatter":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    chart_data['truncated_label'] = chart_data[dimension].apply(lambda x: str(x)[:20] + "..." if isinstance(x, str) and len(x) > 20 else x)
                    fig = px.scatter(
                        chart_data, x=metric, y=y_metric, color='truncated_label',
                        custom_data=[dimension]
                    )
                    fig.update_traces(
                        marker=dict(size=12),
                        hovertemplate="%{customdata[0]}<br>%{x}<br>%{y}"
                    )
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Map":
                    fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Table":
                    st.dataframe(chart_data, use_container_width=True, key=f"table_rec_{idx}")
                elif selected_chart_type == "Pie":
                    fig = px.pie(chart_data, names=dimension, values=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Histogram":
                    fig = px.histogram(chart_data, x=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Bubble":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    fig = px.scatter(chart_data, x=metric, y=y_metric, size=metric, color=dimension, size_max=60)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "BubbleCloud":
                    chart_data["_x"] = np.random.rand(len(chart_data))
                    chart_data["_y"] = np.random.rand(len(chart_data))
                    fig = px.scatter(
                        chart_data, x="_x", y="_y", size=metric, color=dimension, text=dimension, size_max=60
                    )
                    fig.update_traces(
                        textposition='top center', marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
                    )
                    fig.update_layout(
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=30, b=0), height=400
                    )
                    fig.update_layout(**dark_layout)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")

                # Insights Expander
                with st.expander("🔍 Insights", expanded=False):
                    if chart_key in st.session_state.insights_cache:
                        insights = st.session_state.insights_cache[chart_key]
                    else:
                        try:
                            insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                            st.session_state.insights_cache[chart_key] = insights
                            logger.info(f"Cached insights for key: {chart_key}")
                        except Exception as e:
                            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
                            insights = ["Unable to generate insights at this time."]

                    if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                        for insight in insights:
                            st.markdown(f"🔹 {insight}")
                    else:
                        st.markdown("No insights available. Try a different chart type or check the data.")

                # Data Table Expander
                with st.expander("📋 View Chart Data", expanded=False):
                    st.dataframe(chart_data, use_container_width=True, key=f"data_table_rec_{idx}")
                    if metric in chart_df.columns and pd.api.types.is_numeric_dtype(chart_df[metric]):
                        st.markdown("### Basic Statistics")
                        stats = calculate_statistics(chart_df, metric)
                        if stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats['mean']:.2f}")
                            with col2:
                                st.metric("Median", f"{stats['median']:.2f}")
                            with col3:
                                st.metric("Min", f"{stats['min']:.2f}")
                            with col4:
                                st.metric("Max", f"{stats['max']:.2f}")

            st.markdown("---")

    # Render prompt input box with sample prompts
    with st.container():
        st.markdown("### Add Custom Chart")
        sample_prompts = []

        # Generate 4 sample prompts
        sample_prompts = generate_sample_prompts(dimensions, measures, dates, working_df, max_prompts=4)
        sample_prompts = [p.split(". ", 1)[1] if ". " in p else p for p in sample_prompts]

        if sample_prompts:
            st.markdown('<div class="sample-prompts">', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, prompt in enumerate(sample_prompts[:4]):
                unique_key = f"sample_prompt_{i}_{prompt.replace(' ', '_')}"
                with cols[i]:
                    if st.button(prompt, key=unique_key):
                        existing_prompts = [p for p, _ in recommendations] + [p for p, _ in st.session_state.custom_charts]
                        if prompt not in existing_prompts:
                            with st.spinner("Generating chart..."):
                                parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                                chart_type = parsed[0] if parsed else "Bar"
                                st.session_state.custom_charts.append((prompt, chart_type))
                                logger.info(f"Added sample prompt chart: {prompt}")
                            try:
                                st.toast("Chart added!")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.warning(f"Chart with prompt '{prompt}' already exists.")
            st.markdown('</div>', unsafe_allow_html=True)

        user_prompt = st.text_input(
            "📝 Ask about your data (e.g., 'Sales vs Profit by City' or 'Top 5 Cities by Sales'):",
            key="rec_manual_prompt"
        )

        if st.button("📈 Generate Chart", key="rec_manual_prompt_button"):
            if user_prompt:
                existing_prompts = [p for p, _ in recommendations] + [p for p, _ in st.session_state.custom_charts]
                if user_prompt not in existing_prompts:
                    with st.spinner("Generating chart..."):
                        parsed = rule_based_parse(user_prompt, working_df, dimensions, measures, dates)
                        chart_type = parsed[0] if parsed else "Bar"
                        st.session_state.custom_charts.append((user_prompt, chart_type))
                        logger.info(f"Added custom prompt chart: {user_prompt}")
                    try:
                        st.toast("Chart added!")
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.warning(f"Chart with prompt '{user_prompt}' already exists.")



    # Dashboard saving section
    render_save_dashboard_section(supabase)



def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    try:
        logger.info("Generating executive summary for %d charts", len(chart_history))

        # Initialize collections for insights
        key_metrics = {}
        trends = {}
        correlations = {}
        recommendations = []

        # Aggregate insights from recommended and custom charts
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result is None:
                continue

            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result

            # Calculate statistics for numeric metrics
            stats = calculate_statistics(working_df, metric) if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]) else None

            if stats:
                key_metrics[metric] = {
                    'mean': stats['mean'],
                    'max': stats['max'],
                    'min': stats['min'],
                    'std_dev': stats['std_dev']
                }

                # Analyze trends for date-based dimensions
                if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                    monthly_avg = chart_data.groupby(chart_data[dimension].dt.to_period('M'))[metric].mean()
                    trends[metric] = {
                        'peak': (monthly_avg.idxmax(), monthly_avg.max()),
                        'low': (monthly_avg.idxmin(), monthly_avg.min()),
                        'trend': 'increasing' if monthly_avg.iloc[-1] > monthly_avg.iloc[0] else 'decreasing'
                    }

                # Identify correlations between measures
                for other_metric in measures:
                    if other_metric != metric and other_metric in chart_data.columns and pd.api.types.is_numeric_dtype(chart_data[other_metric]):
                        corr = chart_data[[metric, other_metric]].corr().iloc[0, 1]
                        if abs(corr) > 0.5:
                            correlations[(metric, other_metric)] = corr

        # Build concise executive summary
        summary = []

        # Key Metrics Section
        if key_metrics:
            summary.append("**Key Performance Metrics**")
            for metric, values in key_metrics.items():
                summary.append(f"- {metric}: Avg ${values['mean']:.2f} (Range: ${values['min']:.2f} to ${values['max']:.2f})")
                if values['std_dev'] > values['mean'] * 0.5:
                    recommendations.append(f"- Standardize processes to reduce high variability in {metric}")
                if values['min'] < values['mean'] * 0.5:
                    recommendations.append(f"- Address underperforming segments in {metric}")

        # Trends Section
        if trends:
            summary.append("\n**Key Trends**")
            for metric, trend_data in trends.items():
                summary.append(f"- {metric} is {trend_data['trend']}, peaking at ${trend_data['peak'][1]:.2f} in {trend_data['peak'][0]}")
                if trend_data['trend'] == 'decreasing':
                    recommendations.append(f"- Develop strategies to reverse declining {metric} trend")
                else:
                    recommendations.append(f"- Scale initiatives driving {metric} growth")

        # Correlations Section
        if correlations:
            summary.append("\n**Key Relationships**")
            for (metric1, metric2), corr in correlations.items():
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                summary.append(f"- {strength.title()} {direction} correlation ({corr:.2f}) between {metric1} and {metric2}")
                if corr > 0.7:
                    recommendations.append(f"- Leverage synergy between {metric1} and {metric2} for cross-promotion")
                elif corr < -0.7:
                    recommendations.append(f"- Investigate trade-offs between {metric1} and {metric2}")

        # Strategic Recommendations
        if recommendations:
            summary.append("\n**Strategic Recommendations**")
            for rec in recommendations[:5]:  # Limit to top 5 recommendations
                summary.append(f"- {rec}")

        # Fallback if no insights
        if not summary:
            summary = ["No significant insights could be generated from the charts."]

        logger.info(f"Generated executive summary with {len(summary)} points")
        return summary

    except Exception as e:
        logger.error("Failed to generate executive summary: %s", str(e))
        return [
            "Error: Unable to generate summary.",
            "Please ensure valid chart data and try again."
        ]

# Helper function for PDF generation
def generate_pdf_summary(summary_points, overall_analysis):
    """
    Generate a PDF summary from summary points and overall analysis.
    Args:
        summary_points (list): List of summary points
        overall_analysis (list): List of overall analysis points
    Returns:
        bytes: PDF content as bytes
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Executive Summary Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Summary of Dashboard Analysis
        story.append(Paragraph("Summary of Dashboard Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in summary_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Overall Data Analysis
        story.append(PageBreak())
        story.append(Paragraph("Overall Data Analysis and Findings", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in overall_analysis:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        #logger.info("Generated PDF summary successfully")
        return pdf_content
    except Exception as e:
        logger.error(f"Failed to generate PDF summary: {str(e)}", exc_info=True)
        raise

@st.cache_data
def cached_generate_executive_summary(chart_history, df, dimensions, measures, dates):
    """Cached version of generate_executive_summary to reduce reruns."""
    return generate_executive_summary(chart_history, df, dimensions, measures, dates)

@st.cache_data
def cached_generate_overall_data_analysis(df, dimensions, measures, dates):
    """Cached version of generate_overall_data_analysis to reduce reruns."""
    return generate_overall_data_analysis(df, dimensions, measures, dates)

def executive_summary_tab(df):
    """
    Render the Executive Summary tab with dashboard analysis and overall data analysis.
    Args:
        df (pd.DataFrame): The dataset to use for generating summaries.
    """
    st.subheader("📜 Executive Summary")

    # Initialize session state
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = None
    if "overall_analysis" not in st.session_state:
        st.session_state.overall_analysis = None
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None

    if df is None:
        st.warning("Please upload a dataset in the 'Data Manager' tab to view the executive summary.")
        return

    dimensions = st.session_state.field_types.get("dimension", [])
    measures = st.session_state.field_types.get("measure", [])
    dates = st.session_state.field_types.get("date", [])

    # Custom CSS for styled PDF button
    st.markdown("""
        <style>
        .pdf-download-button {
            background-color: #26A69A !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            cursor: pointer !important;
            font-weight: 500 !important;
            text-align: center !important;
            display: inline-block !important;
            text-decoration: none !important;
        }
        .pdf-download-button:hover {
            background-color: #2E7D32 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Generate Summaries Button
    if st.button("📑 Generate Summaries", key="generate_summaries"):
        with st.spinner("Generating summaries..."):
            try:
                # Gather chart history from recommendations and custom charts
                chart_history = [
                    {"prompt": prompt, "chart_type": st.session_state.get(f"chart_type_rec_{i}", chart_type)}
                    for i, (prompt, chart_type) in enumerate(st.session_state.get("recommendations", [])[:8])
                    if not st.session_state.get(f"delete_chart_{i}", False)
                ] + [
                    {"prompt": prompt, "chart_type": st.session_state.get(f"chart_type_rec_{len(st.session_state.get('recommendations', [])) + i}", chart_type)}
                    for i, (prompt, chart_type) in enumerate(st.session_state.get("custom_charts", []))
                    if not st.session_state.get(f"delete_chart_{len(st.session_state.get('recommendations', [])) + i}", False)
                ]
                # Generate both summaries
                st.session_state.executive_summary = cached_generate_executive_summary(chart_history, df, dimensions, measures, dates)
                st.session_state.overall_analysis = cached_generate_overall_data_analysis(df, dimensions, measures, dates)
                st.success("Summaries generated successfully!")
            except Exception as e:
                logger.error(f"Error generating summaries: {str(e)}")
                st.error(f"Failed to generate summaries: {str(e)}")

    # Display Dashboard Analysis Summary
    st.markdown("### Summary of Dashboard Analysis")
    with st.expander("View Dashboard Analysis", expanded=True):
        if st.session_state.executive_summary is None:
            st.info("Click 'Generate Summaries' to view the dashboard analysis.")
        elif st.session_state.executive_summary == ["No significant insights could be generated from the charts."]:
            st.info("No significant insights found. Generate charts in the 'Recommended Charts & Insights' tab.")
        else:
            for point in st.session_state.executive_summary:
                st.markdown(f"- {point}")

    # Display Overall Data Analysis
    st.markdown("### Overall Data Analysis")
    with st.expander("View Overall Data Analysis", expanded=False):
        if st.session_state.overall_analysis is None:
            st.info("Click 'Generate Summaries' to view the overall data analysis.")
        else:
            for point in st.session_state.overall_analysis:
                st.markdown(f"- {point}")

    # PDF Export
    def generate_and_store_pdf():
        try:
            if st.session_state.executive_summary and st.session_state.overall_analysis:
                pdf_content = generate_pdf_summary(st.session_state.executive_summary, st.session_state.overall_analysis)
                st.session_state.pdf_content = pdf_content
            else:
                st.error("Generate summaries before exporting to PDF.")
        except ImportError:
            logger.error("ReportLab not installed for PDF generation")
            st.error("Please install reportlab: `pip install reportlab`")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            st.error(f"Failed to generate PDF: {str(e)}")

    # Text Fallback
    def generate_text_summary():
        try:
            if st.session_state.executive_summary and st.session_state.overall_analysis:
                text_content = "Executive Summary Report\n\nSummary of Dashboard Analysis\n" + "\n".join([f"- {p}" for p in st.session_state.executive_summary])
                text_content += "\n\nOverall Data Analysis and Findings\n" + "\n".join([f"- {p}" for p in st.session_state.overall_analysis])
                return text_content.encode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Failed to generate text summary: {str(e)}")
            st.error(f"Failed to generate text summary: {str(e)}")
            return None

    if st.button("📄 Export Summaries to PDF", key="export_summaries_pdf", on_click=generate_and_store_pdf):
        pass  # Button click triggers the callback

    if st.session_state.pdf_content:
        st.download_button(
            label="Download PDF",
            data=st.session_state.pdf_content,
            file_name=f"Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_pdf_button",
            help="Download the executive summary and overall data analysis as a PDF."
        )
    else:
        text_content = generate_text_summary()
        if text_content:
            st.download_button(
                label="Download Summary as Text",
                data=text_content,
                file_name=f"Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_txt_button",
                help="Download the executive summary and overall data analysis as a text file."
            )

import openai 

def render_breadcrumb():
    cols = st.columns([1] * (len(st.session_state.breadcrumb) * 2 - 1))
    for i, crumb in enumerate(st.session_state.breadcrumb):
        col_idx = i * 2
        with cols[col_idx]:
            if i < len(st.session_state.breadcrumb) - 1:
                if st.button(
                    crumb["label"],
                    key=f"breadcrumb_{i}_{crumb['label']}_{hash(str(crumb))}",
                    help=f"Navigate to {crumb['label']}",
                    use_container_width=False
                ):
                    st.session_state.dashboard_view = crumb["view"]
                    if crumb["view"] == "dashboards":
                        st.session_state.selected_project = crumb["project_id"]
                        st.session_state.selected_dashboard = None
                        # Set current_project only if directory exists
                        project_dir = f"projects/{crumb['project_id']}"
                        if os.path.exists(project_dir):
                            st.session_state.current_project = crumb["project_id"]
                        else:
                            st.session_state.current_project = "my_project"
                            logger.warning(f"Project directory {project_dir} does not exist, falling back to my_project")
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "dashboards", "project_id": crumb["project_id"]}
                    elif crumb["view"] == "dashboard":
                        st.session_state.selected_project = crumb["project_id"]
                        st.session_state.selected_dashboard = crumb["dashboard_id"]
                        project_dir = f"projects/{crumb['project_id']}"
                        if os.path.exists(project_dir):
                            st.session_state.current_project = crumb["project_id"]
                        else:
                            st.session_state.current_project = "my_project"
                            logger.warning(f"Project directory {project_dir} does not exist, falling back to my_project")
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "dashboard", "project_id": crumb["project_id"], "dashboard_id": crumb["dashboard_id"]}
                    else:
                        st.session_state.selected_project = None
                        st.session_state.selected_dashboard = None
                        st.session_state.current_project = "my_project"
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "projects"}
                    st.session_state.breadcrumb = st.session_state.breadcrumb[:i + 1]
                    st.rerun()
            else:
                st.markdown(f"<span class='breadcrumb-current'>{crumb['label']}</span>", unsafe_allow_html=True)
        if i < len(st.session_state.breadcrumb) - 1:
            with cols[col_idx + 1]:
                st.markdown("<span class='breadcrumb-separator'>></span>", unsafe_allow_html=True)
    logger.debug(f"Breadcrumb state: {st.session_state.breadcrumb}")

def generate_overall_data_analysis(df, dimensions, measures, dates):
    """
    Generate overall data analysis and findings.
    Args:
        df (pd.DataFrame): The dataset
        dimensions (list): List of dimension columns
        measures (list): List of measure columns
        dates (list): List of date columns
    Returns:
        list: List of analysis points
    """
    try:
        analysis = []
        
        # Basic dataset overview
        analysis.append(f"**Dataset Overview:** {len(df)} records with {len(dimensions)} dimensions and {len(measures)} measures")
        
        # Key metrics analysis
        for measure in measures:
            if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                stats = calculate_statistics(df, measure)
                analysis.append(f"\n**{measure} Analysis:**")
                analysis.append(f"- Average: ${stats['mean']:.2f}")
                analysis.append(f"- Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                analysis.append(f"- Variability: {stats['std_dev']:.2f} ({(stats['std_dev']/stats['mean']*100):.1f}% of mean)")
        
        # Dimension analysis
        for dimension in dimensions:
            if dimension in df.columns:
                unique_values = df[dimension].nunique()
                analysis.append(f"\n**{dimension} Analysis:**")
                analysis.append(f"- {unique_values} unique values")
                if unique_values < 10:  # For categorical dimensions with few values
                    value_counts = df[dimension].value_counts()
                    analysis.append("- Distribution:")
                    for value, count in value_counts.items():
                        analysis.append(f"  * {value}: {count} ({count/len(df)*100:.1f}%)")
        
        # Date analysis
        for date_col in dates:
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                date_range = df[date_col].max() - df[date_col].min()
                analysis.append(f"\n**{date_col} Analysis:**")
                analysis.append(f"- Date Range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")
                analysis.append(f"- Span: {date_range.days} days")
        
        # Correlation analysis
        if len(measures) > 1:
            analysis.append("\n**Measure Correlations:**")
            for i, m1 in enumerate(measures):
                for m2 in measures[i+1:]:
                    if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                        corr = df[[m1, m2]].corr().iloc[0, 1]
                        if abs(corr) > 0.3:  # Only show meaningful correlations
                            analysis.append(f"- {m1} and {m2}: {corr:.2f}")
        
        return analysis
    except Exception as e:
        logger.error("Failed to generate overall data analysis: %s", str(e))
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

@st.cache_data
def filter_dashboards(dashboards, search_query, project_filter, type_filter, date_filter, tag_filter, _supabase):
    start_time = time.time()
    filtered = dashboards.copy()
    
    # Apply non-charts-dependent filters first
    if search_query:
        filtered = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]
    if project_filter:
        filtered = filtered[filtered["project_id"].isin(project_filter)]
    if date_filter:
        filtered = filtered[pd.to_datetime(filtered["created_at"]).dt.date >= date_filter]
    if tag_filter:
        tags = [tag.strip() for tag in tag_filter.split(",")]
        filtered = filtered[filtered["tags"].apply(lambda t: any(tag in t for tag in tags if t))]
    
    # Fetch charts for remaining dashboards if type_filter is applied
    if type_filter and not filtered.empty:
        dashboard_ids = filtered["id"].tolist()
        charts_data = fetch_dashboard_charts(_supabase, dashboard_ids)
        filtered["charts"] = filtered["id"].map(charts_data).fillna([])
        filtered["analytics_types"] = filtered["charts"].apply(
            lambda charts: list(set(get_analytics_type(chart) for chart in charts)) if charts else []
        )
        filtered = filtered[filtered["analytics_types"].apply(lambda types: any(t in type_filter for t in types))]
    
    logger.info(f"Filtering dashboards took {time.time() - start_time:.2f} seconds")
    return filtered





def load_data(file):
    """
    Load a CSV file with multiple encoding attempts.
    Args:
        file: Uploaded file object from st.file_uploader
    Returns:
        pandas.DataFrame: Loaded dataset
    Raises:
        UnicodeDecodeError: If all encodings fail
        Exception: For other file reading errors
    """
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            logger.debug(f"Attempting to read file with encoding: {encoding}")
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file, encoding=encoding)
            #logger.info(f"Successfully read file with encoding: {encoding}")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read file with encoding {encoding}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error reading file: {str(e)}")
            raise
    raise UnicodeDecodeError("Failed to decode file with supported encodings (UTF-8, ISO-8859-1, Windows-1252)", b"", 0, 0, "All encoding attempts failed")


### Helper Function To support field type retrieval, add this helper function (assumed to be in `utils.py` or similar):
def get_field_type(col, field_types):
    for t in ["dimension", "measure", "date", "id"]:
        if col in field_types.get(t, []):
            return t.capitalize()
    return "Other"

# Tabs

if "uploaded_data" not in st.session_state or st.session_state.uploaded_data is None:
 
   


#elif st.session_state.current_project or st.session_state.uploaded_data:
    tab1,  tab2, tab6,  tab4, tab5 = st.tabs(["📊 Data", "🛠️ Field Editor" , "📊 Recommended Charts & Insights",  "📜 Executive Summary", "💾 Saved Dashboards"])
    

    with tab1:
        st.subheader("📊 Data Management")

        st.markdown("### 📤 Upload Dataset")
        with st.container():
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="upload_csv_unique")
            if uploaded_file:
                try:
                    df = load_data(uploaded_file)
                    st.session_state.dataset = df
                    st.session_state.dataset_hash = compute_dataset_hash(df)
                    st.session_state.classified = False
                    st.session_state.sample_prompts = []
                    st.session_state.used_sample_prompts = []
                    st.session_state.sample_prompt_pool = []
                    st.session_state.last_used_pool_index = 0
                    st.session_state.field_types = {}
                    st.session_state.chart_cache = {}
                    st.session_state.insights_cache = {}
                    os.makedirs(f"projects/{st.session_state.current_project}", exist_ok=True)
                    df.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
                    st.session_state.onboarding_seen = True  # Auto-collapse onboarding modal
                    st.success("✅ Dataset uploaded!")
                    #logger.info("Uploaded dataset for project: %s, Hash: %s", st.session_state.current_project, st.session_state.dataset_hash)
                except UnicodeDecodeError as e:
                    st.error("Failed to upload dataset: Unable to decode file with supported encodings (UTF-8, ISO-8859-1, Windows-1252). Please ensure the file is a valid CSV with a supported encoding.")
                    logger.error("Failed to upload dataset for project %s: %s", st.session_state.current_project, str(e))
                except Exception as e:
                    st.error(f"Failed to upload dataset: {str(e)}")
                    logger.error("Failed to upload dataset for project %s: %s", st.session_state.current_project, str(e))
            else:
                st.info("Please upload a CSV file to begin.")

        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                    df = preprocess_dates(df)
                    st.session_state.field_types = {
                        "dimension": dimensions,
                        "measure": measures,
                        "date": dates,
                        "id": ids,
                    }
                    st.session_state.classified = True
                    st.session_state.dataset = df
                    save_field_types(st.session_state.current_project, st.session_state.field_types)
                    #logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                   #             st.session_state.current_project, dimensions, measures, dates, ids)
                except Exception as e:
                    st.error(f"Failed to classify columns: {str(e)}")
                    logger.error("Failed to classify columns: %s", str(e))
                    st.stop()

            dimensions = st.session_state.field_types.get("dimension", [])
            measures = st.session_state.field_types.get("measure", [])
            dates = st.session_state.field_types.get("date", [])
            ids = st.session_state.field_types.get("id", [])

            # Dark theme for Plotly charts
            dark_layout = {
                'paper_bgcolor': '#1f2a44',
                'plot_bgcolor': '#1f2a44',
                'font': {'color': 'white'},
                'xaxis': {'gridcolor': '#444444'},
                'yaxis': {'gridcolor': '#444444'},
                'legend': {'font': {'color': 'white'}},
                'template': 'plotly_dark'
            }

            # Data Explorer Section
            st.markdown("### 🔍 Data Explorer")
            with st.container():
                num_rows = st.slider("Number of rows to display:", min_value=1, max_value=100, value=100, key="data_explorer_rows")
                st.write("Sample Data with Field Type Editor:")
                # Prepare columns with dynamic field type dropdowns
                col_configs = {}
                for col in df.columns:
                    dtype = df[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                        col_configs[col] = st.column_config.NumberColumn(col, format="%s")
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        col_configs[col] = st.column_config.DatetimeColumn(col)
                    else:
                        col_configs[col] = st.column_config.TextColumn(col)
                    col_configs[f"{col}_type"] = st.column_config.SelectboxColumn(
                        f"{col}_type",
                        options=["Dimension", "Measure", "Date", "ID", "Other"],
                        default=get_field_type(col, st.session_state.field_types),
                        width="small"
                    )
                # Create sample dataframe with type columns
                sample_df = df.head(num_rows).copy()
                type_data = {f"{col}_type": [get_field_type(col, st.session_state.field_types)] * num_rows for col in df.columns}
                display_df = sample_df.join(pd.DataFrame(type_data))
                try:
                    edited_df = st.data_editor(
                        display_df,
                        column_config=col_configs,
                        use_container_width=True,
                        key="data_explorer_editor"
                    )
                    # Process field type changes
                    type_cols = [col for col in edited_df.columns if col.endswith("_type")]
                    changes_made = False
                    for col in df.columns:
                        type_col = f"{col}_type"
                        new_type = edited_df[type_col].iloc[0]  # Assume all rows have the same type
                        current_type = get_field_type(col, st.session_state.field_types)
                        if new_type != current_type:
                            changes_made = True
                            # Remove from current type
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            # Add to new type
                            if new_type.lower() != "other":
                                st.session_state.field_types.setdefault(new_type.lower(), []).append(col)
                            #logger.info(f"Changed field type for {col} to {new_type} in Data Explorer")
                    if changes_made:
                        save_field_types(st.session_state.current_project, st.session_state.field_types)
                        st.session_state.sample_prompts = []
                        st.session_state.used_sample_prompts = []
                        st.session_state.sample_prompt_pool = []
                        st.session_state.last_used_pool_index = 0
                        st.session_state.chart_cache = {}
                        st.session_state.insights_cache = {}
                        st.session_state.dataset_hash = compute_dataset_hash(df)
                        st.success("Field types updated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to render data editor: {str(e)}")
                    logger.error(f"Failed to render data editor: {str(e)}")
                    st.dataframe(display_df, use_container_width=True)

            # Dimension Unique Values Section
            st.markdown("---")
            st.markdown("### 📋 Dimension Fields and Unique Values")
            if dimensions:
                st.markdown("Dimensions:")
                for dim in dimensions:
                    unique_vals = df[dim].dropna().unique()
                    vals_display = ", ".join(map(str, unique_vals[:10])) + ("..." if len(unique_vals) > 10 else "")
                    st.markdown(f"- **{dim}**: {vals_display}")
            else:
                st.info("No dimension fields identified.")

            # Correlation Heatmap Section
            st.markdown("---")
            st.markdown("### 📈 Correlation Heatmap for Measures")
            if measures and len(measures) >= 2:
                try:
                    corr_matrix = df[measures].corr()
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        text_auto=".2f",
                        aspect="auto"
                    )
                    fig.update_layout(**dark_layout, title="Correlation Between Measures")
                    st.plotly_chart( fig, use_container_width=True, key="correlation_heatmap")
                except Exception as e:
                    st.error(f"Failed to generate correlation heatmap: {str(e)}")
                    logger.error(f"Failed to generate correlation heatmap: {str(e)}")
            else:
                st.info("Need at least two measure fields for correlation heatmap.")
# Fields Tab
    with tab2:
        st.session_state.onboarding_seen = True
        if st.session_state.dataset is not None:
            st.subheader("🛠️ Field Editor")
            df = st.session_state.dataset

            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                    df = preprocess_dates(df)  # force parsing of any detected date columns
                    st.session_state.field_types = {
                        "dimension": dimensions,
                        "measure": measures,
                        "date": dates,
                        "id": ids,
                    }
                    st.session_state.classified = True
                    st.session_state.dataset = df
                    #logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                      #          st.session_state.current_project, dimensions, measures, dates, ids)
                except Exception as e:
                    st.error(f"Failed to classify columns: %s", str(e))
                    logger.error("Failed to classify columns: %s", str(e))
                    st.stop()

            st.markdown("### 🔧 Manage Fields and Types")
            with st.expander("Manage Fields and Types", expanded=False):
                for col in df.columns:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{col}**")
                    with col2:
                        current_type = "Other"
                        if col in st.session_state.field_types.get("dimension", []):
                            current_type = "Dimension"
                        elif col in st.session_state.field_types.get("measure", []):
                            current_type = "Measure"
                        elif col in st.session_state.field_types.get("date", []):
                            current_type = "Date"
                        elif col in st.session_state.field_types.get("id", []):
                            current_type = "ID"
                        
                        new_type = st.selectbox(
                            f"Type for {col}",
                            ["Dimension", "Measure", "Date", "ID", "Other"],
                            index=["Dimension", "Measure", "Date", "ID", "Other"].index(current_type),
                            key=f"type_select_{col}",
                            label_visibility="collapsed"
                        )
                        
                        if new_type != current_type:
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            if new_type.lower() != "other":
                                if new_type.lower() in st.session_state.field_types:
                                    st.session_state.field_types[new_type.lower()].append(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.session_state.chart_cache = {}  # Clear chart cache
                            st.session_state.insights_cache = {}  # Clear insights cache
                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                            st.success(f"Field {col} type changed to {new_type}!")
                            #logger.info("Changed field type for %s to %s", col, new_type)
                    with col3:
                        new_name = st.text_input(
                            "Rename",
                            value=col,
                            key=f"rename_{col}",
                            label_visibility="collapsed",
                            placeholder="New name"
                        )
                        if new_name and new_name != col:
                            if new_name in df.columns:
                                st.error("Field name already exists!")
                                logger.warning("Attempted to rename field %s to %s, but name already exists", col, new_name)
                            else:
                                df.rename(columns={col: new_name}, inplace=True)
                                st.session_state.dataset = df
                                for t in ["dimension", "measure", "date", "id"]:
                                    if col in st.session_state.field_types.get(t, []):
                                        st.session_state.field_types[t].remove(col)
                                        st.session_state.field_types[t].append(new_name)
                                save_dataset_changes()
                                st.session_state.sample_prompts = []
                                st.session_state.used_sample_prompts = []
                                st.session_state.sample_prompt_pool = []
                                st.session_state.last_used_pool_index = 0
                                st.session_state.chart_cache = {}  # Clear chart cache
                                st.session_state.insights_cache = {}  # Clear insights cache
                                st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                                st.success(f"Field renamed to {new_name}!")
                                #logger.info("Renamed field %s to %s", col, new_name)
                    with col4:
                        if st.button("Delete", key=f"delete_btn_{col}"):
                            df.drop(columns=[col], inplace=True)
                            st.session_state.dataset = df
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.session_state.chart_cache = {}  # Clear chart cache
                            st.session_state.insights_cache = {}  # Clear insights cache
                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                            st.success(f"Field {col} deleted!")
                            #logger.info("Deleted field: %s", col)

            st.markdown("### ➕ Create Calculated Fields")
            st.markdown("""
            Create a new calculated field by either describing it in plain English, selecting a predefined template, or directly entering a formula. Supported functions: SUM, AVG, COUNT, STDEV, MEDIAN, MIN, MAX, IF-THEN-ELSE-END.
            """)
            
            input_mode = st.radio(
                "Select Input Mode:",
                ["Prompt-based (Plain English)", "Direct Formula Input"],
                key="calc_input_mode"
            )
            
            st.markdown("#### Predefined Calculation Templates")
            template = st.selectbox(
                "Select a Template (Optional):",
                ["None"] + list(PREDEFINED_CALCULATIONS.keys()),
                key="calc_template"
            )
            
            calc_prompt = ""
            formula_input = ""
            
            if template != "None":
                calc_prompt = PREDEFINED_CALCULATIONS[template]["prompt"]
                formula_input = PREDEFINED_CALCULATIONS[template]["formula"]
            
            dimensions = st.session_state.field_types.get("dimension", [])
            group_by = st.selectbox(
                "Group By (Optional, for 'per' aggregations):",
                ["None"] + dimensions,
                key="calc_group_by"
            )
            group_by = None if group_by == "None" else group_by
            
            if input_mode == "Prompt-based (Plain English)":
                st.markdown("#### Describe Your Calculation")
                measures = st.session_state.field_types.get("measure", [])
                dimensions = st.session_state.field_types.get("dimension", [])
                sample_measure1 = measures[0] if measures else "Measure1"
                sample_measure2 = measures[1] if len(measures) > 1 else "Measure2"
                sample_dimension = dimensions[0] if dimensions else "Dimension1"
                
                examples = [
                    f"Mark {sample_measure1} as High if greater than 1000, otherwise Low",
                    f"Calculate the profit margin as {sample_measure1} divided by {sample_measure2}",
                    f"Flag outliers in {sample_measure1} where {sample_measure1} is more than 2 standard deviations above the average",
                    f"Calculate average {sample_measure1} per {sample_dimension} and flag if above overall average",
                    f"If {sample_measure1} is greater than 500 and {sample_measure2} is positive, then High Performer, else if {sample_measure1} is less than 200, then Low Performer, else Medium"
                ]
                
                st.markdown("Examples:")
                for example in examples:
                    st.markdown(f"- {example}")
                
                calc_prompt = st.text_area("Describe Your Calculation in Plain Text:", value=calc_prompt, key="calc_prompt")
            else:
                st.markdown("#### Enter Formula Directly")
                st.markdown("""
                Enter a formula using exact column names (e.g., Sales, not [Sales]). Examples:
                - IF Sales > 1000 THEN 'High' ELSE 'Low' END
                - Profit / Sales
                - IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END
                - IF AVG(Profit) PER Ship Mode > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END
                """)
                formula_input = st.text_area("Enter Formula:", value=formula_input, key="calc_formula_input")
            
            new_field_name = st.text_input("New Field Name:", key="calc_new_field")
            
            if st.button("Create Calculated Field", key="calc_create"):
                if new_field_name in df.columns:
                    st.error("Field name already exists!")
                    logger.warning("Attempted to create field %s, but name already exists", new_field_name)
                elif not new_field_name:
                    st.error("Please provide a new field name!")
                    logger.warning("User attempted to create a calculated field without a name")
                elif (input_mode == "Prompt-based (Plain English)" and not calc_prompt) or (input_mode == "Direct Formula Input" and not formula_input):
                    st.error("Please provide a calculation description or formula!")
                    logger.warning("User attempted to create a calculated field without a description or formula")
                else:
                    with st.spinner("Processing calculation..."):
                        proceed_with_evaluation = True

                        if input_mode == "Prompt-based (Plain English)":
                            formula = generate_formula_from_prompt(
                                calc_prompt,
                                st.session_state.field_types.get("dimension", []),
                                st.session_state.field_types.get("measure", []),
                                df
                            )
                        else:
                            formula = formula_input
                        
                        if not formula:
                            st.error("Could not generate a formula from the prompt.")
                            logger.warning("Failed to generate formula for prompt: %s", calc_prompt)
                            proceed_with_evaluation = False

                        if proceed_with_evaluation:
                            if '=' in formula:
                                parts = formula.split('=', 1)
                                if len(parts) == 2:
                                    formula = parts[1].strip()
                                else:
                                    st.error("Invalid formula format.")
                                    logger.warning("Invalid formula format: %s", formula)
                                    proceed_with_evaluation = False

                            if proceed_with_evaluation:
                                for col in df.columns:
                                    formula = formula.replace(f"[{col}]", col)
                                
                                working_df = df.copy()
                                formula_modified = formula
                                group_averages = {}
                                overall_avg = None
                                group_dim = None
                                
                                per_match = re.search(r'AVG\((\w+)\)\s+PER\s+(\w+(?:\s+\w+)*)', formula_modified, re.IGNORECASE)
                                if per_match:
                                    agg_col = per_match.group(1)
                                    group_dim = per_match.group(2)
                                    if agg_col in working_df.columns and group_dim in working_df.columns:
                                        overall_avg = working_df[agg_col].mean()
                                        group_averages = working_df.groupby(group_dim)[agg_col].mean().to_dict()
                                        formula_modified = formula_modified.replace(f"AVG({agg_col})", str(overall_avg))
                                        formula_modified = re.sub(r'\s+PER\s+\w+(?:\s+\w+)*', '', formula_modified)
                                    else:
                                        st.error("Invalid columns in PER expression.")
                                        logger.error("Invalid columns in PER expression: %s, %s", agg_col, group_dim)
                                        proceed_with_evaluation = False
                                else:
                                    for col in df.columns:
                                        if f"AVG({col})" in formula_modified:
                                            avg_value = working_df[col].mean()
                                            formula_modified = formula_modified.replace(f"AVG({col})", str(avg_value))
                                        if f"STDEV({col})" in formula_modified:
                                            std_value = working_df[col].std()
                                            formula_modified = formula_modified.replace(f"STDEV({col})", str(std_value))
                                
                                if proceed_with_evaluation:
                                    formula_modified = parse_if_statement(formula_modified)
                                    st.markdown(f"**Formula Used:** `{formula}`")
                                    st.markdown(f"**Processed Formula:** `{formula_modified}`")
                                    try:
                                        def evaluate_row(row):
                                            local_vars = row.to_dict()
                                            if group_averages and group_dim in local_vars:
                                                group_value = group_averages.get(local_vars[group_dim], overall_avg)
                                                condition_expr = formula_modified
                                                for col in df.columns:
                                                    condition_expr = condition_expr.replace(col, str(local_vars.get(col, 0)))
                                                condition_expr = condition_expr.replace(str(overall_avg), str(group_value))
                                                return eval(condition_expr, {"__builtins__": None}, {})
                                            else:
                                                return eval(formula_modified, {"__builtins__": None}, local_vars)

                                        result = working_df.apply(evaluate_row, axis=1)
                                        if result is not None:
                                            df[new_field_name] = result
                                            st.session_state.dataset = df
                                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                                            st.session_state.chart_cache = {}  # Clear chart cache
                                            st.session_state.insights_cache = {}  # Clear insights cache
                                            if pd.api.types.is_numeric_dtype(df[new_field_name]):
                                                if "measure" in st.session_state.field_types:
                                                    st.session_state.field_types["measure"].append(new_field_name)
                                            else:
                                                if "dimension" in st.session_state.field_types:
                                                    st.session_state.field_types["dimension"].append(new_field_name)
                                            save_dataset_changes()
                                            st.session_state.sample_prompts = []
                                            st.session_state.used_sample_prompts = []
                                            st.session_state.sample_prompt_pool = []
                                            st.session_state.last_used_pool_index = 0
                                            st.success(f"New field {new_field_name} created!")
                                            #logger.info("Created new calculated field %s with formula: %s", new_field_name, formula)
                                        else:
                                            st.error("Failed to evaluate the formula.")
                                            logger.error("Formula evaluation returned None for prompt: %s", calc_prompt)
                                    except Exception as e:
                                        st.error(f"Error evaluating formula: {str(e)}")
                                        logger.error("Failed to evaluate formula: %s", str(e))


    with tab4:
        st.session_state.onboarding_seen = True
        if st.session_state.dataset is not None:
            executive_summary_tab(st.session_state.dataset)
        else:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab to view the executive summary.")

    
    with tab5:
            st.session_state.onboarding_seen = True
            st.subheader("Saved Dashboards")
    
            if "user_id" not in st.session_state or st.session_state.user_id is None:
                st.error("Please log in to view dashboards.")
            else:
                # Initialize session state for navigation
                if "dashboard_view" not in st.session_state:
                    st.session_state.dashboard_view = "projects"  # Options: "projects", "dashboards", "dashboard"
                if "selected_project" not in st.session_state:
                    st.session_state.selected_project = None
                if "selected_dashboard" not in st.session_state:
                    st.session_state.selected_dashboard = None
                if "breadcrumb" not in st.session_state:
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                if "dashboards_cache" not in st.session_state:
                    st.session_state.dashboards_cache = None
                if "refresh_dashboards" not in st.session_state:
                    st.session_state.refresh_dashboards = False
    
                # Load dashboards before query parameter handling
                page_size = 10
                if "dashboard_page" not in st.session_state:
                    st.session_state.dashboard_page = 0
    
                if st.session_state.dashboards_cache is None or st.session_state.refresh_dashboards:
                    st.session_state.dashboards_cache = load_dashboards(
                        supabase, st.session_state.user_id, st.session_state, limit=page_size, offset=st.session_state.dashboard_page * page_size
                    )
                    st.session_state.refresh_dashboards = False
    
                dashboards = st.session_state.dashboards_cache
    
                # Handle query parameters for navigation
                view_param = st.query_params.get("view", "projects")
                project_id_param = st.query_params.get("project_id")
                dashboard_id_param = st.query_params.get("dashboard_id")
    
                # Update session state based on query parameters
                if view_param != st.session_state.dashboard_view or project_id_param != st.session_state.selected_project or dashboard_id_param != st.session_state.selected_dashboard:
                    st.session_state.dashboard_view = view_param
                    st.session_state.selected_project = project_id_param
                    st.session_state.selected_dashboard = dashboard_id_param
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                    if view_param == "dashboards" and project_id_param:
                        st.session_state.breadcrumb.append({"label": project_id_param, "view": "dashboards", "project_id": project_id_param})
                    elif view_param == "dashboard" and project_id_param and dashboard_id_param:
                        # Check if dashboards is defined and not empty
                        if dashboards is not None and not dashboards.empty:
                            dashboard_name = dashboards[dashboards["id"] == dashboard_id_param]["name"].iloc[0] if not dashboards[dashboards["id"] == dashboard_id_param].empty else dashboard_id_param
                        else:
                            dashboard_name = dashboard_id_param
                        st.session_state.breadcrumb.append({"label": project_id_param, "view": "dashboards", "project_id": project_id_param})
                        st.session_state.breadcrumb.append({"label": dashboard_name, "view": "dashboard", "project_id": project_id_param, "dashboard_id": dashboard_id_param})
    
                # Custom CSS for Tableau-like styling and breadcrumb buttons
                st.markdown("""
                <style>
                .project-container, .dashboard-container, .chart-container {
                    border: 1px solid #475569;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    background-color: #1F2A44;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                .project-container:hover, .dashboard-container:hover {
                    background-color: #334155;
                }
                .project-title, .dashboard-title, .chart-title {
                    font-size: 1.2em;
                    font-weight: 600;
                    color: #FFFFFF;
                    margin-bottom: 0.5rem;
                }
                .project-info, .dashboard-info {
                    font-size: 0.9em;
                    color: #A0AEC0;
                }
                .breadcrumb-container {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 1rem;
                    font-size: 0.9em;
                }
                .breadcrumb-button {
                    background: none !important;
                    border: none !important;
                    color: #3B82F6 !important;
                    text-decoration: none !important;
                    cursor: pointer !important;
                    padding: 0 !important;
                    font-size: 0.9em !important;
                }
                .breadcrumb-button:hover {
                    text-decoration: underline !important;
                }
                .breadcrumb-separator {
                    color: #A0AEC0;
                }
                .breadcrumb-current {
                    color: #A0AEC0;
                    font-size: 0.9em;
                }
                .filter-container {
                    background-color: #1F2A44;
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                }
                .action-button {
                    background-color: #26A69A !important;
                    color: white !important;
                    border-radius: 0.5rem !important;
                    padding: 0.5rem !important;
                    border: none !important;
                    cursor: pointer !important;
                }
                .action-button:hover {
                    background-color: #2E7D32 !important;
                }
                .dataset-status {
                    padding: 0.5rem;
                    border-radius: 4px;
                    margin-bottom: 1rem;
                    font-size: 0.9em;
                }
                .dataset-restored {
                    background-color: #059669;
                    color: white;
                }
                .dataset-sample {
                    background-color: #0891b2;
                    color: white;
                }
                .dataset-missing {
                    background-color: #dc2626;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)
   
    
                # Handle view switching
                if st.session_state.dashboard_view == "projects":
                    render_breadcrumb()
    
                    # Search and filters
                    with st.container():
                        st.markdown("### Filter Projects and Dashboards")
                        with st.container():
                            search_query = st.text_input("Search by name, prompt, or tag", key="dashboard_search")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                project_filter = st.multiselect("Project", options=sorted([p for p in dashboards["project_id"].unique() if p is not None]), key="project_filter")
                            with col2:
                                type_filter = st.multiselect("Analytics Type", options=["Sales", "Customer", "Product", "Other"], key="type_filter")
                            with col3:
                                date_filter = st.date_input("Created After", value=None, key="date_filter")
                            with col4:
                                tag_filter = st.text_input("Tags (comma-separated)", key="tag_filter")
    
                            filtered_dashboards = filter_dashboards(dashboards, search_query, project_filter, type_filter, date_filter, tag_filter, supabase)
    
                    # Pagination controls
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button("Previous", disabled=st.session_state.dashboard_page == 0):
                            st.session_state.dashboard_page -= 1
                            st.session_state.refresh_dashboards = True
                    with col2:
                        st.write(f"Page {st.session_state.dashboard_page + 1}")
                    with col3:
                        if st.button("Next", disabled=len(filtered_dashboards) < page_size):
                            st.session_state.dashboard_page += 1
                            st.session_state.refresh_dashboards = True
    
                    if st.button("Refresh Dashboards", key="refresh_dashboards_btn"):
                        st.session_state.refresh_dashboards = True
                        st.rerun()
    
                    # Display projects
                    if dashboards.empty:
                        st.markdown("No projects or dashboards found.")
                    else:
                        project_ids = sorted([p for p in dashboards["project_id"].unique() if p is not None])
                        for project_id in project_ids:
                            project_dashboards = filtered_dashboards[filtered_dashboards["project_id"] == project_id]
                            with st.container():
                                st.markdown(f"<div class='project-container'>", unsafe_allow_html=True)
                                st.markdown(f"<div class='project-title'>{project_id}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='project-info'>Dashboards: {len(project_dashboards)}</div>", unsafe_allow_html=True)
                                if st.button(f"View Dashboards", key=f"view_project_{project_id}"):
                                    st.session_state.dashboard_view = "dashboards"
                                    st.session_state.selected_project = project_id
                                    st.session_state.breadcrumb.append({"label": project_id, "view": "dashboards", "project_id": project_id})
                                    st.query_params = {"view": "dashboards", "project_id": project_id}
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
    
                elif st.session_state.dashboard_view == "dashboards":
                    render_breadcrumb()
                    project_id = st.session_state.selected_project
                    project_dashboards = dashboards[dashboards["project_id"] == project_id]
    
                    if project_dashboards.empty:
                        st.markdown(f"No dashboards found for project '{project_id}'.")
                        if st.button("Back to Projects", key="back_to_projects"):
                            st.session_state.dashboard_view = "projects"
                            st.session_state.selected_project = None
                            st.query_params = {"view": "projects"}
                            st.rerun()
                    else:
                        st.markdown(f"### Dashboards in {project_id}")
                        for _, dashboard in project_dashboards.iterrows():
                            dashboard_id = dashboard["id"]
                            dashboard_name = dashboard["name"]
                            created_at = dashboard["created_at"]
                            tags = dashboard.get("tags", [])
                            with st.container():
                                st.markdown(f"<div class='dashboard-container'>", unsafe_allow_html=True)
                                st.markdown(f"<div class='dashboard-title'>{dashboard_name}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='dashboard-info'>Created: {created_at} | Tags: {', '.join(tags) if tags else 'None'}</div>", unsafe_allow_html=True)
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    if st.button(f"View Dashboard", key=f"view_dashboard_{dashboard_id}"):
                                        st.session_state.dashboard_view = "dashboard"
                                        st.session_state.selected_dashboard = dashboard_id
                                        st.session_state.breadcrumb.append({"label": dashboard_name, "view": "dashboard", "project_id": project_id, "dashboard_id": dashboard_id})
                                        st.query_params = {"view": "dashboard", "project_id": project_id, "dashboard_id": dashboard_id}
                                        st.rerun()
                                with col2:
                                    if st.button("Delete", key=f"delete_dashboard_{dashboard_id}"):
                                        supabase.table("dashboards").delete().eq("id", dashboard_id).execute()
                                        st.session_state.refresh_dashboards = True
                                        st.success(f"Deleted dashboard '{dashboard_name}'")
                                        st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
    
                elif st.session_state.dashboard_view == "dashboard":
                    render_breadcrumb()
                    dashboard_id = st.session_state.selected_dashboard
                    project_id = st.session_state.selected_project
                    dashboard_data = dashboards[dashboards["id"] == dashboard_id]
                    
                    if dashboard_data.empty:
                        st.error(f"Dashboard with ID {dashboard_id} not found.")
                        if st.button("Back to Dashboards", key="back_to_dashboards"):
                            st.session_state.dashboard_view = "dashboards"
                            st.session_state.selected_dashboard = None
                            st.query_params = {"view": "dashboards", "project_id": project_id}
                            st.rerun()
                    else:
                        dashboard_name = dashboard_data["name"].iloc[0]
                        st.markdown(f"### Dashboard: {dashboard_name}")
                        
                        # Use enterprise dashboard renderer
                        render_dashboard_enterprise(supabase, dashboard_id, project_id, dashboard_name)
    
                        # Dashboard settings
                        with st.expander("⚙️ Dashboard Settings", expanded=False):
                            with st.form(key=f"settings_{dashboard_id}"):
                                new_name = st.text_input("Rename Dashboard", value=dashboard_name)
                                new_tags = st.text_input("Add Tags (comma-separated)")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.form_submit_button("💾 Apply Changes"):
                                        updates = {}
                                        if new_name != dashboard_name:
                                            updates["name"] = new_name
                                        if new_tags:
                                            tag_list = [tag.strip() for tag in new_tags.split(",")]
                                            existing_tags = dashboard_data["tags"].iloc[0] or []
                                            updates["tags"] = list(set(existing_tags + tag_list))
                                        
                                        if updates:
                                            updates["updated_at"] = datetime.utcnow().isoformat()
                                            supabase.table("dashboards").update(updates).eq("id", dashboard_id).execute()
                                            st.success("Dashboard updated!")
                                            st.rerun()
                                
                                with col2:
                                    if st.form_submit_button("🗑️ Delete Dashboard"):
                                        supabase.table("dashboards").delete().eq("id", dashboard_id).execute()
                                        st.success("Dashboard deleted!")
                                        st.session_state.dashboard_view = "dashboards"
                                        st.rerun()
    
                # Reset navigation button for debugging
                if st.button("Reset Navigation", key="reset_navigation"):
                    st.session_state.dashboard_view = "projects"
                    st.session_state.selected_project = None
                    st.session_state.selected_dashboard = None
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                    st.query_params = {"view": "projects"}
                    st.rerun()



    with tab6:
        st.session_state.onboarding_seen = True

        recommended_charts_insights_tab()