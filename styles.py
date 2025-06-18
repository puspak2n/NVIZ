# style.py
import streamlit as st

def load_custom_css():
    """Load custom CSS styles."""
    st.markdown("""
        <style>
        /* Main container styles */
        .main {
            max-width: 1200px;
            padding: 1rem;
        }
        
        /* Block container styles */
        .block-container {
            max-width: 1200px;
            padding: 1rem;
        }
        
        /* Toolbar styles */
        .stToolbar {
            padding: 0.5rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Button styles */
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            padding: 0.5rem;
            font-weight: 500;
        }
        
        /* Chart container styles */
        .chart-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Data table styles */
        .stDataFrame {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Expander styles */
        .streamlit-expanderHeader {
            font-size: 1.1rem;
            font-weight: 500;
            color: #262730 !important;
            background-color: #f0f2f6 !important;
            padding: 0.5rem;
            border-radius: 0.5rem;
        }
        
        /* Insights and View Chart Data expander styles */
        .streamlit-expanderHeader[data-testid="stExpander"] {
            color: #262730 !important;
            background-color: #f0f2f6 !important;
        }
        
        /* Markdown text color */
        .stMarkdown {
            color: #262730 !important;
        }
        
        /* Metric value styles */
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Chart title styles */
        .chart-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #262730 !important;
            margin-bottom: 1rem;
        }
        
        /* Chart description styles */
        .chart-description {
            font-size: 0.9rem;
            color: #666666 !important;
            margin-bottom: 1rem;
        }
        
        /* Chart controls styles */
        .chart-controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Chart type selector styles */
        .chart-type-selector {
            min-width: 150px;
        }
        
        /* Chart actions styles */
        .chart-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        /* Insights and View Chart Data expander styles */
        .streamlit-expanderHeader[data-testid="stExpander"] {
            color: #262730 !important;
            background-color: #f0f2f6 !important;
        }
        
        /* Insights text color */
        .stMarkdown p {
            color: #262730 !important;
        }
        
        /* View Chart Data text color */
        .stDataFrame {
            color: #262730 !important;
        }
        
        /* Basic Statistics text color */
        .stMetric {
            color: #262730 !important;
        }
        
        /* Chart title text color */
        .chart-title {
            color: #262730 !important;
        }
        
        /* Chart description text color */
        .chart-description {
            color: #666666 !important;
        }

        /* Insights and View Chart Data label styles */
        .stMarkdown h3 {
            color: #262730 !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }

        /* Insights and View Chart Data expander header text */
        .streamlit-expanderHeader p {
            color: #262730 !important;
            font-weight: 500 !important;
        }

        /* Insights and View Chart Data expander content */
        .streamlit-expanderContent {
            color: #262730 !important;
        }

        /* Project name styles */
        .stButton button {
            color: #262730 !important;
        }

        /* Prompt text styles */
        .stTextInput input {
            color: #262730 !important;
        }

        /* Label text styles */
        .stTextInput label {
            color: #262730 !important;
        }

        /* Override any white text colors */
        .stMarkdown, .stTextInput, .stButton, .stExpander, .stDataFrame, .stMetric {
            color: #262730 !important;
        }

        /* Ensure all text elements have proper contrast */
        * {
            color: #262730 !important;
        }

        /* Override any specific white text */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        .stMarkdown p, .stMarkdown li, .stMarkdown a,
        .stTextInput label, .stTextInput input,
        .stButton button,
        .stExpander .streamlit-expanderHeader,
        .stDataFrame,
        .stMetric {
            color: #262730 !important;
        }
        </style>
    """, unsafe_allow_html=True)