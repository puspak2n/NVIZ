import re
import logging
import pandas as pd
import streamlit as st
import hashlib
import time
from utils import setup_logging
from calc_utils import detect_outliers
import plotly.express as px
from collections import OrderedDict
import openai
import json
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pio.templates.default = "plotly_dark"

logger = setup_logging()

# Define USE_OPENAI based on openai.api_key
USE_OPENAI = openai.api_key is not None

CHART_THEMES = {
    "corporate_blue": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "accent": "#2ca02c",
        "background": "#f8f9fa",
        "grid": "#e1e5e9",
        "text": "#2c3e50"
    },
    "modern_dark": {
        "primary": "#00d4ff",
        "secondary": "#ff6b6b", 
        "accent": "#4ecdc4",
        "background": "#1a1a1a",
        "grid": "#333333",
        "text": "#ffffff"
    },
    "business_green": {
        "primary": "#27ae60",
        "secondary": "#e74c3c",
        "accent": "#f39c12",
        "background": "#ffffff",
        "grid": "#ecf0f1",
        "text": "#2c3e50"
    },
    "vibrant": {
        "primary": "#e91e63",
        "secondary": "#9c27b0",
        "accent": "#ff9800",
        "background": "#fafafa",
        "grid": "#e0e0e0", 
        "text": "#212121"
    }
}

# Color sequences for multi-category charts
COLOR_SEQUENCES = {
    "corporate": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "modern": ["#00d4ff", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94", "#b4a7d6", "#d4a574"],
    "professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83", "#F2AF29", "#4ECDC4", "#556B8D"],
    "pastel": ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D4A4EB", "#FFB3E6", "#C9BAFF"]
}

def format_number(value, format_type="auto"):
    """Format numbers for better readability."""
    if pd.isna(value) or value == 0:
        return "0"
    
    abs_value = abs(value)
    
    if format_type == "currency":
        if abs_value >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        elif abs_value >= 1_000:
            return f"${value/1_000:.1f}K" 
        else:
            return f"${value:.0f}"
    elif format_type == "percent":
        return f"{value:.1f}%"
    elif format_type == "auto":
        if abs_value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif abs_value >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"
    else:
        return f"{value:.2f}"

def apply_chart_theme(fig, theme_name="corporate_blue", title="", subtitle=""):
    """Apply professional styling theme to any Plotly figure."""
    theme = CHART_THEMES.get(theme_name, CHART_THEMES["corporate_blue"])
    
    # Main title and subtitle
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:14px; color:{theme['text']}80'>{subtitle}</span>"
    
    fig.update_layout(
        # Typography
        title={
            'text': title_text,
            'x': 0.02,
            'y': 0.98,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 24, 'family': 'Arial Black, sans-serif', 'color': theme['text']}
        },
        font=dict(family="Arial, sans-serif", size=12, color=theme['text']),
        
        # Background and layout
        plot_bgcolor=theme['background'],
        paper_bgcolor=theme['background'],
        margin=dict(l=60, r=40, t=80, b=60),
        
        # Grid and axes
        xaxis=dict(
            gridcolor=theme['grid'],
            gridwidth=1,
            tickfont=dict(size=11),
            title_font=dict(size=13, color=theme['text'])
        ),
        yaxis=dict(
            gridcolor=theme['grid'], 
            gridwidth=1,
            tickfont=dict(size=11),
            title_font=dict(size=13, color=theme['text'])
        ),
        
        # Legend
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=theme['grid'],
            borderwidth=1,
            font=dict(size=11),
            x=1.02,
            y=1,
            xanchor='left'
        ),
        
        # Hover styling
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor=theme['primary']
        )
    )
    
    return fig

def create_styled_bar_chart(chart_data, metric, dimension, theme="corporate_blue", top_n=None):
    """Create a professionally styled bar chart."""
    
    # Prepare data
    df = chart_data.copy()
    if top_n:
        df = df.head(top_n)
    
    # Create base chart
    fig = px.bar(
        df, 
        x=dimension, 
        y=metric,
        color_discrete_sequence=[CHART_THEMES[theme]["primary"]]
    )
    
    # Add value labels on bars
    fig.update_traces(
        texttemplate='%{y}',
        textposition='outside',
        textfont_size=11,
        hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:,.0f}}<extra></extra>'
    )
    
    # Style the chart
    title = f"{metric} by {dimension}"
    subtitle = f"Top {top_n} results" if top_n else f"Total: {format_number(df[metric].sum())}"
    fig = apply_chart_theme(fig, theme, title, subtitle)
    
    # Rotate x-axis labels if needed
    if len(df) > 10 or df[dimension].astype(str).str.len().max() > 15:
        fig.update_xaxes(tickangle=45)
    
    return fig

def create_styled_line_chart(chart_data, metric, dimension, secondary_dimension=None, theme="corporate_blue"):
    """Create a professionally styled line chart."""
    
    df = chart_data.copy()
    
    if secondary_dimension:
        # Multi-line chart
        fig = px.line(
            df, 
            x=dimension, 
            y=metric, 
            color=secondary_dimension,
            color_discrete_sequence=COLOR_SEQUENCES["professional"]
        )
        title = f"{metric} Trend by {secondary_dimension}"
    else:
        # Single line chart
        fig = px.line(
            df, 
            x=dimension, 
            y=metric,
            color_discrete_sequence=[CHART_THEMES[theme]["primary"]]
        )
        title = f"{metric} Over Time"
    
    # Enhanced line styling
    fig.update_traces(
        line=dict(width=3),
        mode='lines+markers',
        marker=dict(size=6),
        hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:,.0f}}<extra></extra>'
    )
    
    # Add trend annotation
    if len(df) > 1:
        start_val = df[metric].iloc[0]
        end_val = df[metric].iloc[-1]
        change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
        trend_text = f"{'↗' if change_pct > 0 else '↘'} {abs(change_pct):.1f}% change"
        subtitle = f"Period trend: {trend_text}"
    else:
        subtitle = ""
    
    fig = apply_chart_theme(fig, theme, title, subtitle)
    
    # Format x-axis for dates
    if pd.api.types.is_datetime64_any_dtype(df[dimension]):
        fig.update_xaxes(tickformat='%b %Y')
    
    return fig

def create_styled_scatter_chart(chart_data, metric, second_metric, dimension, theme="corporate_blue"):
    """Create a professionally styled scatter plot."""
    
    df = chart_data.copy()
    
    fig = px.scatter(
        df,
        x=metric,
        y=second_metric, 
        color=dimension,
        size_max=15,
        color_discrete_sequence=COLOR_SEQUENCES["professional"]
    )
    
    # Enhanced scatter styling
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1, color='white')),
        hovertemplate=f'<b>%{{color}}</b><br>{metric}: %{{x:,.0f}}<br>{second_metric}: %{{y:,.0f}}<extra></extra>'
    )
    
    # Add correlation info
    correlation = df[[metric, second_metric]].corr().iloc[0, 1]
    corr_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
    corr_direction = "positive" if correlation > 0 else "negative"
    
    title = f"{metric} vs {second_metric}"
    subtitle = f"{corr_strength} {corr_direction} correlation (r={correlation:.2f})"
    
    fig = apply_chart_theme(fig, theme, title, subtitle)
    
    return fig

def create_styled_pie_chart(chart_data, metric, dimension, theme="corporate_blue", show_percentages=True):
    """Create a professionally styled pie chart."""
    
    df = chart_data.copy()
    
    # Limit to top categories to avoid clutter
    if len(df) > 8:
        top_7 = df.head(7)
        others_sum = df.tail(len(df) - 7)[metric].sum()
        others_row = pd.DataFrame({dimension: ["Others"], metric: [others_sum]})
        df = pd.concat([top_7, others_row], ignore_index=True)
    
    fig = px.pie(
        df,
        values=metric,
        names=dimension,
        color_discrete_sequence=COLOR_SEQUENCES["professional"]
    )
    
    # Enhanced pie styling
    if show_percentages:
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=11,
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )
    else:
        fig.update_traces(
            textposition='inside', 
            textinfo='label',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<extra></extra>'
        )
    
    title = f"{metric} Distribution by {dimension}"
    subtitle = f"Total: {format_number(df[metric].sum())}"
    
    fig = apply_chart_theme(fig, theme, title, subtitle)
    
    return fig

def add_data_labels(fig, chart_type, format_type="auto"):
    """Add formatted data labels to charts."""
    
    if chart_type == "Bar":
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )
    elif chart_type == "Line":
        fig.update_traces(
            mode='lines+markers+text',
            textposition="top center"
        )
    
    return fig

def add_annotations(fig, annotations_list):
    """Add custom annotations to highlight insights."""
    
    for annotation in annotations_list:
        fig.add_annotation(
            x=annotation.get('x', 0.5),
            y=annotation.get('y', 0.95),
            xref="paper" if annotation.get('x_paper', True) else "x",
            yref="paper" if annotation.get('y_paper', True) else "y", 
            text=annotation['text'],
            showarrow=annotation.get('arrow', False),
            font=dict(size=12, color=annotation.get('color', '#666666')),
            bgcolor=annotation.get('bgcolor', 'rgba(255,255,255,0.8)'),
            bordercolor=annotation.get('bordercolor', '#cccccc'),
            borderwidth=1
        )
    
    return fig

# UPDATED render_chart function with styling
def create_styled_chart(chart_data, metric, dimension, chart_type, secondary_dimension=None, theme="corporate_blue", **kwargs):
    """Main function to create styled charts based on type."""
    
    if chart_type == "Bar":
        fig = create_styled_bar_chart(
            chart_data, metric, dimension, theme, 
            top_n=kwargs.get('top_n')

        )
    elif chart_type == "Line":
        fig = create_styled_line_chart(
            chart_data, metric, dimension, secondary_dimension, theme
        )
    elif chart_type == "Scatter":
        second_metric = kwargs.get('second_metric')
        if second_metric:
            fig = create_styled_scatter_chart(
                chart_data, metric, second_metric, dimension, theme
            )
    elif chart_type == "Pie":
        fig = create_styled_pie_chart(
            chart_data, metric, dimension, theme
        )
    else:
        # Fallback to basic styling
        fig = px.bar(chart_data, x=dimension, y=metric)
        fig = apply_chart_theme(fig, theme, f"{metric} by {dimension}")
    
    # Add insights as annotations if provided
    if 'insights' in kwargs:
        insights = kwargs['insights'][:2]  # Show top 2 insights
        annotations = []
        for i, insight in enumerate(insights):
            annotations.append({
                'x': 0.02,
                'y': 0.02 + (i * 0.05),
                'text': f"💡 {insight}",
                'bgcolor': 'rgba(255,255,255,0.9)',
                'font': {'size': 10}
            })
        fig = add_annotations(fig, annotations)
    
    return fig

# Theme selector for Streamlit
def get_theme_selector():
    """Return theme options for Streamlit selectbox."""
    return {
        "Corporate Blue": "corporate_blue",
        "Modern Dark": "modern_dark", 
        "Business Green": "business_green",
        "Vibrant": "vibrant"
    }

def find_column(name, candidates, df, exclude=[]):
    """Fuzzy match a column name from candidates."""
    name_lower = name.strip().lower()
    logger.info(f"Finding column for name: {name_lower} in candidates: {candidates}")
    
    # Handle common plural forms
    singular_forms = {
        'cities': 'city',
        'states': 'state',
        'countries': 'country',
        'regions': 'region',
        'categories': 'category',
        'sub-categories': 'sub-category',
        'products': 'product',
        'customers': 'customer',
        'segments': 'segment'
    }
    
    # Try singular form if the input is plural
    if name_lower in singular_forms:
        name_lower = singular_forms[name_lower]
    
    # First try exact match
    for col in candidates:
        if col.lower() == name_lower and col not in exclude:
            logger.info(f"Found exact match: {col}")
            return col
    
    # Then try contains match
    for col in candidates:
        if name_lower in col.lower() and col not in exclude:
            logger.info(f"Found contains match: {col}")
            return col
    
    # If no match found, log the failure and return None
    logger.warning(f"No match found for {name_lower} in candidates: {candidates}")
    return None

# Safe helper functions that won't break if columns don't exist
def find_column_safe(name, columns, df, exclude=None):
    """Safely find matching column name with fuzzy matching."""
    if not name or not columns:
        return None
    
    name_clean = name.strip().lower()
    exclude = exclude or []
    
    # Exact match first
    for col in columns:
        if col.lower() == name_clean and col not in exclude:
            return col
    
    # Partial match
    for col in columns:
        if name_clean in col.lower() and col not in exclude:
            return col
    
    # Check if it's in the dataframe columns
    for col in df.columns:
        if col.lower() == name_clean and col not in exclude:
            return col
    
    return None

def find_best_date_column(date_columns, df):
    """Find the best date column from available options."""
    if not date_columns:
        return None
    
    # Prefer "Order Date" if it exists
    for col in date_columns:
        if "order" in col.lower() and "date" in col.lower():
            return col
    
    # Otherwise return the first date column
    return date_columns[0] if date_columns else None

def find_date_column(name, date_candidates):
    for col in date_candidates:
        if name.lower() in col.lower():
            return col
    return date_candidates[0] if date_candidates else None

def detect_time_aggregation(name):
    if "year" in name.lower():
        return "year"
    elif "quarter" in name.lower():
        return "quarter"
    elif "month" in name.lower():
        return "month"
    else:
        return "month"

def parse_filter(filter_part, dimensions, measures, df):
    for col in dimensions + measures:
        if col.lower() in filter_part.lower():
            value = filter_part.lower().replace(col.lower(), "").replace("=", "").strip()
            unique_vals = df[col].dropna().astype(str).unique()
            for v in unique_vals:
                if value.lower() in str(v).lower():
                    return col, v
            return col, value
    return None, None

def preprocess_prompt(prompt):
    """Clean and normalize the prompt by removing common action words."""
    # Remove common action prefixes
    action_patterns = [
        r'^show\s+me\s+',
        r'^build\s+a?\s+',
        r'^create\s+(the\s+|a\s+)?',
        r'^plot\s+a?\s+',
        r'^make\s+a?\s+',
        r'^generate\s+a?\s+',
        r'^display\s+',
        r'^visualize\s+',
        r'^draw\s+a?\s+'
    ]
    
    cleaned = prompt.strip()
    for pattern in action_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned

def extract_implicit_filters(prompt, dimensions, measures, df):
    """Extract filters that are implied in natural language without explicit 'where' clause."""
    filters = []
    
    # Skip implicit filter detection for basic "X by Y" patterns
    basic_pattern = r'^[a-zA-Z0-9_\s]+\s+by\s+[a-zA-Z0-9_\s]+$'
    if re.match(basic_pattern, prompt.strip(), re.IGNORECASE):
        return filters  # Return empty - no implicit filters for basic queries
    
    # Only look for explicit filters like "for USA", "in Europe"
    implicit_patterns = [
        r'\s+for\s+([a-zA-Z0-9_\s]+)$',
        r'\s+in\s+([a-zA-Z0-9_\s]+)(?:\s+by|\s*$)',
        r'\s+from\s+([a-zA-Z0-9_\s]+)(?:\s+by|\s*$)',
    ]
    
    for pattern in implicit_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            potential_filter = match.group(1).strip()
            filter_info = find_implicit_filter_column(potential_filter, dimensions, df)
            if filter_info:
                filters.append(filter_info)
    
    return filters


def find_implicit_filter_column(value, dimensions, df):
    """Find which column this value might belong to."""
    value_clean = value.strip().lower()
    
    # Check each dimension column for this value
    for dim in dimensions:
        if dim in df.columns:
            # Check if this value exists in the column
            unique_values = df[dim].astype(str).str.lower().str.strip().unique()
            if value_clean in unique_values:
                return {
                    'column': dim,
                    'value': value.strip(),
                    'operator': '='
                }
    
    return None

def extract_sentiment_and_intent(prompt):
    """Analyze the prompt for chart type hints and sentiment."""
    prompt_lower = prompt.lower()
    
    # Chart type indicators
    chart_hints = {
        'line': ['trend', 'over time', 'change', 'growth', 'decline', 'timeline', 'progression'],
        'bar': ['compare', 'comparison', 'top', 'bottom', 'ranking', 'highest', 'lowest', 'best', 'worst'],
        'scatter': ['vs', 'versus', 'against', 'correlation', 'relationship', 'compare'],
        'pie': ['proportion', 'percentage', 'share', 'distribution', 'breakdown', 'composition'],
        'map': ['by country', 'by region', 'geographic', 'location', 'worldwide', 'global'],
        'bubble': ['bubble', 'sized by', 'size represents']
    }
    
    detected_type = None
    max_score = 0
    
    for chart_type, keywords in chart_hints.items():
        score = sum(1 for keyword in keywords if keyword in prompt_lower)
        if score > max_score:
            max_score = score
            detected_type = chart_type
    
    # Time-based indicators
    time_indicators = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'annual', 
                      'over time', 'trend', 'by month', 'by year', 'by quarter']
    has_time_component = any(indicator in prompt_lower for indicator in time_indicators)
    
    # Comparison indicators
    comparison_indicators = ['vs', 'versus', 'against', 'compared to', 'compare']
    has_comparison = any(indicator in prompt_lower for indicator in comparison_indicators)
    
    return {
        'suggested_chart_type': detected_type,
        'has_time_component': has_time_component,
        'has_comparison': has_comparison,
        'max_score': max_score
    }



def original_rule_based_parse(prompt, df, dimensions, measures, dates):
    logger.info(f"Rule-based parsing prompt: {prompt}")
    logger.info(f"Available dimensions: {dimensions}")
    logger.info(f"Available measures: {measures}")
    prompt_lower = prompt.lower()
    chart_type = None
    metric = None
    dimension = None
    second_metric = None
    filter_col = None
    filter_val = None
    kwargs = {}
    is_two_metric = False
    exclude_list = []
    secondary_dimension = None

    patterns = OrderedDict([
        ("trend_over_time", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]*(?:date|month|year|quarter))\s*$"),
        ("trend_by_group", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]*(?:date|month|year|quarter))\s+and\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("compare_metrics", r"^\s*([a-zA-Z0-9_\s]+)\s+vs\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("top_n", r"^\s*top\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("bottom_n", r"^\s*bottom\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("top_n_filter", r"^\s*top\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+(.+)$"),
        ("map_chart", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+country\s*$"),
        ("outliers", r"^\s*(?:show|find)\s+outliers\s+in\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("filter_category", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+([a-zA-Z0-9_\s]+)\s*=\s*(.+)$"),
        ("filter_value", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+([a-zA-Z0-9_\s]+)\s*(>=|<=|>|<|=)\s*(\d+\.?\d*)\s*$"),
        ("basic_group", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
    ])

    if "bubble" in prompt_lower and "sized by" in prompt_lower:
        parts = prompt_lower.replace("bubble cloud of", "").strip().split("sized by")
        if len(parts) == 2:
            dim_candidate = parts[0].strip()
            metric_candidate = parts[1].strip()
            dimension = find_column(dim_candidate, dimensions, df)
            metric = find_column(metric_candidate, measures, df)
            if dimension and metric:
                chart_type = "BubbleCloud"
                return chart_type, metric, dimension, None, None, None, kwargs, False, [], None
    for pattern_name, pattern in patterns.items():
        match = re.match(pattern, prompt_lower)
        if match:
            logger.info(f"Matched pattern: {pattern_name}")
            groups = match.groups()
            logger.info(f"Matched groups: {groups}")

            if pattern_name == "trend_over_time":
                metric_name, date_field = groups
                chart_type = "Line"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_date_column(date_field.strip(), dates)
                kwargs["time_aggregation"] = detect_time_aggregation(date_field)
                kwargs["sort_by_date"] = True

            elif pattern_name == "trend_by_group":
                metric_name, date_field, dim_name = groups
                chart_type = "Line"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_date_column(date_field.strip(), dates)
                secondary_dimension = find_column(dim_name.strip(), dimensions, df)
                kwargs["time_aggregation"] = detect_time_aggregation(date_field)
                kwargs["sort_by_date"] = True

            elif pattern_name == "compare_metrics":
                metric1_name, metric2_name, dim_name = groups
                chart_type = "Scatter"
                is_two_metric = True
                metric = find_column(metric1_name.strip(), measures, df)
                second_metric = find_column(metric2_name.strip(), measures, df, exclude=[metric])
                dimension = find_column(dim_name.strip(), dimensions, df)

            elif pattern_name in ["top_n", "bottom_n"]:
                n, dim_name, metric_name = groups
                logger.info(f"Processing top/bottom n: n={n}, dim_name={dim_name}, metric_name={metric_name}")
                chart_type = "Bar"
                n = int(n) if n else 5
                dimension = find_column(dim_name.strip(), dimensions, df)
                metric = find_column(metric_name.strip(), measures, df)

                if not dimension:
                    dimension = find_date_column(dim_name.strip(), dates)

                logger.info(f"Found dimension: {dimension}, metric: {metric}")
                if not dimension:
                    logger.error(f"Could not find dimension column for: {dim_name}")
                    return None
                if not metric:
                    logger.error(f"Could not find metric column for: {metric_name}")
                    return None

                if dim_name.strip().lower().endswith(("month", "year", "quarter")):
                    kwargs["time_aggregation"] = detect_time_aggregation(dim_name)
                    kwargs["sort_by_date"] = True

                kwargs["top_n"] = n
                kwargs["is_bottom"] = pattern_name == "bottom_n"

            elif pattern_name == "top_n_filter":
                n, dim_name, metric_name, filter_part = groups
                chart_type = "Bar"
                n = int(n) if n else 5
                dimension = find_column(dim_name.strip(), dimensions, df)
                metric = find_column(metric_name.strip(), measures, df)
                if not dimension or not metric:
                    return None
                kwargs["top_n"] = n
                filter_col, filter_val = parse_filter(filter_part, dimensions, measures, df)

            elif pattern_name == "map_chart":
                metric_name = groups[0]
                chart_type = "Map"
                dimension = "Country"
                metric = find_column(metric_name.strip(), measures, df)
                if "Country" not in dimensions:
                    chart_type = "Bar"
                    logger.warning("Country not in dimensions, fallback to Bar chart.")

            elif pattern_name == "outliers":
                metric_name, dim_name = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                if not dimension or not metric:
                    return None
                kwargs["show_outliers"] = True

            elif pattern_name == "filter_category":
                metric_name, dim_name, filter_col_name, filter_val_raw = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                filter_col = find_column(filter_col_name.strip(), dimensions, df)
                if not dimension or not metric or not filter_col:
                    return None
                filter_val = filter_val_raw.strip()

            elif pattern_name == "filter_value":
                metric_name, dim_name, filter_col_name, operator, value = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                filter_col = find_column(filter_col_name.strip(), dimensions, df)
                if not dimension or not metric or not filter_col:
                    return None
                filter_val = f"{operator}{value}"

            elif pattern_name == "basic_group":
                metric_name, dim_name = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions + dates, df)
            
                if not metric or not dimension:
                    logger.warning(f"basic_group failed: metric={metric}, dimension={dimension}")
                    return None


            return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension

    if not chart_type:
        logger.warning("No pattern matched for prompt: %s", prompt)
        return None

    if not metric or not dimension:
        logger.error("Missing required components. Metric: %s, Dimension: %s", metric, dimension)
        return None

    logger.info("Parsed result: chart_type=%s, metric=%s, dimension=%s, second_metric=%s, filter_col=%s, filter_val=%s, kwargs=%s",
                chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs)
    return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension

def safe_groupby_aggregation(df, dimension, metric):
    """Handle pandas column conflicts during groupby operations"""
    try:
        # Standard groupby
        result = df.groupby(dimension)[metric].sum().reset_index()
        return result
    except ValueError as e:
        if "already exists" in str(e):
            # Handle column conflict by using as_index=False
            result = df.groupby(dimension, as_index=False)[metric].sum()
            return result
        else:
            # Handle same column used for both grouping and measuring
            if dimension == metric:
                result = df[dimension].value_counts().reset_index()
                result.columns = [dimension, f'{metric}_count']
                return result
            raise e

def enhanced_rule_based_parse(prompt, df, dimensions, measures, dates):
    """Enhanced parser with natural language understanding - FIXED VERSION."""
    logger.info(f"Enhanced parsing prompt: {prompt}")
    
    # Step 1: Preprocess to remove action words
    cleaned_prompt = preprocess_prompt(prompt)
    logger.info(f"Cleaned prompt: {cleaned_prompt}")
    
    # Step 2: Extract sentiment and intent
    intent = extract_sentiment_and_intent(cleaned_prompt)
    logger.info(f"Detected intent: {intent}")
    
    # Step 3: Extract implicit filters first
    implicit_filters = extract_implicit_filters(cleaned_prompt, dimensions, measures, df)
    logger.info(f"Implicit filters found: {implicit_filters}")
    
    # Step 4: Remove filter parts from prompt for main parsing
    prompt_for_parsing = cleaned_prompt
    if implicit_filters:
        filter_patterns = [
            r'\s+for\s+[a-zA-Z0-9_\s]+$',
            r'\s+in\s+[a-zA-Z0-9_\s]+(?:\s+by|\s*$)',
            r'\s+from\s+[a-zA-Z0-9_\s]+(?:\s+by|\s*$)'
        ]
        for pattern in filter_patterns:
            prompt_for_parsing = re.sub(pattern, '', prompt_for_parsing, flags=re.IGNORECASE).strip()
    
    logger.info(f"Prompt after filter extraction: {prompt_for_parsing}")
    
    # Initialize ALL variables at the start
    chart_type = None
    metric = None
    dimension = None
    second_metric = None
    filter_col = None
    filter_val = None
    kwargs = {}
    is_two_metric = False
    exclude_list = []
    secondary_dimension = None
    n = None  # Initialize n here!
    
    # Apply implicit filters to kwargs
    if implicit_filters:
        filter_col = implicit_filters[0]['column']
        filter_val = implicit_filters[0]['value']
        kwargs['filter_operator'] = implicit_filters[0]['operator']
    
    # Enhanced patterns with FIXED logic
    enhanced_patterns = OrderedDict([
        ("bubble_cloud", [
            r'^bubble\s+cloud\s+of\s+([a-zA-Z0-9_\s]+)\s+sized\s+by\s+([a-zA-Z0-9_\s]+)$'
        ]),
        ("trend_over_time", [
            r'^([a-zA-Z0-9_\s]+)\s+by\s+(?:order\s+)?(?:date|month|year|quarter)$',
            r'^([a-zA-Z0-9_\s]+)\s+(?:over\s+time|trend)$'
        ]),
        ("trend_by_group", [
            r'^([a-zA-Z0-9_\s]+)\s+by\s+(?:order\s+)?(?:date|month|year|quarter)\s+and\s+([a-zA-Z0-9_\s]+)$'
        ]),
        ("compare_metrics", [
            r'^([a-zA-Z0-9_\s]+)\s+(?:vs|versus|against)\s+([a-zA-Z0-9_\s]+)(?:\s+by\s+([a-zA-Z0-9_\s]+))?$'
        ]),
        ("top_n", [
            r'^top\s+(\d+)\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)$'
        ]),
        ("basic_group", [
            r'^([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)$'
        ])
    ])
    
    # FIXED Pattern matching logic
    for pattern_name, pattern_list in enhanced_patterns.items():
        matched = False
        for pattern in pattern_list:
            match = re.match(pattern, prompt_for_parsing, re.IGNORECASE)
            if match:
                logger.info(f"Matched enhanced pattern: {pattern_name} with pattern: {pattern}")
                groups = match.groups()
                logger.info(f"Matched groups: {groups}")
                
                if pattern_name == "bubble_cloud":
                    dim_name, metric_name = groups
                    chart_type = "BubbleCloud"
                    dimension = find_column_safe(dim_name.strip(), dimensions, df)
                    metric = find_column_safe(metric_name.strip(), measures, df)
                
                elif pattern_name == "trend_over_time":
                    metric_name = groups[0]
                    chart_type = "Line"
                    metric = find_column_safe(metric_name.strip(), measures, df)
                    dimension = find_best_date_column(dates, df)
                    kwargs["time_aggregation"] = "month"
                    kwargs["sort_by_date"] = True
                
                elif pattern_name == "trend_by_group":
                    metric_name, secondary_dim_name = groups
                    chart_type = "Line"
                    metric = find_column_safe(metric_name.strip(), measures, df)
                    dimension = find_best_date_column(dates, df)
                    secondary_dimension = find_column_safe(secondary_dim_name.strip(), dimensions, df)
                    kwargs["time_aggregation"] = "month"
                    kwargs["sort_by_date"] = True
                
                elif pattern_name == "compare_metrics":
                    metric1_name, metric2_name = groups[:2]
                    if len(groups) > 2 and groups[2]:
                        dim_name = groups[2]
                        dimension = find_column_safe(dim_name.strip(), dimensions, df)
                    else:
                        # Extract dimension from metric2_name if it contains "by"
                        if "by" in metric2_name:
                            parts = metric2_name.split("by")
                            metric2_name = parts[0].strip()
                            dimension = find_column_safe(parts[1].strip(), dimensions, df)
                    
                    chart_type = "Scatter"
                    is_two_metric = True
                    metric = find_column_safe(metric1_name.strip(), measures, df)
                    second_metric = find_column_safe(metric2_name.strip(), measures, df)
                
                elif pattern_name == "top_n":
                    # FIXED: Correct order for "Top N X by Y"
                    n_str, dim_name, metric_name = groups
                    chart_type = "Bar"
                    n = int(n_str) if n_str else 5
                    # FIXED: dimension is what we rank (X), metric is what we rank by (Y)
                    dimension = find_column_safe(dim_name.strip(), dimensions, df)
                    metric = find_column_safe(metric_name.strip(), measures, df)
                    kwargs["top_n"] = n
                
                elif pattern_name == "basic_group":
                    metric_name, dim_name = groups
                    chart_type = "Bar"
                    metric = find_column_safe(metric_name.strip(), measures, df)
                    dimension = find_column_safe(dim_name.strip(), dimensions + dates, df)
                    
                    # Handle date dimensions
                    if dimension and dimension in dates:
                        chart_type = "Line"
                        kwargs["time_aggregation"] = "month"
                        kwargs["sort_by_date"] = True
                
                matched = True
                break
        
        if matched:
            break
    
    # Use simple_fallback_parse if no pattern matched
    if not chart_type:
        logger.info("Enhanced patterns failed, using simple fallback")
        return simple_fallback_parse(prompt, df, dimensions, measures, dates)
    
    # Validation and auto-correction
    if not metric or not dimension:
        logger.warning(f"Missing components - trying to find alternatives. Metric: {metric}, Dimension: {dimension}")
        
        # Try to find metric in dimensions if not found in measures
        if not metric and dimension:
            potential_metric = find_column_safe(prompt_for_parsing.split()[0], measures, df)
            if potential_metric:
                metric = potential_metric
        
        # Try to find dimension in measures if not found in dimensions
        if not dimension and metric:
            potential_dim = find_column_safe(prompt_for_parsing.split()[-1], dimensions, df)
            if potential_dim:
                dimension = potential_dim
    
    # Final validation
    if not chart_type or not metric or not dimension:
        logger.error("Enhanced parsing failed: Missing required components. Metric: %s, Dimension: %s", metric, dimension)
        return simple_fallback_parse(prompt, df, dimensions, measures, dates)
    
    logger.info("Enhanced parsed result: chart_type=%s, metric=%s, dimension=%s, second_metric=%s, filter_col=%s, filter_val=%s, kwargs=%s",
                chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs)
    
    return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension


def simple_fallback_parse(prompt, df, dimensions, measures, dates):
    """IMPROVED Simple fallback parser that handles basic cases."""
    prompt_lower = prompt.lower().strip()
    
    # Initialize return values
    chart_type = "Bar"
    metric = None
    dimension = None
    second_metric = None
    filter_col = None
    filter_val = None
    kwargs = {}
    is_two_metric = False
    exclude_list = []
    secondary_dimension = None
    
    # Handle multi-dimensional queries like "Sales by Order Date and Ship Mode"
    if " and " in prompt_lower and "by" in prompt_lower:
        parts = prompt_lower.split("by")
        if len(parts) >= 2:
            metric_part = parts[0].strip()
            dim_parts = parts[1].strip()
            
            if " and " in dim_parts:
                dim_parts_split = dim_parts.split(" and ")
                dim1 = dim_parts_split[0].strip()
                dim2 = dim_parts_split[1].strip()
                
                metric = find_column_safe(metric_part, measures, df)
                dimension = find_column_safe(dim1, dimensions + dates, df)
                secondary_dimension = find_column_safe(dim2, dimensions, df)
                
                if dimension and dimension in dates:
                    chart_type = "Line"
                    kwargs["time_aggregation"] = "month"
                    kwargs["sort_by_date"] = True
                
                if metric and dimension:
                    return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension
    
    # Try standard "X by Y" pattern
    if "by" in prompt_lower:
        parts = prompt_lower.split("by")
        if len(parts) >= 2:
            metric_part = parts[0].strip()
            dim_part = parts[1].strip()
            
            metric = find_column_safe(metric_part, measures, df)
            dimension = find_column_safe(dim_part, dimensions + dates, df)
            
            if dimension and dimension in dates:
                chart_type = "Line"
                kwargs["time_aggregation"] = "month"
                kwargs["sort_by_date"] = True
    
    # If still no components found, use defaults
    if not metric and measures:
        metric = measures[0]
    if not dimension and dimensions:
        dimension = dimensions[0]
    
    return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension


def rule_based_parse(prompt, df, dimensions, measures, dates):
    """
    Main parsing function - now enhanced with natural language understanding.
    This replaces your original rule_based_parse function completely.
    """
    return enhanced_rule_based_parse(prompt, df, dimensions, measures, dates)



# Example usage and test cases
def test_enhanced_parser():
    """Test the enhanced parser with various natural language inputs."""
    test_cases = [
        "Show me sales by segment for USA",
        "Build a trend of revenue over time",
        "Create the top 10 products by profit",
        "Plot sales vs profit by category",
        "Revenue by month for Q1",
        "Show highest 5 customers by revenue",
        "Display breakdown of sales by region",
        "Sales across different segments in Europe",
    ]

#def render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
#    return enhanced_render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None)



def render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """FIXED Render a chart based on the query and data."""
    try:
        logger.info(f"🟢 Rendering chart for prompt: {prompt}")

        parsed = rule_based_parse(prompt, df, dimensions, measures, dates)
        if not parsed:
            raise ValueError("Prompt parsing failed.")

        (
            parsed_chart_type, metric, dimension, second_metric,
            filter_col, filter_val, kwargs,
            is_two_metric, exclude_list, secondary_dimension
        ) = parsed

        if not metric or not dimension:
            raise ValueError(f"Missing metric or dimension. Metric: {metric}, Dimension: {dimension}")

        # Use parsed chart_type if not explicitly passed
        chart_type = chart_type or parsed_chart_type or "Bar"

        working_df = df.copy()

        # Outlier detection logic (unchanged)
        if kwargs.get("show_outliers"):
            from calc_utils import detect_outliers
            try:
                working_df = detect_outliers(working_df, metric, method="std")
                logger.info(f"Outliers added: {working_df['Outlier'].value_counts().to_dict()}")

                original_df = working_df.copy()
                dimension_col = dimension

                grouped = working_df.groupby([dimension_col, "Outlier"])[metric].sum().reset_index()
                all_dims = working_df[[dimension_col]].drop_duplicates()
                all_dims["key"] = 1
                outlier_flags = pd.DataFrame({"Outlier": [True, False], "key": 1})
                dim_flag_combos = pd.merge(all_dims, outlier_flags, on="key").drop(columns="key")

                grouped = pd.merge(dim_flag_combos, grouped, on=[dimension_col, "Outlier"], how="left")
                grouped[metric].fillna(0, inplace=True)
                grouped["Color"] = grouped["Outlier"].map({True: "Outlier", False: "Normal"})

                kwargs["color_by"] = "Color"
                working_df = grouped
                chart_df = grouped
                show_data_df = original_df[[dimension_col, metric, "Outlier"]]
                logger.info("Prepared grouped data with outlier flag.")
            except Exception as e:
                logger.warning(f"⚠️ Outlier detection failed: {e}")

        # Filter
        if filter_col and filter_val is not None:
            try:
                working_df = working_df[
                    working_df[filter_col].astype(str).str.lower().str.strip() ==
                    str(filter_val).lower().strip()
                ]
                logger.info(f"Filter applied: {filter_col} = {filter_val}, Rows left: {len(working_df)}")
            except Exception as e:
                logger.warning(f"Filter error: {e}")

        # Handle date dimensions
        time_agg = None
        if dimension in dates:
            working_df[dimension] = pd.to_datetime(working_df[dimension], errors='coerce')
            time_agg = kwargs.get("time_aggregation", "month")
            if chart_type == "Line":
                if time_agg == "month":
                    working_df[dimension] = working_df[dimension].dt.to_period("M").dt.to_timestamp()
                elif time_agg == "quarter":
                    working_df[dimension] = working_df[dimension].dt.to_period("Q").dt.to_timestamp()
                elif time_agg == "year":
                    working_df[dimension] = working_df[dimension].dt.to_period("Y").dt.to_timestamp()

        # FIXED Grouping logic using safe_groupby_aggregation
        chart_data = None

        if is_two_metric and second_metric:
            chart_data = working_df.groupby(dimension)[[metric, second_metric]].sum().reset_index()
            table_columns = [dimension, metric, second_metric]
        elif secondary_dimension:
            # Multi-dimensional grouping
            try:
                chart_data = working_df.groupby([dimension, secondary_dimension])[metric].sum().reset_index()
            except ValueError as e:
                if "already exists" in str(e):
                    chart_data = working_df.groupby([dimension, secondary_dimension], as_index=False)[metric].sum()
                else:
                    raise e
            table_columns = [dimension, secondary_dimension, metric]
        else:
            if "Outlier" in working_df.columns:
                chart_data = working_df.groupby([dimension, "Outlier"])[metric].sum().reset_index()
            else:
                # FIXED: Use safe groupby aggregation
                chart_data = safe_groupby_aggregation(working_df, dimension, metric)
            table_columns = [dimension, metric]

        # Sorting and top/bottom N
        if "top_n" in kwargs:
            n = kwargs["top_n"]
            chart_data = chart_data.sort_values(by=metric, ascending=kwargs.get("is_bottom", False)).head(n)
        elif dimension in dates:
            chart_data = chart_data.sort_values(by=dimension)
        elif sort_order == "Descending":
            chart_data = chart_data.sort_values(by=metric, ascending=False)
        else:
            chart_data = chart_data.sort_values(by=metric)

        if time_agg:
            kwargs["time_aggregation"] = time_agg


        insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
        
        styled_fig = create_styled_chart(
            chart_data=chart_data,
            metric=metric, 
            dimension=dimension,
            chart_type=chart_type,
            secondary_dimension=secondary_dimension,
    #theme=theme,  # Add theme parameter
    top_n=kwargs.get('top_n'),
    second_metric=second_metric,
    insights=insights
)
        logger.info(f"✅ Chart prepared: {chart_type} | Metric: {metric} | Dimension: {dimension}")
        return chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs

    except Exception as e:
        logger.error(f"❌ Chart rendering failed: {e}")
        raise ValueError(f"Failed to render chart: {str(e)}")


# In chart_utils.py
def generate_insights(chart_data, metric, dimension, secondary_dimension=None):
    """Generate insights from chart data using OpenAI API or fallback to rule-based insights."""
    try:
        logger.info(f"🔍 Generating insights for chart | Metric: {metric}, Dimension: {dimension}, Secondary: {secondary_dimension}, Chart Data Shape: {chart_data.shape}")

        # Validate inputs
        if chart_data.empty:
            logger.warning("Chart data is empty, cannot generate insights.")
            return ["No data available to generate insights."]
        if metric not in chart_data.columns:
            logger.error(f"Metric '{metric}' not found in chart_data columns: {chart_data.columns}")
            return ["Metric not found in data."]

        # If OpenAI is available, try LLM-based insight generation
        if USE_OPENAI and openai.api_key:
            try:
                # Prepare statistics if numeric
                stats = {}
                if pd.api.types.is_numeric_dtype(chart_data[metric]):
                    stats = {
                        "mean": float(chart_data[metric].mean()),
                        "max": float(chart_data[metric].max()),
                        "min": float(chart_data[metric].min()),
                        "total": float(chart_data[metric].sum()),
                        "count": len(chart_data)
                    }
                    logger.info(f"Computed stats for metric '{metric}': {stats}")

                # Compose the insight prompt
                prompt = (
                    f"Generate 3 concise, actionable business insights for a data visualization:\n"
                    f"- Metric: {metric}\n"
                    f"- Dimension: {dimension}\n"
                    f"- Secondary Dimension: {secondary_dimension or 'None'}\n"
                    f"- Data Points: {len(chart_data)}\n"
                )
                if stats:
                    prompt += (
                        f"- Statistics: Mean={stats['mean']:.2f}, Max={stats['max']:.2f}, "
                        f"Min={stats['min']:.2f}, Total={stats['total']:.2f}\n"
                    )
                if pd.api.types.is_datetime64_any_dtype(chart_data.get(dimension, pd.Series())):
                    prompt += f"- Time-based dimension: {dimension} (Date)\n"
                prompt += (
                    "Provide insights that highlight trends, key performers, or anomalies, "
                    "suitable for business decision-making. Format as a list of 3 bullet points."
                )

                # Call OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst providing concise, actionable insights from data visualizations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )

                # Extract insights from response
                content = response.choices[0].message.content.strip()
                insights = [
                    line.strip('- ').strip()
                    for line in content.split('\n')
                    if line.strip().startswith("-")
                ]
                insights = insights[:3]  # Ensure at most 3 insights
                logger.info(f"💡 OpenAI insights generated: {insights}")
                return insights if insights else ["No significant insights generated by OpenAI."]

            except Exception as e:
                logger.error(f"⚠️ OpenAI API call failed: {str(e)}", exc_info=True)

        # Fallback: Rule-based insight generation
        logger.info("🔄 Falling back to rule-based insights")
        insights = []

        if pd.api.types.is_numeric_dtype(chart_data[metric]):
            mean_val = chart_data[metric].mean()
            max_val = chart_data[metric].max()
            min_val = chart_data[metric].min()
            total_val = chart_data[metric].sum()

            # Insight 1: Top performer
            top_row = chart_data.loc[chart_data[metric].idxmax()]
            insights.append(f"Top performer: {top_row[dimension]} with {metric} of {top_row[metric]:.2f}")

            # Insight 2: Value range and average
            insights.append(f"{metric} ranges from {min_val:.2f} to {max_val:.2f}, with an average of {mean_val:.2f}")

            # Handle time-based dimensions (for Line charts)
            if pd.api.types.is_datetime64_any_dtype(chart_data.get(dimension, pd.Series())):
                chart_data = chart_data.sort_values(by=dimension)
                first_val = chart_data[metric].iloc[0]
                last_val = chart_data[metric].iloc[-1]
                trend = "increasing" if last_val > first_val else "decreasing" if last_val < first_val else "stable"
                insights.append(f"{metric} shows a {trend} trend over time from {first_val:.2f} to {last_val:.2f}")

            # Handle Scatter charts (dual metrics)
            elif len(chart_data.columns) > 2 and chart_data.columns[2] != dimension:
                second_metric = chart_data.columns[2]
                if pd.api.types.is_numeric_dtype(chart_data[second_metric]):
                    corr = chart_data[[metric, second_metric]].corr().iloc[0, 1]
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                    direction = "positive" if corr > 0 else "negative"
                    insights.append(f"{strength.capitalize()} {direction} correlation ({corr:.2f}) between {metric} and {second_metric}")

            # Insight 3: Outliers (for non-time dimensions)
            if not pd.api.types.is_datetime64_any_dtype(chart_data.get(dimension, pd.Series())):
                q1 = chart_data[metric].quantile(0.25)
                q3 = chart_data[metric].quantile(0.75)
                iqr = q3 - q1
                outliers = chart_data[
                    (chart_data[metric] < (q1 - 1.5 * iqr)) |
                    (chart_data[metric] > (q3 + 1.5 * iqr))
                ]
                if len(outliers) > 0:
                    insights.append(f"Found {len(outliers)} potential outliers in {metric} for {dimension}")
                else:
                    insights.append(f"No significant outliers detected in {metric} for {dimension}")

            # Secondary dimension breakdown
            if secondary_dimension and secondary_dimension in chart_data.columns:
                top_secondary = (
                    chart_data.groupby(secondary_dimension)[metric]
                    .sum()
                    .sort_values(ascending=False)
                    .head(1)
                )
                if not top_secondary.empty:
                    top_sec_label = top_secondary.index[0]
                    top_sec_value = top_secondary.iloc[0]
                    insights.append(f"Highest {metric} by {secondary_dimension}: {top_sec_label} with {top_sec_value:.2f}")

        insights = insights[:3]  # Limit to 3 insights

        if not insights:
            logger.warning("No insights generated due to insufficient data or incompatible chart type.")
            insights = ["No significant insights could be generated from the data."]

        logger.info(f"✅ Rule-based insights: {insights}")
        return insights

    except Exception as e:
        logger.error(f"❌ Insight generation failed: {str(e)}", exc_info=True)
        return ["Unable to generate insights at this time."]