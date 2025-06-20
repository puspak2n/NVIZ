import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import pandas as pd
import json
import base64
import openai
import os
import numpy as np
from datetime import datetime
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def load_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None

# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Function to create dark-themed charts
def create_dark_chart(fig):
    """Apply consistent dark theme to charts"""
    fig.update_layout(
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#F3F4F6',
        template='plotly_dark',
        margin=dict(t=40, b=40, l=40, r=40),
        title_font_size=16,
        title_font_color='#60A5FA',
        xaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151'
        ),
        yaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151'
        )
    )
    return fig

# Smart data filtering helper
def get_business_relevant_columns(df):
    """Filter out ID columns and other non-analytical fields"""
    avoid_patterns = [
        'id', 'row_id', 'order_id', 'customer_id', 'product_id', 
        'transaction_id', 'invoice_id', 'record_id', 'index', 'key',
        'guid', 'uuid', 'code', 'sku', '_id'
    ]
    business_columns = {
        'categorical': [],
        'numerical': [],
        'temporal': []
    }
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        if any(pattern in col_lower for pattern in avoid_patterns):
            continue
        if df[col].dtype == 'object' and df[col].nunique() > 0.8 * len(df):
            continue
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            business_columns['categorical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col_lower:
            business_columns['temporal'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            business_columns['numerical'].append(col)
    return business_columns

# Updated AI Agent Classes
class DataAnalystAgent:
    def __init__(self):
        self.name = "Data Analyst Agent"
    
    def analyze_dataset(self, df):
        """Perform intelligent data analysis focused on code generation or AI chart specs"""
        try:
            business_cols = get_business_relevant_columns(df)
            profile = self._generate_smart_data_profile(df, business_cols)
            use_code_based = st.session_state.get('use_code_based_charts', True)
            
            analysis_prompt = f"""
            You are an expert data analyst. Generate {'Python code for data visualization' if use_code_based else 'chart specifications for direct plotting'}.
            
            Dataset Overview:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Numerical Columns: {business_cols['numerical'][:10]}
            - Categorical Columns: {business_cols['categorical'][:10]}
            - Temporal Columns: {business_cols['temporal']}
            
            Data Profile:
            {profile}
            
            Generate a JSON response with 4 chart recommendations.
            Each recommendation should include:
            - {'"code": Complete Python code using Plotly Express (px) or Plotly Graph Objects (go), assigning figure to "fig" variable' if use_code_based else '"spec": JSON Plotly chart specification (data, layout)'}
            - "type": Chart type (bar, line, scatter, heatmap, box, violin, etc.)
            - "title": Descriptive title
            - "x_col": X-axis column name
            - "y_col": Y-axis column name (if applicable)
            - "reason": Brief explanation
            - "priority": "high", "medium", or "low"
            
            Important:
            1. {'Use Plotly (px or go) for code-based visualizations' if use_code_based else 'Return JSON chart specs compatible with Plotly'}.
            2. {'Code must assign figure to "fig"' if use_code_based else 'Spec must include data and layout'}.
            3. Different chart types for variety.
            4. Focus on business-relevant columns only.
            
            Return ONLY valid JSON with a "recommendations" array.
            """
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            analysis = json.loads(response_text)
            if 'recommendations' not in analysis:
                analysis['recommendations'] = analysis.get('charts', [])
            return analysis
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._generate_fallback_analysis(df, business_cols)
    
    def _generate_smart_data_profile(self, df, business_cols):
        profile = []
        for col in business_cols['numerical'][:5]:
            if col in df.columns:
                stats = df[col].describe()
                profile.append(f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        for col in business_cols['categorical'][:5]:
            if col in df.columns:
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3).to_dict()
                profile.append(f"- {col}: {unique_count} categories, top: {list(top_values.keys())}")
        if len(business_cols['numerical']) > 1:
            num_cols = [col for col in business_cols['numerical'] if col in df.columns][:5]
            if len(num_cols) > 1:
                corr_matrix = df[num_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            high_corr.append(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_val:.3f}")
                if high_corr:
                    profile.append(f"High correlations: {', '.join(high_corr[:3])}")
        return "\n".join(profile) if profile else "Basic dataset profile generated"
    
    def _generate_fallback_analysis(self, df, business_cols):
        recommendations = []
        use_code_based = st.session_state.get('use_code_based_charts', True)
        if business_cols['categorical'] and business_cols['numerical']:
            cat_col = business_cols['categorical'][0]
            num_col = business_cols['numerical'][0]
            if use_code_based:
                code = f"""import plotly.express as px
import pandas as pd
grouped = df.groupby('{cat_col}')['{num_col}'].mean().sort_values(ascending=False).head(10)
fig = px.bar(x=grouped.index, y=grouped.values,
             title='Average {num_col} by {cat_col}',
             labels={{'x': '{cat_col}', 'y': 'Average {num_col}'}})
fig.update_layout(xaxis_tickangle=-45)"""
                recommendations.append({
                    "code": code,
                    "type": "bar",
                    "title": f"Average {num_col} by {cat_col}",
                    "x_col": cat_col,
                    "y_col": num_col,
                    "reason": "Shows performance metrics by category",
                    "priority": "high"
                })
            else:
                recommendations.append({
                    "spec": {
                        "data": [{"x": [], "y": [], "type": "bar"}],
                        "layout": {
                            "title": f"Average {num_col} by {cat_col}",
                            "xaxis": {"title": cat_col, "tickangle": -45},
                            "yaxis": {"title": f"Average {num_col}"}
                        }
                    },
                    "type": "bar",
                    "title": f"Average {num_col} by {cat_col}",
                    "x_col": cat_col,
                    "y_col": num_col,
                    "reason": "Shows performance metrics by category",
                    "priority": "high"
                })
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            num_col = business_cols['numerical'][0]
            if use_code_based:
                code = f"""import plotly.express as px
import pandas as pd
df_time = df.copy()
df_time['{time_col}'] = pd.to_datetime(df_time['{time_col}'])
time_series = df_time.groupby('{time_col}')['{num_col}'].sum().reset_index()
fig = px.line(time_series, x='{time_col}', y='{num_col}',
              title='{num_col} Over Time',
              labels={{'x': 'Date', 'y': '{num_col}'}})"""
                recommendations.append({
                    "code": code,
                    "type": "line",
                    "title": f"{num_col} Over Time",
                    "x_col": time_col,
                    "y_col": num_col,
                    "reason": "Reveals temporal trends and patterns",
                    "priority": "high"
                })
            else:
                recommendations.append({
                    "spec": {
                        "data": [{"x": [], "y": [], "type": "scatter", "mode": "lines"}],
                        "layout": {
                            "title": f"{num_col} Over Time",
                            "xaxis": {"title": "Date"},
                            "yaxis": {"title": num_col}
                        }
                    },
                    "type": "line",
                    "title": f"{num_col} Over Time",
                    "x_col": time_col,
                    "y_col": num_col,
                    "reason": "Reveals temporal trends and patterns",
                    "priority": "high"
                })
        if business_cols['numerical']:
            num_col = business_cols['numerical'][0]
            if use_code_based:
                code = f"""import plotly.express as px
fig = px.histogram(df, x='{num_col}', 
                   title='Distribution of {num_col}',
                   nbins=30,
                   labels={{'x': '{num_col}', 'y': 'Count'}})
fig.add_vline(x=df['{num_col}'].mean(), line_dash="dash", 
              annotation_text="Mean", annotation_position="top right")"""
                recommendations.append({
                    "code": code,
                    "type": "histogram",
                    "title": f"Distribution of {num_col}",
                    "x_col": num_col,
                    "y_col": None,
                    "reason": "Shows data distribution and outliers",
                    "priority": "medium"
                })
            else:
                recommendations.append({
                    "spec": {
                        "data": [{"x": [], "type": "histogram", "nbinsx": 30}],
                        "layout": {
                            "title": f"Distribution of {num_col}",
                            "xaxis": {"title": num_col},
                            "yaxis": {"title": "Count"},
                            "shapes": [{
                                "type": "line",
                                "x0": df[num_col].mean(),
                                "x1": df[num_col].mean(),
                                "y0": 0,
                                "y1": 1,
                                "yref": "paper",
                                "line": {"dash": "dash"}
                            }]
                        }
                    },
                    "type": "histogram",
                    "title": f"Distribution of {num_col}",
                    "x_col": num_col,
                    "y_col": None,
                    "reason": "Shows data distribution and outliers",
                    "priority": "medium"
                })
        if len(business_cols['numerical']) >= 2:
            x_col = business_cols['numerical'][0]
            y_col = business_cols['numerical'][1]
            if use_code_based:
                code = f"""import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}',
                 title='{y_col} vs {x_col}',
                 labels={{'x': '{x_col}', 'y': '{y_col}'}},
                 trendline="ols")"""
                recommendations.append({
                    "code": code,
                    "type": "scatter",
                    "title": f"{y_col} vs {x_col}",
                    "x_col": x_col,
                    "y_col": y_col,
                    "reason": "Explores relationships between metrics",
                    "priority": "medium"
                })
            else:
                recommendations.append({
                    "spec": {
                        "data": [{"x": [], "y": [], "type": "scatter", "mode": "markers"}],
                        "layout": {
                            "title": f"{y_col} vs {x_col}",
                            "xaxis": {"title": x_col},
                            "yaxis": {"title": y_col}
                        }
                    },
                    "type": "scatter",
                    "title": f"{y_col} vs {x_col}",
                    "x_col": x_col,
                    "y_col": y_col,
                    "reason": "Explores relationships between metrics",
                    "priority": "medium"
                })
        return {"recommendations": recommendations[:4]}

class ChartCreatorAgent:
    def __init__(self):
        self.name = "Chart Creator Agent"
        self.max_charts = 4
    
    def create_intelligent_charts(self, df, analysis):
        """Create charts using either code generation or direct AI specs"""
        charts = []
        business_cols = get_business_relevant_columns(df)
        try:
            if st.session_state.get('use_code_based_charts', True):
                charts = self._create_code_based_charts(df, analysis, business_cols)
            else:
                charts = self._create_ai_generated_charts(df, analysis, business_cols)
            return charts[:self.max_charts]
        except Exception as e:
            print(f"Chart creation error: {str(e)}")
            return self._create_fallback_charts(df, business_cols)[:self.max_charts]

    def _create_fallback_charts(self, df, business_cols):
        charts = []
        if business_cols['categorical'] and business_cols['numerical']:
            cat_col = business_cols['categorical'][0]
            num_col = business_cols['numerical'][0]
            try:
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=grouped.index, 
                    y=grouped.values,
                    title=f"Average {num_col} by {cat_col}",
                    labels={'x': cat_col, 'y': f'Average {num_col}'}
                )
                code = f"""import plotly.express as px
grouped = df.groupby('{cat_col}')['{num_col}'].mean().sort_values(ascending=False).head(10)
fig = px.bar(x=grouped.index, y=grouped.values, 
             title="Average {num_col} by {cat_col}")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"Average {num_col} by {cat_col}",
                    'type': 'bar',
                    'x_col': cat_col,
                    'y_col': num_col,
                    'code': code,
                    'data': df,
                    'priority': 'high',
                    'reason': 'Top performing categories analysis'
                })
            except Exception as e:
                print(f"Bar chart error: {str(e)}")
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            num_col = business_cols['numerical'][0]
            try:
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                df_time = df_time.dropna(subset=[time_col])
                time_series = df_time.groupby(time_col)[num_col].sum().reset_index()
                fig = px.line(
                    time_series, 
                    x=time_col, 
                    y=num_col,
                    title=f"{num_col} Over Time"
                )
                code = f"""import plotly.express as px
df_time = df.copy()
df_time['{time_col}'] = pd.to_datetime(df_time['{time_col}'])
time_series = df_time.groupby('{time_col}')['{num_col}'].sum().reset_index()
fig = px.line(time_series, x='{time_col}', y='{num_col}', 
              title="{num_col} Over Time")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"{num_col} Over Time",
                    'type': 'line',
                    'x_col': time_col,
                    'y_col': num_col,
                    'code': code,
                    'data': df,
                    'priority': 'high',
                    'reason': 'Temporal trend analysis'
                })
            except Exception as e:
                print(f"Time series chart error: {str(e)}")
        if len(business_cols['numerical']) > 0:
            num_col = business_cols['numerical'][0]
            try:
                fig = px.histogram(
                    df, 
                    x=num_col,
                    title=f"Distribution of {num_col}",
                    nbins=30
                )
                code = f"""import plotly.express as px
fig = px.histogram(df, x='{num_col}', 
                   title="Distribution of {num_col}", nbins=30)"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"Distribution of {num_col}",
                    'type': 'histogram',
                    'x_col': num_col,
                    'y_col': None,
                    'code': code,
                    'data': df,
                    'priority': 'medium',
                    'reason': 'Statistical distribution analysis'
                })
            except Exception as e:
                print(f"Histogram chart error: {str(e)}")
        if len(business_cols['numerical']) >= 2:
            x_col = business_cols['numerical'][0]
            y_col = business_cols['numerical'][1]
            try:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    trendline="ols" if len(df) < 1000 else None
                )
                code = f"""import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}',
                 title="{y_col} vs {x_col}", trendline="ols")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"{y_col} vs {x_col}",
                    'type': 'scatter',
                    'x_col': x_col,
                    'y_col': y_col,
                    'code': code,
                    'data': df,
                    'priority': 'medium',
                    'reason': 'Correlation analysis'
                })
            except Exception as e:
                print(f"Scatter chart error: {str(e)}")
        return charts

    
    # Modified _create_code_based_charts method
    def _create_code_based_charts(self, data_frame, columns, target_num_charts=4):
        """Create charts using AI-generated code."""
        charts = []
        
        try:
            # Your AI code generation logic here
            code = self._generate_chart_code(data_frame, columns, target_num_charts)
            
            if code:
                # Execute code and extract charts
                local_vars = {"data": data_frame, "px": px, "go": go, "np": np, "pd": pd}
                exec(code, globals(), local_vars)
                
                for var_name, var_value in local_vars.items():
                    if isinstance(var_value, (go.Figure, plt.Figure)):
                        charts.append({
                            'figure': var_value,
                            'title': f'🤖 AI Custom Chart #{len(charts)+1}',
                            'is_ai_generated': True
                        })
        except Exception as e:
            print(f"Error in generated chart code: {e}")
        
        # IMPORTANT: Only fall back if in mixed mode, not in pure agentic mode
        if len(charts) < target_num_charts and not self.strict_agentic_mode:
            # Add a property to control fallback behavior
            fallback_charts = self._create_fallback_charts(
                data_frame, columns, target_num_charts - len(charts)
            )
            charts.extend(fallback_charts)
        
        return charts
    
    def _create_ai_generated_charts(self, df, analysis, business_cols):
        """Generate charts directly from AI specifications"""
        charts = []
        for i, rec in enumerate(analysis.get('recommendations', [])[:self.max_charts]):
            spec = rec.get('spec', {})
            x_col = rec.get('x_col', '')
            y_col = rec.get('y_col', '')
            chart_type = rec.get('type', 'unknown').lower()
            title = rec.get('title', f'Chart {i+1}')
            if not x_col or (y_col and y_col not in df.columns) or x_col not in df.columns:
                continue
            try:
                if chart_type == 'bar' and y_col:
                    grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=grouped.index, 
                        y=grouped.values,
                        title=title,
                        labels={'x': x_col, 'y': f'Average {y_col}'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                elif chart_type == 'line' and y_col and x_col in business_cols['temporal']:
                    df_time = df.copy()
                    df_time[x_col] = pd.to_datetime(df_time[x_col], errors='coerce')
                    df_time = df_time.dropna(subset=[x_col])
                    time_series = df_time.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.line(
                        time_series,
                        x=x_col,
                        y=y_col,
                        title=title
                    )
                elif chart_type == 'scatter' and y_col:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        trendline="ols" if len(df) < 1000 else None
                    )
                elif chart_type == 'histogram':
                    fig = px.histogram(
                        df,
                        x=x_col,
                        title=title,
                        nbins=30
                    )
                    fig.add_vline(x=df[x_col].mean(), line_dash="dash", annotation_text="Mean")
                else:
                    continue
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': title,
                    'reason': rec.get('reason', 'AI-generated visualization'),
                    'priority': rec.get('priority', 'medium'),
                    'x_col': x_col,
                    'y_col': y_col,
                    'data': df,
                    'type': chart_type,
                    'code': 'AI-generated chart specification (no Python code)'
                })
            except Exception as e:
                print(f"AI-generated chart error: {str(e)}")
        if len(charts) < 3:
            fallback_charts = self._create_fallback_charts(df, business_cols)
            charts.extend(fallback_charts[:self.max_charts - len(charts)])
        return charts
    
    def _verify_code_safety(self, code):
        dangerous_patterns = [
            'import os', 'import sys', '__import__', 'eval(', 'exec(',
            'open(', 'file(', 'input(', 'raw_input', 'compile(',
            'globals(', 'locals(', '__', 'getattr(', 'setattr(',
            'subprocess', 'system(', 'popen('
        ]
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        allowed_imports = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'sklearn']
        lines = code.split('\n')
        for line in lines:
            if line.strip().startswith('import ') or 'from ' in line:
                if not any(lib in line for lib in allowed_imports):
                    return False
        return True
    
    def _execute_chart_code(self, code, df):
        try:
            safe_globals = {
                '__builtins__': {
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'print': print,
                },
                'df': df,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'px': px,
                'go': go,
                'fig': None
            }
            exec(code, safe_globals)
            if 'fig' in safe_globals and safe_globals['fig'] is not None:
                return safe_globals['fig']
            return None
        except Exception as e:
            print(f"Code execution error: {str(e)}")
            return None
    
    def _create_fallback_charts(self, df, business_cols):
        charts = []
        if business_cols['categorical'] and business_cols['numerical']:
            cat_col = business_cols['categorical'][0]
            num_col = business_cols['numerical'][0]
            try:
                grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=grouped.index, 
                    y=grouped.values,
                    title=f"Average {num_col} by {cat_col}",
                    labels={'x': cat_col, 'y': f'Average {num_col}'}
                )
                code = f"""import plotly.express as px
grouped = df.groupby('{cat_col}')['{num_col}'].mean().sort_values(ascending=False).head(10)
fig = px.bar(x=grouped.index, y=grouped.values, 
             title="Average {num_col} by {cat_col}")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"Average {num_col} by {cat_col}",
                    'type': 'bar',
                    'x_col': cat_col,
                    'y_col': num_col,
                    'code': code,
                    'data': df,
                    'priority': 'high',
                    'reason': 'Top performing categories analysis'
                })
            except Exception as e:
                print(f"Bar chart error: {str(e)}")
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            num_col = business_cols['numerical'][0]
            try:
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                df_time = df_time.dropna(subset=[time_col])
                time_series = df_time.groupby(time_col)[num_col].sum().reset_index()
                fig = px.line(
                    time_series, 
                    x=time_col, 
                    y=num_col,
                    title=f"{num_col} Over Time"
                )
                code = f"""import plotly.express as px
df_time = df.copy()
df_time['{time_col}'] = pd.to_datetime(df_time['{time_col}'])
time_series = df_time.groupby('{time_col}')['{num_col}'].sum().reset_index()
fig = px.line(time_series, x='{time_col}', y='{num_col}', 
              title="{num_col} Over Time")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"{num_col} Over Time",
                    'type': 'line',
                    'x_col': time_col,
                    'y_col': num_col,
                    'code': code,
                    'data': df,
                    'priority': 'high',
                    'reason': 'Temporal trend analysis'
                })
            except Exception as e:
                print(f"Time series chart error: {str(e)}")
        if len(business_cols['numerical']) > 0:
            num_col = business_cols['numerical'][0]
            try:
                fig = px.histogram(
                    df, 
                    x=num_col,
                    title=f"Distribution of {num_col}",
                    nbins=30
                )
                code = f"""import plotly.express as px
fig = px.histogram(df, x='{num_col}', 
                   title="Distribution of {num_col}", nbins=30)"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"Distribution of {num_col}",
                    'type': 'histogram',
                    'x_col': num_col,
                    'y_col': None,
                    'code': code,
                    'data': df,
                    'priority': 'medium',
                    'reason': 'Statistical distribution analysis'
                })
            except Exception as e:
                print(f"Histogram chart error: {str(e)}")
        if len(business_cols['numerical']) >= 2:
            x_col = business_cols['numerical'][0]
            y_col = business_cols['numerical'][1]
            try:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    trendline="ols" if len(df) < 1000 else None
                )
                code = f"""import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}',
                 title="{y_col} vs {x_col}", trendline="ols")"""
                charts.append({
                    'figure': create_dark_chart(fig),
                    'title': f"{y_col} vs {x_col}",
                    'type': 'scatter',
                    'x_col': x_col,
                    'y_col': y_col,
                    'code': code,
                    'data': df,
                    'priority': 'medium',
                    'reason': 'Correlation analysis'
                })
            except Exception as e:
                print(f"Scatter chart error: {str(e)}")
        return charts

class InsightGeneratorAgent:
    def __init__(self):
        self.name = "Insight Generator Agent"
    
    def generate_insights(self, df, chart_info, analysis):
        """Generate rule-based insights without using AI (to avoid hallucination)"""
        insights = []
        try:
            x_col = chart_info.get('x_col')
            y_col = chart_info.get('y_col')
            chart_type = chart_info.get('type', '').lower()
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                stats = df[y_col].describe()
                insights.append(f"Average {y_col}: {stats['mean']:.2f} (Â±{stats['std']:.2f} std dev)")
                range_val = stats['max'] - stats['min']
                insights.append(f"Range: {stats['min']:.2f} to {stats['max']:.2f} (span: {range_val:.2f})")
                iqr = stats['75%'] - stats['25%']
                if iqr > 0:
                    insights.append(f"Middle 50% of values: {stats['25%']:.2f} to {stats['75%']:.2f}")
                lower_bound = stats['25%'] - 1.5 * iqr
                upper_bound = stats['75%'] + 1.5 * iqr
                outliers = df[(df[y_col] < lower_bound) | (df[y_col] > upper_bound)]
                if len(outliers) > 0:
                    insights.append(f"Potential outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
            if chart_type == 'bar' and x_col and y_col:
                if x_col in df.columns and y_col in df.columns:
                    try:
                        grouped = df.groupby(x_col)[y_col].agg(['sum', 'mean', 'count'])
                        top_category = grouped['sum'].idxmax()
                        bottom_category = grouped['sum'].idxmin()
                        insights.append(f"Highest {y_col}: {top_category} (total: {grouped.loc[top_category, 'sum']:.2f})")
                        insights.append(f"Lowest {y_col}: {bottom_category} (total: {grouped.loc[bottom_category, 'sum']:.2f})")
                    except:
                        pass
            elif chart_type == 'line' and x_col and y_col:
                if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                    values = df[y_col].dropna()
                    if len(values) > 1:
                        first_val = values.iloc[0]
                        last_val = values.iloc[-1]
                        if first_val != 0:
                            change_pct = ((last_val - first_val) / abs(first_val)) * 100
                            trend = "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable"
                            insights.append(f"Overall trend: {trend} ({change_pct:+.1f}% change)")
                        if len(values) > 2:
                            pct_changes = values.pct_change().dropna()
                            if len(pct_changes) > 0:
                                volatility = pct_changes.std() * 100
                                insights.append(f"Volatility: {volatility:.1f}% (std of period changes)")
            elif chart_type == 'scatter' and x_col and y_col:
                if x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        valid_data = df[[x_col, y_col]].dropna()
                        if len(valid_data) > 2:
                            correlation = valid_data[x_col].corr(valid_data[y_col])
                            strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                            direction = "positive" if correlation > 0 else "negative"
                            insights.append(f"Correlation: {strength} {direction} ({correlation:.3f})")
                            r_squared = correlation ** 2
                            insights.append(f"R-squared: {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
            elif chart_type == 'histogram' and x_col:
                if x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
                    values = df[x_col].dropna()
                    if len(values) > 0:
                        skewness = values.skew()
                        kurtosis = values.kurtosis()
                        skew_interpretation = "right-skewed" if skewness > 0.5 else "left-skewed" if skewness < -0.5 else "roughly symmetric"
                        insights.append(f"Distribution: {skew_interpretation} (skewness: {skewness:.2f})")
                        p10 = values.quantile(0.1)
                        p90 = values.quantile(0.9)
                        insights.append(f"80% of values between {p10:.2f} and {p90:.2f}")
            total_nulls = df[x_col].isnull().sum() if x_col and x_col in df.columns else 0
            if y_col and y_col in df.columns:
                total_nulls += df[y_col].isnull().sum()
            if total_nulls > 0:
                insights.append(f"Missing values: {total_nulls} records")
            return insights[:5] if insights else self._get_default_insights(chart_info)
        except Exception as e:
            print(f"Insight generation error: {str(e)}")
            return self._get_default_insights(chart_info)
    
    def _get_default_insights(self, chart_info):
        chart_type = chart_info.get('type', 'visualization')
        x_col = chart_info.get('x_col', 'X-axis')
        y_col = chart_info.get('y_col', 'Y-axis')
        return [
            f"This {chart_type} shows the relationship between {x_col} and {y_col or 'values'}",
            f"Data patterns visible in this visualization can guide decision-making",
            f"Consider filtering or segmenting for deeper analysis"
        ]

class AgenticAIAgent:
    def __init__(self):
        self.name = "Agentic AI Coordinator"
        self.version = "1.0.0"
        self.data_analyst = None
        self.chart_creator = None
        self.insight_generator = None
        self.agent_status = {
            'data_analyst': 'idle',
            'chart_creator': 'idle',
            'insight_generator': 'idle'
        }
        self.learning_data = {
            'preferred_charts': [],
            'avoided_columns': [],
            'business_context': "",
            'user_feedback': [],
            'successful_patterns': []
        }
        self.conversation_log = []
        self.active_analysis = None
    
    def initialize_agents(self):
        if self.data_analyst is None:
            self.data_analyst = DataAnalystAgent()
            self.log_conversation("System", "Data Analyst Agent initialized")
        if self.chart_creator is None:
            self.chart_creator = ChartCreatorAgent()
            self.log_conversation("System", "Chart Creator Agent initialized")
        if self.insight_generator is None:
            self.insight_generator = InsightGeneratorAgent()
            self.log_conversation("System", "Insight Generator Agent initialized")
    
    def log_conversation(self, agent_name, message, message_type="info", details=None):
        conversation_entry = {
            'agent': agent_name,
            'message': message,
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.conversation_log.append(conversation_entry)
        if 'agent_conversations' in st.session_state:
            st.session_state.agent_conversations.append(conversation_entry)
    
    def update_agent_status(self, agent_name, status):
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status
        if 'agent_status' in st.session_state:
            st.session_state.agent_status[agent_name] = status
    
    def run_full_analysis(self, dataframe, use_code_based_charts=True):
        try:
            self.initialize_agents()
            self.update_agent_status('data_analyst', 'analyzing')
            self.log_conversation("Agentic AI Coordinator", "Starting comprehensive data analysis workflow")
            analysis_results = self.data_analyst.analyze_dataset(dataframe)
            self.active_analysis = analysis_results
            self.log_conversation(
                "Data Analyst Agent", 
                f"Completed analysis: {len(analysis_results.get('recommendations', []))} visualization opportunities identified",
                "analyst",
                analysis_results
            )
            self.update_agent_status('data_analyst', 'complete')
            self.update_agent_status('chart_creator', 'creating')
            charts = self.chart_creator.create_intelligent_charts(dataframe, analysis_results)
            self.log_conversation(
                "Chart Creator Agent",
                f"Generated {len(charts)} visualizations using {'code-based' if use_code_based_charts else 'AI-generated'} approach",
                "creator"
            )
            self.update_agent_status('chart_creator', 'complete')
            self.update_agent_status('insight_generator', 'generating')
            total_insights = 0
            for chart in charts:
                chart['insights'] = self.insight_generator.generate_insights(dataframe, chart, analysis_results)
                total_insights += len(chart.get('insights', []))
            self.log_conversation(
                "Insight Generator Agent",
                f"Generated {total_insights} business insights across all visualizations",
                "insight"
            )
            self.update_agent_status('insight_generator', 'complete')
            for agent in self.agent_status:
                self.update_agent_status(agent, 'idle')
            results = {
                'analysis': analysis_results,
                'charts': charts,
                'total_insights': total_insights,
                'conversation_log': self.conversation_log,
                'agent_status': self.agent_status.copy(),
                'metadata': {
                    'dataset_shape': dataframe.shape,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'use_code_based_charts': use_code_based_charts
                }
            }
            self.log_conversation("Agentic AI Coordinator", "Analysis workflow completed successfully")
            return results
        except Exception as e:
            error_msg = f"Analysis workflow failed: {str(e)}"
            self.log_conversation("Agentic AI Coordinator", error_msg, "error")
            for agent in self.agent_status:
                self.update_agent_status(agent, 'error')
            return {
                'error': error_msg,
                'conversation_log': self.conversation_log,
                'agent_status': self.agent_status.copy()
            }
    
    def create_custom_chart(self, dataframe, user_prompt, business_context=""):
        try:
            self.initialize_agents()
            enhanced_prompt = f"""
            User Request: {user_prompt}
            Business Context: {business_context or self.learning_data.get('business_context', '')}
            Dataset Info:
            - Shape: {dataframe.shape}
            - Columns: {list(dataframe.columns)}
            - Data Types: {dataframe.dtypes.to_dict()}
            """
            custom_analysis = self.data_analyst.analyze_dataset(dataframe)
            charts = self.chart_creator.create_intelligent_charts(dataframe, custom_analysis)
            if charts:
                chart = charts[0]
                chart['prompt'] = user_prompt
                chart['ai_analysis'] = self.insight_generator.generate_insights(dataframe, chart, custom_analysis)
                self.log_conversation(
                    "Custom Chart Creator",
                    f"Created custom visualization for: {user_prompt[:50]}...",
                    "custom"
                )
                return chart
            else:
                return {
                    'error': 'Could not generate chart for the given request',
                    'prompt': user_prompt
                }
        except Exception as e:
            return {
                'error': f'Custom chart creation failed: {str(e)}',
                'prompt': user_prompt
            }
    
    def process_user_feedback(self, feedback_data):
        try:
            self.learning_data['user_feedback'].append(feedback_data)
            rating_value = len(feedback_data.get('chart_rating', '⭐⭐⭐').split('⭐')) - 1
            if rating_value >= 4:
                chart_pattern = f"{feedback_data.get('chart_type')}:{feedback_data.get('x_col')}:{feedback_data.get('y_col')}"
                if chart_pattern not in self.learning_data['preferred_charts']:
                    self.learning_data['preferred_charts'].append(chart_pattern)
                self.learning_data['successful_patterns'].append({
                    'pattern': chart_pattern,
                    'feedback': feedback_data.get('feedback_text', ''),
                    'rating': rating_value,
                    'timestamp': datetime.now().isoformat()
                })
            elif rating_value <= 2:
                x_col = feedback_data.get('x_col')
                if x_col and x_col not in self.learning_data['avoided_columns']:
                    self.learning_data['avoided_columns'].append(x_col)
            if 'agent_learning' in st.session_state:
                st.session_state.agent_learning.update(self.learning_data)
            self.log_conversation(
                "Learning System",
                f"Processed feedback for {feedback_data.get('chart_title', 'chart')} - Rating: {rating_value}/5",
                "feedback"
            )
            return True
        except Exception as e:
            self.log_conversation(
                "Learning System",
                f"Failed to process feedback: {str(e)}",
                "error"
            )
            return False
    
    def get_smart_recommendations(self, dataframe):
        try:
            self.initialize_agents()
            analysis = self.data_analyst.analyze_dataset(dataframe)
            filtered_recommendations = []
            for rec in analysis.get('recommendations', []):
                x_col = rec.get('x_col')
                chart_type = rec.get('type')
                if x_col in self.learning_data.get('avoided_columns', []):
                    continue
                pattern = f"{chart_type}:{x_col}:{rec.get('y_col', '')}"
                if pattern in self.learning_data.get('preferred_charts', []):
                    rec['priority'] = 'high'
                    rec['reason'] += " (User preferred pattern)"
                filtered_recommendations.append(rec)
            return filtered_recommendations
        except Exception as e:
            self.log_conversation(
                "Smart Recommendations",
                f"Failed to generate smart recommendations: {str(e)}",
                "error"
            )
            return []
    
    def export_learning_summary(self):
        return {
            'total_feedback_sessions': len(self.learning_data.get('user_feedback', [])),
            'preferred_chart_patterns': len(self.learning_data.get('preferred_charts', [])),
            'avoided_columns': len(self.learning_data.get('avoided_columns', [])),
            'successful_patterns': len(self.learning_data.get('successful_patterns', [])),
            'average_rating': np.mean([
                len(fb.get('chart_rating', '⭐⭐⭐').split('⭐')) - 1 
                for fb in self.learning_data.get('user_feedback', [])[-10:]
            ]) if self.learning_data.get('user_feedback') else 0,
            'business_context': self.learning_data.get('business_context', ''),
            'last_activity': datetime.now().isoformat()
        }
    
    def reset_agents(self):
        self.data_analyst = None
        self.chart_creator = None  
        self.insight_generator = None
        self.conversation_log = []
        self.active_analysis = None
        for agent in self.agent_status:
            self.update_agent_status(agent, 'idle')
        self.log_conversation("System", "All agents reset successfully")
    
    def get_system_status(self):
        return {
            'agents_initialized': all([
                self.data_analyst is not None,
                self.chart_creator is not None,
                self.insight_generator is not None
            ]),
            'agent_status': self.agent_status.copy(),
            'conversation_entries': len(self.conversation_log),
            'learning_data_summary': self.export_learning_summary(),
            'active_analysis': self.active_analysis is not None,
            'system_version': self.version
        }

class YourVisualizationClass:
    def __init__(self, use_agentic_ai=False, strict_agentic_mode=False):
        """
        Initialize visualization class.
        
        Args:
            use_agentic_ai: Whether to use AI-generated charts.
            strict_agentic_mode: If True, only show AI-generated charts when agentic AI
                                is enabled, with no fallback to standard charts.
        """
        self.use_agentic_ai = use_agentic_ai
        self.strict_agentic_mode = strict_agentic_mode

def track_agentic_ai_usage(user_id, supabase):
    if st.session_state.get("user_role") == "Viewer":
        result = supabase.table("usage").select("count").eq("user_id", user_id).eq("feature", "agentic_ai").execute()
        usage_count = result.data[0]["count"] if result.data else 0
        if usage_count >= 5:
            st.error("Free usage limit reached. Upgrade to Pro for unlimited Agentic AI analyses.")
            return False
        supabase.table("usage").insert({
            "user_id": user_id,
            "feature": "agentic_ai",
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    return True

def agentic_ai_chart_tab():
    agentic_ai = AgenticAIAgent()
    st.subheader("🤖 Real Agentic AI Chart System - Intelligent Analysis")
    if not st.session_state.get("agentic_ai_enabled", False):
        st.warning("Agentic AI is disabled. Enable it in the sidebar to access advanced analytics.")
        return
    if st.session_state.dataset is None:
        st.info("No dataset loaded. Please upload a dataset in the 'Data' tab.")
        return
    if 'agent_conversations' not in st.session_state:
        st.session_state.agent_conversations = []
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'data_analyst': 'idle',
            'chart_creator': 'idle', 
            'insight_generator': 'idle'
        }
    if 'agentic_charts' not in st.session_state:
        st.session_state.agentic_charts = []
    if 'agent_recommendations' not in st.session_state:
        st.session_state.agent_recommendations = []
    if 'saved_dashboards' not in st.session_state:
        st.session_state.saved_dashboards = []
    if 'custom_charts' not in st.session_state:
        st.session_state.custom_charts = []
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = []
    if 'agent_learning' not in st.session_state:
        st.session_state.agent_learning = {
            'preferred_charts': [],
            'avoided_columns': [],
            'business_context': ""
        }
    if 'use_code_based_charts' not in st.session_state:
        st.session_state.use_code_based_charts = True
    st.markdown("""
    <style>
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    .status-idle { background-color: #718096; }
    .status-analyzing { background-color: #F59E0B; animation: pulse 2s infinite; }
    .status-creating { background-color: #3B82F6; animation: pulse 2s infinite; }
    .status-generating { background-color: #8B5CF6; animation: pulse 2s infinite; }
    .status-complete { background-color: #10B981; }
    .status-error { background-color: #EF4444; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .conversation-log {
        background-color: #1F2937;
        color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .agent-message {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-left: 3px solid;
        border-radius: 4px;
        background-color: #374151;
    }
    .analyst-message { border-color: #8B5CF6; }
    .creator-message { border-color: #3B82F6; }
    .insight-message { border-color: #10B981; }
    .chart-container {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        background-color: #1F2937;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        color: #F3F4F6;
    }
    .priority-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .priority-high { background-color: #7F1D1D; color: #FEE2E2; }
    .priority-medium { background-color: #78350F; color: #FEF3C7; }
    .priority-low { background-color: #312E81; color: #E0E7FF; }
    .metric-container {
        font-size: 0.8rem !important;
        background-color: #374151;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        color: #F3F4F6;
        margin: 0.25rem;
        border: 1px solid #4B5563;
    }
    .metric-container .metric-value {
        font-size: 1.2rem !important;
        color: #60A5FA;
        font-weight: bold;
        display: block;
        margin-bottom: 0.25rem;
    }
    .metric-container .metric-label {
        font-size: 0.75rem !important;
        color: #E5E7EB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .insights-container {
        margin-top: 1rem;
        background-color: #374151;
        padding: 1rem;
        border-radius: 8px;
        color: #93C5FD;
        border: 1px solid #4B5563;
    }
    .insights-container h3 {
        color: #60A5FA;
        margin-bottom: 0.75rem;
    }
    .insights-container p, .insights-container li {
        color: #93C5FD;
        line-height: 1.5;
    }
    .insights-container strong {
        color: #DBEAFE;
    }
    .js-plotly-plot {
        background-color: #111827 !important;
    }
    .custom-chart-section {
        border: 2px solid #8B5CF6;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #1F2937;
        color: #93C5FD;
    }
    .custom-chart-section h4 {
        color: #DBEAFE;
    }
    .chart-container {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        background-color: #1F2937;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        color: #93C5FD;
    }
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        color: #DBEAFE;
    }
    .chart-header h4 {
        color: #DBEAFE;
    }
    p, li, span {
        color: #93C5FD;
    }
    strong {
        color: #DBEAFE;
    }
    .code-section {
        background-color: #1F2937;
        padding: 10px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
        color: #93C5FD;
        margin-top: 10px;
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    def save_dashboard():
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': st.session_state.agent_recommendations,
            'conversations': st.session_state.agent_conversations,
            'custom_charts': st.session_state.custom_charts,
            'data_analysis': st.session_state.data_analysis,
            'dataset_info': {
                'rows': len(st.session_state.dataset),
                'columns': list(st.session_state.dataset.columns),
            }
        }
        return dashboard_data
    
    def export_to_pdf():
        df = st.session_state.dataset
        business_cols = get_business_relevant_columns(df)
        chart_images = []
        for idx, chart in enumerate(st.session_state.agent_recommendations):
            try:
                chart_html = chart['figure'].to_html(include_plotlyjs='cdn', div_id=f"chart_{idx}")
                chart_images.append({
                    'title': chart['title'],
                    'html': chart_html,
                    'reason': chart.get('reason', ''),
                    'insights': chart.get('insights', []),
                    'priority': chart.get('priority', 'medium'),
                    'code': chart.get('code', '')
                })
            except:
                chart_images.append({
                    'title': chart['title'],
                    'html': '<div>Chart could not be exported</div>',
                    'reason': chart.get('reason', ''),
                    'insights': chart.get('insights', []),
                    'priority': chart.get('priority', 'medium'),
                    'code': chart.get('code', '')
                })
        custom_chart_images = []
        for idx, custom_chart in enumerate(st.session_state.custom_charts):
            if isinstance(custom_chart, dict) and custom_chart.get('figure'):
                try:
                    chart_html = custom_chart['figure'].to_html(include_plotlyjs='cdn', div_id=f"custom_chart_{idx}")
                    custom_chart_images.append({
                        'prompt': custom_chart.get('prompt', f'Custom Chart {i+1}'),
                        'html': chart_html,
                        'analysis': custom_chart.get('ai_analysis', 'No analysis available'),
                        'code': custom_chart.get('code', '')
                    })
                except:
                    custom_chart_images.append({
                        'prompt': custom_chart.get('prompt', f'Custom Chart {i+1}'),
                        'html': '<div>Custom chart could not be exported</div>',
                        'analysis': custom_chart.get('ai_analysis', 'No analysis available'),
                        'code': custom_chart.get('code', '')})


        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>NarraViz.ai - AI Dashboard Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #1F2937; 
                    color: #93C5FD; 
                    line-height: 1.6;
                }}
                .header {{ 
                    color: #60A5FA; 
                    text-align: center; 
                    border-bottom: 2px solid #374151;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .beta-banner {{
                    background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 8px;
                    text-align: center;
                    margin: 20px 0;
                    font-weight: bold;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }}
                .branding {{
                    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 12px;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .branding h2 {{
                    margin: 0;
                    font-size: 1.5em;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .branding p {{
                    margin: 5px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .section {{ 
                    margin: 30px 0; 
                    padding: 20px; 
                    background-color: #374151; 
                    border-radius: 8px;
                    border: 1px solid #4B5563;
                }}
                .insight {{ 
                    margin: 10px 0; 
                    padding: 15px; 
                    background-color: #1F2937; 
                    border-left: 4px solid #60A5FA;
                    border-radius: 4px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #111827;
                    border-radius: 8px;
                    border: 1px solid #374151;
                }}
                .chart-title {{
                    color: #DBEAFE;
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .priority-badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .priority-high {{ background-color: #7F1D1D; color: #FEE2E2; }}
                .priority-medium {{ background-color: #78350F; color: #FEF3C7; }}
                .priority-low {{ background-color: #312E81; color: #E0E7FF; }}
                .business-context {{
                    background-color: #8B5CF6;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .data-overview {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #1F2937;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #374151;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #60A5FA;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #9CA3AF;
                    text-transform: uppercase;
                }}
                .conversation-log {{
                    background-color: #111827;
                    padding: 15px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 0.9em;
                }}
                .agent-message {{
                    margin: 8px 0;
                    padding: 10px;
                    border-left: 3px solid #8B5CF6;
                    background-color: #374151;
                    border-radius: 4px;
                }}
                h1, h2, h3 {{ color: #DBEAFE; }}
                strong {{ color: #DBEAFE; }}
                .feedback-summary {{
                    background-color: #065F46;
                    color: #D1FAE5;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    background-color: #374151;
                    border: 1px solid #4B5563;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    color: #9CA3AF;
                    font-size: 0.9em;
                    line-height: 1.4;
                }}
                .disclaimer h3 {{
                    color: #F59E0B;
                    margin-top: 0;
                }}
                .footer {{
                    text-align: center; 
                    margin-top: 40px; 
                    padding: 20px; 
                    border-top: 2px solid #374151;
                    background: linear-gradient(135deg, #1F2937 0%, #374151 100%);
                    border-radius: 8px;
                }}
                .footer .logo {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #60A5FA;
                    margin-bottom: 10px;
                }}
                .footer .tagline {{
                    color: #93C5FD;
                    font-size: 1.1em;
                    margin-bottom: 15px;
                }}
                .footer .tech {{
                    color: #9CA3AF;
                    font-size: 0.9em;
                    font-style: italic;
                }}
                .code-section {{
                    background-color: #1F2937;
                    padding: 10px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 0.9em;
                    color: #93C5FD;
                    margin-top: 10px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <div class="branding">
                <h2>🤖 NarraViz.ai - AI Dashboard Report</h2>
                <p>Powered by Real Agentic AI & Machine Learning</p>
            </div>
            <div class="beta-banner">
                🚧 BETA VERSION - Advanced AI Analysis System in Development 🚧
            </div>
            <div class="header">
                <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Dataset:</strong> {len(df):,} rows × {len(df.columns)} columns</p>
            </div>
            <div class="disclaimer">
                <h3>⚠️ Important Beta Disclaimers</h3>
                <ul>
                    <li><strong>Beta Software:</strong> This AI system is in active development. Results should be validated before making business decisions.</li>
                    <li><strong>AI Limitations:</strong> AI-generated insights are based on statistical patterns and may not capture all business context or nuances.</li>
                    <li><strong>Data Responsibility:</strong> Users are responsible for data quality, privacy, and ensuring appropriate use of generated insights.</li>
                    <li><strong>Continuous Learning:</strong> The system improves with feedback - your ratings help train better AI recommendations.</li>
                    <li><strong>Professional Review:</strong> Always have domain experts review AI findings before implementing strategic decisions.</li>
                </ul>
            </div>
            <div class="section">
                <h2>📊 Smart Data Overview</h2>
                <div class="data-overview">
                    <div class="metric-card">
                        <div class="metric-value">{len(df):,}</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal'])}</div>
                        <div class="metric-label">Business Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal'])}</div>
                        <div class="metric-label">Excluded IDs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%</div>
                        <div class="metric-label">Data Quality</div>
                    </div>
                </div>
                <h3>📈 Business Columns Analyzed:</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div>
                        <strong>📊 Numerical (Metrics):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['numerical']])}</ul>
                    </div>
                    <div>
                        <strong>🏷️ Categorical (Dimensions):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['categorical']])}</ul>
                    </div>
                    <div>
                        <strong>📅 Temporal (Time):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['temporal']])}</ul>
                    </div>
                </div>
            </div>
            {f'''
            <div class="business-context">
                <h2>🏢 Business Context</h2>
                <p>{st.session_state.agent_learning.get('business_context', 'No business context provided yet.')}</p>
            </div>
            ''' if st.session_state.agent_learning.get('business_context') else ''}
            <div class="section">
                <h2>🧠 AI Data Analysis Results</h2>
                <div class="insight">
                    {json.dumps(st.session_state.data_analysis, indent=2) if st.session_state.data_analysis else 'AI analysis not yet completed. Run the AI analysis to see intelligent insights here.'}
                </div>
            </div>
            {f'''
            <div class="section">
                <h2>🤖 AI Agent Collaboration Log</h2>
                <div class="conversation-log">
                    {''.join([f'<div class="agent-message"><strong>{msg["agent"]}:</strong> {msg["message"]}</div>' for msg in st.session_state.agent_conversations])}
                </div>
            </div>
            ''' if st.session_state.agent_conversations else ''}
            <div class="section">
                <h2>📊 AI-Generated Intelligent Visualizations</h2>
                <p><em>Our AI agents created {len(chart_images)} optimized visualizations based on intelligent data analysis:</em></p>
                {''.join([f'''
                <div class="chart-container">
                    <div class="chart-title">
                        {chart["title"]}
                        <span class="priority-badge priority-{chart["priority"]}">{chart["priority"].upper()}</span>
                    </div>
                    <p><strong>🤖 AI Reasoning:</strong> {chart["reason"]}</p>
                    <div style="margin: 20px 0;">
                        {chart["html"]}
                    </div>
                    <h3>🧠 AI-Generated Insights:</h3>
                    {''.join([f'<div class="insight">• {insight}</div>' for insight in chart["insights"][:4]])}
                    <h3>💻 Generated Code:</h3>
                    <div class="code-section"><pre>{chart["code"]}</pre></div>
                </div>
                ''' for chart in chart_images])}
            </div>
            {f'''
            <div class="section">
                <h2>🎯 Custom AI-Generated Charts</h2>
                {''.join([f'''
                <div class="chart-container">
                    <div class="chart-title">🤖 Custom Analysis: {custom_chart["prompt"]}</div>
                    <div style="margin: 20px 0;">
                        {custom_chart["html"]}
                    </div>
                    <h3>🤖 AI Analysis:</h3>
                    <div class="insight">{custom_chart["analysis"]}</div>
                    <h3>💻 Generated Code:</h3>
                    <div class="code-section"><pre>{custom_chart["code"]}</pre></div>
                </div>
                ''' for custom_chart in custom_chart_images])}
            </div>
            ''' if custom_chart_images else ''}
            {f'''
            <div class="feedback-summary">
                <h2>📚 AI Learning Progress</h2>
                <div class="data-overview">
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.user_feedback)}</div>
                        <div class="metric-label">Feedback Sessions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.agent_learning.get('preferred_charts', []))}</div>
                        <div class="metric-label">Preferred Patterns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.agent_learning.get('avoided_columns', []))}</div>
                        <div class="metric-label">Avoided Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{np.mean([len(fb['chart_rating'].split('⭐')) - 1 for fb in st.session_state.user_feedback[-10:]]) if st.session_state.user_feedback else 0:.1f}/5</div>
                        <div class="metric-label">Avg Chart Rating</div>
                    </div>
                </div>
                <h3>Recent User Feedback:</h3>
                {''.join([f'<div class="insight"><strong>{fb["chart_title"]}</strong> - {fb["chart_rating"]} - {fb["feedback_text"]}</div>' for fb in st.session_state.user_feedback[-5:]])}
            </div>
            ''' if st.session_state.user_feedback else ''}
            <div class="section">
                <h2>🎯 Summary & Recommendations</h2>
                <div class="insight">
                    <strong>Key Findings:</strong>
                    <ul>
                        <li>AI analyzed {len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal'])} business-relevant columns while excluding {len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal'])} ID fields</li>
                        <li>Generated {len(chart_images)} intelligent visualizations with business-specific insights</li>
                        <li>Created {len(custom_chart_images)} custom analyses based on user requests</li>
                        {f"<li>Incorporated {len(st.session_state.user_feedback)} feedback sessions for continuous AI improvement</li>" if st.session_state.user_feedback else ""}
                    </ul>
                </div>
                <div class="insight">
                    <strong>Next Steps:</strong>
                    <ul>
                        <li>Review the AI-generated insights for actionable business opportunities</li>
                        <li>Focus on high-priority visualizations for strategic decision-making</li>
                        <li>Use the feedback system to train AI for better future recommendations</li>
                        <li>Explore custom analysis requests for specific business questions</li>
                        <li>Validate findings with domain experts before implementation</li>
                        <li>Consider expanding analysis with additional data sources</li>
                    </ul>
                </div>
            </div>
            <div class="footer">
                <div class="logo">🤖 NarraViz.ai</div>
                <div class="tagline">Powered by Real Agentic AI & Intelligent Data Analysis</div>
                <div class="tech">Advanced Machine Learning • Natural Language Processing • Smart Data Analytics</div>
                <br>
                <div style="color: #F59E0B; font-weight: bold;">BETA VERSION - Continuously Learning & Improving</div>
                <div style="color: #9CA3AF; font-size: 0.8em; margin-top: 10px;">
                    Generated by the Real Agentic AI Chart System • Always validate insights with domain experts
                </div>
            </div>
        </body>
        </html>
        """
        return html_content.encode()

    df = st.session_state.dataset
    st.markdown("### 🔍 Smart Data Overview")
    business_cols = get_business_relevant_columns(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Business Columns", len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal']))
    with col3:
        st.metric("Excluded IDs", len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal']))
    with col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    with st.expander("🤖 What AI Will Analyze", expanded=False):
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        with col_diag1:
            st.markdown("**📊 Numerical (Metrics)**")
            for col in business_cols['numerical']:
                st.markdown(f"• {col}")
        with col_diag2:
            st.markdown("**🏷️ Categorical (Dimensions)**") 
            for col in business_cols['categorical']:
                st.markdown(f"• {col}")
        with col_diag3:
            st.markdown("**📅 Temporal (Time)**")
            for col in business_cols['temporal']:
                st.markdown(f"• {col}")
            st.markdown("**🚫 Excluded (IDs)**")
            excluded = [col for col in df.columns if col not in business_cols['categorical'] + business_cols['numerical'] + business_cols['temporal']]
            for col in excluded[:5]:
                st.markdown(f"• {col}")
            if len(excluded) > 5:
                st.markdown(f"• ...and {len(excluded)-5} more")
    if not USE_OPENAI:
        st.error("🔑 OpenAI API key not configured. Please check your secrets.toml file.")
        st.info("💡 Add your OpenAI key to .streamlit/secrets.toml:\n```\n[openai]\napi_key = \"your-key-here\"\n```")
        return
    
    # Initialize AI Agents
    data_analyst = DataAnalystAgent()
    chart_creator = ChartCreatorAgent()
    insight_generator = InsightGeneratorAgent()


    # Chart Generation Mode
    use_code = st.checkbox(
        "Use Code-Based Charts",
         value=st.session_state.get("use_code_based_charts", True),
         help="Enable to generate charts using verified Python code (reduces AI hallucination). Disable for fully AI-generated charts.",
         key="use_code_based_charts"
    )
 

    # Header with controls
    col1, col2, col3, col4 = st.columns([2, 0.8, 0.8, 0.8])
    with col1:
        st.markdown("### 🤖 Real Agentic AI - Intelligent Data Analysis")
        st.info("Watch AI agents analyze your data and create visualizations!")
    with col2:
        if st.button("🧠 Run AI Analysis", type="primary"):
            st.session_state.agent_recommendations = []
            st.session_state.agent_conversations = []
            st.session_state.custom_charts = []
            st.session_state.data_analysis = None
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.agent_recommendations = []
            st.session_state.agent_conversations = []
            st.session_state.custom_charts = []
            st.session_state.data_analysis = None
    with col4:
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("💾 Save"):
                dashboard = save_dashboard()
                st.session_state.saved_dashboards.append(dashboard)
                st.success("Dashboard saved!")
        with save_col2:
            if st.button("📄 Export"):
                pdf_data = export_to_pdf()
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="ai_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # Agent Status Display
    st.markdown("### 🤖 AI Agent Status")
    agent_cols = st.columns(3)
    agents_info = [
        ("🧠 Data Analyst", "data_analyst", "Analyzing patterns with AI..."),
        ("📊 Chart Creator", "chart_creator", "Designing optimal visualizations..."),
        ("💡 Insight Generator", "insight_generator", "Generating business insights...")
    ]
    
    for col, (name, key, description) in zip(agent_cols, agents_info):
        with col:
            status = st.session_state.agent_status[key]
            st.markdown(f"""
            <div class="agent-card">
                <strong>{name}</strong>
                <span class="agent-status status-{status}">{status.upper()}</span>
                <br><small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Run AI Analysis
    if not st.session_state.agent_recommendations and st.session_state.data_analysis is None:
        with st.spinner("🤖 AI Agents are intelligently analyzing your data..."):
            st.session_state.agent_status['data_analyst'] = 'analyzing'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">🧠 Data Analyst Agent is examining data patterns...</div>', unsafe_allow_html=True)
            
            analysis = data_analyst.analyze_dataset(df)
            st.session_state.data_analysis = analysis
            
            st.session_state.agent_conversations.append({
                'agent': 'Data Analyst Agent',
                'message': f"📊 Identified {len(analysis.get('recommendations', []))} visualization opportunities.",
                'type': 'analyst',
                'details': analysis
            })
            
            st.session_state.agent_status['data_analyst'] = 'complete'
            
            st.session_state.agent_status['chart_creator'] = 'creating'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">📊 Chart Creator Agent is generating visualizations...</div>', unsafe_allow_html=True)
            
            charts = chart_creator.create_intelligent_charts(df, analysis)
            st.session_state.agent_conversations.append({
                'agent': 'Chart Creator Agent',
                'message': f"🎨 Created {len(charts)} visualizations using {'code-based' if st.session_state.use_code_based_charts else 'AI-generated'} approach.",
                'type': 'creator'
            })
            
            st.session_state.agent_status['chart_creator'] = 'complete'
            
            st.session_state.agent_status['insight_generator'] = 'generating'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">💡 Insight Generator Agent is analyzing charts...</div>', unsafe_allow_html=True)
            
            for chart in charts:
                chart['insights'] = insight_generator.generate_insights(df, chart, analysis)
            
            st.session_state.agent_recommendations = charts
            
            st.session_state.agent_conversations.append({
                'agent': 'Insight Generator Agent',
                'message': f"💡 Generated {sum(len(chart.get('insights', [])) for chart in charts)} insights across all visualizations.",
                'type': 'insight'
            })
            
            st.session_state.agent_status['insight_generator'] = 'complete'
            
            for key in st.session_state.agent_status:
                st.session_state.agent_status[key] = 'idle'
    
    # Display Agent Conversation Log
    if st.session_state.agent_conversations:
        with st.expander("🤖 AI Agent Collaboration Log", expanded=False):
            st.markdown('<div class="conversation-log">', unsafe_allow_html=True)
            for msg in st.session_state.agent_conversations:
                msg_class = f"{msg['type']}-message"
                st.markdown(f'<div class="agent-message {msg_class}"><strong>{msg["agent"]}:</strong> {msg["message"]}</div>', unsafe_allow_html=True)
                if msg['type'] == 'analyst' and 'details' in msg:
                    details = msg['details']
                    st.markdown(f'<div style="margin-left: 1rem; font-size: 0.9rem; color: #9CA3AF;">')
                    st.markdown(f'<strong>Recommendations:</strong> {len(details.get("recommendations", []))} visualization opportunities')
                    st.markdown('</div>')
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display AI-Generated Charts
    if st.session_state.agent_recommendations:
        st.markdown("### 🧠 AI-Generated Intelligent Visualizations")
        st.markdown(f"*Created {len(st.session_state.agent_recommendations)} visualizations:*")
        
        for i, chart in enumerate(st.session_state.agent_recommendations):
            st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="chart-header">
                <h4>{chart['title']}</h4>
                <span class="priority-badge priority-{chart['priority']}">{chart['priority'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"🤖 AI Reasoning: {chart['reason']}")
            
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.plotly_chart(chart['figure'], use_container_width=True, key=f"rec_chart-{i}")
                
                with st.expander("💻 Generated Code", expanded=False):
                    st.code(chart['code'], language="python")
                
                if chart.get('data') is not None and chart.get('y_col') and hasattr(chart['data'], 'columns') and chart['y_col'] in chart['data'].columns:
                    col_stats = st.columns(4)
                    data = chart['data']
                    y_col = chart['y_col']
                    with col_stats[0]:
                        total_val = f"${data[y_col].sum():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{total_val}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
                    with col_stats[1]:
                        avg_val = f"${data[y_col].mean():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{avg_val}</div><div class="metric-label">Average</div></div>', unsafe_allow_html=True)
                    with col_stats[2]:
                        max_val = f"${data[y_col].max():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{max_val}</div><div class="metric-label">Max</div></div>', unsafe_allow_html=True)
                    with col_stats[3]:
                        count_val = f"{len(data):,}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{count_val}</div><div class="metric-label">Count</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                st.markdown("### 🧠 Insights")
                for insight in chart.get('insights', [])[:4]:
                    st.markdown(f"• **{insight}**")
                
                st.markdown("---")
                st.markdown("#### 📝 Rate & Improve")
                col_rate1, col_rate2 = st.columns(2)
                with col_rate1:
                    chart_rating = st.selectbox(
                        "Chart Usefulness", 
                        ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Great", "⭐⭐⭐⭐⭐ Excellent"],
                        key=f"rating_{i}",
                        index=2
                    )
                with col_rate2:
                    insights_rating = st.selectbox(
                        "Insights Quality",
                        ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Great", "⭐⭐⭐⭐⭐ Excellent"],
                        key=f"insights_rating_{i}",
                        index=2
                    )
                
                feedback_text = st.text_area(
                    "Feedback for AI",
                    placeholder="How can this chart/insight be improved?",
                    key=f"feedback_{i}",
                    height=68
                )
                
                if st.button(f"📚 Train AI", key=f"train_{i}"):
                    feedback_data = {
                        'chart_title': chart['title'],
                        'chart_type': chart['type'],
                        'x_col': chart['x_col'],
                        'y_col': chart['y_col'],
                        'chart_rating': chart_rating,
                        'insights_rating': insights_rating,
                        'feedback_text': feedback_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.user_feedback.append(feedback_data)
                    rating_value = len(chart_rating.split('⭐')) - 1
                    if rating_value >= 4:
                        chart_pattern = f"{chart['type']}:{chart['x_col']}:{chart['y_col']}"
                        if chart_pattern not in st.session_state.agent_learning['preferred_charts']:
                            st.session_state.agent_learning['preferred_charts'].append(chart_pattern)
                    elif rating_value <= 2:
                        if chart['x_col'] not in st.session_state.agent_learning['avoided_columns']:
                            st.session_state.agent_learning['avoided_columns'].append(chart['x_col'])
                    st.success("✅ Feedback saved! AI will learn from this.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Business Context Section
    st.markdown("---")
    st.markdown("### 🏢 Business Context & Custom Analysis")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        business_context = st.text_area(
            "Provide Business Context",
            value=st.session_state.agent_learning.get('business_context', ''),
            placeholder="e.g., We're a retail company focusing on Q4 sales performance...",
            height=100
        )
        if st.button("💾 Save Context"):
            st.session_state.agent_learning['business_context'] = business_context
            st.success("Business context saved!")
    
    with col2:
        custom_prompt = st.text_area(
            "Request Custom Chart",
            placeholder="e.g., Show me seasonal trends in revenue by product category",
            height=100
        )
        if st.button("🎯 Create Custom Chart"):
            with st.spinner("Creating custom visualization..."):
                custom_chart = agentic_ai.create_custom_chart(df, custom_prompt, business_context)
                if 'error' not in custom_chart:
                    st.session_state.custom_charts.append(custom_chart)
                    st.success("Custom chart created!")
                else:
                    st.error(custom_chart['error'])
    
    # Display Custom Charts
    if st.session_state.custom_charts:
        st.markdown("---")
        st.markdown("### 🎯 Custom AI-Generated Charts")
        
        for i, custom_chart in enumerate(st.session_state.custom_charts):
            if isinstance(custom_chart, dict):
                st.markdown(f'<div class="custom-chart-section">', unsafe_allow_html=True)
                st.markdown(f"#### 🤖 AI Custom Chart #{i+1}: {custom_chart.get('prompt', 'Custom Analysis')}")
                
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    if custom_chart.get('figure'):
                        st.plotly_chart(custom_chart['figure'], use_container_width=True, key=f"custom_chart-{i}")
                    with st.expander("💻 Generated Code", expanded=False):
                        st.code(custom_chart.get('code', 'No code available'), language="python")
                
                with col2:
                    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                    st.markdown("### 🤖 AI Analysis")
                    if isinstance(custom_chart.get('ai_analysis'), list):
                        for insight in custom_chart['ai_analysis']:
                            st.markdown(f"• **{insight}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
