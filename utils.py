
import pandas as pd
import logging
import json
import os
import uuid
from supabase import create_client
import re
import time
from io import StringIO

def setup_logging():
    """Configure logging for the entire application."""
    if not logging.getLogger().handlers:  # Only configure if no handlers exist
        logging.basicConfig(
            filename="chartgpt.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize Supabase client
try:
    supabase = create_client("https://fyyvfaqiohdxhnbdqoxu.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5eXZmYXFpb2hkeGhuYmRxb3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NTA2MTYsImV4cCI6MjA2MzEyNjYxNn0.-h6sm3bgPzxDjxlmPhi5LNzsbhMJiz8-0HX80U7FiZc")
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    raise

def classify_columns(df, existing_field_types=None):
    """
    Dataset-agnostic column classification using multiple heuristics.
    Works with any domain: retail, finance, healthcare, etc.
    """
    if existing_field_types is None:
        existing_field_types = {}
    
    dimensions = existing_field_types.get("dimension", [])
    measures = existing_field_types.get("measure", [])
    dates = existing_field_types.get("date", [])
    ids = existing_field_types.get("id", [])
    
    logger.info("Dataset columns: %s", list(df.columns))
    logger.info("Dataset dtypes: %s", {col: df[col].dtype for col in df.columns})
    logger.info("Existing field_types: %s", existing_field_types)
    
    # Dynamic thresholds based on dataset size
    dataset_size = len(df)
    if dataset_size < 100:
        unique_threshold = 0.3  # Smaller datasets need higher threshold
    elif dataset_size < 1000:
        unique_threshold = 0.1
    else:
        unique_threshold = 0.05  # Large datasets use lower threshold
    
    for col in df.columns:
        # Skip already classified columns
        if col in dimensions or col in measures or col in dates or col in ids:
            continue
            
        col_lower = col.lower().strip()
        col_data = df[col]
        
        # RULE 1: ID Detection (multiple patterns)
        id_patterns = ['id', 'key', 'identifier', 'uuid', 'guid']
        if any(pattern in col_lower for pattern in id_patterns):
            ids.append(col)
            logger.info("Classified %s as ID (contains ID pattern)", col)
            continue
            
        # RULE 2: Date Detection (enhanced)
        if pd.api.types.is_datetime64_any_dtype(col_data):
            dates.append(col)
            logger.info("Classified %s as Date (datetime type)", col)
            continue
            
        # Try to convert date-like columns
        date_patterns = ['date', 'time', 'created', 'updated', 'modified', 'timestamp']
        if any(pattern in col_lower for pattern in date_patterns):
            try:
                # Test conversion on a sample
                sample_size = min(100, len(col_data))
                test_sample = col_data.dropna().head(sample_size)
                if len(test_sample) > 0:
                    pd.to_datetime(test_sample)
                    df[col] = pd.to_datetime(col_data, errors='coerce')
                    dates.append(col)
                    logger.info("Explicitly converted %s to datetime due to date pattern", col)
                    continue
            except Exception as e:
                logger.debug("Could not convert %s to datetime: %s", col, str(e))
        
        # RULE 3: Numeric Data Classification
        if pd.api.types.is_numeric_dtype(col_data):
            unique_ratio = col_data.nunique() / len(col_data)
            
            # Geographic/Postal codes (even if numeric)
            geo_patterns = ['postal', 'zip', 'code', 'area', 'region']
            if any(pattern in col_lower for pattern in geo_patterns):
                dimensions.append(col)
                logger.info("Classified %s as Dimension (geographic code)", col)
                continue
            
            # Check if numeric data represents categories
            max_val = col_data.max() if not col_data.isna().all() else 0
            unique_count = col_data.nunique()
            
            # Decision logic for numeric data
            if unique_ratio < unique_threshold and unique_count < 50:
                # Low cardinality numeric = likely categorical
                dimensions.append(col)
                logger.info("Classified %s as Dimension (numeric categorical, ratio=%.3f, unique=%d)", 
                           col, unique_ratio, unique_count)
            elif col_lower in ['year', 'month', 'day', 'quarter', 'week']:
                # Time components are usually dimensions
                dimensions.append(col)
                logger.info("Classified %s as Dimension (time component)", col)
            else:
                # High cardinality or clear numeric measure
                measures.append(col)
                logger.info("Classified %s as Measure (numeric, ratio=%.3f)", col, unique_ratio)
            continue
        
        # RULE 4: Text/String Data
        if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            unique_ratio = col_data.nunique() / len(col_data)
            unique_count = col_data.nunique()
            
            # Check for potential measures stored as text (with currency, percentages, etc.)
            sample_values = col_data.dropna().astype(str).head(50)
            numeric_pattern_count = 0
            
            for val in sample_values:
                val_clean = val.strip()
                # Remove common non-numeric characters
                val_numeric = re.sub(r'[\$\%\,\€\£\¥]', '', val_clean)
                try:
                    float(val_numeric)
                    numeric_pattern_count += 1
                except:
                    pass
            
            # If most text values look like numbers, treat as measure
            if len(sample_values) > 0 and numeric_pattern_count / len(sample_values) > 0.8:
                measures.append(col)
                logger.info("Classified %s as Measure (text-encoded numeric)", col)
                continue
            
            # High cardinality text might be IDs or names
            if unique_ratio > 0.7 and unique_count > 100:
                # Very high cardinality - likely IDs or unique identifiers
                name_patterns = ['name', 'title', 'description', 'comment', 'note']
                if any(pattern in col_lower for pattern in name_patterns):
                    dimensions.append(col)
                    logger.info("Classified %s as Dimension (high cardinality name field)", col)
                else:
                    ids.append(col)
                    logger.info("Classified %s as ID (high cardinality text)", col)
            else:
                # Normal categorical data
                dimensions.append(col)
                logger.info("Classified %s as Dimension (categorical text, ratio=%.3f)", col, unique_ratio)
            continue
        
        # RULE 5: Boolean Data
        if pd.api.types.is_bool_dtype(col_data):
            dimensions.append(col)
            logger.info("Classified %s as Dimension (boolean)", col)
            continue
        
        # RULE 6: Fallback for unknown types
        logger.warning("Unknown data type for column %s (dtype: %s), defaulting to Dimension", col, col_data.dtype)
        dimensions.append(col)
    
    # POST-PROCESSING: Ensure we have at least some measures and dimensions
    if not measures and dimensions:
        # Find the most numeric-looking dimension to convert to measure
        for dim in dimensions[:]:
            if pd.api.types.is_numeric_dtype(df[dim]):
                dimensions.remove(dim)
                measures.append(dim)
                logger.info("Moved %s from Dimension to Measure (ensuring measures exist)", dim)
                break
    
    if not dimensions and measures:
        # Find the most categorical-looking measure to convert to dimension
        for measure in measures[:]:
            if df[measure].nunique() < 20:
                measures.remove(measure)
                dimensions.append(measure)
                logger.info("Moved %s from Measure to Dimension (ensuring dimensions exist)", measure)
                break
    
    logger.info("Final Classified columns - Dimensions: %s, Measures: %s, Dates: %s, IDs: %s", 
                dimensions, measures, dates, ids)
    
    return dimensions, measures, dates, ids

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise

def generate_unique_id():
    return str(uuid.uuid4())

def log_session_state(session_state, context=""):
    """Log session state keys and token excerpts."""
    logger.info(f"Session state {context} keys: {list(session_state.keys())}")
    logger.info(f"Stored access_token {context}: {session_state.get('access_token', 'None')[:10]}...")
    logger.info(f"Stored refresh_token {context}: {session_state.get('refresh_token', 'None')[:10]}...")

def save_dashboard(supabase, project_id, dashboard_name, charts, user_id, session_state, dataset=None):
    logger.info(f"Attempting to save dashboard: name={dashboard_name}, project_id={project_id}, user_id={user_id}, charts={charts}")
    dashboard_id = generate_unique_id()
    try:
        # Log session state for debugging
        log_session_state(session_state, "before saving dashboard")
        # Verify and reapply session
        session = supabase.auth.get_session()
        if not session:
            if "access_token" in session_state and "refresh_token" in session_state:
                logger.info("Attempting to refresh session with stored refresh token")
                supabase.auth.set_session(session_state["access_token"], session_state["refresh_token"])
                session = supabase.auth.get_session()
                if not session:
                    logger.error("Failed to reapply Supabase session")
                    raise ValueError("Authentication failed. Please log in again.")
            else:
                logger.error("No active Supabase session or stored tokens")
                raise ValueError("No valid session. Please log in again.")
        logger.info(f"Supabase session active: access_token={session.access_token[:10] + '...'}")

        # Validate charts
        if not isinstance(charts, list):
            logger.error(f"Invalid charts format: expected list, got {type(charts)}")
            raise ValueError("Charts must be a list of chart configurations")
        unique_charts = []
        seen = set()
        for chart in charts:
            if not isinstance(chart, dict) or "prompt" not in chart:
                logger.warning(f"Skipping invalid chart: {chart}")
                continue
            chart_tuple = (chart.get("prompt", ""), chart.get("chart_type", ""))
            if chart_tuple not in seen:
                seen.add(chart_tuple)
                unique_charts.append(chart)
            else:
                logger.info(f"Removed duplicate chart: prompt={chart.get('prompt')}, chart_type={chart.get('chart_type')}")
        if not unique_charts:
            logger.warning("No valid charts provided after deduplication")
            raise ValueError("At least one valid chart is required")

        # Save dataset to Supabase Storage if provided
        dataset_path = None
        if dataset is not None:
            try:
                bucket_name = "datasets"
                file_name = f"{dashboard_id}/{dashboard_name}_dataset.csv"
                # Convert DataFrame to CSV bytes
                csv_buffer = StringIO()
                dataset.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")
                # Upload to Supabase Storage
                response = supabase.storage.from_(bucket_name).upload(
                    file_name, csv_bytes, file_options={"content-type": "text/csv"}
                )
                logger.info(f"Storage upload response: {response}")
                dataset_path = file_name
                logger.info(f"Uploaded dataset to Supabase Storage: {file_name}")
            except Exception as e:
                logger.error(f"Error uploading dataset to Supabase Storage: {str(e)}")
                raise ValueError(f"Failed to upload dataset: {str(e)}")

        # Insert dashboard with dataset path
        dashboard_data = {
            "id": dashboard_id,
            "project_id": project_id,
            "name": dashboard_name,
            "charts": unique_charts,
            "owner_id": user_id,
            "created_at": datetime.now().isoformat(),
            "dataset_path": dataset_path
        }
        try:
            response = supabase.table("dashboards").insert(dashboard_data).execute()
            logger.info(f"Inserted dashboard: response={response.data}")
        except Exception as e:
            logger.error(f"Supabase insertion failed: {str(e)}")
            raise ValueError(f"Failed to insert dashboard: {str(e)}")

        # Insert permission
        try:
            permission_response = supabase.table("permissions").insert({
                "dashboard_id": dashboard_id,
                "user_id": user_id,
                "role": "Admin"
            }).execute()
            logger.info(f"Inserted permission: response={permission_response.data}")
        except Exception as e:
            logger.warning(f"Failed to insert permission: {str(e)}. Continuing without permission.")

        logger.info(f"Saved dashboard '{dashboard_name}' (ID: {dashboard_id}) for project: {project_id}, user: {user_id}")
        session_state.refresh_dashboards = True
        return dashboard_id
    except Exception as e:
        logger.error(f"Error saving dashboard: {str(e)}", exc_info=True)
        st.error(f"Failed to save dashboard: {str(e)}")
        return None


def load_dashboards(supabase, user_id, session_state, limit=10, offset=0):
    logger.info(f"Loading dashboards for user_id: {user_id}, limit={limit}, offset={offset}")
    try:
        # Verify session
        if "session_verified" not in session_state or not session_state["session_verified"]:
            session = supabase.auth.get_session()
            if not session and "access_token" in session_state and "refresh_token" in session_state:
                supabase.auth.set_session(session_state["access_token"], session_state["refresh_token"])
                session = supabase.auth.get_session()
                if not session:
                    logger.error("Authentication failed. Please log in again.")
                    return pd.DataFrame()
                session_state["session_verified"] = True
                logger.info(f"Session reapplied: access_token={session.access_token[:10] + '...'}")
            elif session:
                session_state["session_verified"] = True
                logger.info(f"Supabase session active: access_token={session.access_token[:10] + '...'}")
            else:
                logger.warning("No active Supabase session or stored tokens")
        
        # Fetch metadata columns including charts
        start_time = time.time()
        response = supabase.table("dashboards").select("id, name, project_id, created_at, tags, dataset_path, charts").or_(
            f"owner_id.eq.{user_id},id.in.(select dashboard_id from permissions where user_id = '{user_id}')"
        ).range(offset, offset + limit - 1).order("created_at", desc=True).execute()
        logger.info(f"Supabase query took {time.time() - start_time:.2f} seconds")
        logger.debug(f"Raw Supabase response: {response.data}")
        
        dashboards = pd.DataFrame(response.data)
        logger.info(f"Loaded {len(dashboards)} dashboards for user {user_id}, columns: {list(dashboards.columns)}")
        if dashboards.empty:
            logger.warning("No dashboards found for user")
        elif "id" not in dashboards.columns or "charts" not in dashboards.columns:
            logger.error(f"Missing required columns in dashboards DataFrame. Available columns: {list(dashboards.columns)}")
            st.error("Dashboard data is invalid. Please contact support.")
        return dashboards
    except Exception as e:
        logger.error(f"Error loading dashboards: {e}", exc_info=True)
        st.error(f"Failed to load dashboards: {str(e)}")
        return pd.DataFrame()

def fetch_dashboard_charts(supabase, dashboard_ids):
    logger.info(f"Fetching charts for dashboard IDs: {dashboard_ids}")
    try:
        response = supabase.table("dashboards").select("id, charts").in_("id", dashboard_ids).execute()
        charts_data = {row["id"]: row["charts"] for row in response.data}
        logger.info(f"Fetched charts for {len(charts_data)} dashboards")
        return charts_data
    except Exception as e:
        logger.error(f"Error fetching charts: {e}")
        return {}

def load_dashboard_dataset(supabase, dataset_path):
    """Load dataset from Supabase Storage given a dataset path."""
    if not dataset_path:
        logger.info("No dataset path provided for dashboard")
        return None
    try:
        bucket_name = "datasets"
        response = supabase.storage.from_(bucket_name).download(dataset_path)
        if response:
            # Read CSV bytes into DataFrame
            csv_content = response.decode("utf-8")
            df = pd.read_csv(StringIO(csv_content))
            logger.info(f"Loaded dataset from Supabase Storage: {dataset_path}, rows={len(df)}")
            return df
        else:
            logger.error(f"Failed to download dataset: {dataset_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset from Supabase Storage: {str(e)}")
        return None

def save_annotation(project_name, dashboard_id, chart_prompt, annotation):
    try:
        os.makedirs(f"projects/{project_name}", exist_ok=True)
        annotation_file = f"projects/{project_name}/annotations.json"
        annotation_data = {
            "dashboard_id": str(dashboard_id),
            "chart_prompt": chart_prompt if chart_prompt else "",
            "annotation": annotation,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        existing_data = []
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        existing_data.append(annotation_data)
        with open(annotation_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        logger.info("Saved annotation for dashboard %s, chart '%s' in project: %s", dashboard_id, chart_prompt, project_name)
    except Exception as e:
        logger.error("Failed to save annotation for project %s: %s", project_name, str(e))
        raise

def load_annotations(project_name):
    try:
        annotation_file = f"projects/{project_name}/annotations.json"
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            df = pd.DataFrame(data)
            logger.info("Loaded annotations for project: %s, rows=%d", project_name, len(df))
            return df
        return pd.DataFrame(columns=["dashboard_id", "chart_prompt", "annotation", "created_at"])
    except Exception as e:
        logger.error("Failed to load annotations for project %s: %s", project_name, str(e))
        return pd.DataFrame(columns=["dashboard_id", "chart_prompt", "annotation", "created_at"])

def delete_dashboard(project_name, dashboard_id):
    try:
        dashboard_file = f"projects/{project_name}/dashboard.json"
        if os.path.exists(dashboard_file):
            with open(dashboard_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            data = [d for d in data if isinstance(d, dict) and str(d["dashboard_id"]) != str(dashboard_id)]
            with open(dashboard_file, "w") as f:
                json.dump(data, f, indent=4)
            logger.info("Deleted dashboard %s from project: %s", dashboard_id, project_name)
        annotation_file = f"projects/{project_name}/annotations.json"
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            if not isinstance(annotations, list):
                annotations = [annotations]
            annotations = [a for a in annotations if str(a["dashboard_id"]) != str(dashboard_id)]
            with open(annotation_file, "w") as f:
                json.dump(annotations, f, indent=4)
            logger.info("Deleted annotations for dashboard %s in project: %s", dashboard_id, project_name)
    except Exception as e:
        logger.error("Failed to delete dashboard %s for project %s: %s", dashboard_id, project_name, str(e))
        raise

def parse_prompt(prompt, dimensions, measures, dates):
    logger.warning("Using deprecated parse_prompt function. Please use rule_based_parse from chart_utils.py instead.")
    from chart_utils import rule_based_parse
    return rule_based_parse(prompt, None, dimensions, measures, dates)

def update_dashboard(project_name, dashboard_id, new_name=None, new_prompts=None):
    try:
        dashboard_file = f"projects/{project_name}/dashboard.json"
        if not os.path.exists(dashboard_file):
            logger.warning("No dashboard file found for project: %s", project_name)
            return
        with open(dashboard_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for d in data:
            if isinstance(d, dict) and str(d["dashboard_id"]) == str(dashboard_id):
                if new_name:
                    d["dashboard_name"] = new_name
                if new_prompts:
                    d["charts"] = [{"prompt": p} for p in new_prompts]
                d["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        with open(dashboard_file, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Updated dashboard %s in project: %s", dashboard_id, project_name)
    except Exception as e:
        logger.error("Failed to update dashboard %s for project %s: %s", dashboard_id, project_name, str(e))
        raise

def load_openai_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in environment.")
        return None
    logger.info("OpenAI API key loaded successfully.")
    return api_key

def generate_gpt_insight_with_fallback(chart_data, dimension, metric):
    return []

def save_field_types(project_name, field_types):
    """Save field types to a JSON file and Supabase."""
    try:
        field_types_file = f"projects/{project_name}/field_types.json"
        os.makedirs(f"projects/{project_name}", exist_ok=True)
        with open(field_types_file, 'w') as f:
            json.dump(field_types, f)
        if 'user_id' in st.session_state and 'current_project' in st.session_state:
            try:
                supabase.table("field_types").insert({
                    "project_id": st.session_state.current_project,
                    "user_id": st.session_state.user_id,
                    "types": field_types
                }).execute()
                logger.info(f"Saved field types to Supabase for project: {project_name}")
            except Exception as e:
                logger.error(f"Failed to save field types to Supabase: {str(e)}")
        logger.info(f"Saved field types for project {project_name}: {field_types}")
    except Exception as e:
        logger.error(f"Failed to save field types for project {project_name}: {str(e)}")
        raise

def get_field_type(col, field_types):
    """Retrieve the field type for a column."""
    for t in ["dimension", "measure", "date", "id"]:
        if col in field_types.get(t, []):
            return t.capitalize()
    return "Other"

def compute_dataset_hash(df):
    """Compute a hash of the DataFrame to detect changes."""
    if df is None:
        return None
    try:
        columns_str = ''.join(sorted(df.columns))
        sample_data = df.head(100).to_csv(index=False)
        hash_input = columns_str + sample_data
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute dataset hash: {str(e)}")
        return None