from __future__ import annotations
import pandas as pd
import json
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

ENGAGEMENTS_CSV_PATH = Path("engagements.csv")
CONFIG_JSON_PATH = Path("configchoice.json")

class DataValidator:
    def __init__(self, choices: Dict):
        """Initializes the validator with pre-loaded choices."""
        self.choices = choices
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Sets up validation rules based on the provided choices."""
        rules = {}
        required_fields = [
            'gics_sector', 'region', 'program', 'country', 
            'milestone_status', 'outcome_status', 'interaction_type'
        ]
        for field, choice_list in self.choices.items():
            rules[field] = {
                'required': field in required_fields,
                'type': 'choice',
                'choices': choice_list
            }
        return rules

    def validate_field(self, field_name: str, value: any) -> Tuple[bool, Optional[str]]:
        """Validates a single field against its rule."""
        if field_name not in self.validation_rules:
            return True, None

        rule = self.validation_rules[field_name]
        if rule['required'] and (value is None or str(value).strip() == ''):
            return False, f"{field_name.replace('_', ' ').title()} is required."
        
        if not rule['required'] and (value is None or str(value).strip() == ''):
            return True, None

        if rule['type'] == 'choice' and value not in rule['choices']:
            return False, f"Invalid value '{value}' for {field_name}."
        
        return True, None
    
    def validate_record(self, record: Dict) -> Dict[str, List[str]]:
        """Validates a full data record."""
        errors = {}
        for field_name, value in record.items():
            # Only validate fields that have rules
            if field_name in self.validation_rules:
                is_valid, error_msg = self.validate_field(field_name, value)
                if not is_valid:
                    errors.setdefault(field_name, []).append(error_msg)
        
        if not record.get('company_name', '').strip():
            errors.setdefault('company_name', []).append("Company name is required.")

        ## Restore: ESG flag validation
        if not any(record.get(flag) for flag in ['e', 's', 'g']):
             errors.setdefault('esg_flags', []).append("At least one ESG flag (E, S, or G) must be selected.")
        
        return errors

# --- DATA LOADING AND SAVING ---

@st.cache_data(ttl=600)
def load_db() -> tuple[pd.DataFrame, dict]:
    """Loads the engagements CSV and config JSON into memory."""
    df = pd.DataFrame()
    config = {}
    try:
        df = pd.read_csv(ENGAGEMENTS_CSV_PATH, encoding='latin1')
        date_cols = ["start_date", "target_date", "last_interaction_date", "next_action_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    except FileNotFoundError:
        st.error(f"Error: Engagements file '{ENGAGEMENTS_CSV_PATH}' not found.")
    except Exception as e:
        st.exception(f"Error loading engagements CSV: {e}")

    try:
        with open(CONFIG_JSON_PATH, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Config file '{CONFIG_JSON_PATH}' not found.")
    except Exception as e:
        st.exception(f"Error loading config JSON: {e}")

    return df, config

def save_engagements_df(df: pd.DataFrame):
    """Saves the engagements DataFrame back to CSV and clears relevant caches."""
    try:
        df.to_csv(ENGAGEMENTS_CSV_PATH, index=False)
        load_db.clear()
    except Exception as e:
        st.error(f"Failed to save engagements data: {e}")

# --- DATA RETRIEVAL AND COMPUTATION ---

def get_latest_view(df: pd.DataFrame) -> pd.DataFrame:
    """Computes dynamic fields on a given DataFrame. Not cached."""
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    now = pd.to_datetime(datetime.now())

    # Ensure date columns are datetime objects for calculations
    date_cols = ['target_date', 'next_action_date']
    for col in date_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

    df_copy["days_to_next_action"] = (df_copy["next_action_date"] - now).dt.days
    df_copy["is_complete"] = df_copy["milestone_status"].str.lower() == "complete"
    df_copy["on_time"] = df_copy["is_complete"] & (df_copy["target_date"] >= now)
    df_copy["late"] = df_copy["is_complete"] & (df_copy["target_date"] < now)
    df_copy["overdue"] = (df_copy["next_action_date"] < now) & (~df_copy["is_complete"])
    df_copy["urgent"] = df_copy["days_to_next_action"] <= 3

    return df_copy

def get_upcoming_tasks(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """Get tasks from a given dataframe due within the specified number of days."""
    if df.empty or 'next_action_date' not in df.columns:
        return pd.DataFrame()

    today = pd.to_datetime(datetime.now().date())
    future_date = today + timedelta(days=days)

    upcoming_mask = (
        (df['next_action_date'] >= today) &
        (df['next_action_date'] <= future_date) &
        (df['milestone_status'].str.lower() != 'complete')
    )
    return df[upcoming_mask].sort_values(by="next_action_date")

def get_interactions_for_company(engagement_id: int) -> List[Dict]:
    """Retrieves the interaction history for a specific engagement."""
    df, _ = load_db()
    if df.empty or 'interactions' not in df.columns:
        return []
    
    record = df[pd.to_numeric(df['engagement_id']) == engagement_id]
    if record.empty:
        return []

    interactions_json = record.iloc[0].get('interactions', '[]')
    try:
        # Handle cases where the JSON might be NaN or other non-string types
        if pd.isna(interactions_json):
            return []
        return json.loads(interactions_json)
    except (json.JSONDecodeError, TypeError):
        return []

# --- DATA MODIFICATION FUNCTIONS ---

def create_engagement(data: Dict) -> Tuple[bool, str]:
    """Creates a new engagement with all fields."""
    validator = st.session_state.validator
    errors = validator.validate_record(data)
    if errors:
        error_messages = [msg for sublist in errors.values() for msg in sublist]
        return False, "Validation failed: " + ", ".join(error_messages)

    df, _ = load_db()
    if not df[df['company_name'].str.lower() == data['company_name'].lower()].empty:
        return False, f"Engagement with '{data['company_name']}' already exists."

    next_id = (df['engagement_id'].max() + 1) if not df.empty and 'engagement_id' in df.columns else 1
    
    ## Restore: Include all fields in the new engagement record
    new_engagement = {
        "engagement_id": next_id, "company_name": data.get("company_name"),
        "isin": data.get("isin"), "aqr_id": data.get("aqr_id"),
        "gics_sector": data.get("gics_sector"), "country": data.get("country"),
        "region": data.get("region"), "program": data.get("program"),
        "theme": data.get("theme"), "objective": data.get("objective"),
        "start_date": data.get("start_date"), "target_date": data.get("target_date"),
        "e": data.get("e", False), "s": data.get("s", False), "g": data.get("g", False),
        "created_date": datetime.now(), "created_by": data.get("created_by", "System"),
        "last_interaction_date": None, "next_action_date": data.get("start_date"), # First action is on start date
        "milestone": "Initiated", "milestone_status": "Amber",
        "escalation_level": "None Required", "outcome_status": "N/A",
        "interactions": "[]",
    }
    
    new_df = pd.concat([df, pd.DataFrame([new_engagement])], ignore_index=True)
    save_engagements_df(new_df)
    return True, f"Engagement for {data['company_name']} created successfully (ID: {next_id})."

def log_interaction(data: Dict) -> Tuple[bool, str]:
    """Logs a detailed interaction for an engagement."""
    df, _ = load_db()
    engagement_id = data.get("engagement_id")
    idx = df[df['engagement_id'] == engagement_id].index
    if idx.empty:
        return False, "Engagement ID not found."
    idx = idx[0]

    # Update the main record with latest interaction info
    update_fields = [
        "last_interaction_date", "next_action_date", "milestone", 
        "milestone_status", "escalation_level", "outcome_status"
    ]
    for key in update_fields:
        if key in data and data[key] is not None:
            df.loc[idx, key] = data[key]
    
    # Append to interaction history
    try:
        interactions_list = json.loads(df.loc[idx, "interactions"]) if pd.notna(df.loc[idx, "interactions"]) else []
    except (json.JSONDecodeError, TypeError):
        interactions_list = []

    ## Restore: Full interaction record
    new_interaction = {
        "interaction_id": str(uuid.uuid4()),
        "interaction_type": data.get("interaction_type"),
        "interaction_summary": data.get("interaction_summary"),
        "interaction_date": pd.to_datetime(data.get("last_interaction_date")).strftime('%Y-%m-%d'),
        "outcome_status": data.get("outcome_status"),
        "milestone": data.get("milestone"),
        "escalation_level": data.get("escalation_level"),
        "logged_by": "System", "logged_date": datetime.now().strftime('%Y-%m-%d')
    }
    interactions_list.append(new_interaction)
    df.loc[idx, "interactions"] = json.dumps(interactions_list, indent=2)

    save_engagements_df(df)
    return True, "Interaction logged successfully."

def update_milestone_status(engagement_id: int, status: str, user: str = "System") -> Tuple[bool, str]:
    """Updates the milestone status, logging it as a formal interaction."""
    return log_interaction({
        "engagement_id": engagement_id,
        "last_interaction_date": datetime.now().date(),
        "interaction_type": "Status Change",
        "interaction_summary": f"Status changed to '{status}' by {user}.",
        "milestone_status": status,
    })


# --- LOOKUP AND ANALYTICS FUNCTIONS ---

@st.cache_data(ttl=600)
def get_lookup_values(field: str) -> List[str]:
    """Gets lookup values for a specific field from the config JSON."""
    _, config_data = load_db()
    return config_data.get(field, [])

@st.cache_data
def get_local_world_topojson():
    """
    Downloads and caches the world topojson file locally to avoid CSP issues.
    Returns the local file path and country mapping.
    """
    import requests
    import json
    from pathlib import Path
    
    # Local file path
    local_topo_file = Path("world_110m.json")
    
    # Download if not exists
    if not local_topo_file.exists():
        try:
            response = requests.get("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json")
            response.raise_for_status()
            
            with open(local_topo_file, 'w') as f:
                json.dump(response.json(), f)
        except Exception as e:
            st.error(f"Failed to download world map data: {e}")
            return None, {}
    
    # Create country name to ID mapping
    try:
        with open(local_topo_file, 'r') as f:
            topo_data = json.load(f)
        
        # Extract country properties for mapping
        countries = topo_data['objects']['countries']['geometries']
        country_mapping = {}
        
        for country in countries:
            props = country.get('properties', {})
            name = props.get('NAME', '')
            country_id = props.get('ADM0_A3', '')  # ISO 3-letter code
            if name and country_id:
                country_mapping[name] = country_id
        
        return str(local_topo_file), country_mapping
        
    except Exception as e:
        st.error(f"Error processing topojson file: {e}")
        return None, {}

@st.cache_data  
def get_country_name_to_id_mapping_local():
    """
    Alternative approach using a simple manual mapping for common countries.
    This avoids external dependencies entirely.
    """
    df, _ = load_db()
    if df.empty:
        return {}
    
    # Manual mapping of common country names to ISO codes
    country_iso_mapping = {
        'United States': 'USA',
        'USA': 'USA',  # Handle both variations
        'US': 'USA',
        'United States of America': 'USA',
        'United Kingdom': 'GBR', 
        'UK': 'GBR',
        'Britain': 'GBR',
        'Great Britain': 'GBR',
        'Germany': 'DEU',
        'France': 'FRA',
        'Japan': 'JPN',
        'Canada': 'CAN',
        'Australia': 'AUS',
        'China': 'CHN',
        'People\'s Republic of China': 'CHN',
        'India': 'IND',
        'Brazil': 'BRA',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Netherlands': 'NLD',
        'Holland': 'NLD',
        'Switzerland': 'CHE',
        'Sweden': 'SWE',
        'Norway': 'NOR',
        'Denmark': 'DNK',
        'Finland': 'FIN',
        'Belgium': 'BEL',
        'Austria': 'AUT',
        'South Korea': 'KOR',
        'Korea': 'KOR',
        'Republic of Korea': 'KOR',
        'Mexico': 'MEX',
        'Russia': 'RUS',
        'Russian Federation': 'RUS',
        'South Africa': 'ZAF',
        'Singapore': 'SGP',
        'Hong Kong': 'HKG',
        'New Zealand': 'NZL',
        'Ireland': 'IRL',
        'Portugal': 'PRT',
        'Poland': 'POL',
        'Czech Republic': 'CZE',
        'Czechia': 'CZE',
        'Hungary': 'HUN',
        'Greece': 'GRC',
        'Turkey': 'TUR',
        'Israel': 'ISR',
        'Thailand': 'THA',
        'Malaysia': 'MYS',
        'Indonesia': 'IDN',
        'Philippines': 'PHL',
        'Taiwan': 'TWN',
        'Republic of China': 'TWN',
        'Chile': 'CHL',
        'Argentina': 'ARG',
        'Colombia': 'COL',
        'Peru': 'PER',
        'Ecuador': 'ECU',
        'Venezuela': 'VEN',
        'Uruguay': 'URY',
        'Paraguay': 'PRY',
        'Bolivia': 'BOL',
    }
    
    # Only return mappings for countries that exist in our data
    countries_in_data = df['country'].unique()
    return {country: iso for country, iso in country_iso_mapping.items() 
            if country in countries_in_data}

@st.cache_data(ttl=600)
def get_engagement_analytics(df: pd.DataFrame) -> Dict:
    """Generates a full suite of analytics from a given DataFrame."""
    if df.empty:
        return {
            "success_rates": pd.DataFrame(), 
            "monthly_trends": pd.DataFrame()
        }

    # Success rates by sector
    success_rates = df.groupby('gics_sector').agg(
        total=('engagement_id', 'count'),
        completed=('is_complete', 'sum')
    ).reset_index()
    success_rates['success_rate'] = (success_rates['completed'] / success_rates['total'] * 100).round(1)

    # Monthly trends
    trends_df = df.copy()
    trends_df['month'] = trends_df['start_date'].dt.to_period('M').dt.to_timestamp()
    monthly_trends = trends_df.groupby('month').agg(
        new_engagements=('engagement_id', 'count')
    ).reset_index()

    return {
        "success_rates": success_rates, 
        "monthly_trends": monthly_trends
    }

def get_lookup_fields() -> List[str]:
    _, config_data = load_db()
    return sorted(list(config_data.keys()))

def get_database_info() -> Dict:
    df, config = load_db()
    return {"engagements": len(df), "config_fields": len(config)}