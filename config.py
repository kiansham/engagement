from pathlib import Path

class Config:
    # Application settings
    APP_TITLE = "ESG Engagement Platform"
    APP_ICON = "📊"
    
    # Color palette
    COLORS = {
        "primary": "#3498db", "success": "#2ecc71",
        "warning": "#f39c12", "danger": "#e74c3c",
    }
    CB_SAFE_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

## Refactor: Centralized page and navigation configuration.
# The keys are the display titles. 'icon' is for the menu, and 'function'
# is the name of the corresponding function in app.py.
PAGES_CONFIG = {
    "Dashboard": {"icon": "speedometer2", "function": "dashboard"},
    "Task Management": {"icon": "list-check", "function": "task_management"},
    "Company Profiles": {"icon": "building", "function": "company_deep_dive"},
    "Engagement Operations": {"icon": "folder-plus", "function": "engagement_management"},
    "Analytics": {"icon": "graph-up-arrow", "function": "analytics"},
}

## Refactor: Centralized CSS styles.
CSS_STYLES = """
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .alert-urgent { 
        background-color: #ffe6e6; border-left: 4px solid #e74c3c; 
        padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; 
    }
    .alert-warning { 
        background-color: #fff3cd; border-left: 4px solid #f39c12; 
        padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; 
    }
    .validation-error { 
        background-color: #fee; border-left: 4px solid #c00; 
        padding: 0.5rem; border-radius: 0.25rem; margin: 0.5rem 0; 
    }
</style>
"""

## Refactor: Centralized navigation menu styles.
NAV_STYLES = {
    "container": {"padding": "0!important", "background-color": "#f0f2f6"},
    "icon": {"font-size": "1.1rem"},
    "nav-link": {
        "font-size": "1rem", "text-align": "left", "margin":"5px",
        "--hover-color": "#e8f4fd"
    },
    "nav-link-selected": {"background-color": Config.COLORS["primary"]},
}

## Refactor: Centralized chart layout configurations.
CHART_CONFIGS = {
    "bar": {
        "height": 400,
        "yaxis": {"tickformat": "d"},
        "margin": {"l": 50, "r": 20, "t": 60, "b": 50},
        "showlegend": False
    },
    "status": {
        "height": 140,
        "barmode": "stack",
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "showlegend": True,
        "xaxis": {"tickformat": "d"}
    },
    "geographic": {
        "height": 700,
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
    }
}
