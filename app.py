from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from typing import Dict, List, Optional, Tuple

## Refactor: Import only the necessary functions from utils
from config import Config, CHART_CONFIGS, CSS_STYLES, NAV_STYLES, PAGES_CONFIG
from utils import (
    get_latest_view, get_upcoming_tasks,
    create_engagement, log_interaction, update_milestone_status,
    get_lookup_values,
    get_engagement_analytics, get_interactions_for_company,
    load_db,
    DataValidator
)

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

# --- REUSABLE UI HELPER FUNCTIONS ---

def create_metric_row(metrics: List[tuple]) -> None:
    """Creates a row of metrics."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

def create_chart(data: pd.Series, title: str, xlab: str, chart_type: str = "bar") -> go.Figure:
    """Creates a generic bar or pie chart."""
    if chart_type == "bar":
        fig = px.bar(x=data.index, y=data.values, title="", labels={"x": "", "y": ""},
                    color=data.index, color_discrete_sequence=Config.CB_SAFE_PALETTE)
        fig.update_layout(**CHART_CONFIGS['bar'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    elif chart_type == "pie":
        fig = px.pie(values=data.values, names=data.index, title="", color_discrete_sequence=Config.CB_SAFE_PALETTE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_status_chart(data: pd.DataFrame) -> go.Figure:
    """Creates the stacked horizontal bar chart for engagement status."""
    if data.empty:
        return go.Figure()

    on_time = data.get("on_time", pd.Series(dtype=bool)).sum()
    late = data.get("late", pd.Series(dtype=bool)).sum()
    active = len(data[~data.get("is_complete", pd.Series(dtype=bool))])

    fig = go.Figure()
    colors = [Config.COLORS["success"], Config.COLORS["danger"], Config.COLORS["warning"]]

    for i, (label, value) in enumerate([("On‑time", on_time), ("Late", late), ("Open", active)]):
        fig.add_bar(y=["Engagements"], x=[value], name=label, orientation="h", marker_color=colors[i])

    fig.update_layout(title="", **CHART_CONFIGS['status'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_outcome_status_chart(data: pd.DataFrame) -> go.Figure:
    """Creates the stacked horizontal bar chart for outcome status."""
    if data.empty or "outcome_status" not in data.columns:
        return go.Figure()

    outcome_counts = data["outcome_status"].value_counts()
    if outcome_counts.empty:
        return go.Figure()

    fig = go.Figure()
    
    for i, (label, value) in enumerate(outcome_counts.items()):
        color = Config.CB_SAFE_PALETTE[i % len(Config.CB_SAFE_PALETTE)]
        fig.add_bar(y=["Outcomes"], x=[value], name=label, orientation="h", marker_color=color)

    fig.update_layout(title="", **CHART_CONFIGS['status'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_info_display(items: List[tuple], use_html: bool = False) -> None:
    """Creates a row of informational text displays."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        if use_html:
            col.markdown(f"**{label}**<br><span style='font-size: 1.1em;'>{value}</span>", unsafe_allow_html=True)
        else:
            col.metric(label, value)

def handle_task_date_display(task_date, today) -> None:
    """Displays overdue/upcoming task dates with styled messages."""
    try:
        if pd.notna(task_date):
            if hasattr(task_date, 'date'):
                task_date = task_date.date()
            else:
                task_date = pd.to_datetime(task_date).date()

            days_left = (task_date - today).days
            if days_left < 0:
                st.error(f"Overdue by {abs(days_left)} days")
            else:
                st.warning(f"{days_left} days left")
        else:
            st.info("No due date set")
    except:
        st.caption("Date error")

def create_alert_section(urgent_tasks: pd.DataFrame, overdue_tasks: pd.DataFrame) -> None:
    """Creates the main alert section for the dashboard."""
    if not urgent_tasks.empty or not overdue_tasks.empty:
        col1, col2 = st.columns(2)

        if not urgent_tasks.empty:
            col1.markdown(f"""<div class="alert-urgent"><strong>⚠️ {len(urgent_tasks)} Urgent Tasks</strong><br>
                         Due within 3 days</div>""", unsafe_allow_html=True)

        if not overdue_tasks.empty:
            col2.markdown(f"""<div class="alert-warning"><strong>📅 {len(overdue_tasks)} Overdue Tasks</strong><br>
                         Past due date</div>""", unsafe_allow_html=True)

def company_selector_widget(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Optional[str]:
    """
    Renders a company selection dropdown.
    """
    if full_df.empty or "company_name" not in full_df.columns:
        st.warning("No company data available.")
        return None

    if not filtered_df.empty and len(filtered_df) < len(full_df):
        available_companies = sorted(filtered_df["company_name"].unique())
        st.info(f"Showing {len(available_companies)} companies based on current filters. Clear filters to see all companies.")
    else:
        available_companies = sorted(full_df["company_name"].unique())

    return st.selectbox("Select Company", [""] + available_companies)

def display_interaction_history(engagement_id: int) -> None:
    """Fetches and displays the interaction history for an engagement."""
    try:
        interactions = get_interactions_for_company(engagement_id)

        if not interactions:
            st.info("No interactions recorded for this company.")
            return

        interactions_df = pd.DataFrame(interactions)
        interactions_df['interaction_date'] = pd.to_datetime(interactions_df['interaction_date'])
        interactions_df = interactions_df.sort_values(by='interaction_date', ascending=False)

        if len(interactions_df) > 1:
            st.markdown("### Interaction Timeline")
            fig = go.Figure()
            unique_types = interactions_df['interaction_type'].unique()
            color_map = {itype: color for itype, color in zip(unique_types, Config.CB_SAFE_PALETTE)}
            
            for _, row in interactions_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["interaction_date"]], y=[1], mode='markers+text',
                    marker=dict(size=15, color=color_map.get(row["interaction_type"], "#333")),
                    text=row.get("interaction_type", "N/A"), textposition="top center",
                    name=row.get("interaction_type", "N/A"),
                    hoverinfo='text',
                    hovertext=f"<b>{row.get('interaction_type')}</b><br>{row.get('interaction_summary')}"
                ))
            
            fig.update_layout(title="", yaxis_visible=False, height=200, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


        st.markdown("### Recent Interactions")
        for _, interaction in interactions_df.iterrows():
            with st.expander(f"{interaction.get('interaction_type', 'N/A')} - {interaction['interaction_date'].strftime('%Y-%m-%d')}"):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Milestone:** {interaction.get('milestone', 'N/A')}")
                col1.markdown(f"**Outcome:** {interaction.get('outcome_status', 'N/A')}")
                col2.markdown(f"**Logged By:** {interaction.get('logged_by', 'N/A')}")
                col2.markdown(f"**Logged Date:** {interaction.get('logged_date', 'N/A')}")
                st.markdown("**Summary:**")
                st.write(interaction.get("interaction_summary", "No summary available."))

    except Exception as e:
        st.error(f"Error loading interaction data: {e}")

# --- FILTER FUNCTIONS ---

def sidebar_filters(df: pd.DataFrame) -> tuple:
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        with st.expander("Alerts", expanded=False):
            show_urgent = st.checkbox("Show urgent only")
            show_overdue = st.checkbox("Show overdue only")
        with st.expander("Company Selection", expanded=False):
            companies = st.multiselect("Company Name", sorted(df["company_name"].unique()) if "company_name" in df.columns else [])
        with st.expander("Engagement Type", expanded=False):
            progs = st.multiselect("Engagement Program", get_lookup_values("program"))
        with st.expander("Geo & Sector", expanded=False):
            region = st.multiselect("Region", get_lookup_values("region"))
            country = st.multiselect("Country", get_lookup_values("country"))
            sector = st.multiselect("GICS Sector", get_lookup_values("gics_sector"))
        with st.expander("Engagement Status", expanded=False):
            mile = st.multiselect("Milestone", get_lookup_values("milestone"))
            status = st.multiselect("Status", get_lookup_values("milestone_status"))
        
        with st.expander("Themes & Content", expanded=False):
            themes = st.multiselect("Theme", get_lookup_values("theme"))
            interaction_search = st.text_input("Search Interaction Summary", help="Search within interaction summaries")

        with st.expander("ESG Focus", expanded=True):
            col_e, col_s, col_g = st.columns(3)
            env = col_e.checkbox("E", value=True)
            soc = col_s.checkbox("S", value=True)
            gov = col_g.checkbox("G", value=True)

    esg = [c for c, b in zip(["e", "s", "g"], [env, soc, gov]) if b] or ["e", "s", "g"]
    return progs, sector, region, country, mile, status, esg, show_urgent, show_overdue, companies, themes, interaction_search

def apply_filters(df: pd.DataFrame, filters: tuple) -> pd.DataFrame:
    if df.empty:
        return df

    progs, sector, region, country, mile, status, esg, show_urgent, show_overdue, companies, themes, interaction_search = filters
    filtered_df = df.copy()

    filter_map = {"program": progs, "gics_sector": sector, "region": region,
                 "country": country, "milestone": mile, "milestone_status": status, "company_name": companies}

    for col, values in filter_map.items():
        if values and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    if themes and "theme" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["theme"].isin(themes)]

    if interaction_search and "interactions" in filtered_df.columns:
        def search_interactions(interactions_json):
            try:
                if pd.isna(interactions_json): return False
                interactions_list = json.loads(interactions_json)
                return any(interaction_search.lower() in interaction.get("interaction_summary", "").lower() for interaction in interactions_list)
            except (json.JSONDecodeError, TypeError):
                return False
        mask = filtered_df["interactions"].apply(search_interactions)
        filtered_df = filtered_df[mask]

    if esg and all(col in filtered_df.columns for col in esg):
        filtered_df = filtered_df[filtered_df[esg].any(axis=1)]

    if show_urgent and "urgent" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["urgent"]]
    if show_overdue and "overdue" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["overdue"]]

    return filtered_df

# --- PAGE-RENDERING FUNCTIONS ---

def dashboard():
    data = st.session_state['DATA']
    
    if data.empty:
        st.warning("No engagement data available or no data matches the current filters.")
        return

    urgent_tasks = data[data.get("urgent", False)]
    overdue_tasks = data[data.get("overdue", False)]
    create_alert_section(urgent_tasks, overdue_tasks)

    st.markdown("### 📈 Key Metrics")
    total = len(data)
    completed = data.get("is_complete", pd.Series(dtype=bool)).sum()
    on_time = data.get("on_time", pd.Series(dtype=bool)).sum()
    late = data.get("late", pd.Series(dtype=bool)).sum()
    active = total - completed
    completion_rate = (completed / total * 100) if total > 0 else 0
    effectiveness = (on_time / (on_time + late) * 100) if (on_time + late) > 0 else 0
    
    create_metric_row([
        ("Total Engagements", total), ("Completion Rate", f"{completion_rate:.1f}%"),
        ("On‑time Effectiveness", f"{effectiveness:.1f}%"), ("Active Engagements", active), ("Overdue", late)
    ])

    # Status breakdown chart
    context_col, chart_col = st.columns([1, 3])
    with context_col:
        st.markdown("#### Status Overview")
        st.markdown("""Shows the breakdown of engagement statuses, displaying those that are:
        \\
        • Currently active \\
        • Complete \\
        • Not yet started""")
    with chart_col:
        st.plotly_chart(create_status_chart(data), use_container_width=True)

    # Outcome status chart
    if "outcome_status" in data.columns and not data["outcome_status"].dropna().empty:
        context_col, chart_col = st.columns([1, 3])
        with context_col:
            st.markdown("#### Outcome Status")
            st.markdown("""Shows the distribution of interaction outcomes, including:
            \\
            • Positive responses \\
            • Neutral engagements \\
            • Negative feedback \\
            • Pending outcomes""")
        with chart_col:
            st.plotly_chart(create_outcome_status_chart(data), use_container_width=True)

    # Sector distribution chart
    if "gics_sector" in data.columns and not data["gics_sector"].dropna().empty:
        context_col, chart_col = st.columns([1, 2])
        with context_col:
            st.markdown("#### 🏢 Sector Distribution")
            st.write("Distribution of engagements across different GICS sectors, helping identify which industries are most actively engaged.")
        with chart_col:
            st.plotly_chart(create_chart(data["gics_sector"].value_counts(), "Engagements by Sector", "Sector"), use_container_width=True)

    # Regional distribution chart
    if "region" in data.columns and not data["region"].dropna().empty:
        context_col, chart_col = st.columns([1, 2])
        with context_col:
            st.markdown("#### 🌍 Regional Distribution")
            st.write("Geographic spread of engagements by region, providing insights into global engagement coverage and regional focus areas.")
        with chart_col:
            st.plotly_chart(create_chart(data["region"].value_counts(), "Engagements by Region", "Region"), use_container_width=True)

    # Milestone progress chart
    if "milestone" in data.columns and not data["milestone"].dropna().empty:
        context_col, chart_col = st.columns([1, 2])
        with context_col:
            st.markdown("#### 🎯 Milestone Progress")
            st.write("Current milestone stages across all engagements, showing progress through the engagement lifecycle from initiation to completion.")
        with chart_col:
            st.plotly_chart(create_chart(data["milestone"].value_counts(), "Engagements by Milestone Stage", "Milestone"), use_container_width=True)

def engagement_operations():
    tab1, tab2 = st.tabs(["➕ Create Engagement", "📝 Log Interaction"])
    
    with tab1:
        with st.form("new_engagement", clear_on_submit=True):
            st.markdown('### Log New Engagement Target')
            col1, col2 = st.columns(2)
            company_name = col1.text_input("Company Name *")
            gics_sector = col2.selectbox("GICS Sector *", [""] + get_lookup_values("gics_sector"))

            col1, col2, col3 = st.columns(3)
            isin = col1.text_input("ISIN", help="International Securities Identification Number")
            aqr_id = col2.text_input("AQR ID", help="Internal AQR identifier")
            program = col3.selectbox("Program *", [""] + get_lookup_values("program"))
            
            col1, col2, col3 = st.columns(3)
            country = col1.selectbox("Country *", [""] + get_lookup_values("country"))
            region = col2.selectbox("Region *", [""] + get_lookup_values("region"))
            theme = col3.selectbox("Theme", [""] + get_lookup_values("theme"))
            
            objective = st.selectbox("Objective", [""] + get_lookup_values("objective"))

            st.markdown("### ESG Focus Areas *")
            col_e, col_s, col_g = st.columns(3)
            esg_flags = {
                "e": col_e.checkbox("Environmental"),
                "s": col_s.checkbox("Social"),
                "g": col_g.checkbox("Governance"),
            }

            st.markdown("### Timeline")
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start Date *", value=datetime.now().date())
            target_date = col2.date_input("Target Date", value=datetime.now().date() + timedelta(days=90))

            if st.form_submit_button("Create Engagement", type="primary"):
                engagement_data = {
                    "company_name": company_name, "gics_sector": gics_sector, "region": region,
                    "isin": isin, "aqr_id": aqr_id, "program": program, "country": country,
                    "theme": theme, "objective": objective,
                    "start_date": start_date, "target_date": target_date, "created_by": "System",
                    **esg_flags
                }
                success, message = create_engagement(engagement_data)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with tab2:
        st.markdown("### Log Interaction")
        full_df = st.session_state['FULL_DATA']
        filtered_df = st.session_state['DATA']
        
        selected_company = company_selector_widget(full_df, filtered_df)
        
        if not selected_company:
            st.info("Please select a company to log an interaction.")
            return

        engagement_data = full_df[full_df["company_name"] == selected_company].iloc[0]

        with st.expander("Current Engagement Status", expanded=True):
            create_info_display([
                ("Current Milestone", engagement_data.get("milestone", "N/A")),
                ("Status", engagement_data.get("milestone_status", "N/A")),
                ("Escalation", engagement_data.get("escalation_level", "N/A"))
            ], use_html=True)

        with st.form("log_interaction"):
            st.markdown("### Interaction Details")
            col1, col2 = st.columns(2)
            interaction_type = col1.selectbox("Interaction Type *", [""] + get_lookup_values("interaction_type"))
            interaction_date = col2.date_input("Interaction Date *", value=datetime.now().date())
            
            col1, col2 = st.columns(2)
            outcome_status = col1.selectbox("Outcome Status *", [""] + get_lookup_values("outcome_status"))
            escalation_level = col2.selectbox("New Escalation Level", [""] + get_lookup_values("escalation_level"))

            interaction_summary = st.text_area("Interaction Summary *", height=150)
            
            st.markdown("### Milestone Update")
            col1, col2 = st.columns(2)
            milestone = col1.selectbox("New Milestone", [""] + get_lookup_values("milestone"))
            milestone_status = col2.selectbox("New Milestone Status", [""] + get_lookup_values("milestone_status"))

            if st.form_submit_button("Log Interaction", type="primary"):
                if not interaction_summary.strip() or not interaction_type:
                    st.error("Interaction type and summary are required.")
                else:
                    interaction_log_data = {
                        "engagement_id": engagement_data["engagement_id"],
                        "last_interaction_date": interaction_date,
                        "next_action_date": datetime.now().date() + timedelta(days=14),
                        "interaction_summary": interaction_summary,
                        "interaction_type": interaction_type,
                        "outcome_status": outcome_status,
                        "escalation_level": escalation_level if escalation_level else engagement_data.get("escalation_level"),
                        "milestone": milestone if milestone else engagement_data.get("milestone"),
                        "milestone_status": milestone_status if milestone_status else engagement_data.get("milestone_status"),
                    }
                    success, message = log_interaction(interaction_log_data)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

def task_management():
    filtered_df = st.session_state['DATA']
    
    if filtered_df.empty:
        st.warning("No tasks available for the current filters.")
        return

    all_upcoming_tasks = get_upcoming_tasks(df=filtered_df, days=14)
    urgent_tasks = all_upcoming_tasks[all_upcoming_tasks['days_to_next_action'] <= 3]
    warning_tasks = all_upcoming_tasks[(all_upcoming_tasks['days_to_next_action'] > 3) & (all_upcoming_tasks['days_to_next_action'] <= 7)]

    create_metric_row([
        ("Urgent (≤3 days)", len(urgent_tasks)),
        ("Warning (≤7 days)", len(warning_tasks)),
        ("Upcoming (≤14 days)", len(all_upcoming_tasks))
    ])

    if len(filtered_df) < len(st.session_state['FULL_DATA']):
        st.info(f"Tasks filtered to show only companies matching current filter criteria.")

    tab1, tab2, tab3 = st.tabs(["🚨 Urgent", "⚠️ This Week", "📅 Upcoming"])
    today = datetime.now().date()

    task_tabs_data = [
        (tab1, urgent_tasks, "Urgent"),
        (tab2, warning_tasks, "This Week"),
        (tab3, all_upcoming_tasks, "Upcoming")
    ]

    for tab, tasks, label in task_tabs_data:
        with tab:
            if not tasks.empty:
                for _, task in tasks.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        col1.markdown(f"**{task['company_name']}**")
                        col1.caption(f"Milestone: {task.get('milestone', 'N/A')}")
                        with col2:
                            handle_task_date_display(task['next_action_date'], today)
                        if col3.button("Mark Complete", key=f"task_{task['engagement_id']}"):
                            success, msg = update_milestone_status(task['engagement_id'], "Complete")
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        st.divider()
            else:
                st.info(f"No {label.lower()} tasks! 🎉")

def enhanced_analysis():
    df = st.session_state['DATA']
    analytics_data = get_engagement_analytics(df) 
    
    if df.empty:
        st.warning("No data available for analysis.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Engagement Analysis", "🌍 Geographic Analysis", "⏱️ Monthly Trends", "📈 Engagement Effectiveness"])

    with tab1:
        st.markdown("### ESG Focus Distribution")
        esg_data = pd.Series({
            "Environmental": df.get("e", pd.Series(dtype=bool)).astype(bool).sum(),
            "Social": df.get("s", pd.Series(dtype=bool)).astype(bool).sum(),
            "Governance": df.get("g", pd.Series(dtype=bool)).astype(bool).sum()
        })
        if not esg_data.empty and esg_data.sum() > 0:
            st.plotly_chart(create_chart(esg_data, "ESG Focus Distribution", "Flag", "pie"), use_container_width=True)
        else:
            st.info("No ESG focus data to display.")

    with tab2:
        st.markdown("### Geographic Engagement Distribution")
        if "country" in df.columns:
            country_data = df.groupby("country").size().reset_index(name="count")
            fig = px.choropleth(country_data, locations="country", locationmode="country names",
                              color="count", color_continuous_scale="Blues", title="")
            if 'geographic' in CHART_CONFIGS:
                fig.update_layout(**CHART_CONFIGS['geographic'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Monthly Engagement Trends")
        if not analytics_data["monthly_trends"].empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=analytics_data["monthly_trends"]["month"],
                y=analytics_data["monthly_trends"]["new_engagements"],
                mode='lines+markers', name="New Engagements", line=dict(color=Config.COLORS["primary"])
            ))
            fig.update_layout(title="", xaxis_title="", yaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Engagement Effectiveness by Sector")
        if not analytics_data["success_rates"].empty:
            fig = px.bar(analytics_data["success_rates"], x="gics_sector", y="success_rate", 
                         title="", labels={"success_rate": "", "gics_sector": ""})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


def company_deep_dive():
    full_df = st.session_state['FULL_DATA']
    filtered_df = st.session_state['DATA']

    selected_company = company_selector_widget(full_df, filtered_df)
    
    if not selected_company:
        return

    company_data = full_df[full_df["company_name"] == selected_company].iloc[0]
    
    st.markdown(f"## {company_data['company_name']}")
    create_info_display([
        ("Sector", company_data.get("gics_sector", "N/A")),
        ("Region", company_data.get("region", "N/A")),
        ("Status", company_data.get("milestone_status", "N/A")),
        ("Escalation", company_data.get("escalation_level", "N/A"))
    ], use_html=True)

    st.markdown("### ESG Focus Areas")
    esg_focus = [label for flag, label in [("e", "Environmental"), ("s", "Social"), ("g", "Governance")] if company_data.get(flag)]
    st.write(", ".join(esg_focus) if esg_focus else "Not specified")

    st.markdown("### Interaction History")
    display_interaction_history(company_data['engagement_id'])

# --- APP NAVIGATION AND EXECUTION ---

PAGE_FUNCTIONS = {
    "dashboard": dashboard,
    "engagement_management": engagement_operations,
    "task_management": task_management,
    "analytics": enhanced_analysis,
    "company_deep_dive": company_deep_dive,
}

def navigation():
    """Sets up the sidebar navigation and triggers data filtering."""
    with st.sidebar:
        st.markdown(f"### {Config.APP_TITLE}")
        
        page_titles = list(PAGES_CONFIG.keys())
        page_icons = [PAGES_CONFIG[p]['icon'] for p in page_titles]

        selected_page_title = option_menu(
            "Navigation", page_titles,
            icons=page_icons,
            menu_icon="cast", default_index=0,
            styles=NAV_STYLES
        )
        
        st.markdown("---")
        filters = sidebar_filters(st.session_state['FULL_DATA'])
        st.session_state['DATA'] = apply_filters(st.session_state['FULL_DATA'], filters)
    
    return selected_page_title

def main():
    """Main application entry point."""
    st.title(Config.APP_TITLE.split(" Platform")[0])
    
    if 'validator' not in st.session_state:
        df, choices = load_db()
        if df.empty and not choices:
             st.error("Failed to load data or config. The application cannot start.")
             return
        st.session_state.validator = DataValidator(choices)
        st.session_state.FULL_DATA = get_latest_view(df)
        st.session_state.DATA = st.session_state.FULL_DATA.copy()
    
    if st.session_state.FULL_DATA.empty:
        st.warning("No engagement data found. Please add an engagement to begin.")
        engagement_operations()
        return

    try:
        selected_page_title = navigation()
        
        page_function_name = PAGES_CONFIG[selected_page_title]['function']
        page_function_to_call = PAGE_FUNCTIONS[page_function_name]
        page_function_to_call()

    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}")
        st.exception(e)
        if st.button("Clear Cache and Reload"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()