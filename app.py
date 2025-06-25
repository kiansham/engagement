from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
from typing import Dict, List, Optional, Tuple

from config import Config, CHART_CONFIGS, ENHANCED_CSS, NAV_STYLES, PAGES_CONFIG
from utils import (
    get_latest_view, get_upcoming_tasks,
    create_engagement, log_interaction, update_milestone_status,
    get_lookup_values,
    get_engagement_analytics, get_interactions_for_company,
    load_db,
    DataValidator,
    render_metrics, create_chart, create_esg_gauge,
    handle_task_date_display, company_selector_widget, display_interaction_history,
    get_themes_for_row, render_icon_header, render_hr, create_aggrid_component,
    get_esg_selection, fix_column_names
)

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    '<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">',
    unsafe_allow_html=True
)

st.markdown(ENHANCED_CSS, unsafe_allow_html=True)

def create_alert_section(this_week_tasks, this_month_tasks):
    col1, col2 = st.columns(2)
    col1.markdown(f"""<div class=\"alert-urgent\"><strong>📅 {len(this_week_tasks)} Meetings This Week</strong><br>
                     Meetings within 7 days</div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class=\"alert-warning\"><strong>🗓️ {len(this_month_tasks)} Meeting This Month</strong><br>
                     Meetings within 30 days</div>""", unsafe_allow_html=True)

def sidebar_filters(df):
    st.markdown(
        f'''
        <span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:18px;font-weight:300;">{Config.HEADER_ICONS["filter"]}</span>
        <span style="vertical-align:middle;font-size:18px;font-weight:600;">Filters</span>
        ''',
        unsafe_allow_html=True
    )

    with st.expander("⚠️ Alerts", expanded=False):
        st.markdown('<span style="font-size:14px; font-weight:200; margin-bottom:-35px; display:block;">Select to Show Upcoming Events</span>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            show_urgent = st.toggle("Urgent", value=False)
        with col2:
            show_upcoming = st.toggle("Upcoming", value=False)

    with st.expander("🏛️ Company Filters", expanded=False):
        companies = st.multiselect("Company Name", sorted(df["company_name"].unique()) if "company_name" in df.columns else [])
        region = st.multiselect("Region", get_lookup_values("region"))
        country = st.multiselect("Country", get_lookup_values("country"))
        sector = st.multiselect("GICS Sector", get_lookup_values("gics_sector"))
    with st.expander("🗣️ Engagement Type", expanded=False):
        progs = st.multiselect("Engagement Program", get_lookup_values("program"))
        themes = st.multiselect("Theme", get_lookup_values("theme"))
        objectives = st.multiselect("Objective", get_lookup_values("objective"))
        esg_option = st.radio("ESG Focus", ["All", "E", "S", "G"], index=0, horizontal=True)
    if esg_option == "All":
        esg = ["e", "s", "g"]
    else:
        esg = [esg_option.lower()]
    with st.expander("👥 Engagement Status", expanded=False):
        mile = st.multiselect("Milestone", get_lookup_values("milestone"))
        status = st.multiselect("Status", get_lookup_values("milestone_status"))

    return progs, sector, region, country, mile, status, esg, show_urgent, show_upcoming, companies, themes, objectives

def apply_filters(df, filters):
    if df.empty:
        return df

    progs, sector, region, country, mile, status, esg, show_urgent, show_upcoming, companies, themes, objectives = filters
    filtered_df = df.copy()

    filter_map = {"program": progs, "gics_sector": sector, "region": region,
                 "country": country, "milestone": mile, "milestone_status": status, "company_name": companies}

    for col, values in filter_map.items():
        if values and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    if themes and "theme" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["theme"].isin(themes)]

    if esg and all(col in filtered_df.columns for col in esg):
        filtered_df = filtered_df[filtered_df[esg].any(axis=1)]

    if show_urgent and "urgent" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["urgent"]]
    if show_upcoming and "next_action_date" in filtered_df.columns:
        today = pd.to_datetime(datetime.now().date())
        filtered_df = filtered_df[(pd.to_datetime(filtered_df["next_action_date"]) - today).dt.days.between(0, 30, inclusive="both")]

    if objectives and "objective" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["objective"].isin(objectives)]

    return filtered_df

def dashboard():
    data = st.session_state['DATA']
    
    if data.empty:
        st.warning("No engagement data available or no data matches the current filters.")
        return

    today = pd.to_datetime(datetime.now().date())
    this_week = data[(pd.to_datetime(data.get("next_action_date")) - today).dt.days.between(0, 6, inclusive="both")]
    this_month = data[(pd.to_datetime(data.get("next_action_date")) - today).dt.days.between(7, 30, inclusive="both")]
    create_alert_section(this_week, this_month)

    render_icon_header(Config.HEADER_ICONS["metrics"], "Key Metrics")
    
    total = len(data)
    # Active Engagements: milestone not in the excluded list
    exclude_milestones_active = ["not started", "verified", "success", "cancelled"]
    active = data["milestone"].str.lower().apply(lambda x: x not in exclude_milestones_active if pd.notna(x) else False).sum() if "milestone" in data.columns else 0

    # Completed Engagements: milestone in the completed list
    completed_milestones = ["success", "full disclosure", "partial disclosure", "verified"]
    completed = data["milestone"].str.lower().isin([m.lower() for m in completed_milestones]).sum() if "milestone" in data.columns else 0

    # Success Rate
    success_milestones = ["Success", "Full Disclosure", "Partial Disclosure", "Verified"]
    success_count = data["milestone"].isin(success_milestones).sum() if "milestone" in data.columns else 0
    success_rate = round((success_count / total * 100)) if total > 0 else 0

    # Not Started count: milestone == 'Not Started' (case-insensitive)
    not_started_count = data["milestone"].str.lower().eq("not started").sum() if "milestone" in data.columns else 0

    # Failed Engagements: milestone == 'cancelled' (case-insensitive)
    failed_count = data["milestone"].str.lower().eq("cancelled").sum() if "milestone" in data.columns else 0

    # Fail Rate: percentage of engagements where milestone == 'cancelled'
    fail_rate = round((failed_count / total * 100)) if total > 0 else 0

    # Engagement Health Status (distribution of active engagements by milestone_status)
    active_engagements = data[data["milestone_status"].str.lower() != "complete"] if "milestone_status" in data.columns else data
    health_counts = (
        active_engagements["milestone_status"].value_counts()
        if "milestone_status" in active_engagements.columns else pd.Series()
    )
    health_statuses = ["Red", "Amber", "Green"]
    health_data = {status: health_counts.get(status, 0) for status in health_statuses}

    # Layout: metrics in two columns, milestone chart on right
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        st.metric("Total Engagements", total)
        st.markdown(" ")
        st.metric("Not Started", not_started_count)
        st.markdown(" ")
        st.metric("Success Rate", f"{success_rate}%")
    with col2:
        st.metric("Active Engagements", active)
        st.markdown(" ")
        st.metric("Completed", completed)
        st.markdown(" ")
        st.metric("Fail Rate", f"{fail_rate}%")
    with col3:
        st.markdown(f'<div style="margin-top:-50px; margin-bottom:8px;"><span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:40px;font-weight:100;">{Config.HEADER_ICONS["milestone"]}</span><span style="vertical-align:middle;font-size:28px;font-weight:600;margin-left:10px;">Milestone Progress</span></div>', unsafe_allow_html=True)
        st.write(Config.CHART_CONTEXTS["milestone"])
        if "milestone" in data.columns and not data["milestone"].dropna().empty:
            chart_data = data["milestone"].value_counts()
            fig = create_chart(chart_data, chart_type="bar")
            st.plotly_chart(fig, use_container_width=True)

    render_icon_header(Config.HEADER_ICONS["esg"], "ESG Engagement Focus Areas")
    st.markdown(
        '<div style="margin-bottom:-100px;">'
        '<span style="font-size:16px; color:#6c757d;">Distribution of engagements across CDP <b>Climate Change</b>, <b>Water</b>, <b>Forests</b>, and <b>Other</b> themes. Shows active engagements with percentage of total portfolio.</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    esg_data = {}
    total_unfiltered = len(st.session_state['FULL_DATA'])
    
    if total > 0 and total_unfiltered > 0:
        theme_columns = ["Climate Change", "Water", "Forests", "Other"]
        total_theme_ys = sum((data[col] == "Y").sum() for col in theme_columns if col in data.columns)
        for theme, col in [("Climate Change", "Climate Change"), ("Water", "Water"), ("Forests", "Forests"), ("Other", "Other")]:
            if col in data.columns:
                count = (data[col] == "Y").sum()
                percentage_of_total = round((count / total_theme_ys) * 100) if total_theme_ys > 0 else 0
                esg_data[theme] = (count, percentage_of_total)
            else:
                esg_data[theme] = (0, 0)
    
    if esg_data:
        cols = st.columns(4)
        for i, (theme, (count, percentage)) in enumerate(esg_data.items()):
            with cols[i]:
                gauge_percentage = round((count / total_theme_ys) * 100) if total_theme_ys > 0 else 0
                option = create_esg_gauge(theme, count, Config.ESG_COLORS[theme], gauge_percentage)
                option["series"][0]["detail"]["formatter"] = str(count)
                option["series"][0]["data"][0]["value"] = gauge_percentage
                st_echarts(options=option, height="300px", key=f"esg-{theme}")

    render_icon_header(Config.HEADER_ICONS["table"], "Engagement Table")
    create_aggrid_component(data, Config.AGGRID_COLUMNS)

    render_hr(margin_top=100, margin_bottom=100)

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
            esg = get_esg_selection()

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
                    **{"e": "e" in esg, "s": "s" in esg, "g": "g" in esg}
                }
                success, message = create_engagement(engagement_data)
                if success:
                    st.success(message)
                    st.balloons()
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
            cols = st.columns(3)
            cols[0].markdown(f"**Current Milestone**<br>{engagement_data.get('milestone', 'N/A')}", unsafe_allow_html=True)
            cols[1].markdown(f"**Status**<br>{engagement_data.get('milestone_status', 'N/A')}", unsafe_allow_html=True)
            cols[2].markdown(f"**Escalation**<br>{engagement_data.get('escalation_level', 'N/A')}", unsafe_allow_html=True)

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
                    with st.spinner('Logging interaction...'):
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

    all_upcoming_tasks = get_upcoming_tasks(df=filtered_df, days=Config.UPCOMING_DAYS)
    urgent_tasks = all_upcoming_tasks[all_upcoming_tasks['days_to_next_action'] <= Config.URGENT_DAYS]
    warning_tasks = all_upcoming_tasks[(all_upcoming_tasks['days_to_next_action'] > Config.URGENT_DAYS) & 
                                      (all_upcoming_tasks['days_to_next_action'] <= Config.WARNING_DAYS)]

    render_metrics([
        (f"Urgent (≤{Config.URGENT_DAYS} days)", len(urgent_tasks)),
        (f"Warning (≤{Config.WARNING_DAYS} days)", len(warning_tasks)),
        (f"Upcoming (≤{Config.UPCOMING_DAYS} days)", len(all_upcoming_tasks))
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
                        render_hr(margin_top=4, margin_bottom=8)
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
        context_col, chart_col = st.columns([1.5, 1])
        with context_col:
            st.markdown(f'<div style="margin-top:8px; margin-bottom:8px;"><span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:40px;font-weight:100;">{Config.HEADER_ICONS["esg"]}</span><span style="vertical-align:middle;font-size:28px;font-weight:600;margin-left:10px;">ESG Focus Distribution</span></div>', unsafe_allow_html=True)
            st.write("Distribution of engagements by ESG focus area (Environmental, Social, Governance). Shows count of active engagements.")
        with chart_col:
            esg_data = pd.Series({
                "Environmental": df.get("e", pd.Series(dtype=bool)).astype(bool).sum(),
                "Social": df.get("s", pd.Series(dtype=bool)).astype(bool).sum(),
                "Governance": df.get("g", pd.Series(dtype=bool)).astype(bool).sum()
            })
            if not esg_data.empty and esg_data.sum() > 0:
                fig = create_chart(esg_data, chart_type="pie")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ESG focus data to display.")

        if "gics_sector" in df.columns and not df["gics_sector"].dropna().empty:
            context_col, chart_col = st.columns([1, 2])
            with context_col:
                st.markdown(f'<div style="margin-top:8px; margin-bottom:8px;"><span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:40px;font-weight:100;">{Config.HEADER_ICONS["sector"]}</span><span style="vertical-align:middle;font-size:28px;font-weight:600;margin-left:10px;">Sector Distribution</span></div>', unsafe_allow_html=True)
                st.write(Config.CHART_CONTEXTS["sector"])
            with chart_col:
                chart_data = df["gics_sector"].value_counts()
                fig = create_chart(chart_data, chart_type="bar")
                st.plotly_chart(fig, use_container_width=True)
        if "region" in df.columns and not df["region"].dropna().empty:
            regions = sorted(df["region"].dropna().unique())
            selected_region = st.selectbox("Select Region to break out by country", ["All Regions"] + regions, key="analytics_region_selector")
            if selected_region == "All Regions":
                region_df = df
            else:
                region_df = df[df["region"] == selected_region]
            context_col, chart_col = st.columns([1, 2])
            with context_col:
                st.markdown(f'<div style="margin-top:8px; margin-bottom:8px;"><span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:40px;font-weight:100;">{Config.HEADER_ICONS["region"]}</span><span style="vertical-align:middle;font-size:28px;font-weight:600;margin-left:10px;">Regional Distribution</span></div>', unsafe_allow_html=True)
                st.write(Config.CHART_CONTEXTS["region"])
                if selected_region != "All Regions":
                    st.write(f"Showing country breakdown for region: **{selected_region}**")
            with chart_col:
                if not region_df.empty and "country" in region_df.columns:
                    chart_data = region_df["country"].value_counts()
                    fig = create_chart(chart_data, chart_type="bar")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No country data to display for this region.")

    with tab2:
        st.markdown("### Geographic Engagement Distribution")
        
        available_regions = sorted(df["region"].unique())
        selected_region = st.selectbox(
            "Select Region",
            ["Global"] + available_regions,
            index=0,
            help="Select a region to view focused stats and map.",
            key="geo_region_selector"
        )

        if selected_region != "Global":
            geo_df = df[df["region"] == selected_region]
        else:
            geo_df = df

        col1, col2 = st.columns([1, 3])

        with col1:
            if not geo_df.empty:
                total_engagements = len(geo_df)
                countries_engaged = geo_df["country"].nunique()
                most_active_country = geo_df["country"].mode()[0] if not geo_df["country"].empty else "N/A"
                
                st.metric("Total Engagements", total_engagements)
                st.metric("Countries Engaged", countries_engaged)
                st.metric("Most Active Country", most_active_country)
            else:
                st.info("No engagement data for the selected region.")

        with col2:
            if not geo_df.empty and "country" in geo_df.columns:
                country_data = geo_df.groupby("country").size().reset_index(name="count")
                
                country_data['iso_code'] = country_data['country'].map(Config.COUNTRY_ISO_MAP)
                mapped_countries = country_data.dropna(subset=['iso_code']).copy()
                
                if not mapped_countries.empty:
                    custom_colorscale = [
                        [0.0, Config.CB_SAFE_PALETTE[4]],
                        [0.25, Config.CB_SAFE_PALETTE[0]],
                        [0.5, Config.CB_SAFE_PALETTE[2]],
                        [0.75, Config.CB_SAFE_PALETTE[1]],
                        [1.0, Config.CB_SAFE_PALETTE[3]]
                    ]
                    
                    fig = create_chart(
                        mapped_countries,
                        chart_type="choropleth",
                        locations="iso_code",
                        color="count",
                        hover_name="country",
                        color_continuous_scale=custom_colorscale,
                        range_color=[0, mapped_countries['count'].max()]
                    )
                    
                    fig.update_layout(
                        geo=dict(
                            bgcolor='rgba(0,0,0,0)',
                            showframe=False,
                            showcoastlines=True,
                            coastlinecolor="rgba(68, 68, 68, 0.15)",
                            projection_type='natural earth',
                            showcountries=True,
                            countrycolor="rgba(68, 68, 68, 0.15)",
                            showland=True,
                            landcolor='rgb(243, 243, 243)',
                            showocean=True,
                            oceancolor='rgb(230, 235, 240)',
                            showlakes=True,
                            lakecolor='rgb(230, 235, 240)',
                            fitbounds="locations",
                            visible=True
                        ),
                        height=400,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    
                    fig.update_coloraxes(
                        colorbar=dict(
                            thickness=15,
                            len=0.7,
                            x=1.02,
                            xpad=10,
                            y=0.5
                        )
                    )
                    
                    fig.update_traces(
                        hovertemplate="<b>%{hovertext}</b><br>Engagements: %{z}<extra></extra>"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = create_chart(
                        country_data.sort_values('count', ascending=True).tail(10),
                        chart_type="bar",
                        x='count', y='country', orientation='h',
                        color='count', color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geographic data to display.")

    with tab3:
        st.subheader("Monthly Engagement Trends")
        if not analytics_data["monthly_trends"].empty:
            with st.spinner('Loading trend data...'):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=analytics_data["monthly_trends"]["month"],
                    y=analytics_data["monthly_trends"]["new_engagements"],
                    mode='lines+markers', 
                    name="New Engagements", 
                    line=dict(color=Config.CB_SAFE_PALETTE[0], width=3),
                    marker=dict(size=8, color=Config.CB_SAFE_PALETTE[0])
                ))
                fig.update_layout(
                    title="", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified'
                )
                fig.update_xaxes(title="")
                fig.update_yaxes(title="")
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Engagement Effectiveness by Sector")
        if not analytics_data["success_rates"].empty:
            with st.spinner('Loading effectiveness data...'):
                fig = px.bar(
                    analytics_data["success_rates"], 
                    x="gics_sector", 
                    y="success_rate", 
                    title="", 
                    color="success_rate",
                    color_continuous_scale=[Config.CB_SAFE_PALETTE[4], Config.CB_SAFE_PALETTE[0]]
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.update_xaxes(title="")
                fig.update_yaxes(title="")
                st.plotly_chart(fig, use_container_width=True)

def company_deep_dive():
    full_df = st.session_state['FULL_DATA']
    filtered_df = st.session_state['DATA']

    selected_company = company_selector_widget(full_df, filtered_df)
    if not selected_company:
        return

    company_data = full_df[full_df["company_name"] == selected_company].iloc[0]

    # --- Header ---
    with st.container(border=True):
        render_icon_header("apartment", f"Company Snapshot: {company_data['company_name']}", icon_size=32, text_size=28)
        # --- Company Info Card ---
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.markdown(f"**Sector:** {company_data.get('gics_sector', 'N/A')}")
        with col2:
            st.markdown(f"**Region:** {company_data.get('region', 'N/A')}")
        with col3:
            st.markdown(f"**Country:** {company_data.get('country', 'N/A')}")
        render_hr(margin_top=0, margin_bottom=0)

    with st.container(border=True):
        # --- Metrics Snapshot ---
        days_since_contact = (datetime.now() - company_data['last_interaction_date']).days if pd.notna(company_data['last_interaction_date']) else 'N/A'
        days_to_next_action = (company_data['next_action_date'] - datetime.now()).days if pd.notna(company_data['next_action_date']) else 'N/A'
        total_interactions = len(get_interactions_for_company(company_data['engagement_id']))
        render_icon_header("camera_alt", f"Engagement Snapshot", icon_size=38, text_size=28)
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.markdown(f"**Program:** {company_data.get('program', 'N/A')}")
        with col2:
            st.markdown(f"**Objective:** {company_data.get('objective', 'N/A')}")
        with col3:
            st.markdown(f"**Engagement Health:** {company_data.get('milestone_status', 'N/A')}")

    #st.markdown("**Themes**")
    #themes = [label for flag, label in [("Climate Change", "Climate Change"), ("Forests", "Forests"), ("Water", "Water"), ("Other", "Other")] if company_data.get(flag)]
   # st.write(", ".join(themes) if themes else "Not specified")

   # st.markdown("**Milestones**")
#**Days Since Last Contact:** {days_since_contact}  
#**Next Action Due (Days):** {days_to_next_action}  
#**Total Interactions:** {total_interactions}  
#**Escalation Level:** {company_data.get('escalation_level', 'N/A')}  
#**Status:** {company_data.get('milestone', 'N/A')}
#""")
    render_hr(margin_top=10, margin_bottom=10)

    #--- Interaction History ---
    render_icon_header("history", "Interaction History")
    display_interaction_history(company_data['engagement_id'])
    render_hr(margin_top=10, margin_bottom=10)

PAGE_FUNCTIONS = {
    "dashboard": dashboard,
    "engagement_management": engagement_operations,
    "task_management": task_management,
    "analytics": enhanced_analysis,
    "company_deep_dive": company_deep_dive,
}

def navigation():
    with st.sidebar:
        st.markdown(""" \n """)
        
        page_titles = list(PAGES_CONFIG.keys())
        page_icons = [PAGES_CONFIG[p]['icon'] for p in page_titles]
        
        try:
            default_index = page_titles.index(st.session_state.get('selected_page', 'Dashboard'))
        except ValueError:
            default_index = 0

        selected_page_title = option_menu(
            "Navigation", page_titles,
            icons=page_icons,
            menu_icon="cast", 
            default_index=default_index,
            styles=NAV_STYLES
        )
        
        st.session_state.selected_page = selected_page_title

        render_hr(margin_top=0, margin_bottom=-0)

        # Filtering heading
        col1, col2 = st.columns([5, 2.5])
        with col1:
            st.markdown(
                f'''
                <div style="margin-left:15px; margin-top:0px; margin-bottom:2px;">
                    <span class="material-icons-outlined" style="vertical-align:middle;color:#333333;font-size:22px;font-weight:300;">{Config.HEADER_ICONS["filter"]}</span>
                    <span style="vertical-align:middle;font-size:20px;font-weight:500;margin-left:5px;">Toggle Filtering</span>
                </div>
                ''',
                unsafe_allow_html=True
            )
        with col2:
            enable_filtering = st.toggle(
                "",
                label_visibility = "visible",
                value=False,
                key="enable_filtering_toggle",
                help="Filtering is active when toggled on. To reset filters Toggle off." 
            )

        render_hr(margin_top=-0, margin_bottom=8)
        
        if enable_filtering:
            filters = sidebar_filters(st.session_state['FULL_DATA'])
            st.session_state['DATA'] = apply_filters(st.session_state['FULL_DATA'], filters)
        else:
            st.session_state['DATA'] = st.session_state['FULL_DATA'].copy()

def main():
    render_icon_header(Config.HEADER_ICONS["app_title"], Config.APP_TITLE, icon_size=32, text_size=32)
    st.markdown('<div style="margin-top:-33px;"></div>', unsafe_allow_html=True)
    render_hr()
    
    if 'validator' not in st.session_state:
        with st.spinner('Loading data...'):
            df, choices = load_db()
            df = fix_column_names(df)
            if df.empty and not choices:
                st.error("Failed to load data or config. The application cannot start.")
                return
            st.session_state.validator = DataValidator(choices)
            st.session_state.FULL_DATA = get_latest_view(df)
            st.session_state.DATA = st.session_state.FULL_DATA.copy()
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'Dashboard'

    if st.session_state.FULL_DATA.empty:
        st.warning("No engagement data found. Please add an engagement to begin.")
        engagement_operations()
        return

    try:
        navigation()
        
        page_function_name = PAGES_CONFIG[st.session_state.selected_page]['function']
        page_function_to_call = PAGE_FUNCTIONS[page_function_name]
        
        with st.spinner(f'Loading {st.session_state.selected_page}...'):
            page_function_to_call()

    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}")
        st.exception(e)
        if st.button("Clear Cache and Reload"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()