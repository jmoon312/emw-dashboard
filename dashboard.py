"""
8-K Item Timeliness and Materiality Dashboard

This dashboard analyzes regression results examining the timeliness (currentness) 
and materiality of 8-K disclosure items.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import io

# Page configuration
st.set_page_config(
    page_title="8-K Item Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("8-K Item Timeliness and Materiality Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load regression results and residual standard deviations"""
    try:
        results = pd.read_csv('regression_results.csv')
        residual_sds = pd.read_csv('residual_sds.csv')
        return results, residual_sds
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'regression_results.csv' and 'residual_sds.csv' are in the same directory.")
        st.stop()

results_df, residual_sds_df = load_data()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Time horizon selection
time_horizon = st.sidebar.selectbox(
    "Time Horizon",
    options=["48 hours", "24 hours", "2 hours", "1 hour"],
    index=1  # Default to 24 hours
)

# Map time horizon to window names
horizon_map = {
    "48 hours": ("48h_before", "48h_after"),
    "24 hours": ("24h_before", "24h_after"),
    "2 hours": ("2h_before", "2h_after"),
    "1 hour": ("1h_before", "1h_after")
}
pre_window, post_window = horizon_map[time_horizon]

# Significance threshold
sig_threshold = st.sidebar.slider(
    "Significance Level for 'Current' Classification (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=1.0
) / 100

# Materiality threshold
materiality_threshold = st.sidebar.slider(
    "Materiality Threshold (as % of Residual SD)",
    min_value=5.0,
    max_value=50.0,
    value=20.0,
    step=1.0
) / 100

# Coefficient scaling toggle
scale_by_sd = st.sidebar.checkbox(
    "Scale coefficients by residual standard deviations",
    value=False
)

st.sidebar.markdown("---")
st.sidebar.markdown("*Adjust controls to dynamically update results*")

# Helper functions
def get_critical_value(alpha):
    """Get critical value for two-tailed test"""
    return stats.norm.ppf(1 - alpha/2)

def is_significant(t_stat, alpha):
    """Check if t-statistic is significant at given alpha level"""
    critical_value = get_critical_value(alpha)
    return abs(t_stat) > critical_value

def get_significance_stars(t_stat):
    """Get significance stars based on t-statistic"""
    if abs(t_stat) > stats.norm.ppf(1 - 0.01/2):
        return "***"
    elif abs(t_stat) > stats.norm.ppf(1 - 0.05/2):
        return "**"
    elif abs(t_stat) > stats.norm.ppf(1 - 0.10/2):
        return "*"
    else:
        return ""

def get_coefficient_value(item, dv, window, scaled=False):
    """Get coefficient value for specific item, DV, and window"""
    row = results_df[
        (results_df['Item'] == item) & 
        (results_df['Dependent_Variable'] == dv) & 
        (results_df['Window'] == window)
    ]
    
    if row.empty:
        return None, None
    
    coef = row['Coefficient'].values[0]
    t_stat = row['T_Statistic'].values[0]
    
    if scaled:
        sd_row = residual_sds_df[
            (residual_sds_df['Dependent_Variable'] == dv) & 
            (residual_sds_df['Window'] == window)
        ]
        if not sd_row.empty:
            sd = sd_row['Residual_SD'].values[0]
            coef = coef / sd
    
    return coef, t_stat

def is_current(item):
    """Check if item is current based on total volume"""
    post_coef, post_t = get_coefficient_value(item, 'Total_Volume', post_window)
    pre_coef, pre_t = get_coefficient_value(item, 'Total_Volume', pre_window)
    
    if post_coef is None or pre_coef is None:
        return False
    
    # Post must be positive and significant
    post_significant = is_significant(post_t, sig_threshold)
    post_positive = post_coef > 0
    
    # Pre must be insignificant or negative
    pre_insignificant = not is_significant(pre_t, sig_threshold)
    pre_negative = pre_coef <= 0
    
    return post_positive and post_significant and (pre_insignificant or pre_negative)

def is_material(item):
    """Check if item is material based on absolute returns"""
    post_coef, _ = get_coefficient_value(item, 'Absolute_Returns', post_window)
    pre_coef, _ = get_coefficient_value(item, 'Absolute_Returns', pre_window)
    
    if post_coef is None or pre_coef is None:
        return False
    
    # Get residual SD for returns
    sd_row = residual_sds_df[
        (residual_sds_df['Dependent_Variable'] == 'Absolute_Returns') & 
        (residual_sds_df['Window'] == post_window)
    ]
    
    if sd_row.empty:
        return False
    
    sd = sd_row['Residual_SD'].values[0]
    
    # Sum of pre and post > threshold * SD
    total_effect = pre_coef + post_coef
    return total_effect > (materiality_threshold * sd)

def has_retail_flag(item):
    """Check if retail responds when total volume doesn't"""
    # Retail post must be positive and significant
    retail_post_coef, retail_post_t = get_coefficient_value(item, 'Retail_Volume', post_window)
    if retail_post_coef is None:
        return False
    
    retail_significant = is_significant(retail_post_t, sig_threshold)
    retail_positive = retail_post_coef > 0
    
    # Total volume post must be insignificant or not positive
    total_post_coef, total_post_t = get_coefficient_value(item, 'Total_Volume', post_window)
    if total_post_coef is None:
        return False
    
    total_insignificant = not is_significant(total_post_t, sig_threshold)
    total_not_positive = total_post_coef <= 0
    
    return retail_positive and retail_significant and (total_insignificant or total_not_positive)

# Classify all items
items = results_df['Item'].unique()
classifications = {
    'current_material': [],
    'current_not_material': [],
    'material_not_current': [],
    'neither': []
}

for item in items:
    current = is_current(item)
    material = is_material(item)
    
    if current and material:
        classifications['current_material'].append(item)
    elif current and not material:
        classifications['current_not_material'].append(item)
    elif material and not current:
        classifications['material_not_current'].append(item)
    else:
        classifications['neither'].append(item)

# Summary statistics
st.header("Summary")
summary_text = f"""Under current settings, **{len(classifications['current_material'])} items** are Current & Material, 
**{len(classifications['current_not_material'])} items** are Current but Not Material, 
**{len(classifications['material_not_current'])} items** are Material but Not Current, and 
**{len(classifications['neither'])} items** are Neither Current nor Material."""
st.markdown(summary_text)
st.markdown("---")

# Build classification table
def build_classification_table():
    """Build the full classification table"""
    table_data = []
    
    for category in ['current_material', 'current_not_material', 'material_not_current', 'neither']:
        for item in classifications[category]:
            # Get item description
            desc_row = results_df[results_df['Item'] == item].iloc[0]
            description = desc_row['Item_Description']
            
            # Get all coefficients
            row_data = {
                'Category': category.replace('_', ' ').title(),
                'Item': item,
                'Description': description
            }
            
            # Total Volume
            tv_pre_coef, tv_pre_t = get_coefficient_value(item, 'Total_Volume', pre_window, scale_by_sd)
            tv_post_coef, tv_post_t = get_coefficient_value(item, 'Total_Volume', post_window, scale_by_sd)
            row_data['Total_Vol_Pre'] = f"{tv_pre_coef:.4f}{get_significance_stars(tv_pre_t)}"
            row_data['Total_Vol_Post'] = f"{tv_post_coef:.4f}{get_significance_stars(tv_post_t)}"
            
            # Retail Volume
            rv_pre_coef, rv_pre_t = get_coefficient_value(item, 'Retail_Volume', pre_window, scale_by_sd)
            rv_post_coef, rv_post_t = get_coefficient_value(item, 'Retail_Volume', post_window, scale_by_sd)
            row_data['Retail_Vol_Pre'] = f"{rv_pre_coef:.4f}{get_significance_stars(rv_pre_t)}"
            row_data['Retail_Vol_Post'] = f"{rv_post_coef:.4f}{get_significance_stars(rv_post_t)}"
            
            # Absolute Returns
            ar_pre_coef, ar_pre_t = get_coefficient_value(item, 'Absolute_Returns', pre_window, scale_by_sd)
            ar_post_coef, ar_post_t = get_coefficient_value(item, 'Absolute_Returns', post_window, scale_by_sd)
            row_data['Returns_Pre'] = f"{ar_pre_coef:.4f}{get_significance_stars(ar_pre_t)}"
            row_data['Returns_Post'] = f"{ar_post_coef:.4f}{get_significance_stars(ar_post_t)}"
            
            # Retail flag
            row_data['Retail_Flag'] = "🔔" if has_retail_flag(item) else ""
            
            table_data.append(row_data)
    
    return pd.DataFrame(table_data)

# Create and display table
st.header("Classification Table")
classification_df = build_classification_table()

# Display table grouped by category
for category in ['Current Material', 'Current Not Material', 'Material Not Current', 'Neither']:
    st.subheader(category)
    category_data = classification_df[classification_df['Category'] == category]
    if not category_data.empty:
        display_df = category_data.drop('Category', axis=1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.write("*No items in this category*")

# Download button for table
st.markdown("---")
csv_buffer = io.StringIO()
classification_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Table as CSV",
    data=csv_buffer.getvalue(),
    file_name="8k_classification_table.csv",
    mime="text/csv"
)

st.markdown("---")

# Visualization section
st.header("Coefficient Visualizations")
st.markdown("*Bars show raw coefficients with 95% confidence intervals*")

def create_bar_chart(dv, dv_name):
    """Create bar chart for a specific dependent variable"""
    # Get data for all items
    chart_data = []
    
    for item in items:
        pre_coef, pre_t = get_coefficient_value(item, dv, pre_window, scaled=False)
        post_coef, post_t = get_coefficient_value(item, dv, post_window, scaled=False)
        
        if pre_coef is not None and post_coef is not None:
            # Calculate standard errors and confidence intervals
            pre_se = pre_coef / pre_t if pre_t != 0 else 0
            post_se = post_coef / post_t if post_t != 0 else 0
            
            pre_ci_lower = pre_coef - 1.96 * abs(pre_se)
            pre_ci_upper = pre_coef + 1.96 * abs(pre_se)
            post_ci_lower = post_coef - 1.96 * abs(post_se)
            post_ci_upper = post_coef + 1.96 * abs(post_se)
            
            chart_data.append({
                'item': item,
                'pre_coef': pre_coef,
                'post_coef': post_coef,
                'pre_ci_lower': pre_ci_lower,
                'pre_ci_upper': pre_ci_upper,
                'post_ci_lower': post_ci_lower,
                'post_ci_upper': post_ci_upper
            })
    
    # Sort by post coefficient (descending)
    chart_data = sorted(chart_data, key=lambda x: x['post_coef'], reverse=True)
    
    # Create figure
    fig = go.Figure()
    
    # Add post-period bars (green)
    fig.add_trace(go.Bar(
        name='Post-Period',
        y=[d['item'] for d in chart_data],
        x=[d['post_coef'] for d in chart_data],
        orientation='h',
        marker_color='green',
        error_x=dict(
            type='data',
            symmetric=False,
            array=[d['post_ci_upper'] - d['post_coef'] for d in chart_data],
            arrayminus=[d['post_coef'] - d['post_ci_lower'] for d in chart_data]
        )
    ))
    
    # Add pre-period bars (red)
    fig.add_trace(go.Bar(
        name='Pre-Period',
        y=[d['item'] for d in chart_data],
        x=[d['pre_coef'] for d in chart_data],
        orientation='h',
        marker_color='red',
        error_x=dict(
            type='data',
            symmetric=False,
            array=[d['pre_ci_upper'] - d['pre_coef'] for d in chart_data],
            arrayminus=[d['pre_coef'] - d['pre_ci_lower'] for d in chart_data]
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{dv_name} Coefficients (Sorted by Post-Period)",
        xaxis_title="Coefficient Estimate",
        yaxis_title="8-K Item",
        barmode='group',
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# Create charts for each dependent variable
st.subheader("Total Volume")
fig_tv = create_bar_chart('Total_Volume', 'Total Volume')
st.plotly_chart(fig_tv, use_container_width=True)

st.subheader("Retail Volume")
fig_rv = create_bar_chart('Retail_Volume', 'Retail Volume')
st.plotly_chart(fig_rv, use_container_width=True)

st.subheader("Absolute Returns")
fig_ar = create_bar_chart('Absolute_Returns', 'Absolute Returns')
st.plotly_chart(fig_ar, use_container_width=True)

# Download button for charts (requires kaleido and Chrome)
st.markdown("---")
try:
    col1, col2, col3 = st.columns(3)

    with col1:
        img_bytes_tv = fig_tv.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Total Volume Chart",
            data=img_bytes_tv,
            file_name="total_volume_chart.png",
            mime="image/png"
        )

    with col2:
        img_bytes_rv = fig_rv.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Retail Volume Chart",
            data=img_bytes_rv,
            file_name="retail_volume_chart.png",
            mime="image/png"
        )

    with col3:
        img_bytes_ar = fig_ar.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Returns Chart",
            data=img_bytes_ar,
            file_name="returns_chart.png",
            mime="image/png"
        )
except Exception as e:
    st.info("💡 Chart downloads require Kaleido and Chrome. You can still interact with the charts above and use your browser's screenshot feature to save them.")

# Footer
st.markdown("---")
st.markdown("*Dashboard for analyzing 8-K item timeliness and materiality | Data updates dynamically with control changes*")
