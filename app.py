import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Bulacan Solar - Daily Inverter Screening Report",
    page_icon="☀️",
    layout="wide"
)

# =========================================================================
# EQUIPMENT CAPACITY LOOKUP TABLE
# =========================================================================
DC_CAPACITIES = {
    'Inverter1-1': 4.4892, 'Inverter1-2': 4.4892,
    'Inverter2-1': 4.4892, 'Inverter2-2': 4.4892,
    'Inverter3-1': 4.1022, 'Inverter3-2': 4.4892,
    'Inverter4-1': 4.4763, 'Inverter4-2': 4.4892,
    'Inverter5-1': 4.4892, 'Inverter5-2': 4.4505,
    'Inverter6-1': 4.4505, 'Inverter6-2': 4.4505,
    'Inverter7-1': 4.4634, 'Inverter7-2': 4.4505,
    'Inverter8-1': 4.4505, 'Inverter8-2': 4.4505,
    'Inverter9-1': 4.4505, 'Inverter9-2': 4.4505
}

# =========================================================================
# SIDEBAR DATA UPLOAD
# =========================================================================
st.sidebar.header("📁 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Daily SCADA Report (.xls / .xlsx)", type=["xls", "xlsx"])

if uploaded_file is not None:
    # 1. Load Data
    @st.cache_data
    def load_data(file):
        df_raw = pd.read_excel(file, sheet_name="Inverter Generation Report", header=None)
        return df_raw

    df_raw = load_data(uploaded_file)
    
    # Extract Report Date from Row 0 if available (fallback to current or placeholder)
    report_date_str = "Previous Operating Day"
    try:
        cell_val = str(df_raw.iloc[0, 0])
        date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})', cell_val)
        if date_match:
            report_date_str = date_match.group(1)
    except Exception:
        pass

    # 2. Parse Headers & Time
    inverter_names_row = df_raw.iloc[1].ffill()
    parameters_row = df_raw.iloc[2]

    columns = ['Time']
    for col_idx in range(1, df_raw.shape[1]):
        columns.append(f"{inverter_names_row.iloc[col_idx]}_{parameters_row.iloc[col_idx]}")

    df_data = df_raw.iloc[3:].copy()
    df_data.columns = columns
    df_data = df_data.dropna(subset=['Time'])

    def parse_time_str(t_str):
        match = re.match(r'^(\d+)Time(\d+)Minutes$', str(t_str).strip())
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return pd.Timedelta(hours=hour, minutes=minute), f"{hour:02d}:{minute:02d}"
        return pd.NaT, None

    time_deltas, time_labels = [], []
    for t in df_data['Time']:
        td, tl = parse_time_str(t)
        time_deltas.append(td)
        time_labels.append(tl)

    df_data['TimeOffset'] = time_deltas
    df_data['TimeLabel'] = time_labels
    df_data = df_data.dropna(subset=['TimeOffset']).sort_values('TimeOffset')
    df_data.set_index('TimeOffset', inplace=True)

    # =========================================================================
    # EXECUTIVE HEADER SECTION
    # =========================================================================
    head_col1, head_col2 = st.columns([0.15, 0.85])
    with head_col1:
        st.markdown("### ⚡ MGEN")
        st.markdown("**MGEN Renewables**")
    with head_col2:
        st.markdown("## Bulacan Solar — Daily Inverter Screening Report")
        st.markdown(f"**Operating Date:** {report_date_str} | **Performance & Analytics Engineering**")
    
    st.markdown("---")

    # 3. Extract Active Power, Irradiance & Daylight Filter
    p_cols = [c for c in df_data.columns if 'P(kW)' in c]
    power_kw = df_data[p_cols].apply(pd.to_numeric, errors='coerce')
    power_kw.columns = [c.split('_')[0] for c in p_cols]

    # Extract Irradiance (W/m2)
    irr_cols = [c for c in df_data.columns if 'Irradiance' in c or 'W/m2' in c]
    if irr_cols:
        irradiance = pd.to_numeric(df_data[irr_cols[0]], errors='coerce').fillna(0)
    else:
        irradiance = pd.Series(0, index=df_data.index)

    hours_numeric = [td.total_seconds() / 3600.0 for td in power_kw.index]
    daylight_mask = [(6.0 <= h <= 18.0) for h in hours_numeric]

    power_kw_day = power_kw.loc[daylight_mask]
    irradiance_day = irradiance.loc[daylight_mask]
    time_labels_day = [df_data.loc[td, 'TimeLabel'] for td in power_kw_day.index]

    # 4. Curtailment & Robust Z-Score Engine (Threshold set to -3.5)
    total_plant_mw_day = power_kw_day.sum(axis=1) / 1000.0
    curtailment_active_mask = (total_plant_mw_day >= 54.5).to_numpy()

    normalized_power = pd.DataFrame(index=power_kw_day.index)
    for inv in power_kw_day.columns:
        dc_cap = DC_CAPACITIES.get(inv, 4.4505)
        normalized_power[inv] = (power_kw_day[inv] / 1000.0) / dc_cap

    peer_median_norm = normalized_power.median(axis=1)
    residuals_norm = normalized_power.sub(peer_median_norm, axis=0)
    mad_norm = residuals_norm.abs().median(axis=1).replace(0, 1e-6).clip(lower=0.001)
    residual_median_norm = residuals_norm.median(axis=1)
    robust_z = 0.6745 * residuals_norm.sub(residual_median_norm, axis=0).div(mad_norm, axis=0)

    valid_sun_mask = (peer_median_norm > 0.02).to_numpy()
    total_sun_intervals = valid_sun_mask.sum()

    results = []
    total_negative_anomalies = 0

    for inv in normalized_power.columns:
        z_series = robust_z[inv].to_numpy()
        ratio_series = normalized_power[inv].to_numpy()
        median_series = peer_median_norm.to_numpy()
        
        dc_cap = DC_CAPACITIES.get(inv, 4.4505)
        deficit_series = np.clip(median_series - ratio_series, 0, None) * dc_cap * 1000.0
        
        # Updated anomaly threshold to -3.5
        flag_mask = (z_series < -3.5) & valid_sun_mask & (~curtailment_active_mask)
        flags_count = flag_mask.sum()
        total_negative_anomalies += flags_count
        
        pct_active = (flags_count / total_sun_intervals) * 100.0 if total_sun_intervals > 0 else 0.0
        z_unconstrained = np.where(curtailment_active_mask, 0.0, z_series)
        worst_z = z_unconstrained[valid_sun_mask].min() if valid_sun_mask.any() else 0.0
        max_deficit = deficit_series[valid_sun_mask].max() if deficit_series[valid_sun_mask].any() else 0.0
        
        if flags_count > 10 or worst_z < -10:
            severity = "HIGH PRIORITY"
            sev_rank = 1
        elif flags_count > 0 or worst_z < -3.5:
            severity = "MONITOR"
            sev_rank = 2
        else:
            severity = "NORMAL"
            sev_rank = 3
            
        results.append({
            'Inverter': inv,
            'Flags': int(flags_count),
            '% active': f"{pct_active:.1f}%",
            'Worst Z': round(worst_z, 2),
            'Max Deficit (kW)': round(max_deficit, 1),
            'Severity': severity,
            '_sev_rank': sev_rank,
            '_sort_flags': flags_count,
            '_sort_z': worst_z
        })

    res_df = pd.DataFrame(results)
    # Sorted primarily by Severity Rank so High Priority is always on top
    res_df = res_df.sort_values(by=['_sev_rank', '_sort_flags', '_sort_z'], ascending=[True, False, True]).reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = 'Rank'
    res_df = res_df.drop(columns=['_sev_rank', '_sort_flags', '_sort_z'])

    top_10_df = res_df.head(10)
    top_1_inverter = top_10_df.iloc[0]['Inverter'] if len(top_10_df) > 0 else "None"

    # =========================================================================
    # METRICS DISPLAY (3 Executive Boxes)
    # =========================================================================
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Inverters Monitored", "18 Units")
    m_col2.metric("Negative Anomalies Flagged (Z < -3.5)", f"{total_negative_anomalies} Intervals")
    m_col3.metric("Highest Priority Unit", top_1_inverter)
    
    st.markdown("---")

    # =========================================================================
    # DUAL-PANEL VISUALIZATION (Plant Power & Irradiance on Right Axis)
    # =========================================================================
    st.subheader("📈 Plant Active Power, Irradiance & Top 5 Performance Timeline")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Plant Active Power (Left Axis) & Irradiance (Right Axis)
    color_pow = '#1f77b4'
    ax1.set_ylabel('Plant Active Power ($MW_{ac}$)', color=color_pow, fontsize=10, fontweight='bold')
    line1 = ax1.plot(time_labels_day, total_plant_mw_day, color=color_pow, linewidth=2.5, label='Total Plant Active Power')
    ax1.axhline(y=55.0, color='red', linestyle='--', linewidth=1.5, label='Grid Export Limit (55 MWac Cap)')
    if curtailment_active_mask.any():
        ax1.fill_between(time_labels_day, 0, 55, where=curtailment_active_mask, color='red', alpha=0.12, label='Active Curtailment Window')
    ax1.tick_params(axis='y', labelcolor=color_pow)
    ax1.grid(True, linestyle='-', alpha=0.3)

    # Create secondary axis for Irradiance on the right side
    ax1_irr = ax1.twinx()
    color_irr = '#ff7f0e'
    ax1_irr.set_ylabel('Irradiance ($W/m^2$)', color=color_irr, fontsize=10, fontweight='bold')
    line2 = ax1_irr.plot(time_labels_day, irradiance_day, color=color_irr, linewidth=1.8, linestyle='-', alpha=0.85, label='Plane of Array Irradiance')
    ax1_irr.tick_params(axis='y', labelcolor=color_irr)
    ax1_irr.grid(False) # avoid overlapping gridlines

    # Combined Legend for Panel 1
    lines_p1 = line1 + line2 + ([ax1.get_lines()[1]] if len(ax1.get_lines()) > 1 else [])
    labels_p1 = [l.get_label() for l in lines_p1]
    ax1.legend(lines_p1, labels_p1, loc='lower right', frameon=True, fontsize=9)
    ax1.set_title('Plant-Wide Active Power & Solar Irradiance Profile', fontsize=12, fontweight='bold')

    # Panel 2: Clean Top 5 Inverter Timeline Plot
    ax2.plot(time_labels_day, peer_median_norm, color='black', linestyle='--', linewidth=2.2, label='Plant Peer Median Ratio', zorder=4)

    top_5_inverters = top_10_df.head(5)['Inverter'].tolist()
    palette = ['#d62728', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd']

    for idx, inv in enumerate(top_5_inverters):
        norm_vals = normalized_power[inv]
        z_series = robust_z[inv].to_numpy()
        flag_mask = (z_series < -3.5) & valid_sun_mask & (~curtailment_active_mask)
        
        lw = 2.6 if idx == 0 else 1.2
        alpha_val = 0.95 if idx == 0 else 0.5
        color = palette[idx]
        
        flag_count_val = top_10_df[top_10_df['Inverter'] == inv]['Flags'].values[0]
        label_str = f"Rank {idx+1}: {inv} ({flag_count_val} flags)"
        ax2.plot(time_labels_day, norm_vals, alpha=alpha_val, linewidth=lw, color=color, label=label_str, zorder=2)
        
        if flag_mask.any():
            flagged_times = [time_labels_day[i] for i in range(len(flag_mask)) if flag_mask[i]]
            flagged_vals = norm_vals[flag_mask]
            ax2.scatter(flagged_times, flagged_vals, color='#9c0006', s=50, zorder=6, marker='X', edgecolor='white', linewidth=0.5)

    ax2.set_title('Top 5 Screened Inverters Capacity-Normalized Performance Ratio & Flag Timeline (Z < -3.5)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time of Day (15-Min Intervals)', fontsize=10)
    ax2.set_ylabel('Normalized Ratio ($MW_{ac} / MW_{dc}$)', fontsize=10)

    ax2.set_xticks(ticks=range(0, len(time_labels_day), 4))
    ax2.set_xticklabels(time_labels_day[::4], rotation=45, ha='right')

    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # =========================================================================
    # TOP 10 O&M PRIORITY TABLE
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Fleet O&M Priority Table (Top 10 Ranked Inverters)")
    st.dataframe(top_10_df, use_container_width=True)

    # =========================================================================
    # OPERATIONAL RECOMMENDATIONS SOP
    # =========================================================================
    st.markdown("---")
    st.subheader("🛠️ Operational Recommendations & SOP Work-Queue")
    st.markdown("""
    * **🔴 HIGH PRIORITY (Flags > 10 or Worst Z < -10):** Immediate dispatch of O&M field team to inspect DC string fuses, inverter IGBT status, and communication card connectivity for the flagged unit.
    * **🟡 MONITOR (0 < Flags <= 10 or -10 <= Worst Z < -3.5):** Log into daily engineering watch-list. Review weather logs and string-level MPPT tracking during the next routine preventive maintenance cycle.
    * **🟢 NORMAL (Flags = 0 and Worst Z >= -3.5):** Normal healthy operation tracking plant-wide peer median.
    """)

else:
    st.info("👈 Please upload a daily SCADA report Excel file (`.xls` or `.xlsx`) in the sidebar to generate the screening dashboard.")
