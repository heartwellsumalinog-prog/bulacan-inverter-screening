import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Bulacan Solar - Fleet Inverter Screening Platform",
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

def parse_time_str(t_str):
    match = re.match(r'^(\d+)Time(\d+)Minutes$', str(t_str).strip())
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return pd.Timedelta(hours=hour, minutes=minute), f"{hour:02d}:{minute:02d}"
    return pd.NaT, None

# =========================================================================
# SIDEBAR MULTI-FILE UPLOAD
# =========================================================================
st.sidebar.header("📁 Fleet Data Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Daily SCADA Reports (.xls / .xlsx)", 
    type=["xls", "xlsx"], 
    accept_multiple_files=True
)

if uploaded_files:
    daily_records = {}

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        date_match = re.search(r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})', filename, re.IGNORECASE)
        if date_match:
            report_date_str = date_match.group(1)
        else:
            report_date_str = filename.replace('.xls', '').replace('.xlsx', '')

        try:
            df_raw = pd.read_excel(uploaded_file, sheet_name="Inverter Generation Report", header=None)
            df_gen_raw = pd.read_excel(uploaded_file, sheet_name="Generation Report", header=None)
        except Exception as e:
            st.sidebar.error(f"❌ Failed to parse file: `{filename}`. Error: {str(e)}")
            continue

        try:
            inverter_names_row = df_raw.iloc[1].ffill()
            parameters_row = df_raw.iloc[2]
            columns = ['Time']
            for col_idx in range(1, df_raw.shape[1]):
                columns.append(f"{inverter_names_row.iloc[col_idx]}_{parameters_row.iloc[col_idx]}")

            df_data = df_raw.iloc[3:].copy()
            df_data.columns = columns
            df_data = df_data.dropna(subset=['Time'])

            time_deltas, time_labels = [], []
            for t in df_data['Time']:
                td, tl = parse_time_str(t)
                time_deltas.append(td)
                time_labels.append(tl)

            df_data['TimeOffset'] = time_deltas
            df_data['TimeLabel'] = time_labels
            df_data = df_data.dropna(subset=['TimeOffset']).sort_values('TimeOffset')
            df_data.set_index('TimeOffset', inplace=True)

            df_gen = df_gen_raw.iloc[6:].copy()
            def parse_t(t_str):
                match = re.match(r'^(\d+)Time(\d+)Minutes$', str(t_str).strip())
                if match:
                    return pd.Timedelta(hours=int(match.group(1)), minutes=int(match.group(2)))
                return pd.NaT

            df_gen['TimeOffset'] = df_gen[0].apply(parse_t)
            df_gen['Irradiance'] = pd.to_numeric(df_gen[17], errors='coerce').fillna(0)
            df_gen = df_gen.dropna(subset=['TimeOffset']).set_index('TimeOffset')

            p_cols = [c for c in df_data.columns if 'P(kW)' in c]
            power_kw = df_data[p_cols].apply(pd.to_numeric, errors='coerce')
            power_kw.columns = [c.split('_')[0] for c in p_cols]

            hours_numeric = [td.total_seconds() / 3600.0 for td in power_kw.index]
            daylight_mask = [(6.0 <= h <= 18.0) for h in hours_numeric]

            power_kw_day = power_kw.loc[daylight_mask]
            time_labels_day = [df_data.loc[td, 'TimeLabel'] for td in power_kw_day.index]
            irradiance_day = df_gen['Irradiance'].reindex(power_kw_day.index).fillna(0)

            total_plant_mw_day = power_kw_day.sum(axis=1) / 1000.0
            curtailment_active_mask = (total_plant_mw_day >= 54.5).to_numpy()

            normalized_power = pd.DataFrame(index=power_kw_day.index)
            for inv in power_kw_day.columns:
                dc_cap = DC_CAPACITIES.get(inv, 4.4505)
                # Preserves distinction between true zero and missing measurements
                normalized_power[inv] = (power_kw_day[inv] / 1000.0) / dc_cap

            peer_median_norm = normalized_power.median(axis=1)
            residuals_norm = normalized_power.sub(peer_median_norm, axis=0)
            mad_norm = residuals_norm.abs().median(axis=1).replace(0, 1e-6).clip(lower=0.001)
            residual_median_norm = residuals_norm.median(axis=1)
            robust_z = 0.6745 * residuals_norm.sub(residual_median_norm, axis=0).div(mad_norm, axis=0)

            valid_sun_mask = (peer_median_norm > 0.02).to_numpy()
            eligible_intervals_mask = valid_sun_mask & (~curtailment_active_mask)

            results = []
            total_negative_anomalies = 0
            inv_flags_dict = {}

            for inv in normalized_power.columns:
                z_series = robust_z[inv].to_numpy()
                ratio_series = normalized_power[inv].to_numpy()
                median_series = peer_median_norm.to_numpy()
                
                dc_cap = DC_CAPACITIES.get(inv, 4.4505)
                deficit_series = np.clip(median_series - ratio_series, 0, None) * dc_cap * 1000.0
                
                # Strict per-inverter finite check for available measurements
                inv_eligible_mask = (
                    eligible_intervals_mask
                    & np.isfinite(z_series)
                    & np.isfinite(deficit_series)
                )
                inv_eligible_count = int(inv_eligible_mask.sum())

                if inv_eligible_count == 0:
                    flags_count = 0
                    pct_active = np.nan
                    worst_z = np.nan
                    max_deficit = np.nan
                    severity = "NO DATA"
                    sev_rank = 4

                else:
                    flag_mask = (z_series < -3.5) & inv_eligible_mask
                    flags_count = flag_mask.sum()
                    total_negative_anomalies += flags_count

                    pct_active = (flags_count / inv_eligible_count) * 100.0

                    worst_z = z_series[inv_eligible_mask].min()
                    max_deficit = deficit_series[inv_eligible_mask].max()

                    if flags_count > 10 or worst_z < -10:
                        severity = "HIGH PRIORITY"
                        sev_rank = 1

                    elif flags_count > 0 or worst_z < -3.5:
                        severity = "MONITOR"
                        sev_rank = 2

                    else:
                        severity = "NORMAL"
                        sev_rank = 3
                
                inv_flags_dict[inv] = flags_count
                    
                results.append({
                    'Inverter': inv,
                    'Flags': int(flags_count),
                    'Anomaly Rate (%)': f"{pct_active:.1f}%" if not np.isnan(pct_active) else "N/A",
                    'Worst Z': round(worst_z, 2) if not np.isnan(worst_z) else "N/A",
                    'Max Deficit (kW)': round(max_deficit, 1) if not np.isnan(max_deficit) else "N/A",
                    'Severity': severity,
                    '_sev_rank': sev_rank,
                    '_sort_flags': flags_count,
                    '_sort_z': worst_z if not np.isnan(worst_z) else 0.0
                })

            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values(by=['_sev_rank', '_sort_flags', '_sort_z'], ascending=[True, False, True]).reset_index(drop=True)
            res_df.index += 1
            res_df.index.name = 'Rank'
            res_df = res_df.drop(columns=['_sev_rank', '_sort_flags', '_sort_z'])

            daily_records[report_date_str] = {
                'priority_df': res_df,
                'total_anomalies': total_negative_anomalies,
                'inv_flags': inv_flags_dict,
                'power_df': power_kw_day,
                'total_plant_mw': total_plant_mw_day,
                'time_labels': time_labels_day,
                'irradiance': irradiance_day,
                'peer_median': peer_median_norm,
                'normalized_power': normalized_power,
                'robust_z': robust_z,
                'valid_sun': valid_sun_mask,
                'curtailment': curtailment_active_mask,
                'eligible_mask': eligible_intervals_mask,
                'active_inverters_count': len(normalized_power.columns)
            }
        except Exception as e:
            st.sidebar.error(f"⚠️ Error processing columns/data in: `{filename}`. Details: {str(e)}")
            continue

    if daily_records:
        # =========================================================================
        # HEADER & VIEW SELECTOR
        # =========================================================================
        head_col1, head_col2 = st.columns([0.22, 0.78])
        with head_col1:
            logo_path = "mgen_logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, width=180)
            else:
                st.markdown("### ⚡ MGEN Renewables")
                
        with head_col2:
            st.markdown("## Bulacan Solar — Fleet Inverter Screening Platform")
            st.markdown(f"**Loaded Files:** {len(daily_records)} Daily Reports")

        st.markdown("---")

        tab_daily, tab_weekly = st.tabs(["📅 Daily Screening Report", "📊 Weekly / Multi-Day Fleet Summary"])

        # -------------------------------------------------------------------------
        # TAB 1: DAILY REPORT VIEW
        # -------------------------------------------------------------------------
        with tab_daily:
            available_dates = list(daily_records.keys())
            selected_date = st.selectbox("Select Operating Date to View:", available_dates)

            if selected_date:
                rec = daily_records[selected_date]
                top_10_df = rec['priority_df'].head(10)
                top_1_inv = top_10_df.iloc[0]['Inverter'] if len(top_10_df) > 0 else "None"

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Inverters Monitored", f"{rec['active_inverters_count']} Units")
                m2.metric("Negative Anomalies Flagged", f"{rec['total_anomalies']} Intervals")
                m3.metric("Highest Priority Unit", top_1_inv)

                st.markdown("---")
                st.subheader(f"📈 Operational Profiles — {selected_date}")

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Panel 1
                line1 = ax1.plot(rec['time_labels'], rec['total_plant_mw'], color='#1f77b4', linewidth=2.5, label='Total Plant Active Power')
                line_limit = ax1.axhline(y=55.0, color='red', linestyle='--', linewidth=1.5, label='Grid Export Limit (55 MWac Cap)')
                fill_curt = None
                if rec['curtailment'].any():
                    fill_curt = ax1.fill_between(rec['time_labels'], 0, 55, where=rec['curtailment'], color='red', alpha=0.12, label='Active Curtailment')
                
                ax1.set_ylabel('Plant Power ($MW_{ac}$)', color='#1f77b4', fontweight='bold')
                ax1.grid(True, alpha=0.3)

                ax1_irr = ax1.twinx()
                line2 = ax1_irr.plot(rec['time_labels'], rec['irradiance'], color='#ff7f0e', linewidth=1.8, alpha=0.85, label='Irradiance')
                ax1_irr.set_ylabel('Irradiance ($W/m^2$)', color='#ff7f0e', fontweight='bold')
                ax1_irr.grid(False)

                lines_p1 = line1 + line2 + [line_limit] + ([fill_curt] if fill_curt is not None else [])
                labels_p1 = [l.get_label() for l in lines_p1]
                ax1.legend(lines_p1, labels_p1, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)
                ax1.set_title(f'Plant Active Power & Irradiance — {selected_date}', fontsize=12, fontweight='bold')

                # Panel 2
                ax2.plot(rec['time_labels'], rec['peer_median'], color='black', linestyle='--', linewidth=2.2, label='Peer Median Ratio', zorder=4)
                top_5_invs = top_10_df.head(5)['Inverter'].tolist()
                palette = ['#d62728', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd']

                for idx, inv in enumerate(top_5_invs):
                    norm_vals = rec['normalized_power'][inv]
                    z_series = rec['robust_z'][inv].to_numpy()
                    inv_elig = rec['eligible_mask'] & np.isfinite(z_series) & np.isfinite(norm_vals.to_numpy())
                    flag_mask = (z_series < -3.5) & inv_elig
                    flag_cnt = rec['inv_flags'][inv]
                    
                    ax2.plot(rec['time_labels'], norm_vals, linewidth=2.0 if idx==0 else 1.2, color=palette[idx], label=f"Rank {idx+1}: {inv} ({flag_cnt} flags)", zorder=2)
                    
                    if flag_mask.any():
                        flagged_labels = [rec['time_labels'][i] for i, is_flagged in enumerate(flag_mask) if is_flagged]
                        # FIXED: Using clean boolean indexing instead of .values attribute on NumPy array
                        flagged_values = norm_vals[flag_mask].values
                        ax2.scatter(flagged_labels, flagged_values, color='#9c0006', s=30, marker='X', linewidths=0.8, zorder=6)

                ax2.set_xlabel('Time of Day (15-Min Intervals)', fontweight='bold')
                ax2.set_ylabel('Normalized Ratio ($MW_{ac} / MW_{dc}$)', fontweight='bold')
                ax2.set_xticks(ticks=range(0, len(rec['time_labels']), 4))
                ax2.set_xticklabels(rec['time_labels'][::4], rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("📋 Fleet O&M Priority Table")
                st.dataframe(top_10_df, use_container_width=True)

        # -------------------------------------------------------------------------
        # TAB 2: WEEKLY / MULTI-DAY FLEET SUMMARY
        # -------------------------------------------------------------------------
        with tab_weekly:
            st.subheader("📊 Fleet Multi-Day Summary & Chronic Anomaly Ranking")
            st.markdown("This summary aggregates all uploaded daily reports to identify chronic, recurring underperforming inverters across the fleet—supporting your weekly O&M maintenance planning.")

            fleet_totals = {}
            for date_str, rec in daily_records.items():
                for inv, f_count in rec['inv_flags'].items():
                    if inv not in fleet_totals:
                        fleet_totals[inv] = {'DaysFlagged': 0, 'TotalIntervals': 0}
                    if f_count > 0:
                        fleet_totals[inv]['DaysFlagged'] += 1
                        fleet_totals[inv]['TotalIntervals'] += f_count

            summary_list = [{
                'Inverter': inv, 
                'Days Flagged': data['DaysFlagged'], 
                'Total Anomaly Intervals': data['TotalIntervals']
            } for inv, data in fleet_totals.items()]
            
            df_summary = pd.DataFrame(summary_list).sort_values(
                by=['Total Anomaly Intervals', 'Days Flagged'], 
                ascending=[False, False]
            ).reset_index(drop=True)
            
            df_summary.index += 1
            df_summary.index.name = 'Rank'

            fig_sum, ax_sum = plt.subplots(figsize=(10, 5))
            bars = ax_sum.barh(df_summary['Inverter'], df_summary['Total Anomaly Intervals'], color='#1f77b4', edgecolor='black', alpha=0.85)
            ax_sum.set_xlabel('Total Cumulative Anomaly Intervals (Z < -3.5)', fontsize=11, fontweight='bold')
            ax_sum.set_ylabel('Inverter ID', fontsize=11, fontweight='bold')
            ax_sum.set_title(f'Multi-Day Fleet Chronic Anomaly Summary ({len(daily_records)} Reports Analyzed)', fontsize=12, fontweight='bold')
            ax_sum.invert_yaxis()
            ax_sum.grid(axis='x', linestyle='--', alpha=0.5)

            for bar in bars:
                width = bar.get_width()
                ax_sum.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)} Intervals', va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

            plt.tight_layout()
            st.pyplot(fig_sum)

            st.markdown("### 📋 Cumulative Fleet Summary Table")
            st.dataframe(df_summary, use_container_width=True)
    else:
        st.warning("⚠️ No valid daily reports were successfully parsed from the uploaded files.")
else:
    st.info("👈 Please upload multiple daily SCADA report Excel files in the sidebar to activate the multi-day fleet platform and weekly summary analytics.")
