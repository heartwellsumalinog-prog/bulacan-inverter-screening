        generating = flags & p[inv].gt(0)
        count, valid = int(flags.sum()), int(eligible.sum())
        worst = z.loc[eligible, inv].min()
        if count > HIGH_FLAGS or worst < EXTREME_Z:
            severity, rank = 'HIGH PRIORITY', 1
        elif count > 0:
            severity, rank = 'MONITOR', 2
        elif valid < min_daily:
            severity, rank = 'INSUFFICIENT DATA', 4
        else:
            severity, rank = 'NORMAL', 3
        rows.append({'Inverter':inv, 'Flags':count, 'Zero-output flags':int(zero.sum()), 'Nonzero-output flags':int(generating.sum()), 'Eligible intervals':valid, 'Anomaly Rate (%)':100*count/valid if valid else np.nan, 'Worst Z':worst, 'Max Deficit (kW)':deficit.loc[eligible,inv].max(), 'Flagged Energy Gap (kWh)':deficit.loc[flags,inv].sum()*0.25 if valid else np.nan, 'Severity':severity, 'Coverage': 'Limited' if valid < min_daily else 'Meets minimum', '_rank':rank})
        details.append(pd.DataFrame({'Time':p.index, 'Inverter':inv,'AC Power (kW)':p[inv].values,'Normalized Ratio':norm[inv].values,'Fleet Median':median.values,'MAD':mad.values,'Robust Z':z[inv].values,'Eligible':eligible.values,'Flag':flags.values,'Zero-output flag':zero.values,'Nonzero-output flag':generating.values,'Peer Deficit (kW)':deficit[inv].values}))
    ranked = pd.DataFrame(rows).sort_values(['_rank','Flags','Worst Z','Inverter'], ascending=[True,False,True,True]).drop(columns='_rank').reset_index(drop=True)
    ranked.index = ranked.index+1
    ranked.index.name='Rank'
    return ranked, pd.concat(details, ignore_index=True), p, norm, median, total_mw, high

def csv(df, index=False):
    return df.to_csv(index=index).encode('utf-8-sig')

def main():
    st.set_page_config(page_title='Bulacan Solar — Fleet Inverter Screening', page_icon='☀️', layout='wide')
    left,right=st.columns([1,4])
    with left:
        if Path('mgen_logo.png').exists(): st.image('mgen_logo.png',width=180)
        else: st.markdown('**MGEN Renewables**')
    with right:
        st.title('Bulacan Solar — Fleet Inverter Screening Platform')
    st.sidebar.header('Fleet data ingestion')
    files=st.sidebar.file_uploader('Daily SCADA reports (.xls / .xlsx)',type=['xls','xlsx'],accept_multiple_files=True)
    st.sidebar.caption(f'Flag: Z < {FLAG_Z}; High Priority: > {HIGH_FLAGS} flags or Z < {EXTREME_Z}.')
    st.sidebar.caption(f'Proposed minimum daily coverage: {MIN_DAILY_INTERVALS} eligible intervals; minimum other peers: {MIN_VALID_PEERS}.')
    if not files:
        st.info('Upload daily SCADA reports to begin.'); return
    records={}
    for file in files:
        try:
            date=report_date(file.name)
            if date in records: raise ValueError('Duplicate operating date; first successful report retained.')
            p,irr=read_report(file)
            rec=screen(p)
            records[date]=(rec,irr)
            missing=p.columns[p.isna().all()].tolist()
            if missing: st.warning(f'{date}: missing inverter data: {", ".join(missing)}. Fleet total eligibility may be unavailable.')
        except Exception as e:
            st.error(f'{file.name}: {e}')
    if not records: st.stop()
    dates=sorted(records)
    st.caption(f'{len(dates)} daily reports loaded | {dates[0]} to {dates[-1]}')
    daily,multi=st.tabs(['Daily Screening Report','Weekly / Multi-Day Fleet Summary'])
    with daily:
        date=st.selectbox('Operating date',dates,index=len(dates)-1)
        (ranked,detail,p,norm,median,total,high),irr=records[date]
        priority=ranked.loc[ranked.Severity.isin(['HIGH PRIORITY','MONITOR'])]
        a,b,c,d=st.columns(4)
        a.metric('Inverters screened',f'{int(ranked["Eligible intervals"].gt(0).sum())}/18')
        b.metric('Anomaly flags',int(ranked.Flags.sum()))
        c.metric('Nonzero-output flags',int(ranked['Nonzero-output flags'].sum()))
        d.metric('Top investigation priority',priority.iloc[0].Inverter if not priority.empty else 'None')
        st.caption('Nonzero-output flags indicate measured generation, not independently confirmed online status or post-restoration operation.')
        x=p.index.total_seconds()/3600
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)
        ax1.plot(x,total,label='Sum of inverter AC power (MW)')
        ax1.axhline(55,color='red',ls='--',label='55 MW reference')
        ax1.fill_between(x,0,55,where=high.to_numpy(),alpha=.12,color='red',label='Excluded high-output intervals')
        ax1.set_ylabel('MW'); ax1.legend(loc='upper left'); ax1.grid(alpha=.3)
        if irr.notna().any():
            twin=ax1.twinx(); twin.plot(x,irr.reindex(p.index),color='orange',alpha=.7); twin.set_ylabel('Irradiance (W/m²)')
        ax2.plot(x,median,'k--',label='Fleet median')
        selected=st.multiselect('Inverters to plot',list(DC_CAPACITIES),default=ranked.head(5).Inverter.tolist())
        for inv in selected:
            ax2.plot(x,norm[inv],label=inv)
            mask=detail.loc[detail.Inverter.eq(inv),'Flag'].to_numpy()
            ax2.scatter(x[mask],norm[inv].to_numpy()[mask],marker='x',color='darkred',s=25)
        ax2.set_ylabel('MWac / MWdc'); ax2.set_xlabel('Hour of day'); ax2.set_xticks(range(6,19));ax2.grid(alpha=.3);ax2.legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.subheader('Fleet O&M priority table')
        st.dataframe(ranked.round(2),use_container_width=True)
        st.caption('Energy gap is estimated from flagged 15-minute samples only; not verified recoverable energy. Limited coverage does not suppress observed anomalies.')
        st.download_button('Download daily rankings',csv(ranked,True),f'{date}_rankings.csv','text/csv')
        detail['Time']=detail.Time.map(lambda t:f'{int(t.total_seconds()//3600):02d}:{int(t.total_seconds()//60%60):02d}')
        st.subheader('Flagged timestamps')
        st.dataframe(detail.loc[detail.Flag].round(4),use_container_width=True)
        st.download_button('Download interval audit data',csv(detail),f'{date}_intervals.csv','text/csv')
    with multi:
        frames=[records[d][0][0].assign(Date=d) for d in dates]
        all_days=pd.concat(frames,ignore_index=True)
        summary=all_days.groupby('Inverter').agg(**{'Days Flagged':('Flags',lambda x:int(x.gt(0).sum())), 'Days Screened':('Eligible intervals',lambda x:int(x.gt(0).sum())), 'Days Nonzero-output Flagged':('Nonzero-output flags',lambda x:int(x.gt(0).sum())), 'Total Anomaly Intervals':('Flags','sum'), 'Zero-output Flags':('Zero-output flags','sum'), 'Nonzero-output Flags':('Nonzero-output flags','sum'), 'Eligible Intervals':('Eligible intervals','sum'), 'Worst Z':('Worst Z','min'), 'Flagged Energy Gap (kWh)':('Flagged Energy Gap (kWh)',lambda x:x.sum(min_count=1))}).reset_index()
        summary['Anomaly Rate (%)']=100*summary['Total Anomaly Intervals']/summary['Eligible Intervals'].replace(0,np.nan)
        summary=summary.sort_values(['Total Anomaly Intervals','Days Flagged','Inverter'],ascending=[False,False,True]).reset_index(drop=True)
        summary.index+=1;summary.index.name='Rank'
        st.subheader('Cumulative fleet anomaly ranking')
        st.bar_chart(summary.set_index('Inverter')[['Zero-output Flags','Nonzero-output Flags']],horizontal=True,stack=True)
        st.dataframe(summary.round(2),use_container_width=True)
        st.download_button('Download multi-day rankings',csv(summary,True),'fleet_summary.csv','text/csv')
        st.subheader('Daily nonzero-output anomaly history')
        history=all_days.pivot(index='Inverter',columns='Date',values='Nonzero-output flags')
        coverage=all_days.pivot(index='Inverter',columns='Date',values='Eligible intervals')
        st.dataframe(history.where(coverage.gt(0)),use_container_width=True)
        st.caption('Ranking covers uploaded dates only, which may not be consecutive. Blank history cells indicate no eligible observations.')

if __name__=='__main__':
    main()
