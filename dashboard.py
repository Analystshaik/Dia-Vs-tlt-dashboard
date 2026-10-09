from __future__ import annotations
import json, sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Dow 30 vs TLT Multi-Timeframe Leadership', layout='wide')
cfg=json.loads(Path('config.json').read_text())

@st.cache_data(ttl=60)
def latest(table):
    con=sqlite3.connect(cfg['database'])
    try:
        return pd.read_sql_query(f"SELECT * FROM {table} WHERE asof_date=(SELECT MAX(asof_date) FROM {table})",con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()

frames={
    'Daily':latest(cfg['daily_analysis_table']),
    'Weekly':latest(cfg['weekly_analysis_table']),
    'Monthly':latest(cfg['monthly_analysis_table'])
}

st.title('Dow 30 vs TLT — Multi-Timeframe Leadership')
st.caption('Concurrent Leaders = best_lag 0. True Followers = best_lag 1 or more.')

def show_board(df,tf,follower):
    title='True Followers' if follower else 'Concurrent Leaders'
    if df.empty:
        st.warning(f'No {tf.lower()} data. Run the analyzer first.'); return
    work=df[df['best_lag'].fillna(0).astype(int)>=1].copy() if follower else df[df['best_lag'].fillna(-1).astype(int)==0].copy()
    st.markdown(f'#### {title}')
    st.caption('TLT leads by at least one bar.' if follower else 'Strongest relationship occurs in the same bar.')
    if work.empty:
        st.info('No qualifying stocks on this run.'); return
    work['abs_score']=work['relationship_score'].abs()
    work=work.sort_values(['abs_score','best_lag_corr'],ascending=[False,False])
    n=st.selectbox(f'{tf} {title} rows',[10,20,30,50,100],index=1,key=f'{tf}_{title}')
    view=work.head(n)
    fig=px.bar(view,x='ticker',y='relationship_score',hover_data=['best_lag','best_lag_corr','best_lag_beta','relationship_type','follower_alert'],title=f'{tf} — {title}')
    fig.add_hline(y=0); st.plotly_chart(fig,use_container_width=True)
    base=['ticker','relationship_score','relationship_type','best_lag','best_lag_corr','best_lag_beta','follower_alert']
    extra={'Daily':['corr_20d','corr_60d','corr_120d','rs_5d','rs_20d'],
           'Weekly':['corr_13w','corr_26w','corr_52w','rs_4w','rs_13w'],
           'Monthly':['corr_6m','corr_12m','corr_24m','rs_3m','rs_6m']}[tf]
    cols=[c for c in base+extra if c in view.columns]
    st.dataframe(view[cols],use_container_width=True,hide_index=True)

tabs=st.tabs(['Daily','Weekly','Monthly','Cross-Timeframe Followers'])
for tf,tab in zip(['Daily','Weekly','Monthly'],tabs[:3]):
    with tab:
        left,right=st.columns(2)
        with left: show_board(frames[tf],tf,False)
        with right: show_board(frames[tf],tf,True)

with tabs[3]:
    st.subheader('Cross-Timeframe True Followers')
    parts=[]
    for tf,df in frames.items():
        if df.empty: continue
        x=df[df['best_lag'].fillna(0).astype(int)>=1][['ticker','relationship_score','best_lag','best_lag_corr','follower_alert']].copy()
        p=tf.lower()
        x=x.rename(columns={'relationship_score':f'{p}_score','best_lag':f'{p}_lag','best_lag_corr':f'{p}_lag_corr','follower_alert':f'{p}_alert'})
        parts.append(x.set_index('ticker'))
    if not parts:
        st.info('No follower data available.')
    else:
        m=pd.concat(parts,axis=1).reset_index()
        sc=[c for c in m.columns if c.endswith('_score')]
        m['timeframes_following']=m[sc].notna().sum(axis=1)
        m['combined_strength']=m[sc].abs().mean(axis=1)
        m=m.sort_values(['timeframes_following','combined_strength'],ascending=[False,False])
        st.dataframe(m,use_container_width=True,hide_index=True)
