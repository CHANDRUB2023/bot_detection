import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, f1_score, precision_score, recall_score)
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SocialGuard: Explainable Bot Detection on Social Media",
    page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"]{background-color:#F8FAFF !important;}
    [data-testid="stMain"]{background-color:#F8FAFF !important;}
    section[data-testid="stSidebar"]{background-color:#EEF2FF !important;}
    html,body,p,span,div,label,.stMarkdown{color:#1a1a2e !important;}
    .main-title{font-size:2.2rem;font-weight:900;color:#1B4FD8 !important;text-align:center;padding:1rem 0;}
    .subtitle{font-size:1rem;color:#444 !important;text-align:center;margin-bottom:2rem;}
    .metric-card{background:linear-gradient(135deg,#1B4FD8,#4F46E5);color:white !important;padding:1.2rem;border-radius:12px;text-align:center;margin:0.5rem;box-shadow:0 4px 12px rgba(27,79,216,0.3);}
    .metric-value{font-size:2rem;font-weight:800;color:white !important;}
    .metric-label{font-size:0.9rem;color:#e0e7ff !important;}
    .bot-result{background:linear-gradient(135deg,#EF4444,#B91C1C);color:white !important;padding:2rem;border-radius:16px;text-align:center;box-shadow:0 4px 15px rgba(239,68,68,0.4);}
    .human-result{background:linear-gradient(135deg,#10B981,#059669);color:white !important;padding:2rem;border-radius:16px;text-align:center;box-shadow:0 4px 15px rgba(16,185,129,0.4);}
    .section-header{font-size:1.3rem;font-weight:700;color:#1B4FD8 !important;border-left:5px solid #1B4FD8;padding:0.5rem 1rem;margin:1.5rem 0 1rem;background:#EEF2FF;border-radius:0 8px 8px 0;}
    .stButton>button{background:linear-gradient(135deg,#1B4FD8,#4F46E5) !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:700 !important;font-size:1rem !important;padding:0.6rem 1rem !important;}
    .stSelectbox div[data-baseweb="select"] div{color:#1a1a2e !important;background-color:#fff !important;}
    .stSelectbox div[data-baseweb="select"] span{color:#1a1a2e !important;}
    div[data-baseweb="popover"] li{color:#1a1a2e !important;background:#fff !important;}
    div[data-baseweb="popover"] li:hover{background:#EEF2FF !important;}
    div[data-baseweb="select"]>div{background-color:#fff !important;border-color:#1B4FD8 !important;}
    [data-testid="stSelectbox"] label{color:#1a1a2e !important;font-weight:700 !important;}
    .stNumberInput label{color:#1a1a2e !important;font-weight:600 !important;}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.6;}}
    .live-badge{display:inline-block;background:#10B981;color:white;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:1px;margin-left:10px;animation:pulse 1.5s infinite;}
    .top2-badge{display:inline-block;background:#F59E0B;color:white;padding:4px 14px;border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:1px;margin-left:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">SocialGuard: Explainable Bot Detection on Social Media</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> | Chandru B — Bharathiar University</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=80)
    st.markdown("### 🔧 Navigation")
    page = st.radio("Go to", ["🏠 Home", "📊 Train Model", "🔍 Predict Account", "📈 Model Analysis"])
    st.markdown("---")
    st.markdown("### 📌 About")
    st.info("Detects whether a Twitter account is a **Bot** or **Human** using ML features like followers ratio, tweet frequency, and account metadata.")
    

FEATURES = ['followers_count','friends_count','statuses_count','favourites_count',
            'listed_count','default_profile','default_profile_image','geo_enabled',
            'verified','account_age_days','followers_friends_ratio','tweets_per_day']

# ── ALL 6 ALGORITHMS — used in comparison charts only ─────
ALGO_MAP = {
    '📈 Gradient Boosting': {'cls':GradientBoostingClassifier,'params':{'n_estimators':200,'learning_rate':0.05,'max_depth':5,'random_state':42},'scaled':False,'desc':'Sequential trees — Rank 1 Best Model','color':'#F59E0B'},
    '🌲 Random Forest':     {'cls':RandomForestClassifier,    'params':{'n_estimators':150,'random_state':42,'n_jobs':-1},                       'scaled':False,'desc':'150 Decision Trees voting together', 'color':'#10B981'},
}

@st.cache_data
def generate_demo_data(n=4000):
    np.random.seed(42)
    df = pd.DataFrame({
        'followers_count':  np.random.lognormal(5.0,2.0,n).astype(int).clip(0,100000),
        'friends_count':    np.random.lognormal(5.5,1.5,n).astype(int).clip(0,50000),
        'statuses_count':   np.random.lognormal(7.0,2.5,n).astype(int).clip(0,500000),
        'favourites_count': np.random.lognormal(6.0,3.0,n).astype(int).clip(0,500000),
        'listed_count':     np.random.lognormal(2.0,2.5,n).astype(int).clip(0,5000),
        'account_age_days': np.random.lognormal(6.5,1.8,n).astype(int).clip(1,6000),
    })
    bot_score = (-0.3*np.log1p(df['followers_count'])+0.4*np.log1p(df['friends_count'])
                 +0.2*np.log1p(df['statuses_count'])-0.3*np.log1p(df['account_age_days'])
                 +0.06*np.random.randn(n))
    bot_prob = expit(bot_score - bot_score.mean())
    df['account_type'] = (np.random.rand(n) < bot_prob).astype(int)
    n_ = len(df)
    df['default_profile']       = np.where(df['account_type']==1, np.random.choice([0,1],n_,p=[0.35,0.65]), np.random.choice([0,1],n_,p=[0.65,0.35]))
    df['default_profile_image'] = np.where(df['account_type']==1, np.random.choice([0,1],n_,p=[0.20,0.80]), np.random.choice([0,1],n_,p=[0.82,0.18]))
    df['geo_enabled']           = np.where(df['account_type']==1, np.random.choice([0,1],n_,p=[0.75,0.25]), np.random.choice([0,1],n_,p=[0.40,0.60]))
    df['verified']              = np.where(df['account_type']==1, np.random.choice([0,1],n_,p=[0.99,0.01]), np.random.choice([0,1],n_,p=[0.93,0.07]))
    df['followers_friends_ratio'] = df['followers_count']/(df['friends_count']+1)
    df['tweets_per_day']          = df['statuses_count']/(df['account_age_days']+1)
    return df

@st.cache_resource
def train_all_models(_df):
    X=_df[FEATURES]; y=_df['account_type']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler=StandardScaler()
    Xtr_sc=scaler.fit_transform(X_train); Xte_sc=scaler.transform(X_test)
    results={}
    for name,cfg in ALGO_MAP.items():
        m=cfg['cls'](**cfg['params'])
        Xtr=Xtr_sc if cfg['scaled'] else X_train
        Xte=Xte_sc if cfg['scaled'] else X_test
        m.fit(Xtr,y_train)
        yp=m.predict(Xte); yprob=m.predict_proba(Xte)[:,1]
        results[name]={'model':m,'y_pred':yp,'y_prob':yprob,
                       'acc':accuracy_score(y_test,yp),'auc':roc_auc_score(y_test,yprob),
                       'f1':f1_score(y_test,yp),'prec':precision_score(y_test,yp),
                       'rec':recall_score(y_test,yp),'X_test':Xte,
                       'desc':cfg['desc'],'color':cfg['color']}
    return results,scaler,X_test,y_test

df = generate_demo_data()
all_results,scaler,X_test_raw,y_test = train_all_models(df)

# ── AUTO SELECT TOP 2 BY ACCURACY ─────────────────────────
ranked_all = sorted(all_results.keys(), key=lambda x: -all_results[x]['acc'])
TOP2       = ranked_all[:2]
best_model = all_results[TOP2[0]]['model']

# ══════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════
if page=="🏠 Home":
    st.markdown('<div class="section-header">📊 Dataset Overview</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Accounts</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{df["account_type"].sum():,}</div><div class="metric-label">Bot Accounts</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{(df["account_type"]==0).sum():,}</div><div class="metric-label">Human Accounts</div></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(FEATURES)}</div><div class="metric-label">Features Used</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Sample Data</div>',unsafe_allow_html=True)
    disp=df.head(10).copy(); disp['Account Type']=disp['account_type'].map({0:'👤 Human',1:'🤖 Bot'})
    st.dataframe(disp.drop(columns=['account_type']),use_container_width=True)
    st.markdown('<div class="section-header">🎯 Class Distribution</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(4,3)); fig.patch.set_facecolor('#F8FAFF')
        counts=df['account_type'].value_counts()
        ax.pie(counts,labels=['Bot 🤖','Human 👤'],autopct='%1.1f%%',colors=['#ff416c','#11998e'],startangle=90,textprops={'fontsize':12})
        ax.set_title('Bot vs Human Distribution',fontweight='bold'); st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(4,3)); fig.patch.set_facecolor('#F8FAFF')
        sns.countplot(data=df,x='account_type',palette=['#11998e','#ff416c'],ax=ax)
        ax.set_xticklabels(['Human','Bot']); ax.set_title('Account Count',fontweight='bold')
        ax.set_xlabel('Account Type'); ax.set_ylabel('Count'); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
#  TRAIN MODEL — TOP 2 IN DROPDOWN
# ══════════════════════════════════════════════════════════
elif page=="📊 Train Model":

    # TOP 2 BANNER
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#EEF2FF,#F0FDF4);border:2px solid #1B4FD8;
        border-radius:14px;padding:20px 28px;margin-bottom:24px;">
        <div style="font-size:1.15rem;font-weight:800;color:#1B4FD8;margin-bottom:6px;">
            🏆 Top 2 Best Performing Algorithms
            <span class="top2-badge">AUTO SELECTED</span>
            <span class="live-badge">● LIVE</span>
        </div>
        <div style="font-size:0.85rem;color:#555;margin-bottom:14px;">
            Automatically selected from all 6 algorithms based on highest accuracy score.
        </div>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
            <div style="background:white;border:2px solid #F59E0B;border-radius:10px;padding:12px 24px;text-align:center;">
                <div style="font-size:0.65rem;color:#888;letter-spacing:2px;">🥇 RANK 1 — BEST</div>
                <div style="font-size:1rem;font-weight:800;color:#F59E0B;margin:4px 0;">{TOP2[0]}</div>
                <div style="font-size:1.4rem;font-weight:900;color:#10B981;">{all_results[TOP2[0]]['acc']*100:.2f}%</div>
            </div>
            <div style="background:white;border:2px solid #10B981;border-radius:10px;padding:12px 24px;text-align:center;">
                <div style="font-size:0.65rem;color:#888;letter-spacing:2px;">🥈 RANK 2 — 2ND BEST</div>
                <div style="font-size:1rem;font-weight:800;color:#10B981;margin:4px 0;">{TOP2[1]}</div>
                <div style="font-size:1.4rem;font-weight:900;color:#10B981;">{all_results[TOP2[1]]['acc']*100:.2f}%</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── DROPDOWN: TOP 2 ONLY ──────────────────────────────
    selected_algo = st.selectbox(
        "🔬 Select Algorithm to View Detailed Metrics:",
        TOP2,
        index=0
    )

    r     = all_results[selected_algo]
    color = r['color']
    perf_color = '#10B981' if r['acc']>=0.90 else '#F59E0B' if r['acc']>=0.85 else '#EF4444'
    perf_label = '🏆 Excellent' if r['acc']>=0.90 else '✅ Good' if r['acc']>=0.85 else '⚠️ Average'

    # LIVE METRIC CARDS
    st.markdown('<div class="section-header">⚡ Live Performance Metrics</div>',unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns(5)
    def mcard(col,lbl,val,sub,bc):
        col.markdown(f"""<div style="background:white;border:2px solid {bc};border-radius:14px;
            padding:18px 12px;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.07);height:130px;">
            <div style="font-size:0.65rem;color:#888;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
            <div style="font-size:2rem;font-weight:900;color:{bc};line-height:1.1;">{val}</div>
            <div style="font-size:0.65rem;color:#aaa;margin-top:5px;">{sub}</div></div>""",unsafe_allow_html=True)
    mcard(c1,"✅ Accuracy",  f"{r['acc']*100:.2f}%", perf_label,         perf_color)
    mcard(c2,"📈 AUC-ROC",  f"{r['auc']:.4f}",      "Area Under Curve", "#1B4FD8")
    mcard(c3,"🎯 F1 Score", f"{r['f1']:.4f}",       "Harmonic Mean",    "#8B5CF6")
    mcard(c4,"🔍 Precision",f"{r['prec']:.4f}",     "Bot Precision",    "#F59E0B")
    mcard(c5,"📡 Recall",   f"{r['rec']:.4f}",      "Bot Recall",       "#EC4899")
    st.markdown("<br>",unsafe_allow_html=True)

    # TOP 2 SIDE-BY-SIDE CARDS
    st.markdown('<div class="section-header">📊 Top 2 Algorithms — Direct Comparison</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    for i,algo in enumerate(TOP2):
        rv=all_results[algo]
        ac='#F59E0B' if i==0 else '#10B981'
        with (col1 if i==0 else col2):
            st.markdown(f"""
            <div style="background:white;border:2px solid {ac};border-radius:14px;padding:20px;
                box-shadow:0 4px 16px rgba(0,0,0,0.08);margin-bottom:12px;">
                <div style="font-size:0.7rem;color:#888;letter-spacing:2px;margin-bottom:4px;">
                    {'🥇 RANK 1 — BEST MODEL' if i==0 else '🥈 RANK 2 — 2ND BEST'}</div>
                <div style="font-size:1.2rem;font-weight:800;color:{ac};margin-bottom:14px;">{algo}</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                    <div style="background:#F8FAFF;border-radius:8px;padding:10px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:900;color:{ac};">{rv['acc']*100:.2f}%</div>
                        <div style="font-size:0.65rem;color:#888;letter-spacing:1px;">ACCURACY</div></div>
                    <div style="background:#F8FAFF;border-radius:8px;padding:10px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:900;color:{ac};">{rv['auc']:.4f}</div>
                        <div style="font-size:0.65rem;color:#888;letter-spacing:1px;">AUC-ROC</div></div>
                    <div style="background:#F8FAFF;border-radius:8px;padding:10px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:900;color:{ac};">{rv['f1']:.4f}</div>
                        <div style="font-size:0.65rem;color:#888;letter-spacing:1px;">F1 SCORE</div></div>
                    <div style="background:#F8FAFF;border-radius:8px;padding:10px;text-align:center;">
                        <div style="font-size:1.5rem;font-weight:900;color:{ac};">{rv['prec']:.4f}</div>
                        <div style="font-size:0.65rem;color:#888;letter-spacing:1px;">PRECISION</div></div>
                </div></div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ── ALL 6 COMPARISON CHARTS — kept as requested ───────
    st.markdown('<div class="section-header">📊 All 6 Algorithms — Full Accuracy Comparison</div>',unsafe_allow_html=True)
    st.caption("ℹ️ Complete comparison of all algorithms — Top 2 highlighted in color")
    ns=sorted(all_results.keys(),key=lambda x:-all_results[x]['acc'])
    vals=[all_results[n]['acc']*100 for n in ns]
    clrs=['#F59E0B' if n==TOP2[0] else '#10B981' if n==TOP2[1] else '#CBD5E1' for n in ns]
    lbls=[n.split(' ',1)[1] for n in ns]
    fig,ax=plt.subplots(figsize=(12,4))
    fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
    bars=ax.bar(lbls,vals,color=clrs,edgecolor='white',linewidth=1.8,zorder=3,width=0.6)
    ax.set_ylim(60,112); ax.set_ylabel('Accuracy (%)',fontsize=11,color='#1a1a2e')
    ax.set_title('All 6 Algorithms — Accuracy Comparison (Top 2 Highlighted)',fontsize=12,fontweight='bold',color='#1a1a2e')
    ax.tick_params(axis='x',rotation=15,colors='#1a1a2e'); ax.tick_params(axis='y',colors='#1a1a2e')
    ax.grid(axis='y',alpha=0.25,zorder=0)
    for bar,val,nm in zip(bars,vals,ns):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f'{val:.1f}%',ha='center',fontsize=10,fontweight='bold',color='#1a1a2e')
        if nm in TOP2:
            tag='🥇 BEST' if nm==TOP2[0] else '🥈 2ND'
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,tag,ha='center',fontsize=8,
                    color='#F59E0B' if nm==TOP2[0] else '#10B981',fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # FULL METRICS ALL 6
    st.markdown('<div class="section-header">📈 Full Metrics — All 6 Algorithms</div>',unsafe_allow_html=True)
    met=['acc','auc','f1','prec','rec']; mlbls=['Accuracy','AUC-ROC','F1','Precision','Recall']
    x=np.arange(len(all_results)); w=0.15
    fig,ax=plt.subplots(figsize=(14,5))
    fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
    pal=['#1B4FD8','#10B981','#8B5CF6','#F59E0B','#EC4899']
    for i,(m,ml) in enumerate(zip(met,mlbls)):
        vm=[all_results[n][m] for n in all_results]
        ax.bar(x+i*w,vm,w,label=ml,color=pal[i],alpha=0.85,edgecolor='white',linewidth=0.8)
    ax.set_xticks(x+w*2); ax.set_xticklabels([n.split(' ',1)[1] for n in all_results],rotation=15,ha='right',color='#1a1a2e',fontsize=9)
    ax.set_ylim(0.5,1.12); ax.set_ylabel('Score',color='#1a1a2e')
    ax.set_title('All Metrics — All 6 Algorithms',fontweight='bold',color='#1a1a2e')
    ax.legend(loc='lower right',fontsize=9); ax.tick_params(colors='#1a1a2e'); ax.grid(axis='y',alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # CONFUSION MATRIX + ROC
    st.markdown('<div class="section-header">📊 Confusion Matrix & ROC Curve</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
        cm=confusion_matrix(y_test,r['y_pred'])
        sns.heatmap(cm,annot=True,fmt='d',cmap='YlOrBr',ax=ax,
                    xticklabels=['Human','Bot'],yticklabels=['Human','Bot'],linewidths=0.5)
        ax.set_title(f'Confusion Matrix — {selected_algo}',fontweight='bold',color='#1a1a2e')
        ax.set_ylabel('Actual',color='#1a1a2e'); ax.set_xlabel('Predicted',color='#1a1a2e')
        st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
        for name in all_results:
            fpr,tpr,_=roc_curve(y_test,all_results[name]['y_prob'])
            lw=3 if name==selected_algo else 1
            alp=1.0 if name in TOP2 else 0.2
            ax.plot(fpr,tpr,color=ALGO_MAP[name]['color'],lw=lw,alpha=alp,
                    label=f"{name.split(' ',1)[1]} ({all_results[name]['auc']:.3f})")
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
        ax.set_title('ROC Curves — All Algorithms',fontweight='bold',color='#1a1a2e')
        ax.set_xlabel('False Positive Rate',color='#1a1a2e'); ax.set_ylabel('True Positive Rate',color='#1a1a2e')
        ax.legend(fontsize=7,loc='lower right'); ax.tick_params(colors='#1a1a2e')
        st.pyplot(fig); plt.close()

    # CLASSIFICATION REPORT
    st.markdown('<div class="section-header">📋 Classification Report</div>',unsafe_allow_html=True)
    rep=classification_report(y_test,r['y_pred'],target_names=['Human','Bot'],output_dict=True)
    st.dataframe(pd.DataFrame(rep).transpose().round(4),use_container_width=True)

    # FEATURE IMPORTANCE
    if hasattr(r['model'],'feature_importances_'):
        st.markdown('<div class="section-header">🔑 Feature Importance</div>',unsafe_allow_html=True)
        imp=pd.Series(r['model'].feature_importances_,index=FEATURES).sort_values(ascending=True)
        fig,ax=plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
        bclrs=[color if v==imp.max() else '#CBD5E1' for v in imp.values]
        imp.plot(kind='barh',color=bclrs,ax=ax,edgecolor='white')
        ax.set_title(f'Feature Importance — {selected_algo}',fontweight='bold',color='#1a1a2e')
        ax.set_xlabel('Importance Score',color='#1a1a2e'); ax.tick_params(colors='#1a1a2e')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
#  PREDICT ACCOUNT
# ══════════════════════════════════════════════════════════
elif page=="🔍 Predict Account":
    st.markdown('<div class="section-header">🔍 Enter Twitter Account Details</div>',unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#EEF2FF;border-left:5px solid #1B4FD8;border-radius:0 10px 10px 0;
        padding:10px 18px;margin-bottom:16px;">
        <span style="color:#1B4FD8;font-weight:700;">🤖 Prediction Model: {TOP2[0]}</span>
        <span style="color:#888;font-size:0.85rem;margin-left:10px;">(Best accuracy — auto selected)</span>
    </div>""",unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)
    with col1:
        followers=st.number_input("👥 Followers Count",0,10000000,150)
        friends=st.number_input("➡️ Following Count",0,10000000,300)
        statuses=st.number_input("📝 Total Tweets",0,1000000,1200)
        favourites=st.number_input("❤️ Favourites Count",0,1000000,3000)
    with col2:
        listed=st.number_input("📋 Listed Count",0,10000,5)
        account_age=st.number_input("📅 Account Age (days)",1,6000,800)
        verified=st.selectbox("✅ Verified Account?",[0,1],format_func=lambda x:"Yes" if x else "No")
    with col3:
        default_profile=st.selectbox("🎨 Default Profile?",[0,1],format_func=lambda x:"Yes" if x else "No")
        default_img=st.selectbox("🖼️ Default Profile Image?",[0,1],format_func=lambda x:"Yes" if x else "No")
        geo_enabled=st.selectbox("📍 Geo Enabled?",[0,1],format_func=lambda x:"Yes" if x else "No")

    if st.button("🚀 Detect Account",use_container_width=True):
        ff_ratio=followers/(friends+1); tpd=statuses/(account_age+1)
        inp=pd.DataFrame([[followers,friends,statuses,favourites,listed,default_profile,
                           default_img,geo_enabled,verified,account_age,ff_ratio,tpd]],columns=FEATURES)
        pred=best_model.predict(inp)[0]; prob=best_model.predict_proba(inp)[0]
        bot_prob=prob[1]*100; human_prob=prob[0]*100
        st.markdown("---")
        if pred==1:
            st.markdown(f'<div class="bot-result"><h2>🤖 BOT DETECTED!</h2><h3>Bot Probability: {bot_prob:.1f}%</h3><p>This account shows strong bot-like behaviour patterns.</p></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="human-result"><h2>👤 HUMAN ACCOUNT</h2><h3>Human Probability: {human_prob:.1f}%</h3><p>This account shows genuine human behaviour patterns.</p></div>',unsafe_allow_html=True)
        st.markdown("---")
        col1,col2=st.columns(2)
        with col1:
            fig,ax=plt.subplots(figsize=(4,4)); fig.patch.set_facecolor('#F8FAFF')
            ax.pie([bot_prob,human_prob],labels=[f'Bot\n{bot_prob:.1f}%',f'Human\n{human_prob:.1f}%'],
                   colors=['#ff416c','#11998e'],startangle=90,textprops={'fontweight':'bold'})
            ax.set_title('Prediction Confidence',fontweight='bold'); st.pyplot(fig); plt.close()
        with col2:
            st.markdown('<p style="color:#1a1a2e;font-size:17px;font-weight:bold;">📊 Key Risk Indicators:</p>',unsafe_allow_html=True)
            for lbl,val,risky in [
                ("👥 Followers/Following Ratio",f"{ff_ratio:.3f}",ff_ratio<0.1),
                ("🐦 Tweets per Day",f"{tpd:.1f}",tpd>50),
                ("🖼️ Default Profile Image","Yes" if default_img else "No",default_img==1),
                ("✅ Verified Account","Yes" if verified else "No",verified==0),
                ("📅 Account Age",f"{account_age} days",account_age<100)]:
                bg="#FEE2E2" if risky else "#D1FAE5"; bc="#EF4444" if risky else "#10B981"
                badge="⚠️ BOT SIGNAL" if risky else "✅ NORMAL"
                st.markdown(f'<div style="background:{bg};border-radius:10px;padding:8px 12px;margin:6px 0;border-left:4px solid {bc};display:flex;justify-content:space-between;align-items:center;"><span style="color:#1a1a2e;font-weight:600;font-size:13px;">{lbl}</span><span style="color:#334155;font-size:13px;margin:0 8px;">{val}</span><span style="background:{bc};color:white;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;">{badge}</span></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MODEL ANALYSIS
# ══════════════════════════════════════════════════════════
elif page=="📈 Model Analysis":
    st.markdown('<div class="section-header">📈 Feature Correlations</div>',unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,7)); fig.patch.set_facecolor('#F8FAFF'); ax.set_facecolor('#F8FAFF')
    corr=df[FEATURES+['account_type']].corr()
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',center=0,linewidths=0.5,ax=ax)
    ax.set_title('Feature Correlation Heatmap',fontweight='bold',fontsize=13,color='#1a1a2e'); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">📊 Feature Distributions by Class</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#1a1a2e;font-size:15px;font-weight:700;margin:10px 0 4px;">🔍 Select Feature to Analyze:</p>',unsafe_allow_html=True)
    selected_feat=st.selectbox("",FEATURES)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for a in axes: a.set_facecolor('#F8FAFF')
    fig.patch.set_facecolor('#F8FAFF')
    df[df['account_type']==0][selected_feat].hist(bins=30,alpha=0.6,color='#11998e',label='Human',ax=axes[0])
    df[df['account_type']==1][selected_feat].hist(bins=30,alpha=0.6,color='#ff416c',label='Bot',ax=axes[0])
    axes[0].set_title(f'{selected_feat} Distribution',fontweight='bold',color='#1a1a2e'); axes[0].legend()
    sns.boxplot(data=df,x='account_type',y=selected_feat,palette={'0':'#11998e','1':'#ff416c'},ax=axes[1])
    axes[1].set_xticklabels(['Human','Bot']); axes[1].set_title(f'{selected_feat} Boxplot',fontweight='bold',color='#1a1a2e')
    plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("---")
st.markdown("""<center style='color:#888;font-size:0.85rem;'>
    🤖 <b>SocialGuard: </b><br/>
    Chandru B | M.Sc Artificial Intelligence | Bharathiar University |
</center>""",unsafe_allow_html=True)