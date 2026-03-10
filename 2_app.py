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

st.markdown('<div class="main-title">SocialGuard: Explainable Bot Detection on Twitter </div>', unsafe_allow_html=True)
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

    # PIE CHART
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#F8FAFF')
        counts = df['account_type'].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=['Bot 🤖', 'Human 👤'],
            autopct='%1.1f%%',
            colors=['#ff416c', '#11998e'],
            startangle=90,
            textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title('Bot vs Human Distribution', fontweight='bold', fontsize=13, color='#1a1a2e', pad=12)
        ax.legend(wedges, ['Bot 🤖', 'Human 👤'],
                  title="Account Type", loc="lower left",
                  fontsize=9, title_fontsize=9, frameon=True, framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # BAR CHART
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#F8FAFF')
        ax.set_facecolor('#F8FAFF')
        from matplotlib.patches import Patch
        bar_colors = ['#11998e', '#ff416c']
        counts_list = [df[df['account_type']==0].shape[0], df[df['account_type']==1].shape[0]]
        bars = ax.bar(['Human 👤', 'Bot 🤖'], counts_list, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
        ax.set_title('Account Count by Type', fontweight='bold', fontsize=13, color='#1a1a2e')
        ax.set_xlabel('Account Type', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.set_ylabel('Number of Accounts', fontsize=11, color='#1a1a2e', labelpad=8)
        # xlim and ylim
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(counts_list) * 1.2)
        ax.tick_params(axis='both', colors='#1a1a2e', labelsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#aaa')
        legend_elements = [Patch(facecolor='#11998e', label='Human'), Patch(facecolor='#ff416c', label='Bot')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, framealpha=0.8)
        for bar, val in zip(bars, counts_list):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    f'{val:,}', ha='center', fontsize=11, fontweight='bold', color='#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════
#  TRAIN MODEL
# ══════════════════════════════════════════════════════════
elif page=="📊 Train Model":

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#EEF2FF,#F0FDF4);border:2px solid #1B4FD8;
        border-radius:14px;padding:20px 28px;margin-bottom:24px;">
        <div style="font-size:1.15rem;font-weight:800;color:#1B4FD8;margin-bottom:6px;">
            🏆 Top 2 Best Performing Algorithms
            <span class="top2-badge">AUTO SELECTED</span>
            <span class="live-badge">● LIVE</span>
        </div>
        <div style="font-size:0.85rem;color:#555;margin-bottom:14px;">
            Automatically selected from 2 algorithms based on highest accuracy score.
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

    selected_algo = st.selectbox("🔬 Select Algorithm to View Detailed Metrics:", TOP2, index=0)
    r     = all_results[selected_algo]
    color = r['color']
    perf_color = '#10B981' if r['acc']>=0.90 else '#F59E0B' if r['acc']>=0.85 else '#EF4444'
    perf_label = '🏆 Excellent' if r['acc']>=0.90 else '✅ Good' if r['acc']>=0.85 else '⚠️ Average'

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

    # TOP 2 COMPARISON CARDS
    st.markdown('<div class="section-header">📊 Best 2 Algorithms — Direct Comparison</div>',unsafe_allow_html=True)
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

    # ACCURACY COMPARISON CHART
    st.markdown('<div class="section-header">📊 All 2 Algorithms — Full Accuracy Comparison</div>',unsafe_allow_html=True)
    st.caption("ℹ️ Complete comparison of all algorithms — Top 2 highlighted in color")
    ns   = sorted(all_results.keys(), key=lambda x: -all_results[x]['acc'])
    vals = [all_results[n]['acc']*100 for n in ns]
    clrs = ['#F59E0B' if n==TOP2[0] else '#10B981' if n==TOP2[1] else '#CBD5E1' for n in ns]
    lbls = [n.split(' ',1)[1] for n in ns]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#F8FAFF')
    ax.set_facecolor('#F8FAFF')
    bars = ax.bar(lbls, vals, color=clrs, edgecolor='white', linewidth=1.8, zorder=3, width=0.6)
    # xlim and ylim
    ax.set_xlim(-0.6, len(ns) - 0.4)
    ax.set_ylim(60, 115)
    ax.set_ylabel('Accuracy (%)', fontsize=12, color='#1a1a2e', labelpad=10)
    ax.set_xlabel('Algorithm', fontsize=12, color='#1a1a2e', labelpad=10)
    ax.set_title('All 2 Algorithms — Accuracy Comparison (Top 2 Highlighted)',
                 fontsize=13, fontweight='bold', color='#1a1a2e', pad=14)
    ax.tick_params(axis='x', rotation=15, colors='#1a1a2e', labelsize=10)
    ax.tick_params(axis='y', colors='#1a1a2e', labelsize=10)
    ax.grid(axis='y', alpha=0.25, zorder=0, linestyle='--', color='#aaa')
    for bar, val, nm in zip(bars, vals, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        if nm in TOP2:
            tag = '🥇 BEST' if nm==TOP2[0] else '🥈 2ND'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, tag,
                    ha='center', fontsize=9, color='#F59E0B' if nm==TOP2[0] else '#10B981', fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F59E0B', label=f'Rank 1 — {TOP2[0].split(" ",1)[1]}'),
        Patch(facecolor='#10B981', label=f'Rank 2 — {TOP2[1].split(" ",1)[1]}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, framealpha=0.85)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # FULL METRICS CHART
    st.markdown('<div class="section-header">📈 Full Metrics — All 2 Algorithms</div>',unsafe_allow_html=True)
    met   = ['acc','auc','f1','prec','rec']
    mlbls = ['Accuracy','AUC-ROC','F1 Score','Precision','Recall']
    x     = np.arange(len(all_results))
    w     = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#F8FAFF')
    ax.set_facecolor('#F8FAFF')
    pal = ['#1B4FD8','#10B981','#8B5CF6','#F59E0B','#EC4899']
    for i,(m,ml) in enumerate(zip(met,mlbls)):
        vm = [all_results[n][m] for n in all_results]
        ax.bar(x+i*w, vm, w, label=ml, color=pal[i], alpha=0.85, edgecolor='white', linewidth=0.8)
    ax.set_xticks(x+w*2)
    ax.set_xticklabels([n.split(' ',1)[1] for n in all_results], rotation=15, ha='right', color='#1a1a2e', fontsize=10)
    # xlim and ylim
    ax.set_xlim(-0.3, len(all_results) - 0.2)
    ax.set_ylim(0.50, 1.12)
    ax.set_ylabel('Score', fontsize=12, color='#1a1a2e', labelpad=10)
    ax.set_xlabel('Algorithm', fontsize=12, color='#1a1a2e', labelpad=10)
    ax.set_title('All Metrics Comparison — All 2 Algorithms',
                 fontsize=13, fontweight='bold', color='#1a1a2e', pad=12)
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.85, title='Metrics', title_fontsize=9)
    ax.tick_params(colors='#1a1a2e', labelsize=10)
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='#aaa')
    ax.axhline(y=0.80, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # CONFUSION MATRIX + ROC
    st.markdown('<div class="section-header">📊 Confusion Matrix & ROC Curve</div>',unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#F8FAFF')
        ax.set_facecolor('#F8FAFF')
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', ax=ax,
                    xticklabels=['Human','Bot'], yticklabels=['Human','Bot'],
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'Confusion Matrix\n{selected_algo}',
                     fontweight='bold', fontsize=12, color='#1a1a2e', pad=10)
        ax.set_ylabel('Actual Label', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.set_xlabel('Predicted Label', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.tick_params(axis='both', colors='#1a1a2e', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#F8FAFF')
        ax.set_facecolor('#F8FAFF')
        for name in all_results:
            fpr, tpr, _ = roc_curve(y_test, all_results[name]['y_prob'])
            lw  = 3 if name == selected_algo else 1.2
            alp = 1.0 if name in TOP2 else 0.25
            ax.plot(fpr, tpr, color=ALGO_MAP[name]['color'], lw=lw, alpha=alp,
                    label=f"{name.split(' ',1)[1]} (AUC={all_results[name]['auc']:.3f})")
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4, label='Random Baseline')
        # xlim and ylim
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title('ROC Curves — All Algorithms', fontweight='bold', fontsize=12, color='#1a1a2e', pad=10)
        ax.set_xlabel('False Positive Rate', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.set_ylabel('True Positive Rate', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.tick_params(axis='both', colors='#1a1a2e', labelsize=10)
        ax.legend(fontsize=8, loc='lower right', frameon=True, framealpha=0.85, title='Algorithms')
        ax.grid(alpha=0.2, linestyle='--', color='#aaa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # CLASSIFICATION REPORT
    st.markdown('<div class="section-header">📋 Classification Report</div>',unsafe_allow_html=True)
    rep = classification_report(y_test, r['y_pred'], target_names=['Human','Bot'], output_dict=True)
    st.dataframe(pd.DataFrame(rep).transpose().round(4), use_container_width=True)

    # FEATURE IMPORTANCE
    if hasattr(r['model'],'feature_importances_'):
        st.markdown('<div class="section-header">🔑 Feature Importance</div>',unsafe_allow_html=True)
        imp = pd.Series(r['model'].feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#F8FAFF')
        ax.set_facecolor('#F8FAFF')
        bclrs = [color if v == imp.max() else '#CBD5E1' for v in imp.values]
        imp.plot(kind='barh', color=bclrs, ax=ax, edgecolor='white', linewidth=0.8)
        # xlim and ylim
        ax.set_xlim(0, imp.max() * 1.25)
        ax.set_ylim(-0.6, len(imp) - 0.4)
        ax.set_title(f'Feature Importance — {selected_algo}',
                     fontweight='bold', fontsize=13, color='#1a1a2e', pad=12)
        ax.set_xlabel('Importance Score', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.set_ylabel('Feature Name', fontsize=11, color='#1a1a2e', labelpad=8)
        ax.tick_params(axis='both', colors='#1a1a2e', labelsize=10)
        ax.grid(axis='x', alpha=0.25, linestyle='--', color='#aaa')
        for i, (val, feat) in enumerate(zip(imp.values, imp.index)):
            ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8, color='#1a1a2e')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color,     label='Top Feature'),
            Patch(facecolor='#CBD5E1', label='Other Features'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True, framealpha=0.85)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

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
        followers  = st.number_input("👥 Followers Count",   0, 10000000, 150)
        friends    = st.number_input("➡️ Following Count",   0, 10000000, 300)
        statuses   = st.number_input("📝 Total Tweets",      0, 1000000,  1200)
        favourites = st.number_input("❤️ Favourites Count",  0, 1000000,  3000)
    with col2:
        listed      = st.number_input("📋 Listed Count",         0, 10000, 5)
        account_age = st.number_input("📅 Account Age (days)",   1, 6000,  800)
        verified    = st.selectbox("✅ Verified Account?",  [0,1], format_func=lambda x:"Yes" if x else "No")
    with col3:
        default_profile = st.selectbox("🎨 Default Profile?",       [0,1], format_func=lambda x:"Yes" if x else "No")
        default_img     = st.selectbox("🖼️ Default Profile Image?", [0,1], format_func=lambda x:"Yes" if x else "No")
        geo_enabled     = st.selectbox("📍 Geo Enabled?",           [0,1], format_func=lambda x:"Yes" if x else "No")

    if st.button("🚀 Detect Account", use_container_width=True):
        ff_ratio = followers / (friends + 1)
        tpd      = statuses  / (account_age + 1)
        inp = pd.DataFrame([[followers, friends, statuses, favourites, listed,
                             default_profile, default_img, geo_enabled, verified,
                             account_age, ff_ratio, tpd]], columns=FEATURES)
        pred     = best_model.predict(inp)[0]
        prob     = best_model.predict_proba(inp)[0]
        bot_prob   = prob[1] * 100
        human_prob = prob[0] * 100

        st.markdown("---")
        if pred == 1:
            st.markdown(f'<div class="bot-result"><h2>🤖 BOT DETECTED!</h2><h3>Bot Probability: {bot_prob:.1f}%</h3><p>This account shows strong bot-like behaviour patterns.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="human-result"><h2>👤 HUMAN ACCOUNT</h2><h3>Human Probability: {human_prob:.1f}%</h3><p>This account shows genuine human behaviour patterns.</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        # PREDICTION PIE CHART
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#F8FAFF')
            wedges, texts, autotexts = ax.pie(
                [bot_prob, human_prob],
                labels=[f'Bot\n{bot_prob:.1f}%', f'Human\n{human_prob:.1f}%'],
                colors=['#ff416c','#11998e'],
                startangle=90,
                textprops={'fontweight':'bold', 'fontsize':11},
                wedgeprops={'edgecolor':'white','linewidth':2},
                autopct='%1.1f%%'
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
            ax.set_title('Prediction Confidence', fontweight='bold', fontsize=13, color='#1a1a2e', pad=12)
            ax.legend(wedges, ['Bot 🤖', 'Human 👤'],
                      title="Prediction", loc="lower left",
                      fontsize=9, title_fontsize=9, frameon=True, framealpha=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown('<p style="color:#1a1a2e;font-size:17px;font-weight:bold;">📊 Key Risk Indicators:</p>', unsafe_allow_html=True)
            for lbl, val, risky in [
                ("👥 Followers/Following Ratio", f"{ff_ratio:.3f}", ff_ratio < 0.1),
                ("🐦 Tweets per Day",            f"{tpd:.1f}",      tpd > 50),
                ("🖼️ Default Profile Image",    "Yes" if default_img  else "No", default_img  == 1),
                ("✅ Verified Account",           "Yes" if verified     else "No", verified     == 0),
                ("📅 Account Age",               f"{account_age} days", account_age < 100)]:
                bg    = "#FEE2E2" if risky else "#D1FAE5"
                bc    = "#EF4444" if risky else "#10B981"
                badge = "⚠️ BOT SIGNAL" if risky else "✅ NORMAL"
                st.markdown(f'<div style="background:{bg};border-radius:10px;padding:8px 12px;margin:6px 0;border-left:4px solid {bc};display:flex;justify-content:space-between;align-items:center;"><span style="color:#1a1a2e;font-weight:600;font-size:13px;">{lbl}</span><span style="color:#334155;font-size:13px;margin:0 8px;">{val}</span><span style="background:{bc};color:white;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;">{badge}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MODEL ANALYSIS
# ══════════════════════════════════════════════════════════
elif page=="📈 Model Analysis":

    # CORRELATION HEATMAP
    st.markdown('<div class="section-header">📈 Feature Correlations</div>',unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#F8FAFF')
    ax.set_facecolor('#F8FAFF')
    corr = df[FEATURES + ['account_type']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14, color='#1a1a2e', pad=14)
    ax.set_xlabel('Features', fontsize=11, color='#1a1a2e', labelpad=8)
    ax.set_ylabel('Features', fontsize=11, color='#1a1a2e', labelpad=8)
    ax.tick_params(axis='both', colors='#1a1a2e', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # FEATURE DISTRIBUTION
    st.markdown('<div class="section-header">📊 Feature Distributions by Class</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#1a1a2e;font-size:15px;font-weight:700;margin:10px 0 4px;">🔍 Select Feature to Analyze:</p>', unsafe_allow_html=True)
    selected_feat = st.selectbox("", FEATURES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for a in axes:
        a.set_facecolor('#F8FAFF')
    fig.patch.set_facecolor('#F8FAFF')

    human_data = df[df['account_type']==0][selected_feat]
    bot_data   = df[df['account_type']==1][selected_feat]

    # Use 95th percentile for xlim/ylim — removes outlier stretching problem
    q05 = df[selected_feat].quantile(0.05)
    q95 = df[selected_feat].quantile(0.95)
    x_min = max(0, q05 * 0.9) if q05 >= 0 else q05 * 1.1
    x_max = q95 * 1.15

    # ── HISTOGRAM ────────────────────────────────────────
    axes[0].hist(human_data, bins=40, alpha=0.65, color='#11998e',
                 label='Human 👤', edgecolor='white', linewidth=0.5)
    axes[0].hist(bot_data,   bins=40, alpha=0.65, color='#ff416c',
                 label='Bot 🤖',   edgecolor='white', linewidth=0.5)

    # plt.xlim() — focus on 95th percentile, avoids long empty tail
    axes[0].set_xlim(x_min, x_max)
    # plt.ylim() — auto with 20% headroom for labels
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.2)

    # Axis modification
    axes[0].set_title(f'{selected_feat} — Distribution',
                      fontweight='bold', fontsize=12, color='#1a1a2e', pad=10)
    axes[0].set_xlabel(f'{selected_feat}  (up to 95th percentile)',
                       fontsize=10, color='#1a1a2e', labelpad=8)
    axes[0].set_ylabel('Frequency (No. of Accounts)',
                       fontsize=10, color='#1a1a2e', labelpad=8)
    axes[0].tick_params(axis='both', colors='#1a1a2e', labelsize=9)
    axes[0].grid(axis='y', alpha=0.25, linestyle='--', color='#aaa')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # plt.legend()
    axes[0].legend(
        fontsize=10, frameon=True, framealpha=0.85,
        loc='upper right',
        title='Account Type', title_fontsize=9,
        edgecolor='#ccc'
    )

    # ── BOXPLOT ──────────────────────────────────────────
    sns.boxplot(data=df, x='account_type', y=selected_feat,
                palette=['#11998e', '#ff416c'], ax=axes[1],
                linewidth=1.5, fliersize=2,
                flierprops=dict(alpha=0.3, marker='o'))

    # plt.xlim()
    axes[1].set_xlim(-0.6, 1.6)
    # plt.ylim() — clamp to 95th percentile so boxes are not squished at bottom
    axes[1].set_ylim(x_min, x_max)

    # Axis modification
    axes[1].set_xticklabels(['Human 👤', 'Bot 🤖'], fontsize=11, color='#1a1a2e')
    axes[1].set_title(f'{selected_feat} — Boxplot by Class',
                      fontweight='bold', fontsize=12, color='#1a1a2e', pad=10)
    axes[1].set_xlabel('Account Type', fontsize=11, color='#1a1a2e', labelpad=8)
    axes[1].set_ylabel(f'{selected_feat} Value',
                       fontsize=11, color='#1a1a2e', labelpad=8)
    axes[1].tick_params(axis='both', colors='#1a1a2e', labelsize=9)
    axes[1].grid(axis='y', alpha=0.25, linestyle='--', color='#aaa')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # plt.legend()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#11998e', label='Human 👤'),
        Patch(facecolor='#ff416c', label='Bot 🤖')
    ]
    axes[1].legend(
        handles=legend_elements, loc='upper right',
        fontsize=9, frameon=True, framealpha=0.85,
        title='Account Type', title_fontsize=9,
        edgecolor='#ccc'
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.markdown("""<center style='color:#888;font-size:0.85rem;'>
    🤖 <b>SocialGuard: </b><br/>
    Chandru B | M.Sc Artificial Intelligence | Bharathiar University |
</center>""", unsafe_allow_html=True)
