import streamlit as st

st.set_page_config(
    page_title="Nilinpach IS Project",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --peach:  #E69D45;
    --coral:  #D45769;
    --mauve:  #308695;
    --purple: #455054;
    --teal:   #308695;
    --bg:     #212427;
    --card:   #354044;
    --card2:  #2e3a3e;
    --border: #3d5055;
    --text:   #D4CFC9;
    --sub:    #9aaba8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
}

div[data-testid="stSpinner"] > div {
    border: 3px solid rgba(212,87,105,0.2) !important;
    border-top: 3px solid #D45769 !important;
    border-radius: 50% !important;
    width: 28px !important; height: 28px !important;
    animation: cspin 0.85s linear infinite !important;
    background: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stSpinner"] > div > * { display:none !important; }
@keyframes cspin { to { transform: rotate(360deg); } }

[data-testid="stSidebar"] {
    background: #212427;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 9px 14px; margin: 4px 0;
    cursor:pointer; transition: all 0.2s; font-size:0.88rem;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(212,87,105,0.15); border-color: var(--coral);
}

[data-testid="stAppViewContainer"] { background: var(--bg); }
.main .block-container { padding-top: 2rem; max-width: 1200px; }
h1,h2,h3 { font-family:'DM Sans',sans-serif !important; color:var(--text) !important; }

.info-card {
    background: linear-gradient(135deg,var(--card),var(--card2));
    border: 1px solid var(--border);
    border-radius: 16px; padding: 24px; margin: 12px 0;
}
.badge {
    display:inline-block; background:var(--mauve); color:white;
    border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; margin:2px;
}
.badge-green { background:#4a8c6f; }

.stButton > button {
    background: linear-gradient(135deg,var(--coral),var(--mauve));
    color:white; border:none; border-radius:10px;
    font-weight:600; padding:0.5rem 2rem; transition:all 0.2s;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 25px rgba(212,87,105,0.35);
}

p,li,label,.stMarkdown { color:#c8b8c0 !important; }
.stDataFrame { border-radius:12px; overflow:hidden; }
[data-testid="stMetricValue"] { color:var(--peach) !important; font-size:2rem !important; }
[data-testid="stMetricLabel"] { color:var(--sub) !important; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 12px;'>
        <p style='font-size:1.73rem;color:#f0e8e0;'>Nilinpach</p>
        <p style='font-size:1rem;color:#f0e8e0 !important;margin:0;'>6704062610241</p>
        <p style='font-size:0.73rem;color:#9aaba8 !important;margin:0;'>Heart Disease & Titanic</p>
    </div>
    <hr style='border-color:#3d5055;margin:14px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Home", "Machine Learning Models", "Neural Network Models", "ทดสอบ Heart Disease", "ทดสอบ Titanic"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:#3d5055;margin:14px 0;'>
    <p style='font-size:0.7rem;color:#6a8080 !important;text-align:center;'>Dataset: Heart Disease & Titanic<br>Models: Ensemble ML & Neural Network</p>
    """, unsafe_allow_html=True)

if page == "Home":
    from pages_content import page_overview; page_overview()
elif page == "Machine Learning Models":
    from pages_content import page_combined_ml; page_combined_ml()
elif page == "Neural Network Models":
    from pages_content import page_combined_nn; page_combined_nn()
elif page == "ทดสอบ Heart Disease":
    from pages_content import page_test_heart; page_test_heart()
elif page == "ทดสอบ Titanic":
    from pages_content import page_test_titanic; page_test_titanic()
