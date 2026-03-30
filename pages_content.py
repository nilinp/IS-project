import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from model_utils import (load_heart, load_titanic, train_heart_ml, train_heart_nn, train_titanic_ml, train_titanic_nn)
from chart_utils import (confusion_matrix_fig, roc_curve_fig, feature_importance_fig, training_history_fig, distribution_fig, correlation_fig,
                         ACCENT, GREEN, RED, CARD_BG, TEXT, SUBTEXT, LAYOUT)

# HELPERS

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def section(title, icon=""):
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; margin:28px 0 12px;'>
        <span style='font-size:1.4rem;'>{icon}</span>
        <h3 style='margin:0; color:#D4CFC9;'>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def badge(text, color="#308695"):
    return f"<span style='background:{color};color:white;border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;margin:2px;display:inline-block;'>{text}</span>"


def result_card(title, content, border_color="#D45769"):
    st.markdown(f"""
    <div style='background:#354044; border:1px solid {border_color}; border-left: 4px solid {border_color}; border-radius:12px; padding:20px; margin:10px 0;'>
        <b style='color:#D4CFC9; font-size:1rem;'>{title}</b><br>
        <span style='color:#bab5b0;'>{content}</span>
    </div>
    """, unsafe_allow_html=True)

# PAGE: OVERVIEW

def page_overview():
    st.markdown("""
    <h1 style='font-size:2.4rem; margin-bottom:4px;'> ML Model Explorer</h1>
    <p style='color:#9aaba8; margin-top:0; font-size:1rem;'>
        สำรวจและเปรียบเทียบ Machine Learning vs Neural Network บน 2 datasets
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='info-card'>
            <h3 style='color:#D4CFC9; margin:8px 0 0px;'>Heart Disease</h3>
            <p style='color:#9aaba8; font-size:0.85rem; margin:0 0 12px;'>ทำนายความเสี่ยงโรคหัวใจจากข้อมูลทางการแพทย์ 13 features</p>
            {badge("303 samples")} {badge("Binary Classification")} {badge("13 features", "#4a9e78")}
            <hr style='border-color:#3d5055; margin:14px 0;'>
            <b style='color:#ccc8c0; font-size:0.85rem;'>ML Model:</b>
            <p style='color:#9aaba8; font-size:0.82rem; margin:4px 0 8px;'>VotingClassifier (Random Forest + Gradient Boosting + Logistic Regression)</p>
            <b style='color:#ccc8c0; font-size:0.85rem;'>Neural Network:</b>
            <p style='color:#9aaba8; font-size:0.82rem; margin:4px 0;'>4-layer Dense NN (128→64→32→1) with BatchNorm, Dropout, EarlyStopping</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='info-card'>
            <h3 style='color:#D4CFC9; margin:8px 0 0px;'>Titanic Survival</h3>
            <p style='color:#9aaba8; font-size:0.85rem; margin:0 0 12px;'>ทำนายการรอดชีวิตจากภัยพิบัติเรือไทแทนิก</p>
            {badge("891 samples")} {badge("Binary Classification")} {badge("Feature Engineered", "#4a9e78")}
            <hr style='border-color:#3d5055; margin:14px 0;'>
            <b style='color:#ccc8c0; font-size:0.85rem;'>ML Model:</b>
            <p style='color:#9aaba8; font-size:0.82rem; margin:4px 0 8px;'>VotingClassifier (Random Forest + Gradient Boosting + SVM)</p>
            <b style='color:#ccc8c0; font-size:0.85rem;'>Neural Network:</b>
            <p style='color:#9aaba8; font-size:0.82rem; margin:4px 0;'>5-layer Dense NN (256→128→64→32→1) with L2 Regularization, Dropout</p>
        </div>
        """, unsafe_allow_html=True)

    section("ขั้นตอนการพัฒนาโมเดล")
    steps = [
        ("1. Data Loading", "โหลดข้อมูลจาก CSV ตรวจสอบ shape, dtypes, missing values"),
        ("2. Feature Engineering", "สร้าง features ใหม่ เช่น age_group, hr_age_ratio, risk_score, FamilySize, Title"),
        ("3. Preprocessing", "StandardScaler สำหรับ normalization, train/test split 80:20"),
        ("4. Model Training", "ฝึก ML Ensemble และ Neural Network พร้อม callbacks"),
        ("5. Evaluation", "วัดผล Accuracy, AUC, Confusion Matrix, Classification Report"),
    ]
    c1, c2 = st.columns(2)
    for i, (title, desc) in enumerate(steps):
        (c1 if i % 2 == 0 else c2).markdown(f"""
        <div style='background:#2b3538; border:1px solid #3d5055; border-radius:10px; padding:14px; margin:6px 0;'>
            <b style='color:#E69D45;'>{title}</b>
            <p style='color:#9aaba8; font-size:0.82rem; margin:4px 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    section("แหล่งอ้างอิง")
    refs = [
        ("Scikit-learn", "https://scikit-learn.org", "Random Forest, GradientBoosting, SVM, Logistic Regression"),
        ("TensorFlow / Keras", "https://www.tensorflow.org", "Sequential Neural Network, Dense, BatchNorm, Dropout"),
        ("Heart Disease Dataset", "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset", "UCI Heart Disease Data"),
        ("Titanic Dataset", "https://www.kaggle.com/competitions/titanic", "Kaggle Titanic Competition"),
        ("Plotly", "https://plotly.com/python/", "Interactive visualizations"),
        ("Streamlit", "https://streamlit.io", "Web application framework"),
    ]
    c1, c2, c3 = st.columns(3)
    for i, (name, url, desc) in enumerate(refs):
        col = [c1, c2, c3][i % 3]
        col.markdown(f"""
        <div style='background:#2b3538; border:1px solid #3d5055; border-radius:10px; padding:12px; margin:4px 0;'>
            <a href='{url}' target='_blank' style='color:#E69D45; text-decoration:none; font-weight:600;'>{name}</a>
            <p style='color:#607878; font-size:0.78rem; margin:4px 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE: HEART DISEASE — ML

def page_heart_ml():
    st.markdown("<h1>️ Heart Disease — Machine Learning Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Ensemble: Random Forest + Gradient Boosting + Logistic Regression</p>", unsafe_allow_html=True)

    # Data Overview
    section("Dataset Overview")
    df, X, y = load_heart()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df))
    c2.metric("Features (raw)", "13")
    c3.metric("Features (engineered)", len(X.columns))
    c4.metric("Positive cases", f"{y.sum()} ({y.mean()*100:.0f}%)")

    with st.expander(" ดูข้อมูล 10 แถวแรก"):
        st.dataframe(df.head(10), use_container_width=True)

    tab1, tab2 = st.tabs([" Distribution", " Correlation"])
    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(distribution_fig(df, "age", "target", "Age by Target"), use_container_width=True)
        c2.plotly_chart(distribution_fig(df, "thalach", "target", "Max Heart Rate by Target"), use_container_width=True)
    with tab2:
        st.plotly_chart(correlation_fig(df), use_container_width=True)

    # Algorithm
    section("Algorithm")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='info-card'>
            <b style='color:#E69D45;'> Random Forest</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 200</li>
                <li>max_depth = 8</li>
                <li>min_samples_split = 4</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='info-card'>
            <b style='color:#4a9e78;'> Gradient Boosting</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 150</li>
                <li>learning_rate = 0.1</li>
                <li>max_depth = 4, subsample=0.8</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='info-card'>
            <b style='color:#f59e0b;'> Logistic Regression</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>C = 1.0</li>
                <li>max_iter = 1000</li>
                <li>Soft voting weight = 1</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # Train & Results
    section("Result")
    with st.spinner("กำลังเทรนโมเดล... ⏳"):
        res = train_heart_ml()

    metric_row([
        ("Accuracy", f"{res['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res['auc']:.4f}",      None),
        ("Precision (class 1)", f"{res['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (class 1)",    f"{res['cr']['1']['recall']*100:.1f}%",    None),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res["cm"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res["fpr"], res["tpr"], res["auc"]), use_container_width=True)

    st.plotly_chart(
        feature_importance_fig(res["feature_names"], res["feature_importances"], "Feature Importances (RF + GB avg)"),
        use_container_width=True)

    section("Feature Engineering")
    st.markdown(f"""
    <div class='info-card'>
        {badge("age_group")} แบ่งอายุเป็น 4 กลุ่ม (0–40, 40–55, 55–70, 70+)<br><br>
        {badge("hr_age_ratio")} อัตราส่วนชีพจรสูงสุด ÷ อายุ — บ่งบอกสมรรถภาพหัวใจ<br><br>
        {badge("risk_score")} ผลรวมของ cp + exang + slope — score ความเสี่ยงรวม
    </div>
    """, unsafe_allow_html=True)

# PAGE: HEART DISEASE NN

def page_heart_nn():
    st.markdown("<h1> Heart Disease — Neural Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Deep Learning: 4-layer Sequential (128→64→32→1) with EarlyStopping</p>", unsafe_allow_html=True)

    section("Neural Network Architecture")
    st.markdown("""
    <div class='info-card'>
        <div style='font-family: "JetBrains Mono", monospace; font-size:0.82rem; color:#E69D45; line-height:1.8;'>
            Input(shape=(17,))<br>
            → Dense(128, relu) → BatchNormalization → Dropout(0.3)<br>
            → Dense(64, relu) → BatchNormalization → Dropout(0.2)<br>
            → Dense(32, relu) → Dropout(0.1)<br>
            → Dense(1, sigmoid)
        </div>
        <hr style='border-color:#3d5055; margin:14px 0;'>
        <b style='color:#ccc8c0; font-size:0.85rem;'>Compile:</b>
        <span style='color:#9aaba8; font-size:0.82rem;'>Adam(lr=0.001) · BinaryCrossentropy · Metrics: Accuracy, AUC</span><br>
        <b style='color:#ccc8c0; font-size:0.85rem;'>Callbacks:</b>
        <span style='color:#9aaba8; font-size:0.82rem;'>EarlyStopping(monitor=val_auc, patience=15) · ReduceLROnPlateau(patience=7)</span>
    </div>
    """, unsafe_allow_html=True)

    section("Result")
    with st.spinner("กำลังเทรน Neural Network... ⏳"):
        res = train_heart_nn()

    metric_row([
        ("Accuracy", f"{res['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res['auc']:.4f}",      None),
        ("Precision (class 1)", f"{res['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (class 1)",    f"{res['cr']['1']['recall']*100:.1f}%",    None),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res["cm"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res["fpr"], res["tpr"], res["auc"]), use_container_width=True)

    section("Training History", "")
    c1, c2 = st.columns(2)
    c1.plotly_chart(training_history_fig(res["history"], "accuracy"), use_container_width=True)
    c2.plotly_chart(training_history_fig(res["history"], "loss"), use_container_width=True)

    section("เหตุผลที่เลือก Neural Network", "")
    st.markdown(f"""
    <div class='info-card'>
        {badge("Non-linear boundaries", "#4a9e78")} Neural Net จับ pattern ที่ซับซ้อนได้ดีกว่า linear models<br><br>
        {badge("BatchNormalization")} ช่วยให้ gradient flow ดีขึ้น ลด internal covariate shift<br><br>
        {badge("Dropout")} ป้องกัน overfitting โดยสุ่ม deactivate neurons ระหว่าง training<br><br>
        {badge("EarlyStopping")} หยุดเทรนเมื่อ validation AUC ไม่ดีขึ้น 15 epochs ติดต่อกัน
    </div>
    """, unsafe_allow_html=True)

# PAGE: TITANIC ML

def page_titanic_ml():
    st.markdown("<h1> Titanic — Machine Learning Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Ensemble: Random Forest + Gradient Boosting + SVM (Soft Voting)</p>", unsafe_allow_html=True)

    section("Dataset Overview")
    df, X, y = load_titanic()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", 891)
    c2.metric("Features (engineered)", len(X.columns))
    c3.metric("Survived", f"{y.sum()} ({y.mean()*100:.0f}%)")
    c4.metric("Did not survive", f"{(~y.astype(bool)).sum()}")

    with st.expander(" ดูข้อมูล 10 แถวแรก"):
        raw_df = pd.read_csv("Titanic-Dataset.csv")
        st.dataframe(raw_df.head(10), use_container_width=True)

    section("Feature Engineering")
    st.markdown(f"""
    <div class='info-card'>
        {badge("Title")} ดึง Title จากชื่อ (Mr, Mrs, Miss, Master, Rare) — บ่งบอก social status<br><br>
        {badge("FamilySize")} SibSp + Parch + 1 — ขนาดครอบครัว<br><br>
        {badge("IsAlone")} 1 ถ้า FamilySize == 1 — เดินทางคนเดียว<br><br>
        {badge("Age*Class")} Age × Pclass — interaction feature บ่งบอก vulnerability<br><br>
        {badge("Cabin encoding")} เก็บ deck letter (A–G, U) แทน full cabin code
    </div>
    """, unsafe_allow_html=True)

    section("Algorithm")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='info-card'>
            <b style='color:#E69D45;'> Random Forest</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 300</li>
                <li>max_depth = 6</li>
                <li>min_samples_split = 4</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card'>
            <b style='color:#4a9e78;'> Gradient Boosting</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 200</li>
                <li>learning_rate = 0.05</li>
                <li>max_depth = 4</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card'>
            <b style='color:#f59e0b;'> SVM (RBF kernel)</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>C = 1.5</li>
                <li>kernel = rbf</li>
                <li>probability = True</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    section("Result")
    with st.spinner("กำลังเทรนโมเดล... ⏳"):
        res = train_titanic_ml()

    metric_row([
        ("Accuracy", f"{res['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res['auc']:.4f}",      None),
        ("Precision (survived)", f"{res['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (survived)",    f"{res['cr']['1']['recall']*100:.1f}%",    None),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res["cm"], ["Not Survived","Survived"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res["fpr"], res["tpr"], res["auc"]), use_container_width=True)

    st.plotly_chart(
        feature_importance_fig(res["feature_names"], res["feature_importances"], "Feature Importances (RF + GB avg)"),
        use_container_width=True)

# PAGE: TITANIC NN

def page_titanic_nn():
    st.markdown("<h1> Titanic — Neural Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Deep Learning: 5-layer Sequential (256→128→64→32→1) with L2 Regularization</p>", unsafe_allow_html=True)

    section("Neural Network Architecture")
    st.markdown("""
    <div class='info-card'>
        <div style='font-family: "JetBrains Mono", monospace; font-size:0.82rem; color:#E69D45; line-height:1.8;'>
            Input(shape=(N,))<br>
            → Dense(256, relu, L2=1e-4) → BatchNormalization → Dropout(0.3)<br>
            → Dense(128, relu, L2=1e-4) → BatchNormalization → Dropout(0.2)<br>
            → Dense(64, relu, L2=1e-4) → Dropout(0.1)<br>
            → Dense(32, relu)<br>
            → Dense(1, sigmoid)
        </div>
        <hr style='border-color:#3d5055; margin:14px 0;'>
        <b style='color:#ccc8c0; font-size:0.85rem;'>Compile:</b>
        <span style='color:#9aaba8; font-size:0.82rem;'>Adam(lr=0.0005) · BinaryCrossentropy · Metric: Accuracy</span><br>
        <b style='color:#ccc8c0; font-size:0.85rem;'>Training:</b>
        <span style='color:#9aaba8; font-size:0.82rem;'>100 epochs · batch_size=16 · validation_split=0.2</span>
    </div>
    """, unsafe_allow_html=True)

    section("Result")
    with st.spinner("กำลังเทรน Neural Network... ⏳"):
        res = train_titanic_nn()

    metric_row([
        ("Accuracy", f"{res['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res['auc']:.4f}",      None),
        ("Precision (survived)", f"{res['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (survived)",    f"{res['cr']['1']['recall']*100:.1f}%",    None),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res["cm"], ["Not Survived","Survived"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res["fpr"], res["tpr"], res["auc"]), use_container_width=True)

    section("Training History", "")
    c1, c2 = st.columns(2)
    c1.plotly_chart(training_history_fig(res["history"], "accuracy"), use_container_width=True)
    c2.plotly_chart(training_history_fig(res["history"], "loss"), use_container_width=True)

    section("L2 Regularization คืออะไร?", "")
    st.markdown(f"""
    <div class='info-card'>
        {badge("L2 (Ridge)", "#4a9e78")} เพิ่ม penalty = λ·Σw² ไปใน loss function<br><br>
        <p style='color:#9aaba8; font-size:0.85rem;'>ทำให้ weights มีค่าน้อยลงโดยรวม ช่วยลด overfitting<br> ต่างจาก L1 ที่ทำให้บาง weight → 0 (sparse), L2 จะ shrink ทุก weight พร้อมกัน<br>
        ใช้ค่า λ = 0.0001 เพื่อ regularization เบาๆ โดยไม่ under-fit</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE: TEST HEART DISEASE

def page_test_heart():
    st.markdown("<h1> ทดสอบโมเดล — Heart Disease</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>กรอกข้อมูลผู้ป่วยเพื่อทำนายความเสี่ยงโรคหัวใจ</p>", unsafe_allow_html=True)

    with st.spinner("โหลดโมเดล..."):
        res_ml = train_heart_ml()
        res_nn = train_heart_nn()

    c1, c2, c3 = st.columns(3)
    with c1:
        age        = st.slider("อายุ (age)", 20, 80, 55)
        sex        = st.selectbox("เพศ (sex)", [("ชาย", 1), ("หญิง", 0)], format_func=lambda x: x[0])[1]
        cp         = st.selectbox("ประเภทเจ็บหน้าอก (cp)", [0,1,2,3], format_func=lambda x: {0:"0 — Typical Angina (เจ็บจากหัวใจขาดเลือด)",1:"1 — Atypical Angina (เจ็บผิดปกติ)",2:"2 — Non-anginal Pain (ไม่ใช่ angina)",3:"3 — Asymptomatic (ไม่มีอาการ)"}[x])
        trestbps   = st.slider("ความดันโลหิต (trestbps)", 90, 200, 130)
        chol       = st.slider("คอเลสเตอรอล (chol)", 100, 400, 230)
    with c2:
        fbs        = st.selectbox("น้ำตาลในเลือด > 120 mg/dl (fbs)", [0, 1], format_func=lambda x: "ใช่" if x else "ไม่ใช่")
        restecg    = st.selectbox("ผล ECG ขณะพัก (restecg)", [0,1,2], format_func=lambda x: {0:"Normal (ปกติ)", 1:"ST-T Wave Abnormality (คลื่นไฟฟ้าผิดปกติ)", 2:"Left Ventricular Hypertrophy (หัวใจห้องล่างซ้ายโต)"}[x])
        thalach    = st.slider("ชีพจรสูงสุด (thalach)", 70, 210, 150)
        exang      = st.selectbox("เจ็บหน้าอกขณะออกกำลัง (exang)", [0, 1], format_func=lambda x: "ใช่" if x else "ไม่ใช่")
    with c3:
        oldpeak    = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope      = st.selectbox("Slope ของ ST segment (slope)", [0,1,2], format_func=lambda x: {0:"Upsloping (ลาดขึ้น — ดีที่สุด)", 1:"Flat (แบน — ปานกลาง)", 2:"Downsloping (ลาดลง — เสี่ยงสูง)"}[x])
        ca         = st.selectbox("จำนวนหลอดเลือดหลักที่แคบ (ca)", [0,1,2,3], format_func=lambda x: {0:"ไม่มีหลอดเลือดแคบ (ปกติ)", 1:"แคบ 1 เส้น", 2:"แคบ 2 เส้น", 3:"แคบ 3 เส้น (รุนแรง)"}[x])
        thal       = st.selectbox("Thalassemia (thal)", [0,1,2,3], format_func=lambda x: {0:"Normal (ปกติ)", 1:"Fixed Defect (ขาดเลือดถาวร)", 2:"Reversible Defect (ขาดเลือดชั่วคราว ฟื้นตัวได้)", 3:"Unknown / Other"}[x])

    if st.button(" ทำนาย Heart Disease"):
        input_dict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        df_input = pd.DataFrame([input_dict])
        df_input["age_group"]    = pd.cut(df_input["age"], bins=[0,40,55,70,100], labels=[0,1,2,3]).astype(int)
        df_input["hr_age_ratio"] = df_input["thalach"] / df_input["age"]
        df_input["risk_score"]   = (df_input["cp"] + df_input["exang"] + df_input["slope"]).astype(int)

        X_scaled_ml = res_ml["scaler"].transform(df_input)
        X_scaled_nn = res_nn["scaler"].transform(df_input)

        pred_ml    = res_ml["model"].predict(X_scaled_ml)[0]
        proba_ml   = res_ml["model"].predict_proba(X_scaled_ml)[0][1]
        proba_nn   = float(res_nn["model"].predict(X_scaled_nn, verbose=0).flatten()[0])
        pred_nn    = int(proba_nn >= 0.5)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            color = "#D45769" if pred_ml == 1 else "#4a9e78"
            label = "️ มีความเสี่ยงโรคหัวใจ" if pred_ml == 1 else " ความเสี่ยงต่ำ"
            st.markdown(f"""
            <div style='background:#354044; border:2px solid {color}; border-radius:16px; padding:24px; text-align:center;'>
                <h3 style='color:{color}; margin:0;'>{label}</h3>
                <p style='color:#9aaba8; margin:8px 0 0;'>ML Ensemble</p>
                <div style='font-size:2.5rem; color:{color}; font-weight:700;'>{proba_ml*100:.1f}%</div>
                <p style='color:#607878; font-size:0.8rem; margin:0;'>ความน่าจะเป็น</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            color2 = "#D45769" if pred_nn == 1 else "#4a9e78"
            label2 = "️ มีความเสี่ยงโรคหัวใจ" if pred_nn == 1 else " ความเสี่ยงต่ำ"
            st.markdown(f"""
            <div style='background:#354044; border:2px solid {color2}; border-radius:16px; padding:24px; text-align:center;'>
                <h3 style='color:{color2}; margin:0;'>{label2}</h3>
                <p style='color:#9aaba8; margin:8px 0 0;'>Neural Network</p>
                <div style='font-size:2.5rem; color:{color2}; font-weight:700;'>{proba_nn*100:.1f}%</div>
                <p style='color:#607878; font-size:0.8rem; margin:0;'>ความน่าจะเป็น</p>
            </div>
            """, unsafe_allow_html=True)

        avg_proba = (proba_ml + proba_nn) / 2
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=avg_proba*100,
            title={"text": "ค่าเฉลี่ยความเสี่ยง (%) (ML + NN)", "font": {"color": "#D4CFC9"}},
            number={"suffix": "%", "font": {"color": "#D4CFC9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aaba8"},
                "bar": {"color": "#D45769"},
                "bgcolor": "#354044",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "#0d3321"},
                    {"range": [30, 60], "color": "#3a2a00"},
                    {"range": [60, 100],"color": "#3a0a0a"},
                ],
                "threshold": {"line": {"color": "#D45769", "width": 3}, "value": 50},
            },
        ))
        fig.update_layout(paper_bgcolor="#2b3336", font_color="#D4CFC9", height=280)
        st.plotly_chart(fig, use_container_width=True)

        st.info("️ ผลลัพธ์นี้เป็นเพียงการทดสอบโมเดล ML เท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์")

# PAGE: TEST TITANIC

def page_test_titanic():
    st.markdown("<h1> ทดสอบโมเดล — Titanic Survival</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>กรอกข้อมูลผู้โดยสารเพื่อทำนายโอกาสรอดชีวิต</p>", unsafe_allow_html=True)

    with st.spinner("โหลดโมเดล..."):
        res_ml = train_titanic_ml()
        res_nn = train_titanic_nn()

    # Get feature names to match training
    _, X_template, _ = load_titanic()
    all_features = list(X_template.columns)

    c1, c2 = st.columns(2)
    with c1:
        pclass   = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
        sex      = st.selectbox("เพศ (Sex)", [("ชาย", 0), ("หญิง", 1)], format_func=lambda x: x[0])[1]
        age      = st.slider("อายุ (Age)", 1, 80, 30)
        sibsp    = st.slider("พี่น้อง/คู่สมรส (SibSp)", 0, 8, 0)
        parch    = st.slider("พ่อแม่/ลูก (Parch)", 0, 6, 0)
    with c2:
        fare     = st.slider("ราคาตั๋ว (Fare)", 0.0, 550.0, 30.0, 0.5)
        title    = st.selectbox("Title", [0,1,2,3,4], format_func=lambda x: {0:"Mr",1:"Miss",2:"Mrs",3:"Master",4:"Rare"}[x])
        embarked = st.selectbox("ท่าเรือ (Embarked)", ["Southampton", "Cherbourg", "Queenstown"])
        cabin    = st.selectbox("Deck", ["U","A","B","C","D","E","F","G"])

    if st.button(" ทำนายการรอดชีวิต"):
        family_size = sibsp + parch + 1
        is_alone    = int(family_size == 1)
        age_class   = age * pclass

        row = {f: 0 for f in all_features}
        for col_name in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Title", "FamilySize", "IsAlone", "Age*Class"]:
            if col_name in row:
                row[col_name] = locals()[col_name.lower().replace("*","_star_")] if col_name not in ("FamilySize", "IsAlone", "Age*Class") else {
                    "FamilySize": family_size, "IsAlone": is_alone, "Age*Class": age_class}[col_name]

        row["Pclass"]     = pclass
        row["Sex"]        = sex
        row["Age"]        = age
        row["SibSp"]      = sibsp
        row["Parch"]      = parch
        row["Fare"]       = fare
        row["Title"]      = title
        row["FamilySize"] = family_size
        row["IsAlone"]    = is_alone
        row["Age*Class"]  = age_class

        if f"Embarked_{embarked}" in row:
            row[f"Embarked_{embarked}"] = 1
        if f"Cabin_{cabin}" in row:
            row[f"Cabin_{cabin}"] = 1

        df_input = pd.DataFrame([row])
        X_scaled_ml = res_ml["scaler"].transform(df_input)
        X_scaled_nn = res_nn["scaler"].transform(df_input)

        pred_ml  = res_ml["model"].predict(X_scaled_ml)[0]
        proba_ml = res_ml["model"].predict_proba(X_scaled_ml)[0][1]
        proba_nn = float(res_nn["model"].predict(X_scaled_nn, verbose=0).flatten()[0])
        pred_nn  = int(proba_nn >= 0.5)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            color = "#4a9e78" if pred_ml == 1 else "#D45769"
            label = " รอดชีวิต" if pred_ml == 1 else " ไม่รอดชีวิต"
            st.markdown(f"""
            <div style='background:#354044; border:2px solid {color}; border-radius:16px; padding:24px; text-align:center;'>
                <h3 style='color:{color}; margin:0;'>{label}</h3>
                <p style='color:#9aaba8; margin:8px 0 0;'>ML Ensemble</p>
                <div style='font-size:2.5rem; color:{color}; font-weight:700;'>{proba_ml*100:.1f}%</div>
                <p style='color:#607878; font-size:0.8rem; margin:0;'>โอกาสรอดชีวิต</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            color2 = "#4a9e78" if pred_nn == 1 else "#D45769"
            label2 = " รอดชีวิต" if pred_nn == 1 else " ไม่รอดชีวิต"
            st.markdown(f"""
            <div style='background:#354044; border:2px solid {color2}; border-radius:16px; padding:24px; text-align:center;'>
                <h3 style='color:{color2}; margin:0;'>{label2}</h3>
                <p style='color:#9aaba8; margin:8px 0 0;'>Neural Network</p>
                <div style='font-size:2.5rem; color:{color2}; font-weight:700;'>{proba_nn*100:.1f}%</div>
                <p style='color:#607878; font-size:0.8rem; margin:0;'>โอกาสรอดชีวิต</p>
            </div>""", unsafe_allow_html=True)

        avg = (proba_ml + proba_nn) / 2
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=avg*100,
            title={"text":"โอกาสรอดชีวิต (%) — ML + NN", "font":{"color":"#D4CFC9"}},
            number={"suffix":"%","font":{"color":"#D4CFC9"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#9aaba8"},
                "bar":{"color":"#4a9e78"},
                "bgcolor":"#354044",
                "borderwidth":0,
                "steps":[
                    {"range":[0,30],"color":"#3a0a0a"},
                    {"range":[30,60],"color":"#3a2a00"},
                    {"range":[60,100],"color":"#0d3321"},
                ],
                "threshold":{"line":{"color":"#4a9e78","width":3},"value":50},
            },
        ))
        fig.update_layout(paper_bgcolor="#2b3336", font_color="#D4CFC9", height=280)
        st.plotly_chart(fig, use_container_width=True)

# PAGE: COMBINED ML (Titanic + Heart Disease)

def page_combined_ml():
    st.markdown("<h1> Machine Learning Models</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Ensemble Learning บน 2 datasets — Titanic Survival & Heart Disease</p>", unsafe_allow_html=True)

    # TITANIC
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin:32px 0 16px; padding:16px 20px; background:linear-gradient(135deg,#354044,#2e3a3e);
                border:1px solid #3d5055; border-left:4px solid #D45769; border-radius:12px;'>
        <span style='font-size:2rem;'></span>
        <div>
            <h2 style='margin:0; color:#D4CFC9; font-size:1.4rem;'>Titanic Survival Prediction</h2>
            <p style='margin:2px 0 0; color:#9aaba8; font-size:0.82rem;'>
                VotingClassifier: Random Forest + Gradient Boosting + SVM (Soft Voting)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_t, X_t, y_t = load_titanic()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", 891)
    c2.metric("Features (engineered)", len(X_t.columns))
    c3.metric("Survived", f"{y_t.sum()} ({y_t.mean()*100:.0f}%)")
    c4.metric("Did not survive", f"{(~y_t.astype(bool)).sum()}")

    section("Feature Engineering (Titanic)")
    st.markdown(f"""
    <div class='info-card'>
        {badge("Title")} ดึง Title จากชื่อ (Mr, Mrs, Miss, Master, Rare) — บ่งบอก social status<br><br>
        {badge("FamilySize")} SibSp + Parch + 1 — ขนาดครอบครัว &nbsp;
        {badge("IsAlone")} 1 ถ้า FamilySize == 1 &nbsp;
        {badge("Age*Class")} Age × Pclass — interaction feature
    </div>
    """, unsafe_allow_html=True)

    section("Algorithm (Titanic)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-card'>
            <b style='color:#E69D45;'> Random Forest</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 300, max_depth = 6</li>
                <li>min_samples_split = 4</li>
            </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='info-card'>
            <b style='color:#4a9e78;'> Gradient Boosting</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 200, lr = 0.05</li>
                <li>max_depth = 4</li>
            </ul></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='info-card'>
            <b style='color:#f59e0b;'> SVM (RBF kernel)</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>C = 1.5, kernel = rbf</li>
                <li>probability = True</li>
            </ul></div>""", unsafe_allow_html=True)

    section("Result (Titanic)")
    with st.spinner("กำลังเทรนโมเดล Titanic ML... ⏳"):
        res_t = train_titanic_ml()

    metric_row([
        ("Accuracy", f"{res_t['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res_t['auc']:.4f}",      None),
        ("Precision (survived)", f"{res_t['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (survived)",    f"{res_t['cr']['1']['recall']*100:.1f}%",    None),
    ])
    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res_t["cm"], ["Not Survived","Survived"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res_t["fpr"], res_t["tpr"], res_t["auc"]), use_container_width=True)
    st.plotly_chart(
        feature_importance_fig(res_t["feature_names"], res_t["feature_importances"], "Feature Importances — Titanic (RF + GB avg)"),
        use_container_width=True)

    st.divider()

    # HEART DISEASE
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin:32px 0 16px; padding:16px 20px; background:linear-gradient(135deg,#354044,#2e3a3e);
                border:1px solid #3d5055; border-left:4px solid #D45769; border-radius:12px;'>
        <span style='font-size:2rem;'>️</span>
        <div>
            <h2 style='margin:0; color:#D4CFC9; font-size:1.4rem;'>Heart Disease Prediction</h2>
            <p style='margin:2px 0 0; color:#9aaba8; font-size:0.82rem;'>
                VotingClassifier: Random Forest + Gradient Boosting + Logistic Regression (Soft Voting)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_h, X_h, y_h = load_heart()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df_h))
    c2.metric("Features (raw)", "13")
    c3.metric("Features (engineered)", len(X_h.columns))
    c4.metric("Positive cases", f"{y_h.sum()} ({y_h.mean()*100:.0f}%)")

    section("Feature Engineering (Heart Disease)")
    st.markdown(f"""
    <div class='info-card'>
        {badge("age_group")} แบ่งอายุเป็น 4 กลุ่ม (0–40, 40–55, 55–70, 70+) &nbsp;
        {badge("hr_age_ratio")} ชีพจรสูงสุด ÷ อายุ — สมรรถภาพหัวใจ &nbsp;
        {badge("risk_score")} cp + exang + slope — score ความเสี่ยงรวม
    </div>
    """, unsafe_allow_html=True)

    section("Algorithm (Heart Disease)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-card'>
            <b style='color:#E69D45;'> Random Forest</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 200, max_depth = 8</li>
                <li>min_samples_split = 4</li>
            </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='info-card'>
            <b style='color:#4a9e78;'> Gradient Boosting</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>n_estimators = 150, lr = 0.1</li>
                <li>max_depth = 4, subsample = 0.8</li>
            </ul></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='info-card'>
            <b style='color:#f59e0b;'> Logistic Regression</b>
            <ul style='color:#9aaba8; font-size:0.82rem; margin:8px 0 0; padding-left:18px;'>
                <li>C = 1.0, max_iter = 1000</li>
                <li>Soft voting weight = 1</li>
            </ul></div>""", unsafe_allow_html=True)

    section("Result (Heart Disease)")
    with st.spinner("กำลังเทรนโมเดล Heart Disease ML... ⏳"):
        res_h = train_heart_ml()

    metric_row([
        ("Accuracy", f"{res_h['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res_h['auc']:.4f}",      None),
        ("Precision (class 1)", f"{res_h['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (class 1)",    f"{res_h['cr']['1']['recall']*100:.1f}%",    None),
    ])
    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res_h["cm"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res_h["fpr"], res_h["tpr"], res_h["auc"]), use_container_width=True)
    st.plotly_chart(
        feature_importance_fig(res_h["feature_names"], res_h["feature_importances"],
                               "Feature Importances — Heart Disease (RF + GB avg)"),
        use_container_width=True)

# PAGE: COMBINED NN (Titanic + Heart Disease)

def page_combined_nn():
    st.markdown("<h1> Neural Network Models</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9aaba8;'>Deep Learning บน 2 datasets — Titanic Survival & Heart Disease</p>", unsafe_allow_html=True)

    # TITANIC
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin:32px 0 16px; padding:16px 20px; background:linear-gradient(135deg,#354044,#2e3a3e);
                border:1px solid #3d5055; border-left:4px solid #D45769; border-radius:12px;'>
        <span style='font-size:2rem;'></span>
        <div>
            <h2 style='margin:0; color:#D4CFC9; font-size:1.4rem;'>Titanic — Neural Network</h2>
            <p style='margin:2px 0 0; color:#9aaba8; font-size:0.82rem;'>
                5-layer Sequential (256→128→64→32→1) · L2 Regularization · Dropout
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("Architecture (Titanic)")
    st.markdown("""
    <div class='info-card'>
        <div style='font-family: "JetBrains Mono", monospace; font-size:0.82rem; color:#E69D45; line-height:1.8;'>
            Input(shape=(N,))<br>
            → Dense(256, relu, L2=1e-4) → BatchNorm → Dropout(0.3)<br>
            → Dense(128, relu, L2=1e-4) → BatchNorm → Dropout(0.2)<br>
            → Dense(64, relu, L2=1e-4) → Dropout(0.1)<br>
            → Dense(32, relu) → Dense(1, sigmoid)
        </div>
        <hr style='border-color:#3d5055; margin:12px 0;'>
        <span style='color:#9aaba8; font-size:0.82rem;'>
            Adam(lr=0.0005) · BinaryCrossentropy · 100 epochs · batch_size=16 · val_split=0.2
        </span>
    </div>
    """, unsafe_allow_html=True)

    section("Result (Titanic)")
    with st.spinner("กำลังเทรน Titanic Neural Network... ⏳"):
        res_t = train_titanic_nn()

    metric_row([
        ("Accuracy", f"{res_t['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res_t['auc']:.4f}",      None),
        ("Precision (survived)", f"{res_t['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (survived)",    f"{res_t['cr']['1']['recall']*100:.1f}%",    None),
    ])
    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res_t["cm"], ["Not Survived","Survived"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res_t["fpr"], res_t["tpr"], res_t["auc"]), use_container_width=True)

    section("Training History (Titanic)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(training_history_fig(res_t["history"], "accuracy"), use_container_width=True)
    c2.plotly_chart(training_history_fig(res_t["history"], "loss"), use_container_width=True)

    st.markdown(f"""
    <div class='info-card'>
        {badge("L2 Regularization", "#4a9e78")} เพิ่ม penalty λ·Σw² ใน loss — ทำให้ weights เล็กลง ลด overfitting &nbsp;
        {badge("Dropout")} ปิด neuron สุ่มระหว่าง train — ป้องกันการ memorize ข้อมูล
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # HEART DISEASE
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin:32px 0 16px; padding:16px 20px; background:linear-gradient(135deg,#354044,#2e3a3e);
                border:1px solid #3d5055; border-left:4px solid #D45769; border-radius:12px;'>
        <span style='font-size:2rem;'>️</span>
        <div>
            <h2 style='margin:0; color:#D4CFC9; font-size:1.4rem;'>Heart Disease — Neural Network</h2>
            <p style='margin:2px 0 0; color:#9aaba8; font-size:0.82rem;'>
                4-layer Sequential (128→64→32→1) · BatchNorm · EarlyStopping · ReduceLROnPlateau
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("Architecture (Heart Disease)")
    st.markdown("""
    <div class='info-card'>
        <div style='font-family: "JetBrains Mono", monospace; font-size:0.82rem; color:#E69D45; line-height:1.8;'>
            Input(shape=(17,))<br>
            → Dense(128, relu) → BatchNorm → Dropout(0.3)<br>
            → Dense(64, relu)  → BatchNorm → Dropout(0.2)<br>
            → Dense(32, relu)  → Dropout(0.1)<br>
            → Dense(1, sigmoid)
        </div>
        <hr style='border-color:#3d5055; margin:12px 0;'>
        <span style='color:#9aaba8; font-size:0.82rem;'>
            Adam(lr=0.001) · BinaryCrossentropy · AUC metric · EarlyStopping(patience=15) · ReduceLROnPlateau(patience=7)
        </span>
    </div>
    """, unsafe_allow_html=True)

    section("Result (Heart Disease)")
    with st.spinner("กำลังเทรน Heart Disease Neural Network... ⏳"):
        res_h = train_heart_nn()

    metric_row([
        ("Accuracy", f"{res_h['acc']*100:.2f}%", None),
        ("AUC-ROC",  f"{res_h['auc']:.4f}",      None),
        ("Precision (class 1)", f"{res_h['cr']['1']['precision']*100:.1f}%", None),
        ("Recall (class 1)",    f"{res_h['cr']['1']['recall']*100:.1f}%",    None),
    ])
    c1, c2 = st.columns(2)
    c1.plotly_chart(confusion_matrix_fig(res_h["cm"]), use_container_width=True)
    c2.plotly_chart(roc_curve_fig(res_h["fpr"], res_h["tpr"], res_h["auc"]), use_container_width=True)

    section("Training History (Heart Disease)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(training_history_fig(res_h["history"], "accuracy"), use_container_width=True)
    c2.plotly_chart(training_history_fig(res_h["history"], "loss"), use_container_width=True)

    st.markdown(f"""
    <div class='info-card'>
        {badge("EarlyStopping", "#4a9e78")} หยุดเทรนเมื่อ val_auc ไม่ดีขึ้น 15 epochs ติดต่อกัน — ประหยัดเวลาและลด overfit<br><br>
        {badge("ReduceLROnPlateau")} ลด learning rate ครึ่งหนึ่งเมื่อ val_loss ไม่ดีขึ้น 7 epochs — ช่วยให้ converge ดีขึ้น<br><br>
        {badge("BatchNormalization")} normalize activation ระหว่าง layers — ทำให้ gradient ไหลได้ดี เทรนเร็วขึ้น
    </div>
    """, unsafe_allow_html=True)
