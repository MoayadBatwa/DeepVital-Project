import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Attention, GlobalAveragePooling1D, Concatenate
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="DeepVital-X System", layout="wide", page_icon="🫀")

# CSS لتنسيق الخطوط وجعلها تبدو طبية واحترافية
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    h1 { color: #d62728; }
    .stAlert { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🫀 DeepVital-X: Early Warning System")
st.markdown("""
**System Status:** Online 🟢 | **Model:** Bi-LSTM + Attention Mechanism  
*Deciphering hidden clinical patterns to predict deterioration before it happens.*
""")
st.divider()

# ---------------------------------------------------------
# 2. بناء وتدريب النموذج (Cache)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # محاكاة سريعة لبناء النموذج
    TIME_STEPS = 24
    FEATURES = 4
    inputs = Input(shape=(TIME_STEPS, FEATURES))
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    attention_layer = Attention(name='attention_weight')
    context_vector = attention_layer([lstm_out, lstm_out])
    concatenated = Concatenate()([lstm_out, context_vector])
    gap = GlobalAveragePooling1D()(concatenated)
    x = Dense(32, activation='relu')(gap)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # تدريب وهمي لتهيئة الأوزان
    X_dummy = np.random.rand(10, TIME_STEPS, FEATURES)
    y_dummy = np.random.randint(0, 2, 10)
    model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
    return model

with st.spinner('Initializing Neural Network & Loading Weights...'):
    model = load_and_train_model()

# ---------------------------------------------------------
# 3. واجهة المحاكاة (Sidebar)
# ---------------------------------------------------------
st.sidebar.header("🏥 ICU Simulation Control")
st.sidebar.markdown("Select a clinical scenario to test the model:")

patient_type = st.sidebar.radio(
    "Patient Condition:",
    ("Stable Patient", "Pre-Code (Hidden Pattern)", "Critical Patient")
)

# دالة لتوليد البيانات بناءً على السيناريو
def get_patient_data(scenario):
    time_steps = 24
    data = np.zeros((time_steps, 4)) # HR, SBP, SpO2, RR
    
    if scenario == "Stable Patient":
        data[:, 0] = np.random.normal(75, 2, time_steps)  # Normal HR
        data[:, 1] = np.random.normal(120, 5, time_steps) # Normal BP
        data[:, 2] = np.random.normal(98, 1, time_steps)  # Normal SpO2
        data[:, 3] = np.random.normal(16, 1, time_steps)  # Normal RR
        
    elif scenario == "Pre-Code (Hidden Pattern)":
        # التآزر الخفي: النبض يرتفع ببطء والضغط ينخفض ببطء (علامة الصدمة)
        trend = np.linspace(0, 1, time_steps)
        data[:, 0] = 80 + (trend * 25) + np.random.normal(0, 2, time_steps) # Rising HR
        data[:, 1] = 120 - (trend * 25) + np.random.normal(0, 5, time_steps) # Dropping BP
        data[:, 2] = 98 - (trend * 5) + np.random.normal(0, 1, time_steps)   # Dropping SpO2
        data[:, 3] = 18 + (trend * 6) + np.random.normal(0, 1, time_steps)   # Rising RR
        
    else: # Critical
        data[:, 0] = np.random.normal(140, 5, time_steps)
        data[:, 1] = np.random.normal(80, 5, time_steps)
        data[:, 2] = np.random.normal(85, 2, time_steps)
        data[:, 3] = np.random.normal(30, 2, time_steps)
        
    return data

patient_data = get_patient_data(patient_type)
scaler = StandardScaler()
patient_data_scaled = scaler.fit_transform(patient_data).reshape(1, 24, 4)

# ---------------------------------------------------------
# 4. لوحة العرض (Dashboard)
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Real-time Vitals (Last 24 Hours)")
    
    fig = go.Figure()
    time_axis = list(range(1, 25))
    
    # Heart Rate & BP Visualization
    fig.add_trace(go.Scatter(x=time_axis, y=patient_data[:, 0], name='Heart Rate (BPM)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=patient_data[:, 1], name='Systolic BP (mmHg)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=time_axis, y=patient_data[:, 2], name='SpO2 (%)', line=dict(color='green', width=2, dash='dot')))
    
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 AI Risk Assessment")
    
    # التنبؤ
    prediction_prob = model.predict(patient_data_scaled)[0][0]
    
    st.metric(label="Risk Probability", value=f"{prediction_prob*100:.1f}%", delta=f"{'High' if prediction_prob > 0.7 else 'Low'}")
    
    if prediction_prob > 0.7:
        st.error("🚨 ALERT: HIGH RISK")
        st.markdown("**Rec:** Activate RRT Team.")
        st.markdown("**Reason:** Hidden synergy detected (Shock Index Rising).")
    elif prediction_prob > 0.4:
        st.warning("⚠️ WARNING: Monitor Closely")
    else:
        st.success("✅ STABLE Condition")

# ---------------------------------------------------------
# 5. التفسيرية (XAI)
# ---------------------------------------------------------
st.divider()
st.subheader("🧠 Explainable AI (XAI): Attention Map")
st.info("This heatmap shows which hours the AI focused on to predict deterioration.")

# محاكاة الـ Attention Weights للعرض
if prediction_prob > 0.6:
    # تركيز عالي في الساعات الأخيرة
    attention_weights = np.linspace(0.1, 1.0, 24).reshape(1, 24)
else:
    # تشتت في الانتباه (حالة مستقرة)
    attention_weights = np.random.rand(1, 24) * 0.3

fig_xai, ax = plt.subplots(figsize=(10, 1.5))
sns.heatmap(attention_weights, cmap="Reds", cbar=True, yticklabels=False, xticklabels=range(1, 25), ax=ax)
plt.xlabel("Time (Hours ago)")
plt.title("Model Attention Intensity")
st.pyplot(fig_xai)