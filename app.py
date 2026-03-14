import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# ==============================
# STREAMLIT PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Crop Stress Detection",
    page_icon="🌿",
    layout="centered",
)

# ==============================
# LOAD MODEL AND CLASS NAMES
# ==============================
@st.cache_resource
def load_disease_model():
    model = load_model("plant_disease_final_model.keras")
    return model

@st.cache_resource
def load_class_names():
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return class_names

# Small helper so UI can show class labels (disease names)
class_names = load_class_names()
HISTORY_FILE = "analysis_history.csv"

# ==============================
# HELPER FUNCTIONS
# ==============================
def extract_leaf_name(disease_name: str) -> str:
    parts = disease_name.split("__")
    if len(parts) > 0:
        return parts[0].replace("_", " ")
    return "Unknown"

def get_stress_level(disease_name: str, confidence: float):
    disease_lower = disease_name.lower()
    disease_part = disease_lower.split("__")[-1] if "__" in disease_lower else disease_lower

    if "healthy" in disease_lower:
        return {
            "level": "🟢 NO STRESS",
            "color": "green",
            "emoji": "✅",
            "action": "No action needed - Plant is healthy",
            "severity": 0,
        }

    severe_diseases = ["scab", "blight", "rust", "mildew", "rot", "canker", "wilt", "late_blight"]
    moderate_diseases = ["spot", "mosaic", "curl", "virus", "bacterial", "early_blight"]
    mild_diseases = ["leaf", "yellow", "streak"]

    if any(d in disease_lower for d in severe_diseases):
        base_severity = 3
        category = "SEVERE"
        emoji = "🔴"
        action = "⚠️ URGENT: Immediate treatment required! Apply fungicide/pesticide."
    elif any(d in disease_lower for d in moderate_diseases):
        base_severity = 2
        category = "MODERATE"
        emoji = "🟠"
        action = "⚕️ Start treatment within 2-3 days. Remove affected leaves."
    elif any(d in disease_lower for d in mild_diseases):
        base_severity = 1
        category = "MILD"
        emoji = "🟡"
        action = "🔍 Monitor plant daily. Consider organic treatment."
    else:
        base_severity = 1
        category = "MILD"
        emoji = "🟡"
        action = "🔍 Monitor closely. Consult if symptoms worsen."

    if confidence > 0.9:
        stress_level = f"{emoji} HIGH {category} STRESS"
        if base_severity == 3:
            action = "⚠️ CRITICAL: Immediate professional consultation required!"
    elif confidence > 0.7:
        stress_level = f"{emoji} MODERATE {category} STRESS"
    else:
        stress_level = f"{emoji} LOW {category} STRESS"

    return {
        "level": stress_level,
        "action": action,
        "emoji": emoji,
    }

def predict_leaf_disease_from_pil(pil_img: Image.Image):
    # Lazy-load model and class names so UI can render even if loading is slow
    model = load_disease_model()
    class_names = load_class_names()

    img = pil_img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(preds)
    disease_name = class_names[top_idx]
    confidence = preds[top_idx]

    leaf_name = extract_leaf_name(disease_name)
    stress_info = get_stress_level(disease_name, confidence)

    return {
        "leaf_name": leaf_name,
        "disease": disease_name,
        "confidence": float(confidence),
        "stress_level": stress_info["level"],
        "action": stress_info["action"],
        "emoji": stress_info["emoji"],
        "raw_probs": preds,
    }


def log_analysis_to_history(
    disease_name: str,
    stress_level: str,
    confidence: float,
    moisture: int,
    temperature: int,
    stress_score: int,
) -> None:
    """Append one analysis record to a simple CSV history file."""
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": disease_name,
        "stress_level": stress_level,
        "confidence": round(confidence, 4),
        "soil_moisture": moisture,
        "temperature_c": temperature,
        "stress_score": stress_score,
    }

    if os.path.exists(HISTORY_FILE):
        df_existing = pd.read_csv(HISTORY_FILE)
        df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
    else:
        df_new = pd.DataFrame([record])

    df_new.to_csv(HISTORY_FILE, index=False)

# ==============================
# STREAMLIT UI
# ==============================
st.title("🌿 Crop Leaf Stress & Disease Detection")
st.write(
    "Upload a clear photo of a single leaf. The model will estimate the "
    "**disease** and **stress level**, and you can explore how stress changes "
    "across crop stages."
)

st.sidebar.header("How to use")
st.sidebar.markdown(
    "- Upload a single leaf image (JPG/PNG)\n"
    "- Tap the leaf preview to highlight stress level\n"
    "- Swipe/scroll through stages to see growth vs stress trends\n"
    "- Use the sliders to explore how moisture / temperature affect stress"
)

# Global CSS for subtle farm background, water / soil animations, leaf emojis and ripple
st.markdown(
    """
    <style>
    /* Main app background – brighter sky and fields */
    .stApp {
        background:
          /* sky */
          radial-gradient(circle at 15% 5%, rgba(255, 239, 170, 0.9) 0, transparent 35%),
          linear-gradient(180deg, rgba(178, 216, 245, 1) 0, rgba(216, 237, 252, 0.96) 32%, rgba(234, 246, 236, 0.98) 55%, #ffffff 78%),
          /* distant hills */
          radial-gradient(circle at 5% 86%, rgba(129, 183, 117, 0.85) 0, transparent 55%),
          radial-gradient(circle at 55% 92%, rgba(98, 160, 107, 0.85) 0, transparent 58%),
          radial-gradient(circle at 105% 88%, rgba(156, 196, 130, 0.8) 0, transparent 48%);
        background-attachment: fixed;
    }
    /* Farm‑related emoji accents floating in background */
    .farm-emoji-layer {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.35;
    }
    .farm-emoji-layer span {
        position: absolute;
        font-size: 1.8rem;
        filter: drop-shadow(0 2px 3px rgba(0,0,0,0.15));
        animation: farm-float 10s ease-in-out infinite alternate;
    }
    .farm-emoji-layer span:nth-child(1) { top: 10%; left: 6%;  animation-delay: 0s; }
    .farm-emoji-layer span:nth-child(2) { top: 16%; right: 6%; animation-delay: 1.0s; }
    .farm-emoji-layer span:nth-child(3) { bottom: 20%; left: 10%; animation-delay: 1.8s; }
    .farm-emoji-layer span:nth-child(4) { bottom: 18%; right: 12%; animation-delay: 2.5s; }
    .farm-emoji-layer span:nth-child(5) { top: 26%; left: 36%; animation-delay: 3.1s; }
    .farm-emoji-layer span:nth-child(6) { top: 32%; right: 32%; animation-delay: 3.7s; }
    .farm-emoji-layer span:nth-child(7) { bottom: 26%; left: 40%; animation-delay: 4.3s; }
    .farm-emoji-layer span:nth-child(8) { bottom: 30%; right: 40%; animation-delay: 4.9s; }

    @keyframes farm-float {
        from { transform: translateY(0px); }
        to   { transform: translateY(10px); }
    }
    .leaf-upload-card {
        padding: 1.2rem;
        border-radius: 1.2rem;
        background: radial-gradient(circle at top, #e3f4e8 0, #f7fff9 40%, #ffffff 100%);
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(163, 196, 163, 0.45);
        transition: box-shadow 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
    }
    .leaf-upload-card:hover {
        box-shadow: 0 10px 26px rgba(0,0,0,0.08);
        border-color: rgba(113, 168, 113, 0.8);
        transform: translateY(-2px);
    }
    .leaf-emoji-row {
        display: flex;
        justify-content: center;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }
    .leaf-emoji {
        font-size: 1.6rem;
        cursor: pointer;
        transition: transform 0.15s ease, opacity 0.15s ease;
        animation: gentle-pop 2.2s ease-in-out infinite;
    }
    .leaf-emoji:hover {
        transform: translateY(-4px) scale(1.08);
    }
    @keyframes gentle-pop {
        0% { transform: translateY(-18px); opacity: 0; }
        30% { transform: translateY(0); opacity: 1; }
        100% { transform: translateY(4px); opacity: 0.8; }
    }
    .water-dot, .soil-dot {
        position: absolute;
        border-radius: 999px;
        opacity: 0.3;
        animation: float 6s ease-in-out infinite alternate;
    }
    .water-dot {
        background: rgba(135, 206, 250, 0.6);
    }
    .soil-dot {
        background: rgba(139, 69, 19, 0.35);
    }
    @keyframes float {
        from { transform: translateY(0px); }
        to   { transform: translateY(-14px); }
    }
    .leaf-preview {
        border-radius: 1rem;
        overflow: hidden;
        position: relative;
        cursor: pointer;
    }
    .leaf-preview::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: radial-gradient(circle at center, rgba(255,255,255,0.0) 0, rgba(0,0,0,0) 45%);
        opacity: 0;
        transition: opacity 0.25s ease;
        pointer-events: none;
    }
    .leaf-preview.leaf-healthy::after {
        background: radial-gradient(circle at center, rgba(144,238,144,0.35) 0, rgba(0,0,0,0) 55%);
    }
    .leaf-preview.leaf-moderate::after {
        background: radial-gradient(circle at center, rgba(255,193,7,0.35) 0, rgba(0,0,0,0) 55%);
    }
    .leaf-preview.leaf-severe::after {
        background: radial-gradient(circle at center, rgba(220,53,69,0.40) 0, rgba(0,0,0,0) 55%);
    }
    .leaf-preview.ripple-active::after {
        opacity: 1;
    }
    .stage-strip {
        display: flex;
        gap: 0.75rem;
        overflow-x: auto;
        padding-bottom: 0.4rem;
    }
    .stage-card {
        min-width: 140px;
        border-radius: 0.9rem;
        padding: 0.7rem 0.9rem;
        background: #f9fbf7;
        border: 1px solid #dde7d4;
        cursor: pointer;
        transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
    }
    .stage-card.active {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        border-color: #4caf50;
        background: #f1fff4;
    }
    .stage-label {
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .stage-sub {
        font-size: 0.78rem;
        color: #617067;
    }
    </style>
    <div class="farm-emoji-layer">
      <span>🌾</span>
      <span>🚜</span>
      <span>🌱</span>
      <span>🌻</span>
      <span>👨‍🌾</span>
      <span>🌽</span>
      <span>🍀</span>
      <span>🏡</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "ripple" not in st.session_state:
    st.session_state.ripple = False
if "selected_stage" not in st.session_state:
    st.session_state.selected_stage = "Early growth"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "moisture" not in st.session_state:
    st.session_state.moisture = 50
if "temperature" not in st.session_state:
    st.session_state.temperature = 25
if "visited_stages" not in st.session_state:
    st.session_state.visited_stages = []

main_col, side_col = st.columns([2, 1])

with main_col:
    st.markdown("#### 1. Upload a leaf image")
    with st.container():
        # Leaf emojis gently popping down
        emoji_html = """
        <div class="leaf-emoji-row">
          <span class="leaf-emoji">🍃</span>
          <span class="leaf-emoji">🌿</span>
          <span class="leaf-emoji">🍀</span>
          <span class="leaf-emoji">🌱</span>
        </div>
        """
        st.markdown(emoji_html, unsafe_allow_html=True)

        st.markdown(
            """
            <div class="leaf-upload-card">
              <div class="water-dot" style="width:28px;height:28px;top:10px;left:12%;"></div>
              <div class="water-dot" style="width:20px;height:20px;top:40px;right:18%;animation-delay:0.3s;"></div>
              <div class="soil-dot" style="width:18px;height:18px;bottom:16px;left:18%;animation-delay:0.6s;"></div>
              <div class="soil-dot" style="width:24px;height:24px;bottom:8px;right:10%;animation-delay:1s;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clickable leaf buttons (no balloons, just visual)
        leaf_cols = st.columns(4)
        leaf_emojis_click = ["🍃", "🌿", "🍀", "🌱"]
        for col, emoji in zip(leaf_cols, leaf_emojis_click):
            with col:
                st.button(emoji, key=f"rain_{emoji}")

        uploaded_file = st.file_uploader(
            "Drop a leaf image here (or browse)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")

            if st.button("Analyze leaf", type="primary"):
                with st.spinner("Analyzing leaf image..."):
                    result = predict_leaf_disease_from_pil(pil_image)
                    st.session_state.last_result = result
                    st.session_state.ripple = True

    # Show analysis if we have a previous result (kept on same page)
    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        disease_clean = result["disease"].split("__")[-1].replace("_", " ")

        # Determine stress class for ripple color
        level_text = result["stress_level"].lower()
        if "no stress" in level_text or "healthy" in level_text:
            stress_class = "leaf-healthy"
        elif "moderate" in level_text:
            stress_class = "leaf-moderate"
        else:
            stress_class = "leaf-severe"

        ripple_class = "ripple-active" if st.session_state.ripple else ""

        st.markdown("#### 2. Tap the leaf to feel the stress level")
        st.markdown(
            f"""
            <div class="leaf-preview {stress_class} {ripple_class}" onclick="this.classList.add('ripple-active'); setTimeout(()=>this.classList.remove('ripple-active'), 400);">
            </div>
            """,
            unsafe_allow_html=True,
        )
        if uploaded_file is not None:
            st.image(pil_image, caption="Tap the leaf to highlight stress", use_column_width=True)

        st.markdown("#### 3. Leaf health result")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Leaf health:** {result['stress_level']}")
            st.markdown(f"**Disease or status:** {disease_clean}")
        with col_b:
            st.markdown(f"**Model confidence:** {result['confidence'] * 100:.1f}%")
            st.progress(min(max(result["confidence"], 0.0), 1.0))

        probs = result["raw_probs"]
        top3_idx = np.argsort(probs)[-3:][::-1]
        st.markdown("**Most likely problems**")
        for i in top3_idx:
            label = class_names[i].split("__")[-1].replace("_", " ")
            st.markdown(f"- {label}: {probs[i] * 100:.1f}%")

        st.markdown("#### 4. Explore conditions")
        col_m, col_t = st.columns(2)
        with col_m:
            st.session_state.moisture = st.slider(
                "Soil moisture (0–100%)",
                0,
                100,
                st.session_state.moisture,
            )
        with col_t:
            st.session_state.temperature = st.slider(
                "Temperature (°C)",
                5,
                45,
                st.session_state.temperature,
            )

        # Simple synthetic indicator reacting to sliders
        moisture = st.session_state.moisture
        temp = st.session_state.temperature
        stress_score = 0
        if moisture < 30 or moisture > 80:
            stress_score += 1
        if temp < 15 or temp > 35:
            stress_score += 1

        # Alert levels and notifications
        if stress_score == 0:
            alert_label = "No action"
            alert_color = "✅"
            st.success("No action: Conditions look gentle. The crop is less likely to be under stress.")
        elif stress_score == 1:
            alert_label = "Watch closely"
            alert_color = "🟠"
            st.warning("Watch closely: One condition is outside the comfortable range. Keep an eye on the plants.")
        else:
            alert_label = "Act soon"
            alert_color = "🔴"
            st.error("Act soon: Moisture and temperature are both stressful. The crop may show strong stress signs.")

        st.markdown(f"**Alert level:** {alert_color} {alert_label}")

        # Save this analysis into simple history so farmers can review later
        log_analysis_to_history(
            disease_name=disease_clean,
            stress_level=result["stress_level"],
            confidence=result["confidence"],
            moisture=moisture,
            temperature=temp,
            stress_score=stress_score,
        )

        st.markdown("#### 5. Your recent checks")
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            # Show last 5 records
            st.write("Recent leaf checks (latest first):")
            st.dataframe(history_df.iloc[::-1].head(5), use_container_width=True)

            # Count “Act soon” alerts in last 7 days (simple time window)
            try:
                history_df["timestamp_dt"] = pd.to_datetime(history_df["timestamp"])
                last_7_days = history_df[
                    history_df["timestamp_dt"]
                    >= (pd.Timestamp.now() - pd.Timedelta(days=7))
                ]
                high_alerts = last_7_days[last_7_days["stress_score"] >= 2]
                count_high = len(high_alerts)
                if count_high > 0:
                    st.warning(
                        f"In the last 7 days, {count_high} leaf checks were at **Act soon** level. "
                        "Consider checking those fields more often or taking action."
                    )
                else:
                    st.info("In the last 7 days, no checks reached the Act soon level.")
            except Exception:
                # If parsing ever fails, just skip the summary
                pass
        else:
            st.write("No history yet. After a few checks, you will see your past results here.")

with side_col:
    st.markdown("#### Crop stage timeline")
    stages = [
        ("Seedling", "Delicate roots, focus on moisture."),
        ("Early growth", "Leaf expansion; watch for early spots."),
        ("Mid growth", "Canopy filling; stress impacts yield most."),
        ("Pre-harvest", "Quality phase; monitor diseases closely."),
    ]
    stage_names = [name for name, _ in stages]

    # Journey progress at the top
    visited = st.session_state.visited_stages
    journey_progress = len(set(visited)) / len(stage_names) if stage_names else 0
    st.markdown("Progress through crop stages")
    st.progress(journey_progress)

    st.write("Tap on each stage below to gently open more information. All details stay on this page.")

    # Inline expanding details for each stage
    for name, desc in stages:
        is_open = st.checkbox(f"{name} stage", key=f"stage_{name}")
        if is_open and name not in st.session_state.visited_stages:
            st.session_state.visited_stages.append(name)

        card_class = "active" if is_open else ""
        stage_html = f"""
        <div class="stage-card {card_class}" title="{desc}">
          <div class="stage-label">{name}</div>
          <div class="stage-sub">{desc}</div>
        </div>
        """
        st.markdown(stage_html, unsafe_allow_html=True)

        if is_open:
            if name == "Seedling":
                st.write(
                    "Seedling stage is when the plant is very young. Roots are small and need gentle, steady moisture. "
                    "Avoid waterlogging and protect from strong sun or wind."
                )
            elif name == "Early growth":
                st.write(
                    "In early growth, new leaves appear quickly. Check leaves often for small spots or color changes, "
                    "so you can react before problems spread."
                )
            elif name == "Mid growth":
                st.write(
                    "Mid growth is when the plant canopy is filling the field. Good care here strongly protects yield. "
                    "Watch for stress from lack of water, heat, or disease."
                )
            else:  # Pre-harvest
                st.write(
                    "Near harvest, the crop is almost ready. Focus on keeping leaves and fruits healthy so quality stays high. "
                    "Try to avoid new infections or sudden stress."
                )

if st.session_state.ripple:
    # Reset ripple flag so effect triggers again only after next analysis
    st.session_state.ripple = False