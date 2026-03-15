import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import os
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.resnet_v2    import preprocess_input as resnet_v2_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Malaria Detection — Grad-CAM & ILCAN",
    page_icon="🔬",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding: 2rem 3rem; }

    .title-box {
        background: linear-gradient(135deg, #1a1f35 0%, #2d3561 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #3d4f8a;
    }
    .title-box h1 { color: #c9d1f5; font-size: 1.9rem; margin: 0; font-weight: 700; }
    .title-box p  { color: #7b8ec8; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

    .pred-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .panel-title { color: #e6edf3; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.2rem; }
    .panel-sub   { color: #8b949e; font-size: 0.72rem; margin-bottom: 0.5rem; }
    .legend-row  { display: flex; align-items: center; gap: 0.5rem;
                   font-size: 0.78rem; color: #e6edf3; margin-bottom: 0.25rem; }
    .legend-dot  { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS — from notebook
# ══════════════════════════════════════════════════════════════
CLASS_NAMES = ['Parasitized', 'Uninfected']

MODEL_INFO = {
    "ResNet-50": {
        "path"      : "resnet50_malaria.keras",
        "preprocess": "resnet50",
        "size"      : (128, 128),
        "last_conv" : "conv5_block3_out",
        "color"     : "#3fb950",
    },
    "VGG-19": {
        "path"      : "vgg19_malaria.keras",
        "preprocess": "rescale",
        "size"      : (224, 224),
        "last_conv" : "block5_conv4",
        "color"     : "#f78166",
    },
    "MobileNetV2": {
        "path"      : "mobilenet_malaria.keras",
        "preprocess": "mobilenet",
        "size"      : (224, 224),
        "last_conv" : "out_relu",
        "color"     : "#58a6ff",
    },
}

# ══════════════════════════════════════════════════════════════
# PREPROCESSING — from notebook cells 4, 6, 7
# ══════════════════════════════════════════════════════════════
def preprocess_image(img_rgb, model_name):
    info = MODEL_INFO[model_name]
    img_resized = cv2.resize(img_rgb, info["size"])
    img_array   = np.expand_dims(img_resized, axis=0).astype(np.float32)
    if info["preprocess"] == "resnet50":
        return resnet50_preprocess(img_array)
    elif info["preprocess"] == "mobilenet":
        return mobilenet_preprocess(img_array)
    else:                                          # rescale — VGG-19
        return img_array / 255.0

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_cached(model_name):
    path = MODEL_INFO[model_name]["path"]
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        return model, None
    except Exception as e:
        return None, str(e)

# ══════════════════════════════════════════════════════════════
# GRAD-CAM — from notebook cell 15
# ══════════════════════════════════════════════════════════════
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        if isinstance(predictions,  (list, tuple)): predictions  = predictions[0]
        if isinstance(conv_outputs, (list, tuple)): conv_outputs = conv_outputs[0]
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# ══════════════════════════════════════════════════════════════
#  Grad Cam— from notebook cell 18
# ══════════════════════════════════════════════════════════════
def make_ilcan_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_outputs, predictions = grad_model(img_tensor)
            if isinstance(predictions,  (list, tuple)): predictions  = predictions[0]
            if isinstance(conv_outputs, (list, tuple)): conv_outputs = conv_outputs[0]
            tape1.watch(conv_outputs)
            tape2.watch(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads  = tape1.gradient(class_channel, conv_outputs)
    grads2 = tape2.gradient(grads, conv_outputs)

    conv_outputs = conv_outputs[0]
    grads        = grads[0]
    grads2       = grads2[0]
    grads3       = grads * grads2
    sum_activations = tf.reduce_sum(conv_outputs, axis=(0, 1))
    denominator     = 2.0 * grads2 + (sum_activations * grads3)
    denominator     = tf.where(tf.abs(denominator) > 1e-8, denominator, tf.ones_like(denominator))
    alphas  = grads2 / denominator
    weights = tf.reduce_sum(alphas * tf.maximum(grads, 0.0), axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# ══════════════════════════════════════════════════════════════
# SUPERIMPOSE — from notebook cell 21
# ══════════════════════════════════════════════════════════════
def superimpose_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    jet             = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet             = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(
        original_img.astype(np.float32), 1 - alpha,
        jet.astype(np.float32),          alpha, 0
    ).astype(np.uint8)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="title-box">
    <h1>🔬 Malaria Detection — Grad-CAM & Modified Grad-CAM++</h1>
    <p>Modified Grad-CAM++ with True 2nd Derivatives ·
       ResNet-50 · VGG-19 · MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_name = st.selectbox("Select Model", list(MODEL_INFO.keys()), index=2)
    alpha      = st.slider("Heatmap Opacity", 0.1, 0.8, 0.4, 0.05)

    st.markdown("---")
    st.markdown("### 📐 Model Input Size")
    for name, info in MODEL_INFO.items():
        selected = "→ " if name == model_name else "   "
        color    = info["color"] if name == model_name else "#8b949e"
        st.markdown(
            f'<div style="color:{color}; font-size:0.83rem; padding:2px 0">'
            f'{selected}<b>{name}</b>: {info["size"][0]}×{info["size"][1]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🎨 Heatmap Legend")
    for color, label in [
        ("#FF0000", "Red    — Highest"),
        ("#FFFF00", "Yellow — High"),
        ("#00FF00", "Green  — Medium"),
        ("#00FFFF", "Cyan   — Low"),
        ("#0000FF", "Blue   — Lowest"),
    ]:
        st.markdown(
            f'<div class="legend-row">'
            f'<div class="legend-dot" style="background:{color}"></div>{label}'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📄 Methods")
    st.markdown("""
<div style="font-size:0.78rem; color:#8b949e; line-height:1.6">
<b style="color:#e6edf3">Grad-CAM</b><br>
Global avg pool of gradients → channel weights<br><br>
<b style="color:#e6edf3">Grad-CAM++</b><br>
True 2nd order derivatives via nested GradientTapes
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload Cell Image",
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
)

if uploaded_file is None:
    st.markdown("""
    <div style="background:#161b22; border:1px dashed #30363d; border-radius:12px;
                padding:3rem; text-align:center; margin-top:1rem">
        <div style="font-size:3rem">🔬</div>
        <div style="color:#58a6ff; font-size:1.1rem; font-weight:600; margin:0.5rem 0">
            Upload a blood smear cell image to begin</div>
        <div style="color:#8b949e; font-size:0.85rem">
            Parasitized or Uninfected · JPG / PNG</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
# READ + PROCESS
# ══════════════════════════════════════════════════════════════
file_bytes   = np.frombuffer(uploaded_file.read(), np.uint8)
img_bgr      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

with st.spinner(f"Loading {model_name}..."):
    model, err = load_model_cached(model_name)

if err:
    st.error(f"⚠️ {err}")
    st.stop()

preprocessed = preprocess_image(img_rgb, model_name)
preds        = model.predict(preprocessed, verbose=0)
pred_idx     = int(np.argmax(preds[0]))
confidence   = float(preds[0][pred_idx])
class_label  = CLASS_NAMES[pred_idx]
model_color  = MODEL_INFO[model_name]["color"]
last_conv    = MODEL_INFO[model_name]["last_conv"]

with st.spinner("Generating Grad-CAM heatmap..."):
    hm_gradcam = make_gradcam_heatmap(preprocessed, model, last_conv, pred_idx)

with st.spinner("Generating Modified Grad-CAM++ heatmap..."):
    hm_ilcan = make_ilcan_heatmap(preprocessed, model, last_conv, pred_idx)

overlay_gradcam = superimpose_heatmap(img_rgb, hm_gradcam, alpha)
overlay_ilcan   = superimpose_heatmap(img_rgb, hm_ilcan,   alpha)

# ══════════════════════════════════════════════════════════════
# PREDICTION CARD
# ══════════════════════════════════════════════════════════════
st.markdown("---")
col_pred, col_model, col_top = st.columns([2, 1, 1])

with col_pred:
    emoji = "🦟" if class_label == "Parasitized" else "✅"
    st.markdown(f"""
    <div class="pred-card">
        <p style="color:{model_color}; font-size:1.2rem; font-weight:700; margin:0">
            {emoji} {class_label}
        </p>
        <p style="color:#8b949e; font-size:0.85rem; margin:0.3rem 0 0.6rem 0">
            Predicted by {model_name}
        </p>
    </div>
    """, unsafe_allow_html=True)
    pct = confidence * 100
    st.markdown(f"""
    <div style="margin-top:-0.3rem">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px">
            <span style="color:#8b949e; font-size:0.75rem">Confidence</span>
            <span style="color:{model_color}; font-weight:700; font-size:0.85rem">{pct:.1f}%</span>
        </div>
        <div style="background:#21262d; border-radius:4px; height:8px; overflow:hidden">
            <div style="background:{model_color}; width:{pct}%; height:100%; border-radius:4px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_model:
    st.markdown(f"""
    <div class="pred-card" style="text-align:center">
        <p style="color:#8b949e; font-size:0.72rem; margin:0; font-weight:600; text-transform:uppercase">Model</p>
        <p style="color:{model_color}; font-size:1.2rem; font-weight:700; margin:0.2rem 0">{model_name}</p>
        <p style="color:#8b949e; font-size:0.78rem; margin:0">
            Input: {MODEL_INFO[model_name]["size"][0]}×{MODEL_INFO[model_name]["size"][1]}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_top:
    top2 = np.argsort(preds[0])[::-1][:2]
    st.markdown('<div class="pred-card">', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e; font-size:0.72rem; font-weight:600; text-transform:uppercase; margin:0 0 0.4rem 0">Top Predictions</p>', unsafe_allow_html=True)
    for rank, idx in enumerate(top2):
        c   = model_color if rank == 0 else "#8b949e"
        pfx = "→ " if rank == 0 else "   "
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; margin-bottom:3px">'
            f'<span style="color:{c}; font-size:0.8rem">{pfx}{CLASS_NAMES[idx]}</span>'
            f'<span style="color:{c}; font-size:0.8rem; font-weight:700">{preds[0][idx]*100:.1f}%</span>'
            f'</div>', unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 3-PANEL — mirrors plot_model_results() from notebook cell 22
# ══════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3)
panel = "background:#161b22; border:1px solid #30363d; border-radius:10px; padding:0.8rem; text-align:center"

with col1:
    st.markdown(f'<div style="{panel}">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Original Image</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-sub">Input cell image</p>', unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div style="{panel}">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Grad-CAM</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-sub">Global avg pool of gradients</p>', unsafe_allow_html=True)
    st.image(overlay_gradcam, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div style="{panel}">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">Modified Grad-CAM++</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-sub">True 2nd derivative</p>', unsafe_allow_html=True)
    st.image(overlay_ilcan, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DIFFERENCE MAP
# ══════════════════════════════════════════════════════════════
with st.expander("📊 Difference Map (Modified Grad-CAM++ − Grad-CAM)"):
    import matplotlib.pyplot as plt
    h, w = img_rgb.shape[:2]
    h_gc = cv2.resize(hm_gradcam, (w, h))
    h_il = cv2.resize(hm_ilcan,   (w, h))
    diff = h_il - h_gc

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#161b22")
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", color="white", fontsize=10)
    axes[1].imshow(overlay_ilcan)
    axes[1].set_title("ILCAN overlay", color="white", fontsize=10)
    im = axes[2].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("ILCAN − Grad-CAM", color="white", fontsize=10)
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#161b22")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
    <div style="font-size:0.78rem; color:#8b949e; margin-top:0.4rem">
    🔴 Red = Modified Grad-CAM++ activates more strongly &nbsp;|&nbsp;
    🔵 Blue = Grad-CAM activates more strongly &nbsp;|&nbsp;
    ⚪ White = both agree
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.75rem; padding:0.4rem 0">
    ILCAN — Malaria Cell Detection · Grad-CAM & Modified Grad-CAM++ ·
    Arabian Journal for Science and Engineering (2022) 47:2305–2314 ·
    NIH Malaria Cell Images Dataset
</div>
""", unsafe_allow_html=True)
