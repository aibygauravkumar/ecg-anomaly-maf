"""
app.py — Streamlit web interface for ECG anomaly detection

Run:
  streamlit run app.py

Three modes:
  1. Upload ECG Image  — upload a real ECG scan or printout
  2. Browse Test Data  — explore real beats from MIT-BIH test set
  3. Anomaly vs Classifier Demo — why unsupervised detection beats supervised classifiers
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from PIL import Image

from src.dataset          import load_raw, INPUT_DIM, CLASS_NAMES
from src.hybrid_model     import HybridECGModel
from src.signal_extractor import extract_signal_from_image, preprocess_for_model
from src.cnn_classifier   import load_cnn, cnn_predict, KNOWN_CLASSES, UNKNOWN_CLASS, CLASS_LABELS


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ECG Anomaly Detector",
    page_icon  = "🫀",
    layout     = "wide"
)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    config_path = Path("outputs/train_config.json")
    model_path  = Path("outputs/best_model.pt")
    eval_path   = Path("outputs/eval_results.json")

    if not all(p.exists() for p in [config_path, model_path, eval_path]):
        return None, None, None, None, None

    with open(config_path) as f:
        config = json.load(f)
    with open(eval_path) as f:
        results = json.load(f)

    model = HybridECGModel(
        input_len   = INPUT_DIM,
        latent_dim  = config.get("latent_dim", 64),
        n_layers    = config["n_layers"],
        hidden_dims = config["hidden_dims"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, results["threshold"], results["auroc"], config, results


@st.cache_resource
def load_test_data():
    X_train, y_train, X_test, y_test = load_raw()
    normal_mask = y_train == 0
    mean = X_train[normal_mask].mean(axis=0)
    std  = X_train[normal_mask].std(axis=0)
    return X_test, y_test, mean, std


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_signal(model, signal_norm: np.ndarray) -> float:
    x = torch.tensor(signal_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = model.log_prob(x).item()
    return float(np.clip(score, -1e6, 1e6))


def make_plot(signal: np.ndarray, color: str, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.plot(signal, color=color, linewidth=2)
    ax.set_xlim(0, len(signal) - 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def verdict_ui(score: float, threshold: float):
    is_anomaly = score < threshold
    c1, c2, c3 = st.columns(3)

    with c1:
        if is_anomaly:
            st.error("## ⚠️ ANOMALY DETECTED")
        else:
            st.success("## ✅ NORMAL")

    with c2:
        st.metric("Model Score  (log p(x))", f"{score:.2f}")
        st.caption("Higher = more normal")

    with c3:
        st.metric("Threshold", f"{threshold:.2f}")
        gap = abs(score - threshold)
        direction = "below" if is_anomaly else "above"
        st.caption(f"Score is {gap:.2f} points {direction} threshold")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫀 ECG Anomaly Detection — CNN + MAF")
st.markdown(
    "A hybrid deep learning model for **unsupervised cardiac anomaly detection**, "
    "built from scratch in PyTorch. "
    "A **1D CNN encoder** compresses each heartbeat into noise-robust features. "
    "A **Masked Autoregressive Flow (MAF)** learns the exact probability density of normal beats. "
    "Any heartbeat with low `log p(x)` is flagged as anomalous — "
    "no anomaly labels needed during training."
)

model, threshold, auroc, config, eval_results = load_model()

if model is None:
    st.error("No trained model found. Run `python main.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("AUROC",     f"{auroc:.4f}")
        st.metric("Recall",    f"{eval_results['recall']:.4f}")
    with c2:
        st.metric("F1 Score",  f"{eval_results['f1']:.4f}")
        st.metric("Precision", f"{eval_results['precision']:.4f}")

    st.divider()
    st.markdown("**Architecture**")
    latent_dim = config.get("latent_dim", 64)
    st.caption(f"CNN encoder : 187 → {latent_dim} features")
    st.caption(f"MAF layers  : {config['n_layers']}")
    st.caption(f"Hidden dims : {config['hidden_dims']}")
    st.caption(f"Parameters  : 2,926,720")
    st.caption(f"Threshold   : {threshold:.2f}")

    st.divider()
    st.markdown("**Dataset**")
    st.caption("MIT-BIH Arrhythmia Database")
    st.caption("87,554 training beats — normal only")
    st.caption("21,892 test beats — all 5 classes")
    st.caption(f"True Positives : {eval_results['true_positives']:,}")
    st.caption(f"False Negatives: {eval_results['false_negatives']:,}")

    st.divider()
    st.markdown("**Training**")
    st.caption(f"Stage 1: CNN autoencoder — {config.get('pretrain_epochs', 100)} epochs")
    st.caption(f"Stage 2: MAF density — {config.get('n_epochs', 400)} epochs")

    st.divider()
    st.markdown(
        "**How it works:**\n\n"
        "The CNN encoder compresses a noisy 187-point heartbeat into "
        f"{latent_dim} clean features, removing noise and baseline wander. "
        "The MAF then learns `p(features)` for normal beats only. "
        "At inference, `log p(features) < threshold` triggers an anomaly alert. "
        "No anomaly examples are ever needed during training."
    )


# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Choose input mode:",
    ["📤 Upload ECG Image", "🔬 Browse Test Examples", "⚡ Anomaly Detector vs Classifier"],
    horizontal=True
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: Upload ECG Image
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📤 Upload ECG Image":

    with st.expander("📋 Supported image types", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
**Automatically handled:**
- ✅ Full 12-lead ECG report (hospital printouts)
  → App detects and extracts the rhythm strip automatically
- ✅ Single lead strip (cropped or screenshot)
  → App processes the full image directly
- ✅ ECG paper with red/pink grid
  → Grid is removed automatically

**Best image quality:**
- Screenshot or scan (not a phone photo)
- 300 DPI or higher for scanned paper
""")
        with col_b:
            st.markdown("""
**May struggle with:**
- Very blurry or low-contrast images
- Heavily annotated images (lots of text over the trace)
- Angled phone photos (glare, distortion)

**Quick test:**
Upload any ECG image from `outputs/` to confirm the pipeline is working.

**Check the tabs below** after uploading to see exactly what was detected and extracted at each step.
""")

    sampling_rate = st.select_slider(
        "ECG sampling rate (Hz) — check your ECG machine specs",
        options=[125, 250, 300, 360, 500],
        value=300,
        help="MIT-BIH uses 360 Hz. Most modern ECG machines use 250–500 Hz."
    )
    st.caption("Full report detection, grid removal, and rhythm strip extraction happen automatically.")

    uploaded = st.file_uploader(
        "Upload your ECG image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)

        with st.spinner("Analysing ECG image..."):
            signal, debug = extract_signal_from_image(
                image, sampling_rate=sampling_rate
            )

        report_type = debug["report_type"]
        st.info(
            f"**Detected:** {'Full 12-lead ECG report' if report_type == 'full_report' else 'Single lead strip'} — "
            f"{'Automatically locating rhythm strip...' if report_type == 'full_report' else 'Processing full image.'}"
        )

        if report_type == "full_report":
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "1️⃣ Original",
                "2️⃣ Detected Regions",
                "3️⃣ Rhythm Strip",
                "4️⃣ Extracted Trace",
                "5️⃣ Final Beat"
            ])
        else:
            tab1, tab3, tab4, tab5 = st.tabs([
                "1️⃣ Original",
                "2️⃣ Cleaned Image",
                "3️⃣ Extracted Trace",
                "4️⃣ Final Beat"
            ])

        with tab1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        if report_type == "full_report":
            with tab2:
                st.image(
                    debug["annotated"],
                    caption="Detected regions — 🔴 Header  🟡 ECG leads  🟢 Rhythm strip (used for analysis)",
                    use_container_width=True
                )
                regions = debug["regions"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Header ends at row",   regions["header_end"])
                c2.metric("ECG grid rows",
                          f"{regions['ecg_start']} → {regions['ecg_end']}")
                c3.metric("Rhythm strip rows",
                          f"{regions['rhythm_start']} → {regions['rhythm_end']}")

            with tab3:
                st.image(
                    debug["rhythm_strip"],
                    caption="Cropped rhythm strip — Lead II (full width, single lead)",
                    use_container_width=True
                )
                st.image(
                    debug["cleaned"],
                    caption="After removing red grid background",
                    use_container_width=True
                )
        else:
            with tab3:
                st.image(
                    debug["cleaned"],
                    caption="After removing grid background",
                    use_container_width=True
                )

        with tab4:
            raw     = debug["raw_trace"]
            n_peaks = debug.get("n_peaks", 0)
            fig = make_plot(raw, "grey", f"Raw extracted trace — {n_peaks} heartbeats detected")
            st.pyplot(fig)
            plt.close()
            st.caption(
                f"{len(raw)} raw time steps extracted. "
                f"R-peak detection found {n_peaks} beats. "
                "The most representative single beat is selected next."
            )
            if n_peaks == 0:
                st.warning("No R-peaks detected. The extracted trace may not be a valid ECG signal.")
            elif n_peaks == 1:
                st.warning("Only 1 beat found — using it directly. More beats = better segmentation.")
            else:
                st.success(f"{n_peaks} beats detected — selecting the most representative one.")

        with tab5:
            looks_ok = signal.std() > 0.05

            if looks_ok:
                fig = make_plot(signal, "steelblue", "Final extracted beat — 187 time steps")
                st.pyplot(fig)
                plt.close()
                st.caption(
                    "This is exactly what the model receives. "
                    "It should show a clear heartbeat shape with one main spike (R-peak)."
                )
            else:
                st.error(
                    "Extracted signal is nearly flat — extraction failed. "
                    "The image may have too much noise or the trace could not be isolated."
                )
                fig = make_plot(signal, "tomato", "Extracted signal (invalid)")
                st.pyplot(fig)
                plt.close()

        st.divider()

        if looks_ok:
            X_test, y_test, mean, std = load_test_data()
            signal_norm = preprocess_for_model(signal, mean, std)
            score = score_signal(model, signal_norm)

            st.subheader("Model Verdict")
            verdict_ui(score, threshold)

            st.divider()
            st.subheader("Score in context")
            col_ctx1, col_ctx2 = st.columns(2)

            margin = score - threshold

            with col_ctx1:
                st.markdown(
                    f"**Score interpretation (threshold = `{threshold:.0f}`):**\n"
                    f"- **Normal:** score above `{threshold:.0f}`\n"
                    f"- **Borderline:** within 50 points of threshold\n"
                    f"- **Anomaly:** score below `{threshold:.0f}`\n"
                    f"- **Your score: `{score:.2f}`** "
                    f"({'**above**' if margin >= 0 else '**below**'} threshold by `{abs(margin):.0f}` points)"
                )

            with col_ctx2:
                if margin > 50:
                    st.success(
                        f"Score is **{margin:.0f} points above** threshold. "
                        "This heartbeat pattern closely matches the normal training distribution."
                    )
                elif margin > 0:
                    st.warning(
                        f"Score is only **{margin:.0f} points above** threshold — borderline. "
                        "The pattern is mostly normal but slightly atypical."
                    )
                elif margin > -100:
                    st.error(
                        f"Score is **{abs(margin):.0f} points below** threshold. "
                        "This pattern is outside the normal distribution — anomaly detected."
                    )
                else:
                    st.error(
                        f"Score is **{abs(margin):.0f} points below** threshold — strong anomaly signal. "
                        "This heartbeat is very different from the normal distribution. "
                        "Verify signal extraction in the tabs above if unexpected."
                    )
        else:
            st.error("Cannot run model — signal extraction failed.")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: Browse Test Examples
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🔬 Browse Test Examples":
    st.subheader("Browse real heartbeats from the MIT-BIH test dataset")
    st.caption(
        "21,892 test beats across 5 classes. "
        "The model was trained on normal beats only — it has never seen any of the anomaly classes."
    )

    X_test, y_test, mean, std = load_test_data()

    col_a, col_b, col_c = st.columns([2, 2, 1])

    with col_a:
        selected_class = st.selectbox(
            "Heartbeat type:",
            options=list(CLASS_NAMES.keys()),
            format_func=lambda x: f"{x} — {CLASS_NAMES[x]}",
        )
    with col_b:
        class_indices = np.where(y_test == selected_class)[0]
        beat_number   = st.slider(
            "Beat number:", min_value=1,
            max_value=min(len(class_indices), 100), value=1
        )
    with col_c:
        st.write("")
        st.write("")
        randomise = st.button("🔀 Random")

    beat_idx   = np.random.choice(class_indices) if randomise else class_indices[beat_number - 1]
    beat       = X_test[beat_idx]
    true_label = int(y_test[beat_idx])

    beat_norm  = (beat - mean) / np.where(std < 1e-6, 1.0, std)
    score      = score_signal(model, beat_norm)
    is_anomaly = score < threshold
    color      = "tomato" if is_anomaly else "steelblue"

    fig = make_plot(beat, color, f"Heartbeat #{beat_idx} — {CLASS_NAMES[true_label]}")
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Model Verdict")
    verdict_ui(score, threshold)

    st.divider()
    is_actually_anomaly = true_label > 0
    correct = is_anomaly == is_actually_anomaly

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("True Label")
        if true_label == 0:
            st.success(f"**{CLASS_NAMES[true_label]}**")
        else:
            st.error(f"**{CLASS_NAMES[true_label]}**")
    with c2:
        st.subheader("Model was:")
        if correct:
            st.success("✓ Correct")
        else:
            st.warning(
                "✗ Wrong — " +
                ("False positive (normal beat wrongly flagged)" if is_anomaly else "False negative (anomaly missed)")
            )

    st.divider()
    st.subheader(f"5 example {CLASS_NAMES[selected_class]} beats")
    compare_idx = class_indices[:5]
    cols = st.columns(5)

    for col, idx in zip(cols, compare_idx):
        b       = X_test[idx]
        bn      = (b - mean) / np.where(std < 1e-6, 1.0, std)
        sc      = score_signal(model, bn)
        verdict = "ANOMALY" if sc < threshold else "NORMAL"
        clr     = "tomato" if sc < threshold else "steelblue"
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        ax2.plot(b, color=clr, linewidth=1)
        ax2.set_title(f"{verdict}\n{sc:.0f}", fontsize=7, color=clr)
        ax2.axis('off')
        plt.tight_layout()
        col.pyplot(fig2)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: Anomaly Detector vs Classifier Demo
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "⚡ Anomaly Detector vs Classifier":

    st.subheader("Why unsupervised anomaly detection beats supervised classification")

    st.markdown("""
The core limitation of supervised classifiers (CNN, Random Forest, etc.) is:

> **They can only detect conditions they were explicitly trained to recognise.**

This demo makes that concrete. We train a supervised CNN classifier on 4 known heartbeat types.
Then we show both models a **5th type** — one the CNN has never seen during training.

The CNN+MAF anomaly detector was trained on **normal beats only** — no labels, no anomaly examples.
It detects anything that deviates from normal, including conditions that do not yet have a name.
""")

    cnn_model, cnn_mean, cnn_std = load_cnn(save_dir="outputs")

    if cnn_model is None:
        st.warning(
            "Supervised CNN classifier not trained yet. Run this first:\n\n"
            "```bash\npython main.py --train-cnn\n```"
        )
        st.stop()

    X_test, y_test, maf_mean, maf_std = load_test_data()

    st.divider()

    col_setup1, col_setup2 = st.columns(2)

    with col_setup1:
        st.markdown("### 🤖 Supervised CNN Classifier")
        st.markdown("""
**Training data:** Normal, Supraventricular, Ventricular, Fusion

**Knows:** 4 specific heartbeat types

**Blind to:** Class 4 — Unclassifiable beats
""")
        st.error("❌ Class 4 was hidden from this model during training")

    with col_setup2:
        st.markdown("### 🌊 CNN + MAF Anomaly Detector")
        st.markdown("""
**Training data:** Normal beats only

**Knows:** What a normal heartbeat looks like

**Detects:** Any heartbeat that deviates from normal
""")
        st.success("✅ No anomaly labels needed — flags anything unusual")

    st.divider()

    st.markdown("### Select a heartbeat to test both models")

    beat_class = st.select_slider(
        "Heartbeat type:",
        options=[0, 1, 2, 3, 4],
        value=4,
        format_func=lambda x: (
            f"Class {x}: {CLASS_LABELS.get(x, '')} "
            f"{'← Classifier was trained on this' if x in KNOWN_CLASSES else '← Classifier has NEVER seen this'}"
        )
    )

    class_idx_arr = np.where(y_test == beat_class)[0]
    if len(class_idx_arr) == 0:
        st.warning(f"No class {beat_class} beats in test set.")
        st.stop()

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        beat_num = st.slider("Beat number:", 1, min(len(class_idx_arr), 50), 1)
    with col_b2:
        st.write("")
        st.write("")
        if st.button("🔀 Random beat"):
            beat_num = int(np.random.randint(1, min(len(class_idx_arr), 50)))

    beat       = X_test[class_idx_arr[beat_num - 1]]
    true_class = int(y_test[class_idx_arr[beat_num - 1]])

    is_known   = true_class in KNOWN_CLASSES
    beat_color = "steelblue" if true_class == 0 else ("darkorange" if is_known else "purple")
    fig = make_plot(beat, beat_color, f"Heartbeat — Class {true_class}: {CLASS_LABELS[true_class]}")
    st.pyplot(fig)
    plt.close()

    if not is_known:
        st.info(
            f"ℹ️ You selected **Class 4: Unclassifiable** — "
            "the condition the supervised classifier was never shown. "
            "Watch what each model says."
        )

    st.divider()
    st.markdown("### Model verdicts")

    beat_norm_maf = (beat - maf_mean) / np.where(maf_std < 1e-6, 1.0, maf_std)
    maf_score     = score_signal(model, beat_norm_maf)
    maf_anomaly   = maf_score < threshold
    cnn_result    = cnn_predict(cnn_model, beat, cnn_mean, cnn_std)

    col_maf, col_cnn = st.columns(2)

    with col_maf:
        st.markdown("#### 🌊 CNN + MAF Anomaly Detector")
        if maf_anomaly:
            st.error("## ⚠️ ANOMALY")
            st.caption("Score below threshold — pattern is outside the normal distribution")
        else:
            st.success("## ✅ NORMAL")
            st.caption("Score above threshold — pattern matches the normal distribution")

        st.metric("Score (log p(x))", f"{maf_score:.1f}")
        st.metric("Threshold", f"{threshold:.1f}")

        maf_correct = (maf_anomaly == (true_class != 0))
        if maf_correct:
            st.success("✓ Correct")
        else:
            st.warning("✗ Wrong")

    with col_cnn:
        st.markdown("#### 🤖 Supervised CNN Classifier")
        if cnn_result["says_normal"]:
            st.success(f"## ✅ {cnn_result['predicted_label']}")
        else:
            st.error(f"## ⚠️ {cnn_result['predicted_label']}")

        st.caption(f"Confidence: {cnn_result['confidence']*100:.1f}%")

        st.markdown("**Class probabilities:**")
        for i, prob in enumerate(cnn_result["probabilities"]):
            cls_name = CLASS_LABELS[KNOWN_CLASSES[i]]
            st.progress(float(prob), text=f"{cls_name}: {prob*100:.1f}%")

        cnn_says_anomaly = not cnn_result["says_normal"]
        cnn_correct      = (cnn_says_anomaly == (true_class != 0))
        if cnn_correct:
            st.success("✓ Correct")
        else:
            st.warning("✗ Wrong")

    st.divider()

    if true_class == UNKNOWN_CLASS:
        st.markdown("### 🔑 What just happened")
        if maf_anomaly and not cnn_correct:
            st.error("""
**Classifier failed. CNN + MAF anomaly detector succeeded.**

The supervised CNN has never seen Class 4. It forced the beat into one of its 4 known
categories — it has no mechanism for saying *"I don't recognise this."*

The CNN + MAF model was never shown this class either. But it knows this beat does not
look like the normal beats it learned from. It correctly raised an alert.

This is the key advantage of density estimation in medicine:
new conditions, rare presentations, and previously unseen patterns can all
be flagged — even before anyone has a name for them.
A supervised classifier is always one step behind the unknown.
""")
        elif maf_anomaly and cnn_correct:
            st.success("Both models correctly identified this beat as an anomaly.")
        elif not maf_anomaly:
            recall_pct = eval_results['recall'] * 100
            st.warning(
                f"The anomaly detector missed this one — it scored the beat as normal. "
                f"This is a false negative. "
                f"With recall of {recall_pct:.1f}%, the model misses ~{100-recall_pct:.0f}% of anomalies."
            )
    else:
        if true_class == 0:
            st.info(
                "Both models should say Normal for this beat. "
                "The classifier was trained on this class and the anomaly detector learned the normal distribution."
            )
        else:
            st.info(
                f"Class {true_class} ({CLASS_LABELS[true_class]}) was in the classifier's training data. "
                f"The classifier should recognise it. "
                f"The anomaly detector was never shown any anomalies — it should still flag it as unusual."
            )
