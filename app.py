# app.py
import streamlit as st
import time
from pipeline import analyze
import os
import uuid
from datetime import datetime
import base64

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Face Analytics AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# CYBERPUNK + CLEAN UI CSS
# ---------------------------------------------------
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top, #0a0f24, #000000 60%);
    font-family: 'Segoe UI', sans-serif;
    color: #eee;
}

/* NEON TITLE */
.cyber-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff00f7, #00eaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

.subtitle {
    text-align: center;
    color: #bbb;
    margin-top: -10px;
}

/* GLASS CARD */
.glass {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    backdrop-filter: blur(10px);
}

/* Neon button */
.stButton>button {
    background: linear-gradient(90deg, #ff00f7, #00eaff);
    border: none;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 48px;
    transition: 0.18s;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px #00eaff;
}

/* Loader animation */
.loader {
  border: 5px solid #222;
  border-top: 5px solid #00eaff;
  border-radius: 50%;
  width: 45px;
  height: 45px;
  animation: spin 0.8s linear infinite;
  margin: auto;
}

@keyframes spin {
  100% { transform: rotate(360deg); }
}

/* Gallery image + hover */
.gallery {
    display:inline-block;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
    transition: transform .18s ease, box-shadow .18s ease;
}
.gallery img {
    display:block;
    width: 140px;
    height: auto;
}
.gallery:hover {
    transform: scale(1.06);
    box-shadow: 0 0 18px #ff00f7;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown('<div class="cyber-title">MULTI-TASK FACE ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identity • Emotion • Age • Smart AI Pipeline</div>', unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs(["📤 Analyze Face", "🖼 Identity Gallery", "🛠 Tools", "📜 History"])

# helper: convert local image to base64 data URI
def img_to_data_uri(path):
    ext = os.path.splitext(path)[1].lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/{ext};base64,{b64}"

# ---------------------------------------------------
# TAB 1 — Upload & Analyze
# ---------------------------------------------------
with tabs[0]:

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

        analyze_btn = None
        if uploaded:
            img_path = f"upload_{uuid.uuid4().hex}.jpg"
            with open(img_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Resize preview to avoid scrolling (use_container_width to fill)
            st.image(img_path, caption="Preview", use_container_width=False, width=320)

            st.write("")
            analyze_btn = st.button("⚡ Analyze Now")

        if analyze_btn:
            # placeholder for loader so we can clear it
            loader_placeholder = st.empty()
            with loader_placeholder.container():
                st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
                st.markdown("<div style='text-align:center;color:#aaa;margin-top:8px;'>Processing...</div>", unsafe_allow_html=True)

            # Run analysis (this may take a sec)
            result = analyze(img_path)

            # clear loader
            loader_placeholder.empty()

            if result is None:
                st.error("❗ No face detected.")
            else:
                st.success("Analysis Complete!")

                with right:
                    st.markdown('<div class="glass">', unsafe_allow_html=True)
                    st.markdown("## 🔍 Result Info")

                    st.write(f"**👤 Identity:** {result['identity']}")
                    st.write(f"**🎭 Emotion:** {result['emotion']}")
                    st.write(f"**🎂 Age:** {result['age']} years")
                    st.write(f"**🌀 Similarity:** {result['similarity']:.4f}")

                    st.image(result["face"], width=230)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Save history
                with open("history.txt", "a") as h:
                    h.write(
                        f"{datetime.now()} | {img_path} | {result['identity']} | {result['emotion']} | {result['age']}\n"
                    )

# ---------------------------------------------------
# TAB 2 — Gallery
# ---------------------------------------------------
with tabs[1]:
    st.markdown("## 🖼 Identity Gallery")

    FACE_DIR = "Face Images"
    if os.path.exists(FACE_DIR):
        folders = [f for f in os.listdir(FACE_DIR) if os.path.isdir(os.path.join(FACE_DIR, f))]
        cols = st.columns(5)

        for i, fol in enumerate(folders):
            folder_path = os.path.join(FACE_DIR, fol)
            imgs = [im for im in os.listdir(folder_path) if im.lower().endswith(('.png','.jpg','.jpeg'))]
            if not imgs:
                continue
            img = os.path.join(folder_path, imgs[0])
            data_uri = img_to_data_uri(img)
            col = cols[i % 5]
            with col:
                st.markdown(f"**{fol}**")
                # embed HTML with gallery class so CSS hover works
                html = f'<div class="gallery"><img src="{data_uri}" alt="{fol}" /></div>'
                st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------
# TAB 3 — Tools
# ---------------------------------------------------
with tabs[2]:
    st.markdown("## 🛠 Tools")

    if st.button("🔄 Rebuild FAISS"):
        # run the faiss build script - ensure filename matches your script
        st.experimental_rerun()  # optional quick UI refresh
        os.system("python build_faiss.py")
        st.success("FAISS rebuilt.")

# ---------------------------------------------------
# TAB 4 — History
# ---------------------------------------------------
with tabs[3]:
    st.markdown("## 📜 Upload History")
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as h:
            st.code(h.read())
    else:
        st.info("No history yet.")
