# pipeline.py
import os
import numpy as np
import faiss
from PIL import Image
import torch
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from face_encoder import load_facenet, get_embedding

# ----------------------------
# Load Models & Paths
# ----------------------------
MODEL_DIR = "models"

# FAISS
index = faiss.read_index(os.path.join(MODEL_DIR, "faiss.index"))
labels = np.load(os.path.join(MODEL_DIR, "faiss_labels.npy"))

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# FaceNet + MTCNN
mtcnn = MTCNN(image_size=160, margin=20, device=device)
encoder = load_facenet()

# Emotion Model
emo_ckpt = torch.load(os.path.join(MODEL_DIR, "emotion_model.pth"), map_location=device)
emotion_model = models.resnet18(weights=None)
emotion_model.fc = torch.nn.Linear(emotion_model.fc.in_features, 7)
emotion_model.load_state_dict(emo_ckpt["model_state_dict"])
emotion_model = emotion_model.to(device).eval()

# RAF-DB correct emotion labels (7-class)
EMOTION_LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

emo_map = {i: EMOTION_LABELS[i] for i in range(7)}

# Age Model
age_ckpt = torch.load(os.path.join(MODEL_DIR, "age_model.pth"), map_location=device)
age_model = models.resnet34(weights=None)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, 1)
age_model.load_state_dict(age_ckpt["model_state_dict"])
age_model = age_model.to(device).eval()

# ----------------------------
# Preprocessing for Emotion & Age
# ----------------------------
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ----------------------------
# Main Inference Function
# ----------------------------
def analyze(image_path):

    # Load image
    img = Image.open(image_path).convert("RGB")

    # ---- 1. Detect Face ----
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None

    # ---- 2. Embedding (FaceNet) ----
    emb = get_embedding(encoder, face_tensor).astype("float32")
    emb = emb.reshape(1, -1)

    # ---- 3. FAISS Search (cosine similarity) ----
    # normalize for cosine similarity search
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    D, I = index.search(emb_norm, 1)

    identity = labels[I[0][0]]
    similarity = float(D[0][0])       # cosine distance
    similarity = 1 - similarity       # convert to similarity score

    # ---- 4. Emotion Prediction ----
    emo_input = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emo_out = emotion_model(emo_input)
        emo_idx = emo_out.argmax().item()
        emotion = emo_map[emo_idx]

    # ---- 5. Age Prediction (use cropped FACE, not full image) ----
    face_np_255 = face_tensor.permute(1,2,0).cpu().numpy()
    face_np_255 = ((face_np_255 + 1) * 127.5).astype("uint8")  # convert [-1,1] → [0,255]

    face_img = Image.fromarray(face_np_255)
    age_input = tfm(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        age = age_model(age_input).item()

    age = max(0, min(int(age), 100))  # clamp for sanity

    # ---- 6. Fix face for Streamlit display ----
    face_disp = (face_tensor.permute(1,2,0).cpu().numpy() + 1) / 2
    face_disp = np.clip(face_disp, 0, 1)

    # ---- 7. Return final output ----
    return {
        "identity": identity,
        "similarity": similarity,
        "emotion": emotion,
        "age": age,
        "face": face_disp
    }
