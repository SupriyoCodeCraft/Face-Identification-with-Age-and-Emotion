# build_faiss.py
import os
import numpy as np
import faiss
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from face_encoder import load_facenet, get_embedding

GALLERY = "Face Images"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=160, margin=20, device=device)
encoder = load_facenet()

embeddings = []
labels = []

print("Building FAISS Index...")

for person in sorted(os.listdir(GALLERY)):
    folder = os.path.join(GALLERY, person)
    if not os.path.isdir(folder):
        continue

    print(f"> Processing {person}...")

    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)

        try:
            img = Image.open(path).convert("RGB")
        except:
            print("Could not read:", path)
            continue

        # detect & align
        face_tensor = mtcnn(img)
        if face_tensor is None:
            print("No face:", path)
            continue

        # get embedding
        emb = get_embedding(encoder, face_tensor)
        embeddings.append(emb)
        labels.append(person)

embeddings = np.array(embeddings).astype("float32")
labels = np.array(labels)

print("Total embeddings:", embeddings.shape)

# FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(MODEL_DIR, "faiss.index"))
np.save(os.path.join(MODEL_DIR, "faiss_labels.npy"), labels)

print("\nFAISS saved successfully!")
