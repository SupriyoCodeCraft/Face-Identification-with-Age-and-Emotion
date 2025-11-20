# face_encoder.py
import torch
from facenet_pytorch import InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def get_embedding(model, face_tensor):
    # face_tensor = (3,160,160) tensor normalized [0,1]
    with torch.no_grad():
        emb = model(face_tensor.unsqueeze(0).to(device))
    emb = emb.cpu().numpy()[0]
    return emb
