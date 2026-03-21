from flask import Flask, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
# Autorise ton site Vercel à communiquer avec cette API
CORS(app) 

# --- 1. Copie ta classe Generator V3 ici ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x.view(x.size(0), -1, 1, 1))

# --- 2. Chargement du modèle sur CPU (très important pour le web gratuit) ---
device = torch.device('cpu')
model = Generator(100).to(device)
# On force le map_location='cpu' pour les serveurs web
model.load_state_dict(torch.load('fuze_generator_v3(1).pt', map_location=device))
model.eval()

# --- 3. Route de l'API ---
@app.route('/generate', methods=['GET'])
def generate():
    with torch.no_grad():
        z = torch.randn(1, 100).to(device)
        img_tensor = model(z).squeeze().numpy().transpose(1, 2, 0)
        
        # Formatage de l'image
        img_array = ((img_tensor * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array)

        # Conversion en texte (Base64) pour le web
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"image": f"data:image/png;base64,{img_str}"})

if __name__ == '__main__':
    # Le port par défaut est souvent le 5000
    app.run(host='0.0.0.0', port=5000)
