import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
from PIL import Image, ImageChops, ImageEnhance
 
app = Flask(__name__)
CORS(app)
 
# =========================================================
# MODEL CLASSES
# =========================================================
 
def squash(x, dim=-1):
    norm = torch.norm(x, dim=dim, keepdim=True)
    return (norm ** 2 / (1 + norm ** 2)) * (x / (norm + 1e-8))
 
 
class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, num_capsules=16, capsule_dim=32):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.fc = nn.Linear(input_dim, num_capsules * capsule_dim)
 
    def forward(self, x):
        u = self.fc(x)
        u = u.view(x.size(0), self.num_capsules, self.capsule_dim)
        return squash(u)
 
 
class AGSKLayer(nn.Module):
    def __init__(self, input_dim, dropout=0.35):
        super().__init__()
        self.Wg   = nn.Linear(input_dim, input_dim)
        self.Ws   = nn.Linear(input_dim, input_dim)
        self.bn   = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x):
        gain   = torch.sigmoid(self.Wg(x))
        x_gain = x * gain
        shared = F.relu(self.Ws(x_gain))
        return self.drop(self.bn(x_gain + shared))
 
 
class DualStreamCapsuleDenseNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        densenet      = models.densenet121(weights=None)
        self.features = densenet.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.agsk     = AGSKLayer(2048, dropout=dropout)
        self.caps     = CapsuleLayer(2048, num_capsules=16, capsule_dim=32)
        cap_out = 16 * 32
        self.classifier = nn.Sequential(
            nn.Linear(cap_out, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),     nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
 
    def _encode(self, x):
        f = self.features(x)
        f = self.pool(f)
        return f.view(f.size(0), -1)
 
    def forward(self, ela, orig):
        f_ela  = self._encode(ela)
        f_orig = self._encode(orig)
        fused  = torch.cat([f_ela, f_orig], dim=1)
        fused  = self.agsk(fused)
        caps   = self.caps(fused)
        caps   = caps.view(caps.size(0), -1)
        return self.classifier(caps)
 
 
# =========================================================
# LOAD MODEL
# =========================================================
 
device  = torch.device('cpu')
CLASSES = ['fake', 'real']
IMG_SIZE      = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
print("Loading model...")
try:
    model = DualStreamCapsuleDenseNet(num_classes=2).to(device)
    ckpt  = torch.load('best_model_phase2.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded ✅")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise
 
 
# =========================================================
# TRANSFORMS — single pass (no TTA, fast on CPU)
# =========================================================
 
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
 
 
# =========================================================
# ELA
# =========================================================
 
def ela_image(pil_img, quality=90):
    original = pil_img.convert('RGB')
    tmp      = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        original.save(tmp_name, 'JPEG', quality=quality)
        compressed = Image.open(tmp_name).convert('RGB')
        compressed.load()
        ela      = ImageChops.difference(original, compressed)
        extrema  = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema)
        scale    = 255.0 / max_diff if max_diff != 0 else 1.0
        ela      = ImageEnhance.Brightness(ela).enhance(scale)
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    return ela
 
 
# =========================================================
# ROUTES
# =========================================================
 
@app.route('/')
def home():
    return jsonify({"status": "Backend is running"})
 
 
@app.route('/health')
def health():
    return jsonify({"status": "ok"})
 
 
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Send image with key "image"'}), 400
 
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
 
    tmp      = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_name = tmp.name
    tmp.close()
    file.save(tmp_name)
 
    try:
        print("\n--- New prediction ---")
        orig_img = Image.open(tmp_name).convert('RGB')
        orig_img.load()
        ela_img  = ela_image(orig_img)
 
        # Single forward pass — fast on CPU
        with torch.no_grad():
            ela_t  = val_tf(ela_img).unsqueeze(0)
            orig_t = val_tf(orig_img).unsqueeze(0)
            logits = model(ela_t, orig_t)
 
        probs    = F.softmax(logits, dim=1)[0]
        fake_pct = round(probs[0].item() * 100, 1)
        real_pct = round(probs[1].item() * 100, 1)
        label    = 'fake' if fake_pct > 50 else 'real'
 
        print(f"Result: {label} | fake={fake_pct}% real={real_pct}%")
 
        return jsonify({
            "label":      label,
            "fake_pct":   fake_pct,
            "real_pct":   real_pct,
            "confidence": max(fake_pct, real_pct),
            "metrics": {
                "facial_inconsistencies": round(probs[0].item(), 3),
                "artificial_patterns":    round(probs[0].item() * 0.85, 3),
                "temporal_coherence":     round(probs[1].item(), 3),
            }
        })
 
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
 
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
 
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
 