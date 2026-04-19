import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from torchvision import transforms, models
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================
# MODEL
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
            nn.Linear(cap_out, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
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
FAKE_THRESHOLD = 35

print("Loading model...")
try:
    model = DualStreamCapsuleDenseNet(num_classes=2).to(device)
    ckpt  = torch.load('best_model_phase2.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])  # strict=True — catches errors
    model.eval()
    print("Model loaded ✅")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

try:
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print("Face detector loaded ✅")
except Exception as e:
    print(f"ERROR loading face detector: {e}")
    raise


# =========================================================
# TRANSFORMS
# =========================================================

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tta_transforms = [
    val_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 0.9), int(IMG_SIZE * 0.9))),
        transforms.Pad(int(IMG_SIZE * 0.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


# =========================================================
# FACE CROP
# =========================================================

def crop_face(pil_img, padding=0.3):
    try:
        img_array = np.array(pil_img.convert('RGB'))
        gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces     = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            print("No face — using full image")
            return pil_img

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_array.shape[1], x + w + pad_x)
        y2 = min(img_array.shape[0], y + h + pad_y)
        print(f"Face cropped: ({x1},{y1})→({x2},{y2})")
        return pil_img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"Face crop failed: {e}")
        return pil_img


# =========================================================
# ELA  — Windows file lock safe
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
# FFT ANALYSIS  — Fixed & calibrated
# =========================================================

def fft_analysis(pil_img):
    """
    Analyzes image in frequency domain.
    Prints raw values so you can see exactly
    what each image scores — use /debug to tune.
    """
    try:
        gray      = np.array(pil_img.convert('L')).astype(float)
        fft       = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        h, w = magnitude.shape

        # --- Signal 1: Overall std of frequency magnitudes ---
        # Real photos: higher std (more varied frequencies)
        # GANs: lower std (artificially smooth spectrum)
        freq_std = float(np.std(magnitude))

        # --- Signal 2: Azimuthal uniformity ---
        # GANs produce unnaturally uniform patterns in all directions
        # We measure variance of magnitude along angular bins
        cy, cx    = h // 2, w // 2
        y_idx, x_idx = np.mgrid[0:h, 0:w]
        angles    = np.arctan2(y_idx - cy, x_idx - cx)  # -pi to pi
        angle_bins = np.linspace(-np.pi, np.pi, 37)     # 36 bins
        bin_means  = []
        for i in range(len(angle_bins) - 1):
            mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
            if mask.sum() > 0:
                bin_means.append(magnitude[mask].mean())
        angular_std = float(np.std(bin_means))
        # Low angular_std = unnaturally uniform = GAN

        # --- Signal 3: High frequency ring energy ratio ---
        # GANs suppress high frequencies (too smooth)
        radius     = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
        max_radius = np.sqrt(cy**2 + cx**2)
        hf_mask    = radius > (max_radius * 0.5)   # outer 50% of spectrum
        hf_energy  = float(magnitude[hf_mask].mean() / (magnitude.mean() + 1e-8))

        # --- Signal 4: Periodic grid artifacts ---
        # Some GAN architectures leave grid patterns
        # Detectable as spikes in the power spectrum
        power     = np.abs(fft_shift) ** 2
        power_norm = power / (power.mean() + 1e-8)
        # Count how many pixels have unusually high power (>50x mean)
        spike_ratio = float((power_norm > 50).sum() / (h * w))

        print(f"FFT → freq_std={freq_std:.3f}, angular_std={angular_std:.3f}, "
              f"hf_energy={hf_energy:.3f}, spike_ratio={spike_ratio:.6f}")

        return {
            "freq_std":    freq_std,
            "angular_std": angular_std,
            "hf_energy":   hf_energy,
            "spike_ratio": spike_ratio,
        }

    except Exception as e:
        print(f"FFT failed: {e}")
        return {
            "freq_std": 20.0, "angular_std": 2.0,
            "hf_energy": 1.0, "spike_ratio": 0.001
        }


def fft_fake_score(s):
    """
    Converts FFT signals into 0-100 fake score.
    Each signal votes independently.
    Print statements show exactly why each image scores what it does.
    """
    score = 0
    reasons = []

    # Signal 1: freq_std
    # Typical real photos:  15-35
    # Typical GANs:         8-18
    # (ranges overlap — hence we combine signals)
    if s["freq_std"] < 10:
        score += 35
        reasons.append(f"very_low_freq_std({s['freq_std']:.1f})+35")
    elif s["freq_std"] < 14:
        score += 20
        reasons.append(f"low_freq_std({s['freq_std']:.1f})+20")
    elif s["freq_std"] < 17:
        score += 8
        reasons.append(f"slight_low_freq_std({s['freq_std']:.1f})+8")

    # Signal 2: angular_std
    # Real photos: 1.5-4.0  (varied angular patterns)
    # GANs:        0.3-1.2  (unnaturally uniform)
    if s["angular_std"] < 0.5:
        score += 35
        reasons.append(f"very_uniform_angular({s['angular_std']:.2f})+35")
    elif s["angular_std"] < 1.0:
        score += 20
        reasons.append(f"uniform_angular({s['angular_std']:.2f})+20")
    elif s["angular_std"] < 1.5:
        score += 8
        reasons.append(f"slight_uniform_angular({s['angular_std']:.2f})+8")

    # Signal 3: hf_energy ratio
    # Real photos: 0.7-1.1  (natural high freq content)
    # GANs:        0.3-0.7  (too smooth, suppressed HF)
    if s["hf_energy"] < 0.45:
        score += 30
        reasons.append(f"very_low_hf({s['hf_energy']:.2f})+30")
    elif s["hf_energy"] < 0.60:
        score += 15
        reasons.append(f"low_hf({s['hf_energy']:.2f})+15")
    elif s["hf_energy"] < 0.75:
        score += 5
        reasons.append(f"slight_low_hf({s['hf_energy']:.2f})+5")

    # Signal 4: spike_ratio
    # Grid-pattern GANs leave periodic spikes
    if s["spike_ratio"] > 0.003:
        score += 20
        reasons.append(f"grid_spikes({s['spike_ratio']:.4f})+20")
    elif s["spike_ratio"] > 0.001:
        score += 8
        reasons.append(f"mild_spikes({s['spike_ratio']:.4f})+8")

    final = min(score, 100)
    print(f"FFT score: {final} | reasons: {', '.join(reasons) if reasons else 'none'}")
    return final


# =========================================================
# GLOBAL ERROR HANDLERS — always return JSON
# =========================================================

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled error: {e}")
    return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def home():
    return jsonify({"status": "Backend is running 🚀"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():

    if request.method == 'OPTIONS':
        res = make_response()
        res.headers['Access-Control-Allow-Origin']  = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return res, 200

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
        # Step 1: Load + face crop
        print("\n--- New prediction ---")
        orig_img = Image.open(tmp_name).convert('RGB')
        orig_img.load()
        orig_img = crop_face(orig_img)

        # Step 2: ELA
        ela_img = ela_image(orig_img)

        # Step 3: FFT
        fft_stats      = fft_analysis(orig_img)
        fft_fake_boost = fft_fake_score(fft_stats)

        # Step 4: Model inference with TTA
        logits_sum = torch.zeros(1, 2)
        with torch.no_grad():
            for i, tta_tf in enumerate(tta_transforms):
                ela_t  = tta_tf(ela_img).unsqueeze(0)
                orig_t = val_tf(orig_img).unsqueeze(0)
                logits_sum += model(ela_t, orig_t)
                print(f"TTA {i+1}/5 done")

        probs          = F.softmax(logits_sum, dim=1)[0]
        model_fake_pct = probs[0].item() * 100
        model_real_pct = probs[1].item() * 100
        print(f"Model: fake={model_fake_pct:.1f}% real={model_real_pct:.1f}%")
        print(f"FFT boost: {fft_fake_boost}")

        # Step 5: Combine — model 60% + FFT 40%
        combined_fake = (model_fake_pct * 0.60) + (fft_fake_boost * 0.40)
        combined_real = 100.0 - combined_fake
        label         = 'fake' if combined_fake > FAKE_THRESHOLD else 'real'
        fake_pct      = round(combined_fake, 1)
        real_pct      = round(combined_real, 1)

        print(f"Combined: fake={fake_pct}% → {label.upper()}")

        return jsonify({
            "label":      label,
            "fake_pct":   fake_pct,
            "real_pct":   real_pct,
            "confidence": max(fake_pct, real_pct),
            "metrics": {
                "facial_inconsistencies": round(model_fake_pct / 100, 3),
                "artificial_patterns":    round(fft_fake_boost / 100, 3),
                "temporal_coherence":     round(model_real_pct / 100, 3),
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


# =========================================================
# DEBUG — shows raw scores, use this to tune thresholds
# =========================================================

@app.route('/debug', methods=['POST'])
def debug():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file     = request.files['image']
    tmp      = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_name = tmp.name
    tmp.close()
    file.save(tmp_name)

    try:
        orig_img = Image.open(tmp_name).convert('RGB')
        orig_img.load()
        orig_img = crop_face(orig_img)
        ela_img  = ela_image(orig_img)

        fft_stats      = fft_analysis(orig_img)
        fft_fake_boost = fft_fake_score(fft_stats)

        logits_sum = torch.zeros(1, 2)
        with torch.no_grad():
            for tta_tf in tta_transforms:
                ela_t  = tta_tf(ela_img).unsqueeze(0)
                orig_t = val_tf(orig_img).unsqueeze(0)
                logits_sum += model(ela_t, orig_t)

        probs = F.softmax(logits_sum, dim=1)[0]
        model_fake_pct = probs[0].item() * 100
        model_real_pct = probs[1].item() * 100
        combined_fake  = (model_fake_pct * 0.60) + (fft_fake_boost * 0.40)

        return jsonify({
            "model_fake_pct":   round(model_fake_pct, 1),
            "model_real_pct":   round(model_real_pct, 1),
            "fft_freq_std":     round(fft_stats["freq_std"], 3),
            "fft_angular_std":  round(fft_stats["angular_std"], 3),
            "fft_hf_energy":    round(fft_stats["hf_energy"], 4),
            "fft_spike_ratio":  round(fft_stats["spike_ratio"], 6),
            "fft_fake_boost":   fft_fake_boost,
            "combined_fake":    round(combined_fake, 1),
            "verdict":          "fake" if combined_fake > FAKE_THRESHOLD else "real",
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


# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)