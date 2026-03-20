"""
Reimagine HDR Studio — Railway Backend
=======================================
Deploy to Railway in 3 steps:
  1. Create new project on railway.app → "Deploy from GitHub"
  2. Push this file + requirements.txt to a GitHub repo
  3. Set environment variables in Railway dashboard (see below)

Environment variables to set in Railway:
  REPLICATE_API_TOKEN   = r8_xxxxxxxxxxxxxxxx
  ANTHROPIC_API_KEY     = sk-ant-xxxxxxxxxxxxxxxx
  ALLOWED_ORIGIN        = https://your-frontend.com  (or * for dev)
"""

import os, io, base64, tempfile, httpx, anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import replicate

app = Flask(__name__)
CORS(app, origins=os.environ.get("ALLOWED_ORIGIN", "*"))

REPLICATE_TOKEN  = os.environ.get("REPLICATE_API_TOKEN", "")
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Utility ───────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode()

def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def pil_to_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── HDR Merge (Enfuse / Mertens) ─────────────────────────────────────────────

def merge_hdr(images: list[np.ndarray]) -> np.ndarray:
    """
    Exposure fusion via OpenCV's Mertens algorithm.
    Equivalent to Enfuse — produces natural-looking HDR without tone-mapping artefacts.
    """
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=0.0,   # let contrast/saturation drive it
    )
    fused = merge.process(images)
    # Convert [0,1] float to uint8
    result = (fused * 255).clip(0, 255).astype(np.uint8)
    return result

# ── Claude Vision Analysis ────────────────────────────────────────────────────

def analyze_image(b64_image: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="""You are an expert real estate photography AI.
Return ONLY a JSON object — no markdown, no backticks, no preamble.
{
  "roomType": "string",
  "lightingQuality": "poor|fair|good",
  "windowCount": number,
  "windowIssues": ["array"],
  "skyVisible": boolean,
  "skyType": "string or null",
  "exposureIssues": ["array"],
  "colorIssues": ["array"],
  "recommendedEdits": ["4-6 items"],
  "editingDifficulty": "easy|moderate|complex",
  "professionalNotes": "2-3 sentences"
}""",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_image}},
                {"type": "text", "text": "Analyse this merged HDR real estate photo. Return only JSON."}
            ]
        }]
    )
    import json
    raw = resp.content[0].text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Replicate: Sky Replacement ────────────────────────────────────────────────

def replace_sky(b64_image: str) -> str:
    """Returns base64 of sky-replaced image."""
    client = replicate.Client(api_token=REPLICATE_TOKEN)
    # Using stability-ai inpainting; swap for a dedicated sky model if preferred
    output = client.run(
        "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
        input={
            "image": f"data:image/jpeg;base64,{b64_image}",
            "prompt": "photorealistic blue sky with soft white clouds, golden hour light, real estate photography",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        }
    )
    # output is a URL — fetch and re-encode
    img_bytes = httpx.get(str(output[0])).content
    return base64.b64encode(img_bytes).decode()

# ── Exposure / Color correction (OpenCV) ─────────────────────────────────────

def tone_map(img: np.ndarray) -> np.ndarray:
    """Apply Reinhard global tone mapping + warm grade for real estate."""
    tonemap = cv2.createTonemapReinhard(gamma=1.2, intensity=0.0, light_adapt=0.8, color_adapt=0.0)
    float_img = img.astype(np.float32) / 255.0
    mapped = tonemap.process(float_img)
    result = (mapped * 255).clip(0, 255).astype(np.uint8)
    return result

def warm_grade(img: np.ndarray) -> np.ndarray:
    """Subtle warm real-estate grade — boost reds/yellows, lift shadows."""
    b, g, r = cv2.split(img.astype(np.float32))
    r = (r * 1.04).clip(0, 255)
    g = (g * 1.01).clip(0, 255)
    b = (b * 0.97).clip(0, 255)
    # Lift shadows
    lut = np.array([min(255, int(i + (20 * (1 - i/255.0)))) for i in range(256)], dtype=np.uint8)
    result = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
    result = cv2.LUT(result, lut)
    return result

def remove_glare(img: np.ndarray) -> np.ndarray:
    """Suppress specular highlights — simple bilateral + highlight desaturation."""
    smoothed = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Desaturate very bright areas
    v = hsv[:, :, 2]
    mask = v > 230
    hsv[:, :, 1][mask] *= 0.5
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result

# ── Main /process endpoint ────────────────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process():
    # 1. Read uploaded shots
    files = [request.files[k] for k in sorted(request.files.keys())]
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 exposure shots"}), 400

    np_images = []
    for f in files:
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            np_images.append(img)

    # 2. Resize all to same dimensions (use smallest)
    h = min(img.shape[0] for img in np_images)
    w = min(img.shape[1] for img in np_images)
    np_images = [cv2.resize(img, (w, h)) for img in np_images]

    # 3. HDR merge
    merged = merge_hdr(np_images)

    # 4. Tone map
    graded = tone_map(merged)

    # 5. Warm grade
    graded = warm_grade(graded)

    # 6. Glare removal
    graded = remove_glare(graded)

    # 7. Encode for analysis
    pil_result = np_to_pil(graded)
    b64 = pil_to_b64(pil_result)

    # 8. Claude Vision analysis
    analysis = {}
    if ANTHROPIC_KEY:
        try:
            analysis = analyze_image(b64)
        except Exception as e:
            analysis = {"error": str(e)}

    # 9. Sky replacement (if sky detected & Replicate key set)
    if REPLICATE_TOKEN and analysis.get("skyVisible"):
        try:
            b64 = replace_sky(b64)
        except Exception as e:
            print(f"Sky replacement failed: {e}")

    return jsonify({
        "image_url": f"data:image/jpeg;base64,{b64}",
        "analysis": analysis,
        "shots_merged": len(np_images),
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
