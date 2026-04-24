"""
AstroDetect — Cinematic Space UI
Clean, immersive, zero technical clutter
"""

import os, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter
from collections import Counter

os.environ["PHOTUTILS_FUTURE_COLUMN_NAMES"] = "1"
try:
    import photutils
    photutils.future_column_names = True
except Exception:
    pass
try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    PHOT_OK = True
except ImportError:
    PHOT_OK = False

import gradio as gr

torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(2)
DEVICE = torch.device("cpu")

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "model")
CLF_PATH     = os.path.join(MODEL_DIR, "resnet50_classifier.pt")
YOLO_PATH    = os.path.join(MODEL_DIR, "yolo11s_cosmica.pt")
CLS_TXT      = os.path.join(MODEL_DIR, "cosmica_classes.txt")
CLF_CLASSES  = ["GALAXY", "STAR", "QSO"]
YOLO_CLASSES = ["comet", "galaxy", "globular_cluster", "nebula"]

COLORS = {
    "GALAXY":(229,89,52),"STAR":(59,139,212),"QSO":(29,158,117),
    "galaxy":(229,89,52),"nebula":(139,92,246),
    "globular_cluster":(239,200,39),"comet":(239,159,39),
}
LABELS = {
    "GALAXY":"Galaxy","STAR":"Star","QSO":"Quasar",
    "galaxy":"Galaxy","nebula":"Nebula",
    "globular_cluster":"Cluster","comet":"Comet",
}

def _get_centroids(src):
    for xn,yn in [("x_centroid","y_centroid"),("xcentroid","ycentroid")]:
        try: return src[xn].tolist(), src[yn].tolist()
        except (KeyError,TypeError): continue
    c=src.colnames; return src[c[0]].tolist(), src[c[1]].tolist()

def build_clf():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features,512), nn.BatchNorm1d(512),
        nn.ReLU(), nn.Dropout(0.4), nn.Linear(512,3))
    return m

clf = None
if os.path.exists(CLF_PATH):
    clf  = build_clf()
    ckpt = torch.load(CLF_PATH, map_location="cpu", weights_only=False)
    clf.load_state_dict(ckpt.get("model_state_dict", ckpt))
    clf  = clf.eval()
    print(f"Classifier loaded — {os.path.getsize(CLF_PATH)/1e6:.0f}MB")

yolo = None
try:
    from ultralytics import YOLO as _YOLO
    if os.path.exists(YOLO_PATH):
        yolo = _YOLO(YOLO_PATH)
        if os.path.exists(CLS_TXT):
            YOLO_CLASSES[:] = [l.strip() for l in open(CLS_TXT) if l.strip()]
        print(f"Detector loaded — {YOLO_CLASSES}")
except ImportError:
    pass

INFER_TF = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def run_pipeline(img_pil, sigma=4.0, max_src=150, yolo_conf=0.25):
    img_np = np.array(img_pil.convert("RGB"))
    W,H    = img_pil.size
    dets   = []

    if PHOT_OK and clf is not None:
        gray        = np.mean(img_np,axis=2).astype(np.float64)
        _,med,std   = sigma_clipped_stats(gray, sigma=3.0)
        if std > 0:
            srcs = DAOStarFinder(fwhm=3.0, threshold=sigma*std)(gray-med)
            if srcs is not None and len(srcs)>0:
                xs,ys = _get_centroids(srcs)
                fs    = srcs["flux"].tolist()
                top   = sorted(zip(xs,ys,fs), key=lambda v:-v[2])[:int(max_src)]
                crops, boxes = [], []
                for x,y,_ in top:
                    xi,yi  = int(round(x)), int(round(y))
                    x1,x2  = max(0,xi-16), min(W,xi+16)
                    y1,y2  = max(0,yi-16), min(H,yi+16)
                    if x2-x1<8 or y2-y1<8: continue
                    crops.append(INFER_TF(img_pil.crop((x1,y1,x2,y2)).convert("RGB")))
                    boxes.append((x1,y1,x2,y2))
                if crops:
                    with torch.no_grad():
                        probs = torch.softmax(clf(torch.stack(crops)), dim=1)
                        preds = probs.argmax(1).tolist()
                        confs = probs.max(1).values.tolist()
                    for (x1,y1,x2,y2),pred,conf in zip(boxes,preds,confs):
                        dets.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,
                            class_name=CLF_CLASSES[pred],
                            confidence=round(conf,3), stage="phot"))

    if yolo is not None:
        r = yolo.predict(source=img_np, conf=float(yolo_conf),
                         iou=0.45, imgsz=640, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                bx1,by1,bx2,by2 = [int(v) for v in box.xyxy[0].tolist()]
                cid  = int(box.cls[0])
                name = YOLO_CLASSES[cid] if cid<len(YOLO_CLASSES) else str(cid)
                dets.append(dict(x1=bx1,y1=by1,x2=bx2,y2=by2,
                    class_name=name, confidence=round(float(box.conf[0]),3),
                    stage="yolo"))
    return dets

def draw_detections(img_pil, dets):
    from PIL import ImageFont
    out  = img_pil.copy().convert("RGBA")
    ov   = Image.new("RGBA", out.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    try:    font = ImageFont.load_default(size=13)
    except: font = ImageFont.load_default()

    for d in dets:
        col  = COLORS.get(d["class_name"],(200,200,200))
        lbl  = LABELS.get(d["class_name"], d["class_name"])
        x1,y1,x2,y2 = d["x1"],d["y1"],d["x2"],d["y2"]

        if d["stage"]=="phot" and d["class_name"]=="STAR":
            cx,cy = (x1+x2)//2,(y1+y2)//2
            # glow dot — draw multiple expanding circles
            for r,a in [(6,30),(4,60),(2,120)]:
                draw.ellipse([cx-r,cy-r,cx+r,cy+r],
                    fill=(*col,a), outline=None)
            draw.ellipse([cx-2,cy-2,cx+2,cy+2], fill=(*col,255))
        else:
            # glow box — outer soft glow
            for expand, alpha in [(8,15),(5,25),(3,40),(1,80)]:
                draw.rectangle(
                    [x1-expand,y1-expand,x2+expand,y2+expand],
                    outline=(*col,alpha), width=1)
            # solid border
            draw.rectangle([x1,y1,x2,y2], outline=(*col,220), width=2)
            # label pill
            try:
                bb = draw.textbbox((0,0), lbl, font=font)
                tw,th = bb[2]-bb[0], bb[3]-bb[1]
            except:
                tw,th = len(lbl)*7, 13
            px,py = x1+1, y1-th-8
            draw.rounded_rectangle(
                [px-4, py-2, px+tw+6, py+th+4],
                radius=3, fill=(0,0,0,180))
            draw.text((px+1, py+1), lbl, fill=(*col,255), font=font)

    out = Image.alpha_composite(out, ov).convert("RGB")
    return out

def process(image, max_stars, sigma, conf_thresh):
    if image is None:
        return None, _waiting_html(), "{}"

    t0      = time.time()
    img_pil = Image.fromarray(image).convert("RGB")
    dets    = run_pipeline(img_pil, sigma=sigma,
                           max_src=max_stars, yolo_conf=conf_thresh)
    elapsed = time.time() - t0
    ann     = draw_detections(img_pil, dets)
    counts  = Counter(LABELS.get(d["class_name"], d["class_name"]) for d in dets)
    total   = len(dets)

    result_html = _result_html(total, counts, elapsed)
    json_out    = json.dumps({
        "objects_found": total,
        "scan_time_s":   round(elapsed,2),
        "catalogue":     dict(counts),
        "detections":    [{
            "label":      LABELS.get(d["class_name"],d["class_name"]),
            "confidence": d["confidence"],
            "bbox":       [d["x1"],d["y1"],d["x2"],d["y2"]]
        } for d in dets[:60]]
    }, indent=2)
    return np.array(ann), result_html, json_out

def _waiting_html():
    return """
<div style="
  min-height:180px;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  font-family:'Share Tech Mono',monospace;
  color:rgba(0,255,200,0.35);
  letter-spacing:0.25em; text-transform:uppercase;
  font-size:0.75rem; gap:12px;
">
  <div style="font-size:2rem; opacity:0.3;">◌</div>
  Awaiting telescope image…
</div>"""

def _result_html(total, counts, elapsed):
    # Build class badges
    color_map = {
        "Star":"#3b8bd4","Galaxy":"#e5592e","Quasar":"#1d9e75",
        "Nebula":"#8b5cf6","Cluster":"#efc827","Comet":"#ef9f27",
    }
    badges = ""
    for name, n in sorted(counts.items(), key=lambda x:-x[1]):
        col = color_map.get(name,"#888")
        badges += f"""
<div style="
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,255,255,0.04);
  border:1px solid {col}44;
  border-left:3px solid {col};
  border-radius:4px;
  padding:8px 14px; margin:4px;
">
  <span style="font-family:'Orbitron',monospace;font-size:1.1rem;
    font-weight:700;color:{col};">{n}</span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
    color:rgba(200,220,255,0.7);letter-spacing:0.12em;text-transform:uppercase;">
    {name}{'s' if n!=1 else ''}</span>
</div>"""

    return f"""
<div style="
  font-family:'Share Tech Mono',monospace;
  padding: 4px 0;
">
  <!-- Scan complete header -->
  <div style="
    display:flex; align-items:center; gap:12px;
    margin-bottom:16px;
    padding-bottom:12px;
    border-bottom:1px solid rgba(0,255,200,0.12);
  ">
    <div style="
      width:8px; height:8px; border-radius:50%;
      background:#00ffc8;
      box-shadow: 0 0 10px #00ffc8, 0 0 20px #00ffc8;
      flex-shrink:0;
    "></div>
    <div>
      <div style="font-family:'Orbitron',monospace;font-size:1rem;
        font-weight:700;color:#00ffc8;letter-spacing:0.1em;">
        SCAN COMPLETE
      </div>
      <div style="font-size:0.65rem;color:rgba(0,200,160,0.5);
        letter-spacing:0.15em;margin-top:2px;">
        {total} objects identified &nbsp;·&nbsp; {elapsed:.1f}s
      </div>
    </div>
  </div>

  <!-- Object badges -->
  <div style="display:flex; flex-wrap:wrap; gap:0; margin-bottom:12px;">
    {badges if badges else '<div style="color:rgba(120,140,160,0.5);font-size:0.75rem;padding:8px;">No objects detected — try lowering the threshold</div>'}
  </div>

  <!-- Total count large -->
  <div style="
    background:rgba(0,255,200,0.04);
    border:1px solid rgba(0,255,200,0.15);
    border-radius:4px;
    padding:12px 16px;
    display:flex; align-items:center; justify-content:space-between;
  ">
    <span style="font-size:0.65rem;color:rgba(0,200,160,0.6);
      letter-spacing:0.2em;text-transform:uppercase;">Total Objects</span>
    <span style="font-family:'Orbitron',monospace;font-size:1.6rem;
      font-weight:900;color:#00ffc8;
      text-shadow:0 0 15px rgba(0,255,200,0.4);">{total}</span>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Cinematic, fullscreen space, zero technical clutter
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── Global reset + deep space background ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
    margin: 0; padding: 0;
    min-height: 100vh;
    overflow-x: hidden;
}

body, .gradio-container, gradio-app, #root {
    min-height: 100vh;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #c8e8ff !important;
    /* Layered: gradient nebulae + real space photo + solid fallback */
    background:
        radial-gradient(ellipse at 20% 35%, rgba(100,20,180,0.50) 0%, transparent 48%),
        radial-gradient(ellipse at 78% 18%, rgba(10,50,180,0.45) 0%, transparent 42%),
        radial-gradient(ellipse at 55% 80%, rgba(0,120,140,0.32) 0%, transparent 38%),
        radial-gradient(ellipse at 5%  90%, rgba(80,0,120,0.38) 0%, transparent 38%),
        radial-gradient(ellipse at 90% 65%, rgba(5,80,160,0.28) 0%, transparent 35%),
        url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=1920&q=85') center/cover no-repeat fixed,
        #000510 !important;
}

/* ── Starfield overlay (pure CSS) ── */
body::before {
    content:'';
    position:fixed; inset:0;
    background-image:
        radial-gradient(1px 1px at  7% 10%, rgba(255,255,255,0.95) 0%, transparent 100%),
        radial-gradient(1px 1px at 18% 82%, rgba(200,230,255,0.80) 0%, transparent 100%),
        radial-gradient(1px 1px at 32% 22%, rgba(255,255,255,0.85) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 47% 57%, rgba(180,215,255,0.70) 0%, transparent 100%),
        radial-gradient(1px 1px at 63% 12%, rgba(255,255,255,0.90) 0%, transparent 100%),
        radial-gradient(2px 2px at 71% 78%, rgba(200,175,255,0.65) 0%, transparent 100%),
        radial-gradient(1px 1px at 84% 42%, rgba(255,255,255,0.85) 0%, transparent 100%),
        radial-gradient(1px 1px at 12% 50%, rgba(220,195,255,0.65) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 43% 93%, rgba(255,215,175,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 30%, rgba(195,255,215,0.65) 0%, transparent 100%),
        radial-gradient(2px 2px at  3% 97%, rgba(255,255,255,0.40) 0%, transparent 100%),
        radial-gradient(1px 1px at 97%  5%, rgba(215,238,255,0.85) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 35%, rgba(255,255,255,0.60) 0%, transparent 100%),
        radial-gradient(1px 1px at 26% 65%, rgba(200,230,255,0.50) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 76% 55%, rgba(255,200,230,0.45) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── All gradio wrappers transparent ── */
.gradio-container > * { position: relative; z-index: 1; }

/* ── Glass panels ── */
.block, .gr-box, .panel, .form,
div[data-testid="block"], .wrap {
    background: rgba(1, 4, 16, 0.78) !important;
    border: 1px solid rgba(0, 255, 200, 0.14) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
}

/* ── Labels ── */
label, .label-wrap span, .svelte-1ipelgc {
    color: rgba(0,255,200,0.65) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
}

/* ── Sliders ── */
input[type=range] {
    accent-color: #00ffc8 !important;
}
.wrap.svelte-215172 input[type=range]::-webkit-slider-thumb {
    background: #00ffc8 !important;
    box-shadow: 0 0 8px #00ffc8 !important;
}

/* ── UPLOAD ZONE — beautiful glow border ── */
.upload-container, [data-testid="image"] .wrap,
.image-container, .svelte-116rqfv {
    border: 1px dashed rgba(0,255,200,0.35) !important;
    background: rgba(0,5,18,0.72) !important;
    border-radius: 8px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.upload-container:hover, [data-testid="image"] .wrap:hover {
    border-color: rgba(0,255,200,0.65) !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.12),
                inset 0 0 20px rgba(0,255,200,0.04) !important;
}

/* ── THE SCAN BUTTON ── */
button.primary, button[variant="primary"],
.gr-button-primary {
    background: transparent !important;
    border: 2px solid #00ffc8 !important;
    color: #00ffc8 !important;
    font-family: 'Orbitron', 'Share Tech Mono', monospace !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.28em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 14px 36px !important;
    box-shadow:
        0 0 20px rgba(0,255,200,0.30),
        inset 0 0 20px rgba(0,255,200,0.06) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 4px !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: rgba(0,255,200,0.10) !important;
    box-shadow:
        0 0 40px rgba(0,255,200,0.55),
        inset 0 0 30px rgba(0,255,200,0.12) !important;
    letter-spacing: 0.34em !important;
}

/* ── Secondary buttons ── */
button.secondary, button[variant="secondary"] {
    background: rgba(0,255,200,0.05) !important;
    border: 1px solid rgba(0,255,200,0.25) !important;
    color: rgba(0,255,200,0.7) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.12em !important;
    font-size: 0.72rem !important;
    border-radius: 4px !important;
}

/* ── Accordion ── */
details summary {
    color: rgba(0,255,200,0.5) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
details {
    border: 1px solid rgba(0,255,200,0.10) !important;
    background: rgba(0,3,12,0.7) !important;
    border-radius: 6px !important;
}

/* ── Code/JSON block ── */
.code-wrap, pre, code, .cm-editor {
    background: rgba(0,3,12,0.92) !important;
    border: 1px solid rgba(0,255,200,0.12) !important;
    color: #40e8b0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background: rgba(0,5,15,0.5); }
::-webkit-scrollbar-thumb { background: rgba(0,255,200,0.25); border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,200,0.45); }

/* ── Remove all gradio branding / footers ── */
footer, .footer, .svelte-1ax1toq { display:none !important; }

/* ── Markdown ── */
.gr-markdown, .prose, .md {
    font-family: 'Share Tech Mono', monospace !important;
    color: #90b8d8 !important;
    line-height: 1.7 !important;
}
.gr-markdown h3, .prose h3 {
    font-family: 'Orbitron', monospace !important;
    color: rgba(0,255,200,0.80) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
.gr-markdown code {
    color: #00ffc8 !important;
    background: rgba(0,255,200,0.08) !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-size: 0.9em !important;
}
.gr-markdown strong { color: #ffffff !important; }

/* ── Number display style ── */
.gr-markdown p { color: #80a8c8 !important; font-size:0.82rem !important; }
"""

# ── HERO HTML ─────────────────────────────────────────────────────────────────
HERO = """
<div style="
  position: relative;
  text-align: center;
  padding: 52px 24px 40px;
  overflow: hidden;
">
  <!-- Nebula glow blobs -->
  <div style="position:absolute;inset:0;overflow:hidden;pointer-events:none;">
    <div style="position:absolute;top:-60px;left:8%;width:380px;height:260px;
      background:radial-gradient(ellipse,rgba(100,20,200,0.45) 0%,transparent 65%);
      filter:blur(40px);"></div>
    <div style="position:absolute;top:-40px;right:5%;width:340px;height:220px;
      background:radial-gradient(ellipse,rgba(10,60,200,0.40) 0%,transparent 65%);
      filter:blur(35px);"></div>
    <div style="position:absolute;bottom:-30px;left:40%;width:300px;height:150px;
      background:radial-gradient(ellipse,rgba(0,180,140,0.30) 0%,transparent 65%);
      filter:blur(30px);"></div>
  </div>

  <!-- Eyebrow -->
  <div style="
    font-family:'Share Tech Mono',monospace;
    font-size:0.62rem; letter-spacing:0.45em;
    color:rgba(0,255,200,0.55);
    text-transform:uppercase; margin-bottom:16px;
    position:relative;
  ">◈ &nbsp; DEEP SPACE OBJECT CLASSIFICATION SYSTEM &nbsp; ◈</div>

  <!-- Main wordmark -->
  <div style="
    font-family:'Orbitron',monospace;
    font-size: clamp(2.6rem, 8vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    line-height: 1.0;
    background: linear-gradient(135deg,
      #00ffc8 0%, #00d4ff 30%, #8040ff 58%, #ff40c8 80%, #00ffc8 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(0,255,200,0.35));
    margin-bottom: 14px;
    position: relative;
  ">ASTRODETECT</div>

  <!-- Tagline -->
  <div style="
    font-family:'Share Tech Mono',monospace;
    font-size: clamp(0.7rem, 1.8vw, 0.9rem);
    letter-spacing: 0.20em;
    color: rgba(180,220,255,0.65);
    text-transform: uppercase;
    margin-bottom: 8px;
    position: relative;
  ">Explore the Universe Through AI Vision</div>

  <!-- Divider line -->
  <div style="
    width:120px; height:1px;
    background: linear-gradient(90deg,
      transparent, rgba(0,255,200,0.6), transparent);
    margin: 20px auto 0;
    position: relative;
  "></div>
</div>
"""

FOOTER = """
<div style="
  text-align:center;
  padding: 18px 24px;
  margin-top: 8px;
  border-top: 1px solid rgba(0,255,200,0.08);
  font-family:'Share Tech Mono',monospace;
  font-size:0.58rem;
  color:rgba(0,180,140,0.30);
  letter-spacing:0.14em;
  text-transform:uppercase;
">
  ASTRODETECT &nbsp;·&nbsp; HYBRID AI PIPELINE
  &nbsp;·&nbsp; photutils · PyTorch · Ultralytics
  &nbsp;·&nbsp; CPU optimised
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    title="AstroDetect",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Share Tech Mono"),
    )
) as demo:

    gr.HTML(HERO)

    with gr.Row():

        # ── LEFT: Upload + Controls ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):

            inp_img = gr.Image(
                label="TELESCOPE IMAGE",
                type="numpy",
                height=280,
            )

            detect_btn = gr.Button(
                "⚡  INITIATE DEEP SCAN",
                variant="primary",
                size="lg",
            )

            with gr.Accordion("◈  SCAN PARAMETERS", open=False):
                max_stars = gr.Slider(50, 400, value=150, step=25,
                    label="MAX POINT SOURCES")
                sigma_sl  = gr.Slider(2.0, 8.0, value=4.0, step=0.5,
                    label="DETECTION THRESHOLD σ")
                yolo_sl   = gr.Slider(0.10, 0.60, value=0.25, step=0.05,
                    label="EXTENDED OBJECT SENSITIVITY")

            gr.Markdown("""
**Detected classes**
`Star` · `Galaxy` · `Quasar`
`Nebula` · `Cluster` · `Comet`

**Scan time (CPU)**
`50 sources` → ~20s
`150 sources` → ~90s
""")

        # ── RIGHT: Results ───────────────────────────────────────────────────
        with gr.Column(scale=2):

            out_img = gr.Image(
                label="DETECTION FIELD MAP",
                type="numpy",
                height=420,
                interactive=False,
            )

            result_html = gr.HTML(_waiting_html())

    with gr.Accordion("◈  OBJECT CATALOGUE  [ RAW DATA ]", open=False):
        json_out = gr.Code(language="json", label="")

    detect_btn.click(
        fn=process,
        inputs=[inp_img, max_stars, sigma_sl, yolo_sl],
        outputs=[out_img, result_html, json_out],
    )

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
