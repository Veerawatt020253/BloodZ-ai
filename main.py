import os, io, math, base64, csv
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# =============== YOLO (ultralytics) ===============
from ultralytics import YOLO

app = FastAPI(title="🧬 BLOOD ANALYZER AI v12.0 — API")

# # ===== CORS =====
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

origins = [
    "http://localhost:3000",      # Your local frontend
    "https://blood.ongor.fun",    # Your production domain
    "http://127.0.0.1:3000",      # Alternative local address
    "https://www.blood-z.com",     # <--- (สำคัญ) โดเมนที่ error แจ้งมา
    "https://blood-z.com",         # <--- ควรเผื่อแบบไม่มี www ไว้ด้วย
    "https://blood.ongor.fun"      # โดเมนตัวเอง (เผื่อมีการเรียก internal)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Must be specific list, not ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Global model cache -----------------
MODEL_CACHE: Dict[str, YOLO] = {}

def load_model(model_bytes: Optional[bytes]=None, model_path: Optional[str]=None) -> YOLO:
    """
    Load YOLO model either from uploaded bytes or from a filesystem path / env MODEL_PATH.
    """
    global MODEL_CACHE

    if model_bytes:
        # Cache by hash of bytes to avoid reloading if same upload comes again
        key = f"bytes:{hash(model_bytes)}"
        if key not in MODEL_CACHE:
            tmp_path = "./_uploaded_best.pt"
            with open(tmp_path, "wb") as f:
                f.write(model_bytes)
            MODEL_CACHE[key] = YOLO(tmp_path)
        return MODEL_CACHE[key]

    # Fall back to path
    path = model_path or os.getenv("MODEL_PATH", "./best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO weights not found. Provide 'model' file in request or set MODEL_PATH env (current: {path})."
        )
    if path not in MODEL_CACHE:
        MODEL_CACHE[path] = YOLO(path)
    return MODEL_CACHE[path]

# ----------------- Utils -----------------
def read_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Ensure JPG/PNG formats.")
    return img

def bgr_to_jpeg_base64(img_bgr: np.ndarray, quality: int = 85) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", img_bgr, encode_param)
    if not ok:
        raise ValueError("Failed to encode image to JPEG.")
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def remove_black_border_auto_circle(img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=50, maxRadius=0
    )
    if circles is not None and len(circles[0]) > 0:
        x, y, r = np.uint16(np.around(circles))[0, 0]
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        white_bg = np.where(mask[..., None] == 0, 255, result)
        return white_bg.astype(np.uint8), True
    return img_bgr, False

def zoom_sharpen_and_split(img_bgr: np.ndarray, zoom_factor: float = 3.0, rows: int = 3, cols: int = 3):
    # Denoise
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
    # Zoom
    z = cv2.resize(den, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    # Sharpen
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1, 10, -1],
                             [-1, -1, -1]])
    sharp = cv2.filter2D(z, -1, kernel_sharp)

    h, w = sharp.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    parts = []
    for i in range(rows):
        for j in range(cols):
            block = sharp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            up2 = cv2.resize(block, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            parts.append(up2)
    return parts, (rows, cols)

def parse_disease_csv(csv_bytes: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8")
    disease_db = []
    for _, row in df.iterrows():
        cond = {}
        if isinstance(row.get("conditions", ""), str):
            for c in row["conditions"].split(";"):
                if ":" in c:
                    k, v = c.split(":")
                    cond[k.strip()] = v.strip()
        disease_db.append({
            "name": row.get("name", ""),
            "conditions": cond,
            "details": row.get("details", "")
        })
    return disease_db

def normalize_level(obs_val: float, kind: str) -> bool:
    base = 50
    if kind == "low":
        return obs_val < 0.8 * base
    if kind == "high":
        return obs_val > 1.2 * base
    return True

def softmax_rule_based(obs: Dict[str, float], db: List[Dict[str, Any]], topk: int = 5):
    results = []
    for dis in db:
        cond = dis["conditions"]
        score, total = 0, 0
        for key, val in cond.items():
            if key in obs:
                total += 1
                if isinstance(val, str) and normalize_level(obs[key], val):
                    score += 1
        ratio = score / total if total else 0
        results.append((dis["name"], ratio, dis["details"]))
    if not results:
        return []
    exp_scores = [math.exp(r[1]) for r in results]
    s = sum(exp_scores)
    final = [(r[0], exp_scores[i]/s, r[2]) for i, r in enumerate(results)]
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:topk]

def counts_template() -> Dict[str, int]:
    return {
        "RBC": 0, "Platelet": 0, "Neutrophil": 0,
        "Lymphocyte": 0, "Monocyte": 0,
        "Eosinophil": 0, "Basophil": 0
    }

def classify_label_to_key(label: str) -> Optional[Tuple[str, Tuple[int,int,int]]]:
    l = label.lower()
    if "rbc" in l or "eryth" in l:
        return "RBC", (0, 0, 255)
    if "plate" in l or "plt" in l:
        return "Platelet", (255, 0, 0)
    if "neutro" in l:
        return "Neutrophil", (0, 255, 0)
    if "lymph" in l:
        return "Lymphocyte", (255, 255, 0)
    if "mono" in l:
        return "Monocyte", (255, 165, 0)
    if "eosino" in l:
        return "Eosinophil", (255, 0, 255)
    if "baso" in l:
        return "Basophil", (128, 0, 128)
    return None

def detections_to_dict(res, names) -> List[Dict[str, Any]]:
    out = []
    boxes = res[0].boxes if len(res) > 0 else None
    if boxes is None or len(boxes) == 0:
        return out
    for box in boxes:
        cls_id = int(box.cls)
        label = names[cls_id] if names else str(cls_id)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        out.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2]
        })
    return out

def annotate_from_dets(img_bgr: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    ann = img_bgr.copy()
    for d in dets:
        label = d["label"]
        x1, y1, x2, y2 = d["bbox_xyxy"]
        mapped = classify_label_to_key(label)
        color = (200, 200, 200) if mapped is None else mapped[1]
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ann, label, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return ann

def to_csv_base64(cell_counts: Dict[str, int], top_diag: List[Tuple[str, float, str]]) -> str:
    df = pd.DataFrame([cell_counts])
    for i, (n, s, d) in enumerate(top_diag, 1):
        df[f"Top{i}_Disease"] = n
        df[f"Top{i}_Score"] = s
        df[f"Top{i}_Detail"] = d
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    csv_text = buf.getvalue().encode("utf-8-sig")
    return base64.b64encode(csv_text).decode("utf-8")

def build_text_summary_th(counts: Dict[str,int], top_list: List[Tuple[str,float,str]], disease_db_len: int) -> str:
    # ส่วนรวมจำนวนเซลล์
    counts_str = str(counts)
    # ส่วน Top diagnoses (รูปแบบเหมือนตัวอย่าง)
    lines = []
    for i, (n, s, d) in enumerate(top_list, 1):
        pct = round(s * 100, 1)
        lines.append(f"{i}. {n} — {pct:.1f}% | {d}")
    diag_block = "\n".join(lines) if lines else "(ไม่พบรายการจากฐานข้อมูลโรค)"
    # ประกอบข้อความสรุป
    text = (
        f"รวมจำนวนเซลล์ทั้งหมด: {counts_str}\n"
        f"✅ โหลดฐานข้อมูลโรค ({disease_db_len} รายการ)\n\n"
        f"🧾 Top possible diagnoses:\n{diag_block}"
    )
    return text

# ----------------- Request/Response Models -----------------
class AnalyzeResponse(BaseModel):
    cropped_removed_black_border: bool
    grid_size: List[int]
    cell_counts: Dict[str, int]
    top_diagnoses: List[Dict[str, Any]]
    per_part_detections: List[Dict[str, Any]]
    preview_annotated_part1_jpeg_b64: Optional[str]
    csv_base64_utf8sig: str
    meta: Dict[str, Any]
    # ---- NEW ----
    disease_db_len: int
    summary_text_th: Optional[str]
    chart_bar: Dict[str, Any]
    chart_pie: Dict[str, Any]

# ----------------- API Endpoint -----------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(..., description="Microscope image (jpg/png)"),
    model: Optional[UploadFile] = File(None, description="YOLO .pt (optional if MODEL_PATH is set)"),
    disease_csv: Optional[UploadFile] = File(None, description="CSV with columns: name,conditions,details"),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    return_preview: bool = Form(True),
    # เปิด/ปิดการคืนสรุปข้อความแบบไทย (แนวเดียวกับตัวอย่าง)
    return_text_summary: bool = Form(True)
):
    """
    End-to-end pipeline returning everything as JSON + Thai summary text + chart numbers.
    """
    try:
        # 1) Read image
        raw_image = await image.read()
        img_bgr = read_image_to_bgr(raw_image)

        # 2) Remove black border (auto circle)
        cleaned_bgr, removed = remove_black_border_auto_circle(img_bgr)

        # 3) Zoom + Sharpen + Split 3x3
        parts, (rows, cols) = zoom_sharpen_and_split(cleaned_bgr, zoom_factor=3.0, rows=3, cols=3)

        # 4) Load YOLO
        model_bytes = await model.read() if model is not None else None
        yolo = load_model(model_bytes=model_bytes, model_path=None)
        names = getattr(yolo, "names", None)

        # 5) Detect + Count
        counts = counts_template()
        per_part = []
        preview_b64 = None

        for idx, pimg in enumerate(parts):
            # Run inference: PIL image path-less
            pimg_rgb = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(pimg_rgb)
            res = yolo.predict(source=pil_img, imgsz=imgsz, conf=conf, verbose=False)

            dets = detections_to_dict(res, names)
            # Update counts
            for d in dets:
                mapped = classify_label_to_key(d["label"])
                if mapped:
                    counts[mapped[0]] += 1

            # Annotate (only for preview part 1 to keep JSON light)
            if idx == 0 and return_preview:
                ann = annotate_from_dets(pimg, dets)
                preview_b64 = bgr_to_jpeg_base64(ann, quality=85)

            per_part.append({
                "part_index": idx + 1,
                "detections": dets
            })

        # 6) Disease DB
        if disease_csv is not None:
            disease_bytes = await disease_csv.read()
            disease_db = parse_disease_csv(disease_bytes)
        else:
            disease_db = []

        disease_db_len = len(disease_db)

        # 7) Softmax Rule-based Diagnosis
        top = softmax_rule_based(counts, disease_db, topk=5) if disease_db else []
        top_list = [{"name": n, "score": s, "details": d} for (n, s, d) in top]

        # 8) Export CSV (as base64 text)
        csv_b64 = to_csv_base64(counts, top)

        # 9) Build charts data
        # Bar chart over all 7 cell types
        bar_labels = list(counts.keys())
        bar_values = [int(counts[k]) for k in bar_labels]

        # Pie chart uses top diagnoses (scores already normalized 0..1)
        pie_labels = [n for (n, _, _) in top]
        pie_values = [float(s) for (_, s, _) in top]
        pie_percentages = [round(s * 100, 1) for s in pie_values]

        # 10) Optional Thai text summary
        summary_text = build_text_summary_th(counts, top, disease_db_len) if return_text_summary else None

        # 11) Response
        resp = {
            "cropped_removed_black_border": bool(removed),
            "grid_size": [rows, cols],
            "cell_counts": counts,
            "top_diagnoses": top_list,
            "per_part_detections": per_part,
            "preview_annotated_part1_jpeg_b64": preview_b64,
            "csv_base64_utf8sig": csv_b64,
            "meta": {
                "imgsz": imgsz,
                "conf": conf,
                "model_loaded_from": "upload" if model is not None else os.getenv("MODEL_PATH", "./best.pt"),
                "notes": "Preview only for part 1 to keep JSON compact."
            },
            # ---- NEW ----
            "disease_db_len": disease_db_len,
            "summary_text_th": summary_text,
            "chart_bar": {
                "labels": bar_labels,
                "values": bar_values
            },
            "chart_pie": {
                "labels": pie_labels,
                "values": pie_values,
                "percentages": pie_percentages
            }
        }
        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
