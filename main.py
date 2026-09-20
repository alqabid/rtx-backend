"""
@rtx backend — real /generate-steps implementation.

Two branches, chosen automatically per upload:
  - Portrait branch: MediaPipe face detection + face landmarks. Produces
    guide paths for the face oval, eyebrows, eyes, nose, and lips, in
    that drawing order.
  - Object branch: rembg (U^2-Net, lightweight) isolates the subject from
    the background, then OpenCV contour extraction produces an outer
    silhouette step followed by internal detail steps, largest first.

/analyze-stroke is still the fake stub for now — that's the next piece.

Run it with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

First request will be slow (downloading the MediaPipe models, a few MB) —
after that they're cached on disk and every request is fast.
"""

import io
import os
from collections import defaultdict
from typing import List, Optional
from urllib.request import urlretrieve

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from rembg import remove, new_session
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

app = FastAPI(title="@rtx backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Shapes mirror src/types.ts in the frontend ---

class GuidePath(BaseModel):
    id: str
    points: List[float]
    label: Optional[str] = None


class DrawingStep(BaseModel):
    id: int
    title: str
    instruction: str
    anatomicalTarget: str
    guidePaths: List[GuidePath]
    isCompleted: bool = False


class StepFeedback(BaseModel):
    type: str
    title: str
    message: str
    score: float


# --- Lazily-loaded models — downloaded once on first request, cached after ---

MODEL_DIR = "models"
FACE_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

_face_detector = None
_face_landmarker = None
_rembg_session = None


def _ensure_model(url: str, filename: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        urlretrieve(url, path)
    return path


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        model_path = _ensure_model(FACE_DETECTOR_URL, "blaze_face_short_range.tflite")
        options = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path))
        _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector


def get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        model_path = _ensure_model(FACE_LANDMARKER_URL, "face_landmarker.task")
        options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1)
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2netp")
    return _rembg_session


# --- Portrait branch ---

def connections_to_polylines(connections) -> List[List[int]]:
    """
    MediaPipe gives face regions as unordered lists of (start, end) index
    pairs, not ready-to-draw polylines — and some regions (lips, eyebrows)
    are actually two separate loops/rows, not one. This walks the edge
    graph and returns one ordered list of landmark indices per connected
    piece, so each becomes its own drawable guide path.
    """
    adj = defaultdict(set)
    for c in connections:
        adj[c.start].add(c.end)
        adj[c.end].add(c.start)

    unvisited_edges = {tuple(sorted((c.start, c.end))) for c in connections}
    remaining_nodes = set(adj.keys())
    polylines = []

    while remaining_nodes:
        candidates = [
            n for n in remaining_nodes
            if any(tuple(sorted((n, nb))) in unvisited_edges for nb in adj[n])
        ]
        if not candidates:
            break
        start = next(
            (n for n in candidates
             if sum(1 for nb in adj[n] if tuple(sorted((n, nb))) in unvisited_edges) == 1),
            candidates[0],
        )
        ordered = [start]
        current = start
        while True:
            nxt = None
            for neighbor in adj[current]:
                edge = tuple(sorted((current, neighbor)))
                if edge in unvisited_edges:
                    nxt = neighbor
                    unvisited_edges.discard(edge)
                    break
            if nxt is None:
                break
            ordered.append(nxt)
            current = nxt
        polylines.append(ordered)
        remaining_nodes = {
            n for n in remaining_nodes
            if any(tuple(sorted((n, nb))) in unvisited_edges for nb in adj[n])
        }
    return polylines


def is_portrait(pil_image: Image.Image) -> bool:
    w, h = pil_image.size
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_image))
    result = get_face_detector().detect(mp_image)
    if not result.detections:
        return False
    best = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    face_area_ratio = (best.bounding_box.width * best.bounding_box.height) / (w * h)
    return face_area_ratio > 0.03  # face fills at least ~3% of the frame


def generate_portrait_steps(pil_image: Image.Image) -> List[DrawingStep]:
    w, h = pil_image.size
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_image))
    result = get_face_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return []
    landmarks = result.face_landmarks[0]

    flc = vision.FaceLandmarksConnections
    regions = [
        ("Block in the head shape", "Face silhouette", flc.FACE_LANDMARKS_FACE_OVAL),
        ("Place the eyebrows", "Eyebrows",
         list(flc.FACE_LANDMARKS_LEFT_EYEBROW) + list(flc.FACE_LANDMARKS_RIGHT_EYEBROW)),
        ("Draw the eyes", "Eyes",
         list(flc.FACE_LANDMARKS_LEFT_EYE) + list(flc.FACE_LANDMARKS_RIGHT_EYE)),
        ("Add the nose", "Nose", flc.FACE_LANDMARKS_NOSE),
        ("Draw the mouth", "Lips", flc.FACE_LANDMARKS_LIPS),
    ]

    steps = []
    for i, (title, target, connections) in enumerate(regions):
        polylines = connections_to_polylines(connections)
        guide_paths = []
        for j, poly in enumerate(polylines):
            flat: List[float] = []
            for idx in poly:
                lm = landmarks[idx]
                flat.extend([lm.x * w, lm.y * h])
            guide_paths.append(GuidePath(id=f"step{i+1}-p{j+1}", points=flat, label=target))
        steps.append(DrawingStep(
            id=i + 1,
            title=title,
            instruction=f"Focus on the {target.lower()} — follow the guide line closely.",
            anatomicalTarget=target,
            guidePaths=guide_paths,
        ))
    return steps


# --- Object branch ---

def generate_object_steps(pil_image: Image.Image) -> List[DrawingStep]:
    arr_rgb = np.array(pil_image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=get_rembg_session())
    out_arr = np.array(Image.open(io.BytesIO(out_bytes)))

    alpha = out_arr[:, :, 3] if out_arr.shape[2] == 4 else np.full(out_arr.shape[:2], 255, dtype=np.uint8)
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    outer_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outer_contours:
        return []
    outer = max(outer_contours, key=cv2.contourArea)

    # Internal detail: edges on the raw image, restricted afterwards to an
    # eroded interior region so the outer boundary itself isn't re-detected.
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    eroded = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=2)
    edges_interior = cv2.bitwise_and(edges, edges, mask=eroded)
    edges_interior = cv2.dilate(edges_interior, np.ones((3, 3), np.uint8), iterations=1)

    internal_contours, _ = cv2.findContours(edges_interior, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    internal_contours = [c for c in internal_contours if cv2.contourArea(c) > 40]
    internal_sorted = sorted(internal_contours, key=cv2.contourArea, reverse=True)[:5]

    def to_guide_path(c, id_: str, label: str) -> GuidePath:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        flat = [float(v) for pt in approx.reshape(-1, 2) for v in pt]
        return GuidePath(id=id_, points=flat, label=label)

    steps = [DrawingStep(
        id=1,
        title="Block in the main shape",
        instruction="Start with the overall outline of the subject.",
        anatomicalTarget="Outer silhouette",
        guidePaths=[to_guide_path(outer, "step1-p1", "silhouette")],
    )]
    for i, c in enumerate(internal_sorted):
        steps.append(DrawingStep(
            id=i + 2,
            title=f"Add internal detail {i + 1}",
            instruction="Draw this internal shape on top of your blocked-in outline.",
            anatomicalTarget=f"Internal detail {i + 1}",
            guidePaths=[to_guide_path(c, f"step{i+2}-p1", f"detail-{i+1}")],
        ))
    return steps


# --- Endpoints ---

@app.post("/generate-steps", response_model=List[DrawingStep])
async def generate_steps(image: UploadFile = File(...)):
    raw = await image.read()
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    pil_image.thumbnail((900, 900))  # keep processing fast on Railway's CPU

    try:
        if is_portrait(pil_image):
            steps = generate_portrait_steps(pil_image)
            if steps:
                return steps
    except Exception:
        pass  # any face-pipeline failure falls through to the object branch

    return generate_object_steps(pil_image)


# --- Still a stub — next piece to build ---

import random  # noqa: E402


@app.post("/analyze-stroke", response_model=StepFeedback)
async def analyze_stroke(step_id: int, points: List[float]):
    choice = random.choice(["match", "correction", "neutral"])
    messages = {
        "match": ("Well matched — moving on", "Your line lines up well with the guide."),
        "correction": ("Try adjusting the shape", "Your line drifts from the guide on one side."),
        "neutral": ("Keep going", "You're on the right track — a bit more to go."),
    }
    title, message = messages[choice]
    return StepFeedback(type=choice, title=title, message=message, score=round(random.uniform(0.6, 0.98), 2))


@app.get("/")
async def root():
    return {"status": "ok", "note": "@rtx backend — real /generate-steps, /analyze-stroke still stubbed."}
