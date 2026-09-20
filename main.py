"""
@rtx stub backend.

This is FAKE — it doesn't do any real image analysis yet. It just returns
hardcoded data in the exact shape the frontend already expects, so you can
plug it in and see the whole app work end-to-end before any real AI is built.

Run it with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs to see and test the endpoints in your
browser.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random

app = FastAPI(title="@rtx backend (stub)")

# Allow the frontend (running on localhost:3000) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev, tighten before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- These shapes mirror src/types.ts in the frontend exactly ---

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
    type: str  # "match" | "correction" | "neutral"
    title: str
    message: str
    score: float


# --- Fake step data (stand-in for real segmentation + contour extraction) ---

FAKE_STEPS = [
    DrawingStep(
        id=1,
        title="Block in the main shape",
        instruction="Start with the overall outline — don't worry about details yet.",
        anatomicalTarget="Outer silhouette",
        guidePaths=[GuidePath(id="p1", points=[100, 100, 300, 100, 300, 300, 100, 300], label="silhouette")],
    ),
    DrawingStep(
        id=2,
        title="Add the major internal shapes",
        instruction="Break the main shape into its biggest internal parts.",
        anatomicalTarget="Primary internal masses",
        guidePaths=[GuidePath(id="p2", points=[150, 150, 250, 150, 250, 250, 150, 250], label="core mass")],
    ),
    DrawingStep(
        id=3,
        title="Refine the details",
        instruction="Now add the smaller details on top of your blocked-in shapes.",
        anatomicalTarget="Fine detail contours",
        guidePaths=[GuidePath(id="p3", points=[180, 180, 220, 180, 220, 220, 180, 220], label="detail")],
    ),
]


@app.post("/generate-steps", response_model=List[DrawingStep])
async def generate_steps(image: UploadFile = File(...)):
    """
    Real version will: run segmentation + contour extraction (or MediaPipe
    for portraits) on `image` and return a real ordered step list.
    For now: ignores the image and returns the same fake steps every time.
    """
    return FAKE_STEPS


@app.post("/analyze-stroke", response_model=StepFeedback)
async def analyze_stroke(step_id: int, points: List[float]):
    """
    Real version will: compare `points` against the target guide path for
    `step_id` using point-distance / DTW and return real feedback.
    For now: returns a random one of the three feedback types.
    """
    choice = random.choice(["match", "correction", "neutral"])
    messages = {
        "match": ("Well matched — moving on", "Your line lines up well with the guide."),
        "correction": ("Try adjusting the shape", "Your line drifts from the guide on one side."),
        "neutral": ("Keep going", "You're on the right track — a bit more to go."),
    }
    title, message = messages[choice]
    return StepFeedback(
        type=choice,
        title=title,
        message=message,
        score=round(random.uniform(0.6, 0.98), 2),
    )


@app.get("/")
async def root():
    return {"status": "ok", "note": "This is the @rtx stub backend — fake data only."}
