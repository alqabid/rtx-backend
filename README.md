# @rtx backend

## What's real now

`/generate-steps` does actual image analysis:

- **Portrait branch** (MediaPipe): if a face is detected taking up a
  meaningful part of the photo, it extracts real facial landmarks and
  returns guide paths for the face outline, eyebrows, eyes, nose, and
  lips — in that drawing order.
- **Object branch** (rembg + OpenCV): otherwise, it removes the
  background to isolate the subject, extracts the outer silhouette as
  step 1, then finds internal detail edges (largest first) as the
  remaining steps.

`/analyze-stroke` is **still fake** — it returns a random match/
correction/neutral result. That's the next thing to build (point-
distance / DTW comparison against the target guide path).

## First request will be slow

The MediaPipe face models (~a few MB) download automatically the first
time a portrait is processed, then get cached on disk (`models/`
folder) for every request after that. On Railway, this means: after a
fresh deploy, the *first* portrait upload will be slow (10-30s+
downloading + loading the model); every one after that should be fast.
The object branch doesn't have this issue — its model is tiny (~4.5MB)
and downloads almost instantly.

**Note on Railway's free/hobby tier:** if the service goes to sleep
from inactivity, you'll see the model-download delay again on the next
wake-up, on top of the normal cold-start delay. If this becomes
annoying, look at Railway's always-on / paid tier options, or add a
simple periodic ping to keep it warm.

## Run it locally (if you ever want to)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000/docs to test both endpoints by hand.

## What's next

- Real `/analyze-stroke`: point-distance scoring for fast client-side
  feedback, DTW-based scoring for the server-side "Check my step" call.
- Tuning: the object branch's internal-detail detection is a first
  pass (Canny edges + area sort) — it'll need real-photo testing to
  see if the detail count/ordering actually feels right to draw from.
