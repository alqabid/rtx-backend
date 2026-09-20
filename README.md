# @rtx stub backend

This is a **fake** backend. It doesn't do any real image analysis — it just
sends back hardcoded data shaped exactly like the real thing will be, so you
can see the frontend and backend actually talking to each other.

## Run it (3 commands)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000/docs** in your browser — you'll see the two
endpoints (`/generate-steps`, `/analyze-stroke`) and can test them by hand.

## Connect it to the frontend

In the frontend repo, set this in `.env`:

```
VITE_API_BASE_URL=http://localhost:8000
```

Then update `DrawingContext.tsx` to `fetch()` from `${import.meta.env.VITE_API_BASE_URL}/generate-steps`
instead of reading from `mockData.ts` directly.

## What's fake vs. what's next

| Endpoint          | Right now                          | Next step                                      |
|--------------------|-------------------------------------|-------------------------------------------------|
| `/generate-steps`  | Always returns the same 3 fake steps | Real segmentation + contour extraction on the uploaded image |
| `/analyze-stroke`  | Returns a random feedback type      | Real point-distance/DTW comparison against the guide path |

Nothing here needs to be "smart" yet — the whole point of this stub is to
prove the plumbing works before we build the hard parts.
