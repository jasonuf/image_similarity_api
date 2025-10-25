import os
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import psycopg2
from io import BytesIO

# ------------------ CONFIG ------------------

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("❌ DATABASE_URL environment variable not found!")

# Default parameters
ALPHA_DEFAULT = float(os.getenv("ALPHA_DEFAULT", 0.7))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 5))

# ------------------ INIT --------------------
app = FastAPI(
    title="Art Similarity API",
    description="Find visually and semantically similar artworks using CLIP + pgvector.",
)

# Load CLIP model once on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Create persistent connection
conn = psycopg2.connect(DB_URL)


# ------------------ HELPERS -----------------

def get_image_embedding(file: UploadFile):
    """Generate normalized CLIP image embedding."""
    image = Image.open(BytesIO(file.file.read())).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().cpu().numpy().tolist()


def hybrid_search(vector, alpha=ALPHA_DEFAULT, top_k=TOP_K_DEFAULT):
    """Search top_k matches combining image+text embedding similarity."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT artist, title, year, style, filepath,
                   %s * (1 - (image_embedding <-> %s)) +
                   (1 - %s) * (1 - (text_embedding <-> %s)) AS hybrid_score
            FROM art_embeddings
            ORDER BY hybrid_score DESC
            LIMIT %s;
        """, (alpha, vector, alpha, vector, top_k))
        rows = cur.fetchall()

    results = []
    for rank, row in enumerate(rows, start=1):
        artist, title, year, style, filepath, score = row
        results.append({
            "rank": rank,
            "artist": artist,
            "title": title,
            "year": year,
            "style": style,
            "filepath": filepath,
            "score": float(score),
        })
    return results


# ------------------ ROUTES ------------------

@app.post("/search")
async def search_image(
    file: UploadFile = File(...),
    alpha: float = Query(ALPHA_DEFAULT, ge=0.0, le=1.0),
    top_k: int = Query(TOP_K_DEFAULT, gt=0)
):
    """
    Upload an image → returns top_k most similar artworks.
    """
    try:
        input_vec = get_image_embedding(file)
        results = hybrid_search(input_vec, alpha, top_k)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
def health_check():
    """Simple health endpoint for Railway uptime checks."""
    return {"status": "ok"}
