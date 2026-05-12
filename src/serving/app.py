from fastapi import FastAPI

app = FastAPI(title="Support Query Classifier", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
