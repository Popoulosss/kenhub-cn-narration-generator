from fastapi import FastAPI

app = FastAPI()


@app.get("/")
@app.get("/api/hello")
async def hello():
    return {"hello": "world", "service": "kenhub-debug"}
