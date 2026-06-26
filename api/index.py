"""
Kenhub Narration Generator — Vercel Python Function。

api/index.py 在 Vercel 上默认暴露在 /api 路径（同时 /api/* 内部路由由 FastAPI 处理）。
public/ 目录由 Vercel 自动作为静态资源服务。
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("kenhub")

MINIMAX_BASE = "https://api.minimaxi.com"
MAX_TEXT_LEN = 50_000

DEFAULT_VOICE_ID = "moss_audio_972c74bb-47a5-11f1-915a-0a14aa2b7ca7"
DEFAULT_MODEL = "speech-2.8-hd"
ALLOWED_MODELS = {
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
}

app = FastAPI(title="Kenhub Narration Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 校验 ----------


def _validate_tone(raw) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="pronunciation_dict 必须是对象")
    tone = raw.get("tone")
    if tone is None:
        return []
    if not isinstance(tone, list):
        raise HTTPException(status_code=422, detail="pronunciation_dict.tone 必须是字符串数组")

    cleaned: list[str] = []
    for idx, item in enumerate(tone):
        if not isinstance(item, str):
            raise HTTPException(status_code=422, detail=f"pronunciation_dict.tone[{idx}] 必须是字符串")
        parts = item.split("/")
        if len(parts) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"pronunciation_dict.tone[{idx}] 必须是「原文/替换」格式（恰好一个 /）",
            )
        word, repl = parts[0].strip(), parts[1].strip()
        if not word or not repl:
            raise HTTPException(
                status_code=422,
                detail=f"pronunciation_dict.tone[{idx}] 的原文和替换均不能为空",
            )
        cleaned.append(f"{word}/{repl}")
    return cleaned


def _validate_synthesize(body: dict) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="请求体必须是 JSON 对象")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="text 必填且不能为空")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(status_code=422, detail=f"text 超过 {MAX_TEXT_LEN} 字符上限")

    voice_id = body.get("voice_id") or DEFAULT_VOICE_ID
    if not isinstance(voice_id, str) or not voice_id.strip():
        voice_id = DEFAULT_VOICE_ID

    model = body.get("model") or DEFAULT_MODEL
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=422, detail=f"不支持的 model: {model}")

    def _float(name: str, default: float, lo: float, hi: float) -> float:
        v = body.get(name, default)
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{name} 必须是数字")
        if not (lo <= f <= hi):
            raise HTTPException(status_code=422, detail=f"{name} 必须在 [{lo}, {hi}] 区间")
        return f

    def _int(name: str, default: int, lo: int, hi: int) -> int:
        v = body.get(name, default)
        try:
            i = int(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{name} 必须是整数")
        if not (lo <= i <= hi):
            raise HTTPException(status_code=422, detail=f"{name} 必须在 [{lo}, {hi}] 区间")
        return i

    tone = _validate_tone(body.get("pronunciation_dict"))

    return {
        "text": text,
        "voice_id": voice_id.strip(),
        "model": model,
        "speed": _float("speed", 1.0, 0.5, 2.0),
        "vol": _float("vol", 1.0, 0.0, 10.0),
        "pitch": _int("pitch", 0, -12, 12),
        "tone": tone,
    }


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少 Authorization: Bearer <token> Header")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authorization Header 中的 token 为空")
    return token


async def _raise_for_minimax(resp: httpx.Response, op: str) -> None:
    if resp.is_success:
        return
    detail = f"MiniMax {op} HTTP {resp.status_code}"
    try:
        body = resp.json()
        msg = (body.get("base_resp") or {}).get("status_msg")
        if msg:
            detail = f"{detail}: {msg}"
    except Exception:
        pass
    log.warning("%s failed: %s body=%s", op, detail, resp.text[:300])
    raise HTTPException(status_code=resp.status_code, detail=detail)


# ---------- 路由 ----------


@app.get("/")
async def root():
    return {
        "service": "kenhub-narration-generator",
        "status": "ok",
        "endpoints": ["/api/health", "/api/synthesize (POST)", "/api/status/{task_id}", "/api/download/{file_id}"],
    }


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "kenhub-narration-generator"}


@app.post("/api/synthesize")
async def synthesize(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer(authorization)

    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="请求体不是合法 JSON")
    body = _validate_synthesize(raw)

    payload = {
        "model": body["model"],
        "text": body["text"],
        "voice_setting": {
            "voice_id": body["voice_id"],
            "speed": body["speed"],
            "vol": body["vol"],
            "pitch": body["pitch"],
        },
        "audio_setting": {
            "audio_sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
        "language_boost": "auto",
    }
    if body["tone"]:
        payload["pronunciation_dict"] = {"tone": body["tone"]}

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{MINIMAX_BASE}/v1/t2a_async_v2", json=payload, headers=headers)
    await _raise_for_minimax(resp, "t2a_async_v2")

    data = resp.json()
    base_resp = data.get("base_resp") or {}
    if base_resp.get("status_code") not in (0, None):
        raise HTTPException(status_code=502, detail=base_resp.get("status_msg", "MiniMax 返回错误"))

    task_id = data.get("task_id")
    if not task_id:
        raise HTTPException(status_code=502, detail="MiniMax 未返回 task_id")

    log.info("created task %s (voice=%s, model=%s, chars=%d, tone=%d)",
             task_id, body["voice_id"], body["model"], len(body["text"]), len(body["tone"]))

    return {"task_id": task_id, "usage_characters": data.get("usage_characters", len(body["text"]))}


@app.get("/api/status/{task_id}")
async def status(task_id: str, authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer(authorization)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{MINIMAX_BASE}/v1/query/t2a_async_query_v2",
            params={"task_id": task_id},
            headers=headers,
        )
    await _raise_for_minimax(resp, "t2a_async_query_v2")

    data = resp.json()
    raw_status = (data.get("status") or "").lower()
    mapping = {
        "processing": "processing",
        "success": "finished",
        "failed": "failed",
        "expired": "failed",
    }
    return {
        "status": mapping.get(raw_status, "processing"),
        "raw_status": raw_status,
        "file_id": data.get("file_id"),
    }


@app.get("/api/download/{file_id}")
async def download(file_id: int, authorization: Optional[str] = Header(default=None)):
    """流式下载音频。Vercel Hobby 默认 10s 超时。"""
    token = _extract_bearer(authorization)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        meta_resp = await client.get(
            f"{MINIMAX_BASE}/v1/files/retrieve",
            params={"file_id": file_id},
            headers=headers,
        )
    await _raise_for_minimax(meta_resp, "files/retrieve")

    meta = meta_resp.json()
    download_url = (meta.get("file") or {}).get("download_url")
    if not download_url:
        raise HTTPException(status_code=502, detail="MiniMax 未返回 download_url")

    async def iter_audio() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", download_url) as r:
                if not r.is_success:
                    await r.aread()
                    raise HTTPException(status_code=r.status_code, detail="下载音频失败")
                async for chunk in r.aiter_bytes(chunk_size=64 * 1024):
                    yield chunk

    filename = f"kenhub_narration_{file_id}.mp3"
    return StreamingResponse(
        iter_audio(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------- 错误兜底 ----------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
