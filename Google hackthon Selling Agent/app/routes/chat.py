import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="


class ChatRequest(BaseModel):
    message: str
    shop: str = "this stand"
    inventory: str = ""


@router.post("/message")
async def chat_message(req: ChatRequest):
    key = os.getenv("GEMINI_API_KEY", "AIzaSyBUVo-kDUCEX7zESFvYyVuO1VNb0ASr_oM")
    url = _GEMINI_BASE + key
    system_ctx = (
        f'You are Midori, a cheerful AI sales assistant for VendorGroove Net — '
        f'a smart platform for US vegetable street vendors. '
        f'The vendor\'s shop is "{req.shop}". '
        f'Current inventory: {req.inventory}. '
        f'Help with: restocking decisions, pricing, sales trends, group buying, and produce tips. '
        f'Be concise (2-3 sentences), warm, and practical. '
        f'Respond in the same language the user uses.'
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_ctx}]},
        "contents": [{"role": "user", "parts": [{"text": req.message}]}],
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(url, json=payload)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Gemini error {res.status_code}: {res.text[:200]}")
        data = res.json()
        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"reply": reply}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini request timed out")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Gemini response format: {e}")
