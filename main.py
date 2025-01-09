import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tts import TTSSystem
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(
    title="Text-to-Speech API",
    description="API for converting text to speech using Tacotron2 and HiFi-GAN",
    version="1.0.0"
)

# Create thread pool for running TTS
# executor = ThreadPoolExecutor(max_workers=2)


@app.get("/")
async def root():
    return {"message": "Hello TTS Emtelco"}

# Request model
class TTSRequest(BaseModel):
    text: str
    use_pronunciation: Optional[bool] = True
    superres_strength: Optional[float] = 1.0

@app.post("/tts")
async def tts(request: TTSRequest):
    # Initialize TTS system
    """
    Convert text to speech and return an audio ID.
    """
    tts = TTSSystem(superres_strength=1.0)

    try:
        tts.tts_excecute(request.text, request.use_pronunciation)
        return{"statusCode" : 200 ,"message": "Audio file correct generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
