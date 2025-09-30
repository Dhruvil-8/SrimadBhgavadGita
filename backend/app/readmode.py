# readmode.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from .engine import engine  # Shared engine instance
import os
import copy
import asyncio # <<< NEW: Import the asyncio library

router = APIRouter()

# Path to your data folder (for audio checks)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# --- Helper functions (These do not need to be async) ---
# NOTE: Using the optimized lookup functions we discussed previously.
def get_chapters() -> List[Dict[str, Any]]:
    return list(engine.chapters.values())

def get_chapter_verses(chapter: int) -> List[Dict[str, Any]]:
    # This assumes you have implemented the optimized chapter_to_verse_ids lookup dict
    if hasattr(engine, 'chapter_to_verse_ids'):
        verse_ids = engine.chapter_to_verse_ids.get(chapter, [])
        return [copy.deepcopy(engine.corpus[vid]) for vid in verse_ids]
    else: # Fallback to the old, slower method if not implemented
        ch_verses = []
        for verse_id, v in engine.corpus.items():
            if v.get('verse_details', {}).get('chapter_number') == chapter:
                ch_verses.append(copy.deepcopy(v))
        ch_verses.sort(key=lambda x: x.get('verse_details', {}).get('verse_number', 0))
        return ch_verses

def get_single_verse(chapter: int, verse: int) -> Dict[str, Any] | None:
    # This assumes you have implemented the optimized verse_id_to_verse lookup dict
    if hasattr(engine, 'verse_id_to_verse'):
        key = f"{chapter}.{verse}"
        v = engine.verse_id_to_verse.get(key)
        return copy.deepcopy(v) if v else None
    else: # Fallback to the old, slower method
        for verse_id, v in engine.corpus.items():
            details = v.get('verse_details', {})
            if details.get('chapter_number') == chapter and details.get('verse_number') == verse:
                return copy.deepcopy(v)
        return None

# --- Asynchronous API Endpoints ---

@router.get("/chapters")
async def list_chapters(): # <<< CHANGED: from def to async def
    """
    Asynchronously get list of chapters with metadata.
    """
    chapters_list = await asyncio.to_thread(get_chapters)
    if not chapters_list:
        raise HTTPException(status_code=404, detail="No chapters found in corpus")
    return {"chapters": chapters_list}

@router.get("/chapter/{chapter}")
async def read_chapter(chapter: int): # <<< CHANGED: from def to async def
    """
    Asynchronously get all verses in a chapter.
    """
    verses = await asyncio.to_thread(get_chapter_verses, chapter)
    if not verses:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    # This metadata attachment is also a sync operation, so best to include it in the thread
    def attach_metadata():
        ch_info = engine.chapters.get(chapter, {})
        verses[0]['chapter_name'] = ch_info.get('name_translation', f"Chapter {chapter}")
        verses[0]['chapter_summary'] = ch_info.get('chapter_summary', '')
        verses[0]['verses_count'] = ch_info.get('verses_count', len(verses))
        return verses

    verses_with_meta = await asyncio.to_thread(attach_metadata)
    return verses_with_meta

@router.get("/verse/{chapter}/{verse}")
async def read_verse(chapter: int, verse: int): # <<< CHANGED: from def to async def
    """
    Asynchronously get a specific verse by chapter & verse number.
    """
    v = await asyncio.to_thread(get_single_verse, chapter, verse)
    if not v:
        raise HTTPException(status_code=404, detail="Verse not found")

    # This file check is a blocking I/O operation, perfect for to_thread
    def check_audio_and_get_url():
        audio_path = os.path.join(DATA_DIR, "audio", f"{chapter}-{verse}.mp3")
        if os.path.exists(audio_path):
            return f"/static/audio/{chapter}-{verse}.mp3"
        return None

    audio_url = await asyncio.to_thread(check_audio_and_get_url)
    if audio_url:
        v["audio_url"] = audio_url

    return v