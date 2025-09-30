# engine.py (in app/ subdirectory)
from .gita_engine import GitaEngine  # Relative import within app/

def get_engine():
    if not hasattr(get_engine, 'instance'):
        get_engine.instance = GitaEngine()
    return get_engine.instance
engine = get_engine()

