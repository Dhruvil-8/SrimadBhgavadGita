# rebuild_faiss_from_pickle.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

PICKLE_PATH = 'gita_full_corpus.pkl'
FAISS_PATH = 'gita_verse_index.faiss'
BACKUP_FAISS = 'gita_verse_index.faiss.bak.' + datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

# Backup index if exists
if os.path.exists(FAISS_PATH):
    os.rename(FAISS_PATH, BACKUP_FAISS)
    print("Backed up existing FAISS to", BACKUP_FAISS)

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

corpus = data['corpus']
verse_id_map = data['verse_id_map']  # list or dict

# Ensure ordered docs list: if verse_id_map is dict, sort by key.
if isinstance(verse_id_map, dict):
    ordered_ids = [verse_id_map[i] for i in sorted(verse_id_map.keys())]
else:
    ordered_ids = list(verse_id_map)

# Create document texts same as your generation script:
def first_sentences(text: str, n: int = 2):
    import re
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return " ".join([p.strip() for p in parts[:n] if p.strip()])

documents = []
for vid in ordered_ids:
    d = corpus[vid]
    vd = d['verse_details']
    ch_info = d.get('chapter_info', {})
    ch = vd.get('chapter_number')
    vs = vd.get('verse_number')
    ch_name = ch_info.get('name_translation', f"Chapter {ch}")
    ch_summary_snip = first_sentences(ch_info.get('chapter_summary', ''), 1)
    verse_text = vd.get('sanskrit_text', '')
    eng_trans = next((t['text'] for t in d['translations'] if t['language'] == 'english'), '')
    hindi_trans = next((t['text'] for t in d['translations'] if t['language'] == 'hindi'), '')
    eng_comms = [c['text'] for c in d['commentaries'] if c['language'] == 'english']
    comm_snip = " ".join([first_sentences(c, 2) for c in eng_comms[:3]])
    doc = (f"title: {ch_name} | Chapter {ch}, Verse {vs} | "
           f"chapter_context: {ch_summary_snip} || text: {verse_text} || translation: {eng_trans} || "
           f"hindi: {hindi_trans} || commentary_snippets: {comm_snip}")
    documents.append(doc)

# Load embedding model (use same model used in generation)
model_name = 'google/embeddinggemma-300M'
device = 'cpu'  # or 'cuda'
model = SentenceTransformer(model_name, device=device)
embeddings = model.encode(documents, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
embeddings = embeddings.astype('float32')

# L2 normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-8)

d = embeddings.shape[1]
base = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap(base)
ids = np.arange(len(embeddings)).astype('int64')
index.add_with_ids(embeddings, ids)
faiss.write_index(index, FAISS_PATH)
print("Rebuilt FAISS index and saved to", FAISS_PATH)
