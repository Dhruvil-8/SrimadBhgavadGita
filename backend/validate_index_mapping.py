# validate_index_mapping.py
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

FAISS_PATH = 'gita_verse_index.faiss'
PICKLE_PATH = 'gita_full_corpus.pkl'
MODEL_NAME = 'google/embeddinggemma-300M'

idx = faiss.read_index(FAISS_PATH)
with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

verse_id_map = data['verse_id_map']
if isinstance(verse_id_map, dict):
    ordered_ids = [verse_id_map[i] for i in sorted(verse_id_map.keys())]
else:
    ordered_ids = list(verse_id_map)

print("Loaded", len(ordered_ids), "verse ids. Index ntotal:", idx.ntotal)
assert idx.ntotal == len(ordered_ids), "Index size and verse_id_map length mismatch!"

# Load model and run a sample query
model = SentenceTransformer(MODEL_NAME, device='cpu')
query = "what is dharma?"
q_emb = model.encode([query], convert_to_numpy=True)
# normalize
q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
D, I = idx.search(q_emb.astype('float32'), 5)
print("Top 5 indices:", I[0])
print("Corresponding verse ids:", [ordered_ids[i] for i in I[0]])
