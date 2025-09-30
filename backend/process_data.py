import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import time
import json
import os
import re  # For text cleaning

print("--- Enhanced Multilingual Data Processing with Chapters Integration ---")

# --- 1. Load All JSON Data ---
print("Loading JSON files...")
data_dir = 'data'
required_files = ['verse.json', 'translation.json', 'commentary.json', 'chapters.json']
for file in required_files:
    if not os.path.exists(os.path.join(data_dir, file)):
        raise FileNotFoundError(f"Missing {file} in {data_dir}. Download from https://github.com/gita/gita/tree/main/data")

try:
    with open(os.path.join(data_dir, 'verse.json'), 'r', encoding='utf-8-sig') as f:
        verses_data = json.load(f)
    with open(os.path.join(data_dir, 'translation.json'), 'r', encoding='utf-8-sig') as f:
        translations_data = json.load(f)
    with open(os.path.join(data_dir, 'commentary.json'), 'r', encoding='utf-8-sig') as f:
        commentaries_data = json.load(f)
    with open(os.path.join(data_dir, 'chapters.json'), 'r', encoding='utf-8-sig') as f:
        chapters_data = json.load(f)
    print("Successfully loaded all JSON files.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# --- 2. Build Chapters Dict ---
print("Building chapters metadata...")
chapters = {ch['chapter_number']: ch for ch in chapters_data}
print(f"Loaded {len(chapters)} chapters with summaries and names.")

# --- 3. Build the Rich, Multi-Lingual Corpus ---
print("Building the rich, multi-lingual corpus...")
corpus = {}

for verse in verses_data:
    verse_id = verse['id']  # e.g., "1.1"
    ch_num = verse['chapter_number']
    ch_info = chapters.get(ch_num, {})  # Fallback empty if mismatch
    corpus[verse_id] = {
        'verse_details': {
            'chapter_number': ch_num,
            'verse_number': verse['verse_number'],
            'sanskrit_text': verse['text'],
            'transliteration': verse['transliteration'],
            'word_meanings': verse.get('word_meanings', [])  # Preserve if available
        },
        'chapter_info': ch_info,  # Embed full chapter metadata here for per-verse access
        'translations': [],
        'commentaries': [],
        'tags': []  # Placeholder; add from data if available (e.g., themes like "duty", "devotion")
    }

# Add translations
for trans in translations_data:
    verse_id = trans['verse_id']
    if verse_id in corpus:
        corpus[verse_id]['translations'].append({
            'author': trans['authorName'],
            'language': trans['lang'],
            'text': trans['description']
        })

# Add commentaries
for comm in commentaries_data:
    verse_id = comm['verse_id']
    if verse_id in corpus:
        corpus[verse_id]['commentaries'].append({
            'author': comm['authorName'],
            'language': comm['lang'],
            'text': comm['description']
        })

print(f"Corpus built successfully with {len(corpus)} verses and chapter info.")

# --- 4. Create Documents for Embedding (Now with Chapter Context) ---
print("Creating documents for embedding...")
documents_for_embedding = []
verse_id_map = []

def first_sentences(text: str, n: int = 2) -> str:
    """Extract first N sentences, matching engine."""
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return " ".join([p.strip() for p in parts[:n] if p.strip()])

for verse_id, data in corpus.items():
    vd = data['verse_details']
    ch_info = data.get('chapter_info', {})
    ch = vd['chapter_number']
    vs = vd['verse_number']
    ch_name = ch_info.get('name_translation', f"Chapter {ch}")  # Use proper English name
    ch_summary_snip = first_sentences(ch_info.get('chapter_summary', ''), 1)  # Snippet for context
    verse_text = vd['sanskrit_text']  # Use Sanskrit as 'text'
    eng_trans = next((t['text'] for t in data['translations'] if t['language'] == 'english'), '')
    hindi_trans = next((t['text'] for t in data['translations'] if t['language'] == 'hindi'), '')
    eng_comms = [c['text'] for c in data['commentaries'] if c['language'] == 'english']
    comm_snip = " ".join([first_sentences(c, 2) for c in eng_comms[:3]])
    tags = " ".join(data.get('tags', []))  # Empty for now
    
    # Enhanced: Include chapter summary snippet for broader thematic retrieval
    doc = (f"title: {ch_name} | {ch_info.get('name', '')} | Chapter {ch}, Verse {vs} | "
           f"chapter_context: {ch_summary_snip} || text: {verse_text} || translation: {eng_trans} || "
           f"hindi: {hindi_trans} || tags: {tags} || commentary_snippets: {comm_snip}")
    documents_for_embedding.append(doc)
    verse_id_map.append(verse_id)

print(f"Created {len(documents_for_embedding)} documents for embedding (with chapter context).")

# --- 5. Generate Embeddings with EmbeddingGemma (Using Proper Prompts) ---
print("Loading EmbeddingGemma model...")
model_name = 'google/embeddinggemma-300M'
device = "cpu"  # Change to "cuda" if GPU available
print(f"Using embedding model: {model_name} on device: {device}")
model = SentenceTransformer(model_name, device=device)

print("Generating embeddings for documents (using Retrieval-document prompt)...")
start_time = time.time()
# Key: Use prompt_name for documents
embeddings = model.encode(
    documents_for_embedding,
    prompt_name='Retrieval-document',  # Optimal for docs
    show_progress_bar=True,
    batch_size=32,  # Efficient batching
    convert_to_numpy=True
)
end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

# Normalize (L2) for cosine similarity
embeddings = np.array(embeddings).astype('float32')
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-8)  # Safe normalization

embedding_dimension = embeddings.shape[1]
# Use IndexFlatIP for cosine (inner product on normalized vectors)
index = faiss.IndexFlatIP(embedding_dimension)
index.add(embeddings)

print(f"FAISS index created successfully with {index.ntotal} vectors (cosine-optimized).")

# --- 6. Save the Output Files ---
print("Saving the FAISS index and the enhanced corpus to disk...")
faiss.write_index(index, 'gita_verse_index.faiss')

corpus_to_save = {
    'verse_id_map': verse_id_map,  # List of str IDs
    'corpus': corpus,  # Per-verse data with chapter_info embedded
    'chapters': chapters  # Top-level dict for quick chapter lists/summaries
}

with open('gita_full_corpus.pkl', 'wb') as f:
    pickle.dump(corpus_to_save, f)

print("\n--- Enhanced Processing Complete! ---")
print("Files created:")
print("1. gita_verse_index.faiss - Cosine-optimized index with chapter context.")
print("2. gita_full_corpus.pkl - Self-contained corpus (verses + chapters + metadata).")