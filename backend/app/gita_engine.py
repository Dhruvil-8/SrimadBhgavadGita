# Improved GitaEngine Hybrid with IndexIDMap, BM25, and RRF fusion

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import google.generativeai as genai
import math
import re
import logging
from collections import defaultdict
from functools import lru_cache
import json
from datetime import datetime

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GitaEngineHybrid")

# --- CONFIG ---
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "google/embeddinggemma-300M")
CROSS_ENCODER_ID = os.getenv("CROSS_ENCODER_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "gita_verse_index.faiss")
CORPUS_PICKLE_PATH = os.getenv("CORPUS_PICKLE_PATH", "gita_full_corpus.pkl")
EMBEDDING_DIM: Optional[int] = None  # will be inferred from model output
K_RETRIEVE = int(os.getenv("K_RETRIEVE", "5"))  # Balanced for speed/recall
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
DEVICE = os.getenv("DEVICE", "cpu")
TFIDF_TOP_K = int(os.getenv("TFIDF_TOP_K", "5"))
BATCH_SIZE_EMBED = int(os.getenv("BATCH_SIZE_EMBED", "32"))
TELEMETRY_PATH = os.getenv("TELEMETRY_PATH", "gita_telemetry.jsonl")

# Gemini config
if "GOOGLE_API_KEY" in os.environ:
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])  # safe: will raise if key invalid
        logger.info("Gemini configured successfully.")
    except Exception as e:
        logger.warning("Could not configure Gemini: %s", e)
else:
    logger.info("GOOGLE_API_KEY not set; Gemini-related features will use fallbacks.")

# ---------- Utility functions ----------

def safe_split_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r'^[\-\u2022\s\d\.\)\(]*', '', s).strip()
        if s and len(s) > 2:
            lines.append(s)
    return lines


def l2_normalize(a: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return a / norm


# ---------- Helper algorithms: RRF and MMR (small implementations) ----------

def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 10, phi: float = 60.0) -> List[int]:
    """Simple Reciprocal Rank Fusion implementation.
    rank_lists: list of ranked doc id lists (best first).
    returns top-k fused ids.
    """
    scores = defaultdict(float)
    for rlist in rank_lists:
        for rank, doc in enumerate(rlist, start=1):
            scores[doc] += 1.0 / (phi + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]


def mmr_selection(candidate_embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int = 5, lambda_coef: float = 0.7) -> List[int]:
    """Return indices of candidate_embeddings selected by MMR."""
    if candidate_embeddings.size == 0:
        return []
    sims_to_query = (candidate_embeddings @ query_embedding.T).flatten()
    selected = []
    candidates = list(range(candidate_embeddings.shape[0]))
    while len(selected) < top_k and candidates:
        mmr_scores = []
        for c in candidates:
            diversity = 0.0
            if selected:
                diversity = max((candidate_embeddings[c] @ candidate_embeddings[s].T) for s in selected)
            mmr_scores.append(lambda_coef * sims_to_query[c] - (1 - lambda_coef) * diversity)
        best_idx = int(np.argmax(mmr_scores))
        best = candidates[best_idx]
        selected.append(best)
        candidates.remove(best)
    return selected


# ---------- GitaEngine class (hybrid) ----------
class GitaEngine:
    def __init__(self,
                 embedding_model_id: str = EMBEDDING_MODEL_ID,
                 cross_encoder_id: str = CROSS_ENCODER_ID,
                 index_path: str = FAISS_INDEX_PATH,
                 corpus_pickle: str = CORPUS_PICKLE_PATH,
                 device: str = DEVICE):
        logger.info("Initializing GitaEngineHybrid...")
        self.device = device

        # load sentence-transformer embedding model (fallback safe)
        try:
            logger.info("Loading embedding model: %s", embedding_model_id)
            self.embed_model = SentenceTransformer(embedding_model_id, device=self.device)
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()  # Explicitly set
        except Exception as e:
            logger.exception("Failed to load embedding model '%s' - ensure model name available. Falling back to 'google/embeddinggemma-300M'.", embedding_model_id)
            self.embed_model = SentenceTransformer("google/embeddinggemma-300M", device=self.device)
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        # cross-encoder
        try:
            logger.info("Loading cross-encoder: %s", cross_encoder_id)
            self.cross_encoder = CrossEncoder(cross_encoder_id, device=self.device)
        except Exception as e:
            logger.exception("Failed to load cross-encoder '%s' - some ranking features will be unavailable.", cross_encoder_id)
            self.cross_encoder = None

        # Gemini placeholders with health check
        self.fast_model = None
        self.quality_model = None
        # Only attempt to instantiate if genai is configured
        try:
            if hasattr(genai, 'GenerativeModel'):
                fast_name = os.getenv('GEMINI_FAST', 'gemini-2.5-flash-lite')
                quality_name = os.getenv('GEMINI_QUALITY', 'gemini-2.5-flash-lite')
                self.fast_model = genai.GenerativeModel(fast_name)
                self.quality_model = genai.GenerativeModel(quality_name)
                try:
                    self.fast_model.generate_content("test")
                    logger.info("Gemini models functional.")
                except Exception as health_e:
                    logger.warning("Gemini models loaded but not functional: %s", health_e)
                    self.fast_model = self.quality_model = None
        except Exception as e:
            logger.debug("Gemini models not available: %s", e)

        # storage
        self.index_path = index_path
        self.corpus_pickle = corpus_pickle
        self.corpus: Dict[str, Dict[str, Any]] = {}
        self.chapters: Dict[int, Any] = {}  # To store chapter metadata
        self.verse_id_map: Dict[int, str] = {}
        self.doc_texts: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.index: Optional[faiss.Index] = None
        self.chapter_verses: Dict[int, List[str]] = defaultdict(list)  # Pre-index for O(1) lookups

        # BM25
        self.bm25 = None
        self.tokenized_docs: Optional[List[List[str]]] = None

        # try load corpus
        if os.path.exists(self.corpus_pickle):
            try:
                with open(self.corpus_pickle, 'rb') as f:
                    data = pickle.load(f)
                    self.corpus = data.get('corpus', {})
                    self.chapters = data.get('chapters', {})  # Load chapters data
                    self.verse_id_map = data.get('verse_id_map', {})
                    if isinstance(self.verse_id_map, list):
                        self.verse_id_map = {i: v for i, v in enumerate(self.verse_id_map)}
                # Build chapter_verses index
                for vid, v in self.corpus.items():
                    ch = v.get('verse_details', {}).get('chapter_number')
                    if ch:
                        self.chapter_verses[ch].append(vid)
                logger.info("Chapter index built with %d chapters.", len(self.chapter_verses))
            except Exception as e:
                logger.exception("Failed loading corpus pickle: %s", e)
        else:
            logger.info("Corpus pickle not found at %s", self.corpus_pickle)

        if self.corpus:
            self._prepare_doc_texts_and_tfidf()

        # If an index file exists, load it. Otherwise, build when documents exist.
        if os.path.exists(self.index_path):
            try:
                logger.info("Loading FAISS index from %s", self.index_path)
                self.index = faiss.read_index(self.index_path)
                # If loaded index isn't an IndexIDMap, wrap it so add_with_ids works consistently going forward.
                if not isinstance(self.index, faiss.IndexIDMap):
                    try:
                        self.index = faiss.IndexIDMap(self.index)
                        logger.info("Wrapped FAISS index into IndexIDMap for stable ids.")
                    except Exception:
                        logger.warning("Could not wrap loaded FAISS index into IndexIDMap; incremental adds may be affected.")
            except Exception as e:
                logger.exception("Failed to read FAISS index: %s", e)
                self.index = None

        if self.index is None and self.doc_texts:
            self._build_faiss_index()

        logger.info("Initialization complete.")

    def _first_sentences(self, text: str, n: int = 1) -> str:
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        return " ".join([p for p in parts[:n] if p])

    def _compose_document_text(self, verse_id: str, verse_data: Dict[str, Any]) -> str:
        vd = verse_data
        ch = vd.get('verse_details', {}).get('chapter_number', 'Unknown')
        vs = vd.get('verse_details', {}).get('verse_number', verse_id)
        verse_text = vd.get('verse_text', '')
        translations = vd.get('translations', [])
        eng_trans = next((t['text'] for t in translations if t.get('language','').lower() == 'english'), '')
        if not eng_trans:
            logger.debug(f"No English translation for {verse_id}")
        hindi_trans = next((t['text'] for t in translations if t.get('language','').lower() == 'hindi'), '')
        comms = vd.get('commentaries', [])
        eng_comms = [c['text'] for c in comms if c.get('language','').lower() == 'english']
        comm_snip = " ".join([self._first_sentences(c, 2) for c in eng_comms[:3]])
        tags = " ".join(vd.get('tags', []))
        doc = (f"title: Chapter {ch}, Verse {vs} | "
               f"text: {verse_text} || translation: {eng_trans} || hindi: {hindi_trans} || tags: {tags} || commentary_snippets: {comm_snip}")
        return doc

    def _prepare_doc_texts_and_tfidf(self):
        logger.info("Preparing document texts and TF-IDF...")
        if self.verse_id_map:
            ordered_verse_ids = [self.verse_id_map[i] for i in sorted(self.verse_id_map.keys())]
        else:
            ordered_verse_ids = list(self.corpus.keys())
            self.verse_id_map = {i: vid for i, vid in enumerate(ordered_verse_ids)}

        self.doc_texts = [self._compose_document_text(vid, self.corpus[vid]) for vid in ordered_verse_ids]

        # Prepare TF-IDF
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)
            logger.info("TF-IDF matrix shape: %s", str(self.tfidf_matrix.shape))
        except Exception as e:
            logger.exception("TF-IDF vectorizer failed: %s", e)
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

        # Prepare BM25 if available
        if BM25_AVAILABLE:
            try:
                tokenized = [doc.split() for doc in self.doc_texts]
                self.tokenized_docs = tokenized
                self.bm25 = BM25Okapi(tokenized)
                logger.info("BM25 prepared with %d documents.", len(self.doc_texts))
            except Exception as e:
                logger.exception("BM25 preparation failed: %s", e)
                self.bm25 = None
        else:
            logger.info("rank_bm25 not installed; BM25 disabled.")

    def _build_faiss_index(self):
        if not self.doc_texts:
            logger.warning("No documents to index.")
            return

        logger.info("Computing embeddings for %d documents...", len(self.doc_texts))
        parts = []
        for i in range(0, len(self.doc_texts), BATCH_SIZE_EMBED):
            batch = self.doc_texts[i:i + BATCH_SIZE_EMBED]
            try:
                emb = self.embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                parts.append(emb)
            except Exception as e:
                logger.error(f"Embedding batch {i//BATCH_SIZE_EMBED} failed: {e}")
                continue
        if not parts:
            logger.error("No embeddings generated—index build failed.")
            return
        self.doc_embeddings = np.vstack(parts).astype('float32')
        self.doc_embeddings = l2_normalize(self.doc_embeddings)

        # Build FAISS index and wrap in IDMap for stable ids
        base_index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap(base_index)
        ids = np.arange(len(self.doc_texts)).astype('int64')
        self.index.add_with_ids(self.doc_embeddings, ids)

        # Save with error handling
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info("FAISS index saved.")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _generate_multiple_queries(self, query: str, max_alternatives: int = 3) -> List[str]:
        prompt = (f"You are an expert in query expansion for a spiritual search engine. "
                  f"From the user's question generate {max_alternatives} alternate rephrasings from different perspectives. "
                  f"Return only the alternatives as a numbered or bulleted list.\n\nUser Question: \"{query}\"\n\nAlternatives:")
        if self.fast_model:
            try:
                resp = self.fast_model.generate_content(prompt)
                items = safe_split_lines(resp.text)
                return items[:max_alternatives]
            except Exception as e:
                logger.debug("Gemini multi-query generation failed: %s", e)
        # fallback heuristics
        return [
            f"How does the Bhagavad Gita address: {query}?",
            f"What does Bhagavad Gita say about {query}?",
            f"Verses about {query} in Bhagavad Gita"
        ][:max_alternatives]

    def _generate_hypothetical_answer(self, query: str) -> str:
        prompt = (f"Provide a concise (1-2 sentence) summary answer to this question as if from Bhagavad Gita's perspective. "
                  f"Question: \"{query}\"")
        if self.fast_model:
            try:
                resp = self.fast_model.generate_content(prompt)
                return resp.text.strip().replace('\n', ' ')
            except Exception as e:
                logger.debug("Gemini HyDE failed: %s", e)
        return query

    def _tfidf_search(self, query: str, top_k: int) -> List[int]:
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            return []
        try:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            scores = self.tfidf_matrix.dot(query_tfidf.T).toarray().flatten()
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [int(i) for i in top_indices if scores[i] > 0]
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []

    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        prefixed = [f"task: search result | query: {q}" for q in queries]
        parts = []
        for i in range(0, len(prefixed), BATCH_SIZE_EMBED):
            batch = prefixed[i:i + BATCH_SIZE_EMBED]
            emb = self.embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            parts.append(emb)
        emb_all = np.vstack(parts).astype('float32')
        emb_all = l2_normalize(emb_all)
        return emb_all

    def _dense_search_rank(self, query_embeddings: np.ndarray, k: int = 20) -> List[int]:
        """Return a ranked list of doc ids by aggregating scores across multiple query embeddings."""
        if self.index is None:
            logger.warning("FAISS index missing; dense search skipped.")
            return []
        # perform search for each query embedding
        distances, indices = self.index.search(query_embeddings, k)
        score_map = defaultdict(float)
        for row_idx in range(indices.shape[0]):
            for col_idx in range(indices.shape[1]):
                idx = int(indices[row_idx, col_idx])
                if idx == -1:
                    continue
                score = float(distances[row_idx, col_idx])
                score_map[idx] += score  # accumulate
        if not score_map:
            return []
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]

    def _bm25_rank(self, query: str, k: int = 20) -> List[int]:
        if not BM25_AVAILABLE or self.bm25 is None:
            return []
        tokenized_q = query.split()
        scores = self.bm25.get_scores(tokenized_q)
        ranked = np.argsort(scores)[::-1]
        return [int(i) for i in ranked[:k] if scores[i] > 0]

    def _retrieve_candidates(self, query_embeddings: np.ndarray, query_text: str, k: int = K_RETRIEVE) -> List[int]:
        # Dense ranked list (expand k for better fusion)
        dense_ranked = self._dense_search_rank(query_embeddings, k=(k * 4))
        # BM25 ranked list
        bm25_ranked = self._bm25_rank(query_text, k=(k * 4)) if BM25_AVAILABLE else []
        # If both exist, fuse lists; else prefer dense or bm25
        lists_to_fuse = []
        if dense_ranked:
            lists_to_fuse.append(dense_ranked)
        if bm25_ranked:
            lists_to_fuse.append(bm25_ranked)
        if lists_to_fuse:
            fused = reciprocal_rank_fusion(lists_to_fuse, k=(k * 4))
            return fused[:k]
        # fallback to tfidf
        tfidf_candidates = self._tfidf_search(query_text, top_k=k)
        return tfidf_candidates

    # ---------- rerank ----------
    def _rerank_and_select(self, user_query: str, candidate_indices: List[int], top_n: int = RERANK_TOP_N) -> List[Dict[str, Any]]:
        if not candidate_indices:
            return []
        pairs = []
        metadata = []
        for idx in candidate_indices:
            verse_id = self.verse_id_map.get(idx)
            if verse_id is None:
                continue
            vd = self.corpus.get(verse_id)
            if vd is None:
                continue
            commentaries = [c for c in vd.get('commentaries', []) if c.get('language','').lower() == 'english']
            if commentaries:
                cand_text = self._first_sentences(commentaries[0]['text'], 4)
            else:
                cand_text = self.doc_texts[idx]
            pairs.append((user_query, cand_text))
            metadata.append({
                'index': idx,
                'verse_id': verse_id,
                'chapter_id': vd.get('verse_details', {}).get('chapter_number'),
                'verse_number': vd.get('verse_details', {}).get('verse_number'),
                'author': commentaries[0].get('author') if commentaries else None,
                'candidate_text': cand_text
            })

        if not pairs:
            return []

        # If cross-encoder is available
        if self.cross_encoder is not None:
            try:
                scores = self.cross_encoder.predict(pairs, batch_size=16)
            except Exception as e:
                logger.exception("CrossEncoder predict failed: %s", e)
                scores = np.zeros(len(pairs))
        else:
            # Improved fallback: use TF-IDF cosine similarity
            scores = np.zeros(len(pairs))
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    query_tfidf = self.tfidf_vectorizer.transform([user_query])
                    tfidf_scores = self.tfidf_matrix[np.array(candidate_indices)].dot(query_tfidf.T).toarray().flatten()
                    scores = tfidf_scores
                except Exception as e:
                    logger.warning("TF-IDF scoring failed: %s", e)
            else:
                for i, (_q, ct) in enumerate(pairs):
                    scores[i] = len(set(ct.split()) & set(user_query.split()))

        ranked = sorted(zip(metadata, scores), key=lambda x: x[1], reverse=True)
        top = [m for m, s in ranked[:top_n]]
        return top

    # ---------- synthesis ----------
    def _synthesize_answer(self, user_query: str, selected_sources: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, s in enumerate(selected_sources):
            vd = self.corpus.get(s['verse_id']) if s.get('verse_id') else None
            translations = vd.get('translations', []) if vd else []
            translation = next((t['text'] for t in translations if t.get('language','').lower() == 'english'), 'Translation not available.')
            commentary_author = s.get('author') or "Unknown"
            comp = s.get('candidate_text', '')
            context_parts.append(
                f"Source {i+1}:\n- Chapter {s.get('chapter_id')}, Verse {s.get('verse_number')}\n- Translation: \"{translation}\"\n- Commentary ({commentary_author}): \"{comp}\"\n"
            )
        context_string = "\n\n".join(context_parts)

        final_prompt = (
            "You are a serene and wise spiritual scholar named GitaAI. "
            "Using ONLY the provided SOURCE material, synthesize a single, coherent answer to the user's question. "
            "Be concise (3-5 paragraphs), clear, and compassionate. If the sources are insufficient, explicitly say so and offer suggested verses for further reading.\n\n"
            f"User Question: \"{user_query}\"\n\nSOURCES:\n{context_string}\n\nAnswer:")

        if self.quality_model:
            try:
                resp = self.quality_model.generate_content(final_prompt)
                return resp.text.strip()
            except Exception as e:
                logger.debug("Gemini synthesis failed: %s", e)

        fallback = "Based on the provided sources:\n\n"
        for s in selected_sources:
            fallback += f"- Chapter {s.get('chapter_id')}, Verse {s.get('verse_number')} ({s.get('author')}): {s.get('candidate_text')[:300]}...\n"
        return fallback

    # ---------- telemetry ----------
    def _log_telemetry(self, payload: Dict[str, Any]):
        try:
            payload['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            with open(TELEMETRY_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.debug("Failed to write telemetry: %s", e)

    # ---------- public API ----------
    def get_response(self, query: str) -> Dict[str, Any]:
        normalized_query = query.strip().lower()
        if len(normalized_query) < 2:
            return {"answer": "Please provide a longer question.", "sources": [], "debug": {"reason": "short query"}}
        return self._get_response_cached(normalized_query)
    
    @lru_cache(maxsize=128)
    def _get_response_cached(self, query: str) -> Dict[str, Any]:
        logger.info("User query (normalized): %s", query)

        alt_queries = self._generate_multiple_queries(query)
        all_queries = [query] + [q for q in alt_queries if q and q.lower() != query.lower()]
        logger.debug("Expanded queries: %s", all_queries)

        hyde_docs = [self._generate_hypothetical_answer(q) for q in all_queries]
        query_embeddings = self._embed_queries(hyde_docs)

        candidate_indices = self._retrieve_candidates(query_embeddings, query, k=K_RETRIEVE * 2)
        fallback_used = False
        if not candidate_indices:
            candidate_indices = self._tfidf_search(query, top_k=TFIDF_TOP_K)
            fallback_used = True

        if not candidate_indices:
            self._log_telemetry({
                'query': query,
                'normalized_query': query,
                'candidates': [],
                'selected': [],
                'answer': None,
                'fallback_used': True
            })
            return {"answer": "I could not find relevant verses for your question.", "sources": [], "debug": {"candidates": []}}

        selected_sources_meta = self._rerank_and_select(query, candidate_indices, top_n=RERANK_TOP_N)
        if not selected_sources_meta:
            sources_min = []
            for idx in candidate_indices[:3]:
                vid = self.verse_id_map.get(idx)
                vd = self.corpus.get(vid, {})
                sources_min.append({
                    "chapter": vd.get('verse_details', {}).get('chapter_number'),
                    "verse": vd.get('verse_details', {}).get('verse_number'),
                    "commentary_author": None
                })
            self._log_telemetry({
                'query': query,
                'normalized_query': query,
                'candidates': candidate_indices,
                'selected': [],
                'answer': None,
                'fallback_used': fallback_used
            })
            return {"answer": "Found related verses but could not compute final ranking.", "sources": sources_min, "debug": {"fallback": fallback_used}}

        final_answer = self._synthesize_answer(query, selected_sources_meta)
        final_sources_ui = [{
            "chapter": s.get('chapter_id'),
            "verse": s.get('verse_number'),
            "commentary_author": s.get('author')
        } for s in selected_sources_meta]

        debug = {
            "num_candidates": len(candidate_indices),
            "fallback_used": fallback_used,
            "selected_indices": [s.get('index') for s in selected_sources_meta]
        }

        # telemetry
        try:
            self._log_telemetry({
                'query': query,
                'normalized_query': query,
                'candidates': candidate_indices,
                'selected': [s.get('index') for s in selected_sources_meta],
                'sources': final_sources_ui,
                'answer': final_answer[:1000],
                'debug': debug,
                'fallback_used': fallback_used
            })
        except Exception:
            logger.debug("Telemetry logging failed.")

        return {"answer": final_answer, "sources": final_sources_ui, "debug": debug}

    # ---------- utility: add documents incrementally ----------
    def add_documents(self, new_corpus: Dict[str, Dict[str, Any]], save: bool = True):
        if not new_corpus:
            logger.warning("add_documents called with no new documents.")
            return

        start_idx = max(self.verse_id_map.keys()) + 1 if self.verse_id_map else 0
        new_doc_texts = []
        new_vids = []
        idx_counter = start_idx
        for vid, vdata in new_corpus.items():
            if vid in self.corpus:
                logger.warning(f"Verse ID {vid} already exists in corpus. Skipping.")
                continue

            new_idx = idx_counter
            idx_counter += 1
            self.corpus[vid] = vdata
            self.verse_id_map[new_idx] = vid
            new_vids.append((new_idx, vid))

            doc_text = self._compose_document_text(vid, vdata)
            self.doc_texts.append(doc_text)
            new_doc_texts.append(doc_text)

            ch = vdata.get('verse_details', {}).get('chapter_number')
            if ch:
                self.chapter_verses[ch].append(vid)

        logger.info(f"Adding {len(new_doc_texts)} new documents to the index.")

        if self.index is not None and new_doc_texts:
            logger.info("Generating embeddings for new documents...")
            new_embeddings = self.embed_model.encode(
                new_doc_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE_EMBED
            ).astype('float32')
            new_embeddings = l2_normalize(new_embeddings)

            # compute new ids
            new_ids = np.arange(start_idx, start_idx + new_embeddings.shape[0]).astype('int64')
            logger.info(f"Adding {new_embeddings.shape[0]} new vectors to FAISS index.")
            try:
                self.index.add_with_ids(new_embeddings, new_ids)
            except Exception as e:
                logger.exception("Failed to add_with_ids to FAISS index: %s", e)
                # fallback: rebuild index
                self._build_faiss_index()
        else:
            logger.warning("FAISS index not available or no new documents; full rebuild might be needed.")
            self._build_faiss_index()

        # update BM25 and TF-IDF
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)
            if BM25_AVAILABLE:
                tokenized = [doc.split() for doc in self.doc_texts]
                self.tokenized_docs = tokenized
                self.bm25 = BM25Okapi(tokenized)
        except Exception:
            logger.exception("Could not update TF-IDF/BM25 vectorizer")

        if save:
            try:
                faiss.write_index(self.index, self.index_path)
                logger.info("Saved updated FAISS index to %s", self.index_path)

                with open(self.corpus_pickle, 'wb') as f:
                    pickle.dump({'corpus': self.corpus, 'verse_id_map': self.verse_id_map, 'chapters': self.chapters}, f)
                    logger.info("Saved updated corpus to %s", self.corpus_pickle)
            except Exception:
                logger.exception("Could not save updated index or corpus pickle")


if __name__ == '__main__':
    ge = GitaEngine()
    print('Ready. Load corpus pickle and call get_response(query)')