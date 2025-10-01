# Srimad Bhagavad Gita AI

Srimad Bhagavad Gita AI is a full-stack application designed to provide answers from the sacred text of the Srimad Bhagavad Gita using a Retrieval-Augmented Generation (RAG) system.

Users can ask questions in natural language, and the system retrieves the most relevant verses and scholarly commentaries using semantic search. 

---

## Core Idea

The vision of this project goes beyond the Srimad Bhagavad Gita.  
The core idea is that anyone should be able to explore the wisdom of Sanatan Dharma by simply asking a question — with answers drawn from its sacred foundations: the Vedas, Puranas, Samhitas, and related scriptures. 

---

## Features

- **AI-Powered Q&A:** Ask complex spiritual and philosophical questions.  
- **Source-Grounded Answers:** Every AI answer is backed by specific verses and commentaries.  
- **Reading Mode:** Browse and read the Bhagavad Gita chapter by chapter.    
- **Multilingual Support:** View translations and commentaries in English, Hindi, and Sanskrit.  

---

## Development Story

This project represents a unique collaboration between human direction and artificial intelligence. The entire codebase, both the Python backend and the Next.js frontend, was written by advanced LLMs. My role was that of an architect, providing the vision, and iterative feedback to guide the models.

- Google Gemini- ChatGPT - Grok  

---

## Technology & Data

- **Frontend:** Next.js, React, Tailwind CSS  
- **Backend:** FastAPI (Python)  
- **AI/ML:** Sentence-Transformers & FAISS for semantic search  
- **Data:** Verses & commentaries are sourced from the open-source [Gita GitHub Project](https://github.com/gita/gita/tree/main/data).  

---

## Current Limitations & Future Work

⚠️ *Note:* The current version may sometimes retrieve irrelevant verses.The system and dataset still require further fine-tuning and enrichment to improve contextual relevance and accuracy.  

Future improvements include:  
- Expanding the dataset to cover Vedas, Puranas, Samhitas, Ayurveda, and other Sanatan Dharma texts.  
- More refined embeddings and semantic search for better verse matching.  
- Enhanced multilingual support and commentary alignment.  
