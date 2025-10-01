'use client';

import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';

interface SourcePointer {
  chapter: number;
  verse: number;
  commentary_author: string | null;
}

interface AiResponse {
  answer: string;
  sources: SourcePointer[] | null;
}

interface VerseData {
  verse_details: {
    chapter_number: number;
    verse_number: number;
    sanskrit_text: string;
    transliteration: string;
  };
  translations: { author: string; language: string; text: string }[];
  commentaries: { author: string; language: string; text: string }[];
  audio_url?: string | null;
}

type Language = 'english' | 'hindi' | 'sanskrit';

export default function HomePage() {
  const [mode, setMode] = useState<'ask' | 'read'>('ask');
  const [query, setQuery] = useState('');
  const [aiResponse, setAiResponse] = useState<AiResponse | null>(null);
  const [sourceVerseData, setSourceVerseData] = useState<VerseData[]>([]);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [isSourceLoading, setIsSourceLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('english');

  // Reading-mode state
  const [chapterList, setChapterList] = useState<{ id: number; name?: string; verse_count: number }[]>([]);
  const [currentChapter, setCurrentChapter] = useState<number | null>(null);
  const [chapterData, setChapterData] = useState<VerseData[]>([]);
  const [currentVerseIndex, setCurrentVerseIndex] = useState<number>(0);
  const [bookmarks, setBookmarks] = useState<string[]>([]);
  const [readingProgress, setReadingProgress] = useState<Record<number, number>>({});
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState<string | null>(null);

  const API = process.env.NEXT_PUBLIC_API_URL || '';

  useEffect(() => {
    try {
      const bm = localStorage.getItem('gita_bookmarks');
      if (bm) setBookmarks(JSON.parse(bm));
      const rp = localStorage.getItem('gita_reading_progress');
      if (rp) setReadingProgress(JSON.parse(rp));
    } catch (e) {
      console.warn('Failed to read localStorage', e);
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('gita_bookmarks', JSON.stringify(bookmarks));
    } catch (e) {
      console.warn('Failed to set bookmarks', e);
    }
  }, [bookmarks]);

  useEffect(() => {
    try {
      localStorage.setItem('gita_reading_progress', JSON.stringify(readingProgress));
    } catch (e) {
      console.warn('Failed to set reading progress', e);
    }
  }, [readingProgress]);

  useEffect(() => {
    if (mode === 'read' && chapterList.length === 0) {
      fetch(`${API}/api/chapters`)
        .then((res) => res.json())
        .then((data) => setChapterList(data.chapters))
        .catch((err) => console.error('Could not fetch chapters', err));
    }
  }, [mode, API, chapterList.length]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setError(null);
    setAiResponse(null);
    setSourceVerseData([]);
    setIsAiLoading(true);

    try {
      const askResponse = await fetch(`${API}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!askResponse.ok) throw new Error('Failed to get a response from the AI.');

      const askData: AiResponse = await askResponse.json();
      setAiResponse(askData);
      setIsAiLoading(false);

      if (askData.sources?.length) {
        setIsSourceLoading(true);

        const versePromises = askData.sources.map((s) =>
          fetch(`${API}/api/verse/${s.chapter}/${s.verse}`).then((res) => {
            if (!res.ok) throw new Error(`Failed to retrieve verse ${s.chapter}.${s.verse}`);
            return res.json();
          })
        );

        const versesData: VerseData[] = await Promise.all(versePromises);
        setSourceVerseData(versesData);
        setIsSourceLoading(false);
      }
    } catch (err: any) {
      setError(err.message);
      setIsAiLoading(false);
      setIsSourceLoading(false);
    }
  };

  const loadChapter = async (chapter: number) => {
    setCurrentChapter(chapter);
    setChapterData([]);
    setCurrentVerseIndex(0);
    try {
      const res = await fetch(`${API}/api/chapter/${chapter}`);
      if (!res.ok) throw new Error('Failed to fetch chapter');
      const data: VerseData[] = await res.json();
      setChapterData(data);
      markVerseRead(chapter, data[0]?.verse_details.verse_number ?? 1);
    } catch (e) {
      console.error(e);
      setError('Failed to load chapter');
    }
  };

  const goToVerseIndex = (index: number) => {
    const clamped = Math.max(0, Math.min(index, chapterData.length - 1));
    setCurrentVerseIndex(clamped);
    const vnum = chapterData[clamped]?.verse_details.verse_number;
    if (currentChapter && vnum) markVerseRead(currentChapter, vnum);
    stopAudio();
  };

  const prevVerse = () => goToVerseIndex(currentVerseIndex - 1);
  const nextVerse = () => goToVerseIndex(currentVerseIndex + 1);

  const toggleBookmark = (chapter: number, verse: number) => {
    const key = `${chapter}:${verse}`;
    setBookmarks((prev) => {
      if (prev.includes(key)) return prev.filter((p) => p !== key);
      return [...prev, key];
    });
  };

  const isBookmarked = (chapter: number, verse: number) => bookmarks.includes(`${chapter}:${verse}`);

  const markVerseRead = (chapter: number, verse: number) => {
    setReadingProgress((prev) => {
      const prevVal = prev[chapter] || 0;
      const newVal = Math.max(prevVal, verse);
      return { ...prev, [chapter]: newVal };
    });
  };

  const getProgressForChapter = (chapter: number) => {
    const read = readingProgress[chapter] || 0;
    const total = chapterList.find((c) => c.id === chapter)?.verse_count || chapterData.length || 0;
    const pct = total > 0 ? Math.round((read / total) * 100) : 0;
    return { read, total, pct };
  };

  const playAudio = (path: string) => {
    try {
      if (!audioRef.current) {
        audioRef.current = new Audio(path);
      } else if (audioRef.current.src !== window.location.origin + path) {
        audioRef.current.src = path;
        audioRef.current.load();
      }
      
      audioRef.current.play().then(() => {
          setIsPlaying(true);
          setCurrentlyPlaying(path);
      }).catch(e => console.error("Audio playback failed:", e));

      audioRef.current.onended = () => {
          setIsPlaying(false);
          setCurrentlyPlaying(null);
      };

    } catch (e) {
      console.error('Audio play failed', e);
      setIsPlaying(false);
      setCurrentlyPlaying(null);
    }
  };

  const pauseAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };
  
  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      setCurrentlyPlaying(null);
    }
  };

  const renderVerseBlock = (verse: VerseData, idx: number) => {
    const ch = verse.verse_details.chapter_number;
    const v = verse.verse_details.verse_number;
    const bookmarked = isBookmarked(ch, v);

    const audioPath = `/audio/${ch}/${v}.mp3`;
    const isThisAudioPlaying = isPlaying && currentlyPlaying === audioPath;

    return (
      <div key={`${ch}-${v}`} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-lg font-semibold text-orange-700">Chapter {ch}, Verse {v}</h3>
            <p className="text-sm text-gray-500">Verse {idx + 1} of {chapterData.length}</p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => toggleBookmark(ch, v)}
              className={`px-3 py-1 rounded-md text-sm font-medium border transition ${bookmarked ? 'bg-yellow-400 text-white' : 'bg-white text-orange-600 border-orange-200'}`}
            >
              {bookmarked ? 'Bookmarked' : 'Bookmark'}
            </button>
            
            {/* --- START: MODIFIED AUDIO BUTTON (REMOVED "OPEN" LINK) --- */}
            <button onClick={() => (isThisAudioPlaying ? pauseAudio() : playAudio(audioPath))} className="px-3 py-1 rounded-md border">
              {isThisAudioPlaying ? 'Pause' : 'Play'}
            </button>
            {/* --- END: MODIFIED AUDIO BUTTON --- */}
          </div>
        </div>

        <div className="mt-4 space-y-4">
          <div>
            <h4 className="font-semibold text-gray-600">Sanskrit:</h4>
            <p className="font-serif text-lg leading-relaxed">{verse.verse_details.sanskrit_text}</p>
          </div>
          <div>
            <h4 className="font-semibold text-gray-600">Transliteration:</h4>
            <p className="italic text-gray-500">{verse.verse_details.transliteration}</p>
          </div>
          <div>
            <h4 className="font-semibold text-gray-600">Translations ({selectedLanguage}):</h4>
            {verse.translations.filter(t => t.language?.toLowerCase() === selectedLanguage).map((t, i) => (
              <div key={i} className="p-3 bg-orange-50/50 border border-orange-100 rounded-md mt-2">
                <p className="font-medium text-sm text-orange-900">{t.author}:</p>
                <p className="italic">"{t.text}"</p>
              </div>
            ))}
          </div>
          <div>
            <h4 className="font-semibold text-gray-600">Commentaries ({selectedLanguage}):</h4>
            {verse.commentaries.filter(c => c.language?.toLowerCase() === selectedLanguage).map((c, i) => (
              <div key={i} className="p-3 bg-gray-50 border border-gray-200 rounded-md mt-2">
                <p className="font-medium text-sm text-gray-800">{c.author}:</p>
                <p className="text-gray-700">{c.text}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-6 flex justify-between items-center">
          <div>
            <button onClick={prevVerse} disabled={idx === 0} className="px-4 py-2 rounded-md border">
              ← Previous
            </button>
            <button onClick={nextVerse} disabled={idx === chapterData.length - 1} className="ml-2 px-4 py-2 rounded-md border">
              Next →
            </button>
          </div>
          <div className="text-sm text-gray-500">Progress: {getProgressForChapter(ch).pct}% ({getProgressForChapter(ch).read}/{getProgressForChapter(ch).total})</div>
        </div>
      </div>
    );
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-orange-50 text-gray-800">
      <div className="w-full max-w-4xl">
        {/* START: MODIFIED SECTION */}
        <div className="flex justify-between items-center mb-2">
            {/* 1. Invisible spacer on the left */}
            <div className="invisible">
               <a href="/about" className="text-orange-600 hover:underline font-medium">
                   About
               </a>
            </div>

            {/* 2. Your centered title */}
            <h1 className="text-4xl font-bold text-orange-800 text-center">
            Srimad Bhagavad Gita
            </h1>

            {/* 3. The real "About" link on the right */}
            <Link href="/about" className="text-orange-600 hover:underline font-medium">
               About
            </Link>
        </div>

        <p className="text-center text-gray-600 mb-6">Ask a question or read the Bhagavad Gita.</p>
        {/* END: MODIFIED SECTION */}

        <div className="flex justify-center space-x-4 mb-6">
          <button onClick={() => setMode('ask')} className={`px-4 py-2 rounded-md font-medium ${mode === 'ask' ? 'bg-orange-600 text-white' : 'bg-white border border-orange-200 text-orange-600'}`}>
            Ask AI
          </button>
          <button onClick={() => setMode('read')} className={`px-4 py-2 rounded-md font-medium ${mode === 'read' ? 'bg-orange-600 text-white' : 'bg-white border border-orange-200 text-orange-600'}`}>
            Reading Mode
          </button>
        </div>

        {mode === 'ask' && (
          <>
            <form onSubmit={handleSubmit} className="mb-8">
              <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="What is the nature of duty? / कर्तव्य का स्वरूप क्या है?" className="w-full p-3 border border-orange-200 rounded-lg shadow-sm focus:ring-2 focus:ring-orange-500 focus:outline-none" rows={3} />
              <button type="submit" disabled={isAiLoading || isSourceLoading} className="w-full mt-2 p-3 bg-orange-600 text-white font-semibold rounded-lg shadow-md hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors">
                {isAiLoading ? 'Synthesizing Wisdom...' : 'Seek Wisdom'}
              </button>
            </form>

            {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg" role="alert">{error}</div>}
            {aiResponse && (
              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
                <h2 className="text-2xl font-semibold mb-3 text-orange-700">Synthesized Answer</h2>
                <p className="text-lg leading-relaxed whitespace-pre-wrap">{aiResponse.answer}</p>
              </div>
            )}
            {isSourceLoading && <p className="text-center mt-4">Loading source details...</p>}

            {/* --- START: RESTORED VERSE CONTENT FOR "ASK AI" MODE --- */}
            {sourceVerseData.length > 0 && (
              <div className="mt-6 space-y-6">
                <h2 className="text-2xl font-semibold text-orange-700">Relevant Sources</h2>
                {sourceVerseData.map((verseData) => (
                  <div key={`${verseData.verse_details.chapter_number}-${verseData.verse_details.verse_number}`} className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                    <h3 className="text-xl font-semibold text-orange-700">Bhagavad Gita, Chapter {verseData.verse_details.chapter_number}, Verse {verseData.verse_details.verse_number}</h3>
                    <div className="my-4 flex justify-center space-x-2 border-b pb-4">
                      {(['english', 'hindi', 'sanskrit'] as Language[]).map((lang) => (
                        <button key={lang} onClick={() => setSelectedLanguage(lang)} className={`px-4 py-2 text-sm font-medium rounded-md border transition-colors ${selectedLanguage === lang ? 'bg-orange-600 text-white border-orange-600' : 'bg-white text-orange-600 border-orange-200 hover:bg-orange-100'}`}>
                          {lang.charAt(0).toUpperCase() + lang.slice(1)}
                        </button>
                      ))}
                    </div>
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-semibold text-gray-600">Sanskrit Text:</h4>
                        <p className="text-lg whitespace-pre-wrap font-serif">{verseData.verse_details.sanskrit_text.trim()}</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-600">Transliteration:</h4>
                        <p className="text-md italic text-gray-500">{verseData.verse_details.transliteration.trim()}</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-600 mb-2">Translations ({selectedLanguage}):</h4>
                        {verseData.translations.filter((t) => t.language?.toLowerCase() === selectedLanguage).map((t, i) => (
                          <div key={i} className="p-3 bg-orange-50/50 border border-orange-100 rounded-md">
                            <p className="font-medium text-sm text-orange-900">{t.author}:</p>
                            <p className="italic">"{t.text.trim()}"</p>
                          </div>
                        ))}
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-600 mb-2">Commentaries ({selectedLanguage}):</h4>
                        {verseData.commentaries.filter((c) => c.language?.toLowerCase() === selectedLanguage).map((c, i) => (
                          <div key={i} className="p-3 bg-gray-50 border border-gray-200 rounded-md">
                            <p className="font-medium text-sm text-gray-800">{c.author}:</p>
                            <p className="text-gray-700">{c.text.trim()}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {/* --- END: RESTORED VERSE CONTENT --- */}
          </>
        )}

        {mode === 'read' && (
          <div>
            {!currentChapter ? (
              <div>
                <h2 className="text-2xl font-semibold mb-4 text-orange-700">Chapters</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                  {chapterList.map((ch) => (
                    <button key={ch.id} onClick={() => loadChapter(ch.id)} className="p-4 bg-white shadow rounded-lg border hover:bg-orange-50 transition">
                      <p className="font-semibold text-orange-700">Chapter {ch.id}</p>
                      <p className="text-sm text-gray-500">{ch.name ?? '—'}</p>
                      <p className="text-xs text-gray-400">{ch.verse_count} verses</p>
                      <div className="mt-2 text-sm text-gray-600">Progress: {getProgressForChapter(ch.id).pct}%</div>
                    </button>
                  ))}
                </div>

                <div className="mt-6">
                  <h3 className="text-lg font-medium">Bookmarks</h3>
                  {bookmarks.length === 0 ? <p className="text-sm text-gray-500">No bookmarks yet.</p> : (
                    <div className="mt-2 space-y-2">
                      {bookmarks.map((b) => {
                        const [ch, v] = b.split(':').map(Number);
                        return (
                          <div key={b} className="p-2 bg-white border rounded flex justify-between items-center">
                            <div>
                              <div className="font-medium text-orange-700">Chapter {ch}, Verse {v}</div>
                              <div className="text-sm text-gray-500">Tap to open</div>
                            </div>
                            <div>
                              <button onClick={() => { loadChapter(ch).then(() => { const verseIndex = chapterData.findIndex(cd => cd.verse_details.verse_number === v); setCurrentVerseIndex(verseIndex !== -1 ? verseIndex : 0); }); }} className="px-3 py-1 border rounded text-sm mr-2">Open</button>
                              <button onClick={() => setBookmarks(prev => prev.filter(p => p !== b))} className="px-3 py-1 border rounded text-sm">Remove</button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <button onClick={() => { setCurrentChapter(null); stopAudio(); }} className="text-orange-600 underline">← Back to Chapters</button>
                    <h2 className="text-2xl font-semibold text-orange-700 inline-block ml-4">Chapter {currentChapter}</h2>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="text-sm text-gray-600">Select language:</div>
                    {(['english', 'hindi', 'sanskrit'] as Language[]).map((lang) => (
                      <button key={lang} onClick={() => setSelectedLanguage(lang)} className={`px-3 py-1 rounded-md text-sm ${selectedLanguage === lang ? 'bg-orange-600 text-white' : 'bg-white border'}`}>
                        {lang}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <button onClick={prevVerse} disabled={currentVerseIndex === 0} className="px-4 py-2 rounded-md border">← Prev</button>
                    <button onClick={nextVerse} disabled={currentVerseIndex === chapterData.length - 1} className="ml-2 px-4 py-2 rounded-md border">Next →</button>
                  </div>
                  <div className="text-sm text-gray-600">Verse {currentVerseIndex + 1} / {chapterData.length}</div>
                </div>
                
                {chapterData.length > 0 && chapterData[currentVerseIndex] && renderVerseBlock(chapterData[currentVerseIndex], currentVerseIndex)}

                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">All verses in this chapter</h3>
                  <div className="grid gap-2">
                    {chapterData.map((v, idx) => (
                      <button key={`${v.verse_details.chapter_number}-${v.verse_details.verse_number}`} onClick={() => goToVerseIndex(idx)} className={`text-left p-2 rounded border ${idx === currentVerseIndex ? 'bg-orange-50 border-orange-200' : 'bg-white'}`}>
                        <div className="flex justify-between">
                          <div>Verse {v.verse_details.verse_number}</div>
                          <div className="text-sm text-gray-500">{v.translations.find(t=>t.language==='english')?.text?.slice(0, 60) ?? ''}...</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}