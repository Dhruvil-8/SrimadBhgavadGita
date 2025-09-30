'use client';

import Link from 'next/link';

export default function AboutPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-orange-50 text-gray-800">
      <div className="w-full max-w-4xl">
        
        {/* Header */}
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-orange-800">About This Project</h1>
          <Link href="/" className="text-orange-600 hover:underline font-medium">
            ← Back to App
          </Link>
        </header>

        {/* Content Box */}
        <section className="bg-white p-8 rounded-lg shadow-md border border-gray-200 text-lg leading-relaxed space-y-4">

          <p>
            This project is an AI-powered tool designed to help you explore and reflect on the teachings of the Srimad Bhagavad Gita.
          </p>

          <p>
            You may occasionally find that some retrieved verses are not directly relevant to your question. 
            Please take them as a guide, and be encouraged to return to the original text for deeper study and reflection.
          </p>
          
          <p>
            To learn more about how this project works or to view the source code, visit the{' '}
            <a
              // IMPORTANT: Replace this with the actual URL to your GitHub repository
              href="https://github.com/Dhruvil-8/SrimadBhgavadGita"
              target="_blank"
              rel="noopener noreferrer"
              className="text-orange-600 hover:underline"
            >
              GitHub repository
            </a>.
          </p>
          
        </section>
      </div>
    </main>
  );
}
