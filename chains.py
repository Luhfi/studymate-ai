from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import config


# LLM Model
def get_llm(temperature: float = 0.7):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


# 1. ROUTER CHAIN — Klasifikasi intent user
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah sistem klasifikasi intent yang cerdas untuk asisten belajar.
Tugasmu adalah menganalisis input pengguna dan mengelompokkannya ke dalam SATU kata kunci berikut:

- 'research'   → Jika pengguna meminta penjelasan materi, sejarah, teori, atau menggunakan kata kerja seperti 'jelaskan', 'carikan', 'riset', 'apa itu'.
- 'analysis'   → Jika pengguna ingin menganalisis suatu teks, komentar, sentimen, atau mencari poin penting dari sebuah paragraf.
- 'writing'    → Jika pengguna meminta dibuatkan konten panjang, esai, artikel, draf tulisan, atau rangkuman materi kuliah.
- 'flashcard'  → Jika pengguna secara eksplisit meminta dibuatkan flashcard, kartu belajar, atau tebak-tebakan Q&A untuk hafalan.
- 'qa'         → Jika pengguna hanya menyapa, melakukan chat santai, atau memberikan pertanyaan super pendek.

PENTING: Balas HANYA dengan satu kata saja (research/analysis/writing/flashcard/qa) tanpa tanda baca dan tanpa penjelasan apa pun."""),
    ("human", "{input}")
])

def get_router_chain():
    return ROUTER_PROMPT | get_llm(temperature=0) | StrOutputParser()


# 2. RESEARCH CHAIN — Web search + sintesis
RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah peneliti dan pendidik yang ahli.
Berdasarkan hasil pencarian web yang diberikan, susun jawaban yang komprehensif.

Gunakan format ini:
## 🔍 Ringkasan
[Ringkasan singkat topik dalam 2-3 kalimat]

## 📚 Informasi Utama
[Poin-poin penting dari hasil riset, gunakan bullet points]

## 💡 Penjelasan Mendalam
[Penjelasan detail yang edukatif dengan contoh]

## 🔗 Konsep Terkait
[Hubungan dengan topik atau konsep lain yang relevan]

Gunakan bahasa yang sama dengan pertanyaan pengguna."""),
    ("human", "Pertanyaan: {input}\n\nHasil Pencarian Web:\n{search_results}")
])

def get_research_chain():
    return RESEARCH_PROMPT | get_llm() | StrOutputParser()


# 3. ANALYSIS CHAIN — Analisis teks mendalam
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah analis teks yang sangat ahli.
Lakukan analisis menyeluruh terhadap teks yang diberikan:

## 📊 Analisis Sentimen
[Sentimen: Positif/Negatif/Netral dengan skor estimasi 0-100% dan alasan]

## 🔑 Poin-Poin Kunci
[5-7 poin utama dalam teks sebagai bullet points]

## 📝 Ringkasan Eksekutif
[Ringkasan 2-3 kalimat yang menangkap esensi teks]

## 🧩 Tema & Topik
[Identifikasi tema utama, sub-topik, dan kata kunci penting]

## 💬 Nada & Gaya Penulisan
[Analisis nada, register, dan karakteristik gaya penulisan]

## 💡 Insight
[Observasi menarik atau hal yang mungkin terlewat dari teks]

Gunakan bahasa yang sama dengan teks input."""),
    ("human", "Analisis teks berikut ini:\n\n{input}")
])

def get_analysis_chain():
    return ANALYSIS_PROMPT | get_llm() | StrOutputParser()


# 4. WRITING CHAIN — Buat konten berkualitas
WRITING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah penulis akademik dan pendidik yang sangat mahir.
Buat konten berkualitas tinggi, terstruktur, dan informatif sesuai permintaan.

Gunakan format Markdown yang rapi:
- Judul utama yang menarik
- Bagian-bagian terorganisir dengan heading
- Bullet points dan contoh praktis
- Paragraf yang mengalir dan mudah dibaca
- Kesimpulan / takeaway yang bermakna

Gunakan bahasa yang sama dengan permintaan pengguna."""),
    ("human", "{input}")
])

def get_writing_chain():
    return WRITING_PROMPT | get_llm() | StrOutputParser()


# 5. FLASHCARD CHAIN — Buat kartu belajar
FLASHCARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah pembuat flashcard belajar yang ahli.
Buat 6-8 flashcard berkualitas tinggi dengan format berikut:

---
**FLASHCARD 1**
**❓ Pertanyaan:** [Pertanyaan atau konsep yang diuji]
**✅ Jawaban:** [Jawaban singkat namun informatif]

---
**FLASHCARD 2**
[dst...]

Pastikan flashcard:
- Mencakup konsep-konsep paling penting dari topik
- Pertanyaan spesifik dan jelas
- Jawaban ringkas tapi lengkap
- Berurutan dari konsep dasar ke lanjutan
- Ada variasi: definisi, contoh, perbedaan, penerapan

Gunakan bahasa yang sama dengan permintaan pengguna."""),
    ("human", "Buat flashcard untuk topik: {input}")
])

def get_flashcard_chain():
    return FLASHCARD_PROMPT | get_llm() | StrOutputParser()


# 6. Q&A CHAIN — Tanya jawab langsung
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah tutor berpengetahuan luas dan sangat membantu.
Berikan jawaban yang jelas, akurat, dan edukatif.

Format respons:
## ✅ Jawaban Langsung
[Jawaban singkat dan to-the-point]

## 📖 Penjelasan Detail
[Penjelasan komprehensif dengan logika yang mudah dipahami]

## 🎯 Contoh Praktis
[1-2 contoh konkret jika relevan]

## 🔗 Konsep Terkait
[Konsep yang berkaitan untuk pemahaman lebih dalam]

Gunakan bahasa yang sama dengan pertanyaan pengguna."""),
    ("human", "{input}")
])

def get_qa_chain():
    return QA_PROMPT | get_llm() | StrOutputParser()


# Search Tool
def get_search_tool():
    return TavilySearchResults(
        max_results=3,
        tavily_api_key=config.TAVILY_API_KEY,
        include_answer=True,
    )