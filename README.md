# 🤖 Autonomous Candidate Screening Platform (TalentAI Engine)

> **AI Specialist Technical Assessment Solution**
> Platform seleksi & penapisan CV otomatis berbasis AI/ML end-to-end dengan sistem *Blind Anonymization*, *Multi-Tier Hybrid Matching Engine*, dan *Explainable AI (XAI)*.

---

## 📌 Features
- 🛡️ **Ethical PII Anonymizer:** Menghapus Nama, Foto, Gender, Umur, dan Institusi bergengsi secara otomatis untuk eliminasi bias seleksi.
- ⚡ **Multi-Tier Hybrid Matcher:**
  - **Tier 1:** Hard-filter instan untuk kriteria mutlak (misal: Min. S1, sertifikasi K3).
  - **Tier 2:** Semantic Vector Embedding & Skill Graph Similarity.
  - **Tier 3:** LLM Chain-of-Thought (CoT) Deep Qualitative Evaluation.
- 📊 **Explainable AI (XAI) Output:** Skor fit 0-100%, analisis Kelebihan (Pros), Potensi Gap (Cons), dan Rekomendasi Pertanyaan Wawancara.
- 💻 **Interactive HR Dashboard:** Antarmuka Streamlit untuk manajemen job opening, upload CV, visualisasi ranking kandidat, dan pencetakan laporan.

---

## 🗂️ Project Structure

```
Autonomous_Candidate_Screening_Platform/
├── PRD.md                       # Product Requirement Document lengkap
├── README.md                    # Panduan penggunaan & arsitektur proyek
├── requirements.txt             # Dependensi Python
├── sample_data/                 # Sample dataset untuk pengujian
│   ├── job_descriptions.json    # Contoh kriteria lowongan manufaktur
│   └── sample_cvs.json          # Contoh dataset CV kandidat
└── src/                         # Core Python Modules
    ├── anonymizer.py            # PII Masking & Blind-CV Engine
    ├── parser.py                # Document Entity Extractor
    ├── matcher.py               # 3-Tier Hybrid Scoring & Matching Engine
    └── app.py                   # Streamlit Interactive HR Dashboard
```

---

## 🚀 Quick Start & Installation

### 1. Prasyarat
- Python 3.10 / 3.11 / 3.12
- API Key OpenAI / Gemini / DeepSeek (atau Local LLM via Ollama)

### 2. Install Dependensi
```bash
cd D:\Github\Autonomous_Candidate_Screening_Platform
pip install -r requirements.txt
```

### 3. Jalankan Interactive HR Dashboard PoC
```bash
streamlit run src/app.py
```

---

## 📑 Documentation & Deliverables
* 📄 [Product Requirement Document (PRD)](file:///D:/Github/Autonomous_Candidate_Screening_Platform/PRD.md)
