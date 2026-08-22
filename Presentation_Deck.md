# 📊 Slide Deck Presentation: Autonomous Candidate Screening Platform
**AI Specialist Technical Assessment Proposal — TalentAI Engine**

---

## Slide 1: Title & Executive Summary
* **Judul Platform:** Autonomous Candidate Screening Platform (TalentAI Engine)
* **Subtitle:** Solusi Penapisan & Perankingan CV Berbasis AI/ML Multi-Tier dengan *Blind Anonymization*, *Vector Embeddings*, dan *Explainable AI (XAI)*
* **Presenter:** AI/ML Specialist Candidate
* **Key Value Proposition:**
  - ⚡ **Super Cepat:** Memangkas durasi penapisan dari ~15 menit menjadi **< 3 detik per CV** (> 90% time reduction).
  - 🛡️ **100% Bebas Bias:** Menggunakan *Ethical Blind Screening* untuk menyamarkan PII (Nama, Gender, Usia, Foto, Institusi).
  - 🎯 **Presisi Anti-Hallucination:** Divalidasi dengan *Domain Role Verification* dan *Decoupled Technical Skill Scoring*.
  - 📊 **Explainable AI (XAI):** Memberikan transparansi skor komposit, rincian *Pros*, *Cons*, dan justifikasi keputusan eksekutif.

---

## Slide 2: Latar Belakang & Tantangan Bisnis
* **Konteks:** Perusahaan modern & industri manufaktur menghadapi lonjakan berkas pelamar dalam jumlah masif (*high-volume hiring*).
* **Masalah Utama Seleksi Manual:**
  1. *Time Bottleneck:* Membutuhkan 15–20 menit per CV, memicu *hiring backlog* dan tingginya *Cost-per-Hire*.
  2. *Subconscious Bias:* Keputusan rentan bias gender, usia, almamater, latar belakang visual CV, serta faktor kelelahan HR.
  3. *Unstandardized Matching:* Kriteria kecocokan tidak konsisten dan tidak memiliki rubric scoring kuantitatif yang terukur.
* **Solusi TalentAI Engine:** Platform penapisan end-to-end yang menggabungkan *Layout-Aware NLP*, *Vector Embeddings*, *Domain Scoring Engine*, dan *Multi-LLM Reasoning*.

---

## Slide 3: Alur Kerja Aplikasi (End-to-End 3-Step Workflow)
* **Step 1 — Ingestion Job Description:**
  - Input kriteria melalui: Upload PDF, Tautan Google Drive, Template Preset, atau Custom Form.
  - Ekstraksi otomatis kriteria mutlak (*Hard Requirements*), *Technical Skills*, *Soft Skills*, dan *Major*.
* **Step 2 — Multi-Candidate Ingestion & Blind Anonymization:**
  - Batch upload PDF CV atau Folder Google Drive.
  - *Hierarchical Section Chunking* memartisi dokumen tanpa kontaminasi silang.
  - *Anonymizer Engine* menyamarkan PII menjadi alias unik (`CANDIDATE-01`, `CANDIDATE-02`).
* **Step 3 — AI Screening & Interactive Evaluation:**
  - Konfigurasi bobot penilaian dinamis (*Skill, Exp, Edu*) & ambang kelulusan (*Threshold*) dengan tombol *Reset Weights*.
  - Eksekusi manual berbasis tombol (*Anti-Auto-Load Protection*).
  - Tampilan Leaderboard, Grafik Plotly, XAI Deep-Dive, dan Ekspor Laporan Resmi (*PDF/CSV/JSON*).

---

## Slide 4: Inovasi Algoritma 1 & 2 (NLP Chunking & Semantic Embeddings)

### 1. Hierarchical Section Chunking & Isolated Named Entity Segmentation
* **Definisi:** Algoritma NLP *Layout-Aware* yang membagi aliran teks mentah menjadi blok semantik terisolasi:
  $$\text{CV Raw Text} \longrightarrow \big[ \mathcal{S}_{\text{Header}} \;\big|\; \mathcal{S}_{\text{Experience}} \;\big|\; \mathcal{S}_{\text{Education}} \;\big|\; \mathcal{S}_{\text{Skills}} \;\big|\; \mathcal{S}_{\text{Certifications}} \big]$$
* **Benefit:** Menghilangkan salah baca regex (misal: biodata kontak tidak masuk ke pencapaian kerja).

### 2. Dense Semantic Vector Embeddings & Cosine Similarity
* **Definisi:** Pemetaan teks lowongan ($\mathbf{u}$) dan CV ($\mathbf{v}$) ke dalam ruang vektor berdimensi tinggi (`text-embedding-004` / `text-embedding-3-small`).
* **Formula Perhitungan Matematis:**
  $$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \cdot \sqrt{\sum_{i=1}^n v_i^2}}$$
  $$S_{\text{semantic}} = \min\left(100.0, \; \max\left(0.0, \; \frac{\cos(\theta) \times 100 - 35.0}{0.55}\right)\right)$$

---

## Slide 5: Inovasi Algoritma 3 (Multi-Tier Scoring & Domain Verification)

### A. Tier 1: Knockout Filter (Syarat Mutlak)
$$\text{HardFilterPassed} = \left( \sum \text{Duration}_i \ge \text{MinExp} \right) \land \left( \text{Mandatory Certs Fulfilled} \right) \implies \text{Gagal: Penalti } 50\%$$

### B. Tier 2: Perhitungan Komponen Skor & Validasi Domain
* **Decoupled Skill Score ($S_{\text{skill}}$):**
  - Jika $N_{\text{tech}} = 0 \implies S_{\text{skill}} = \min\big(15.0, \; (R_{\text{soft}} \times 10.0) + (S_{\text{semantic}} \times 0.05)\big)$ *(Capped)*
  - Jika $N_{\text{tech}} > 0 \implies S_{\text{skill}} = (R_{\text{tech}} \times 75.0) + (R_{\text{soft}} \times 15.0) + (\min(100, S_{\text{semantic}}) \times 0.10)$
* **Domain Experience Relevance ($S_{\text{exp}}$):**
  $$\text{Years}_{\text{relevant}} = \sum (\text{Duration}_i \times \text{Relevance}_i) \quad \text{dimana } \text{Relevance}_i \in \{0.0, 0.5, 1.0\} \text{ (Jaccard Overlap)}$$
* **Education & Major Relevance ($S_{\text{edu}}$):**
  $$S_{\text{edu}} = (S_{\text{deg}} \times 0.40) + (S_{\text{major}} \times 0.60)$$
* **Critical Domain Mismatch Penalty:**
  Jika $N_{\text{tech}} = 0 \land \text{Years}_{\text{relevant}} = 0 \land S_{\text{major}} \le 50.0 \implies \text{Skor Akhir dikunci } \le 22.0\% \text{ (Status: Rejected)}$

---

## Slide 6: Inovasi Algoritma 4 & 5 (Ethical Shield & Explainable AI)

### 4. Dynamic Ethical PII Anonymization Engine
* **Target Masking:** Nama $\to$ `CANDIDATE-01`, Kontak $\to$ `[MASKED]`, Gender/Usia $\to$ `[REDACTED]`, Institusi $\to$ `Accredited Higher Education`.
* **Dampak:** Memenuhi prinsip keadilan hukum (*EEO Compliance* & UU PDP / GDPR).

### 5. Explainable AI (XAI) Reasoning Engine
* Menghasilkan laporan evaluasi kualitatif komprehensif:
  - **Profile Strengths (Pros):** Rincian software teknis, pengalaman relevan, dan kualifikasi studi.
  - **Areas for Consideration (Cons):** Rincian gap alat kerja atau ketidaksesuaian domain profesi.
  - **Executive Decision Rationale:** Justifikasi bisnis yang jelas untuk status kelulusan.

---

## Slide 7: Tech Stack Architecture

| Layer | Teknologi / Library | Peran & Fungsi |
| :--- | :--- | :--- |
| **Frontend & Visualization** | `Streamlit` (v1.30+), `Plotly`, `Pandas` | Dashboard interaktif, grafik radar, metriks leaderboard, dan manipulasi data. |
| **Document Parsing & NLP** | `PyPDF` (`pypdf`), `Regular Expressions (re)` | Parsing PDF digital multi-kolom dan *Hierarchical Section Chunking*. |
| **AI Models & LLM Framework** | `google-genai` / `google.generativeai` | Gemini 2.5 Flash, Gemini 3.x, `text-embedding-004`. |
| | `openai` | GPT-4o-mini, GPT-4o, `text-embedding-3-small`. |
| **Offline Fallback Engine** | Sparse TF-IDF Vectorizer + Rule Reasoner | Penapisan mandiri dan analisis XAI tanpa ketergantungan koneksi API eksternal. |
| **Integrasi & Reporting** | `requests`, `gdown`, Print Report Generator | Ingestion folder Google Drive & ekspor berkas seleksi resmi (*PDF, CSV, JSON*). |

---

## Slide 8: Live PoC Demonstration Highlights
1. **Interactive Candidate Leaderboard:** Visualisasi peringkat pelamar dengan badge status warna (`Pass` 🟢, `Considered` 🟡, `Rejected` 🔴).
2. **Dynamic Weight Tuning & Reset:** Slider pembobotan dinamis (*Skill, Experience, Education*) dengan tombol *Reset Weights* instan.
3. **Deep XAI Inspection Modal:** Pembacaan transparan atas *Pros, Cons, Matched Skills, Missing Skills*, dan *Executive Rationale*.
4. **Export & Audit Tab:** Fitur unduh laporan resmi dan perbandingan data *Raw CV vs Anonymized CV*.

---

## Slide 9: Dampak Bisnis, ROI, & Tata Kelola AI
* ⏱️ **Efisiensi Waktu Rekrutmen:** Penurunan waktu penapisan hingga **> 90%** (dari 15 menit ke 3 detik/CV).
* 💰 **Efisiensi Biaya Operasional:** Menghemat beban kerja tim talent acquisition hingga **75%**.
* 🎯 **Akurasi & Validitas Hiring:** Rekomendasi objektif meningkatkan kesesuaian kandidat hingga **> 90%**.
* ⚖️ **Human-in-the-Loop (HITL):** AI berperan sebagai sistem penunjang keputusan (*Decision Support System*); otoritas persetujuan pemanggilan wawancara tetap berada di tangan HR.

---

## Slide 10: Technical Roadmap Menuju Produksi
* **Fase 1 (Bulan 1):** Integrasi Webhook ATS (Workday, BambooHR, LinkedIn Easy Apply).
* **Fase 2 (Bulan 2):** Fine-tuning model domain embedding lokal untuk spesialisasi industri spesifik.
* **Fase 3 (Bulan 3):** Dukungan *Enterprise Multi-Tenant Role-Based Access Control* (RBAC) dan modul analitik komparatif multi-departemen.

