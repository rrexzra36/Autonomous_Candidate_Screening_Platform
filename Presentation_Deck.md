# 📊 Slide Deck Outline: Autonomous Candidate Screening Platform
**AI Specialist Technical Assessment Proposal**

---

## Slide 1: Title & Executive Summary
* **Header:** Autonomous Candidate Screening Platform (TalentAI Engine)
* **Subtitle:** Solusi Penapisan & Perankingan CV Berbasis AI/ML untuk Pabrik & Industri Manufaktur
* **Presenter:** AI/ML Specialist Candidate
* **Key Takeaway:** Otomatisasi seleksi ribuan CV secara cepat (3 detik/CV), objektif (tanpa bias PII), dan transparan (*Explainable AI*).

---

## Slide 2: Latar Belakang & Tantangan Bisnis
* **Konteks:** Perusahaan manufaktur berkembang pesat, volume rekrutmen tinggi (*high-volume hiring*).
* **Masalah Utama Seleksi Manual:**
  1. *Time Bottleneck:* Membutuhkan 15-20 menit per CV (lambat dan mahal).
  2. *Human Bias:* Rentan terhadap bias gender, usia, almamater, dan kelelahan visual HR.
  3. *Unstructured Matching:* Kriteria kecocokan antar recruiter tidak terstandarisasi.
* **Solusi Ditawarkan:** Platform AI End-to-End dengan *Blind-CV Anonymizer* & *3-Tier Matching Engine*.

---

## Slide 3: Arsitektur Sistem & Data Flow
* **Flow Diagram:**
  `Job Portals / Email Ingestion` ➡️ `Blind-CV Anonymizer` ➡️ `Multi-Modal Entity Extractor` ➡️ `3-Tier Matching Engine` ➡️ `Interactive HR Dashboard`
* **Keunggulan Arsitektur:**
  - *Decoupled Microservices:* Komponen parser, anonymizer, dan scoring terpisah.
  - *High Throughput Queue:* Siap menangani 10.000 CV/hari.

---

## Slide 4: Inovasi AI/ML Utama
1. **Blind-CV Anonymizer (Bias Shield):** Menutupi PII (Nama, Foto, Gender, Age, Universitas) sebelum scoring.
2. **Two-Tier Hybrid Matching:**
   - *Tier 1 (Hard Filter):* Knockout instan untuk syarat mutlak (Min S1, Lisensi K3).
   - *Tier 2 (Vector Embedding & Skill Graph):* Kemiripan makna kontekstual (Python = Data Science).
   - *Tier 3 (LLM CoT Reasoning):* Evaluasi kualitas dampak pengalaman kerja.
3. **Explainable AI (XAI) Output:** Menghasilkan skor 0-100%, Pros/Cons, dan Rekomendasi Pertanyaan Wawancara.

---

## Slide 5: Pertimbangan Etika, Transparency & Governance
* **Fairness Guarantee:** Scoring murni 100% dari rekam jejak kapabilitas profesional.
* **Human-in-the-Loop (HITL):** AI memberikan rekomendasi, keputusan panggil wawancara di tangan HR.
* **Audit Trail & Feedback Loop:** Alasan override HR disimpan untuk tuning kriteria di masa mendatang.

---

## Slide 6: Proof of Concept (PoC) Demonstration
* *Tampilan Dashboard Streamlit:*
  - Leaderboard ranking kandidat real-time.
  - XAI Breakdown: Kenapa kandidat diposisikan sebagai `#1 Shortlisted`.
  - Audit tab membandingkan CV mentah vs CV anonymized.

---

## Slide 7: Business Impact & ROI
* **Penghematan Waktu:** Penurunan waktu penapisan hingga **> 90%** (dari 15 menit ke 3 detik/CV).
* **Penghematan Biaya:** Efisiensi operasional tim HR hingga **75%**.
* **Kualitas Hiring:** Rekomendasi objektif meningkatkan tingkat keberhasilan wawancara hingga **> 85%**.

---

## Slide 8: Next Steps & Technical Roadmap
* **Phase 1 (Week 1-2):** Integrasi API ATS & Webhook Jobstreet/LinkedIn.
* **Phase 2 (Week 3-4):** Fine-tuning model embedding lokal untuk istilah manufaktur spesifik.
* **Phase 3 (Month 2):** Launching produksi dengan dukungan multi-tenant department.
