"""
Autonomous Candidate Screening Platform - Streamlit HR Dashboard
Run with: streamlit run src/app.py
"""

import streamlit as st
import json
import os
import pandas as pd
from anonymizer import BlindCVAnonymizer
from matcher import CandidateMatcherEngine
from parser import DocumentParser, EmptyPDFError, InvalidDocumentError, DocumentParsingError
from config import Config

# Page Config
st.set_page_config(
    page_title="TalentAI - Autonomous Candidate Screening Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

@st.cache_data
def load_preset_data():
    with open(os.path.join(DATA_DIR, "job_descriptions.json"), "r", encoding="utf-8") as f:
        jobs = json.load(f)
    with open(os.path.join(DATA_DIR, "sample_cvs.json"), "r", encoding="utf-8") as f:
        cvs = json.load(f)
    return jobs, cvs

preset_jobs, preset_cvs = load_preset_data()

# Header
st.title("🤖 Autonomous Candidate Screening Platform")
st.caption("AI-Powered Talent Acquisition Engine for High-Volume Manufacturing Hiring with Ethical Blind Anonymization & Explainable AI (XAI)")

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Konfigurasi Sistem & AI")

env_gemini_key = Config.GEMINI_API_KEY
gemini_key_input = st.sidebar.text_input(
    "Google Gemini API Key (Opsional):",
    value=env_gemini_key,
    type="password",
    help="Masukkan API key Gemini untuk ekstraksi LLM tingkat lanjut. Jika dikosongkan, sistem otomatis menggunakan heuristic rule engine lokal."
)

active_api_key = Config.get_active_gemini_key(gemini_key_input)
if active_api_key:
    st.sidebar.success("✨ Gemini Generative AI: Terhubung (Pertanyaan Wawancara Personal & Deep Reasoning Aktif)")
else:
    st.sidebar.info("⚡ Mode: Local Intelligent Rule Engine (Offline)")

enable_blind_cv = st.sidebar.toggle("🛡️ Blind-CV Anonymization (Bias Shield)", value=True, help="Otomatis menyamarkan Nama, Foto, Gender, Usia, dan Almamater sebelum scoring.")
min_score = st.sidebar.slider("Minimum Shortlist Score Threshold (%):", 0, 100, 60, 5)

# ==========================================
# STEP 1: JOB DESCRIPTION (KRITERIA LOWONGAN)
# ==========================================
st.header("1️⃣ Pengaturan Posisi & Kriteria Lowongan (Job Description)")

jd_source = st.radio(
    "Pilih Sumber Job Description:",
    ["📄 Upload PDF Job Description", "📂 Gunakan Preset Manufaktur"],
    horizontal=True
)

active_job = None

if jd_source == "📄 Upload PDF Job Description":
    uploaded_jd_pdf = st.file_uploader("Upload Dokumen PDF Job Description (berisi Job Title, Requirements, Responsibilities):", type=["pdf"])
    if uploaded_jd_pdf is not None:
        with st.spinner("🤖 AI sedang membaca & memvalidasi PDF Job Description..."):
            try:
                jd_text = DocumentParser.extract_text_from_pdf(uploaded_jd_pdf.getvalue())
                active_job = DocumentParser.parse_job_description(jd_text, api_key=active_api_key)
                st.success(f"✅ Berhasil mengekstrak kriteria lowongan: **{active_job['title']}**")
            except EmptyPDFError as e:
                st.error(f"❌ **File PDF Tidak Dapat Dibaca / Kosong:** {str(e)}")
                st.info("💡 **Solusi:** Pastikan berkas PDF memiliki teks digital (bukan hasil scan/foto tanpa layer OCR teks).")
                active_job = None
            except InvalidDocumentError as e:
                st.error(f"❌ **Dokumen Tidak Sesuai:** {str(e)}")
                st.warning("💡 **Tips:** Pastikan berkas yang diunggah benar-benar memuat informasi lowongan kerja, kualifikasi/syarat, atau deskripsi pekerjaan.")
                active_job = None
            except Exception as e:
                st.error(f"❌ **Gagal Memproses PDF:** {str(e)}")
                active_job = None
    else:
        st.info("Silakan upload file PDF Job Description, atau beralih ke 'Gunakan Preset Manufaktur'.")
        active_job = preset_jobs[0]
else:
    selected_preset_title = st.selectbox(
        "Pilih Template Lowongan Manufaktur:",
        [j["title"] for j in preset_jobs]
    )
    active_job = next(j for j in preset_jobs if j["title"] == selected_preset_title)

# Display extracted/active Job Criteria
if active_job:
    with st.expander(f"📋 Rincian Kriteria Teridentifikasi: **{active_job['title']}**", expanded=True):
        st.markdown(f"**Posisi:** {active_job.get('title', 'Posisi Pekerjaan')}")
        st.markdown(f"**Jurusan/ Program Studi:** {active_job.get('major', active_job.get('department', 'Semua Jurusan Terkait'))}")
        st.markdown(f"**Min. Pendidikan:** {active_job['hard_requirements'].get('min_education', 'S1')}")
        st.markdown(f"**Min. Pengalaman:** {active_job['hard_requirements'].get('min_experience_years', 1)} Tahun")
        
        t_skills = active_job.get('technical_skills', [])
        s_skills = active_job.get('soft_skills', [])
        if not t_skills and not s_skills:
            t_skills, s_skills = DocumentParser.classify_skills(active_job.get('key_skills', []))
            
        st.markdown(f"**Technical Skills:** {', '.join(t_skills) if t_skills else '-'}")
        st.markdown(f"**Soft Skills:** {', '.join(s_skills) if s_skills else '-'}")
        st.markdown(f"**Tanggung Jawab (Responsibilities):**\n\n{active_job.get('responsibilities', active_job.get('description', ''))}")

st.markdown("---")

# ==========================================
# STEP 2: CV KANDIDAT (UPLOAD / INGESTION)
# ==========================================
st.header("2️⃣ Pengumpulan & Upload CV Kandidat")

cv_source = st.radio(
    "Pilih Sumber CV Kandidat:",
    ["📤 Upload Berkas CV (Multiple PDF)", "📂 Gunakan Dataset Sampel CV"],
    horizontal=True
)

candidates_to_process = []

if cv_source == "📤 Upload Berkas CV (Multiple PDF)":
    if "cv_uploader_key" not in st.session_state:
        st.session_state["cv_uploader_key"] = 0

    col_up_title, col_clear_btn = st.columns([3, 1])
    with col_up_title:
        st.markdown("**Unggah Dokumen CV Kandidat (Multiple PDF):**")
    with col_clear_btn:
        if st.button("🗑️ Hapus Semua CV", help="Klik untuk menghapus/mereset seluruh berkas CV yang telah diunggah."):
            st.session_state["cv_uploader_key"] += 1
            st.rerun()

    uploaded_cv_files = st.file_uploader(
        "Pilih atau drag & drop file PDF CV kandidat:",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"cv_uploader_{st.session_state['cv_uploader_key']}",
        label_visibility="collapsed"
    )
    if uploaded_cv_files:
        st.write(f"📁 **{len(uploaded_cv_files)} berkas CV diunggah untuk diproses.**")
        invalid_cv_count = 0
        with st.spinner("🤖 AI sedang memvalidasi dan mengekstrak seluruh PDF CV..."):
            for cv_file in uploaded_cv_files:
                try:
                    raw_text = DocumentParser.extract_text_from_pdf(cv_file.getvalue())
                    parsed_cv = DocumentParser.parse_candidate_cv(raw_text, filename=cv_file.name, api_key=active_api_key)
                    candidates_to_process.append(parsed_cv)
                except EmptyPDFError:
                    st.warning(f"⚠️ **File Dilewati [{cv_file.name}]:** Berkas kosong atau scan gambar tanpa teks digital.")
                    invalid_cv_count += 1
                except InvalidDocumentError as e:
                    st.warning(f"⚠️ **File Dilewati [{cv_file.name}]:** {str(e)}")
                    invalid_cv_count += 1
                except Exception as e:
                    st.warning(f"⚠️ **Gagal Memproses [{cv_file.name}]:** {str(e)}")
                    invalid_cv_count += 1

        if candidates_to_process:
            st.success(f"✅ Berhasil memproses {len(candidates_to_process)} CV kandidat yang valid.")
        elif invalid_cv_count > 0:
            st.error("❌ Tidak ada CV valid yang dapat diproses. Silakan periksa kembali berkas Anda.")
    else:
        st.info("Silakan upload satu atau beberapa berkas CV dalam format PDF.")
else:
    candidates_to_process = preset_cvs
    st.success(f"✅ Menggunakan {len(preset_cvs)} sampel CV bawaan manufaktur.")

st.markdown("---")

# ==========================================
# STEP 3: MATCHING & DASHBOARD RESULTS
# ==========================================
if candidates_to_process and active_job:
    matcher = CandidateMatcherEngine(gemini_api_key=active_api_key)
    evaluated_results = []

    for raw_cv in candidates_to_process:
        cv_to_process = BlindCVAnonymizer.anonymize_cv(raw_cv) if enable_blind_cv else raw_cv
        eval_res = matcher.evaluate_candidate(cv_to_process, active_job)
        eval_res["raw_cv"] = raw_cv
        eval_res["anonymized_cv"] = cv_to_process
        evaluated_results.append(eval_res)

    evaluated_results.sort(key=lambda x: x["overall_score"], reverse=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard & Hasil Seleksi", "🛡️ Audit Blind-CV (Anti-Bias)", "📊 Distribusi & Analisis"])

    with tab1:
        st.subheader(f"Hasil Evaluasi Kandidat untuk Posisi: {active_job['title']}")
        
        filtered_list = [c for c in evaluated_results if c["overall_score"] >= min_score]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            with st.container(border=True):
                st.metric("📁 Total CV Diproses", len(evaluated_results))
        with m2:
            with st.container(border=True):
                st.metric("🎯 Kandidat Lolos Shortlist", len(filtered_list))
        with m3:
            with st.container(border=True):
                avg_score = round(sum(c['overall_score'] for c in evaluated_results) / len(evaluated_results), 1) if evaluated_results else 0
                st.metric("📈 Rerata Skor Kesesuaian", f"{avg_score}%")

        st.markdown("### 📋 Daftar Kandidat Terurut (Ranking)")

        for rank, item in enumerate(evaluated_results, start=1):
            is_passed = item["overall_score"] >= min_score and item["hard_filter_passed"]
            status_icon = "🟢" if is_passed else ("🟡" if item["overall_score"] >= 50 else "🔴")
            display_name = item["candidate_alias"] if enable_blind_cv else item["raw_cv"]["personal_info"].get("full_name", item["cv_id"])
            
            with st.container(border=True):
                st.markdown(f"#### #{rank} {status_icon} **{display_name}** — Skor Kecocokan: **{item['overall_score']}%**")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.markdown(f"📌 **Status Rekomendasi:** `{item['status']}`")
                col_b.markdown(f"🎯 **Kesesuaian Skill:** `{item['score_breakdown']['skill_match']}%`")
                col_c.markdown(f"⏱️ **Durasi Pengalaman:** `{item['score_breakdown']['experience_depth']}%`")
                
                with st.expander("🔍 Lihat Analisis Transparan AI (Pros, Cons, & Rekomendasi Wawancara)"):
                    st.markdown("**✅ Keunggulan Kandidat (Pros):**")
                    for pro in item["justification"]["pros"]:
                        st.markdown(f"- {pro}")
                    
                    st.markdown("**⚠️ Catatan / Potensi Gap (Cons):**")
                    for con in item["justification"]["cons"]:
                        st.markdown(f"- {con}")
                    
                    st.markdown("**❓ Rekomendasi Pertanyaan Wawancara dari AI:**")
                    for idx, q in enumerate(item["justification"]["interview_questions"], start=1):
                        st.markdown(f"{idx}. *{q}*")
                st.markdown("---")

    with tab2:
        st.subheader("🛡️ Verifikasi Blind-CV Anonymization (Etika AI & Anti-Bias)")
        st.info("Fitur ini menunjukkan bagaimana identitas pribadi (Nama, Gender, Foto, Umur, Nama Universitas) disamarkan sebelum data dikirim ke sistem penilaian, memastikan seleksi 100% berdasarkan rekam jejak & kompetensi.")
        
        cv_options = [c["cv_id"] for c in evaluated_results]
        selected_audit_id = st.selectbox("Pilih CV untuk Diaudit:", cv_options)
        target_audit = next(c for c in evaluated_results if c["cv_id"] == selected_audit_id)
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.error("❌ Data Mentah CV Asli (Rentan Bias Manusia)")
            st.json(target_audit["raw_cv"].get("personal_info", {}))
        with col_right:
            st.success("✅ Data Blind-CV yang Diterima AI (Bebas Bias)")
            st.json(target_audit["anonymized_cv"].get("personal_info", {}))

    with tab3:
        st.subheader("📊 Analisis Distribusi Skor Kandidat")
        df_plot = pd.DataFrame([
            {
                "Kandidat": c["candidate_alias"] if enable_blind_cv else c["raw_cv"]["personal_info"].get("full_name", c["cv_id"]),
                "Total Skor (%)": c["overall_score"],
                "Skor Skill (%)": c["score_breakdown"]["skill_match"],
                "Skor Pengalaman (%)": c["score_breakdown"]["experience_depth"]
            } for c in evaluated_results
        ])
        st.bar_chart(df_plot.set_index("Kandidat"))

elif not active_job:
    st.warning("⚠️ Silakan lengkapi atau perbaiki upload dokumen Job Description yang valid terlebih dahulu.")
else:
    st.info("Silakan unggah CV kandidat untuk memulai proses penapisan.")

st.caption("Autonomous Candidate Screening Platform v1.3.0 | AI Specialist Technical Assessment")
