
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

# Header
st.title("🤖 Autonomous Candidate Screening Platform")
st.caption("AI-Powered Talent Acquisition Engine for Rapid Screening with Ethical Blind Anonymization & Explainable AI (XAI)")

st.markdown("---")

# ==========================================
# SIDEBAR CONFIGURATION (AI PROVIDER & MODEL)
# ==========================================
st.sidebar.header("⚙️ Konfigurasi AI & Model")

provider_choice = st.sidebar.selectbox(
    "Pilih AI Provider:",
    ["Google Gemini", "OpenAI"],
    index=0
)

if provider_choice == "Google Gemini":
    selected_provider = "gemini"
    model_options = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
    default_model = Config.LLM_MODEL_NAME if Config.LLM_MODEL_NAME in model_options else "gemini-1.5-flash"
    selected_model = st.sidebar.selectbox(
        "Pilih Model Gemini:",
        model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        help="gemini-1.5-flash: Cepat & hemat kuota. gemini-1.5-pro: Penalaran mendalam."
    )
    
    env_gemini_key = Config.GEMINI_API_KEY
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key:",
        value=env_gemini_key,
        type="password",
        help="Dapatkan Gemini API Key gratis di https://aistudio.google.com/"
    )
    active_api_key = Config.get_active_gemini_key(api_key_input)

else:
    selected_provider = "openai"
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox(
        "Pilih Model OpenAI:",
        model_options,
        index=0,
        help="gpt-4o-mini: Cepat & efisien biaya. gpt-4o: Flagship reasoning."
    )
    
    env_openai_key = Config.OPENAI_API_KEY
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key:",
        value=env_openai_key,
        type="password",
        help="Dapatkan OpenAI API Key di https://platform.openai.com/api-keys"
    )
    active_api_key = Config.get_active_openai_key(api_key_input)

if active_api_key:
    st.sidebar.success(f"✨ {provider_choice} (Connected)")
else:
    st.sidebar.info("⚡ Local Intelligent Rule Engine (Offline)")

enable_blind_cv = st.sidebar.toggle("🛡️ Blind-CV Anonymization (Bias Shield)", value=True, help="Otomatis menyamarkan Nama, Foto, Gender, Usia, dan Almamater sebelum scoring.")
min_score = st.sidebar.slider("Minimum Shortlist Score Threshold (%):", 0, 100, 60, 5)

# ==========================================
# STEP 1: JOB DESCRIPTION (KRITERIA LOWONGAN)
# ==========================================
st.header("1️⃣ Pengaturan Posisi & Kriteria Lowongan (Job Description)")

jd_input_mode = st.radio(
    "Pilih Metode Input Job Description:",
    ["📄 Upload Dokumen PDF", "✍️ Ketik / Tempel Teks Langsung"],
    horizontal=True
)

active_job = None

if jd_input_mode == "📄 Upload Dokumen PDF":
    uploaded_jd_pdf = st.file_uploader(
        "Upload Dokumen PDF Job Description (berisi Job Title, Requirements, Responsibilities):",
        type=["pdf"],
        key="jd_pdf_uploader"
    )
    if uploaded_jd_pdf is not None:
        with st.spinner(f"🤖 AI ({provider_choice}) sedang membaca & memvalidasi PDF Job Description..."):
            try:
                jd_text = DocumentParser.extract_text_from_pdf(uploaded_jd_pdf.getvalue())
                active_job = DocumentParser.parse_job_description(
                    jd_text,
                    api_key=active_api_key,
                    provider=selected_provider,
                    model_name=selected_model
                )
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
        st.info("📄 Silakan upload berkas PDF Job Description untuk memulai proses seleksi.")

else:
    jd_raw_text = st.text_area(
        "Ketik atau tempel teks rincian lowongan kerja di sini:",
        height=220,
        placeholder=(
            "Contoh:\n"
            "Posisi: Junior Architect\n"
            "Jurusan: Architecture, Interior Design, or a related field\n"
            "Requirements: Minimum 2 years experience in design and build, AutoCAD, SketchUp, Revit, Technical Drawing...\n"
            "Responsibilities: To support project execution and design coordination..."
        ),
        key="jd_text_area"
    )
    if jd_raw_text and len(jd_raw_text.strip()) >= 20:
        with st.spinner(f"🤖 AI ({provider_choice}) sedang memproses teks Job Description..."):
            try:
                active_job = DocumentParser.parse_job_description(
                    jd_raw_text.strip(),
                    api_key=active_api_key,
                    provider=selected_provider,
                    model_name=selected_model
                )
                st.success(f"✅ Berhasil mengekstrak kriteria lowongan: **{active_job['title']}**")
            except InvalidDocumentError as e:
                st.error(f"❌ **Format Teks Kurang Lengkap:** {str(e)}")
                st.warning("💡 **Tips:** Pastikan teks memuat informasi nama posisi, kualifikasi/syarat, atau tanggung jawab pekerjaan.")
                active_job = None
            except Exception as e:
                st.error(f"❌ **Gagal Memproses Teks:** {str(e)}")
                active_job = None
    elif jd_raw_text:
        st.warning("⚠️ Teks terlalu pendek. Masukkan informasi posisi dan kualifikasi lowongan secara lebih lengkap.")
    else:
        st.info("✍️ Silakan ketik atau tempel teks rincian lowongan pekerjaan pada kotak di atas.")

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

candidates_to_process = []

if uploaded_cv_files:
    st.write(f"📁 **{len(uploaded_cv_files)} berkas CV diunggah untuk diproses.**")
    invalid_cv_count = 0
    with st.spinner(f"🤖 AI ({provider_choice}) sedang memvalidasi dan mengekstrak seluruh PDF CV..."):
        for cv_file in uploaded_cv_files:
            try:
                raw_text = DocumentParser.extract_text_from_pdf(cv_file.getvalue())
                parsed_cv = DocumentParser.parse_candidate_cv(
                    raw_text,
                    filename=cv_file.name,
                    api_key=active_api_key,
                    provider=selected_provider,
                    model_name=selected_model
                )
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
    st.info("📤 Silakan upload satu atau beberapa berkas CV kandidat dalam format PDF.")

st.markdown("---")

# ==========================================
# STEP 3: MATCHING & DASHBOARD RESULTS
# ==========================================
if candidates_to_process and active_job:
    matcher = CandidateMatcherEngine(
        api_key=active_api_key,
        provider=selected_provider,
        model_name=selected_model
    )
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
            display_name = item["candidate_alias"] if enable_blind_cv else item["raw_cv"]["personal_info"].get("full_name", item["cv_id"])
            
            with st.container(border=True):
                st.markdown(f"#### #{rank} **{display_name}** — Skor Kecocokan: **{item['overall_score']}%**")
                
                col_a, col_b = st.columns(2)
                col_a.markdown(f"📌 **Status Rekomendasi:** `{item['status']}`")
                col_b.markdown(f"🎯 **Kesesuaian Skill:** `{item['score_breakdown']['skill_match']}%`")
                
                with st.expander("🔍 Lihat Analisis Transparan AI (Pros & Cons)"):
                    st.markdown("**✅ Keunggulan Kandidat (Pros):**")
                    for pro in item["justification"]["pros"]:
                        st.markdown(f"- {pro}")
                    
                    st.markdown("**⚠️ Catatan / Potensi Gap (Cons):**")
                    for con in item["justification"]["cons"]:
                        st.markdown(f"- {con}")
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

elif not active_job and not candidates_to_process:
    st.info("💡 **Langkah Awal:** Silakan unggah dokumen PDF Job Description pada bagian (1) dan PDF CV kandidat pada bagian (2) di atas.")
elif not active_job:
    st.warning("⚠️ Dokumen Job Description yang valid belum diunggah. Silakan unggah berkas PDF lowongan pada bagian (1).")
elif not candidates_to_process:
    st.info("💡 Dokumen lowongan siap. Silakan unggah berkas CV kandidat pada bagian (2) untuk menjalankan evaluasi.")

st.caption("Autonomous Candidate Screening Platform v1.5.0 | AI Specialist Technical Assessment")
