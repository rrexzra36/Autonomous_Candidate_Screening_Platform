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
try:
    from drive_importer import GoogleDriveImporter
except ImportError:
    from src.drive_importer import GoogleDriveImporter

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
    
    env_gemini_key = Config.GEMINI_API_KEY
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key:",
        value=env_gemini_key,
        type="password",
        help="Dapatkan Gemini API Key gratis di https://aistudio.google.com/"
    )
    active_api_key = Config.get_active_gemini_key(api_key_input)

    # Gemini 3.x Models & Custom Input Only
    model_options = [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "Input Model Kustom (Manual)"
    ]

    selected_model_choice = st.sidebar.selectbox(
        "Pilih Model Gemini:",
        model_options,
        index=0,
        help="Pilih model Gemini versi 3 atau pilih 'Input Model Kustom (Manual)' untuk mengetik nama model sendiri."
    )
    
    if selected_model_choice == "Input Model Kustom (Manual)":
        custom_model = st.sidebar.text_input("Ketik Nama Model Gemini:", value="gemini-3.5-flash")
        selected_model = custom_model.strip() if custom_model.strip() else "gemini-3.5-flash"
    else:
        selected_model = selected_model_choice

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

# Session State Initialization for Model Connection
if "api_connected" not in st.session_state:
    st.session_state["api_connected"] = False
if "connected_model" not in st.session_state:
    st.session_state["connected_model"] = ""
if "connected_provider" not in st.session_state:
    st.session_state["connected_provider"] = ""

if active_api_key:
    if not st.session_state["api_connected"]:
        if st.sidebar.button("🔗 Connect to Model", type="primary", use_container_width=True):
            with st.spinner("Menghubungkan ke model AI..."):
                try:
                    test_text = None
                    success_model = selected_model
                    last_err = None
                    
                    if selected_provider == "gemini":
                        try:
                            from google import genai
                            client = genai.Client(api_key=active_api_key)
                            res = client.models.generate_content(
                                model=selected_model,
                                contents="Katakan 'OK' dalam 1 kata."
                            )
                            test_text = res.text
                        except Exception as ec:
                            last_err = ec

                        if not test_text:
                            try:
                                import google.generativeai as legacy_genai
                                legacy_genai.configure(api_key=active_api_key)
                                mod = legacy_genai.GenerativeModel(selected_model)
                                res = mod.generate_content("Katakan 'OK' dalam 1 kata.")
                                test_text = res.text
                            except Exception as el:
                                last_err = el

                        if not test_text and last_err:
                            raise last_err

                    else:
                        import openai
                        client = openai.OpenAI(api_key=active_api_key)
                        res = client.chat.completions.create(
                            model=selected_model or "gpt-4o-mini",
                            messages=[{"role": "user", "content": "Katakan 'OK' dalam 1 kata."}]
                        )
                        test_text = res.choices[0].message.content
                    
                    if test_text:
                        st.session_state["api_connected"] = True
                        st.session_state["connected_model"] = success_model
                        st.session_state["connected_provider"] = provider_choice
                        st.sidebar.success(f"✅ Berhasil terhubung ke **{success_model}**")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Model tidak merespons.")
                except Exception as e:
                    err_str = str(e)
                    st.session_state["api_connected"] = False
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        st.sidebar.error("⚠️ **Limit Kuota (429):** Kuota request akun Anda habis. Tunggu 1 menit atau buat API key baru.")
                    elif "400" in err_str or "API_KEY_INVALID" in err_str:
                        st.sidebar.error("❌ **API Key Tidak Valid (400):** Periksa kembali karakter API key yang Anda masukkan.")
                    elif "404" in err_str:
                        st.sidebar.error(f"❌ **Model '{selected_model}' Tidak Ditemukan (404):** Periksa ketersediaan model pada akun Anda atau gunakan 'Input Model Kustom'.")
                    else:
                        st.sidebar.error(f"❌ **Detail Error:** {err_str}")

    else:
        st.sidebar.success(f"🟢 **Connected**")
        if st.sidebar.button("🔌 Disconnect", use_container_width=True):
            st.session_state["api_connected"] = False
            st.session_state["connected_model"] = ""
            st.session_state["connected_provider"] = ""
            st.rerun()

else:
    st.session_state["api_connected"] = False
    st.sidebar.info("⚡ Mode: Local Intelligent Rule Engine (Offline)")

st.sidebar.markdown("---")

min_score = st.sidebar.slider("Minimum Shortlist Score Threshold (%):", 0, 100, 60, 5)

# Effective API connection parameters
is_ai_connected = st.session_state.get("api_connected", False) and bool(active_api_key)
effective_api_key = active_api_key if is_ai_connected else ""
effective_model = st.session_state.get("connected_model", selected_model) if is_ai_connected else selected_model

# ==========================================
# STEP 1: JOB DESCRIPTION (KRITERIA LOWONGAN)
# ==========================================
st.header("1️⃣ Pengaturan Posisi & Kriteria Lowongan (Job Description)")

if "drive_jd_file" not in st.session_state:
    st.session_state["drive_jd_file"] = None

tab_jd_pdf, tab_jd_drive, tab_jd_text = st.tabs(["📄 Upload Dokumen PDF", "📁 Impor dari Google Drive", "✍️ Ketik / Tempel Teks Langsung"])

active_job = None

with tab_jd_pdf:
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
                    api_key=effective_api_key,
                    provider=selected_provider,
                    model_name=effective_model
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

with tab_jd_drive:
    st.markdown("**Impor Berkas Lowongan (Job Description) dari Google Drive:**")
    st.caption("💡 Masukkan tautan **1 file PDF spesifik** atau **folder Google Drive publik** yang memuat dokumen Job Description.")
    
    col_jd_dr_in, col_jd_dr_btn = st.columns([3, 1])
    with col_jd_dr_in:
        drive_jd_url = st.text_input(
            "Tautan (URL) Job Description Google Drive:",
            placeholder="Contoh: https://drive.google.com/file/d/... atau https://drive.google.com/drive/folders/...",
            help="Salin dan tempelkan link file atau folder Google Drive yang berisi dokumen lowongan kerja.",
            key="drive_jd_input"
        )
    with col_jd_dr_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Impor Lowongan", type="primary", use_container_width=True):
            if drive_jd_url and drive_jd_url.strip():
                with st.spinner("⏳ Menghubungi Google Drive & mengunduh berkas Job Description..."):
                    jd_files, err = GoogleDriveImporter.fetch_pdf_files_from_drive(drive_jd_url)
                    if err:
                        st.error(err)
                        st.session_state["drive_jd_file"] = None
                    else:
                        st.session_state["drive_jd_file"] = jd_files[0]
                        st.success(f"✅ Berhasil mengimpor berkas [{jd_files[0]['name']}] dari Google Drive.")
                        st.rerun()
            else:
                st.warning("⚠️ Masukkan tautan Google Drive terlebih dahulu.")

    if st.session_state.get("drive_jd_file"):
        jd_f = st.session_state["drive_jd_file"]
        col_jdt, col_jdc = st.columns([3, 1])
        with col_jdt:
            st.info(f"📄 **Berkas Terpilih dari Drive:** `{jd_f['name']}` ({round(jd_f['size']/1024, 1)} KB)")
        with col_jdc:
            if st.button("🗑️ Reset Berkas Drive", key="btn_reset_jd_drive"):
                st.session_state["drive_jd_file"] = None
                st.rerun()
        
        if active_job is None:
            with st.spinner(f"🤖 AI ({provider_choice}) sedang memvalidasi Job Description dari Google Drive..."):
                try:
                    jd_text = DocumentParser.extract_text_from_pdf(jd_f["bytes"])
                    active_job = DocumentParser.parse_job_description(
                        jd_text,
                        api_key=effective_api_key,
                        provider=selected_provider,
                        model_name=effective_model
                    )
                    st.success(f"✅ Berhasil mengekstrak kriteria lowongan: **{active_job['title']}**")
                except EmptyPDFError as e:
                    st.error(f"❌ **File PDF Tidak Dapat Dibaca / Kosong:** {str(e)}")
                    active_job = None
                except InvalidDocumentError as e:
                    st.error(f"❌ **Dokumen Tidak Sesuai:** {str(e)}")
                    active_job = None
                except Exception as e:
                    st.error(f"❌ **Gagal Memproses Dokumen:** {str(e)}")
                    active_job = None

with tab_jd_text:
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
        if active_job is None:
            with st.spinner(f"🤖 AI ({provider_choice}) sedang memproses teks Job Description..."):
                try:
                    active_job = DocumentParser.parse_job_description(
                        jd_raw_text.strip(),
                        api_key=effective_api_key,
                        provider=selected_provider,
                        model_name=effective_model
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
# PROTOKOL ANTI-BIAS & PRIVASI (BLIND-CV)
# ==========================================
active_masked_fields = []

with st.container(border=True):
    col_blind_title, col_blind_badge = st.columns([3, 2])
    with col_blind_title:
        st.markdown("#### 🛡️ Blind-CV Anonymization")
        enable_blind_cv = st.toggle(
            "Aktifkan Blind-CV Anonymization Layer",
            value=True,
            help="Otomatis menyamarkan informasi pribadi sensitif (PII) sebelum dievaluasi AI guna memastikan penilaian 100% berbasis keahlian (merit-based)."
        )

    if enable_blind_cv:
        
        st.markdown("**Pilih Parameter Identitas yang Ingin Disamarkan:**")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            if st.checkbox("Nama Lengkap", value=True, key="chk_name"): active_masked_fields.append("full_name")
            if st.checkbox("Alamat Email", value=True, key="chk_email"): active_masked_fields.append("email")
        with col_c2:
            if st.checkbox("Gender / Kelamin", value=True, key="chk_gender"): active_masked_fields.append("gender")
            if st.checkbox("Usia / Umur", value=True, key="chk_age"): active_masked_fields.append("age")
        with col_c3:
            if st.checkbox("Alamat Domisili", value=True, key="chk_address"): active_masked_fields.append("address")
            if st.checkbox("Foto Profil", value=True, key="chk_photo"): active_masked_fields.append("photo_url")
        with col_c4:
            if st.checkbox("Nama Kampus / Univ", value=True, key="chk_univ"): active_masked_fields.append("university")
            if st.checkbox("Nomor Telepon", value=True, key="chk_phone"): active_masked_fields.append("phone")

st.markdown("---")

# ==========================================
# STEP 2: CV KANDIDAT (UPLOAD / INGESTION)
# ==========================================
st.header("2️⃣ Pengumpulan & Upload CV Kandidat")

if "cv_uploader_key" not in st.session_state:
    st.session_state["cv_uploader_key"] = 0
if "parsed_cv_store" not in st.session_state:
    st.session_state["parsed_cv_store"] = {}
if "eval_results_store" not in st.session_state:
    st.session_state["eval_results_store"] = {}
if "drive_cv_files" not in st.session_state:
    st.session_state["drive_cv_files"] = []

tab_upload, tab_drive = st.tabs(["📤 Upload Berkas PDF Manual", "📁 Impor dari Google Drive Folder"])

raw_cv_items = []

with tab_upload:
    col_up_title, col_clear_btn = st.columns([3, 1])
    with col_up_title:
        st.markdown("**Unggah Dokumen CV Kandidat (Multiple PDF):**")
    with col_clear_btn:
        if st.button("🗑️ Hapus Semua CV", help="Klik untuk menghapus/mereset seluruh berkas CV yang telah diunggah."):
            st.session_state["cv_uploader_key"] += 1
            st.session_state["parsed_cv_store"] = {}
            st.session_state["eval_results_store"] = {}
            st.rerun()

    uploaded_cv_files = st.file_uploader(
        "Pilih atau drag & drop file PDF CV kandidat:",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"cv_uploader_{st.session_state['cv_uploader_key']}",
        label_visibility="collapsed"
    )
    if uploaded_cv_files:
        for f in uploaded_cv_files:
            raw_cv_items.append({"name": f.name, "bytes": f.getvalue()})

with tab_drive:
    st.markdown("**Impor Berkas CV dari Google Drive (Folder / 1 File Spesifik):**")
    st.caption("💡 Mendukung tautan **Folder** (multi-CV) maupun tautan **1 File PDF spesifik**. Pastikan izin akses telah diatur ke **'Anyone with the link can view'**.")
    
    col_dr_in, col_dr_btn = st.columns([3, 1])
    with col_dr_in:
        drive_folder_url = st.text_input(
            "Tautan (URL) Folder / File Google Drive:",
            placeholder="Contoh: https://drive.google.com/drive/folders/... atau https://drive.google.com/file/d/...",
            help="Salin dan tempelkan link folder atau link file Google Drive Anda di sini.",
            key="drive_folder_input"
        )
    with col_dr_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Impor dari Drive", type="primary", use_container_width=True):
            if drive_folder_url and drive_folder_url.strip():
                with st.spinner("⏳ Menghubungi Google Drive & mengunduh berkas PDF..."):
                    files, err = GoogleDriveImporter.fetch_pdf_files_from_drive(drive_folder_url)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["drive_cv_files"] = files
                        st.success(f"✅ Berhasil mengimpor {len(files)} berkas PDF dari Google Drive.")
                        st.rerun()
            else:
                st.warning("⚠️ Masukkan tautan folder Google Drive terlebih dahulu.")

    if st.session_state.get("drive_cv_files"):
        d_files = st.session_state["drive_cv_files"]
        col_dt, col_dc = st.columns([3, 1])
        with col_dt:
            st.write(f"📁 **{len(d_files)} berkas CV aktif dari Google Drive.**")
        with col_dc:
            if st.button("🗑️ Reset Drive Files", help="Hapus berkas yang diimpor dari Google Drive."):
                st.session_state["drive_cv_files"] = []
                st.session_state["eval_results_store"] = {}
                st.rerun()
        
        for f in d_files:
            raw_cv_items.append({"name": f["name"], "bytes": f["bytes"]})

candidates_to_process = []

if raw_cv_items:
    st.write(f"📋 **{len(raw_cv_items)} berkas CV siap diproses.**")
    invalid_cv_count = 0
    with st.spinner(f"🤖 Memvalidasi dan memproses CV kandidat..."):
        for item in raw_cv_items:
            fname = item["name"]
            file_bytes = item["bytes"]
            cv_cache_key = f"{fname}_{len(file_bytes)}_{effective_model}_{effective_api_key[:6] if effective_api_key else 'offline'}"
            
            # Cek apakah CV ini sudah pernah di-parse sebelumnya di memori sesi
            if cv_cache_key in st.session_state["parsed_cv_store"]:
                parsed_cv = st.session_state["parsed_cv_store"][cv_cache_key]
                candidates_to_process.append(parsed_cv)
                continue

            try:
                raw_text = DocumentParser.extract_text_from_pdf(file_bytes)
                parsed_cv = DocumentParser.parse_candidate_cv(
                    raw_text,
                    filename=fname,
                    api_key=effective_api_key,
                    provider=selected_provider,
                    model_name=effective_model
                )
                st.session_state["parsed_cv_store"][cv_cache_key] = parsed_cv
                candidates_to_process.append(parsed_cv)
            except EmptyPDFError:
                st.warning(f"⚠️ **File Dilewati [{fname}]:** Berkas kosong atau scan gambar tanpa teks digital.")
                invalid_cv_count += 1
            except InvalidDocumentError as e:
                st.warning(f"⚠️ **File Dilewati [{fname}]:** {str(e)}")
                invalid_cv_count += 1
            except Exception as e:
                st.warning(f"⚠️ **Gagal Memproses [{fname}]:** {str(e)}")
                invalid_cv_count += 1

    if candidates_to_process:
        st.success(f"✅ Berhasil memproses {len(candidates_to_process)} CV kandidat yang valid.")
    elif invalid_cv_count > 0:
        st.error("❌ Tidak ada CV valid yang dapat diproses. Silakan periksa kembali berkas Anda.")

st.markdown("---")

# ==========================================
# STEP 3: MATCHING & DASHBOARD RESULTS
# ==========================================
st.header("3️⃣ Evaluasi & Hasil Screening AI")

if "analysis_triggered" not in st.session_state:
    st.session_state["analysis_triggered"] = False

if not active_job:
    st.info("📋 Silakan atur atau unggah **Job Description** pada **Langkah 1** terlebih dahulu.")
elif not candidates_to_process:
    st.info("📤 Silakan unggah atau impor berkas **CV Kandidat** pada **Langkah 2** terlebih dahulu.")
else:
    with st.container(border=True):
        col_st_info, col_st_btn = st.columns([3, 1])
        with col_st_info:
            st.markdown(f"Siap mengevaluasi **{len(candidates_to_process)} CV kandidat** untuk posisi **{active_job['title']}**.")
            status_text = "🛡️ Protokol Blind-CV Aktif" if enable_blind_cv else "⚪ Mode Penilaian Standar"
            model_info = f"🤖 Engine: {provider_choice} ({effective_model})" if effective_api_key else "⚡ Engine: Local Intelligent Rule Engine (Offline)"
            st.caption(f"{status_text} | {model_info}")
        with col_st_btn:
            st.write("")
            if st.button("🚀 Mulai Analisis AI", type="primary", use_container_width=True):
                st.session_state["analysis_triggered"] = True
                st.rerun()

    if st.session_state.get("analysis_triggered"):
        matcher = CandidateMatcherEngine(
            api_key=effective_api_key,
            provider=selected_provider,
            model_name=effective_model
        )
        evaluated_results = []
        progress_bar = st.progress(0, text="Sedang menganalisis kecocokan kandidat...")

        for idx, raw_cv in enumerate(candidates_to_process):
            cand_name = raw_cv.get("personal_info", {}).get("full_name") or f"Kandidat #{idx+1}"
            progress_bar.progress((idx + 1) / len(candidates_to_process), text=f"🤖 Mengevaluasi {cand_name} ({idx+1}/{len(candidates_to_process)})...")
            cv_to_process = BlindCVAnonymizer.anonymize_cv(raw_cv, enabled_fields=active_masked_fields) if enable_blind_cv else raw_cv
            
            # Cache Key Evaluasi Scoring untuk mencegah Hit API berulang kali saat klik di UI
            eval_cache_key = f"{raw_cv.get('cv_id')}_{active_job.get('job_id')}_{enable_blind_cv}_{'_'.join(sorted(active_masked_fields))}_{effective_model}_{effective_api_key[:6] if effective_api_key else 'offline'}"
            
            if eval_cache_key in st.session_state["eval_results_store"]:
                eval_res = st.session_state["eval_results_store"][eval_cache_key]
            else:
                eval_res = matcher.evaluate_candidate(cv_to_process, active_job)
                eval_res["raw_cv"] = raw_cv
                eval_res["anonymized_cv"] = cv_to_process
                st.session_state["eval_results_store"][eval_cache_key] = eval_res

            evaluated_results.append(eval_res)

        progress_bar.empty()
        evaluated_results.sort(key=lambda x: x["overall_score"], reverse=True)

        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard & Hasil Seleksi", "🛡️ Blind-CV Anonymization", "📊 Distribusi & Analisis"])

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
                raw_personal = item["raw_cv"].get("personal_info", {})
                real_name = raw_personal.get("full_name") or item["candidate_alias"]
                alias_label = f" ({item['candidate_alias']})" if enable_blind_cv else ""
                
                with st.container(border=True):
                    st.markdown(f"#### #{rank} **{real_name}**{alias_label} — Skor Kecocokan: **{item['overall_score']}%**")
                    
                    col_a, col_b = st.columns(2)
                    col_a.markdown(f"📌 **Status Rekomendasi:** `{item['status']}`")
                    col_b.markdown(f"🎯 **Kesesuaian Skill:** `{item['score_breakdown']['skill_match']}%`")
                    if item.get("eval_source"):
                        st.caption(f"⚡ *Engine Analisis:* `{item['eval_source']}`")
                    
                    with st.expander("Review"):
                        active_profile = item["raw_cv"]
                        p_info = active_profile.get("personal_info", {})
                        
                        st.markdown("##### Informasi Kandidat")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.markdown(f"- **Nama:** {p_info.get('full_name', real_name)}")
                            st.markdown(f"- **Email:** {p_info.get('email', '-')}")
                            st.markdown(f"- **Telepon:** {p_info.get('phone', '-')}")
                        with col_info2:
                            age_val = p_info.get("age", "-")
                            age_disp = f"{age_val} tahun" if isinstance(age_val, (int, float)) else str(age_val)
                            st.markdown(f"- **Usia:** {age_disp}")
                            st.markdown(f"- **Gender:** {p_info.get('gender', '-')}")
                            st.markdown(f"- **Domisili:** {p_info.get('address', '-')}")

                        st.markdown("##### Riwayat Pendidikan")
                        edu_list = active_profile.get("education", [])
                        if isinstance(edu_list, list) and edu_list:
                            for edu in edu_list:
                                inst = edu.get("institution", "-")
                                deg = edu.get("degree", "-")
                                per = edu.get("period", "")
                                per_str = f" *({per})*" if per else ""
                                st.markdown(f"- **{inst}**{per_str}: {deg}")
                        elif isinstance(edu_list, dict):
                            st.markdown(f"- **{edu_list.get('institution', '-')}**: {edu_list.get('degree', '-')}")
                        else:
                            st.markdown("- *Tidak tercantum data pendidikan.*")

                        st.markdown("##### Pengalaman Kerja & Rekam Jejak")
                        exp_list = active_profile.get("work_experience", [])
                        if exp_list:
                            for exp in exp_list:
                                role = exp.get("role", "Posisi")
                                comp = exp.get("company", "Perusahaan")
                                period = exp.get("period", f"{exp.get('duration_years', 0)} Tahun")
                                st.markdown(f"- **{role}** di **{comp}** *({period})*")
                        else:
                            st.markdown("- *Tidak ada riwayat kerja spesifik.*")

                        col_sk1, col_sk2 = st.columns(2)
                        with col_sk1:
                            st.markdown("##### Keahlian Teknis (Technical Skills)")
                            tech_list = active_profile.get("technical_skills", [])
                            if tech_list:
                                for t in tech_list:
                                    st.markdown(f"- {t}")
                            else:
                                st.markdown("- *Tidak tercantum skill teknis khusus.*")

                        with col_sk2:
                            st.markdown("##### Keahlian Perilaku (Soft Skills)")
                            soft_list = active_profile.get("soft_skills", [])
                            if soft_list:
                                for s in soft_list:
                                    st.markdown(f"- {s}")
                            else:
                                st.markdown("- *Tidak tercantum soft skill khusus.*")

                        certs = active_profile.get("certifications", [])
                        if certs:
                            st.markdown("##### Sertifikasi & Pencapaian")
                            for c in certs:
                                st.markdown(f"- {c}")

                        st.markdown("---")
                        st.markdown("##### Analisis Kesesuaian AI (Pros & Cons)")
                        col_pro, col_con = st.columns(2)
                        with col_pro:
                            st.markdown("**Keunggulan Profil (Pros):**")
                            for pro in item["justification"]["pros"]:
                                st.markdown(f"- {pro}")
                        with col_con:
                            st.markdown("**Area Pertimbangan / Gap (Cons):**")
                            for con in item["justification"]["cons"]:
                                st.markdown(f"- {con}")

        with tab2:
            st.subheader("🛡️ Blind-CV Anonymization")
            st.info("Fitur ini memungkinkan Anda memilih secara spesifik informasi pribadi (PII) yang ingin disamarkan sebelum data dikirim ke sistem penilaian, memastikan evaluasi 100% berbasis kompetensi & rekam jejak.")
            
            cv_options = [c["cv_id"] for c in evaluated_results]
            selected_audit_id = st.selectbox("Pilih CV untuk Diaudit & Dibandingkan:", cv_options)
            target_audit = next(c for c in evaluated_results if c["cv_id"] == selected_audit_id)
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.error("❌ Data Mentah CV Asli (Lengkap)")
                st.json(target_audit["raw_cv"])
            with col_right:
                st.success("✅ Data Blind-CV yang Diterima AI (Lengkap & Terlindungi)")
                st.json(target_audit["anonymized_cv"])

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

st.caption("Autonomous Candidate Screening Platform v1.6.0 | AI Specialist Technical Assessment")
