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
st.sidebar.header("⚙️ AI & Model Configuration")

provider_choice = st.sidebar.selectbox(
    "Select AI Provider:",
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
        help="Get your free Gemini API Key at https://aistudio.google.com/"
    )
    active_api_key = Config.get_active_gemini_key(api_key_input)

    # Gemini 3.x Models & Custom Input Only
    model_options = [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "Input Custom Model (Manual)"
    ]

    selected_model_choice = st.sidebar.selectbox(
        "Select Gemini Model:",
        model_options,
        index=0,
        help="Select a Gemini version 3 model or choose 'Input Custom Model (Manual)' to specify your own model name."
    )
    
    if selected_model_choice == "Input Custom Model (Manual)":
        custom_model = st.sidebar.text_input("Type Gemini Model Name:", value="gemini-3.5-flash")
        selected_model = custom_model.strip() if custom_model.strip() else "gemini-3.5-flash"
    else:
        selected_model = selected_model_choice

else:
    selected_provider = "openai"
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox(
        "Select OpenAI Model:",
        model_options,
        index=0,
        help="gpt-4o-mini: Fast & cost-effective. gpt-4o: Flagship reasoning."
    )
    
    env_openai_key = Config.OPENAI_API_KEY
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key:",
        value=env_openai_key,
        type="password",
        help="Get your OpenAI API Key at https://platform.openai.com/api-keys"
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
            with st.spinner("Connecting to AI model..."):
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
                                contents="Say 'OK' in 1 word."
                            )
                            test_text = res.text
                        except Exception as ec:
                            last_err = ec

                        if not test_text:
                            try:
                                import google.generativeai as legacy_genai
                                legacy_genai.configure(api_key=active_api_key)
                                mod = legacy_genai.GenerativeModel(selected_model)
                                res = mod.generate_content("Say 'OK' in 1 word.")
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
                            messages=[{"role": "user", "content": "Say 'OK' in 1 word."}]
                        )
                        test_text = res.choices[0].message.content
                    
                    if test_text:
                        st.session_state["api_connected"] = True
                        st.session_state["connected_model"] = success_model
                        st.session_state["connected_provider"] = provider_choice
                        st.sidebar.success(f"Successfully connected to **{success_model}**")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Model did not respond.")
                except Exception as e:
                    err_str = str(e)
                    st.session_state["api_connected"] = False
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        st.sidebar.error("⚠️ **Quota Limit (429):** Account quota exhausted. Please wait 1 minute or generate a new API key.")
                    elif "400" in err_str or "API_KEY_INVALID" in err_str:
                        st.sidebar.error("❌ **Invalid API Key (400):** Please check the API key you entered.")
                    elif "404" in err_str:
                        st.sidebar.error(f"❌ **Model '{selected_model}' Not Found (404):** Check model availability or use 'Input Custom Model (Manual)'.")
                    else:
                        st.sidebar.error(f"❌ **Error Details:** {err_str}")

    else:
        st.sidebar.success(f"**Connected to {st.session_state.get('connected_model')}**")
        if st.sidebar.button("🔌 Disconnect", use_container_width=True):
            st.session_state["api_connected"] = False
            st.session_state["connected_model"] = ""
            st.session_state["connected_provider"] = ""
            st.rerun()

else:
    st.session_state["api_connected"] = False
    st.sidebar.info("Local Intelligent Rule Engine (Offline)")

st.sidebar.markdown("---")

# Effective API connection parameters
is_ai_connected = st.session_state.get("api_connected", False) and bool(active_api_key)
effective_api_key = active_api_key if is_ai_connected else ""
effective_model = st.session_state.get("connected_model", selected_model) if is_ai_connected else selected_model

# ==========================================
# STEP 1: JOB DESCRIPTION (POSITION CRITERIA)
# ==========================================
st.header("1️⃣ Job Position & Criteria Setup (Job Description)")

if "jd_uploader_key" not in st.session_state:
    st.session_state["jd_uploader_key"] = 0
if "drive_jd_file" not in st.session_state:
    st.session_state["drive_jd_file"] = None

tab_jd_pdf, tab_jd_drive, tab_jd_text = st.tabs([
    "📤 PDF Upload", 
    "📁 Import Google Drive", 
    "✍️ Type Text"
])

active_job = None

with tab_jd_pdf:
    col_jd_title, col_jd_reset = st.columns([3, 1], vertical_alignment="center")
    with col_jd_title:
        st.markdown("**Upload Job Description PDF Document:**")
    with col_jd_reset:
        if st.button("🗑️ Reset PDF", use_container_width=True, help="Click to reset the uploaded Job Description PDF."):
            st.session_state["jd_uploader_key"] += 1
            st.session_state["drive_jd_file"] = None
            st.session_state["executed_config_sig"] = ""
            st.rerun()

    uploaded_jd_pdf = st.file_uploader(
        "Upload Job Description PDF Document:",
        type=["pdf"],
        key=f"jd_pdf_uploader_{st.session_state['jd_uploader_key']}",
        label_visibility="collapsed"
    )
    if uploaded_jd_pdf is not None:
        with st.spinner(f"🤖 AI ({provider_choice}) is reading & validating the Job Description PDF..."):
            try:
                jd_text = DocumentParser.extract_text_from_pdf(uploaded_jd_pdf.getvalue())
                active_job = DocumentParser.parse_job_description(
                    jd_text,
                    api_key=effective_api_key,
                    provider=selected_provider,
                    model_name=effective_model
                )
                st.success(f"✅ Successfully extracted job criteria: **{active_job['title']}**")
            except EmptyPDFError as e:
                st.error(f"❌ **Unreadable / Empty PDF File:** {str(e)}")
                st.info("💡 **Solution:** Ensure the PDF contains digital text (not a scanned image without OCR text).")
                active_job = None
            except InvalidDocumentError as e:
                st.error(f"❌ **Invalid Document:** {str(e)}")
                st.warning("💡 **Tip:** Ensure the uploaded document contains genuine job vacancy requirements or responsibilities.")
                active_job = None
            except Exception as e:
                st.error(f"❌ **Failed to Process PDF:** {str(e)}")
                active_job = None

with tab_jd_drive:
    col_jd_dr_title, col_jd_dr_reset = st.columns([3, 1], vertical_alignment="center")
    with col_jd_dr_title:
        st.markdown("**Import Job Description from Google Drive:**")
    with col_jd_dr_reset:
        if st.button("🗑️ Reset Drive File", key="btn_reset_jd_drive", use_container_width=True, help="Click to reset the Job Description imported from Google Drive."):
            st.session_state["drive_jd_file"] = None
            st.session_state["executed_config_sig"] = ""
            st.rerun()

    st.caption("💡 Supports a **specific single PDF file link** or a **public Google Drive folder** containing job vacancy documents.")
    
    col_jd_dr_in, col_jd_dr_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_jd_dr_in:
        drive_jd_url = st.text_input(
            "Google Drive Job Description URL / Link:",
            placeholder="e.g., https://drive.google.com/file/d/... or https://drive.google.com/drive/folders/...",
            help="Copy and paste the Google Drive file or folder link containing the job description document.",
            key="drive_jd_input",
            label_visibility="collapsed"
        )
    with col_jd_dr_btn:
        if st.button("📥 Import Job from Drive", type="primary", use_container_width=True):
            if drive_jd_url and drive_jd_url.strip():
                with st.spinner("⏳ Connecting to Google Drive & downloading Job Description..."):
                    jd_files, err = GoogleDriveImporter.fetch_pdf_files_from_drive(drive_jd_url)
                    if err:
                        st.error(err)
                        st.session_state["drive_jd_file"] = None
                    else:
                        st.session_state["drive_jd_file"] = jd_files[0]
                        st.session_state["executed_config_sig"] = ""
                        st.success(f"✅ Successfully imported [{jd_files[0]['name']}] from Google Drive.")
                        st.rerun()
            else:
                st.warning("⚠️ Please enter a Google Drive link first.")

    if st.session_state.get("drive_jd_file"):
        jd_f = st.session_state["drive_jd_file"]
        if active_job is None:
            with st.spinner(f"🤖 AI ({provider_choice}) is validating the Job Description from Google Drive..."):
                try:
                    jd_text = DocumentParser.extract_text_from_pdf(jd_f["bytes"])
                    active_job = DocumentParser.parse_job_description(
                        jd_text,
                        api_key=effective_api_key,
                        provider=selected_provider,
                        model_name=effective_model
                    )
                    st.success(f"✅ Successfully extracted job criteria: **{active_job['title']}**")
                except EmptyPDFError as e:
                    st.error(f"❌ **Unreadable / Empty PDF File:** {str(e)}")
                    active_job = None
                except InvalidDocumentError as e:
                    st.error(f"❌ **Invalid Document:** {str(e)}")
                    active_job = None
                except Exception as e:
                    st.error(f"❌ **Failed to Process Document:** {str(e)}")
                    active_job = None

with tab_jd_text:
    col_jd_txt_title, col_jd_txt_reset = st.columns([3, 1], vertical_alignment="center")
    with col_jd_txt_title:
        st.markdown("**Type or Paste Job Description Text Directly:**")
    with col_jd_txt_reset:
        if st.button("🗑️ Clear Text", key="btn_reset_jd_text", use_container_width=True, help="Click to clear the job description text."):
            st.session_state["jd_text_area"] = ""
            st.session_state["executed_config_sig"] = ""
            st.rerun()

    jd_raw_text = st.text_area(
        "Job Description Text:",
        height=220,
        placeholder=(
            "Example:\n"
            "Position: Junior Architect\n"
            "Major: Architecture, Interior Design, or a related field\n"
            "Requirements: Minimum 2 years experience in design and build, AutoCAD, SketchUp, Revit, Technical Drawing...\n"
            "Responsibilities: Support project execution, 3D visualization, and site supervision..."
        ),
        key="jd_text_area",
        label_visibility="collapsed"
    )
    if jd_raw_text and len(jd_raw_text.strip()) >= 20:
        if active_job is None:
            with st.spinner(f"🤖 AI ({provider_choice}) is processing Job Description text..."):
                try:
                    active_job = DocumentParser.parse_job_description(
                        jd_raw_text.strip(),
                        api_key=effective_api_key,
                        provider=selected_provider,
                        model_name=effective_model
                    )
                    st.success(f"✅ Successfully extracted job criteria: **{active_job['title']}**")
                except InvalidDocumentError as e:
                    st.error(f"❌ **Incomplete Text Format:** {str(e)}")
                    st.warning("💡 **Tip:** Ensure text includes position title, requirements, or responsibilities.")
                    active_job = None
                except Exception as e:
                    st.error(f"❌ **Failed to Process Text:** {str(e)}")
                    active_job = None
    elif jd_raw_text:
        st.warning("⚠️ Text is too short. Please provide comprehensive job description details.")

# Display extracted/active Job Criteria
if active_job:
    with st.expander(f"📋 Identified Criteria Summary: **{active_job['title']}**", expanded=True):
        st.markdown(f"**Position:** {active_job.get('title', 'Professional Role')}")
        st.markdown(f"**Required Major / Discipline:** {active_job.get('major', active_job.get('department', 'All Related Disciplines'))}")
        st.markdown(f"**Min. Education:** {active_job['hard_requirements'].get('min_education', 'Bachelor Degree')}")
        st.markdown(f"**Min. Experience:** {active_job['hard_requirements'].get('min_experience_years', 1)} Years")
        
        t_skills = active_job.get('technical_skills', [])
        s_skills = active_job.get('soft_skills', [])
        if not t_skills and not s_skills:
            t_skills, s_skills = DocumentParser.classify_skills(active_job.get('key_skills', []))
            
        st.markdown(f"**Technical Skills:** {', '.join(t_skills) if t_skills else '-'}")
        st.markdown(f"**Soft Skills:** {', '.join(s_skills) if s_skills else '-'}")
        st.markdown(f"**Responsibilities:**\n\n{active_job.get('responsibilities', active_job.get('description', ''))}")

st.markdown("---")

# ==========================================
# STEP 2: CANDIDATE CV INGESTION & UPLOAD
# ==========================================
st.header("2️⃣ Candidate CV Ingestion & Upload")

# ==========================================
# ANTI-BIAS & PRIVACY PROTOCOL (BLIND-CV)
# ==========================================
active_masked_fields = []

with st.container(border=True):
    col_blind_title, col_blind_badge = st.columns([3, 2])
    with col_blind_title:
        enable_blind_cv = st.toggle(
            "Blind-CV Anonymization",
            value=True,
            help="Automatically masks sensitive Personally Identifiable Information (PII) before AI evaluation to ensure 100% merit-based scoring."
        )

    if enable_blind_cv:
        st.markdown("**Select Personally Identifiable Information (PII) to Mask:**")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            if st.checkbox("Full Name", value=True, key="chk_name"): active_masked_fields.append("full_name")
            if st.checkbox("Email Address", value=True, key="chk_email"): active_masked_fields.append("email")
        with col_c2:
            if st.checkbox("Gender", value=True, key="chk_gender"): active_masked_fields.append("gender")
            if st.checkbox("Age", value=True, key="chk_age"): active_masked_fields.append("age")
        with col_c3:
            if st.checkbox("Domicile / Address", value=True, key="chk_address"): active_masked_fields.append("address")
            if st.checkbox("Profile Photo", value=True, key="chk_photo"): active_masked_fields.append("photo_url")
        with col_c4:
            if st.checkbox("University / Institution", value=True, key="chk_univ"): active_masked_fields.append("university")
            if st.checkbox("Phone Number", value=True, key="chk_phone"): active_masked_fields.append("phone")

if "cv_uploader_key" not in st.session_state:
    st.session_state["cv_uploader_key"] = 0
if "parsed_cv_store" not in st.session_state:
    st.session_state["parsed_cv_store"] = {}
if "eval_results_store" not in st.session_state:
    st.session_state["eval_results_store"] = {}
if "drive_cv_files" not in st.session_state:
    st.session_state["drive_cv_files"] = []
if "prev_uploaded_cv_names" not in st.session_state:
    st.session_state["prev_uploaded_cv_names"] = []

tab_upload, tab_drive = st.tabs(["📄 PDF Upload", "📁 Import Google Drive"])

raw_cv_items = []

with tab_upload:
    col_up_title, col_clear_btn = st.columns([3, 1], vertical_alignment="center")
    with col_up_title:
        st.markdown("**Upload Candidate CV Documents (Multiple PDF):**")
    with col_clear_btn:
        if st.button("🗑️ Clear All CVs", use_container_width=True, help="Click to clear and reset all uploaded candidate CV files."):
            st.session_state["cv_uploader_key"] += 1
            st.session_state["parsed_cv_store"] = {}
            st.session_state["eval_results_store"] = {}
            st.session_state["executed_config_sig"] = ""
            st.session_state["prev_uploaded_cv_names"] = []
            st.rerun()

    uploaded_cv_files = st.file_uploader(
        "Select or drag & drop candidate CV PDF files:",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"cv_uploader_{st.session_state['cv_uploader_key']}",
        label_visibility="collapsed"
    )
    if uploaded_cv_files:
        current_up_names = [f.name for f in uploaded_cv_files]
        if current_up_names != st.session_state.get("prev_uploaded_cv_names"):
            st.session_state["prev_uploaded_cv_names"] = current_up_names
            st.session_state["executed_config_sig"] = ""
        for f in uploaded_cv_files:
            raw_cv_items.append({"name": f.name, "bytes": f.getvalue()})
    else:
        if st.session_state.get("prev_uploaded_cv_names"):
            st.session_state["prev_uploaded_cv_names"] = []
            st.session_state["executed_config_sig"] = ""

with tab_drive:
    col_dr_title, col_dr_reset = st.columns([3, 1], vertical_alignment="center")
    with col_dr_title:
        st.markdown("**Import Candidate CVs from Google Drive:**")
    with col_dr_reset:
        if st.button("🗑️ Reset Drive Files", key="btn_reset_cv_drive", use_container_width=True, help="Click to clear and reset all files imported from Google Drive."):
            st.session_state["drive_cv_files"] = []
            st.session_state["eval_results_store"] = {}
            st.session_state["executed_config_sig"] = ""
            st.rerun()

    st.caption("💡 Supports **Google Drive Folder** (multi-CV ingestion) or a **specific single PDF file link**. Ensure access is set to **'Anyone with the link can view'**.")
    
    col_dr_in, col_dr_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_dr_in:
        drive_folder_url = st.text_input(
            "Google Drive Folder / File URL:",
            placeholder="e.g., https://drive.google.com/drive/folders/... or https://drive.google.com/file/d/...",
            help="Copy and paste your Google Drive folder or file link here.",
            key="drive_folder_input",
            label_visibility="collapsed"
        )
    with col_dr_btn:
        if st.button("📥 Import from Drive", type="primary", use_container_width=True):
            if drive_folder_url and drive_folder_url.strip():
                with st.spinner("⏳ Connecting to Google Drive & downloading PDF documents..."):
                    files, err = GoogleDriveImporter.fetch_pdf_files_from_drive(drive_folder_url)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["drive_cv_files"] = files
                        st.session_state["executed_config_sig"] = ""
                        st.success(f"✅ Successfully imported {len(files)} PDF documents from Google Drive.")
                        st.rerun()
            else:
                st.warning("⚠️ Please enter a Google Drive link first.")

    if st.session_state.get("drive_cv_files"):
        for f in st.session_state["drive_cv_files"]:
            raw_cv_items.append({"name": f["name"], "bytes": f["bytes"]})

candidates_to_process = []

if raw_cv_items:
    invalid_cv_count = 0
    for item in raw_cv_items:
        fname = item["name"]
        file_bytes = item["bytes"]
        cv_cache_key = f"{fname}_{len(file_bytes)}_{effective_model}_{effective_api_key[:6] if effective_api_key else 'offline'}"
        
        # Check if CV has already been parsed in session memory
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
            st.warning(f"⚠️ **File Skipped [{fname}]:** Empty document or pure scanned image without digital OCR text.")
            invalid_cv_count += 1
        except InvalidDocumentError as e:
            st.warning(f"⚠️ **File Skipped [{fname}]:** {str(e)}")
            invalid_cv_count += 1
        except Exception as e:
            st.warning(f"⚠️ **Failed to Process [{fname}]:** {str(e)}")
            invalid_cv_count += 1

st.markdown("---")

# ==========================================
# STEP 3: MATCHING & DASHBOARD RESULTS
# ==========================================
st.header("3️⃣ AI Screening & Evaluation Results")

if "weights_reset_key" not in st.session_state:
    st.session_state["weights_reset_key"] = 0

if not active_job:
    st.info("📋 Please setup or upload a **Job Description** in **Step 1** first.")
elif not candidates_to_process:
    st.info("📤 Please upload or import **Candidate CVs** in **Step 2** first.")
else:
    with st.container(border=True):
        st.markdown(f"Ready to evaluate **{len(candidates_to_process)} candidate CVs** for **{active_job['title']}**.")
        
        st.markdown("**⚙️ Scoring Weights & Criteria Configuration:**")
        col_w0, col_w1, col_w2, col_w3 = st.columns(4)
        with col_w0:
            threshold_score = st.number_input(
                "📊 Threshold (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                key=f"score_threshold_input_{st.session_state['weights_reset_key']}",
                help="Minimum overall score percentage for a candidate to qualify for the shortlist."
            )
        with col_w1:
            w_skill = st.number_input(
                "🎯 Skill Match (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                key=f"weight_skill_{st.session_state['weights_reset_key']}",
                help="Weight percentage for candidate technical and soft skills match."
            )
        with col_w2:
            w_exp = st.number_input(
                "💼 Experience Depth (%)",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                key=f"weight_exp_{st.session_state['weights_reset_key']}",
                help="Weight percentage for work experience duration and relevant industry track record."
            )
        with col_w3:
            w_edu = st.number_input(
                "🎓 Education (%)",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
                key=f"weight_edu_{st.session_state['weights_reset_key']}",
                help="Weight percentage for formal degree and academic background."
            )

        total_weight = w_skill + w_exp + w_edu
        custom_weights = {
            "skill": float(w_skill),
            "experience": float(w_exp),
            "education": float(w_edu)
        }

        # Unique state fingerprint for current input configuration
        current_config_sig = f"{active_job.get('job_id', '')}_{len(candidates_to_process)}_{'_'.join(sorted([c.get('cv_id', '') for c in candidates_to_process]))}_{enable_blind_cv}_{'_'.join(sorted(active_masked_fields))}_{w_skill}_{w_exp}_{w_edu}_{threshold_score}_{effective_model}_{effective_api_key[:6] if effective_api_key else 'offline'}"

        col_warn, col_reset_btn, col_btn = st.columns([2, 1, 1], vertical_alignment="center")
        with col_warn:
            if total_weight != 100:
                st.warning(f"⚠️ Total scoring weight must equal 100% (Current Total: **{total_weight}%**).")
        with col_reset_btn:
            if st.button("🔄 Reset Weights", use_container_width=True, help="Reset scoring weights to default values (60% Threshold, 50% Skills, 30% Experience, 20% Education)."):
                st.session_state["weights_reset_key"] += 1
                st.rerun()
        with col_btn:
            is_disabled = (total_weight != 100)
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True, disabled=is_disabled):
                st.session_state["executed_config_sig"] = current_config_sig
                st.rerun()

    # The evaluation results and leaderboards ONLY execute and display if the user has explicitly clicked Start AI Analysis for the current configuration
    if st.session_state.get("executed_config_sig") == current_config_sig:
        matcher = CandidateMatcherEngine(
            api_key=effective_api_key,
            provider=selected_provider,
            model_name=effective_model
        )
        evaluated_results = []
        progress_bar = st.progress(0, text="Analyzing candidate compatibility...")

        for idx, raw_cv in enumerate(candidates_to_process):
            cand_name = raw_cv.get("personal_info", {}).get("full_name") or f"Candidate #{idx+1}"
            progress_bar.progress((idx + 1) / len(candidates_to_process), text=f"🤖 Evaluating {cand_name} ({idx+1}/{len(candidates_to_process)})...")
            cv_to_process = BlindCVAnonymizer.anonymize_cv(raw_cv, enabled_fields=active_masked_fields) if enable_blind_cv else raw_cv
            
            # Cache Key for scoring evaluation to prevent redundant API hits on UI clicks
            eval_cache_key = f"{raw_cv.get('cv_id')}_{active_job.get('job_id')}_{enable_blind_cv}_{'_'.join(sorted(active_masked_fields))}_{w_skill}_{w_exp}_{w_edu}_{threshold_score}_{effective_model}_{effective_api_key[:6] if effective_api_key else 'offline'}"
            
            if eval_cache_key in st.session_state["eval_results_store"]:
                eval_res = st.session_state["eval_results_store"][eval_cache_key]
            else:
                eval_res = matcher.evaluate_candidate(cv_to_process, active_job, weights=custom_weights, threshold=float(threshold_score))
                eval_res["raw_cv"] = raw_cv
                eval_res["anonymized_cv"] = cv_to_process
                st.session_state["eval_results_store"][eval_cache_key] = eval_res

            evaluated_results.append(eval_res)

        progress_bar.empty()
        evaluated_results.sort(key=lambda x: x["overall_score"], reverse=True)

        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard & Screening Results", "🛡️ Blind-CV Anonymization", "📊 Analytics & Distribution"])

        with tab1:
            st.subheader(f"Candidate Evaluation Results for: {active_job['title']}")
            
            filtered_list = [c for c in evaluated_results if c["overall_score"] >= threshold_score]
            
            m1, m2, m3 = st.columns(3)
            with m1:
                with st.container(border=True):
                    st.metric("📁 Total CVs Processed", len(evaluated_results))
            with m2:
                with st.container(border=True):
                    st.metric("🎯 Shortlisted Candidates", len(filtered_list))
            with m3:
                with st.container(border=True):
                    avg_score = round(sum(c['overall_score'] for c in evaluated_results) / len(evaluated_results), 1) if evaluated_results else 0
                    st.metric("📈 Average Match Score", f"{avg_score}%")

            st.markdown("### 📋 Ranked Candidates (Leaderboard)")

            for rank, item in enumerate(evaluated_results, start=1):
                raw_personal = item["raw_cv"].get("personal_info", {})
                real_name = raw_personal.get("full_name") or item["candidate_alias"]
                alias_label = f" ({item['candidate_alias']})" if enable_blind_cv else ""
                
                with st.container(border=True):
                    st.markdown(f"#### #{rank} **{real_name}**{alias_label} — Match Score: **{item['overall_score']}%**")
                    
                    col_a, col_b = st.columns(2)
                    status_val = item["status"]
                    if status_val == "Pass":
                        status_styled = '<span style="color:#16a34a; font-weight:bold; font-size:1.05rem;">Pass</span>'
                    elif status_val == "Considered":
                        status_styled = '<span style="color:#d97706; font-weight:bold; font-size:1.05rem;">Considered</span>'
                    else:
                        status_styled = '<span style="color:#dc2626; font-weight:bold; font-size:1.05rem;">Rejected</span>'
                    
                    col_a.markdown(f"📌 **Recommendation Status:** {status_styled}", unsafe_allow_html=True)
                    col_b.markdown(f"🎯 **Skill Compatibility:** `{item['score_breakdown']['skill_match']}%`")
                    
                    with st.expander("Review"):
                        active_profile = item["raw_cv"]
                        p_info = active_profile.get("personal_info", {})
                        
                        st.markdown("##### Candidate Information")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.markdown(f"- **Name:** {p_info.get('full_name', real_name)}")
                            st.markdown(f"- **Email:** {p_info.get('email', '-')}")
                            st.markdown(f"- **Phone:** {p_info.get('phone', '-')}")
                        with col_info2:
                            age_val = p_info.get("age", "-")
                            age_disp = f"{age_val} years old" if isinstance(age_val, (int, float)) else str(age_val)
                            st.markdown(f"- **Age:** {age_disp}")
                            st.markdown(f"- **Gender:** {p_info.get('gender', '-')}")
                            st.markdown(f"- **Domicile / Address:** {p_info.get('address', '-')}")

                        st.markdown("##### Education Background")
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
                            st.markdown("- *No formal education history specified.*")

                        st.markdown("##### Work Experience & Track Record")
                        exp_list = active_profile.get("work_experience", [])
                        if exp_list:
                            for exp in exp_list:
                                role = exp.get("role", "Role")
                                comp = exp.get("company", "Company")
                                period = exp.get("period", f"{exp.get('duration_years', 0)} Years")
                                st.markdown(f"- **{role}** at **{comp}** *({period})*")
                        else:
                            st.markdown("- *No specific work experience listed.*")

                        col_sk1, col_sk2 = st.columns(2)
                        with col_sk1:
                            st.markdown("##### Technical Skills")
                            tech_list = active_profile.get("technical_skills", [])
                            if tech_list:
                                for t in tech_list:
                                    st.markdown(f"- {t}")
                            else:
                                st.markdown("- *No specific technical skills listed.*")

                        with col_sk2:
                            st.markdown("##### Soft Skills")
                            soft_list = active_profile.get("soft_skills", [])
                            if soft_list:
                                for s in soft_list:
                                    st.markdown(f"- {s}")
                            else:
                                st.markdown("- *No specific soft skills listed.*")

                        certs = active_profile.get("certifications", [])
                        if certs:
                            st.markdown("##### Certifications & Achievements")
                            for c in certs:
                                st.markdown(f"- {c}")

                        st.markdown("---")
                        st.markdown("##### AI Fit Analysis (Explainable AI / XAI)")
                        col_pro, col_con = st.columns(2)
                        with col_pro:
                            st.markdown("**Profile Strengths (Pros):**")
                            for pro in item["justification"]["pros"]:
                                st.markdown(f"- {pro}")
                        with col_con:
                            st.markdown("**Areas for Consideration / Gaps (Cons):**")
                            for con in item["justification"]["cons"]:
                                st.markdown(f"- {con}")

                        rec_reason = item["justification"].get("recommendation_reason")
                        if rec_reason:
                            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                            st.markdown("**Executive Status Explanation (Decision Rationale):**")
                            if item["status"] == "Pass":
                                st.success(f"**Accepted (Pass):** {rec_reason}")
                            elif item["status"] == "Considered":
                                st.warning(f"**Under Consideration (Considered):** {rec_reason}")
                            else:
                                st.error(f"**Rejected:** {rec_reason}")

        with tab2:
            st.subheader("🛡️ Blind-CV Anonymization Audit")
            st.info("This feature allows you to granularly select personal identifiable information (PII) to be masked before data is evaluated, guaranteeing 100% merit-based assessment.")
            
            cv_options = [c["cv_id"] for c in evaluated_results]
            selected_audit_id = st.selectbox("Select CV to Audit & Compare:", cv_options)
            target_audit = next(c for c in evaluated_results if c["cv_id"] == selected_audit_id)
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.error("❌ Original Raw CV Data (Unmasked)")
                st.json(target_audit["raw_cv"])
            with col_right:
                st.success("✅ Blind-CV Data Evaluated by AI (Protected & Anonymized)")
                st.json(target_audit["anonymized_cv"])

        with tab3:
            st.subheader("📊 Candidate Score Distribution Analytics")
            df_plot = pd.DataFrame([
                {
                    "Candidate": c["candidate_alias"] if enable_blind_cv else c["raw_cv"]["personal_info"].get("full_name", c["cv_id"]),
                    "Overall Score (%)": c["overall_score"],
                    "Skill Match Score (%)": c["score_breakdown"]["skill_match"],
                    "Experience Score (%)": c["score_breakdown"]["experience_depth"]
                } for c in evaluated_results
            ])
            st.bar_chart(df_plot.set_index("Candidate"))

st.caption("Autonomous Candidate Screening Platform v1.6.0 | AI Specialist Technical Assessment")
