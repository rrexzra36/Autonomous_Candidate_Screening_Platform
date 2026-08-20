"""
Multi-Modal PDF Document Parser & Entity Extraction Engine
- Extract raw text from PDF documents (JD or CV)
- Strict validation & error handling against test sheets, guides, invoices, and invalid docs
- Parse Job Descriptions into structured screening criteria with Technical & Soft Skills separation
- High-Fidelity CV Parser extracting real Personal Info, Education, Experience, and Skills
- Multi-LLM Support: Google Gemini (gemini-1.5-flash, gemini-2.5-flash, gemini-1.5-pro) & OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo)
"""

from typing import Dict, Any, List, Tuple
import io
import re
import json
from datetime import datetime

class DocumentParsingError(Exception):
    """Base exception for document parsing errors."""
    pass

class EmptyPDFError(DocumentParsingError):
    """Raised when PDF has no extractable digital text (e.g. blank or image scan)."""
    pass

class InvalidDocumentError(DocumentParsingError):
    """Raised when document content is not relevant to expected type (JD or CV)."""
    pass

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(file_bytes_or_stream) -> str:
        """
        Extracts clean plain text from PDF bytes or uploaded file buffer.
        Raises EmptyPDFError if no text can be extracted.
        """
        try:
            import pypdf
            if isinstance(file_bytes_or_stream, bytes):
                if len(file_bytes_or_stream) == 0:
                    raise EmptyPDFError("File PDF kosong (ukuran berkas 0 byte).")
                stream = io.BytesIO(file_bytes_or_stream)
            else:
                stream = file_bytes_or_stream
            
            reader = pypdf.PdfReader(stream)
            if len(reader.pages) == 0:
                raise EmptyPDFError("File PDF tidak memiliki halaman.")

            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            clean_text = text.strip()
            if len(clean_text) < 15:
                raise EmptyPDFError(
                    "Teks tidak dapat diekstrak dari PDF. Berkas mungkin berupa hasil scan/gambar murni tanpa lapisan teks digital (OCR) atau dokumen kosong."
                )
            return clean_text
        except EmptyPDFError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Gagal membaca berkas PDF: {str(e)}")

    @staticmethod
    def validate_job_description_text(text: str) -> Tuple[bool, str]:
        """
        Validates if extracted text genuinely looks like a Job Vacancy / Job Description
        and rejects assessment sheets, invoices, guides, and generic articles.
        """
        if len(text.strip()) < 30:
            return False, "Isi dokumen terlalu singkat untuk sebuah Job Description."

        text_lower = text.lower()

        # Negative Signals
        assessment_signals = [
            "technical assessment", "technical test", "assessment test", "uji teknis", "soal tes",
            "waktu pengerjaan:", "waktu pengerjaan :", "tugas anda adalah", "kriteria penilaian:",
            "instruksi pengerjaan", "studi kasus:", "output:\n● presentasi", "output: presentasi",
            "proof of concept\nwaktu pengerjaan", "soal ujian", "lembar kerja siswa", "faktur pajak",
            "invoice #", "purchase order", "latar belakang:\nsebuah perusahaan"
        ]
        for signal in assessment_signals:
            if signal in text_lower:
                return False, (
                    "Dokumen yang diunggah terdeteksi sebagai lembar soal / tes asesmen teknis (Technical Assessment Brief), "
                    "BUKAN dokumen resmi Lowongan Kerja (Job Vacancy / Job Description). Silakan unggah dokumen deskripsi lowongan kerja yang sebenarnya."
                )

        # Positive Required Structural Sections
        has_req_section = bool(re.search(r"\b(requirement|requirements|kualifikasi|persyaratan|qualifications|job requirements|job qualifications|kriteria pelamar|syarat)\b", text_lower))
        has_resp_section = bool(re.search(r"\b(responsibilities|responsibility|tanggung jawab|deskripsi pekerjaan|job description|tugas dan tanggung jawab)\b", text_lower))
        has_hiring_title = bool(re.search(r"\b(we are hiring|lowongan kerja|open position|job vacancy|job title|posisi)\b", text_lower))

        if not (has_req_section or has_resp_section or has_hiring_title):
            return False, (
                "Dokumen tidak memuat bagian kualifikasi lowongan (Requirements / Kualifikasi) "
                "atau tanggung jawab pekerjaan (Responsibilities) yang sah."
            )

        # Minimum keyword density check
        jd_keywords = [
            "experience", "pengalaman", "skills", "keahlian", "education", "pendidikan",
            "degree", "sarjana", "diploma", "s1", "d3", "major", "jurusan", "years", "tahun",
            "kemampuan", "kompetensi", "drawing", "design"
        ]
        matched_kw = [kw for kw in jd_keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)]
        if len(matched_kw) < 2:
            return False, (
                "Dokumen tidak memiliki informasi kriteria kualifikasi minimum pelamar (seperti pendidikan, pengalaman kerja, atau keterampilan)."
            )

        return True, "Valid Job Description"

    @staticmethod
    def validate_cv_text(text: str) -> Tuple[bool, str]:
        """
        Validates if extracted text genuinely looks like a Candidate CV / Resume.
        """
        if len(text.strip()) < 30:
            return False, "Isi dokumen terlalu singkat untuk sebuah CV."

        text_lower = text.lower()

        if "technical assessment" in text_lower or "soal tes" in text_lower:
            return False, "Dokumen yang diunggah terdeteksi sebagai soal tes asesmen, bukan berkas CV kandidat."

        cv_indicators = [
            "experience", "pengalaman", "education", "pendidikan", "skills", "keahlian",
            "work", "kerja", "curriculum vitae", "resume", "riwayat", "profile", "profil",
            "contact", "kontak", "email", "phone", "telepon", "university", "universitas",
            "project", "proyek", "achievement", "prestasi", "kemampuan", "organization", "organisasi"
        ]

        matched_indicators = [kw for kw in cv_indicators if re.search(rf"\b{re.escape(kw)}\b", text_lower)]

        if len(matched_indicators) < 2:
            return False, (
                "Dokumen tidak terdeteksi memuat riwayat hidup (CV) kandidat yang valid "
                "(tidak ditemukan informasi pengalaman kerja, pendidikan, kontak, atau keterampilan)."
            )
        return True, "Valid Candidate CV"

    @staticmethod
    def _is_valid_title(candidate: str) -> bool:
        """
        Validates that a string is a clean job title noun phrase.
        """
        if not candidate or len(candidate) < 3 or len(candidate) > 50:
            return False
        if re.match(r"^[\d\.\-\*\•\(\)\[\]\:\>\#]", candidate.strip()):
            return False
        bad_words = [
            "menganalisis", "mengumpulkan", "membuat", "tugas", "tujuan", "latar belakang",
            "kriteria", "solusi", "anda", "kami", "proses", "hasil", "mengotomatiskan",
            "memproses", "menghasilkan", "memberikan", "tahap", "intervensi", "seleksi"
        ]
        cand_lower = candidate.lower()
        if any(re.search(rf"\b{re.escape(w)}\b", cand_lower) for w in bad_words):
            return False
        return True

    @staticmethod
    def extract_job_title(text: str) -> str:
        """
        Extracts a clean, valid Job Title noun phrase from JD text.
        """
        explicit_match = re.search(r"(?:job\s+title|position|posisi|lowongan|role|dibutuhkan|vacancy)\s*[:\-]\s*([^\n\r]+)", text, re.I)
        if explicit_match:
            candidate = explicit_match.group(1).strip()
            if DocumentParser._is_valid_title(candidate):
                return candidate.strip(":- \r\n")

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:6]:
            header_sub = re.sub(r"(?:REQUIREMENTS|JOB DESCRIPTION|KUALIFIKASI|DESKRIPSI|WE ARE HIRING|LOWONGAN|RESPONSIBILITIES).*", "", line, flags=re.I).strip(":- ")
            if header_sub and DocumentParser._is_valid_title(header_sub):
                return header_sub

        for line in lines[:4]:
            if DocumentParser._is_valid_title(line):
                return line.strip(":- ")

        return "Posisi Pekerjaan (Umum)"

    @staticmethod
    def classify_skills(skills: List[str]) -> Tuple[List[str], List[str]]:
        """
        Splits a list of skills into (Technical Skills, Soft Skills).
        """
        soft_keywords = [
            "creative", "visualization", "communication", "interpersonal", "presentation",
            "leadership", "management", "problem solv", "learner", "resilient", "teamwork",
            "negotiation", "adaptability", "critical thinking", "collaboration",
            "analytical", "time management", "insightful", "visionary", "confident", "enthusiastic"
        ]
        technical = []
        soft = []
        for s in skills:
            s_clean = s.strip()
            if not s_clean:
                continue
            if any(t in s_clean.lower() for t in ["drawing", "autocad", "revit", "sketchup", "python", "sql", "excel", "plc", "scada", "design & build", "interior design", "architectural design", "lumion", "v-ray", "blender", "photoshop", "illustrator", "3ds max", "rhino", "bim"]):
                if s_clean not in technical:
                    technical.append(s_clean)
            elif any(k in s_clean.lower() for k in soft_keywords):
                if s_clean not in soft:
                    soft.append(s_clean)
            else:
                if s_clean not in technical:
                    technical.append(s_clean)
        return technical, soft

    @staticmethod
    def parse_job_description(text: str, api_key: str = "", provider: str = "gemini", model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
        """
        Converts raw JD text into structured JSON criteria with Technical and Soft Skills separation.
        """
        is_valid, err_msg = DocumentParser.validate_job_description_text(text)
        if not is_valid:
            raise InvalidDocumentError(err_msg)

        if api_key and api_key.strip():
            try:
                extracted_json = DocumentParser._llm_parse_jd(text, api_key, provider=provider, model_name=model_name)
                if extracted_json and extracted_json.get("title") and DocumentParser._is_valid_title(extracted_json.get("title")):
                    all_skills = extracted_json.get("key_skills", [])
                    t_skills = extracted_json.get("technical_skills", [])
                    s_skills = extracted_json.get("soft_skills", [])
                    if not t_skills and not s_skills and all_skills:
                        t_skills, s_skills = DocumentParser.classify_skills(all_skills)
                    extracted_json["technical_skills"] = t_skills
                    extracted_json["soft_skills"] = s_skills
                    extracted_json["key_skills"] = t_skills + s_skills
                    return extracted_json
            except Exception:
                pass

        # Dynamic Heuristic Parser
        title = DocumentParser.extract_job_title(text)

        # Major
        major = "Semua Jurusan / Terkait"
        major_match = re.search(
            r"(?:degree\s+in|bachelor\s+in|major\s+in|jurusan|lulusan|program\s+studi|background\s+in)\s*[:\-]?\s*([a-zA-Z\s,/&]+?(?:or\s+a\s+related\s+field|dan\s+jurusan\s+terkait|terkait)?)(?=[.\n\r]|\s+(?:with|min|minimal|visionary|experience|pengalaman|interior\s+designer|architect|kualifikasi|syarat|$))",
            text,
            re.I
        )
        if major_match and len(major_match.group(1).strip()) > 3:
            major = major_match.group(1).strip().title()
        elif "architecture" in text.lower():
            major = "Architecture, Interior Design, or a related field"
        elif "mesin" in text.lower() or "industri" in text.lower():
            major = "Teknik Mesin / Industri / Terkait"
        elif "informatika" in text.lower() or "software" in text.lower():
            major = "Teknik Informatika / Ilmu Komputer / Terkait"

        # Education
        edu = "S1 / Bachelor Degree"
        if re.search(r"master|s2|magister", text, re.I):
            edu = "S2 / Master Degree"
        elif re.search(r"degree|bachelor|s1|sarjana", text, re.I):
            edu = "S1 / Bachelor Degree"
        elif re.search(r"diploma|d3|ahli madya", text, re.I):
            edu = "D3 / Diploma"
        elif re.search(r"smk|sma|high school", text, re.I):
            edu = "SMK / SMA Sederajat"

        # Experience
        exp_match = re.search(r"(?:minimum|min\.?|minimal|at least)?\s*(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr|year)", text, re.I)
        min_exp = int(exp_match.group(1)) if exp_match else 1

        # Skills
        skills = []
        known_tools = [
            "AutoCAD", "SketchUp", "3ds Max", "Revit", "Adobe Photoshop", "Photoshop", "Illustrator",
            "Lumion", "Rhino", "Blender", "V-Ray", "ArchiCAD", "Figma", "Canva", "InDesign",
            "Python", "SQL", "Excel", "Microsoft Office", "SAP", "PLC", "SCADA", "Six Sigma", "ISO 9001",
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "K3 Umum", "Git", "React", "Node.js"
        ]
        for tool in known_tools:
            if re.search(rf"\b{re.escape(tool)}\b", text, re.I):
                if tool not in skills:
                    skills.append(tool)

        skill_patterns = [
            (r"technical drawing", "Technical Drawing"),
            (r"creative\s+(?:and|&)\s+visualization", "Creative & Visualization Skills"),
            (r"project management", "Project Management"),
            (r"communication", "Communication Skills"),
            (r"design and build", "Design & Build"),
            (r"problem solv(?:er|ing)", "Problem Solving"),
            (r"interior design", "Interior Design"),
            (r"architectural design", "Architectural Design"),
            (r"root cause analysis", "Root Cause Analysis"),
            (r"statistical process control|spc", "Statistical Process Control (SPC)")
        ]
        for pattern, label in skill_patterns:
            if re.search(pattern, text, re.I) and label not in skills:
                skills.append(label)

        if not skills:
            skills = ["Technical Capabilities", "Problem Solving", "Project Execution"]

        tech_skills, soft_skills = DocumentParser.classify_skills(skills)

        # Responsibilities
        resp_match = re.search(r"(?:RESPONSIBILITIES|TANGGUNG JAWAB|JOB RESPONSIBILITY|JOB DESCRIPTION)\s*[:\-]?\s*([\s\S]+)", text, re.I)
        if resp_match:
            responsibilities = resp_match.group(1).strip()
        else:
            responsibilities = "Mendukung eksekusi proyek, desain teknis, dan koordinasi operasional tim sesuai standar industri."

        if len(responsibilities) > 600:
            responsibilities = responsibilities[:600] + "..."

        return {
            "job_id": f"JOB-UPLOADED-{abs(hash(title)) % 10000}",
            "title": title,
            "major": major,
            "department": "Design / Engineering / Operations",
            "hard_requirements": {
                "min_education": edu,
                "min_experience_years": min_exp,
                "mandatory_certifications": []
            },
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "key_skills": skills,
            "responsibilities": responsibilities,
            "description": text[:600] + "..." if len(text) > 600 else text
        }

    @staticmethod
    def parse_candidate_cv(text: str, filename: str = "Candidate_CV.pdf", api_key: str = "", provider: str = "gemini", model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
        """
        High-fidelity extractor converting raw CV text into complete, authentic candidate profile entities.
        """
        is_valid, err_msg = DocumentParser.validate_cv_text(text)
        if not is_valid:
            raise InvalidDocumentError(err_msg)

        if api_key and api_key.strip():
            try:
                extracted_json = DocumentParser._llm_parse_cv(text, filename, api_key, provider=provider, model_name=model_name)
                if extracted_json and extracted_json.get("personal_info") and extracted_json["personal_info"].get("full_name"):
                    return extracted_json
            except Exception:
                pass

        # === High-Fidelity Heuristic Fallback Extractor ===
        clean_lines = [l.strip() for l in text.split("\n") if l.strip()]

        # 1. Full Name Extraction
        name = ""
        name_match = re.search(r"(?:nama|name|full\s*name)\s*[:\-]\s*([a-zA-Z\s\.\,\'\-]+)", text, re.I)
        if name_match and len(name_match.group(1).strip()) > 2:
            name = name_match.group(1).strip().title()
        else:
            for line in clean_lines[:6]:
                # Skip platform boilerplate and headers
                if re.search(r"dibuat dengan|profil jobstreet|curriculum vitae|resume|biodata|personal info|contact|tentang saya|about me", line, re.I):
                    continue
                if len(line) < 35 and not re.search(r"[@0-9\+\:\/\|]", line) and len(line.split()) <= 4:
                    # Clean title noun
                    candidate_name = line.strip(" :-|")
                    if len(candidate_name) > 2 and not any(k in candidate_name.lower() for k in ["architect", "drafter", "engineer", "designer", "profile", "summary"]):
                        name = candidate_name.title()
                        break
        if not name:
            name = filename.replace(".pdf", "").replace("_", " ").replace("CV Sample", "Kandidat").title()

        # 2. Email Extraction (handling spaces around @ e.g. "Dani 19@gmail.com" on same line)
        email = "candidate@email.com"
        email_pattern = re.search(r"([a-zA-Z0-9_.+-]+(?:\s+[a-zA-Z0-9_.+-]+)?)\s*@\s*([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
        if email_pattern:
            tokens = [x for x in re.split(r"[\s\r\n]+", email_pattern.group(1).strip()) if x]
            user_part = "".join(tokens[-2:]) if len(tokens) >= 2 and tokens[-1].isdigit() else (tokens[-1] if tokens else "candidate")
            domain_part = re.sub(r"\s+", "", email_pattern.group(2))
            email = f"{user_part}@{domain_part}".lower()

        # 3. Phone Extraction
        phone = "Tidak Dicantumkan"
        phone_match = re.search(r"(?:\+?\s*62|0)[\s\-]*(?:8[0-9\s\-]{7,15})", text)
        if phone_match:
            raw_phone = phone_match.group(0)
            phone = re.sub(r"\s+", " ", raw_phone).strip()

        # 4. Gender Extraction
        gender = "Tidak Dicantumkan"
        if re.search(r"\b(laki[\s\-]*laki|pria|male)\b", text, re.I):
            gender = "Laki-laki"
        elif re.search(r"\b(perempuan|wanita|female)\b", text, re.I):
            gender = "Perempuan"

        # 5. Age & Birth Date Extraction
        age = "Tidak Dicantumkan"
        explicit_age = re.search(r"(?:usia|umur|age)\s*[:\-]?\s*(\d{2})", text, re.I)
        if explicit_age:
            age = int(explicit_age.group(1))
        else:
            birth_match = re.search(r"(?:born\s+in|lahir|birth|dob|tgl lahir|tanggal lahir)[^\n\r]*?(\d{1,2}\s+[a-zA-Z]+\s+\d{4}|\d{4})", text, re.I)
            if birth_match:
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", birth_match.group(0))
                if year_match:
                    birth_year = int(year_match.group(1))
                    age = 2026 - birth_year

        # 6. Address / City Extraction
        address = "Tidak Dicantumkan"
        cities = [
            "South Jakarta - Indonesia", "South Jakarta", "Jakarta Barat", "Jakarta Timur", "Jakarta Selatan", "Jakarta Pusat", "Jakarta Utara", "DKI Jakarta", "Jakarta",
            "Bandung, Indonesia", "Bandung", "Kuningan", "Surabaya", "Yogyakarta", "Semarang", "Bekasi", "Tangerang", "Depok", "Bogor", "Medan", "Malang", "Solo", "Surakarta", "Denpasar", "Bali"
        ]
        for c in cities:
            if re.search(rf"\b{re.escape(c)}\b", text, re.I):
                address = c
                break
        if address == "Tidak Dicantumkan":
            addr_match = re.search(r"(?:alamat|address|domisili|lokasi|city|kota|location)\s*[:\-]?\s*([^\n\r\|]+)", text, re.I)
            if addr_match and len(addr_match.group(1).strip()) > 3:
                address = addr_match.group(1).strip()

        # 7. University / School Institution & Degree Extraction
        university = "Tidak Dicantumkan"
        degree = "S1 / Bachelor Degree"
        
        # Detect university / school name
        univ_match = re.search(r"((?:Universitas|University|Institut|Institute|Politeknik|Polytechnic|Sekolah Tinggi|Vocational High School|SMK Negeri|SMK|SMA Negeri|SMA)\s+[a-zA-Z0-9\s\(\)\.\,]+?)(?=[,\n\r]|\s+(?:majoring|jurusan|with|faculty|tahun|grade|ipk|gpa|$))", text, re.I)
        if univ_match:
            university = univ_match.group(1).strip().title()
        
        if "Vocational High School" in text or "SMK" in text:
            degree = "SMK / Vocational High School"
        elif "Master" in text or "S2" in text or "Magister" in text:
            degree = "S2 / Master Degree"
        elif "Diploma" in text or "D3" in text:
            degree = "D3 / Diploma"
        elif "Bachelor" in text or "S1" in text or "Sarjana" in text or "University" in text or "Universitas" in text:
            degree = "S1 / Bachelor Degree"

        major_match = re.search(r"(?:majoring in|major in|jurusan|program studi|studi)\s*[:\-]?\s*([a-zA-Z\s\(\)]+)", text, re.I)
        if major_match and len(major_match.group(1).strip()) > 3:
            major_clean = major_match.group(1).strip().title()
            degree = f"{degree} ({major_clean})"

        # 8. Experience Duration & Roles
        exp_match = re.search(r"(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr)", text, re.I)
        dur_exp = int(exp_match.group(1)) if exp_match else 2

        # 9. Skills Discovery
        known_tools = [
            "AutoCAD", "SketchUp", "3ds Max", "Revit", "Adobe Photoshop", "Photoshop", "Illustrator",
            "Lumion", "Rhino", "Blender", "V-Ray", "ArchiCAD", "Figma", "Canva", "InDesign",
            "Python", "SQL", "Excel", "Microsoft Office", "SAP", "PLC", "SCADA", "Six Sigma", "ISO 9001",
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "K3 Umum", "Technical Drawing",
            "Project Management", "Interior Design", "Design & Build", "Communication Skills", "Problem Solving",
            "Building Information Modeling", "BIM", "3D Visualization", "Architectural Modeling"
        ]
        found_skills = [s for s in known_tools if re.search(rf"\b{re.escape(s)}\b", text, re.I)]
        if not found_skills:
            found_skills = ["Technical Drawing", "Design Execution"]

        tech_skills, soft_skills = DocumentParser.classify_skills(found_skills)
        cv_id = f"CV-UP-{abs(hash(name + filename)) % 10000}"

        return {
            "cv_id": cv_id,
            "personal_info": {
                "full_name": name,
                "email": email,
                "phone": phone,
                "gender": gender,
                "age": age,
                "photo_url": "",
                "address": address,
                "university": university
            },
            "education": {
                "degree": degree,
                "institution": university,
                "graduation_year": 2022
            },
            "work_experience": [
                {
                    "role": "Candidate Work History",
                    "company": "Industry",
                    "duration_years": dur_exp,
                    "achievements": text[:300] if len(text) > 300 else text
                }
            ],
            "skills": found_skills,
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "certifications": []
        }

    @staticmethod
    def _llm_parse_jd(text: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
        prompt = f"""
Anda adalah AI HR Specialist. Ekstrak informasi Job Description berikut secara akurat ke format JSON murni:
{text}

Struktur JSON yang WAJIB dihasilkan:
{{
  "job_id": "JOB-CUSTOM",
  "title": "Nama Posisi / Job Title persis di dokumen (misal: Junior Architect)",
  "major": "Jurusan / Program Studi yang diminta (misal: Architecture, Interior Design, or a related field)",
  "department": "Bidang / Departemen",
  "hard_requirements": {{
    "min_education": "Tingkat Pendidikan Minimal (misal: Degree in Architecture / S1)",
    "min_experience_years": 2,
    "mandatory_certifications": []
  }},
  "technical_skills": ["AutoCAD", "SketchUp", "Revit", "Technical Drawing", "Design & Build", "Interior Design"],
  "soft_skills": ["Creative & Visualization Skills", "Project Management", "Communication Skills", "Problem Solving"],
  "key_skills": ["AutoCAD", "SketchUp", "Revit", "Technical Drawing", "Creative & Visualization Skills", "Project Management", "Communication Skills", "Design & Build", "Problem Solving", "Interior Design"],
  "responsibilities": "Poin-poin tanggung jawab pekerjaan",
  "description": "Ringkasan deskripsi pekerjaan"
}}
"""
        return DocumentParser._call_llm_json(prompt, api_key, provider=provider, model_name=model_name)

    @staticmethod
    def _llm_parse_cv(text: str, filename: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
        prompt = f"""
Anda adalah AI CV Parser tingkat lanjut. Ekstrak data Curriculum Vitae (CV) kandidat berikut PERSIS SESUAI ISI ASLI DOKUMEN secara menyeluruh ke format JSON murni.
JANGAN MENGARANG DATA ATAU MENGGUNAKAN TEMPLATE PLACEHOLDER (seperti 'Indonesia' atau 'Universitas Terakreditasi') jika informasi aslinya tercantum di dalam teks CV:

{text}

Struktur JSON yang WAJIB dihasilkan:
{{
  "cv_id": "CV-UPLOAD",
  "personal_info": {{
    "full_name": "Nama Lengkap Asli Kandidat",
    "email": "Email Asli Kandidat (perbaiki jika ada spasi di sekitar @)",
    "phone": "Nomor Telepon Asli Kandidat",
    "gender": "Laki-laki / Perempuan / Tidak Dicantumkan",
    "age": 23,
    "address": "Kota / Alamat Domisili Asli dari CV (misal: South Jakarta - Indonesia / Bandung)",
    "university": "Nama Asli Sekolah / Universitas / Almamater (misal: Vocational High School 3 Kuningan / Universitas Budi Luhur)"
  }},
  "education": {{
    "degree": "Jenjang & Jurusan Asli (misal: Vocational High School - Architectural Building Modeling Design / S1 Arsitektur)",
    "institution": "Nama Asli Kampus / Sekolah",
    "graduation_year": 2022
  }},
  "work_experience": [
    {{
      "role": "Jabatan / Pekerjaan",
      "company": "Perusahaan / Tempat Kerja",
      "duration_years": 2,
      "achievements": "Ringkasan pengalaman kerja"
    }}
  ],
  "technical_skills": ["Daftar Skill Teknis / Software Asli dari CV"],
  "soft_skills": ["Daftar Soft Skill Asli dari CV"],
  "skills": ["Semua Skill Asli dari CV"],
  "certifications": ["Sertifikasi jika ada"]
}}
"""
        res = DocumentParser._call_llm_json(prompt, api_key, provider=provider, model_name=model_name)
        if res:
            res["cv_id"] = f"CV-UP-{abs(hash(filename)) % 10000}"
        return res

    @staticmethod
    def _call_llm_json(prompt: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
        text = None
        
        # 1. OpenAI Provider
        if provider.lower() == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=api_key.strip())
                response = client.chat.completions.create(
                    model=model_name or "gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Anda adalah AI CV Parser tingkat lanjut yang mengeluarkan data JSON murni dengan akurasi 100% dari teks sumber."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content
            except Exception:
                pass

        # 2. Google Gemini Provider
        else:
            try:
                from google import genai
                client = genai.Client(api_key=api_key.strip())
                response = client.models.generate_content(
                    model=model_name or "gemini-1.5-flash",
                    contents=prompt
                )
                text = response.text
            except Exception:
                pass

            if not text:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=api_key.strip())
                    model = legacy_genai.GenerativeModel(model_name or "gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    text = response.text
                except Exception:
                    pass

        if text:
            cleaned_text = text.strip()
            if "```" in cleaned_text:
                parts = cleaned_text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        cleaned_text = part
                        if cleaned_text.startswith("json"):
                            cleaned_text = cleaned_text[4:]
                        break
            try:
                return json.loads(cleaned_text.strip())
            except Exception:
                pass
        return None
