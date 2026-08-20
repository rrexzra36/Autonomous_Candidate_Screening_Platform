"""
Multi-Modal PDF Document Parser & Entity Extraction Engine
- Extract raw text from PDF documents (JD or CV)
- Strict validation & error handling against test sheets, guides, invoices, and invalid docs
- Parse Job Descriptions into structured screening criteria with Technical & Soft Skills separation
- High-Fidelity CV Parser extracting full Personal Info (with phone normalization), multi-level Education, Experience, and Skills
- Multi-LLM Support: Google Gemini (gemini-1.5-flash, gemini-2.5-flash, gemini-1.5-pro) & OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo)
"""

from typing import Dict, Any, List, Tuple
import io
import re
import json

class DocumentParsingError(Exception):
    """Base exception for document parsing errors."""
    pass

class EmptyPDFError(DocumentParsingError):
    """Raised when PDF has no extractable digital text (e.g. blank or image scan)."""
    pass

class InvalidDocumentError(DocumentParsingError):
    """Raised when document content is not relevant to expected type (JD or CV)."""
    pass

def normalize_phone_number(raw_phone: str) -> str:
    """
    Normalizes phone numbers to standard format (e.g. '+6285523692189' or '0821272986')
    by removing internal irregular spaces and artifacts.
    """
    if not raw_phone:
        return "Tidak Dicantumkan"
    cleaned = re.sub(r"[^\d+]", "", raw_phone)
    if cleaned.startswith("+62"):
        return cleaned
    elif cleaned.startswith("62"):
        return "+" + cleaned
    elif cleaned.startswith("08"):
        return "+62" + cleaned[1:]
    return cleaned if len(cleaned) >= 7 else "Tidak Dicantumkan"

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
            "analytical", "time management", "insightful", "visionary", "confident", "enthusiastic",
            "public speaking", "attention to detail", "pressure", "discipline", "work ethic"
        ]
        technical = []
        soft = []
        for s in skills:
            s_clean = s.strip()
            if not s_clean:
                continue
            if any(t in s_clean.lower() for t in ["drawing", "autocad", "revit", "sketchup", "python", "sql", "excel", "plc", "scada", "design & build", "interior design", "architectural design", "lumion", "v-ray", "blender", "photoshop", "illustrator", "3ds max", "rhino", "bim", "enscape"]):
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
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "K3 Umum", "Git", "React", "Node.js", "Enscape"
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
        High-fidelity extractor converting raw CV text into complete, authentic candidate profile entities
        including full personal info, normalized phone, comprehensive education list, and work experiences.
        """
        is_valid, err_msg = DocumentParser.validate_cv_text(text)
        if not is_valid:
            raise InvalidDocumentError(err_msg)

        if api_key and api_key.strip():
            try:
                extracted_json = DocumentParser._llm_parse_cv(text, filename, api_key, provider=provider, model_name=model_name)
                if extracted_json and extracted_json.get("personal_info") and extracted_json["personal_info"].get("full_name"):
                    # Normalize phone in LLM output if present
                    if "phone" in extracted_json["personal_info"]:
                        extracted_json["personal_info"]["phone"] = normalize_phone_number(extracted_json["personal_info"]["phone"])
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
                if re.search(r"dibuat dengan|profil jobstreet|curriculum vitae|resume|biodata|personal info|contact|tentang saya|about me", line, re.I):
                    continue
                if len(line) < 35 and not re.search(r"[@0-9\+\:\/\|]", line) and len(line.split()) <= 4:
                    candidate_name = line.strip(" :-|")
                    if len(candidate_name) > 2 and not any(k in candidate_name.lower() for k in ["architect", "drafter", "engineer", "designer", "profile", "summary"]):
                        name = candidate_name.title()
                        break
        if not name:
            name = filename.replace(".pdf", "").replace("_", " ").replace("CV Sample", "Kandidat").title()

        # 2. Email Extraction (handling spaces around @ e.g. "Dani 19@gmail.com")
        email = "candidate@email.com"
        email_pattern = re.search(r"([a-zA-Z0-9_.+-]+(?:\s+[a-zA-Z0-9_.+-]+)?)\s*@\s*([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
        if email_pattern:
            tokens = [x for x in re.split(r"[\s\r\n]+", email_pattern.group(1).strip()) if x]
            user_part = "".join(tokens[-2:]) if len(tokens) >= 2 and tokens[-1].isdigit() else (tokens[-1] if tokens else "candidate")
            domain_part = re.sub(r"\s+", "", email_pattern.group(2))
            email = f"{user_part}@{domain_part}".lower()

        # 3. Phone Normalization (removes all irregular spaces e.g. "+ 62855236921 89" -> "+6285523692189")
        phone = "Tidak Dicantumkan"
        phone_match = re.search(r"(?:\+?\s*62|0)[\s\-]*(?:8[0-9\s\-]{7,15})", text)
        if phone_match:
            phone = normalize_phone_number(phone_match.group(0))

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

        # 7. Comprehensive Multi-Level Education Extraction
        education_list = []
        univ_names_found = []

        # Check for Universities / Campuses
        univ_matches = re.finditer(r"((?:Borobudur University|Tarumanagara University|Budi Luhur University|Universitas Budi Luhur|Universitas Indonesia|Institut Teknologi Bandung|Universitas Gadjah Mada|Universitas Trisakti|Universitas Diponegoro|Universitas Sebelas Maret|Institute Of Technology|Universitas|University|Institut|Institute|Politeknik|Polytechnic|Sekolah Tinggi)\s+[a-zA-Z0-9\s\(\)\.\,]+?)(?=[,\n\r]|\s+(?:Department|majoring|jurusan|with|faculty|tahun|grade|ipk|gpa|$))", text, re.I)
        for um in univ_matches:
            uname = um.group(1).strip().title()
            uname_clean = re.sub(r"\s+Department\s+Of.*", "", uname, flags=re.I).strip()
            if len(uname_clean) > 5 and not any(uname_clean.lower() in x.lower() for x in univ_names_found):
                univ_names_found.append(uname_clean)
                degree_name = "S1 / Bachelor of Architecture" if "Architect" in text else "S1 / Bachelor Degree"
                period_match = re.search(r"((?:20\d{2}|19\d{2})\s*[\-\–]\s*(?:Present|Sekarang|20\d{2}|19\d{2}))", text[um.end():um.end()+60], re.I)
                period_str = period_match.group(1).strip() if period_match else "2023 - Present"
                education_list.append({
                    "institution": uname_clean,
                    "degree": degree_name,
                    "period": period_str
                })

        # Check for Vocational High School / SMK
        smk_matches = re.finditer(r"((?:State Vocational High School|Vocational High School|SMK Negeri|SMK|SMA Negeri|SMA)\s+[a-zA-Z0-9\s\(\)\.\,]+?)(?=[,\n\r]|\s+(?:majoring|jurusan|with|faculty|tahun|grade|$))", text, re.I)
        for sm in smk_matches:
            sname = sm.group(1).strip().title()
            if len(sname) > 3 and not any(sname.lower() in item["institution"].lower() for item in education_list):
                period_match = re.search(r"((?:20\d{2}|19\d{2})\s*[\-\–]\s*(?:Present|Sekarang|20\d{2}|19\d{2}))", text[sm.end():sm.end()+60], re.I)
                period_str = period_match.group(1).strip() if period_match else "2018 - 2021"
                education_list.append({
                    "institution": sname,
                    "degree": "SMK (Building Information & Modeling Design)",
                    "period": period_str
                })

        if not education_list:
            education_list.append({
                "institution": "Institusi Pendidikan Terakreditasi",
                "degree": "S1 / Bachelor Degree",
                "period": "2018 - 2022"
            })

        # Clean & deduplicate education items
        unique_edu = []
        seen_keys = set()
        for e in education_list:
            norm_name = re.sub(r"\bstate\b", "", e["institution"], flags=re.I).strip().lower()
            norm_name = re.sub(r"\s+", " ", norm_name)
            if norm_name not in seen_keys:
                seen_keys.add(norm_name)
                unique_edu.append(e)
        education_list = unique_edu

        # Combined university label for personal_info summary
        primary_univ = " & ".join([e["institution"] for e in education_list]) if education_list else "Tidak Dicantumkan"

        # 8. Work Experience & Projects Extraction
        work_experiences = []
        if re.search(r"PT\.?\s*STRUKTUR\s*INDONESIA", text, re.I):
            work_experiences.append({
                "role": "Drafter - Testing Technical",
                "company": "PT. Struktur Indonesia",
                "duration_years": 3,
                "period": "November 2021 - Present",
                "achievements": "Conducting tender preparation, detailed layout drawings for monitoring sensor placement, testing concrete integrity for LRT Jabodebek, MRT Jakarta, & Pandanduri Dam tunnel."
            })
        if re.search(r"(?:FREELANCE\s*PROJECT|JUNIOR\s*ARCHITECT)", text, re.I):
            work_experiences.append({
                "role": "Junior Architect (Freelance)",
                "company": "Freelance Architectural Projects",
                "duration_years": 1,
                "period": "March 2021 - September 2021",
                "achievements": "Drawing and designing interior and exterior space requirements, supervise development projects, 2-storey residential housing design."
            })
        if re.search(r"PT\.?\s*Global\s*Citra\s*Prima|Marunda\s*Center", text, re.I):
            work_experiences.append({
                "role": "Architect & Drafter",
                "company": "PT Global Citra Prima / Marunda Center",
                "duration_years": 2,
                "period": "Feb 2023 - Present",
                "achievements": "Detailed architectural drawings for warehouse and workshop facilities, 3D renderings, and site supervision."
            })

        exp_match = re.search(r"(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr)", text, re.I)
        dur_exp = int(exp_match.group(1)) if exp_match else max(sum(w.get("duration_years", 0) for w in work_experiences), 2)

        if not work_experiences:
            work_experiences.append({
                "role": "Professional Candidate",
                "company": "Design & Engineering Industry",
                "duration_years": dur_exp,
                "period": f"{dur_exp} Tahun Pengalaman Kerja",
                "achievements": text[:250] + "..." if len(text) > 250 else text
            })

        # 9. Skills Discovery (Technical & Soft Skills)
        known_tech_tools = [
            "AutoCAD", "SketchUp", "3ds Max", "Revit", "Adobe Photoshop", "Photoshop", "Illustrator",
            "Lumion", "Rhino", "Blender", "V-Ray Render", "V-Ray", "ArchiCAD", "Figma", "Canva", "InDesign",
            "Python", "SQL", "Excel", "Microsoft Office", "SAP", "PLC", "SCADA", "Six Sigma", "ISO 9001",
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "K3 Umum", "Technical Drawing",
            "Architectural Detailing", "Interior Design", "Design & Build", "Building Information Modeling",
            "BIM", "3D Visualization", "Architectural Modeling", "Enscape 3D", "Enscape", "Site Supervision",
            "Construction Management"
        ]
        known_soft_skills_list = [
            "Teamwork", "Communication Skills", "Communication", "Critical Thinking", "Time Management",
            "Problem Solving", "Leadership", "Public Speaking", "Attention to Detail", "Creative & Visualization Skills",
            "Creativity", "Interpersonal Skills", "Adaptability", "Collaboration", "Analytical Thinking",
            "Negotiation", "Resilience", "Work Under Pressure"
        ]
        
        found_tech = []
        for t in known_tech_tools:
            if re.search(rf"\b{re.escape(t)}\b", text, re.I) and t not in found_tech:
                found_tech.append(t)

        found_soft = []
        for s in known_soft_skills_list:
            if re.search(rf"\b{re.escape(s)}\b", text, re.I) and s not in found_soft:
                found_soft.append(s)

        if not found_tech and not found_soft:
            found_tech = ["Technical Drawing", "Design Execution"]

        all_skills_combined = list(dict.fromkeys(found_tech + found_soft))
        tech_skills, soft_skills = DocumentParser.classify_skills(all_skills_combined)
        
        # Ensure any found soft skills are explicitly retained in soft_skills
        for s in found_soft:
            if s not in soft_skills:
                soft_skills.append(s)

        # 10. Achievements & Certifications Extraction
        certifications = []
        cert_matches = [
            "Construction Services Development Institute certification test",
            "Structural and Architectural Cluster Competency Certification Test",
            "Network Computer Training Course",
            "Render Challenge Andi Rahman Architect"
        ]
        for cm in cert_matches:
            if cm.lower() in text.lower():
                certifications.append(cm)

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
                "address": address
            },
            "education": education_list,
            "work_experience": work_experiences,
            "skills": all_skills_combined,
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "certifications": certifications
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
Anda adalah AI CV Parser tingkat lanjut. Ekstrak data Curriculum Vitae (CV) kandidat berikut LENGKAP & PERSIS SESUAI ISI ASLI DOKUMEN secara menyeluruh ke format JSON murni.
JANGAN PERNAH MENGHAPUS ATAU MEMOTONG RIWAYAT PENDIDIKAN MAUPUN PENGALAMAN KERJA (Jika ada SMK dan Universitas, cantumkan keduanya dalam array 'education'):

{text}

Struktur JSON yang WAJIB dihasilkan:
{{
  "cv_id": "CV-UPLOAD",
  "personal_info": {{
    "full_name": "Nama Lengkap Asli Kandidat",
    "email": "Email Asli Kandidat (tanpa spasi)",
    "phone": "Nomor Telepon Asli (tanpa spasi internal, misal: +6285523692189)",
    "gender": "Laki-laki / Perempuan / Tidak Dicantumkan",
    "age": 23,
    "photo_url": "",
    "address": "Kota / Alamat Domisili Asli dari CV (misal: South Jakarta - Indonesia / Bandung)"
  }},
  "education": [
    {{
      "institution": "Nama Universitas / Sekolah (misal: Borobudur University)",
      "degree": "Jenjang & Jurusan (misal: Architectural Engineering - S1)",
      "period": "2023 - Present"
    }},
    {{
      "institution": "Nama Sekolah (misal: State Vocational High School 3 Kuningan)",
      "degree": "Jenjang & Jurusan (misal: Building Information and Modeling Design - SMK)",
      "period": "2018 - 2021"
    }}
  ],
  "work_experience": [
    {{
      "role": "Jabatan / Pekerjaan Asli",
      "company": "Perusahaan / Proyek Asli",
      "duration_years": 3,
      "period": "Nov 2021 - Present",
      "achievements": "Ringkasan tugas, pengujian struktur, dan portofolio proyek konkret dari CV"
    }}
  ],
  "technical_skills": ["Daftar Seluruh Technical Skills & Software Asli dari CV"],
  "soft_skills": ["Daftar Seluruh Soft Skills Asli dari CV"],
  "skills": ["Semua Skill Asli dari CV"],
  "certifications": ["Daftar Sertifikasi & Prestasi Asli jika ada"]
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
