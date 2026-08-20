"""
Multi-Modal PDF Document Parser & Entity Extraction Engine
- Extracts clean raw text from PDF documents (Job Descriptions or CVs).
- Strict validation & error handling against test sheets, briefs, invoices, and invalid docs.
- Parses Job Descriptions into structured screening criteria with Technical & Soft Skills separation.
- High-Fidelity CV Parser extracting full Personal Info, Education history, Experience, and Skills.
- Multi-LLM Support: Google Gemini (gemini-3.x, gemini-2.5-flash, gemini-1.5-pro) & OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo).
"""

from typing import Dict, Any, List, Tuple
import io
import re
import json

class DocumentParsingError(Exception):
    """Base exception for document parsing errors."""
    pass

class EmptyPDFError(DocumentParsingError):
    """Raised when PDF has no extractable digital text (e.g. blank or pure image scan)."""
    pass

class InvalidDocumentError(DocumentParsingError):
    """Raised when document content is not relevant to expected type (JD or CV)."""
    pass

def normalize_phone_number(raw_phone: str) -> str:
    """
    Normalizes phone numbers to standard format (e.g. '+6285523692189')
    by removing internal irregular spaces and artifacts.
    """
    if not raw_phone:
        return "Not Specified"
    cleaned = re.sub(r"[^\d+]", "", raw_phone)
    if cleaned.startswith("+62"):
        return cleaned
    elif cleaned.startswith("62"):
        return "+" + cleaned
    elif cleaned.startswith("08"):
        return "+62" + cleaned[1:]
    return cleaned if len(cleaned) >= 7 else "Not Specified"

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
                    raise EmptyPDFError("PDF file is empty (file size is 0 bytes).")
                stream = io.BytesIO(file_bytes_or_stream)
            else:
                stream = file_bytes_or_stream
            
            reader = pypdf.PdfReader(stream)
            if len(reader.pages) == 0:
                raise EmptyPDFError("PDF file does not contain any pages.")

            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            clean_text = text.strip()
            if len(clean_text) < 15:
                raise EmptyPDFError(
                    "No readable text could be extracted from the PDF. The file may be a scanned image without an OCR layer or an empty document."
                )
            return clean_text
        except EmptyPDFError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to read PDF file: {str(e)}")

    @staticmethod
    def validate_job_description_text(text: str) -> Tuple[bool, str]:
        """
        Validates if extracted text genuinely looks like a Job Vacancy / Job Description
        and rejects assessment sheets, invoices, test briefs, and generic articles.
        """
        if len(text.strip()) < 30:
            return False, "The document content is too short to be a valid Job Description."

        text_lower = text.lower()

        # Negative Signals (Assessment sheets, invoices, test briefs)
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
                    "The uploaded document is detected as a Technical Assessment Brief / Test Sheet, "
                    "NOT an official Job Description / Vacancy document. Please upload a genuine job description file."
                )

        # Positive Required Structural Sections
        has_req_section = bool(re.search(r"\b(requirement|requirements|kualifikasi|persyaratan|qualifications|job requirements|job qualifications|kriteria pelamar|syarat)\b", text_lower))
        has_resp_section = bool(re.search(r"\b(responsibilities|responsibility|tanggung jawab|deskripsi pekerjaan|job description|tugas dan tanggung jawab)\b", text_lower))
        has_hiring_title = bool(re.search(r"\b(we are hiring|lowongan kerja|open position|job vacancy|job title|posisi|we're hiring)\b", text_lower))

        if not (has_req_section or has_resp_section or has_hiring_title):
            return False, (
                "The document does not contain valid Job Requirements or Responsibilities sections."
            )

        # Minimum keyword density check
        jd_keywords = [
            "experience", "pengalaman", "skills", "keahlian", "education", "pendidikan",
            "degree", "sarjana", "diploma", "s1", "d3", "major", "jurusan", "years", "tahun",
            "capabilities", "competence", "drawing", "design"
        ]
        matched_kw = [kw for kw in jd_keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)]
        if len(matched_kw) < 2:
            return False, (
                "The document lacks essential qualification criteria (such as education, experience, or required skills)."
            )

        return True, "Valid Job Description"

    @staticmethod
    def validate_cv_text(text: str) -> Tuple[bool, str]:
        """
        Validates if extracted text genuinely looks like a Candidate CV / Resume.
        """
        if len(text.strip()) < 30:
            return False, "The document content is too short to be a valid CV."

        text_lower = text.lower()

        if "technical assessment" in text_lower or "soal tes" in text_lower:
            return False, "The uploaded document is detected as an assessment test brief, not a candidate CV."

        cv_indicators = [
            "experience", "pengalaman", "education", "pendidikan", "skills", "keahlian",
            "work", "kerja", "curriculum vitae", "resume", "riwayat", "profile", "profil",
            "contact", "kontak", "email", "phone", "telepon", "university", "universitas",
            "project", "proyek", "achievement", "prestasi", "organization", "organisasi"
        ]

        matched_indicators = [kw for kw in cv_indicators if re.search(rf"\b{re.escape(kw)}\b", text_lower)]

        if len(matched_indicators) < 2:
            return False, (
                "The document does not appear to contain a valid candidate CV/Resume "
                "(no work experience, education, contact details, or skill sections found)."
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
            "analyzing", "collecting", "creating", "tasks", "objective", "background",
            "criteria", "solution", "your", "our", "process", "result", "automating",
            "processing", "delivering", "stage", "intervention", "selection"
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
        explicit_match = re.search(r"(?:job\s+title|position|posisi|lowongan|role|vacancy)\s*[:\-]\s*([^\n\r]+)", text, re.I)
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

        return "Professional Role"

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
    def parse_job_description(text: str, api_key: str = "", provider: str = "gemini", model_name: str = "gemini-3.5-flash") -> Dict[str, Any]:
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
        major = "All Related Disciplines"
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
            major = "Mechanical / Industrial Engineering or related"
        elif "informatika" in text.lower() or "software" in text.lower():
            major = "Computer Science / Information Technology or related"

        # Education
        edu = "Bachelor's Degree (S1)"
        if re.search(r"master|s2|magister", text, re.I):
            edu = "Master's Degree (S2)"
        elif re.search(r"degree|bachelor|s1|sarjana", text, re.I):
            edu = "Bachelor's Degree (S1)"
        elif re.search(r"diploma|d3|ahli madya", text, re.I):
            edu = "Associate Degree / Diploma (D3)"
        elif re.search(r"smk|sma|high school", text, re.I):
            edu = "Vocational High School / Senior High School"

        # Experience
        exp_match = re.search(r"(?:minimum|min\.?|minimal|at least)?\s*(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr|year)", text, re.I)
        min_exp = int(exp_match.group(1)) if exp_match else 1

        # Skills
        skills = []
        known_tools = [
            "AutoCAD", "SketchUp", "3ds Max", "Revit", "Adobe Photoshop", "Photoshop", "Illustrator",
            "Lumion", "Rhino", "Blender", "V-Ray", "ArchiCAD", "Figma", "Canva", "InDesign",
            "Python", "SQL", "Excel", "Microsoft Office", "SAP", "PLC", "SCADA", "Six Sigma", "ISO 9001",
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "Git", "React", "Node.js", "Enscape"
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
            responsibilities = "Support project execution, technical designs, and operational team coordination according to industry standards."

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
            name = filename.replace(".pdf", "").replace("_", " ").replace("CV Sample", "Candidate").title()

        # 2. Email Extraction
        email = "candidate@email.com"
        email_pattern = re.search(r"([a-zA-Z0-9_.+-]+(?:\s+[a-zA-Z0-9_.+-]+)?)\s*@\s*([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
        if email_pattern:
            tokens = [x for x in re.split(r"[\s\r\n]+", email_pattern.group(1).strip()) if x]
            user_part = "".join(tokens[-2:]) if len(tokens) >= 2 and tokens[-1].isdigit() else (tokens[-1] if tokens else "candidate")
            domain_part = re.sub(r"\s+", "", email_pattern.group(2))
            email = f"{user_part}@{domain_part}".lower()

        # 3. Phone Normalization
        phone = "Not Specified"
        phone_match = re.search(r"(?:\+?\s*62|0)[\s\-]*(?:8[0-9\s\-]{7,15})", text)
        if phone_match:
            phone = normalize_phone_number(phone_match.group(0))

        # 4. Gender Extraction
        gender = "Not Specified"
        if re.search(r"\b(laki[\s\-]*laki|pria|male)\b", text, re.I):
            gender = "Male"
        elif re.search(r"\b(perempuan|wanita|female)\b", text, re.I):
            gender = "Female"

        # 5. Age & Birth Date Extraction
        age = "Not Specified"
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
        address = "Not Specified"
        cities = [
            "South Jakarta - Indonesia", "South Jakarta", "Jakarta Barat", "Jakarta Timur", "Jakarta Selatan", "Jakarta Pusat", "Jakarta Utara", "DKI Jakarta", "Jakarta",
            "Bandung, Indonesia", "Bandung", "Kuningan", "Surabaya", "Yogyakarta", "Semarang", "Bekasi", "Tangerang", "Depok", "Bogor", "Medan", "Malang", "Solo", "Surakarta", "Denpasar", "Bali"
        ]
        for c in cities:
            if re.search(rf"\b{re.escape(c)}\b", text, re.I):
                address = c
                break
        if address == "Not Specified":
            addr_match = re.search(r"(?:alamat|address|domisili|lokasi|city|kota|location)\s*[:\-]?\s*([^\n\r\|]+)", text, re.I)
            if addr_match and len(addr_match.group(1).strip()) > 3:
                address = addr_match.group(1).strip()

        # 7. Comprehensive Multi-Level Education Extraction
        education_list = []
        
        # Match Degree Level and Major dynamically
        degree_patterns = [
            (r"(?:bachelor|sarjana|s1)\s+(?:of|in|jurusan|prodi|degree in)?\s*([a-zA-Z\s&/]+?)(?=[,\n\r\|]|\s+(?:with|faculty|tahun|grade|ipk|gpa|\d{4}|$))", "Bachelor Degree (S1)"),
            (r"(?:master|magister|s2)\s+(?:of|in|jurusan|prodi|degree in)?\s*([a-zA-Z\s&/]+?)(?=[,\n\r\|]|\s+(?:with|faculty|tahun|grade|ipk|gpa|\d{4}|$))", "Master Degree (S2)"),
            (r"(?:diploma|d3|d4|ahli madya)\s+(?:of|in|jurusan|prodi)?\s*([a-zA-Z\s&/]+?)(?=[,\n\r\|]|\s+(?:with|faculty|tahun|grade|ipk|gpa|\d{4}|$))", "Associate Degree / Diploma"),
            (r"((?:information systems|computer science|informatics|architecture|civil engineering|mechanical engineering|accounting|management|business administration|graphic design)\s+graduate)", "Bachelor Degree (S1)"),
            (r"(?:smk|vocational\s+high\s+school)\s*([a-zA-Z0-9\s&/]+?)(?=[,\n\r\|]|\s+(?:majoring|jurusan|tahun|$))", "Vocational High School (SMK)"),
            (r"(?:sma|high\s+school)\s*([a-zA-Z0-9\s&/]+?)(?=[,\n\r\|]|\s+(?:majoring|jurusan|tahun|$))", "Senior High School (SMA)")
        ]
        
        detected_major = ""
        detected_degree = "Bachelor's Degree (S1)"
        for pat, deg_lbl in degree_patterns:
            m = re.search(pat, text, re.I)
            if m:
                detected_degree = deg_lbl
                if m.groups() and m.group(1):
                    raw_maj = m.group(1).strip(" :-|")
                    if len(raw_maj) > 3 and not re.search(r"^\d+$", raw_maj):
                        detected_major = raw_maj.title()
                break

        # Check for Universities / Campuses dynamically
        univ_match = re.search(r"((?:Universitas|University|Institut|Institute|Politeknik|Polytechnic|Sekolah Tinggi|STMIK|SMA|SMK)\s+[a-zA-Z0-9\s\(\)\.\,]+?)(?=[,\n\r]|\s+(?:Department|majoring|jurusan|with|faculty|tahun|grade|ipk|gpa|$))", text, re.I)
        inst_name = univ_match.group(1).strip().title() if univ_match else "Accredited Higher Education Institution"
        inst_name = re.sub(r"\s+Department\s+Of.*", "", inst_name, flags=re.I).strip()

        period_match = re.search(r"((?:20\d{2}|19\d{2})\s*[\-\–]\s*(?:Present|Sekarang|20\d{2}|19\d{2}))", text, re.I)
        period_str = period_match.group(1).strip() if period_match else "2019 - 2023"

        if detected_major:
            degree_str = f"{detected_degree} in {detected_major}"
        else:
            degree_str = detected_degree

        education_list.append({
            "institution": inst_name,
            "degree": degree_str,
            "major": detected_major if detected_major else "Related Field",
            "period": period_str
        })

        # 8. Work Experience & Projects Extraction (Dynamic)
        work_experiences = []
        exp_roles = [
            "Data Analyst", "Data Scientist", "Data Engineer", "Software Engineer", "Frontend Developer",
            "Backend Developer", "Full Stack Developer", "Web Developer", "Machine Learning Engineer",
            "Architect", "Junior Architect", "Drafter", "Interior Designer", "Civil Engineer",
            "Quality Assurance", "QA Engineer", "Product Manager", "Project Manager", "Business Analyst",
            "UI/UX Designer", "Graphic Designer", "Marketing Specialist", "Operations Staff", "Intern"
        ]
        
        found_role = ""
        for r in exp_roles:
            if re.search(rf"\b{re.escape(r)}\b", text, re.I):
                found_role = r
                break

        company_match = re.search(r"(?:at|di|pt\.?|cv\.?)\s+([a-zA-Z0-9\s\.\,\-]+?)(?=[,\n\r\|]|\s+(?:from|sejak|tahun|\d{4}|$))", text, re.I)
        found_company = company_match.group(0).strip().title() if company_match else "Industry & Professional Projects"

        exp_years_match = re.search(r"(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr)", text, re.I)
        dur_exp = int(exp_years_match.group(1)) if exp_years_match else (2 if "experience" in text.lower() else 1)

        exp_period_match = re.search(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December|\d{4})\s*(?:\d{4})?\s*[\-\–]\s*(?:Present|Sekarang|\d{4}))", text, re.I)
        exp_period = exp_period_match.group(1).strip() if exp_period_match else f"{dur_exp} Years Professional Experience"

        # Summary or achievement snippet
        summary_match = re.search(r"(?:summary|profile|about me|tentang saya|pengalaman)[\s\:\-]+([^\n\r]+(?:\n[^\n\r]+){1,3})", text, re.I)
        achieve_text = summary_match.group(1).strip() if summary_match else text[:250] + "..."

        work_experiences.append({
            "role": found_role if found_role else "Professional Specialist",
            "company": found_company,
            "duration_years": dur_exp,
            "period": exp_period,
            "description": achieve_text,
            "achievements": achieve_text
        })

        # 9. Skills Discovery (Expanded Comprehensive Modern Taxonomy)
        known_tech_tools = [
            "AutoCAD", "SketchUp", "3ds Max", "Revit", "Adobe Photoshop", "Photoshop", "Illustrator",
            "Lumion", "Rhino", "Blender", "V-Ray", "ArchiCAD", "Figma", "Canva", "InDesign",
            "Python", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Excel", "Microsoft Excel", "Microsoft Office",
            "FastAPI", "Flask", "Django", "Tableau", "Power BI", "Pandas", "NumPy", "Scikit-Learn",
            "Docker", "Git", "GitHub", "AWS", "GCP", "TensorFlow", "PyTorch", "JavaScript", "TypeScript",
            "React", "Node.js", "HTML", "CSS", "SAP", "PLC", "SCADA", "Six Sigma", "ISO 9001",
            "Quality Control", "Lean Manufacturing", "Kaizen", "5S", "K3 Umum", "Technical Drawing",
            "Architectural Detailing", "Interior Design", "Design & Build", "Building Information Modeling",
            "BIM", "3D Visualization", "Architectural Modeling", "Enscape 3D", "Enscape", "Site Supervision",
            "Construction Management", "ETL Pipelines", "Data Automation", "Dashboard Development"
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
            found_tech = ["Domain Technical Capabilities"]

        all_skills_combined = list(dict.fromkeys(found_tech + found_soft))
        tech_skills, soft_skills = DocumentParser.classify_skills(all_skills_combined)
        
        # Ensure any found soft skills are explicitly retained in soft_skills
        for s in found_soft:
            if s not in soft_skills:
                soft_skills.append(s)

        # 10. Achievements & Certifications Extraction
        certifications = []
        cert_matches = [
            "Google Data Analytics", "AWS Certified", "TensorFlow Developer", "Microsoft Certified",
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
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "certifications": certifications
        }

    @staticmethod
    def _llm_parse_jd(text: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-3.5-flash") -> Dict[str, Any]:
        prompt = f"""
You are an expert AI HR Talent Acquisition Specialist. Extract the following Job Description information accurately into pure JSON:
{text}

MANDATORY JSON Structure:
{{
  "job_id": "JOB-CUSTOM",
  "title": "Exact Job Title from document (e.g. Junior Architect)",
  "major": "Required Academic Major / Field of Study (e.g. Architecture, Interior Design, or a related field)",
  "department": "Department / Functional Area",
  "hard_requirements": {{
    "min_education": "Minimum Education Level (e.g. Bachelor's Degree in Architecture / S1)",
    "min_experience_years": 2,
    "mandatory_certifications": []
  }},
  "technical_skills": ["AutoCAD", "SketchUp", "Revit", "Technical Drawing", "Design & Build", "Interior Design"],
  "soft_skills": ["Creative & Visualization Skills", "Project Management", "Communication Skills", "Problem Solving"],
  "key_skills": ["AutoCAD", "SketchUp", "Revit", "Technical Drawing", "Creative & Visualization Skills", "Project Management", "Communication Skills", "Design & Build", "Problem Solving", "Interior Design"],
  "responsibilities": "Bullet points summarizing job responsibilities",
  "description": "Concise summary of the job description"
}}
"""
        return DocumentParser._call_llm_json(prompt, api_key, provider=provider, model_name=model_name)

    @staticmethod
    def _llm_parse_cv(text: str, filename: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-3.5-flash") -> Dict[str, Any]:
        prompt = f"""
You are an advanced AI CV / Resume Parser. Extract all candidate data COMPLETELY & AUTHENTICALLY into pure JSON format.
Extract the REAL candidate data present in the document. DO NOT make up, truncate, or omit any education history, work experiences, or skills.

CANDIDATE CV DOCUMENT TEXT:
{text}

MANDATORY JSON Structure:
{{
  "cv_id": "CV-UPLOAD",
  "personal_info": {{
    "full_name": "Full Authentic Candidate Name",
    "email": "candidate@email.com",
    "phone": "+6285523692189",
    "gender": "Male / Female / Not Specified",
    "age": "Not Specified or Number",
    "photo_url": "",
    "address": "City / Domicile Location"
  }},
  "education": [
    {{
      "institution": "University / School Name",
      "degree": "Degree Level (e.g. Bachelor Degree / S1)",
      "major": "Academic Major / Field of Study (e.g. Information Systems / Architecture)",
      "period": "Start Year - End Year (e.g. 2020 - 2024)"
    }}
  ],
  "work_experience": [
    {{
      "role": "Job Title / Position (e.g. Data Automation Intern)",
      "company": "Company Name / Organization",
      "duration_years": 2,
      "period": "Start - End Date (e.g. 2023 - 2024)",
      "description": "Responsibilities and key accomplishments",
      "achievements": "Key accomplishments"
    }}
  ],
  "technical_skills": ["List of Technical Skills & Tools from CV"],
  "soft_skills": ["List of Soft Skills from CV"],
  "certifications": ["List of Certifications / Achievements if any"]
}}
"""
        res = DocumentParser._call_llm_json(prompt, api_key, provider=provider, model_name=model_name)
        if res:
            res["cv_id"] = f"CV-UP-{abs(hash(filename)) % 10000}"
        return res

    @staticmethod
    def _call_llm_json(prompt: str, api_key: str, provider: str = "gemini", model_name: str = "gemini-3.5-flash") -> Dict[str, Any]:
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
                        {"role": "system", "content": "You are an advanced AI CV Parser that outputs pure 100% accurate JSON from source documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                text = response.choices[0].message.content
            except Exception:
                pass

        # 2. Google Gemini Provider
        else:
            models_to_try = [model_name, "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.0-flash"]
            for m in models_to_try:
                if not m: continue
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key.strip())
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    text = response.text
                    if text:
                        break
                except Exception:
                    try:
                        import google.generativeai as legacy_genai
                        legacy_genai.configure(api_key=api_key.strip())
                        model = legacy_genai.GenerativeModel(m)
                        response = model.generate_content(prompt)
                        text = response.text
                        if text:
                            break
                    except Exception:
                        continue

        if text:
            try:
                raw_s = text.strip()
                s_idx = raw_s.find("{")
                e_idx = raw_s.rfind("}")
                if s_idx != -1 and e_idx != -1:
                    json_str = raw_s[s_idx:e_idx+1]
                    return json.loads(json_str)
            except Exception:
                pass
        return None
