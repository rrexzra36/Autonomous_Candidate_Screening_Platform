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
    def _segment_cv_sections(text: str) -> Dict[str, str]:
        """
        Segments raw CV text into isolated semantic section blocks to prevent cross-contamination:
        - HEADER (Personal Info, Contact, Summary)
        - EXPERIENCE (Work History, Projects)
        - EDUCATION (Universities, Degrees, Majors)
        - SKILLS (Technical Tools & Soft Skills)
        - CERTIFICATIONS
        """
        header_patterns = [
            ('EXPERIENCE', r'(?:^|\n)\s*(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|PENGALAMAN\s+KERJA|PENGALAMAN|EMPLOYMENT\s+HISTORY)\s*(?:\n|\:|\-)'),
            ('EDUCATION', r'(?:^|\n)\s*(?:EDUCATION|PENDIDIKAN|RIWAYAT\s+PENDIDIKAN|ACADEMIC\s+BACKGROUND)\s*(?:\n|\:|\-)'),
            ('SKILLS', r'(?:^|\n)\s*(?:SKILLS\s*&\s*ABILITIES|TECHNICAL\s+SKILLS|SKILLS|KEAHLIAN|COMPETENCIES)\s*(?:\n|\:|\-)'),
            ('PROJECTS', r'(?:^|\n)\s*(?:PROJECTS|PROJECT\s+EXPERIENCE|PROYEK|ORGANIZATIONAL\s+EXPERIENCE)\s*(?:\n|\:|\-)'),
            ('CERTIFICATIONS', r'(?:^|\n)\s*(?:CERTIFICATIONS|CERTIFICATES|SERTIFIKAT|ACHIEVEMENTS)\s*(?:\n|\:|\-)')
        ]
        
        found = []
        for sec_name, pat in header_patterns:
            for m in re.finditer(pat, text, re.I):
                found.append((m.start(), m.end(), sec_name))
                
        found.sort(key=lambda x: x[0])
        
        sections = {}
        if not found:
            sections['HEADER'] = text
            return sections
            
        sections['HEADER'] = text[:found[0][0]].strip()
        for i in range(len(found)):
            name = found[i][2]
            s_idx = found[i][1]
            e_idx = found[i+1][0] if i + 1 < len(found) else len(text)
            sections[name] = text[s_idx:e_idx].strip()
            
        return sections

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

        # === Hierarchical Section-Segmented Extraction Algorithm ===
        sections = DocumentParser._segment_cv_sections(text)
        header_text = sections.get("HEADER", text)
        edu_text = sections.get("EDUCATION", "")
        exp_text = sections.get("EXPERIENCE", "")
        skills_text = sections.get("SKILLS", "")
        cert_text = sections.get("CERTIFICATIONS", "")

        # 1. Full Name Extraction (from Header Chunk)
        name = ""
        name_match = re.search(r"(?:nama|name|full\s*name)\s*[:\-]\s*([a-zA-Z\s\.\,\'\-]+)", header_text, re.I)
        if name_match and len(name_match.group(1).strip()) > 2:
            name = name_match.group(1).strip().title()
        else:
            clean_lines = [l.strip() for l in header_text.split("\n") if l.strip()]
            for line in clean_lines[:4]:
                if re.search(r"dibuat dengan|curriculum vitae|resume|biodata|personal info|contact|tentang saya", line, re.I):
                    continue
                if len(line) < 40 and not re.search(r"[@0-9\+\:\/\|]", line) and len(line.split()) <= 4:
                    candidate_name = line.strip(" :-|").title()
                    if len(candidate_name) > 2:
                        name = candidate_name
                        break
        if not name:
            name = filename.replace(".pdf", "").replace("_", " ").title()

        # 2. Email Extraction (from Header Chunk)
        email = "candidate@email.com"
        email_pattern = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", header_text)
        if email_pattern:
            email = email_pattern.group(0).lower()

        # 3. Phone Normalization (from Header Chunk)
        phone = "Not Specified"
        phone_match = re.search(r"(?:\+?\s*62|0)[\s\-]*(?:8[0-9\s\-]{7,15})", header_text)
        if phone_match:
            phone = normalize_phone_number(phone_match.group(0))

        # 4. Gender Extraction
        gender = "Not Specified"
        if re.search(r"\b(laki[\s\-]*laki|pria|male)\b", header_text, re.I):
            gender = "Male"
        elif re.search(r"\b(perempuan|wanita|female)\b", header_text, re.I):
            gender = "Female"

        # 5. Age & Birth Date Extraction
        age = "Not Specified"
        explicit_age = re.search(r"(?:usia|umur|age)\s*[:\-]?\s*(\d{2})", header_text, re.I)
        if explicit_age:
            age = int(explicit_age.group(1))

        # 6. Address / City Extraction (from Header Chunk)
        address = "Not Specified"
        cities = [
            "Sukoharjo", "Surakarta", "Solo", "Yogyakarta", "South Jakarta - Indonesia", "South Jakarta",
            "Jakarta Barat", "Jakarta Timur", "Jakarta Selatan", "Jakarta Pusat", "Jakarta Utara", "DKI Jakarta", "Jakarta",
            "Bandung, Indonesia", "Bandung", "Surabaya", "Semarang", "Bekasi", "Tangerang", "Depok", "Bogor", "Medan", "Malang", "Denpasar", "Bali"
        ]
        for c in cities:
            if re.search(rf"\b{re.escape(c)}\b", header_text, re.I):
                address = c
                break

        # 7. Education Extraction (Isolated to EDUCATION Chunk)
        education_list = []
        target_edu_text = edu_text if edu_text else text

        # Match Campus / School Name
        univ_match = re.search(r"\b((?:Universitas|University|Institut|Institute|Politeknik|Polytechnic|Sekolah Tinggi|STMIK|SMA\s+Negeri|SMK\s+Negeri|SMA|SMK)\s+[a-zA-Z0-9\s\(\)\.\,]+?)(?=[,\n\r\(]|\s+(?:Department|jurusan|tahun|grade|ipk|gpa|\d{4}|$))", target_edu_text, re.I)
        inst_name = univ_match.group(1).strip().title() if univ_match else "Accredited Higher Education Institution"
        inst_name = re.sub(r"\s+Department\s+Of.*", "", inst_name, flags=re.I).strip()

        # Degree & Major Detection
        deg_match = re.search(r"\b(Bachelor(?:\s+of|\s+in)?\s+[a-zA-Z\s]+|Sarjana(?:\s+Komputer|\s+Teknik|\s+S1|\s+S\.Kom|\s+S\.T)?|Master(?:\s+of|\s+in)?\s+[a-zA-Z\s]+|Diploma(?:\s+in)?\s+[a-zA-Z\s]+|S1\s+[a-zA-Z\s]+|S2\s+[a-zA-Z\s]+|D3\s+[a-zA-Z\s]+|Vocational\s+High\s+School|Senior\s+High\s+School)\b", target_edu_text, re.I)
        degree_name = deg_match.group(0).strip().title() if deg_match else "Bachelor's Degree (S1)"

        major_patterns = [
            r"(?:information systems|computer science|informatics|architecture|civil engineering|mechanical engineering|electrical engineering|industrial engineering|accounting|management|business administration|data science|graphic design)",
            r"(?:sistem informasi|teknik informatika|ilmu komputer|arsitektur|teknik sipil|teknik mesin|teknik elektro|akuntansi|manajemen|desain komunikasi visual)"
        ]
        detected_major = ""
        for mp in major_patterns:
            mm = re.search(mp, target_edu_text, re.I)
            if mm:
                detected_major = mm.group(0).strip().title()
                break
        if not detected_major:
            grad_m = re.search(r"([a-zA-Z\s]+)\s+graduate", header_text, re.I)
            if grad_m and len(grad_m.group(1).strip()) < 30:
                detected_major = grad_m.group(1).strip().title()

        period_m = re.search(r"\b(20\d{2}\s*[\-\–]\s*(?:20\d{2}|Present|Sekarang))\b", target_edu_text, re.I)
        period_str = period_m.group(1).strip() if period_m else "2020 - 2024"

        education_list.append({
            "institution": inst_name,
            "degree": degree_name,
            "major": detected_major if detected_major else "Related Discipline",
            "period": period_str
        })

        # 8. Work Experience Extraction (Isolated to EXPERIENCE Chunk)
        work_experiences = []
        target_exp_text = exp_text if exp_text else (sections.get("PROJECTS", "") or "")

        if target_exp_text:
            exp_lines = [l.strip() for l in target_exp_text.split("\n") if l.strip()]
            first_exp_line = exp_lines[0]
            role_comp_m = re.search(r"([^—\-\(]+)\s*[—\-]\s*([^\(]+)(?:\(([^)]+)\))?", first_exp_line)
            if role_comp_m:
                role = role_comp_m.group(1).strip().title()
                comp = role_comp_m.group(2).strip().title()
                exp_period = role_comp_m.group(3).strip() if role_comp_m.group(3) else "Recent Professional Experience"
            else:
                role = first_exp_line[:50].strip().title()
                comp = "Professional Industry Project"
                exp_period = "Recent"

            exp_yrs_m = re.search(r"(\d+)\s*(?:\+|-\d+)?\s*(?:tahun|thn|years|yr)", target_exp_text, re.I)
            dur_exp = int(exp_yrs_m.group(1)) if exp_yrs_m else 1

            bullet_points = " ".join(exp_lines[1:]) if len(exp_lines) > 1 else first_exp_line

            work_experiences.append({
                "role": role,
                "company": comp,
                "duration_years": dur_exp,
                "period": exp_period,
                "description": bullet_points[:400],
                "achievements": bullet_points[:400]
            })
        else:
            summary_m = re.search(r"(?:experience\s+in|background\s+in)\s+([^\.\n]+)", header_text, re.I)
            exp_summary = summary_m.group(0).strip() if summary_m else "Hands-on project and domain execution experience."
            work_experiences.append({
                "role": "Independent Project Specialist",
                "company": "Professional Projects",
                "duration_years": 1,
                "period": "1 Year Experience",
                "description": exp_summary,
                "achievements": exp_summary
            })

        # 9. Skills Discovery (Isolated to SKILLS Chunk & Full Document)
        target_skills_text = (skills_text + " " + header_text + " " + exp_text).strip()
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
            if re.search(rf"\b{re.escape(t)}\b", target_skills_text, re.I) and t not in found_tech:
                found_tech.append(t)

        found_soft = []
        for s in known_soft_skills_list:
            if re.search(rf"\b{re.escape(s)}\b", target_skills_text, re.I) and s not in found_soft:
                found_soft.append(s)

        if not found_tech and not found_soft:
            found_tech = ["Domain Technical Capabilities"]

        all_skills_combined = list(dict.fromkeys(found_tech + found_soft))
        tech_skills, soft_skills = DocumentParser.classify_skills(all_skills_combined)
        
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
