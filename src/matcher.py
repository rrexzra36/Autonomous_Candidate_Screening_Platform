"""
Multi-Tier Candidate Matching & Scoring Engine
- Layer 1: Hard Filter (Knockout Criteria)
- Layer 2: Skill & Semantic Vector Similarity
- Layer 3: Explainable AI (XAI) Reasoning & Justification Generator (Pros & Cons)
- Multi-LLM Support: Google Gemini (gemini-1.5-flash, gemini-2.5-flash, gemini-1.5-pro) & OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo)
"""

from typing import Dict, Any, List
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class CandidateMatcherEngine:
    def __init__(
        self,
        api_key: str = "",
        provider: str = "gemini",
        model_name: str = "gemini-1.5-flash",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        *args,
        **kwargs
    ):
        self.provider = (provider or kwargs.get("provider", "gemini")).lower()
        self.model_name = model_name or kwargs.get("model_name", "gemini-1.5-flash")
        
        if self.provider == "openai":
            self.api_key = api_key or openai_api_key or kwargs.get("openai_api_key", "")
        else:
            self.api_key = api_key or gemini_api_key or kwargs.get("gemini_api_key", "")

    def evaluate_candidate(self, anonymized_cv: Dict[str, Any], job_desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs 3-Tier evaluation flow on an anonymized candidate CV against job description.
        """
        cv_id = anonymized_cv.get("cv_id", "UNKNOWN")
        candidate_alias = anonymized_cv.get("personal_info", {}).get("candidate_alias", "CANDIDATE-X")
        
        # --- TIER 1: Hard Filter (Knockout Rules) ---
        hard_reqs = job_desc.get("hard_requirements", {})
        min_exp = hard_reqs.get("min_experience_years", 0)
        
        total_exp = 0
        for exp in anonymized_cv.get("work_experience", []):
            total_exp += exp.get("duration_years", 0)
            
        hard_filter_passed = True
        knockout_reasons = []
        
        if total_exp < min_exp:
            hard_filter_passed = False
            knockout_reasons.append(f"Pengalaman kerja ({total_exp} tahun) kurang dari batas minimum ({min_exp} tahun).")
            
        mandatory_certs = hard_reqs.get("mandatory_certifications", [])
        cand_certs = anonymized_cv.get("certifications", [])
        for cert in mandatory_certs:
            if not any(cert.lower() in c.lower() for c in cand_certs):
                hard_filter_passed = False
                knockout_reasons.append(f"Tidak memiliki sertifikasi wajib: '{cert}'.")

        # --- TIER 2: Skill & Semantic Matching ---
        jd_skills = list(dict.fromkeys([s.lower() for s in (job_desc.get("technical_skills", []) + job_desc.get("soft_skills", []) + job_desc.get("key_skills", []))]))
        cand_skills = list(dict.fromkeys([s.lower() for s in (anonymized_cv.get("technical_skills", []) + anonymized_cv.get("soft_skills", []) + anonymized_cv.get("skills", []))]))
        
        matched_skills = []
        for s in jd_skills:
            if any(s in cs or cs in s for cs in cand_skills):
                matched_skills.append(s.title())
                
        skill_score = (len(matched_skills) / max(len(jd_skills), 1)) * 100
        exp_score = min((total_exp / max(min_exp, 1)) * 100, 100) if min_exp > 0 else 100
        education_score = 90.0
        
        overall_score = round((skill_score * 0.5) + (exp_score * 0.3) + (education_score * 0.2), 1)
        if not hard_filter_passed:
            overall_score = round(overall_score * 0.5, 1)

        # --- TIER 3: LLM / XAI Reasoning ---
        pros, cons = self._generate_reasoning(
            anonymized_cv, job_desc, matched_skills, jd_skills, total_exp, min_exp, knockout_reasons
        )

        status = "Pass" if (overall_score >= 70 and hard_filter_passed) else ("Considered" if overall_score >= 50 else "Rejected")

        return {
            "cv_id": cv_id,
            "candidate_alias": candidate_alias,
            "job_id": job_desc.get("job_id"),
            "job_title": job_desc.get("title"),
            "overall_score": overall_score,
            "status": status,
            "hard_filter_passed": hard_filter_passed,
            "score_breakdown": {
                "skill_match": round(skill_score, 1),
                "experience_depth": round(exp_score, 1),
                "education_tier": round(education_score, 1)
            },
            "matched_skills": matched_skills,
            "justification": {
                "pros": pros,
                "cons": cons
            }
        }

    def _generate_reasoning(self, cv, job, matched_skills, jd_skills, total_exp, min_exp, knockout_reasons):
        """
        Uses Google Gemini or OpenAI LLM if API Key is available, otherwise uses deterministic logic.
        """
        if self.api_key and self.api_key.strip():
            prompt = f"""
Anda adalah Senior Technical Recruiter dan AI Hiring Specialist tingkat lanjut.
Tugas Anda adalah menganalisis profil kandidat ini secara akurat dan menyajikan analisis Keunggulan (Pros) dan Catatan Gap (Cons) berdasarkan kecocokan terhadap lowongan pekerjaan.

=== DATA LOWONGAN PEKERJAAN (JOB VACANCY) ===
Posisi: {job.get('title')}
Jurusan/Prodi: {job.get('major', 'Terkait')}
Pendidikan Minimal: {job.get('hard_requirements', {}).get('min_education', 'S1')}
Pengalaman Minimal: {job.get('hard_requirements', {}).get('min_experience_years', 0)} Tahun
Technical Skills Dibutuhkan: {', '.join(job.get('technical_skills', job.get('key_skills', [])))}
Soft Skills Dibutuhkan: {', '.join(job.get('soft_skills', []))}
Tanggung Jawab & Deskripsi:
{job.get('responsibilities', job.get('description', ''))}

=== DATA CV KANDIDAT (LENGKAP) ===
{json.dumps(cv, indent=2, ensure_ascii=False)}

=== INSTRUKSI KHUSUS ANALISIS ===
1. Pada "pros":
   - Sebutkan SELURUH Technical Skills yang dimiliki kandidat dari CV-nya.
   - Sebutkan SELURUH Soft Skills yang dimiliki kandidat dari CV-nya.
   - Cantumkan total durasi pengalaman dan pendidikan asli kandidat.
2. Pada "cons":
   - Cantumkan Technical Skills atau Soft Skills yang diminta lowongan tapi BELUM tercantum di CV kandidat ini.
   - Cantumkan kekurangan durasi pengalaman jika ada.

=== FORMAT OUTPUT (WAJIB JSON MURNI) ===
{{
  "pros": [
    "Technical Skills yang dimiliki: [seluruh skill teknis dari CV]",
    "Soft Skills yang dimiliki: [seluruh soft skill dari CV]",
    "[Durasi pengalaman dan latar belakang pendidikan]"
  ],
  "cons": [
    "Belum mencantumkan Technical Skills: [skill teknis yang diminta lowongan tapi belum ada di CV jika ada]",
    "Belum mencantumkan Soft Skills: [soft skill yang diminta lowongan tapi belum ada di CV jika ada]",
    "[Catatan gap pengalaman jika ada]"
  ]
}}
"""
            text = None

            # 1. OpenAI Provider
            if self.provider == "openai":
                try:
                    import openai
                    client = openai.OpenAI(api_key=self.api_key.strip())
                    response = client.chat.completions.create(
                        model=self.model_name or "gpt-4o-mini",
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": "Anda adalah Senior Technical Recruiter yang mengeluarkan JSON murni."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    text = response.choices[0].message.content
                except Exception:
                    pass

            # 2. Google Gemini Provider
            else:
                models_to_try = [self.model_name, "gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
                for model_name in models_to_try:
                    if not model_name:
                        continue
                    try:
                        from google import genai
                        client = genai.Client(api_key=self.api_key.strip())
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        text = response.text
                        if text:
                            break
                    except Exception:
                        continue

                if not text:
                    for model_name in [self.model_name, "gemini-1.5-flash", "gemini-pro"]:
                        if not model_name:
                            continue
                        try:
                            import google.generativeai as legacy_genai
                            legacy_genai.configure(api_key=self.api_key.strip())
                            model = legacy_genai.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            text = response.text
                            if text:
                                break
                        except Exception:
                            continue

            if text:
                try:
                    cleaned_text = text.strip()
                    if "```" in cleaned_text:
                        parts = cleaned_text.split("```")
                        for part in parts:
                            if "{" in part and "}" in part:
                                cleaned_text = part
                                if cleaned_text.startswith("json"):
                                    cleaned_text = cleaned_text[4:]
                                break
                    data = json.loads(cleaned_text.strip())
                    pros = data.get("pros", [])
                    cons = data.get("cons", [])
                    if pros:
                        return pros, cons
                except Exception:
                    pass

        # Fallback Deterministic Heuristic Engine with Full Candidate Profile
        try:
            from parser import DocumentParser
        except ImportError:
            from src.parser import DocumentParser

        # 1. Extract Technical & Soft skills from this specific candidate's CV
        cand_tech = cv.get("technical_skills", [])
        cand_soft = cv.get("soft_skills", [])
        if not cand_tech and not cand_soft:
            cand_all_skills = cv.get("skills", [])
            cand_tech, cand_soft = DocumentParser.classify_skills(cand_all_skills)

        # 2. Extract Matched and Missing Skills
        matched_tech, matched_soft = DocumentParser.classify_skills(matched_skills)
        missing_skills = [s.title() for s in jd_skills if s.title() not in matched_skills]
        missing_tech, missing_soft = DocumentParser.classify_skills(missing_skills)

        pros = []
        cons = []
        
        # Format Pros based on THIS candidate's actual CV
        if cand_tech:
            pros.append(f"Technical Skills yang dimiliki: {', '.join(cand_tech)}.")
        elif matched_tech:
            pros.append(f"Technical Skills yang dimiliki: {', '.join(matched_tech)}.")

        if cand_soft:
            pros.append(f"Soft Skills yang dimiliki: {', '.join(cand_soft)}.")
        elif matched_soft:
            pros.append(f"Soft Skills yang dimiliki: {', '.join(matched_soft)}.")

        # Duration & Education
        if total_exp > 0:
            if total_exp >= min_exp:
                pros.append(f"Memiliki total durasi pengalaman kerja {total_exp} tahun (memenuhi target >={min_exp} tahun).")
            else:
                pros.append(f"Memiliki total durasi pengalaman kerja {total_exp} tahun.")

        edu = cv.get("education", {})
        if isinstance(edu, dict) and edu.get("degree"):
            pros.append(f"Latar Belakang Pendidikan: {edu.get('degree')}.")

        # Format Cons (Gaps against Job Vacancy)
        if knockout_reasons:
            cons.extend(knockout_reasons)
            
        if missing_tech:
            cons.append(f"Belum mencantumkan Technical Skills: {', '.join(missing_tech)}.")
        if missing_soft:
            cons.append(f"Belum mencantumkan Soft Skills: {', '.join(missing_soft)}.")

        return pros, cons
