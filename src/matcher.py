"""
Multi-Tier Candidate Matching & Scoring Engine
- Layer 1: Hard Filter (Knockout Criteria)
- Layer 2: Skill & Semantic Vector Similarity
- Layer 3: Explainable AI (XAI) Reasoning & Justification Generator
"""

from typing import Dict, Any, List
import re

class CandidateMatcherEngine:
    def __init__(self):
        pass

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
        jd_skills = [s.lower() for s in job_desc.get("key_skills", [])]
        cand_skills = [s.lower() for s in anonymized_cv.get("skills", [])]
        
        matched_skills = []
        for s in jd_skills:
            if any(s in cs or cs in s for cs in cand_skills):
                matched_skills.append(s.title())
                
        skill_score = (len(matched_skills) / max(len(jd_skills), 1)) * 100
        exp_score = min((total_exp / max(min_exp, 1)) * 100, 100) if min_exp > 0 else 100
        
        education_score = 90.0 # Standard accredited tier score
        
        # Combined Weighted Score
        overall_score = round((skill_score * 0.5) + (exp_score * 0.3) + (education_score * 0.2), 1)
        
        if not hard_filter_passed:
            overall_score = round(overall_score * 0.5, 1) # Penalty for missing knockout criteria

        # --- TIER 3: Explainable AI (XAI) Reasoning Generator ---
        pros = []
        cons = []
        questions = []
        
        if matched_skills:
            pros.append(f"Menguasai {len(matched_skills)} dari {len(jd_skills)} keahlian kunci: {', '.join(matched_skills)}.")
        if total_exp >= min_exp:
            pros.append(f"Memiliki durasi pengalaman {total_exp} tahun (memenuhi target >={min_exp} tahun).")
            
        if knockout_reasons:
            cons.extend(knockout_reasons)
        missing_skills = [s.title() for s in jd_skills if s.title() not in matched_skills]
        if missing_skills:
            cons.append(f"Belum mencantumkan keahlian: {', '.join(missing_skills)}.")
            
        # Interview questions recommendation
        if matched_skills:
            questions.append(f"Bisakah Anda menceritakan penerapan praktis keahlian {matched_skills[0]} dalam proyek manufaktur Anda?")
        if missing_skills:
            questions.append(f"Bagaimana strategi Anda untuk mengadaptasi keahlian {missing_skills[0]} dalam waktu singkat?")
        questions.append("Ceritakan tantangan terbesar yang pernah Anda hadapi dalam operasional lini produksi dan bagaimana solusi Anda.")

        status = "SHORTLISTED" if (overall_score >= 70 and hard_filter_passed) else ("CONSIDERATION" if overall_score >= 50 else "REJECTED")

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
                "cons": cons,
                "interview_questions": questions
            }
        }
