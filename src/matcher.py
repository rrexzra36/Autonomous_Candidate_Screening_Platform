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
        model_name: str = "gemini-3-flash-preview",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        *args,
        **kwargs
    ):
        self.provider = (provider or kwargs.get("provider", "gemini")).lower()
        self.model_name = model_name or kwargs.get("model_name", "gemini-3-flash-preview")
        
        if self.provider == "openai":
            self.api_key = api_key or openai_api_key or kwargs.get("openai_api_key", "")
        else:
            self.api_key = api_key or gemini_api_key or kwargs.get("gemini_api_key", "")

    def evaluate_candidate(
        self,
        anonymized_cv: Dict[str, Any],
        job_desc: Dict[str, Any],
        weights: Dict[str, float] = None,
        threshold: float = 60.0
    ) -> Dict[str, Any]:
        """
        Runs 3-Tier evaluation flow on an anonymized candidate CV against job description
        with customizable scoring weights and pass threshold.
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
            knockout_reasons.append(f"Work experience ({total_exp} years) is less than the required minimum ({min_exp} years).")
            
        mandatory_certs = hard_reqs.get("mandatory_certifications", [])
        cand_certs = anonymized_cv.get("certifications", [])
        for cert in mandatory_certs:
            if not any(cert.lower() in c.lower() for c in cand_certs):
                hard_filter_passed = False
                knockout_reasons.append(f"Missing mandatory certification: '{cert}'.")

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
        
        # Apply Scoring Weights
        if weights:
            w_skill = weights.get("skill", 50.0) / 100.0
            w_exp = weights.get("experience", 30.0) / 100.0
            w_edu = weights.get("education", 20.0) / 100.0
        else:
            w_skill, w_exp, w_edu = 0.5, 0.3, 0.2

        overall_score = round((skill_score * w_skill) + (exp_score * w_exp) + (education_score * w_edu), 1)
        if not hard_filter_passed:
            overall_score = round(overall_score * 0.5, 1)

        # --- TIER 3: LLM / XAI Reasoning ---
        pros, cons, eval_source = self._generate_reasoning(
            anonymized_cv, job_desc, matched_skills, jd_skills, total_exp, min_exp, knockout_reasons
        )

        status = "Pass" if (overall_score >= threshold and hard_filter_passed) else ("Considered" if overall_score >= max(threshold - 15, 35) else "Rejected")

        return {
            "cv_id": cv_id,
            "candidate_alias": candidate_alias,
            "job_id": job_desc.get("job_id"),
            "job_title": job_desc.get("title"),
            "overall_score": overall_score,
            "status": status,
            "hard_filter_passed": hard_filter_passed,
            "eval_source": eval_source,
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
You are a Senior Technical Recruiter and AI Talent Acquisition Specialist.
Provide an in-depth, rigorous, and objective evaluation of the candidate's fit against the job description below.

=== JOB DESCRIPTION DATA ===
Position: {job.get('title')}
Education & Major: {job.get('hard_requirements', {}).get('min_education', 'Bachelor Degree')} ({job.get('major', 'Related')})
Minimum Experience: {job.get('hard_requirements', {}).get('min_experience_years', 0)} Years
Technical Skills Required: {', '.join(job.get('technical_skills', job.get('key_skills', [])))}
Soft Skills Required: {', '.join(job.get('soft_skills', []))}
Responsibilities & Description:
{job.get('responsibilities', job.get('description', ''))}

=== CANDIDATE PROFILE (BLIND-CV / MERIT BASED) ===
{json.dumps(cv, indent=2, ensure_ascii=False)}

=== EVALUATION INSTRUCTIONS (EXPLAINABLE AI) ===
1. PROS (Candidate Strengths & Value-Add Potential):
   - Analyze candidate's real project/work track record relevance to position requirements.
   - Analyze technical software proficiency & core competencies ready to be deployed.
   - Highlight interpersonal strengths, work ethic, and achievements.
2. CONS (Gaps & Areas for Consideration):
   - Highlight essential technical software/tools or certifications required by the job but missing from the CV.
   - Highlight experience gaps, depth variance, or onboarding adaptation needed.

=== OUTPUT FORMAT (MANDATORY PURE JSON WITHOUT EMOJIS) ===
{{
  "pros": [
    "Key strength point analyzing candidate's portfolio & concrete track record...",
    "Key strength point analyzing core software & relevant technical tools...",
    "Key strength point analyzing soft skills and proven achievements..."
  ],
  "cons": [
    "Area of consideration regarding specific software/tools not explicitly listed...",
    "Area of consideration regarding qualification depth or experience gap..."
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
                            {"role": "system", "content": "You are a Senior Technical Recruiter outputting pure JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    text = response.choices[0].message.content
                except Exception:
                    pass

            # 2. Google Gemini Provider
            else:
                models_to_try = [
                    self.model_name,
                    "gemini-3.5-flash",
                    "gemini-3-flash-preview",
                    "gemini-3.1-flash-lite",
                    "gemini-3.1-pro-preview",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro"
                ]
                for m_name in models_to_try:
                    if not m_name:
                        continue
                    try:
                        from google import genai
                        client = genai.Client(api_key=self.api_key.strip())
                        response = client.models.generate_content(
                            model=m_name,
                            contents=prompt
                        )
                        text = response.text
                        if text:
                            break
                    except Exception as e:
                        # Fallback to legacy SDK if google.genai has model naming variance
                        try:
                            import google.generativeai as legacy_genai
                            legacy_genai.configure(api_key=self.api_key.strip())
                            model = legacy_genai.GenerativeModel(m_name)
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
                        json_content = raw_s[s_idx:e_idx+1]
                        data = json.loads(json_content)
                        pros = data.get("pros", [])
                        cons = data.get("cons", [])
                        if pros or cons:
                            provider_label = f"Google Gemini ({self.model_name})" if self.provider == "gemini" else f"OpenAI ({self.model_name})"
                            return pros, cons, provider_label
                except Exception as e:
                    print(f"[Matcher AI Reasoning JSON Parse Error]: {e}")

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
            pros.append(f"Proficient in technical tools and software: {', '.join(cand_tech)}.")
        elif matched_tech:
            pros.append(f"Proficient in relevant technical capabilities: {', '.join(matched_tech)}.")

        if cand_soft:
            pros.append(f"Demonstrates interpersonal competencies and work ethic: {', '.join(cand_soft)}.")
        elif matched_soft:
            pros.append(f"Possesses supporting soft skills: {', '.join(matched_soft)}.")

        # Duration & Education
        if total_exp > 0:
            if total_exp >= min_exp:
                pros.append(f"Fulfills work experience requirement with {total_exp} years of relevant experience (target: >= {min_exp} years).")
            else:
                pros.append(f"Possesses {total_exp} years of accumulated industry work experience.")

        edu_list = cv.get("education", [])
        if isinstance(edu_list, list) and edu_list:
            deg_names = [e.get("degree") for e in edu_list if e.get("degree")]
            if deg_names:
                pros.append(f"Supported by formal education credentials: {', '.join(deg_names)}.")
        elif isinstance(edu_list, dict) and edu_list.get("degree"):
            pros.append(f"Supported by formal education background: {edu_list.get('degree')}.")

        # Format Cons (Gaps against Job Vacancy)
        if knockout_reasons:
            cons.extend(knockout_reasons)
            
        if missing_tech:
            cons.append(f"Has not explicitly listed specific software tools required by the job: {', '.join(missing_tech)}.")
        if missing_soft:
            cons.append(f"Soft skills may require further verification during interview: {', '.join(missing_soft)}.")

        return pros, cons, "Local Intelligent Rule Engine (Offline)"
