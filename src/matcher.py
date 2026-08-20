"""
Multi-Tier Candidate Matching & Scoring Engine
- Layer 1: Hard Filter (Knockout Criteria)
- Layer 2: Skill & Semantic Vector Embedding Similarity (Cosine Similarity on Dense Embeddings)
- Layer 3: Explainable AI (XAI) Reasoning & Justification Generator (Pros & Cons)
- Multi-LLM Support: Google Gemini (gemini-3-flash-preview, gemini-3.1-flash-lite, etc.) & OpenAI (gpt-4o-mini, gpt-4o)
- Embedding Models: Google text-embedding-004 / OpenAI text-embedding-3-small with Offline Cosine fallback
"""

from typing import Dict, Any, List, Optional
import json
import os
import math
import re
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two numeric vectors [0.0 - 1.0]."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))

def compute_fallback_sparse_vector(text: str, vocabulary: Dict[str, int]) -> List[float]:
    """Generates a term frequency vector against a shared vocabulary."""
    tokens = re.findall(r'\b[a-zA-Z0-9_+#.-]{2,}\b', text.lower())
    counts = Counter(tokens)
    vec = [counts.get(word, 0) for word in vocabulary]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else [0.0] * len(vocabulary)

def extract_domain_keywords(title: str, major: str, tech_skills: List[str]) -> set:
    """Extracts distinctive domain keywords for role matching."""
    text = f"{title} {major} {' '.join(tech_skills)}".lower()
    stopwords = {
        "and", "or", "the", "in", "of", "for", "with", "a", "an", "to", "at", "by", "on", "is",
        "junior", "senior", "lead", "staff", "intern", "officer", "specialist", "manager", "associate"
    }
    tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
    return {t for t in tokens if t not in stopwords}

def calculate_text_domain_overlap(text: str, domain_keywords: set) -> float:
    """Calculates domain overlap ratio [0.0 - 1.0]."""
    if not domain_keywords or not text:
        return 0.0
    text_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
    intersection = text_tokens.intersection(domain_keywords)
    return len(intersection) / max(len(domain_keywords), 1)

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

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates dense vector embeddings using Google Gemini (text-embedding-004) or OpenAI (text-embedding-3-small).
        Returns None if offline or if API is unreachable.
        """
        if not self.api_key or not self.api_key.strip() or not text or not text.strip():
            return None

        clean_text = text.strip()[:8000]

        # 1. OpenAI Embedding
        if self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key.strip())
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=clean_text
                )
                return response.data[0].embedding
            except Exception:
                return None

        # 2. Google Gemini Embedding
        else:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key.strip())
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=clean_text
                )
                if hasattr(response, "embedding") and hasattr(response.embedding, "values"):
                    return list(response.embedding.values)
                elif hasattr(response, "embeddings") and response.embeddings:
                    return list(response.embeddings[0].values)
            except Exception:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.api_key.strip())
                    res = legacy_genai.embed_content(
                        model="models/text-embedding-004",
                        content=clean_text
                    )
                    if "embedding" in res:
                        return list(res["embedding"])
                except Exception:
                    return None

        return None

    def evaluate_candidate(
        self,
        anonymized_cv: Dict[str, Any],
        job_desc: Dict[str, Any],
        weights: Dict[str, float] = None,
        threshold: float = 60.0
    ) -> Dict[str, Any]:
        """
        Runs 3-Tier evaluation flow on an anonymized candidate CV against job description
        with customizable scoring weights, semantic vector embeddings, and pass threshold.
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

        # --- TIER 2: Skill & Semantic Vector Embedding Matching with Domain Role Verification ---
        try:
            from parser import DocumentParser
        except ImportError:
            from src.parser import DocumentParser

        jd_tech_skills = job_desc.get("technical_skills", [])
        jd_soft_skills = job_desc.get("soft_skills", [])
        if not jd_tech_skills and not jd_soft_skills:
            jd_tech_skills, jd_soft_skills = DocumentParser.classify_skills(job_desc.get("key_skills", []))

        cand_tech_skills = anonymized_cv.get("technical_skills", [])
        cand_soft_skills = anonymized_cv.get("soft_skills", [])
        if not cand_tech_skills and not cand_soft_skills:
            cand_tech_skills, cand_soft_skills = DocumentParser.classify_skills(anonymized_cv.get("skills", []))

        domain_keywords = extract_domain_keywords(
            job_desc.get("title", ""),
            job_desc.get("major", ""),
            jd_tech_skills
        )

        # 1. Technical & Soft Skills Matching
        matched_tech = []
        for s in jd_tech_skills:
            if any(s.lower() in cs.lower() or cs.lower() in s.lower() for cs in cand_tech_skills):
                matched_tech.append(s.title())

        matched_soft = []
        for s in jd_soft_skills:
            if any(s.lower() in cs.lower() or cs.lower() in s.lower() for cs in cand_soft_skills):
                matched_soft.append(s.title())

        matched_skills = matched_tech + matched_soft
        tech_ratio = len(matched_tech) / max(len(jd_tech_skills), 1)
        soft_ratio = len(matched_soft) / max(len(jd_soft_skills), 1) if jd_soft_skills else 1.0

        # 2. Dense Semantic Vector Similarity Analysis
        jd_profile_text = f"{job_desc.get('title', '')}. Key Requirements: {', '.join(jd_tech_skills + jd_soft_skills)}. Responsibilities: {job_desc.get('responsibilities', '')} {job_desc.get('description', '')}"
        cv_profile_text = f"Technical Skills: {', '.join(cand_tech_skills)}. Soft Skills: {', '.join(cand_soft_skills)}. " + " ".join([f"{e.get('role', '')} at {e.get('company', '')}: {e.get('description', '')}" for e in anonymized_cv.get("work_experience", [])])

        jd_vec = self._compute_embedding(jd_profile_text)
        cv_vec = self._compute_embedding(cv_profile_text)

        if jd_vec and cv_vec:
            semantic_similarity = calculate_cosine_similarity(jd_vec, cv_vec) * 100.0
            semantic_score = min(100.0, max(0.0, (semantic_similarity - 35.0) / 0.55))
        else:
            vocab = {word: idx for idx, word in enumerate(set(re.findall(r'\b[a-zA-Z0-9_+#.-]{2,}\b', (jd_profile_text + " " + cv_profile_text).lower())))}
            v_jd = compute_fallback_sparse_vector(jd_profile_text, vocab)
            v_cv = compute_fallback_sparse_vector(cv_profile_text, vocab)
            sparse_sim = calculate_cosine_similarity(v_jd, v_cv) * 100.0
            semantic_score = min(100.0, max(0.0, sparse_sim * 1.5))

        # 3. Composite Skill Score (Technical Skills carry primary 75-80% weight)
        if len(jd_tech_skills) > 0:
            if len(matched_tech) == 0:
                # ZERO core technical skills matched! Soft skills alone cannot qualify for a technical role
                skill_score = round(min(15.0, (soft_ratio * 10.0) + (semantic_score * 0.05)), 1)
            else:
                skill_score = round((tech_ratio * 75.0) + (soft_ratio * 15.0) + (min(100.0, semantic_score) * 0.10), 1)
        else:
            skill_score = round((soft_ratio * 60.0) + (min(100.0, semantic_score) * 0.40), 1)

        # 4. Work Experience Domain Relevance (Not Just Raw Duration)
        relevant_exp_years = 0.0
        for exp in anonymized_cv.get("work_experience", []):
            role_text = f"{exp.get('role', '')} {exp.get('description', '')}"
            overlap = calculate_text_domain_overlap(role_text, domain_keywords)
            has_tech = any(s.lower() in role_text.lower() for s in jd_tech_skills)
            
            if overlap >= 0.12 or has_tech:
                relevance = 1.0
            elif overlap > 0.04:
                relevance = 0.5
            else:
                relevance = 0.0
                
            relevant_exp_years += exp.get("duration_years", 0) * relevance

        if min_exp > 0:
            if relevant_exp_years > 0:
                exp_score = min((relevant_exp_years / min_exp) * 100.0, 100.0)
            else:
                # 0 relevant experience: nominal transferable points only
                exp_score = min(10.0, (total_exp / min_exp) * 10.0)
        else:
            exp_score = 100.0 if relevant_exp_years > 0 else 30.0

        # 5. Education Level & Major / Discipline Relevance
        edu_list = anonymized_cv.get("education", [])
        cand_degree_text = " ".join([e.get("degree", "") for e in edu_list if isinstance(e, dict)]).lower()
        cand_major_text = " ".join([e.get("major", "") for e in edu_list if isinstance(e, dict)]).lower()

        # Degree Level Assessment
        req_deg = hard_reqs.get("min_education", "Bachelor Degree").lower()
        if "master" in req_deg or "s2" in req_deg:
            deg_level_score = 100.0 if any(d in cand_degree_text for d in ["master", "s2", "phd", "magister"]) else 50.0
        elif "bachelor" in req_deg or "s1" in req_deg or "sarjana" in req_deg:
            deg_level_score = 90.0 if any(d in cand_degree_text for d in ["bachelor", "s1", "sarjana", "master", "s2", "phd"]) else 45.0
        else:
            deg_level_score = 80.0

        # Major Relevance Assessment
        major_overlap = calculate_text_domain_overlap(cand_major_text, domain_keywords)
        jd_major_keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', job_desc.get("major", "").lower()))
        direct_major_match = bool(set(re.findall(r'\b[a-zA-Z]{3,}\b', cand_major_text)).intersection(jd_major_keywords))

        if direct_major_match or major_overlap >= 0.20:
            major_score = 95.0
        elif major_overlap > 0.05 or ("engineering" in cand_major_text and "engineering" in job_desc.get("major", "").lower()):
            major_score = 65.0
        elif any(w in cand_major_text for w in ["building", "bim", "design", "architecture", "interior"]):
            major_score = 55.0
        else:
            major_score = 20.0  # Unrelated major

        education_score = round((deg_level_score * 0.4) + (major_score * 0.6), 1)

        # 6. Apply Scoring Weights
        if weights:
            w_skill = weights.get("skill", 50.0) / 100.0
            w_exp = weights.get("experience", 30.0) / 100.0
            w_edu = weights.get("education", 20.0) / 100.0
        else:
            w_skill, w_exp, w_edu = 0.5, 0.3, 0.2

        raw_overall = (skill_score * w_skill) + (exp_score * w_exp) + (education_score * w_edu)

        # Critical Domain Mismatch Filter
        is_domain_mismatch = (len(matched_tech) == 0 and relevant_exp_years == 0)
        if is_domain_mismatch and major_score <= 50.0:
            overall_score = round(min(22.0, raw_overall), 1)
        elif is_domain_mismatch:
            overall_score = round(min(28.0, raw_overall), 1)
        else:
            overall_score = round(raw_overall, 1)

        if not hard_filter_passed:
            overall_score = round(overall_score * 0.5, 1)

        # 7. Final Recommendation Status
        status = "Pass" if (overall_score >= threshold and hard_filter_passed) else ("Considered" if overall_score >= max(threshold - 15, 45) else "Rejected")

        missing_tech = [s.title() for s in jd_tech_skills if s.title() not in matched_tech]
        missing_soft = [s.title() for s in jd_soft_skills if s.title() not in matched_soft]

        pros, cons, rec_reason, eval_source = self._generate_reasoning(
            anonymized_cv, job_desc, matched_skills, jd_tech_skills + jd_soft_skills, total_exp, min_exp, knockout_reasons,
            overall_score, threshold, hard_filter_passed, status, is_domain_mismatch, relevant_exp_years, missing_tech, missing_soft
        )

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
                "semantic_similarity": round(semantic_score, 1),
                "experience_depth": round(exp_score, 1),
                "education_tier": round(education_score, 1)
            },
            "matched_skills": matched_skills,
            "justification": {
                "pros": pros,
                "cons": cons,
                "recommendation_reason": rec_reason
            }
        }

    def _generate_reasoning(
        self,
        cv,
        job,
        matched_skills,
        jd_skills,
        total_exp,
        min_exp,
        knockout_reasons,
        overall_score=0,
        threshold=60,
        hard_filter_passed=True,
        status="Considered",
        is_domain_mismatch=False,
        relevant_exp_years=0.0,
        missing_tech=None,
        missing_soft=None
    ):
        """
        Uses Google Gemini or OpenAI LLM if API Key is available, otherwise uses deterministic logic.
        """
        missing_tech = missing_tech or []
        missing_soft = missing_soft or []

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

=== EVALUATION METRICS ===
Match Score: {overall_score}%
Passing Threshold: {threshold}%
Decision Status: {status}
Domain Relevance Status: {"CRITICAL DOMAIN MISMATCH (0 Relevant Tech Skills, 0 Years Relevant Experience)" if is_domain_mismatch else "Domain Relevant"}

=== EVALUATION INSTRUCTIONS (EXPLAINABLE AI) ===
1. PROS (Candidate Strengths & Value-Add Potential):
   - If Domain Mismatch is active: Do NOT describe unrelated tools (e.g. Python/SQL for Architecture role) as relevant job strengths. Instead describe general transferable soft skills or formal education level.
   - If candidate is aligned: Analyze candidate's real project/work track record relevance and technical software proficiency ready to deploy.
2. CONS (Gaps & Areas for Consideration):
   - If Domain Mismatch is active: Explicitly identify the major career domain and technical skill gap as the primary reason for non-qualification.
   - Highlight missing required software/tools: {', '.join(missing_tech) if missing_tech else 'None'}.
3. RECOMMENDATION REASON:
   - Provide a direct, professional 1-2 sentence executive explanation of WHY this candidate is {status} based on {overall_score}% match score against the {threshold}% threshold, core skill alignment, and domain experience match.

=== OUTPUT FORMAT (MANDATORY PURE JSON WITHOUT EMOJIS) ===
{{
  "pros": [
    "Key strength point analyzing candidate's actual relevant capabilities or transferable work ethic...",
    "Key strength point analyzing supporting soft skills or credentials..."
  ],
  "cons": [
    "Critical gap analyzing missing required technical tools ({', '.join(missing_tech[:3]) if missing_tech else 'None'})...",
    "Area of consideration regarding domain experience variance or qualification depth..."
  ],
  "recommendation_reason": "Executive rationale explaining why the candidate was {status}..."
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
                        rec_reason = data.get("recommendation_reason", "")
                        if pros or cons:
                            provider_label = f"Google Gemini ({self.model_name})" if self.provider == "gemini" else f"OpenAI ({self.model_name})"
                            if not rec_reason:
                                if status == "Pass":
                                    rec_reason = f"Candidate exceeds the qualification threshold ({overall_score}% vs {threshold}% min) with proven technical proficiency and relevant experience."
                                elif status == "Considered":
                                    rec_reason = f"Candidate achieves a match score of {overall_score}%, showing potential but presenting minor skill or depth gaps relative to the {threshold}% threshold."
                                else:
                                    rec_reason = f"Candidate does not meet the minimum qualification threshold ({overall_score}% vs {threshold}% min)."
                            return pros, cons, rec_reason, provider_label
                except Exception as e:
                    print(f"[Matcher AI Reasoning JSON Parse Error]: {e}")

        # Fallback Deterministic Heuristic Engine with Rigorous Domain Alignment
        cand_tech = cv.get("technical_skills", [])
        cand_soft = cv.get("soft_skills", [])
        edu_list = cv.get("education", [])
        cand_major = " ".join([e.get("major", "") for e in edu_list if isinstance(e, dict)])

        pros = []
        cons = []

        if is_domain_mismatch:
            if cand_soft:
                pros.append(f"Demonstrates transferable interpersonal competencies: {', '.join(cand_soft)}.")
            if total_exp > 0:
                pros.append(f"Possesses {total_exp} years of general work tenure in an adjacent or different professional domain.")
            if cand_major:
                pros.append(f"Educational foundation in: {cand_major.title()}.")

            cons.append(f"Critical Domain Mismatch: Candidate background is outside the target domain and lacks core technical tools: {', '.join(missing_tech) if missing_tech else 'Required Technical Skills'}.")
            if total_exp > 0 and relevant_exp_years == 0:
                cons.append(f"Work experience ({total_exp} years) is in an unrelated field, resulting in 0 years of relevant experience for {job.get('title', 'this role')}.")
            if missing_soft:
                cons.append(f"Soft skills may require further verification during interview: {', '.join(missing_soft)}.")

            rec_reason = f"Candidate is rejected with a match score of {overall_score}%, falling far below the {threshold}% threshold due to critical domain and technical skillset mismatch with the {job.get('title', 'target')} role."
        else:
            if cand_tech:
                matched_cand_tech = [t for t in cand_tech if t.title() in matched_skills]
                if matched_cand_tech:
                    pros.append(f"Proficient in relevant technical capabilities: {', '.join(matched_cand_tech)}.")
                else:
                    pros.append(f"Demonstrates technical software proficiency: {', '.join(cand_tech)}.")

            if cand_soft:
                pros.append(f"Demonstrates interpersonal competencies and work ethic: {', '.join(cand_soft)}.")

            if relevant_exp_years >= min_exp and min_exp > 0:
                pros.append(f"Fulfills work experience requirement with {relevant_exp_years} years of relevant domain experience (target: >= {min_exp} years).")
            elif total_exp > 0:
                pros.append(f"Possesses {total_exp} years of accumulated work experience.")

            if edu_list:
                deg_names = [e.get("degree") for e in edu_list if isinstance(e, dict) and e.get("degree")]
                if deg_names:
                    pros.append(f"Supported by formal education credentials: {', '.join(deg_names)}.")

            if knockout_reasons:
                cons.extend(knockout_reasons)
            if missing_tech:
                cons.append(f"Has not explicitly listed specific software tools required by the job: {', '.join(missing_tech)}.")
            if missing_soft:
                cons.append(f"Soft skills may require further verification during interview: {', '.join(missing_soft)}.")

            if status == "Pass":
                rec_reason = f"Candidate successfully passed evaluation with a match score of {overall_score}%, exceeding the {threshold}% threshold. Profile shows strong technical alignment and meets required experience expectations."
            elif status == "Considered":
                rec_reason = f"Candidate is under consideration with a match score of {overall_score}%. Demonstrates foundational competencies but shows minor skill or experience depth gaps relative to the {threshold}% threshold."
            else:
                reasons_suffix = f" due to mandatory knockout criteria: {'; '.join(knockout_reasons)}" if knockout_reasons else f" falling below the minimum {threshold}% threshold"
                rec_reason = f"Candidate is rejected with an overall score of {overall_score}%,{reasons_suffix}."

        return pros, cons, rec_reason, "Local Intelligent Rule Engine (Offline)"
