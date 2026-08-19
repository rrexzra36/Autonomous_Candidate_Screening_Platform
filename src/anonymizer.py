"""
Blind-CV Anonymizer Engine (Bias Mitigation Layer)
Bertanggung jawab menghapus Person Identifiable Information (PII) 
sebelum data CV dikirim ke Matching & Scoring Engine.
"""

from typing import Dict, Any

class BlindCVAnonymizer:
    """
    Anonymizes sensitive PII fields from candidate CV data
    to enforce ethical, unbiased screening.
    """
    
    PII_FIELDS_TO_STRIP = [
        "full_name",
        "email",
        "gender",
        "age",
        "photo_url",
        "address"
    ]
    
    @classmethod
    def anonymize_cv(cls, raw_cv: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw CV dictionary and returns anonymized version with masked PII.
        """
        anonymized = raw_cv.copy()
        cv_id = anonymized.get("cv_id", "UNKNOWN_CV")
        
        # Mask Personal Info
        personal_info = anonymized.get("personal_info", {})
        masked_info = {
            "candidate_alias": f"CANDIDATE-{cv_id.split('-')[-1]}",
            "location_tier": "Regional (Masked)",
            "demographics_status": "PII Masked for Unbiased Evaluation"
        }
        anonymized["personal_info"] = masked_info
        
        # Mask Institution specific names to institutional accreditation tier
        education = anonymized.get("education", {})
        if isinstance(education, dict) and "institution" in education:
            education["institution_tier"] = "Accredited Higher Education Institution (Masked)"
            del education["institution"]
        
        anonymized["is_anonymized"] = True
        return anonymized

if __name__ == "__main__":
    import json
    sample_cv = {
        "cv_id": "CV-RAW-001",
        "personal_info": {"full_name": "Budi Santoso", "gender": "Laki-laki", "age": 27},
        "education": {"degree": "S1 Teknik Mesin", "institution": "Universitas Indonesia"}
    }
    result = BlindCVAnonymizer.anonymize_cv(sample_cv)
    print(json.dumps(result, indent=2))
