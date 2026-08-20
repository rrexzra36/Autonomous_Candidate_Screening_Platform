"""
Blind-CV Anonymizer Engine (Bias Mitigation Layer)
Bertanggung jawab menghapus Person Identifiable Information (PII) 
secara fleksibel berdasarkan checklist pilihan pengguna sebelum data CV diproses.
"""

from typing import Dict, Any, List
import copy

class BlindCVAnonymizer:
    """
    Anonymizes sensitive PII fields from candidate CV data
    to enforce ethical, unbiased screening with granular field control.
    """
    
    DEFAULT_PII_FIELDS = [
        "full_name",
        "email",
        "gender",
        "age",
        "photo_url",
        "address",
        "university"
    ]
    
    @classmethod
    def anonymize_cv(cls, raw_cv: Dict[str, Any], enabled_fields: List[str] = None) -> Dict[str, Any]:
        """
        Takes raw CV dictionary and returns anonymized version with masked PII based on enabled fields.
        """
        if enabled_fields is None:
            enabled_fields = cls.DEFAULT_PII_FIELDS

        anonymized = copy.deepcopy(raw_cv)
        cv_id = anonymized.get("cv_id", "UNKNOWN_CV")
        
        # Mask Personal Info
        personal_info = anonymized.get("personal_info", {})
        masked_info = copy.deepcopy(personal_info)
        
        if "full_name" in enabled_fields:
            candidate_num = cv_id.split('-')[-1]
            masked_info["candidate_alias"] = f"CANDIDATE-{candidate_num}"
            masked_info["full_name"] = f"CANDIDATE-{candidate_num} (Anonymized)"
        else:
            masked_info["candidate_alias"] = personal_info.get("full_name", f"CANDIDATE-{cv_id.split('-')[-1]}")
            
        if "email" in enabled_fields:
            masked_info["email"] = "[MASKED_EMAIL@ANONYMIZED.LOCAL]"
            
        if "gender" in enabled_fields:
            masked_info["gender"] = "[MASKED_GENDER]"
            
        if "age" in enabled_fields:
            masked_info["age"] = "[MASKED_AGE]"
            
        if "photo_url" in enabled_fields:
            masked_info["photo_url"] = ""
            
        if "address" in enabled_fields:
            masked_info["address"] = "Regional (Masked)"
            
        if "university" in enabled_fields:
            masked_info["university"] = "Accredited Higher Education Institution (Masked)"
            education = anonymized.get("education", {})
            if isinstance(education, dict) and "institution" in education:
                education["institution"] = "Accredited Higher Education Institution (Masked)"
                education["institution_tier"] = "Accredited Institution (Masked)"

        anonymized["personal_info"] = masked_info
        anonymized["is_anonymized"] = True
        return anonymized

if __name__ == "__main__":
    import json
    sample_cv = {
        "cv_id": "CV-RAW-001",
        "personal_info": {"full_name": "Budi Santoso", "email": "budi@email.com", "gender": "Laki-laki", "age": 27, "address": "Jakarta", "university": "Universitas Indonesia"},
        "education": {"degree": "S1 Teknik Mesin", "institution": "Universitas Indonesia"}
    }
    result = BlindCVAnonymizer.anonymize_cv(sample_cv, ["full_name", "email", "gender"])
    print(json.dumps(result, indent=2))
