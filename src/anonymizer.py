"""
Blind-CV Anonymizer Engine (Bias Mitigation Layer)
Responsible for dynamically masking Personally Identifiable Information (PII) 
based on granular user configuration across all CV profile fields.
"""

from typing import Dict, Any, List
import copy

class BlindCVAnonymizer:
    """
    Anonymizes sensitive PII fields from candidate CV data
    to enforce ethical, merit-based, and unbiased candidate screening.
    """
    
    DEFAULT_PII_FIELDS = [
        "full_name",
        "email",
        "phone",
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
        candidate_num = cv_id.split('-')[-1]
        
        # 1. Mask Personal Info
        personal_info = anonymized.get("personal_info", {})
        masked_info = copy.deepcopy(personal_info)
        
        if "full_name" in enabled_fields:
            masked_info["candidate_alias"] = f"CANDIDATE-{candidate_num}"
            masked_info["full_name"] = f"CANDIDATE-{candidate_num} (Anonymized)"
        else:
            masked_info["candidate_alias"] = personal_info.get("full_name", f"CANDIDATE-{candidate_num}")
            
        if "email" in enabled_fields:
            masked_info["email"] = "[MASKED_EMAIL@ANONYMIZED.LOCAL]"
            
        if "phone" in enabled_fields or "email" in enabled_fields:
            masked_info["phone"] = "[MASKED_PHONE]"
            
        if "gender" in enabled_fields:
            masked_info["gender"] = "[MASKED_GENDER]"
            
        if "age" in enabled_fields:
            masked_info["age"] = "[MASKED_AGE]"
            
        if "photo_url" in enabled_fields:
            masked_info["photo_url"] = ""
            
        if "address" in enabled_fields:
            masked_info["address"] = "Regional Location (Masked)"
            
        anonymized["personal_info"] = masked_info
        
        # 2. Mask Education Institution Names
        if "university" in enabled_fields:
            edu_data = anonymized.get("education", [])
            if isinstance(edu_data, list):
                for item in edu_data:
                    if isinstance(item, dict) and "institution" in item:
                        item["institution"] = "Accredited Higher Education Institution (Masked)"
            elif isinstance(edu_data, dict):
                if "institution" in edu_data:
                    edu_data["institution"] = "Accredited Higher Education Institution (Masked)"
                    edu_data["institution_tier"] = "Accredited Institution (Masked)"

        anonymized["is_anonymized"] = True
        return anonymized
