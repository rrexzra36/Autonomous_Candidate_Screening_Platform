"""
Google Drive Importer Module
Enables downloading and extracting PDF CVs/Job Descriptions directly from public Google Drive folder or specific file links.
"""

import os
import tempfile
from typing import List, Dict, Any, Tuple

try:
    import gdown
except ImportError:
    gdown = None


class GoogleDriveImporter:
    """
    Utility class to download and extract PDF documents from Google Drive.
    Supports both folder links (multi-file batch ingestion) and specific single file links.
    """

    @staticmethod
    def extract_id_from_url(url_or_id: str) -> str:
        """
        Extracts the folder ID or file ID from various Google Drive URL formats.
        """
        if not url_or_id:
            return ""
        url_or_id = url_or_id.strip()
        if "folders/" in url_or_id:
            return url_or_id.split("folders/")[1].split("?")[0].split("/")[0]
        if "file/d/" in url_or_id:
            return url_or_id.split("file/d/")[1].split("?")[0].split("/")[0]
        if "id=" in url_or_id:
            return url_or_id.split("id=")[1].split("&")[0]
        return url_or_id

    @classmethod
    def fetch_pdf_files_from_drive(cls, drive_url_or_id: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Downloads all PDF files from a Google Drive folder or specific file link into memory.
        Returns:
            Tuple[List[Dict[name, bytes, size]], error_message]
        """
        if not gdown:
            return [], "The 'gdown' library is not installed in the current environment."

        if not drive_url_or_id or not drive_url_or_id.strip():
            return [], "Google Drive URL or ID cannot be empty."

        raw_input = drive_url_or_id.strip()
        drive_id = cls.extract_id_from_url(raw_input)
        if not drive_id:
            return [], "Unrecognized Google Drive URL format."

        is_explicit_file = "file/d/" in raw_input or "open?id=" in raw_input or "uc?id=" in raw_input
        is_explicit_folder = "folders/" in raw_input
        
        with tempfile.TemporaryDirectory() as temp_dir:
            last_err = None
            
            # SCENARIO 1: Specific Single File Link
            if is_explicit_file:
                try:
                    out_path = os.path.join(temp_dir, "")
                    downloaded_file = gdown.download(
                        id=drive_id,
                        output=out_path,
                        quiet=True,
                        use_cookies=False
                    )
                    if not downloaded_file or not os.path.exists(downloaded_file):
                        fallback_path = os.path.join(temp_dir, "Document.pdf")
                        gdown.download(id=drive_id, output=fallback_path, quiet=True, use_cookies=False)
                except Exception as ef:
                    last_err = ef

            # SCENARIO 2: Folder Link (or auto-detect fallback)
            else:
                try:
                    gdown.download_folder(
                        id=drive_id,
                        output=temp_dir,
                        quiet=True,
                        use_cookies=False
                    )
                except Exception as e_f:
                    last_err = e_f
                    if not is_explicit_folder:
                        try:
                            out_path = os.path.join(temp_dir, "")
                            gdown.download(
                                id=drive_id,
                                output=out_path,
                                quiet=True,
                                use_cookies=False
                            )
                        except Exception as e_s:
                            last_err = e_s

            # Scan and collect all downloaded PDF documents
            pdf_results = []
            for root, _, files in os.walk(temp_dir):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    if fname.lower().endswith(".pdf") or os.path.isfile(fpath):
                        try:
                            with open(fpath, "rb") as f:
                                b_data = f.read()
                            # Check PDF magic bytes (%PDF-)
                            if len(b_data) > 0:
                                clean_name = fname if fname.lower().endswith(".pdf") else f"{fname}.pdf"
                                pdf_results.append({
                                    "name": clean_name,
                                    "bytes": b_data,
                                    "size": len(b_data)
                                })
                        except Exception:
                            continue

            if not pdf_results:
                err_detail = f"\nDetails: {str(last_err)}" if last_err else ""
                return [], (
                    f"Failed to retrieve PDF files from Google Drive. Please ensure the access permission is set to "
                    f"'Anyone with the link can view'.{err_detail}"
                )

            return pdf_results, ""
