"""
Google Drive Importer Module
Memungkinkan pengunduhan dan pembacaan berkas PDF CV langsung dari tautan folder maupun tautan 1 file spesifik Google Drive.
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
    Menyediakan fungsi utilitas untuk mengunduh dan mengekstrak berkas PDF dari Google Drive.
    Mendukung tautan folder (multi-file) dan tautan 1 file PDF spesifik.
    """

    @staticmethod
    def extract_id_from_url(url_or_id: str) -> str:
        """
        Mengekstrak ID folder atau file dari berbagai format URL Google Drive.
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
        Mengunduh seluruh file PDF dari folder/file Google Drive ke dalam memori.
        Mendukung tautan folder maupun tautan 1 file PDF individual.
        Returns:
            Tuple[List[Dict[name, bytes, size]], error_message]
        """
        if not gdown:
            return [], "Library 'gdown' belum terpasang di environment sistem."

        if not drive_url_or_id or not drive_url_or_id.strip():
            return [], "Tautan atau ID Google Drive tidak boleh kosong."

        raw_input = drive_url_or_id.strip()
        drive_id = cls.extract_id_from_url(raw_input)
        if not drive_id:
            return [], "Format tautan Google Drive tidak dikenali."

        is_explicit_file = "file/d/" in raw_input or "open?id=" in raw_input or "uc?id=" in raw_input
        is_explicit_folder = "folders/" in raw_input
        
        with tempfile.TemporaryDirectory() as temp_dir:
            last_err = None
            
            # SKENARIO 1: Tautan 1 File Spesifik
            if is_explicit_file:
                try:
                    # Unduh 1 file spesifik (gdown akan otomatis mengambil nama asli file)
                    out_path = os.path.join(temp_dir, "")
                    downloaded_file = gdown.download(
                        id=drive_id,
                        output=out_path,
                        quiet=True,
                        use_cookies=False
                    )
                    if not downloaded_file or not os.path.exists(downloaded_file):
                        # Fallback jika nama file default diperlukan
                        fallback_path = os.path.join(temp_dir, "CV_Candidate.pdf")
                        gdown.download(id=drive_id, output=fallback_path, quiet=True, use_cookies=False)
                except Exception as ef:
                    last_err = ef

            # SKENARIO 2: Tautan Folder (atau deteksi otomatis)
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
                    # Jika gagal sebagai folder, coba sebagai 1 file spesifik
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

            # Pindai dan kumpulkan semua berkas PDF yang berhasil diunduh
            pdf_results = []
            for root, _, files in os.walk(temp_dir):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    if fname.lower().endswith(".pdf") or os.path.isfile(fpath):
                        try:
                            with open(fpath, "rb") as f:
                                b_data = f.read()
                            # Validasi magic bytes PDF (%PDF-)
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
                err_detail = f"\nDetail: {str(last_err)}" if last_err else ""
                return [], (
                    f"Gagal mengunduh berkas PDF dari Google Drive. Pastikan izin akses file/folder telah diatur ke "
                    f"'Siapa saja yang memiliki link' (Anyone with the link can view).{err_detail}"
                )

            return pdf_results, ""
