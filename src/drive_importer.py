"""
Google Drive Importer Module
Memungkinkan pengunduhan dan pembacaan berkas PDF CV langsung dari tautan folder Google Drive publik/terbagikan.
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
        Returns:
            Tuple[List[Dict[name, bytes, size]], error_message]
        """
        if not gdown:
            return [], "Library 'gdown' belum terpasang di environment sistem."

        drive_id = cls.extract_id_from_url(drive_url_or_id)
        if not drive_id:
            return [], "Tautan atau ID Google Drive tidak boleh kosong."

        folder_url = f"https://drive.google.com/drive/folders/{drive_id}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded = None
            try:
                # 1. Coba unduh sebagai folder
                downloaded = gdown.download_folder(
                    url=folder_url,
                    output=temp_dir,
                    quiet=True,
                    remaining_ok=True
                )
            except Exception as e_folder:
                # 2. Coba unduh sebagai single file jika bukan folder
                try:
                    target_file = os.path.join(temp_dir, "document.pdf")
                    single_res = gdown.download(
                        id=drive_id,
                        output=target_file,
                        quiet=True
                    )
                    if single_res and os.path.exists(single_res):
                        downloaded = [single_res]
                except Exception as e_single:
                    return [], (
                        f"Gagal mengakses Google Drive: Pastikan izin folder/file telah diatur ke "
                        f"'Siapa saja yang memiliki link' (Anyone with the link can view).\n"
                        f"Detail: {str(e_folder)}"
                    )

            # Temukan semua berkas PDF yang berhasil diunduh
            pdf_results = []
            for root, _, files in os.walk(temp_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(".pdf"):
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, "rb") as f:
                                b_data = f.read()
                            if len(b_data) > 0:
                                pdf_results.append({
                                    "name": fname,
                                    "bytes": b_data,
                                    "size": len(b_data)
                                })
                        except Exception:
                            continue

            if not pdf_results:
                return [], "Tidak ditemukan berkas PDF (.pdf) di dalam folder Google Drive tersebut."

            return pdf_results, ""
