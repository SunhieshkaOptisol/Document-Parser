"""
This module provides services to interact with SharePoint, including retrieving and downloading files.
"""

from typing import Dict, List, Any
import os
import requests
from elsai_core.config.loggerConfig import setup_logger
from elsai_core.config.sharepoint_auth_service import get_access_token

class SharePointService:
    """
    A service class to interact with SharePoint for file retrieval and download.
    """
    def __init__(self):
        self.logger = setup_logger()

    def retrieve_sharepoint_files_from_folder(self, folder_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve files from a specified folder in SharePoint.

        Args:
            folder_name (str): The name of the folder in SharePoint Document Library from which to fetch files.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing file information with the following structure:
                {
                    "files": [
                        {
                            "file-name": str,  # The name of the file
                            "file-id": str     # The unique ID of the file
                        },
                        ...
                    ]
                }

        Raises:
            Exception: If any error occurs while fetching files, such as issues with authentication, 
                       site ID retrieval, drive ID retrieval, or folder/file access.
        """
        try:
            # Step 1: Get Site ID
            self.logger.info("Getting access token...")
            access_token = get_access_token()
            self.logger.info("Starting to retrieve files from SharePoint folder: %s", folder_name)
            self.logger.info("Fetching site information")

            site_hostname = os.getenv("SITE_HOSTNAME")
            site_path = os.getenv("SITE_PATH")
            site_url = f"https://graph.microsoft.com/v1.0/sites/{site_hostname}:{site_path}"
            headers = {"Authorization": f"Bearer {access_token}"}
            self.logger.info("Making GET request to %s for site info.", site_url)
            response = requests.get(site_url, headers=headers, timeout=10)
            response.raise_for_status()
            site_id = response.json()["id"]

            self.logger.info("Successfully retrieved site ID: %s", site_id)

            # Step 2: Get Drive ID
            self.logger.info("Fetching drives information.")
            drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
            response = requests.get(drives_url, headers=headers, timeout=10)
            response.raise_for_status()
            drives = response.json()["value"]
            
            self.logger.info("Drives available: %s", drives)
            drive_name = os.getenv("DRIVE_NAME")

            drive_id = next((drive["id"] for drive in drives if drive["name"] == drive_name), None)
            if not drive_id:
                self.logger.error("Drive '%s' not found in site '%s'.", drive_name, site_path)
                raise ValueError(f"Drive '{drive_name}' not found in site '{site_path}'.")

            self.logger.info("Found drive ID: %s for drive name: %s", drive_id, drive_name)

            # Step 3: Fetch Files in Folder
            self.logger.info("Fetching files in folder: %s", folder_name)
            folder_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{folder_name}:/children"
            response = requests.get(folder_url, headers=headers, timeout=10)
            response.raise_for_status()
            files = response.json().get("value", [])
            if not files:
                self.logger.warning("No files found in folder: %s", folder_name)
            files_info = []
            for file in files:
                if file.get("file"):
                    file_name = file["name"]
                    file_id = file["id"]
                    files_info.append({
                        "file-name": file_name,
                        "file-id": file_id
                    })
                    self.logger.info("Found file: %s (ID: %s)", file_name, file_id)
            self.logger.info(
                "Successfully retrieved %d files from folder: %s",
                len(files_info), 
                folder_name
            )
            return {"files": files_info}
        except requests.exceptions.RequestException as e:
            self.logger.error("HTTP error occurred: %s", str(e))
            raise
        except ValueError as e:
            self.logger.error("Value error: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Unexpected error: %s", str(e))
            raise

    def download_file_from_sharepoint(self, file_id: str, target_folder: str):
        """
        Download a file from SharePoint using its file ID.

        Args:
            file_id (str): The ID of the file to download.

        Returns:
            bytes: The binary content of the downloaded file.

        Raises:
            Exception: If file download fails.
        """
        try:
            self.logger.info("Getting access token...")
            access_token = get_access_token()
            headers = {"Authorization": f"Bearer {access_token}"}
            drive_id = os.getenv("DRIVE_ID")
            download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
            self.logger.info("Downloading file with ID: %s", file_id)
            
            response = requests.get(download_url, headers=headers, timeout=10)
            response.raise_for_status()
           
            self.logger.info("Writing file....")
            metadata_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}"
            metadata_response = requests.get(metadata_url, headers=headers, timeout=10)
            metadata_response.raise_for_status()
            file_name = metadata_response.json().get("name")
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                self.logger.info("Created folder: %s", target_folder)
            local_file_path = os.path.join(target_folder, file_name)

            with open(local_file_path, 'wb') as f:
                f.write(response.content)
                self.logger.info("File Saved Successfully")
            return local_file_path
        
        except requests.exceptions.RequestException as e:
            self.logger.error("HTTP error occurred: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Unexpected error: %s", str(e))
            raise
