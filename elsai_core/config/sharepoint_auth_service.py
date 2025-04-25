import os
import requests
def get_access_token():
    """
    Retrieves an OAuth2 access token for SharePoint using client credentials.
    Returns:
        str: Access token for authenticating SharePoint API requests.
    Raises:
        requests.exceptions.RequestException: If the request to acquire the access token fails.
    """
    tenant_id = os.getenv("TENANT_ID")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }
    response = requests.post(token_url, data=data, timeout=15)

    if response.status_code == 200:
        return response.json().get("access_token")
    raise requests.exceptions.RequestException("Failed to acquire access token.")
    