import requests
class PezzoPromptRenderer:
    """
    A class to get prompts from Pezzo.
    """
    def __init__(
        self,
        api_key: str,
        project_id: str,
        environment: str,
        server_url: str,
    ):
        """
        api_key: str = Pezzo API key
        project_id: str = Pezzo project ID
        environment: str = Pezzo environment name
        server_url: str = Pezzo server host URL
        """
        self.api_key = api_key
        self.project_id = project_id
        self.environment = environment
        self.server_url = server_url

    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt from Pezzo using the prompt name.
        """
        url = f"{self.server_url}/api/prompts/v2/deployment"
        params = {
            "name": prompt_name,
            "environmentName": self.environment,
            "projectId": self.project_id
        }
        headers = {
            "Content-Type": "application/json",
            "x-pezzo-api-key": self.api_key,
            "x-pezzo-project-id": self.project_id
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        if not response.ok:
            error_message = data.get("message",
                                    f"Error fetching {prompt_name} for {self.environment} "
                                    f"({data.get('statusCode')}).")
            raise Exception(error_message)

        return data.get("content")['prompt']
