import requests

def analyze_incident(incident_data):
    """Sends time-window telemetry to local LLM for explanation."""
    prompt = f"""
    You are a SOC analyst. Review this anomalous system activity that occurred at {incident_data['timestamp']}:
    - Failed Logins: {incident_data['failed_logins']}
    - Successful Logins: {incident_data['success_logins']}
    - Unique IPs observed: {incident_data['unique_ips']}
    - Hour of Day: {incident_data['hour_of_day']}:00
    
    In exactly two short sentences, explain why this time window is suspicious and what attack is likely occurring.
    """
    
    try:
        response = requests.post('http://localhost:11434/api/generate', 
                                 json={"model": "llama3", "prompt": prompt, "stream": False})
        if response.status_code == 200:
            return response.json()['response']
        return "Error: LLM returned non-200 status."
    except requests.exceptions.ConnectionError:
        return "Connection Error: Is Ollama running locally?"
