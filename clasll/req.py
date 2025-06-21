import requests
import json

sentence = (
    "Mumbai (formerly called Bombay) is a densely populated city on India’s west coast. "
    "A financial center, it's India's largest city. On the Mumbai Harbour waterfront stands "
    "the iconic Gateway of India stone arch, built by the British Raj in 1924. Offshore, nearby "
    "Elephanta Island holds ancient cave temples dedicated to the Hindu god Shiva. The city's also "
    "famous as the heart of the Bollywood film industry."
)

headers = {"Content-Type": "application/json"}

# --- Analyze for tags ---
analyze_url = "http://127.0.0.1:5000/analyze"
analyze_data = {"sentence": sentence}

analyze_response = requests.post(analyze_url, headers=headers, data=json.dumps(analyze_data))

if analyze_response.status_code == 200:
    print("✅ Final Combined Tags:")
    print(json.dumps(analyze_response.json(), indent=2))
else:
    print("❌ Analyze Error:", analyze_response.status_code)
    print(analyze_response.text)


