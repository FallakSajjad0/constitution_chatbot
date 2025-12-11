# # llm_free.py - Free LLM API (no key needed)
# import requests
# import json

# class FreeLLM:
#     def __init__(self):
#         # Using Together.ai free endpoint (no key needed for trial)
#         self.api_url = "https://api.together.xyz/v1/chat/completions"
#         self.api_key = "YOUR_FREE_KEY"  # Get from https://api.together.ai
        
#     def invoke(self, prompt: str) -> str:
#         """Use free LLM API"""
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         payload = {
#             "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ],
#             "max_tokens": 512,
#             "temperature": 0.1
#         }
        
#         try:
#             response = requests.post(
#                 self.api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 return response.json()["choices"][0]["message"]["content"]
#             else:
#                 return f"API Error: {response.status_code}"
                
#         except:
#             # Fallback to simple rule-based response
#             return self._fallback_response(prompt)
    
#     def _fallback_response(self, prompt: str) -> str:
#         """Simple fallback if API fails"""
#         if "?" in prompt.lower():
#             return "Based on the documents provided, I can see relevant information but cannot generate a detailed answer at the moment."
#         return "I have retrieved relevant document information. For a complete answer, please check the system configuration."