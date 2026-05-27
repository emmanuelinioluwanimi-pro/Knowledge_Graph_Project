# main.py
from pdf_extractor import extract_all_pages
from vlm_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import json
from PIL import Image
import io
import base64
from neo4j_loader import Neo4jLoader
import requests
import ollama


def call_vlm(page_data):
    url = "http://localhost:11434/api/chat"
    
    user_message = USER_PROMPT_TEMPLATE.format(
        page_number=page_data["page_number"],
        text=page_data.get("text", "")[:1200]
    )
    
    # Ollama format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_message,
            "images": [page_data["image_base64"]]   # ← Correct way
        }
    ]
    
    payload = {
        "model": "llava:7b",
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192
        }
    }
    
    try:
        print(f"   ⏳ Calling Qwen2.5-VL for page {page_data['page_number']}...")
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        raw_content = result["message"]["content"]
        
        print(f"\n--- RAW OUTPUT (Page {page_data['page_number']}) ---")
        print(raw_content[:800] + "..." if len(raw_content) > 800 else raw_content)
        print("--- END RAW OUTPUT ---\n")
        
        return json.loads(raw_content)
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"❌ VLM Error on page {page_data['page_number']}: {e}")
        return None


import time

def process_manual(pdf_path, product_code):
    print(f"Processing {pdf_path}...")
    pages = extract_all_pages(pdf_path)
    
    all_data = {
        "manual": {"productName": product_code, "productCode": product_code},
        "steps": [],
        "parts": [],
        "tools": [],
        "figures": []
    }
    
    for page in pages:
        print(f"  Processing page {page['page_number']}...")
        page_result = call_vlm(page)
        
        if page_result:
            # Merge whatever we get
            if isinstance(page_result, dict):
                if "steps" in page_result:
                    all_data["steps"].extend(page_result.get("steps", []))
                if "parts" in page_result or "parts_list" in page_result:
                    all_data["parts"].extend(page_result.get("parts", page_result.get("parts_list", [])))
                if "tools" in page_result or "tools_required" in page_result:
                    all_data["tools"].extend(page_result.get("tools", page_result.get("tools_required", [])))
        
        time.sleep(12)  # Wait 12 seconds between calls to respect free tier limits
    
    return all_data


# ==================== CONFIG ====================
NEO4J_URI = "****************************"
NEO4J_USER = "*******"
NEO4J_PASSWORD = "***************************"
# ===============================================

if __name__ == "__main__":
    print("🚀 Starting IKEA Manual Processing with Gemini...\n")
    
    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    data = process_manual("manuals/sandsberg-table.pdf", "SANDSBERG")
    
    if data:
        loader.load_manual_data(data, "SANDSBERG")
        print("✅ Processing completed!")
    
    loader.close()
