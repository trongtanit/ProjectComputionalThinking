# server.py - Backend API với Gemini AI
"""
Backend server để share AI cho cả nhóm
Deploy lên Render/Railway (free tier)

Usage:
1. Set GEMINI_API_KEY trong environment variables trên server
2. Deploy lên Render/Railway
3. Client chỉ cần gọi API, không cần API key
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from typing import Dict, Any
import datetime

# Import dialog manager (sẽ dùng Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Configure Gemini từ environment variable (set trên server)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI configured")
else:
    print("⚠️ Gemini not available, using fallback")

# Store sessions in-memory (production nên dùng Redis)
sessions = {}

class SimpleDialogManager:
    """Simplified dialog manager cho server"""
    
    def __init__(self):
        self.state = {
            "location": None,
            "food_type": None,
            "play_type": None,
        }
        self.history = []
        
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.use_ai = True
        else:
            self.model = None
            self.use_ai = False
    
    def extract_slots(self, user_text: str) -> Dict:
        """Extract slots với Gemini"""
        if not self.use_ai:
            return self._extract_with_rules(user_text)
        
        prompt = f"""Extract info from: "{user_text}"

Current state:
- Location: {self.state.get('location') or 'none'}
- Food: {self.state.get('food_type') or 'none'}
- Vibe: {self.state.get('play_type') or 'none'}

Return JSON:
{{
  "location": "quận 1" | "quận 3" | ... | null,
  "food_type": "lẩu" | "nướng" | "cafe" | ... | null,
  "play_type": "chill" | "view đẹp" | "any" | null
}}

Rules:
- If user says district name → extract it
- If user says "gì cũng được" → play_type: "any"
- Only extract what user mentioned
- Return valid JSON only, no markdown"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(text)
            
            # Update state
            if result.get("location"):
                self.state["location"] = result["location"]
            if result.get("food_type"):
                self.state["food_type"] = result["food_type"]
            if result.get("play_type"):
                self.state["play_type"] = result["play_type"]
            
            return self.state
            
        except Exception as e:
            print(f"AI error: {e}")
            return self._extract_with_rules(user_text)
    
    def _extract_with_rules(self, text: str) -> Dict:
        """Fallback rules"""
        import re
        text_lower = text.lower()
        
        # Location
        match = re.search(r"quận\s*(\d+)", text_lower)
        if match:
            self.state["location"] = f"quận {match.group(1)}"
        
        # Food
        foods = {"lẩu": "lẩu", "nướng": "nướng", "cafe": "cafe", 
                "hải sản": "hải sản", "đồ hàn": "đồ hàn"}
        for k, v in foods.items():
            if k in text_lower:
                self.state["food_type"] = v
                break
        
        # Vibe
        if any(x in text_lower for x in ["gì cũng được", "không quan tâm"]):
            self.state["play_type"] = "any"
        else:
            vibes = {"chill": "chill", "view": "view đẹp", "bar": "bar"}
            for k, v in vibes.items():
                if k in text_lower:
                    self.state["play_type"] = v
                    break
        
        return self.state
    
    def get_missing_slot(self):
        """Check missing slots"""
        if not self.state.get("location"):
            return "location"
        if not self.state.get("food_type"):
            return "food_type"
        if not self.state.get("play_type"):
            return "play_type"
        return None


# ==================================================
# API ENDPOINTS
# ==================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "ai_available": GEMINI_AVAILABLE and bool(GEMINI_API_KEY)
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    
    Request body:
    {
        "session_id": "user123",
        "message": "tôi muốn ăn lẩu"
    }
    
    Response:
    {
        "reply": "Bạn muốn ăn ở quận nào?",
        "state": {...},
        "complete": false
    }
    """
    try:
        data = request.json
        session_id = data.get("session_id", "default")
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "Message required"}), 400
        
        # Get or create session
        if session_id not in sessions:
            sessions[session_id] = SimpleDialogManager()
        
        dm = sessions[session_id]
        
        # Extract slots
        state = dm.extract_slots(user_message)
        
        # Check missing
        missing = dm.get_missing_slot()
        
        if missing:
            # Generate question
            questions = {
                "location": "Bạn muốn tìm ở quận nào? (VD: quận 1, quận 3...)",
                "food_type": "Bạn thích ăn gì? (VD: lẩu, nướng, cafe...)",
                "play_type": "Bạn thích không gian nào? (VD: chill, view đẹp... hoặc 'gì cũng được')"
            }
            
            return jsonify({
                "reply": questions.get(missing, "Bạn có thể nói rõ hơn?"),
                "state": state,
                "complete": False,
                "missing_slot": missing
            })
        
        # All slots filled → Return success
        return jsonify({
            "reply": "✅ Đủ thông tin! Đang tìm kiếm...",
            "state": state,
            "complete": True,
            "missing_slot": None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset conversation session"""
    data = request.json
    session_id = data.get("session_id", "default")
    
    if session_id in sessions:
        del sessions[session_id]
    
    return jsonify({"status": "reset"})


# ==================================================
# MAIN
# ==================================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    
    print("=" * 60)
    print("🚀 BACKEND SERVER STARTING")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"AI Mode: {'Gemini' if (GEMINI_AVAILABLE and GEMINI_API_KEY) else 'Rules'}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)