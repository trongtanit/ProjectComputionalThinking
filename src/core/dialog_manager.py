# dialog_manager.py - Conversational AI Version
from typing import Dict, Any, Optional, List
import datetime
import json
import requests

from recommender import recommend

# ==================================================
# CONVERSATIONAL AI with Ollama
# ==================================================
class ConversationalAI:
    """AI hiểu ngữ cảnh hội thoại như ChatGPT"""
    
    def __init__(self, model="llama3.2:1b"):
        self.api_url = "http://localhost:11434/api/generate"
        self.model = model
        self.conversation_history: List[str] = []
        
    def understand_intent(self, user_text: str, current_state: Dict) -> Dict:
        """Hiểu ý định user trong ngữ cảnh hội thoại"""
        
        # Build conversation context
        history = "\n".join(self.conversation_history[-6:])  # Last 3 turns
        
        prompt = f"""You are a helpful restaurant recommendation assistant in Vietnam.

CONVERSATION HISTORY:
{history}

CURRENT STATE:
- Location: {current_state.get('location') or 'not set'}
- Food preference: {current_state.get('food_type') or 'not set'}
- Atmosphere: {current_state.get('play_type') or 'not set'}

USER'S NEW MESSAGE: "{user_text}"

Analyze the user's intent and return JSON:
{{
  "action": "update_location" | "update_food" | "update_vibe" | "remove_constraint" | "search" | "clarify",
  "location": "quận X" or null,
  "food_type": "lẩu" | "nướng" | "cafe" | "hải sản" | "đồ hàn" | "buffet" | null,
  "play_type": "chill" | "view đẹp" | "live music" | "bar" | "any" | null,
  "remove_constraint": "play_type" | "food_type" | "location" | null,
  "user_sentiment": "frustrated" | "satisfied" | "neutral" | "changing_mind"
}}

INTENT DETECTION RULES:
- If user says "không quan tâm X nữa", "thôi X", "bỏ X đi" → action: "remove_constraint", remove_constraint: "play_type" or relevant field
- If user says "chỉ cần ngon thôi", "gì cũng được" → play_type: "any" (accept any vibe)
- If user is frustrated (no results) → be flexible, suggest alternatives
- If user changes preference → update relevant field
- Vietnamese: "không quan tâm view" means play_type: "any"

Return ONLY valid JSON, no explanation."""

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = json.loads(response.json()["response"])
                return result
            else:
                print(f"⚠️ AI error: {response.status_code}")
                return {"action": "clarify"}
                
        except Exception as e:
            print(f"⚠️ AI failed: {e}")
            return {"action": "clarify"}
    
    def generate_response(self, state: Dict, results: List[Dict], no_results: bool = False) -> str:
        """Generate natural response like ChatGPT"""
        
        if no_results:
            prompt = f"""User is looking for {state.get('food_type')} restaurant with {state.get('play_type')} vibe in {state.get('location')}, but no results found.

Generate a helpful, friendly response in Vietnamese that:
1. Acknowledges the request
2. Suggests relaxing ONE constraint (vibe, location, or food type)
3. Asks what they prefer to change

Keep it natural, conversational, under 2 sentences."""
        else:
            prompt = f"""Generate a brief, friendly introduction in Vietnamese for {len(results)} {state.get('food_type')} restaurants.

Keep it under 1 sentence, natural tone."""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            }
            
            response = requests.post(self.api_url, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()["response"].strip()
        except:
            pass
        
        # Fallback
        if no_results:
            return "Hmm, mình không tìm thấy quán phù hợp. Bạn muốn thử đổi khu vực, món ăn, hoặc không gian không?"
        return f"Mình tìm được {len(results)} quán {state.get('food_type')} cho bạn:"
    
    def add_to_history(self, role: str, message: str):
        """Add message to conversation history"""
        self.conversation_history.append(f"{role}: {message}")


# ==================================================
# SMART DIALOG MANAGER
# ==================================================
class DialogManager:
    REQUIRED_SLOTS = ["location", "food_type", "play_type"]

    def __init__(self, use_ai=True):
        self.state: Dict[str, Any] = {
            "location": None,
            "food_type": None,
            "play_type": None,
            "budget_vnđ": None,
        }
        
        self.use_ai = use_ai
        self.ai = ConversationalAI() if use_ai else None
        
        if self.use_ai:
            print("✅ Conversational AI mode (ChatGPT-like)")
        else:
            print("⚠️ Rule-based mode")

    def _extract_with_rules_fallback(self, text: str):
        """Fallback extraction"""
        text_lower = text.lower()
        
        # Detect constraint removal
        if any(phrase in text_lower for phrase in ["không quan tâm", "thôi", "bỏ", "gì cũng được", "chỉ cần ngon"]):
            # User wants to remove play_type constraint
            if "view" in text_lower or "không gian" in text_lower or "gì cũng được" in text_lower:
                self.state["play_type"] = "any"
                return
        
        # Location
        districts = ["quận 1", "quận 2", "quận 3", "quận 7", "quận 10", 
                    "bình thạnh", "gò vấp", "tân bình", "phú nhuận", "thủ đức"]
        for d in districts:
            if d in text_lower:
                if not d.startswith("quận") and d != "thủ đức":
                    d = "quận " + d
                self.state["location"] = d
                return
        
        # Food
        foods = {"lẩu": "lẩu", "nướng": "nướng", "cafe": "cafe", "hải sản": "hải sản", 
                "đồ hàn": "đồ hàn", "buffet": "buffet"}
        for k, v in foods.items():
            if k in text_lower:
                self.state["food_type"] = v
                return
        
        # Vibe
        vibes = {"chill": "chill", "view": "view đẹp", "nhạc sống": "live music", "bar": "bar"}
        for k, v in vibes.items():
            if k in text_lower:
                self.state["play_type"] = v
                return

    def process(self, user_text: str) -> str:
        # Add to conversation history
        if self.ai:
            self.ai.add_to_history("User", user_text)
        
        # 1) Understand intent với AI
        if self.use_ai:
            intent = self.ai.understand_intent(user_text, self.state)
            
            # Handle actions
            action = intent.get("action")
            
            if action == "remove_constraint":
                constraint = intent.get("remove_constraint")
                if constraint == "play_type":
                    self.state["play_type"] = "any"
                    print("🔄 Removed vibe constraint")
            
            # Update state from AI
            if intent.get("location"):
                self.state["location"] = intent["location"]
            if intent.get("food_type"):
                self.state["food_type"] = intent["food_type"]
            if intent.get("play_type"):
                if intent["play_type"] == "any":
                    self.state["play_type"] = "any"
                else:
                    self.state["play_type"] = intent["play_type"]
        else:
            self._extract_with_rules_fallback(user_text)
        
        # Debug
        print(f"🔍 State: location={self.state.get('location')}, "
              f"food={self.state.get('food_type')}, "
              f"vibe={self.state.get('play_type')}")
        
        # 2) Check missing slots
        missing = self._missing_slot()
        if missing:
            question = self._ask_question(missing)
            if self.ai:
                self.ai.add_to_history("Bot", question)
            return question
        
        # 3) Build search preference
        pref = {
            "intents_requested": {"Food_Dining", "Entertainment_Leisure"},
            "budget_vnđ": self.state.get("budget_vnđ"),
            "time_hhmm": f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}",
            "max_results": 10
        }
        
        # Add vibe constraint (nếu không phải "any")
        if self.state["play_type"] != "any":
            pref["vibes"] = {self.state["play_type"]}
        
        # Location
        location_coords = {
            "quận 1": (10.775658, 106.700424),
            "quận 3": (10.784369, 106.688079),
            "quận 7": (10.736772, 106.721987),
            "quận bình thạnh": (10.8142, 106.7054),
            "quận phú nhuận": (10.7992, 106.6803),
        }
        loc = self.state["location"]
        pref["start_lat"], pref["start_lon"] = location_coords.get(loc, (10.776, 106.700))
        
        # 4) Search
        results = recommend(pref)
        
        # Filter by food type manually
        food_type = self.state["food_type"]
        if food_type and results:
            food_keywords = {
                "lẩu": ["lẩu", "hot pot", "hotpot"],
                "nướng": ["nướng", "bbq", "grill"],
                "cafe": ["cafe", "coffee", "cà phê"],
                "hải sản": ["seafood", "hải sản"],
                "đồ hàn": ["korean", "hàn"],
                "buffet": ["buffet"],
            }
            
            keywords = food_keywords.get(food_type, [food_type])
            filtered = []
            
            for r in results:
                name_lower = r.get("name", "").lower()
                cat_lower = r.get("category", "").lower()
                
                if any(kw in name_lower or kw in cat_lower for kw in keywords):
                    filtered.append(r)
            
            results = filtered[:10]
        
        # 5) Generate response
        if not results:
            response = self.ai.generate_response(self.state, [], no_results=True) if self.ai else \
                f"Không tìm thấy quán {food_type}. Bạn thử đổi khu vực hoặc món ăn?"
            
            if self.ai:
                self.ai.add_to_history("Bot", response)
            return response
        
        # Format results
        intro = self.ai.generate_response(self.state, results) if self.ai else \
            f"✨ Có {len(results)} quán {food_type}:"
        
        reply = intro + "\n\n"
        for i, r in enumerate(results, 1):
            reply += f"{i}. {r['name']}\n"
            reply += f"   💰 {r['avg_cost']} VNĐ | 📍 {r.get('vibe', 'N/A')}\n"
        
        if self.ai:
            self.ai.add_to_history("Bot", reply)
        return reply

    def _missing_slot(self) -> Optional[str]:
        for slot in self.REQUIRED_SLOTS:
            val = self.state.get(slot)
            if not val or val == "any":
                if slot == "play_type" and val == "any":
                    continue  # "any" is valid for play_type
                if not val:
                    return slot
        return None

    def _ask_question(self, slot: str) -> str:
        if slot == "location":
            return "Bạn muốn tìm ở quận nào? (VD: quận 1, Bình Thạnh...)"
        if slot == "food_type":
            return "Bạn thích ăn gì? (VD: lẩu, nướng, hải sản, cafe, đồ Hàn...)"
        if slot == "play_type":
            return "Bạn thích không gian kiểu nào? (VD: chill, view đẹp, nhạc sống... hoặc 'gì cũng được')"
        return "Bạn có thể nói rõ hơn không?"


if __name__ == "__main__":
    print("=" * 70)
    print("🤖 CONVERSATIONAL AI CHATBOT (ChatGPT-like)")
    print("=" * 70)
    print("\n⚠️  Lưu ý: Chạy 'ollama serve' trước!")
    print("💡 Tip: Nói tự nhiên như chat bình thường\n")
    
    dm = DialogManager(use_ai=True)
    print("Gõ 'exit' để thoát\n")
    
    while True:
        msg = input("Bạn: ")
        if msg.strip().lower() == "exit":
            break
        reply = dm.process(msg)
        print(f"\n🤖 Bot: {reply}\n")