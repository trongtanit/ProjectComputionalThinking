# dialog_manager.py - Client kết nối tới Backend API
"""
Client chatbot kết nối tới backend server
- Không cần API key
- Chạy được trên mọi máy
- Backend xử lý AI

Setup:
1. Deploy server.py lên Render/Railway (free)
2. Thay SERVER_URL bên dưới
3. Chạy: python dialog_manager.py
"""

import requests
import json
import uuid
from typing import Dict, Any, Optional
import datetime

from recommender import recommend

# ==================================================
# CONFIG - Thay URL này sau khi deploy
# ==================================================
SERVER_URL = "http://localhost:5000"  # Local testing
# SERVER_URL = "https://your-app.onrender.com"  # Production

class DialogManager:
    """Client dialog manager gọi API backend"""
    
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())  # Unique session per user
        self.state = {}
        
        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                ai_mode = "AI" if data.get("ai_available") else "Rules"
                print(f"✅ Connected to server ({ai_mode} mode)")
            else:
                print("⚠️ Server not responding, using offline mode")
                self._use_offline_mode()
        except Exception as e:
            print(f"⚠️ Cannot connect to server: {e}")
            print("📴 Using offline mode")
            self._use_offline_mode()
    
    def _use_offline_mode(self):
        """Fallback to offline rule-based mode"""
        self.offline_mode = True
        self.offline_state = {
            "location": None,
            "food_type": None,
            "play_type": None,
        }
    
    def _offline_extract(self, text: str):
        """Simple offline extraction"""
        import re
        text_lower = text.lower()
        
        # Location
        match = re.search(r"quận\s*(\d+)", text_lower)
        if match:
            self.offline_state["location"] = f"quận {match.group(1)}"
        
        districts = {
            "bình thạnh": "quận bình thạnh",
            "gò vấp": "quận gò vấp",
            "tân bình": "quận tân bình",
        }
        for k, v in districts.items():
            if k in text_lower:
                self.offline_state["location"] = v
                break
        
        # Food
        foods = {"lẩu": "lẩu", "nướng": "nướng", "cafe": "cafe", 
                "hải sản": "hải sản", "đồ hàn": "đồ hàn", "buffet": "buffet"}
        for k, v in foods.items():
            if k in text_lower:
                self.offline_state["food_type"] = v
                break
        
        # Vibe
        if any(x in text_lower for x in ["gì cũng được", "không quan tâm"]):
            self.offline_state["play_type"] = "any"
        else:
            vibes = {"chill": "chill", "view": "view đẹp", "nhạc sống": "live music", "bar": "bar"}
            for k, v in vibes.items():
                if k in text_lower:
                    self.offline_state["play_type"] = v
                    break
    
    def _offline_process(self, user_text: str) -> str:
        """Offline processing"""
        self._offline_extract(user_text)
        
        # Check missing
        if not self.offline_state.get("location"):
            return "Bạn muốn tìm ở quận nào? (VD: quận 1, quận 3...)"
        if not self.offline_state.get("food_type"):
            return "Bạn thích ăn gì? (VD: lẩu, nướng, cafe...)"
        if not self.offline_state.get("play_type"):
            return "Bạn thích không gian nào? (VD: chill, view đẹp... hoặc 'gì cũng được')"
        
        # Search
        return self._search_and_format(self.offline_state)
    
    def process(self, user_text: str) -> str:
        """
        Process user message
        - Online: Call API backend
        - Offline: Use local rules
        """
        # Offline mode
        if hasattr(self, 'offline_mode') and self.offline_mode:
            return self._offline_process(user_text)
        
        # Online mode: Call API
        try:
            response = requests.post(
                f"{self.server_url}/chat",
                json={
                    "session_id": self.session_id,
                    "message": user_text
                },
                timeout=10
            )
            
            if response.status_code != 200:
                print("⚠️ Server error, switching to offline mode")
                self._use_offline_mode()
                return self._offline_process(user_text)
            
            data = response.json()
            
            # Update local state
            self.state = data.get("state", {})
            
            # Debug
            print(f"🔍 State: {self.state}")
            
            # If complete, do search
            if data.get("complete"):
                return self._search_and_format(self.state)
            
            # Return bot's question
            return data.get("reply", "Xin lỗi, tôi không hiểu.")
            
        except requests.exceptions.Timeout:
            print("⏱️ Server timeout, using offline mode")
            self._use_offline_mode()
            return self._offline_process(user_text)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại."
    
    def _search_and_format(self, state: Dict) -> str:
        """Search venues và format kết quả"""
        # Build preference
        pref = {
            "intents_requested": {"Food_Dining", "Entertainment_Leisure"},
            "time_hhmm": f"{datetime.datetime.now().hour:02d}:{datetime.datetime.now().minute:02d}",
            "max_results": 15
        }
        
        # Add vibe
        if state.get("play_type") and state["play_type"] != "any":
            pref["vibes"] = {state["play_type"]}
        
        # Add location
        location_coords = {
            "quận 1": (10.775658, 106.700424),
            "quận 3": (10.784369, 106.688079),
            "quận 7": (10.736772, 106.721987),
            "quận bình thạnh": (10.8142, 106.7054),
            "quận gò vấp": (10.8379, 106.6752),
            "quận tân bình": (10.7992, 106.6522),
        }
        loc = state.get("location", "quận 1")
        pref["start_lat"], pref["start_lon"] = location_coords.get(loc, (10.776, 106.700))
        
        # Search
        print("🔎 Đang tìm kiếm...")
        results = recommend(pref)
        
        # Filter by food
        food_type = state.get("food_type")
        if food_type and results:
            food_keywords = {
                "lẩu": ["lẩu", "lau", "hot pot"],
                "nướng": ["nướng", "nuong", "bbq"],
                "cafe": ["cafe", "coffee"],
                "hải sản": ["seafood", "hai san"],
                "đồ hàn": ["korean", "han"],
                "buffet": ["buffet"],
            }
            
            keywords = food_keywords.get(food_type, [food_type])
            
            filtered = []
            for r in results:
                name_cat = (r.get("name", "") + r.get("category", "")).lower()
                if any(kw in name_cat for kw in keywords):
                    filtered.append(r)
            
            results = filtered[:10]
        
        # Format
        if not results:
            return f"😔 Không tìm thấy quán {food_type} ở {loc}. Thử đổi khu vực hoặc món ăn?"
        
        reply = f"✨ Có {len(results)} quán {food_type} ở {loc}:\n\n"
        for i, r in enumerate(results, 1):
            reply += f"{i}. {r['name']}\n   💰 {r['avg_cost']} VNĐ\n"
        
        return reply
    
    def reset(self):
        """Reset conversation"""
        try:
            requests.post(
                f"{self.server_url}/reset",
                json={"session_id": self.session_id},
                timeout=5
            )
        except:
            pass
        
        self.session_id = str(uuid.uuid4())
        self.state = {}
        
        if hasattr(self, 'offline_mode'):
            self.offline_state = {
                "location": None,
                "food_type": None,
                "play_type": None,
            }
        
        print("🔄 Đã reset")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 CHATBOT GỢI Ý ĐỊA ĐIỂM - SMART AI VERSION")
    print("=" * 70)
    print(f"\n🌐 Server: {SERVER_URL}")
    print("\n💡 Hướng dẫn:")
    print("   - Nói tự nhiên như chat bình thường")
    print("   - Gõ 'reset' để bắt đầu lại")
    print("   - Gõ 'exit' để thoát\n")
    
    dm = DialogManager()
    
    while True:
        try:
            user_input = input("Bạn: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("👋 Tạm biệt!")
                break
            
            if user_input.lower() == "reset":
                dm.reset()
                continue
            
            response = dm.process(user_input)
            print(f"\n🤖: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break