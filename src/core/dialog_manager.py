# src/core/dialog_manager.py
import os
import json
import re
from typing import Any, Dict, List, Optional

# Azure SDK (the style you requested)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# DataManager (bạn đã tạo trước đó)
from core.data_manager import DataManager

# -------------------------
# CONFIG
# -------------------------
ENDPOINT = "https://models.github.ai/inference"
MODEL = "gpt-4o-mini"   # đổi nếu bạn có model khác được cấp quyền
TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("❌ Chưa đặt biến môi trường GITHUB_TOKEN!")

client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))

# load dataset (DataManager sẽ in thông báo)
DATA = DataManager("../data/travel_poi_data_ranked.csv")

# -------------------------
# UTIL: cố gắng parse JSON từ LLM output
# -------------------------
def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # loại bỏ markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    # Some LLMs prepend/explain — try to find first JSON {...}
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        candidate = text
    # try loads
    try:
        return json.loads(candidate)
    except Exception:
        # try a relaxed transform: replace single quotes to double, null/none -> null
        t2 = candidate.replace("'", '"')
        t2 = re.sub(r"\bNone\b|\bnone\b", "null", t2)
        t2 = re.sub(r"\bNone\b|\bNull\b", "null", t2)
        try:
            return json.loads(t2)
        except Exception:
            return None

# -------------------------
# UTIL: simple rule-based fallback extractor if JSON parse fails
# (keeps system working even when LLM is noisy)
# -------------------------
def fallback_extract(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()
    prefs = {
        "food": [],
        "entertainment": [],
        "vibe": "",
        "area": "",
        "budget": 0
    }
    # food keywords rough heuristics (add more if needed)
    food_keywords = ["hàn", "korean", "nhật", "sushi", "hải sản", "cơm gà", "bún", "phở", "cafe", "cà phê", "trà sữa", "pizza"]
    for kw in food_keywords:
        if kw in text:
            prefs["food"].append(kw)
    # entertainment
    ent_keywords = ["vui chơi", "bar", "quán bia", "club", "công viên", "rạp", "bowling", "khu vui chơi", "cafe", "chill"]
    for kw in ent_keywords:
        if kw in text:
            prefs["entertainment"].append(kw)
    # vibe
    for v in ["chill", "thư giãn", "sang trọng", "lãng mạn", "thân thiện", "năng động", "yên tĩnh"]:
        if v in text:
            prefs["vibe"] = v
            break
    # area (quận)
    m_area = re.search(r"quận\s*(\d{1,2})", text)
    if m_area:
        prefs["area"] = f"quận {m_area.group(1)}"
    # budget numeric like "300k", "300 nghìn", "3tr", "300000"
    m_budget = re.search(r"(\d+(\.\d+)?)(\s*)(k|nghìn|tr|triệu|m|vnd)?", text)
    if m_budget:
        num = float(m_budget.group(1))
        unit = (m_budget.group(4) or "").lower()
        if unit in ["k", "nghìn"]:
            prefs["budget"] = int(num * 1000)
        elif unit in ["tr", "triệu", "m"]:
            prefs["budget"] = int(num * 1_000_000)
        elif unit in ["vnd", ""]:
            prefs["budget"] = int(num)
    return prefs

# -------------------------
# MAIN: gọi LLM, parse, fallback, tìm data, in kết quả
# -------------------------
def understand_and_suggest(user_input: str):
    # 1) call LLM
    system_msg = (
        "You are a Vietnamese travel assistant for Ho Chi Minh City. "
        "From user input, extract JSON with keys: "
        "food (list of short strings), entertainment (list), area (string, e.g. 'quận 1'), "
        "vibe (string), budget (integer VND or null). "
        "Return only the JSON object and nothing else."
    )
    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a helpful AI travel assistant for Ho Chi Minh City."),
                UserMessage(system_msg + "\nUser: " + user_input),
            ],
            model=MODEL
        )
        llm_text = response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        llm_text = ""

    # 2) parse LLM output
    parsed = try_parse_json(llm_text)
    if parsed is None:
        # try a simpler parse: sometimes model returns plain keys like 'food: ...'
        parsed = try_parse_json(user_input)  # unlikely but harmless

    if parsed is None:
        # fallback: use rule-based heuristic
        print("⚠️ LLM output not parseable → dùng fallback rule-based extractor.")
        prefs = fallback_extract(user_input)
    else:
        # Normalize parsed to expected fields; accept many key names
        prefs = {
            "food": [],
            "entertainment": [],
            "vibe": "",
            "area": "",
            "budget": 0
        }
        # possible keys to try
        # food may be "food", "foods", "food_preferences", "food_preferences"
        for k in ("food", "foods", "food_preferences", "food_preference", "dish"):
            if k in parsed and parsed[k]:
                if isinstance(parsed[k], list):
                    prefs["food"] = [str(x).strip() for x in parsed[k] if x]
                else:
                    # comma split
                    prefs["food"] = [s.strip() for s in re.split(r"[;,/]|và|,| and ", str(parsed[k])) if s.strip()]
                break
        # entertainment
        for k in ("entertainment", "entertainment_preferences", "activities", "activity"):
            if k in parsed and parsed[k]:
                if isinstance(parsed[k], list):
                    prefs["entertainment"] = [str(x).strip() for x in parsed[k] if x]
                else:
                    prefs["entertainment"] = [s.strip() for s in re.split(r"[;,/]|và|,| and ", str(parsed[k])) if s.strip()]
                break
        # vibe
        for k in ("vibe", "mood", "atmosphere"):
            if k in parsed and parsed[k]:
                prefs["vibe"] = str(parsed[k]).strip()
                break
        # area
        for k in ("area", "location", "place", "district"):
            if k in parsed and parsed[k]:
                prefs["area"] = str(parsed[k]).strip()
                break
        # budget
        for k in ("budget", "budget_range", "price", "price_max"):
            if k in parsed and parsed[k]:
                try:
                    prefs["budget"] = int(parsed[k] or 0)
                except Exception:
                    # try parse numeric with suffix
                    btxt = str(parsed[k])
                    m = re.search(r"(\d+(\.\d+)?)", btxt.replace(",", ""))
                    if m:
                        v = float(m.group(1))
                        if "k" in btxt.lower():
                            prefs["budget"] = int(v * 1000)
                        elif "tr" in btxt.lower() or "triệu" in btxt.lower():
                            prefs["budget"] = int(v * 1_000_000)
                        else:
                            prefs["budget"] = int(v)
                break

    # Ensure lists exist
    prefs["food"] = prefs.get("food") or []
    prefs["entertainment"] = prefs.get("entertainment") or []
    # Print what we have
    print("\n💬 Parsed preferences (final):")
    print(json.dumps(prefs, ensure_ascii=False, indent=2))

    # 3) Use DataManager to find places
    # merge food + entertainment keywords for searching
    search_food = ", ".join(prefs["food"]) if prefs["food"] else None
    search_ent = ", ".join(prefs["entertainment"]) if prefs["entertainment"] else None

    results = DATA.find_places(
        food=search_food,
        vibe=prefs.get("vibe"),
        area=prefs.get("area"),
        budget=prefs.get("budget", 0),
        limit=8
    )

    if results is None or results.empty:
        # try relaxed search: search by any keyword across dataset
        print("\n🔎 Không có kết quả chính xác — thử mở rộng tìm kiếm theo từ khóa...")
        keywords = prefs["food"] + prefs["entertainment"]
        # if still empty, try searching name/category for each keyword
        agg = DATA.df.copy()
        if keywords:
            pattern = "|".join([re.escape(k.lower()) for k in keywords])
            agg_cols = []
            for c in ["name", "category_detail", "poi_type", "tags", "description", "address"]:
                if c in agg.columns:
                    agg_cols.append(c)
            if agg_cols:
                agg["__joint"] = agg[agg_cols].astype(str).agg(" | ".join, axis=1).str.lower()
                matched = agg[agg["__joint"].str.contains(pattern, na=False)]
                # apply budget if possible
                if prefs.get("budget") and "avg_cost" in matched.columns:
                    matched = matched[matched["avg_cost"] <= prefs["budget"] * 1.2]
                if not matched.empty:
                    matched = matched.sort_values(by=["recommendation_score"] if "recommendation_score" in matched.columns else [], ascending=False)
                    print("\n✨ Gợi ý (relaxed):")
                    print(matched.head(8)[["name", "category_detail", "avg_cost", "simulated_rating"]].to_string(index=False))
                    return
        print("\n😢 Không tìm thấy địa điểm phù hợp (even relaxed). Thử nói rõ hơn (ví dụ: 'quán cơm gà quận 3 tầm 150k').")
        return

    # 4) Print results (sorted by DataManager)
    print("\n✨ Gợi ý từ dữ liệu (từ gần nhất / điểm đề xuất):")
    # DataManager.find_places returns a DataFrame; print useful cols
    cols_to_show = [c for c in ["name", "category_detail", "avg_cost", "simulated_rating", "vibe", "address"] if c in results.columns]
    print(results[cols_to_show].to_string(index=False))

# -------------------------
# CLI loop
# -------------------------
def main():
    print("=== 💬 SMART TRAVEL ASSISTANT (HCM) ===")
    print("Gõ 'exit' để thoát.\n")
    while True:
        text = input("Bạn: ").strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "bye"):
            print("Tạm biệt!")
            break
        understand_and_suggest(text)

if __name__ == "__main__":
    main()
