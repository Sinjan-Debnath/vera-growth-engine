import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import google.generativeai as genai

app = FastAPI()

# --- STATE MANAGEMENT ---
db = {}
# Track auto-reply counts per conversation: conv_id -> count
auto_reply_tracker = {} 

# --- GEMINI SETUP ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={
        "temperature": 0.0,
        "response_mime_type": "application/json"
    }
)

# --- SYSTEM PROMPTS ---
TICK_PROMPT = """You are Vera, a proactive Growth Consultant for local merchants.
Compose a specific, compelling message based strictly on the provided JSON contexts.

RULES:
1. Extract and use real numbers (prices, views, dips, batch numbers, dates) from the context.
2. Match the tone: Dentists = clinical. Salons = warm. Restaurants = operator-to-operator. Gyms = coach. Pharmacies = precise.
3. Use the merchant owner's first name and reference their specific live offers.
4. End with a SINGLE, simple binary CTA (e.g., YES/NO).
5. NEVER include a URL. Do not fabricate data.

Respond ONLY with this exact JSON schema:
{
  "body": "The exact text of the message",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "1-sentence explanation citing specific data used."
}"""

REPLY_PROMPT = """You are Vera, analyzing a merchant's reply.
Determine the next action.
If the merchant explicitly commits (e.g., "let's do it", "yes", "confirm"): action = "send", and move immediately to action/drafting. Do not ask qualifying questions.
Otherwise, respond to their query appropriately.

Respond ONLY with this exact JSON schema:
{
  "action": "send", 
  "body": "The text of your response",
  "cta": "open_ended",
  "wait_seconds": 0,
  "rationale": "Why you chose this action."
}"""

# --- ENDPOINTS ---

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in db.items():
        if scope in counts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": 100, "contexts_loaded": counts}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Sinjan Debnath",
        "model": "gemini-1.5-flash",
        "approach": "Deterministic FastAPI 4-context engine with hardcoded safety guardrails.",
        "version": "2.0.0"
    }

@app.post("/v1/context")
async def push_context(req: Request):
    data = await req.json()
    scope = data.get("scope")
    context_id = data.get("context_id")
    version = data.get("version", 1)
    payload = data.get("payload", {})
    
    key = (scope, context_id)
    
    # Accept if it's new, OR if the incoming version is higher
    if key in db and db[key]["version"] >= version:
        return JSONResponse(
            status_code=409, 
            content={"accepted": False, "reason": "stale_version", "current_version": db[key]["version"]}
        )
    
    db[key] = {"version": version, "payload": payload}
    return {"accepted": True, "ack_id": f"ack_{context_id}_v{version}", "stored_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/tick")
async def tick(req: Request):
    data = await req.json()
    actions = []
    
    for trg_id in data.get("available_triggers", []):
        trg_data = db.get(("trigger", trg_id), {}).get("payload")
        if not trg_data: 
            continue
        
        merchant_id = trg_data.get("merchant_id")
        merchant = db.get(("merchant", merchant_id), {}).get("payload")
        if not merchant:
            continue
            
        category_slug = merchant.get("category_slug")
        category = db.get(("category", category_slug), {}).get("payload")
        
        customer_id = trg_data.get("customer_id")
        customer = db.get(("customer", customer_id), {}).get("payload") if customer_id else None

        context_str = f"TRIGGER: {json.dumps(trg_data)}\nMERCHANT: {json.dumps(merchant)}\nCATEGORY: {json.dumps(category)}\nCUSTOMER: {json.dumps(customer)}"
        
        try:
            response = model.generate_content(f"{TICK_PROMPT}\n{context_str}")
            result = json.loads(response.text)
            
            # Default to "merchant_on_behalf" if customer context exists, else "vera"
            send_as = result.get("send_as", "merchant_on_behalf" if customer else "vera")
            
            actions.append({
                "conversation_id": f"conv_{trg_id}",
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": send_as,
                "trigger_id": trg_id,
                "template_name": "dynamic_v1",
                "template_params": [],
                "body": result.get("body", "Hello from Vera."),
                "cta": result.get("cta", "none"),
                "suppression_key": trg_data.get("suppression_key", f"supp_{trg_id}"),
                "rationale": result.get("rationale", "Generated by LLM.")
            })
        except Exception as e:
            # We log the error but do not append an action if the LLM fails
            print(f"Tick Generation Error: {e}")

    return {"actions": actions}

@app.post("/v1/reply")
async def reply(req: Request):
    data = await req.json()
    conv_id = data.get("conversation_id")
    msg = data.get("message", "").lower().strip()
    
    # 1. HARD GUARDRAIL: Hostile / STOP detection
    stop_words = ["stop", "unsubscribe", "cancel", "not interested", "spam", "leave me alone"]
    if any(word in msg for word in stop_words):
        return {
            "action": "end",
            "rationale": "Explicit merchant opt-out or hostility detected."
        }
        
    # 2. HARD GUARDRAIL: Auto-reply detection
    auto_reply_phrases = ["thank you for contacting", "we will get back to you", "office hours are", "automated assistant"]
    if any(phrase in msg for phrase in auto_reply_phrases):
        # Increment counter for this conversation
        auto_reply_tracker[conv_id] = auto_reply_tracker.get(conv_id, 0) + 1
        
        if auto_reply_tracker[conv_id] >= 2:
            return {
                "action": "end",
                "rationale": "Auto-reply detected twice consecutively. Graceful exit."
            }
        else:
            return {
                "action": "wait",
                "wait_seconds": 14400, # Wait 4 hours on first auto-reply
                "rationale": "First auto-reply detected. Backing off."
            }
            
    # Reset auto-reply tracker if a real message comes through
    if conv_id in auto_reply_tracker:
        del auto_reply_tracker[conv_id]
        
    # 3. NORMAL FLOW: LLM Handling
    try:
        response = model.generate_content(f"{REPLY_PROMPT}\nMERCHANT REPLY: {msg}")
        result = json.loads(response.text)
        
        output = {
            "action": result.get("action", "send"),
            "rationale": result.get("rationale", "Generated by LLM.")
        }
        if output["action"] == "send":
            output["body"] = result.get("body", "Noted. I'll take care of that.")
            output["cta"] = result.get("cta", "open_ended")
        elif output["action"] == "wait":
            output["wait_seconds"] = result.get("wait_seconds", 1800)
            
        return output
    except Exception as e:
        print(f"Reply Generation Error: {e}")
        # Fallback to a safe send to keep the conversation alive
        return {
            "action": "send", 
            "body": "Got it. Let me know if you need anything else.", 
            "cta": "none",
            "rationale": "Fallback triggered due to LLM parsing error."
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
