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
conversations = {} 

# --- GEMINI SETUP ---
# Ensure you set the GEMINI_API_KEY environment variable in Render
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Using gemini-1.5-flash for speed. Enforcing JSON output and 0 temperature.
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={
        "temperature": 0.0,
        "response_mime_type": "application/json"
    }
)

# --- SYSTEM PROMPTS ---
TICK_PROMPT = """You are Vera, a proactive Growth Consultant for local merchants.
You must compose a highly specific, compelling message based strictly on the provided JSON contexts.

SCORING RULES (CRITICAL):
1. SPECIFICITY: You MUST extract and use real numbers (prices, views, dips, batch numbers, JIDA pages) from the context. Do not round or invent numbers.
2. CATEGORY FIT: Match the tone. Dentists = clinical/peer. Salons = warm/practical. Restaurants = busy/operator.
3. MERCHANT FIT: Use the owner's first name. Reference their specific live offers (Service + Price).
4. ENGAGEMENT: Use loss aversion, curiosity, or reciprocity. End with a SINGLE, simple YES/NO or binary Call to Action (CTA).
5. NO URLS: Never include a link.
6. NO FABRICATION: Do not cite papers, competitors, or offers not in the text.

Respond ONLY with this exact JSON schema:
{
  "body": "The WhatsApp message text",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "1-sentence explanation of your strategy citing specific data used."
}

--- USER CONTEXT DATA BELOW ---
"""

REPLY_PROMPT = """You are Vera, analyzing a merchant's reply.
Determine the next action based on these rules:
1. If the message looks like a canned auto-reply (e.g., "Thank you for contacting..."): action = "wait", wait_seconds = 14400.
2. If the merchant is hostile, says "stop", or opts out: action = "end".
3. If the merchant explicitly commits (e.g., "let's do it", "yes"): action = "send", and DO NOT ask more qualifying questions. Move immediately to action/drafting.

Respond ONLY with this exact JSON schema:
{
  "action": "send", 
  "body": "Your response (leave empty if wait or end)",
  "cta": "open_ended",
  "wait_seconds": 0,
  "rationale": "Why you chose this action."
}

--- MERCHANT REPLY BELOW ---
"""

# --- ENDPOINTS ---

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in db.items():
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": 100, "contexts_loaded": counts}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Sinjan Debnath",
        "model": "gemini-1.5-flash",
        "approach": "Deterministic FastAPI 4-context engine using Gemini JSON mode",
        "version": "1.0.0"
    }

@app.post("/v1/context")
async def push_context(req: Request):
    data = await req.json()
    key = (data["scope"], data["context_id"])
    
    if key in db and db[key]["version"] >= data["version"]:
        return JSONResponse(
            status_code=409, 
            content={"accepted": False, "reason": "stale_version", "current_version": db[key]["version"]}
        )
    
    db[key] = {"version": data["version"], "payload": data["payload"]}
    return {"accepted": True, "ack_id": f"ack_{data['context_id']}_v{data['version']}", "stored_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/tick")
async def tick(req: Request):
    data = await req.json()
    actions = []
    
    for trg_id in data.get("available_triggers", []):
        trg_data = db.get(("trigger", trg_id), {}).get("payload")
        if not trg_data: continue
        
        merchant = db.get(("merchant", trg_data.get("merchant_id")), {}).get("payload")
        category = db.get(("category", merchant.get("category_slug")), {}).get("payload")
        customer = db.get(("customer", trg_data.get("customer_id")), {}).get("payload") if trg_data.get("customer_id") else None

        # Build Context String for LLM
        context_str = f"TRIGGER: {json.dumps(trg_data)}\nMERCHANT: {json.dumps(merchant)}\nCATEGORY: {json.dumps(category)}\nCUSTOMER: {json.dumps(customer)}"
        
        try:
            # Trigger Gemini
            response = model.generate_content(f"{TICK_PROMPT}\n{context_str}")
            result = json.loads(response.text)
            
            # Send_as logic based on scope
            send_as = "merchant_on_behalf" if trg_data.get("scope") == "customer" else "vera"
            
            actions.append({
                "conversation_id": f"conv_{trg_id}",
                "merchant_id": merchant["merchant_id"],
                "customer_id": customer["customer_id"] if customer else None,
                "send_as": result.get("send_as", send_as),
                "trigger_id": trg_id,
                "template_name": "dynamic_v1",
                "template_params": [],
                "body": result["body"],
                "cta": result["cta"],
                "suppression_key": trg_data.get("suppression_key", f"supp_{trg_id}"),
                "rationale": result["rationale"]
            })
        except Exception as e:
            print(f"LLM Error on Tick: {e}")
            pass

    return {"actions": actions}

@app.post("/v1/reply")
async def reply(req: Request):
    data = await req.json()
    conv_id = data["conversation_id"]
    msg = data["message"]
    
    conversations.setdefault(conv_id, []).append(msg)
    
    try:
        response = model.generate_content(f"{REPLY_PROMPT}\n{msg}")
        result = json.loads(response.text)
        
        output = {
            "action": result.get("action", "wait"),
            "rationale": result.get("rationale", "")
        }
        if output["action"] == "send":
            output["body"] = result.get("body", "")
            output["cta"] = result.get("cta", "open_ended")
        elif output["action"] == "wait":
            output["wait_seconds"] = result.get("wait_seconds", 14400)
            
        return output
    except Exception:
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Fallback wait due to error."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))