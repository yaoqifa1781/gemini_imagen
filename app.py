import httpx
import re
import base64
import os
import asyncio
import time
import gc
from datetime import datetime
from collections import deque  # 引入双端队列
from PIL import Image
from io import BytesIO
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. 配置区域
# ==========================================
CONCURRENCY_LIMITER = asyncio.Semaphore(3)

# --- 🚀 动态违禁词缓存 (核心修改) ---
# maxlen=200: 自动保持最新的200个，旧的自动丢弃
DYNAMIC_BLOCK_CACHE = deque(maxlen=200)

# 硬编码的基础高危词 (保留最基本的底线，防止API挂了时裸奔)
SENSITIVE_WORDS = [
    "nsfw", "nude", "naked", "sex", "porn", "hentai", 
    "裸", "色情", "全裸", "blood", "kill", "murder", 
    "血腥", "尸体", "毒品", "杀"
]

AR_BASES = {
    "1:1 (SDXL标准 1024)": (1024, 1024),
    "16:9 (SDXL标准 768p)": (1344, 768),
    "16:9 (1080p Full HD)": (1920, 1080),
    "9:16 (手机竖屏)": (768, 1344),
    "4:3 (标准)":  (1152, 896),
    "21:9 (宽屏)": (1536, 640)
}
SCALE_MULTIPLIERS = {"1k": 1.0, "2k": 1.5, "4k": 2.0}

# IP 限流记录
IP_REQUEST_HISTORY = {}

# ==========================================
# 2. 辅助功能
# ==========================================

def get_real_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def check_ip_rate_limit(ip: str, limit: int = 5, window: int = 60):
    now = time.time()
    history = IP_REQUEST_HISTORY.get(ip, [])
    valid_history = [t for t in history if now - t < window]
    if len(valid_history) >= limit:
        IP_REQUEST_HISTORY[ip] = valid_history
        return True
    valid_history.append(now)
    IP_REQUEST_HISTORY[ip] = valid_history
    if len(IP_REQUEST_HISTORY) > 5000: IP_REQUEST_HISTORY.clear()
    return False

def contains_local_sensitive_words(text: str):
    """
    检查本地缓存
    1. 检查最近被百度封杀的200个Prompt (精确匹配)
    2. 检查硬编码的关键词 (模糊匹配)
    """
    # 1. 检查动态缓存 (O(n) 遍历，但n最大200，极快)
    if text in DYNAMIC_BLOCK_CACHE:
        return True, "最近违规记录 (已缓存)"

    # 2. 检查静态关键词
    text_lower = text.lower()
    for word in SENSITIVE_WORDS:
        if word and word.lower() in text_lower:
            return True, word
            
    return False, None

# ==========================================
# 3. 百度审核 & 生成
# ==========================================

_BAIDU_TOKEN_CACHE = {"token": None, "expires_at": 0}

async def get_baidu_access_token():
    ak = os.getenv("BAIDU_API_KEY", "").strip()
    sk = os.getenv("BAIDU_SECRET_KEY", "").strip()
    if not ak or not sk: return None
    now = time.time()
    if _BAIDU_TOKEN_CACHE["token"] and now < _BAIDU_TOKEN_CACHE["expires_at"]:
        return _BAIDU_TOKEN_CACHE["token"]
    
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": ak, "client_secret": sk}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, params=params)
            data = resp.json()
            token = data.get("access_token")
            if token:
                _BAIDU_TOKEN_CACHE["token"] = token
                _BAIDU_TOKEN_CACHE["expires_at"] = now + 1728000
                return token
    except: return None

async def check_baidu_text_censor(text):
    token = await get_baidu_access_token()
    if not token: return False, None

    url = f"https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined?access_token={token}"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'text': text}
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, data=data, headers=headers)
            result = resp.json()
            
            if result.get('conclusionType') == 2:
                msg = [i.get('msg') for i in result.get('data', [])]
                reason = ",".join(msg) if msg else "违规"
                
                # 🚀 发现新词：存入内存队列
                if text not in DYNAMIC_BLOCK_CACHE:
                    DYNAMIC_BLOCK_CACHE.append(text)
                    print(f"🔒 [Cache] 缓存违规Prompt ({len(DYNAMIC_BLOCK_CACHE)}/200): {text[:15]}...")
                
                return True, reason
            elif result.get('conclusionType') == 3: return True, "疑似违规"
            return False, None
    except: return False, None

class GenerateRequest(BaseModel):
    api_key: str
    model: str
    prompt: str
    ratio: str
    scale: str
    custom_size: str
    format: str

def process_image_in_memory(image_bytes: bytes, target_format: str) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            out = BytesIO()
            fmt = "JPEG" if target_format == "jpg" else "PNG"
            img = img.convert("RGB") if target_format == "jpg" else img
            img.save(out, format=fmt, quality=95)
            return base64.b64encode(out.getvalue()).decode('utf-8')
    except Exception as e: raise Exception(f"IMG Error: {e}")
    finally: gc.collect()

async def download_image_as_bytes(url):
    print(f"📥 下载: {url}")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        if r.status_code == 200: return r.content
        raise Exception(f"HTTP {r.status_code}")

async def core_generate(req: GenerateRequest):
    fs = "1024x1024"
    if req.ratio == "custom": fs = req.custom_size.strip()
    else:
        w, h = AR_BASES.get(req.ratio, (1024, 1024))
        m = SCALE_MULTIPLIERS.get(req.scale, 1.0)
        fs = f"{int(w*m)}x{int(h*m)}"
    
    fp = req.prompt
    if "gemini" in req.model.lower():
        suffix = f"--resolution {fs}" if req.ratio=="custom" else f"--ar {req.ratio.split()[0]} --resolution {fs}"
        fp = f"{req.prompt} {suffix}"

    client = AsyncOpenAI(api_key=req.api_key, base_url="https://api.cloudapp.ink/v1", max_retries=0, timeout=60.0)

    img_b = None
    if "gemini" in req.model.lower():
        print(f"🟣 Gemini: {req.model}")
        res = await client.chat.completions.create(model=req.model, messages=[{"role":"user","content":fp}], extra_body={"modalities":["image","text"]})
        c = res.choices[0].message.content
        u = re.search(r"!\[.*?\]\((https?://[^\)]+)\)", c) or re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", c)
        if u: img_b = await download_image_as_bytes(u.group(1))
        elif "base64," in c: img_b = base64.b64decode(c.split("base64,")[1].split(")")[0].strip())
        else: raise Exception("No Image")
    else:
        print(f"🔵 Standard: {req.model}")
        res = await client.images.generate(model=req.model, prompt=fp, n=1, size=fs, response_format="b64_json")
        d = res.data[0]
        if getattr(d,'b64_json',None): img_b = base64.b64decode(d.b64_json)
        elif hasattr(d,'url'): img_b = await download_image_as_bytes(d.url)
        else: raise Exception("No Data")
        
    return process_image_in_memory(img_b, req.format)

# ==========================================
# 4. FastAPI App
# ==========================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index(): return FileResponse('static/index.html')

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/generate")
async def generate_api(req: GenerateRequest, request: Request):
    ip = get_real_ip(request)
    print(f"\n🚀 [{datetime.now().strftime('%H:%M:%S')}] IP: {ip} | Prompt: {req.prompt[:20]}...")

    if check_ip_rate_limit(ip, 5, 60):
        return JSONResponse({"status": "error", "message": "请求过快"}, 429)

    is_loc, word = contains_local_sensitive_words(req.prompt)
    if is_loc:
        print(f"🚫 [Local] 拦截: {word}")
        return JSONResponse({"status": "error", "message": f"违规内容: {word}"}, 200)

    is_bd, reason = await check_baidu_text_censor(req.prompt)
    if is_bd:
        print(f"🚫 [Baidu] 拦截: {reason}")
        return JSONResponse({"status": "error", "message": f"审核未通过: {reason}"}, 200)

    async with CONCURRENCY_LIMITER:
        try:
            b64 = await core_generate(req)
            return {"status": "success", "image_base64": b64}
        except Exception as e:
            print(f"❌ Error: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, 200)
        finally:
            gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)