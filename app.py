import httpx
import re
import base64
import os
import asyncio
import time
import gc
from datetime import datetime
from collections import deque
from PIL import Image
from io import BytesIO
from openai import AsyncOpenAI
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. 配置区域
# ==========================================
CONCURRENCY_LIMITER = asyncio.Semaphore(5)

DYNAMIC_BLOCK_CACHE = deque(maxlen=200)

SENSITIVE_WORDS = [
    "nsfw", "nude", "naked", "sex", "porn", "hentai", 
    "裸", "色情", "全裸", "blood", "kill", "murder", 
    "血腥", "尸体", "毒品", "杀"
]

# 精确的枚举表
RESOLUTION_MAP = {
    "1:1":  {"1k": (1024, 1024), "2k": (2048, 2048), "4k": (4096, 4096)},
    "16:9": {"1k": (1280, 720),  "2k": (2560, 1440), "4k": (3840, 2160)},
    "9:16": {"1k": (720, 1280),  "2k": (1440, 2560), "4k": (2160, 3840)},
    "4:3":  {"1k": (1152, 864),  "2k": (2048, 1536), "4k": (2880, 2160)},
    "3:4":  {"1k": (864, 1152),  "2k": (1536, 2048), "4k": (2160, 2880)},
    "21:9": {"1k": (1680, 720),  "2k": (3360, 1440), "4k": (5040, 2160)},
    "2:3":  {"1k": (852, 1280),  "2k": (1704, 2560), "4k": (2440, 3660)}
}

IP_REQUEST_HISTORY = {}
app_base_url = os.getenv("APP_BASE_URL", "").strip()

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
    if text in DYNAMIC_BLOCK_CACHE: return True, text
    text_lower = text.lower()
    for word in SENSITIVE_WORDS:
        if word and word.lower() in text_lower: return True, word
    return False, None

# ==========================================
# 3. 百度审核 & 图片处理逻辑
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
            token = resp.json().get("access_token")
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
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, data=data, headers=headers)
            result = resp.json()
            if result.get('conclusionType') == 2:
                msg = [i.get('msg') for i in result.get('data', [])]
                reason = ",".join(msg) if msg else "违规"
                if text not in DYNAMIC_BLOCK_CACHE:
                    DYNAMIC_BLOCK_CACHE.append(text)
                    print(f"🔒 [Cache] 新增违规: {text[:10]}...")
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
    format: str 
    init_image: Optional[str] = None  # 支持图生图

def process_image_in_memory(image_bytes: bytes, target_format: str, target_size: tuple = None) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            out = BytesIO()
            final_img = img
            status_msg = f"[原图直出] 尺寸: {img.size}"

            if target_size:
                w_diff = abs(img.size[0] - target_size[0])
                h_diff = abs(img.size[1] - target_size[1])
                if w_diff > 50 or h_diff > 50:
                    print(f"⚠️ [Resize] {img.size} -> {target_size}")
                    final_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    status_msg = f"[触发放大] -> {target_size}"
                else:
                    status_msg = f"[原图直出] 符合预期"
            print(f"✅ {status_msg}")

            fmt = "JPEG" if target_format == "jpg" else "PNG"
            final_img = final_img.convert("RGB") if target_format == "jpg" else final_img
            final_img.save(out, format=fmt, quality=95)
            return base64.b64encode(out.getvalue()).decode('utf-8')
    except Exception as e: 
        raise Exception(f"Image Error: {e}")
    finally: 
        gc.collect()

def sanitize_input_image(base64_str: str) -> str:
    """
    清洗上传的图片：
    1. 剥离 data URI 前缀
    2. 转为 RGB 模式（去除 Alpha 通道，解决 500 错误）
    3. 限制最大边长 1024px（解决超时问题）
    4. 返回纯 Base64 字符串
    """
    if not base64_str: return None
    
    if "base64," in base64_str:
        base64_data = base64_str.split("base64,")[1]
    else:
        base64_data = base64_str

    try:
        img_data = base64.b64decode(base64_data)
        with Image.open(BytesIO(img_data)) as img:
            # 强制转为 RGB，防止 PNG 透明通道导致 API 报错
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 缩放过大的图片，加速上传和处理
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"📉 [Input] 参考图已压缩至: {img.size}")

            out = BytesIO()
            img.save(out, format="JPEG", quality=85)
            return base64.b64encode(out.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 图片预处理失败，将尝试使用原始数据: {e}")
        return base64_data

async def download_image_as_bytes(url):
    print(f"📥 下载: {url}")
    async with httpx.AsyncClient(timeout=360) as client:
        r = await client.get(url)
        if r.status_code == 200: return r.content
        raise Exception(f"HTTP {r.status_code}")

async def core_generate(req: GenerateRequest):
    # 1. 纯查表逻辑：计算目标尺寸
    ratio_key = req.ratio.split(' ')[0]
    target_w, target_h = 1024, 1024
    if ratio_key in RESOLUTION_MAP:
        if req.scale in RESOLUTION_MAP[ratio_key]:
            target_w, target_h = RESOLUTION_MAP[ratio_key][req.scale]
        else:
            target_w, target_h = RESOLUTION_MAP[ratio_key]["1k"]
    api_size_str = f"{target_w}x{target_h}"
    
    # 2. Prompt 处理
    final_prompt = req.prompt
    if "gemini" in req.model.lower():
        quality_prompt = "standard quality"
        if req.scale == "4k": quality_prompt = "Extreme High Quality, 4K Resolution"
        elif req.scale == "2k": quality_prompt = "High Quality, 2K Resolution"
        suffix = f". (Settings: Aspect Ratio {ratio_key}, Quality {quality_prompt}, Target Size {api_size_str})"
        final_prompt = f"{req.prompt} {suffix}"
    
    # 3. 预处理参考图（清洗 Base64，限制尺寸并转为 RGB）
    clean_init_image = None
    if req.init_image:
        print("🧹 正在预处理参考图...")
        clean_init_image = sanitize_input_image(req.init_image)

    # 4. 初始化 OpenAI 客户端
    client = AsyncOpenAI(api_key=req.api_key, base_url=f"{app_base_url}/v1", max_retries=0, timeout=360.0)
    img_b = None

    # --- 分支 A: Gemini 模型 (使用 Chat Completion 多模态) ---
    if "gemini" in req.model.lower():
        print(f"🟣 Gemini: {req.model}")
        content_parts = [{"type": "text", "text": final_prompt}]
        if clean_init_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{clean_init_image}"}
            })
        
        res = await client.chat.completions.create(
            model=req.model, 
            messages=[{"role":"user","content": content_parts}], 
            extra_body={"modalities":["image","text"]},
            timeout=360.0
        )
        c = res.choices[0].message.content
        u = re.search(r"!\[.*?\]\((https?://[^\)]+)\)", c) or re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", c)
        if u: img_b = await download_image_as_bytes(u.group(1))
        elif "base64," in c: img_b = base64.b64decode(c.split("base64,")[1].split(")")[0].strip())
        else: raise Exception("Gemini 返回中未找到图片数据")

    # --- 分支 B: 非 Gemini 模型 (Nano Banana 等) ---
    else:
        # B1: 如果有参考图，采用 edit 接口进行图生图
        if clean_init_image:
            print(f"🔵 [Edit 接口] 图生图模式: {req.model}")
            
            # 将清洗后的 Base64 转回二进制流
            image_data = base64.b64decode(clean_init_image)
            image_file = BytesIO(image_data)
            image_file.name = "init_image.jpg"  # 必须提供文件名，部分 SDK 内部校验需要

            try:
                # 调用 edit 接口
                res = await client.images.edit(
                    model=req.model,
                    image=image_file,
                    prompt=final_prompt,
                    n=1,
                    size=api_size_str,
                    response_format="b64_json",
                    extra_body={"strength": 0.75} # 图生图通常需要重绘强度参数
                )
            except Exception as e:
                # 如果 edit 接口报 404 或不支持，尝试回退到普通的 generate 强行传递
                print(f"⚠️ Edit 接口调用失败 ({e})，尝试使用 Generations 兼容模式...")
                res = await client.images.generate(
                    model=req.model,
                    prompt=final_prompt,
                    size=api_size_str,
                    response_format="b64_json",
                    extra_body={"image": clean_init_image, "strength": 0.75}
                )

        # B2: 如果没有参考图，采用普通的 generate 接口进行文生图
        else:
            print(f"🔵 [Generate 接口] 文生图模式: {req.model}")
            res = await client.images.generate(
                model=req.model,
                prompt=final_prompt,
                n=1,
                size=api_size_str,
                response_format="b64_json"
            )

        # 处理返回的 Image 对象
        d = res.data[0]
        if getattr(d, 'b64_json', None):
            img_b = base64.b64decode(d.b64_json)
        elif hasattr(d, 'url') and d.url:
            img_b = await download_image_as_bytes(d.url)
        else:
            raise Exception("API 未返回有效的图片数据")

    # 5. 最后进行图片后处理（缩放、格式转换等）
    return process_image_in_memory(img_b, req.format, target_size=(target_w, target_h))

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
    
    # --- 🔒 API Key 脱敏处理 ---
    key_suffix = req.api_key[-8:] if len(req.api_key) >= 8 else req.api_key
    masked_key = f"******{key_suffix}"
    
    # --- 📝 增强日志记录 ---
    timestamp = datetime.now().strftime('%H:%M:%S')
    mode = "Img2Img" if req.init_image else "Txt2Img"
    print("\n" + "="*60)
    print(f"🚀 [Req] {timestamp} | IP: {ip}")
    print(f"🔑 Key: {masked_key}")
    print(f"📌 Model: {req.model} | Mode: {mode} | Scale: {req.scale} | Ratio: {req.ratio}")
    print("-" * 60)
    print(f"💡 Prompt (Full):")
    print(req.prompt)
    print("="*60 + "\n")

    # 1. IP 限流
    if check_ip_rate_limit(ip, 3, 60):
        print(f"⛔ [Rate-Limit] IP {ip} 请求过快")
        return JSONResponse({"status": "error", "message": "请求过快"}, 429)

    # 2. 本地敏感词拦截
    is_loc, word = contains_local_sensitive_words(req.prompt)
    if is_loc:
        print(f"🚫 [Local] 拦截: {word}")
        return JSONResponse({"status": "error", "message": f"违规内容: {word}"}, 200)

    # 3. 百度 API 检查
    is_bd, reason = await check_baidu_text_censor(req.prompt)
    if is_bd:
        print(f"🚫 [Baidu] 拦截: {reason}")
        return JSONResponse({"status": "error", "message": f"审核未通过: {reason}"}, 200)

    # 4. 执行生成
    async with CONCURRENCY_LIMITER:
        try:
            start_time = time.time()
            b64 = await core_generate(req)
            print(f"✅ [Success] 耗时: {time.time()-start_time:.2f}s")
            return {"status": "success", "image_base64": b64}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ [Error] {e}")
            return JSONResponse({"status": "error", "message": str(e)}, 200)
        finally:
            gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)