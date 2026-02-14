from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import os
from huggingface_hub import login
import torch
import logging
import asyncio

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 環境変数からトークンを取得
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

app = FastAPI()

# Jinja2 テンプレートの設定
templates = Jinja2Templates(directory="templates")

# モデルとトークナイザーのロード
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
CACHE_DIR = "/models"

# 4-bit量子化の設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_quant_type="nf4",  # または "fp4"
    llm_int8_enable_fp32_cpu_offload = True
    # 「llm_int8_enable_fp32_cpu_offload」はコメントアウト推奨（4bitとは競合しやすい）
)

logger.info("トークナイザーをロード中...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)
logger.info("トークナイザーのロードが完了しました。")

logger.info("モデルをロード中...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # max_memory = {
    #     0: "24GiB",    # 0番目のGPU
    #     "cpu": "64GiB" # CPU
    # }
)
logger.info("モデルのロードが完了しました。")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    logger.info("フォームページをレンダリングしています...")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate_stream/")
async def generate_stream(prompt: str):
    logger.info(f"GET /generate_stream/ エンドポイントにプロンプトが送信されました: {prompt}")

    async def token_stream():
        try:
            # 入力をトークナイズし、GPUに移動 (GPUがない環境なら "cpu" に変更)
            logger.info("プロンプトをトークナイズしています...")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            logger.info("トークナイズ完了。モデルに入力を送信します。")

            # 非同期でテキストを生成
            logger.info("モデルによるテキスト生成を開始します...")
            outputs = await asyncio.to_thread(model.generate, **inputs, max_new_tokens=128)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("テキスト生成が完了しました。")

            # 生成されたテキストをストリームとして送信
            if generated_text:
                logger.info(f"生成されたテキスト: {generated_text}")
                yield f"data: {generated_text}\n\n"

            # ストリーム終了通知
            yield "data: [DONE]\n\n"
            logger.info("ストリーミングレスポンスを完了しました。")

        except Exception as e:
            logger.error(f"テキスト生成中にエラーが発生しました: {e}", exc_info=True)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")

