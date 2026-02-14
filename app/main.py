from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os
import logging
import asyncio
import queue

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding='utf-8')
logger = logging.getLogger(__name__)

app = FastAPI()

# Jinja2 テンプレートの設定
templates = Jinja2Templates(directory="templates")

# モデル設定
MODEL_REPO = \
"bartowski/Llama-3.3-70B-Instruct-GGUF"
# "mmnga/tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.3-gguf"
MODEL_FILENAME = \
"Llama-3.3-70B-Instruct-Q4_K_M.gguf"
# "tokyotech-llm-Llama-3.1-Swallow-70B-Instruct-v0.3-IQ2_S.gguf" 
CACHE_DIR = "/models"
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_FILENAME)

# モデルのダウンロードまたはスキップ
if not os.path.exists(MODEL_PATH):
    logger.info(f"Model not found. Downloading {MODEL_FILENAME}...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=CACHE_DIR
    )
    logger.info(f"Model downloaded successfully: {model_path}")
else:
    logger.info(f"Model already exists at {MODEL_PATH}. Skipping download.")
    model_path = MODEL_PATH

# モデルのロード
try:
    logger.info("Loading model...")
    llm = Llama(model_path=model_path, n_gpu_layers=-1)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    logger.info("Rendering the form page...")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate_stream/")
async def generate_stream(prompt: str):
    logger.info(f"Received prompt via GET: {prompt}")

    async def token_stream():
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "あなたは有能で誠実な日本語アシスタントです。"
                        "必ず日本語で回答してください。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            # llama-cpp-python の stream generator は同期なので、別スレッドで回して逐次取り出す
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def producer():
                try:
                    for chunk in llm.create_chat_completion(
                        messages=messages,
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=256,
                        stream=True,
                    ):
                        # chunk例: {'choices':[{'delta':{'content':'...'}}], ...}
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # 終了合図

            # producerを別スレッドで開始
            asyncio.create_task(asyncio.to_thread(producer))

            # queueから逐次取り出してSSEで返す
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                # テキスト断片の取り出し（delta形式）
                choice = item.get("choices", [{}])[0]

                delta = choice.get("delta", {}).get("content", "")
                if not delta:
                    # バージョン差でこちらに出ることがある
                    delta = choice.get("text", "")

                if delta:
                    yield f"data: {delta}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx等を挟む時のバッファ抑制（いまは無くてもOK）
            "X-Accel-Buffering": "no",
        },
    )
