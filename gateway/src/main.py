import os
import logging

# FastAPI本体
from fastapi import FastAPI, Request

# レスポンス型（HTML / JSON / ストリーミング）
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# vLLMバックエンド
from .backends.vllm_client import VLLMBackend

# モデル一覧を見て model を自動補正する（新規追加）
from .backends.model_registry import ModelRegistry

# 非ストリーム時にvLLMへ普通にPOSTするために使う（streamはbackendで中継）
import httpx


# ログ設定（encoding='utf-8' はWindowsで文字化けしにくくする意図）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# FastAPIアプリを作る
app = FastAPI()

# バックエンド（今はvLLM固定。後でTRT-LLMに差し替え可能）
backend = VLLMBackend()

# モデル解決（DEFAULT_MODELやクライアント指定がズレても404にしないため）
# ttlやリトライは環境変数で調整可能にしておく（必要なければ固定でもOK）
model_registry = ModelRegistry(
    ttl_sec=float(os.getenv("MODEL_REFRESH_TTL", "60")),
    retries=int(os.getenv("MODEL_FETCH_RETRIES", "10")),
    retry_delay_sec=float(os.getenv("MODEL_FETCH_RETRY_DELAY", "1.0")),
)


# UI（静的HTML）のパス。src/static/index.html を返すだけ
INDEX_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")

# アプリ全体で使い回すHTTPクライアント（/v1/models取得と非stream POSTに流用する）
http_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def on_startup():
    """起動時にHTTPクライアントを作って、可能ならモデル一覧を先読みしておく"""
    global http_client
    http_client = httpx.AsyncClient(timeout=None)

    # vLLMがまだ起動していない可能性があるので、warmupは失敗しても落とさない設計
    try:
        await model_registry.warmup(http_client)
    except Exception as e:
        logger.warning(f"Model warmup failed (ignored): {e}")


@app.on_event("shutdown")
async def on_shutdown():
    """終了時にHTTPクライアントを閉じる"""
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """デバッグ用UIを返す（本番では消してもOK）"""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI互換のチャットAPI。
    受け取ったJSONを（必要なら補正して）vLLMへ中継する。
    """

    # クライアントが送ったJSONをそのまま辞書にする
    payload = await request.json()

    # modelは「DEFAULT_MODEL or クライアント指定」だが、
    # vLLMの /v1/models 一覧を見て「存在するID」に自動補正する
    requested_model = payload.get("model")
    default_model = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # http_clientはstartupで作っている想定（念のためNoneなら作る）
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=None)

    resolved = await model_registry.resolve_model(requested_model, default_model, http_client)
    if resolved:
        payload["model"] = resolved
    else:
        # どうしても決められない（モデル一覧が取れず、requested/defaultも空）なら、従来通りsetdefault
        payload.setdefault("model", default_model)

    # ここは「企業のgatewayでよくやる場所」：
    # 例）APIキー検証、利用者ごとのモデル制限、ログ、レート制限、監査対応など
    #
    # 今回は「日本語強制」を例として入れる（不要ならFORCE_JA=0でOFF）
    if os.getenv("FORCE_JA", "1") == "1":
        msgs = payload.get("messages") or []
        # 先頭がsystemでないならsystemを挿入する（上書きしたい場合は別ロジックに）
        if not (len(msgs) > 0 and msgs[0].get("role") == "system"):
            msgs = [{"role": "system", "content": "必ず日本語で回答してください。"}] + msgs
        payload["messages"] = msgs

    # stream=trueならSSEで返す（OpenAI互換）
    stream = bool(payload.get("stream", False))

    # vLLM側も同じパスなので、pathを固定で使う
    path = "/v1/chat/completions"

    if stream:
        # streamingの場合は「SSEのbytes」をそのまま返す
        async def gen():
            # backendはbytesチャンクをyieldする
            async for chunk in backend.stream_openai_sse(path=path, payload=payload):
                yield chunk

        # StreamingResponseでSSEとして返す
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",  # SSEのMIMEタイプ
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx等のバッファ抑制（挟まないなら実質不要）
            },
        )

    # 非streamの場合：通常のJSONレスポンスとして返す
    # backendを拡張してもいいが、まずは分かりやすく直でPOSTする
    vllm_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000").rstrip("/") + path

    # 念のため None なら作る（startup未完了/例外などの保険）
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=None)

    r = await http_client.post(
        vllm_url,
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    # もし「model not found 404」なら、モデル一覧を取り直して置換→再試行（最大1回）
    if r.status_code == 404:
        try:
            resp_json = r.json()
        except Exception:
            resp_json = None

        new_model = await model_registry.maybe_retry_on_model_404(
            resp_status=r.status_code,
            resp_json=resp_json,
            requested=requested_model,
            default_model=default_model,
            client=http_client,
        )
        if new_model and new_model != payload.get("model"):
            payload["model"] = new_model
            r = await http_client.post(
                vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

    

    # vLLMが返したJSONをそのまま返す（OpenAI互換）
    try:
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse(content={"raw": r.text}, status_code=r.status_code)