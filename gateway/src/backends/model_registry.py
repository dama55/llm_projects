# gateway/src/backends/model_registry.py

from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx


class ModelRegistry:
    """
    vLLM(OpenAI互換)の /v1/models を見て、
    - 利用可能なモデルID一覧をキャッシュ
    - リクエストの model を「存在するID」に自動補正
    するための小さなヘルパ。

    目的：
    - DEFAULT_MODEL がズレていても 404 を避ける
    - クライアント指定 model が存在しない場合も自動で置換する
    - vLLMがまだ起動してない等で /v1/models が取れない場合でも落ちない
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        ttl_sec: float = 60.0,
        retries: int = 10,
        retry_delay_sec: float = 1.0,
    ) -> None:
        # vLLMのベースURL（docker-composeで VLLM_BASE_URL=http://vllm:8000 を渡す想定）
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://vllm:8000")).rstrip("/")
        # モデル一覧キャッシュのTTL（秒）
        self.ttl_sec = ttl_sec
        # startup時や再取得時のリトライ回数（vLLM起動待ちを想定）
        self.retries = retries
        self.retry_delay_sec = retry_delay_sec

        # キャッシュ本体
        self._models: List[str] = []
        self._last_fetch: float = 0.0

        # 多重fetch防止（同時に何リクエストも来た時に /v1/models を連打しない）
        self._lock = asyncio.Lock()

    # ---- 公開API ----

    async def warmup(self, client: httpx.AsyncClient) -> None:
        """
        起動時に呼んで /v1/models を取っておく（失敗してもOK）。
        取得できない時は空のまま。
        """
        await self.refresh(client, force=True)

    async def resolve_model(self, requested: Optional[str], default_model: Optional[str], client: httpx.AsyncClient) -> str:
        """
        requested / default_model / models[0] の順で解決する。
        - modelsが取れていれば、その中にあるものを採用
        - modelsが取れない場合は、requested→default→"" の順で返す（従来挙動に近い）
        """
        # TTLが切れていたら軽く更新を試みる（失敗しても落とさない）
        await self.refresh(client, force=False)

        models = list(self._models)  # スナップショット

        candidates: List[str] = []
        if requested:
            candidates.append(requested)
        if default_model:
            candidates.append(default_model)

        # 1) 候補が models に含まれていればそれ
        for c in candidates:
            if c in models:
                return c

        # 2) models が取れていれば先頭にフォールバック
        if models:
            return models[0]

        # 3) modelsが取れないなら：requested → default → ""（最後は空文字）
        if requested:
            return requested
        if default_model:
            return default_model
        return ""

    async def maybe_retry_on_model_404(
        self,
        resp_status: int,
        resp_json: Optional[Dict[str, Any]],
        requested: Optional[str],
        default_model: Optional[str],
        client: httpx.AsyncClient,
    ) -> Optional[str]:
        """
        vLLM側が「model not found (404)」を返したっぽい時に、
        - /v1/models を強制再取得
        - もう一度 resolve_model し直す
        ためのヘルパ。

        戻り値:
          - 置換すべきモデルID（= 再解決結果）。再試行するならそれを使う。
          - 再試行しても意味がなさそうなら None
        """
        if resp_status != 404:
            return None

        if not self._is_model_not_found_404(resp_json or {}):
            return None

        # 404が来たということは、モデル一覧が古い/未取得の可能性があるので取り直す
        await self.refresh(client, force=True)

        new_model = await self.resolve_model(requested, default_model, client)
        return new_model if new_model else None

    # ---- 内部処理 ----

    async def refresh(self, client: httpx.AsyncClient, force: bool) -> None:
        """
        TTLが切れたら /v1/models を取り直す。
        force=True の場合はTTL無視で必ず更新を試みる。
        """
        # 既に新鮮なら何もしない（forceでなければ）
        if (not force) and self._is_fresh():
            return

        async with self._lock:
            # lock待ちの間に他が更新してる可能性があるので再チェック
            if (not force) and self._is_fresh():
                return

            # /v1/models を取りに行く（リトライ付き）
            last_exc: Optional[Exception] = None
            for _ in range(max(1, self.retries if force else 1)):
                try:
                    ids = await self._fetch_models(client)
                    if ids:
                        self._models = ids
                        self._last_fetch = time.time()
                    return
                except Exception as e:
                    last_exc = e
                    # forceのときだけ待ってリトライする（非forceは1回で諦める）
                    if force:
                        await asyncio.sleep(self.retry_delay_sec)

            # 失敗しても落とさない（キャッシュがあるならそれを使い続ける）
            if last_exc:
                print(f"[gateway] WARN: failed to fetch /v1/models from vLLM: {last_exc}")

    async def _fetch_models(self, client: httpx.AsyncClient) -> List[str]:
        url = f"{self.base_url}/v1/models"
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        # {"object":"list","data":[{"id":"..."}]}
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]

    def _is_fresh(self) -> bool:
        if self.ttl_sec <= 0:
            return False
        return (time.time() - self._last_fetch) < self.ttl_sec

    def _is_model_not_found_404(self, resp_json: Dict[str, Any]) -> bool:
        """
        vLLM(OpenAI互換)のエラーJSONから「model not found」を雑に判定する。
        例: {"error": {"message": "... does not exist", "code": 404}}
        """
        err = resp_json.get("error") or {}
        code = err.get("code")
        msg = (err.get("message") or "").lower()
        return (code == 404) or ("does not exist" in msg) or ("not found" in msg)