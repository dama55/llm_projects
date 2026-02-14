from __future__ import annotations

# 環境変数を読むため
import os

# 型ヒント
from typing import AsyncIterator, Any, Dict, Optional

# HTTPクライアント（FastAPIの外側でHTTPリクエストを投げる）
import httpx

# 自分で定義したインターフェース
from .base import LLMBackend


class VLLMBackend(LLMBackend):
    """
    vLLMはOpenAI互換APIを提供するので、gatewayは基本的に
    "受け取ったJSONをそのままvLLMへ投げて、返り値もそのまま返す"
    だけで良い。
    """

    def __init__(self, base_url: Optional[str] = None):
        # docker-composeで VLLM_BASE_URL=http://vllm:8000 を渡す想定
        # base_urlが引数で渡されたらそれを優先し、無ければ環境変数から読む
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://vllm:8000")).rstrip("/")
        # rstrip("/") は末尾スラッシュを消して、URL連結ミスを防ぐ

    async def stream_openai_sse(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        # vLLMのエンドポイントURLを組み立て
        url = f"{self.base_url}{path}"

        # 基本ヘッダ（JSON送信）
        req_headers: Dict[str, str] = {"Content-Type": "application/json"}

        # 呼び出し元からヘッダが渡されたら上書き・追加
        if headers:
            req_headers.update(headers)

        # AsyncClient(timeout=None) は「長時間のストリーム」でもタイムアウトしないようにする
        async with httpx.AsyncClient(timeout=None) as client:
            # client.stream(...) はレスポンスをチャンク単位で受け取れる
            async with client.stream("POST", url, json=payload, headers=req_headers) as r:
                # HTTPエラー（4xx/5xx）なら例外にする
                r.raise_for_status()

                # aiter_bytes() で、SSEの生データを bytes チャンクで読み続ける
                async for chunk in r.aiter_bytes():
                    if chunk:
                        # gatewayは「中身を解釈せず」そのまま返す方針なので、
                        # bytesチャンクをそのままyieldする
                        yield chunk