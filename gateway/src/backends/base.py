# Pythonの将来互換（型ヒントが前方参照できる）
from __future__ import annotations

# ABC = 抽象クラス（インターフェースっぽいもの）を作るための標準ライブラリ
from abc import ABC, abstractmethod

# 型ヒント用（AsyncIterator = async for で回せるイテレータ、Any = なんでも、Dict = dict）
from typing import AsyncIterator, Any, Dict, Optional


class LLMBackend(ABC):
    """
    どの推論エンジン（vLLM / TRT-LLM / Ollama など）でも同じ呼び方で使えるようにするIF。
    今回は「OpenAI互換のSSE(JSON)を、そのままbytesで返す」のを共通仕様にする。
    """

    @abstractmethod
    async def stream_openai_sse(
        self,
        path: str,                          # 例: "/v1/chat/completions"
        payload: Dict[str, Any],            # OpenAI互換のリクエストJSON（model/messages/stream等）
        headers: Optional[Dict[str, str]] = None,  # 認証等のヘッダ（必要なら）
    ) -> AsyncIterator[bytes]:
        """
        OpenAI互換のストリーミングレスポンス（SSE）を
        生の bytes チャンクとして返す（gatewayは中身を解釈しない）。
        """
        raise NotImplementedError