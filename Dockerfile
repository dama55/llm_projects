# ベースイメージを指定
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# ビルド時にCMAKE_ARGSを受け取る
ARG CMAKE_ARGS
ENV CMAKE_ARGS=$CMAKE_ARGS

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    python3-pip python3-dev \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# アプリケーションコードをコピー
COPY app/ /app/


# デフォルトのコマンドを設定
CMD ["tail", "-f", "/dev/null"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]