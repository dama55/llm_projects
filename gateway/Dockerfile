FROM python:3.11-slim

WORKDIR /app

# 依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY src ./src

EXPOSE 8000

# デフォルトのコマンドを設定
CMD ["tail", "-f", "/dev/null"]
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]