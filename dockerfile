FROM python:3.12.0

## 必要なパッケージのインストール
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    build-essential \
    curl \
    git \
    vim \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=true \
    UV_LINK_MODE=copy

## uvインストール
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# ## ホストのUID/GIDでユーザー作成 ※ターミナルに"id"を実行すると確認可能
# RUN groupadd -r -g 1000 appgroup && \
#     useradd -r -u 1000 -g appgroup appuser

## 任意のディレクトリパス + 作業ディレクトリの所有者を設定
WORKDIR /app
# RUN chown appgroup:appuser /app

COPY . .

## pyproject.tomlがある場合は以下をコメント解除
# RUN uv sync && \
#     uv pip install -r pyproject.toml

# # 非rootユーザで実行
# USER appuser

CMD ["bash"]