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

## uvインストール. どの権限でも(システム全体で)uvが使えるように"/usr/local/bin"にコピー
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    rm /uv-installer.sh

# ## ホストのUID/GIDでユーザー作成 ※ターミナルに"id"を実行すると確認可能
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

## 任意のディレクトリパス + 作業ディレクトリの所有者を設定
ARG DIR
WORKDIR /${DIR}

COPY . .

## pyproject.tomlがある場合は以下をコメント解除
# RUN uv sync && \
#     uv pip install -r pyproject.toml

# # 非rootユーザで実行
RUN chown -R appuser:appgroup /${DIR}
USER appuser

CMD ["bash"]