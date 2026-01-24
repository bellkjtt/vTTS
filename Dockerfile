FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    cmake \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리
WORKDIR /app

# pip 업그레이드
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 앱 복사
COPY pyproject.toml ./
COPY vtts/ ./vtts/

# Python 패키지 설치
RUN pip3 install --no-cache-dir ".[all]"

# 추가 의존성 설치
RUN pip3 install --no-cache-dir pytorch_lightning matplotlib

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 엔트리포인트
ENTRYPOINT ["vtts"]
CMD ["--help"]
