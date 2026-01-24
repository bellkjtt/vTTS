FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 앱 복사
COPY . .

# pip 업그레이드 및 빌드 도구 설치
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel build

# Python 패키지 설치 (all 엔진 포함)
RUN pip3 install --no-cache-dir .

# 포트 노출
EXPOSE 8000

# 환경 변수
ENV PYTHONUNBUFFERED=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 엔트리포인트
ENTRYPOINT ["vtts"]
CMD ["--help"]
