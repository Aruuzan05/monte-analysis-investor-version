# Keepp Capital — Monte Carlo simulator
#
# Built to be assembled from INSIDE mainland China, where pypi.org and Docker
# Hub are slow or unreachable. Both mirrors are build args, so the same file
# works outside China by overriding them:
#
#   docker build \
#     --build-arg BASE=python:3.12-slim \
#     --build-arg PIP_INDEX=https://pypi.org/simple \
#     -t keepp-mc .

# Docker Hub is throttled from the mainland; this is Aliyun's mirror of the
# same official image. Swap for a Tencent (ccr.ccs.tencentyun.com) mirror if
# you are hosting on Tencent Cloud.
ARG BASE=registry.cn-hangzhou.aliyuncs.com/library/python:3.12-slim
FROM ${BASE}

ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_HOST=mirrors.aliyun.com

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url ${PIP_INDEX} --trusted-host ${PIP_HOST} \
        -r requirements.txt

COPY *.py jpm_correlations.csv ./
COPY .streamlit/config.toml .streamlit/config.toml

# Run as a non-root user. The only thing the app writes is the SQLite
# database (accounts + saved runs) in /data — mount a volume there so it
# survives a rebuild. Set KEEPP_DATABASE_URL to use Postgres instead.
ENV KEEPP_DATA_DIR=/data
RUN useradd --create-home --uid 10001 keepp \
    && mkdir -p /data && chown -R keepp:keepp /app /data
VOLUME ["/data"]
USER keepp

EXPOSE 8501

# Streamlit's own health endpoint. Docker will restart the container if the
# server wedges rather than leaving managers on a blank page.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
