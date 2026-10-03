# The platform image (simulated GPUs: workloads sleep; nothing runs a model).
FROM python:3.12-slim
LABEL cvproject=ai-workload-platform org.opencontainers.image.source="ai-workload-platform"

ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# A package index or mirror can be passed at build time (--build-arg PIP_INDEX_URL=...); nothing is set here.
ARG PIP_INDEX_URL
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 awp \
    && mkdir -p /data && chown awp /data
USER awp
EXPOSE 18400
# Inside a container the service listens on all interfaces; publish it on 127.0.0.1 only (see docker-compose.yml).
ENTRYPOINT ["python", "-m", "ai_workload_platform"]
CMD ["up", "--host", "0.0.0.0", "--port", "18400", "--db", "/data/awp.db"]
