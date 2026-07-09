FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ATDP_DATA_DIR=/data

COPY pyproject.toml README.md ./
COPY atdp_proxy ./atdp_proxy

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "atdp_proxy.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
