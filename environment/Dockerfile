FROM python:3.12.8-slim-bookworm

RUN useradd --create-home --uid 10001 worker
WORKDIR /workspace
COPY workspace/ /workspace/
RUN chown -R worker:worker /workspace
USER worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["bash"]
