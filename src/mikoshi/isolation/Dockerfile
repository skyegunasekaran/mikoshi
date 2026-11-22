FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ripgrep ca-certificates procps nodejs npm cargo rustc build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*
# Docker bind mounts can report host ownership differently. Trust only the
# explicitly supplied checkout, not arbitrary repositories in the container.
RUN git config --system --add safe.directory /workspace
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
CMD ["/bin/sh"]
