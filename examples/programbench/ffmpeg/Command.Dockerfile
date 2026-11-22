ARG EVALUATOR_IMAGE=mikoshi-programbench-ffmpeg:895f65916bc5-evaluator
FROM ${EVALUATOR_IMAGE}
USER root
WORKDIR /
# Keep the query-only reference outside the writable, source-mounted workspace.
RUN mkdir -p /opt/programbench-reference \
    && mv /workspace/executable /opt/programbench-reference/executable \
    && chown root:root /opt/programbench-reference/executable \
    && chmod 111 /opt/programbench-reference/executable \
    && rm -rf /workspace \
    && mkdir /workspace \
    && git config --system --add safe.directory /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_INDEX=0
WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
