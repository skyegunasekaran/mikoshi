ARG FULL_IMAGE=programbench/ffmpeg_1776_ffmpeg.360a402@sha256:a2028263dd1bc65d914f64708446f7da095a2b70a8b585a8275ca93796a3a006
FROM ${FULL_IMAGE}
USER root
WORKDIR /
COPY requirements.lock /opt/programbench-grading/requirements.lock
COPY pytest_environment.py /usr/local/lib/python3.10/dist-packages/mikoshi_environment.py
RUN python3 -m pip download --require-hashes -r /opt/programbench-grading/requirements.lock -d /opt/programbench-grading/wheels \
    && python3 -m pip install --no-index --no-build-isolation --find-links=/opt/programbench-grading/wheels --require-hashes -r /opt/programbench-grading/requirements.lock
ENV PYTEST_PLUGINS=mikoshi_environment \
    PIP_NO_INDEX=1 \
    PIP_FIND_LINKS=/opt/programbench-grading/wheels \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Operator-only positive control. Never select this image for agent commands.
RUN mkdir -p /opt/programbench-control \
    && mv /workspace/ffmpeg /opt/programbench-control/gold \
    && rm -rf /workspace \
    && mkdir /workspace
WORKDIR /workspace
