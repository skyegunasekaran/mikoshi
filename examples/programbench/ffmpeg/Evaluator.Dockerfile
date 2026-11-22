ARG CLEANROOM_IMAGE=programbench/ffmpeg_1776_ffmpeg.360a402@sha256:fbbf551125339eca7451eddd08e4d8afb3f3b26af0f1e5fc30c2dbba9d216322
FROM ${CLEANROOM_IMAGE}
USER root
COPY requirements.lock /opt/programbench-grading/requirements.lock
COPY pytest_environment.py /usr/local/lib/python3.10/dist-packages/mikoshi_environment.py
# Public test scripts invoke pip themselves. Satisfy those calls offline without
# editing their tests, run.sh files, or scoring rules.
RUN python3 -m pip download --require-hashes -r /opt/programbench-grading/requirements.lock -d /opt/programbench-grading/wheels \
    && python3 -m pip install --no-index --no-build-isolation --find-links=/opt/programbench-grading/wheels --require-hashes -r /opt/programbench-grading/requirements.lock
ENV PYTEST_PLUGINS=mikoshi_environment \
    PIP_NO_INDEX=1 \
    PIP_FIND_LINKS=/opt/programbench-grading/wheels \
    PIP_DISABLE_PIP_VERSION_CHECK=1
