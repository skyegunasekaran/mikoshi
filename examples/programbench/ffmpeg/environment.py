#!/usr/bin/env python3
"""Prepare and grade the FFmpeg cleanroom. Never starts inference or paid runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
STATE = ROOT / "tmp" / "programbench" / "ffmpeg"
BENCHMARK = STATE / "benchmark"
LOCK = json.loads((BASE / "environment.lock.json").read_text())
RECEIPT = STATE / "images.json"
DOCS = [
    "README.md",
    "LICENSE.md",
    "COPYING.GPLv2",
    "COPYING.GPLv3",
    "COPYING.LGPLv2.1",
    "COPYING.LGPLv3",
    "assets",
]


def run(*args, capture=False, **kwargs):
    kwargs.setdefault("stdout", subprocess.PIPE if capture else None)
    result = subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)
    return result.stdout.strip() if capture else None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recipe_digest():
    h = hashlib.sha256()
    for name in (
        "environment.lock.json",
        "Command.Dockerfile",
        "Control.Dockerfile",
        "Evaluator.Dockerfile",
        "requirements.lock",
        "pytest_environment.py",
    ):
        h.update(name.encode() + b"\0" + (BASE / name).read_bytes())
    return h.hexdigest()


def check_benchmark():
    if run("git", "-C", BENCHMARK, "rev-parse", "HEAD", capture=True) != LOCK["benchmark_commit"]:
        raise ValueError("Benchmark revision mismatch")
    if run("git", "-C", BENCHMARK, "status", "--porcelain", capture=True):
        raise ValueError("Pinned benchmark checkout has modifications")


def inspect_image(image):
    info = json.loads(run("docker", "image", "inspect", image, capture=True))[0]
    if (info["Os"], info["Architecture"]) != ("linux", "amd64"):
        raise ValueError(f"Incorrect image platform: {image}")
    return {key: info[key] for key in ("Id", "RepoDigests", "Os", "Architecture", "Size")}


def images():
    check_benchmark()
    data = json.loads(RECEIPT.read_text())
    if data["recipe_sha256"] != recipe_digest():
        raise ValueError("Recipe changed; rerun setup")
    for role in ("cleanroom", "full", "command", "evaluator", "control"):
        if inspect_image(data[role]["Id"])["Id"] != data[role]["Id"]:
            raise ValueError(f"Image changed for {role}; rerun setup")
    return data


def check_fixtures():
    manifest = json.loads((STATE / "fixtures.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if digest(STATE / "fixtures" / relative) != expected:
            raise ValueError(f"Fixture checksum mismatch: {relative}")
    branches = json.loads(
        (BENCHMARK / "src/programbench/data/tasks" / LOCK["instance_id"] / "tests.json").read_text()
    )["branches"]
    for branch, info in branches.items():
        if not info.get("ignored"):
            relative = f"{LOCK['instance_id']}/tests/{branch}.tar.gz"
            if relative not in manifest["sha256"]:
                raise ValueError(f"Missing fixture: {relative}")
    return manifest


def setup(args):
    STATE.mkdir(parents=True, exist_ok=True)
    if not BENCHMARK.exists():
        run("git", "init", "-q", BENCHMARK)
        run("git", "-C", BENCHMARK, "remote", "add", "origin", LOCK["benchmark_repository"])
        run("git", "-C", BENCHMARK, "fetch", "--depth", "1", "origin", LOCK["benchmark_commit"])
        run("git", "-C", BENCHMARK, "checkout", "--detach", "FETCH_HEAD")
    check_benchmark()
    run("uv", "sync", "--frozen", "--python", sys.executable, cwd=BENCHMARK)
    if not (STATE / "fixtures.json").exists():
        run(
            BENCHMARK / ".venv/bin/python",
            "-c",
            "from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id="
            + repr(LOCK["hf_repository"])
            + ", repo_type='dataset', revision="
            + repr(LOCK["hf_revision"])
            + ", allow_patterns="
            + repr(LOCK["instance_id"] + "/**")
            + ", local_dir="
            + repr(str(STATE / "fixtures"))
            + ", max_workers=4)",
        )
        fixture_root = STATE / "fixtures"
        write_json(
            STATE / "fixtures.json",
            {
                "repository": LOCK["hf_repository"],
                "revision": LOCK["hf_revision"],
                "sha256": {
                    p.relative_to(fixture_root).as_posix(): digest(p)
                    for p in sorted((fixture_root / LOCK["instance_id"]).rglob("*"))
                    if p.is_file()
                },
            },
        )
    check_fixtures()
    if RECEIPT.exists() and not args.rebuild:
        try:
            images()
            print(f"Reusing verified images and fixtures: {STATE}")
            return
        except (ValueError, subprocess.CalledProcessError):
            pass
    for role in ("cleanroom", "full"):
        try:
            inspect_image(LOCK[role + "_image"])
        except subprocess.CalledProcessError:
            run("docker", "pull", "--platform", LOCK["platform"], LOCK[role + "_image"])
    for role, parent, arg in (
        ("evaluator", "cleanroom", "CLEANROOM_IMAGE"),
        ("command", "evaluator", "EVALUATOR_IMAGE"),
        ("control", "full", "FULL_IMAGE"),
    ):
        run(
            "docker",
            "build",
            "--provenance=false",
            "--platform",
            LOCK["platform"],
            "--build-arg",
            arg + "=" + LOCK.get(parent + "_image", LOCK.get(parent + "_tag")),
            "-f",
            BASE / (role.title() + ".Dockerfile"),
            "-t",
            LOCK[role + "_tag"],
            BASE,
        )
    data = {"recipe_sha256": recipe_digest(), "recorded_at": datetime.now(timezone.utc).isoformat()}
    for role in ("cleanroom", "full", "command", "evaluator", "control"):
        ref = LOCK.get(role + "_image", LOCK.get(role + "_tag"))
        data[role] = {"ref": ref, **inspect_image(ref)}
    write_json(RECEIPT, data)
    write_json(
        BASE / "mikoshi-deployment.json",
        {
            "deployment": {
                "docker": {
                    "image": data["command"]["Id"],
                    "network": LOCK["inference_network"],
                    "platform": LOCK["platform"],
                }
            }
        },
    )
    print(f"Ready: {STATE}")


def prepare(args):
    data = images()
    target = args.target.resolve()
    if target.exists():
        raise ValueError(f"Refusing to overwrite target: {target}")
    if (
        target == STATE
        or target in STATE.parents
        or any(
            target == private or private in target.parents
            for private in (BENCHMARK, STATE / "fixtures", BASE)
        )
    ):
        raise ValueError("Target must be separate from environment recipes and private fixtures")
    target.mkdir(parents=True)
    cid = run(
        "docker", "create", "--platform", LOCK["platform"], data["cleanroom"]["Id"], capture=True
    )
    try:
        for name in DOCS:
            run("docker", "cp", f"{cid}:/workspace/{name}", target / name)
    finally:
        run("docker", "rm", "-f", cid, capture=True)
    (target / ".gitignore").write_text("/executable\n/build/\n__pycache__/\n*.py[cod]\n")
    run("git", "-C", target, "-c", "init.defaultBranch=main", "init", "-q")
    run("git", "-C", target, "add", "-A")
    run(
        "git",
        "-C",
        target,
        "-c",
        "user.name=ProgramBench setup",
        "-c",
        "user.email=programbench@local",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Pristine ProgramBench FFmpeg cleanroom materials",
    )
    run("git", "-C", target, "tag", "baseline")
    write_json(
        target.with_name(target.name + ".provenance.json"),
        {
            "cleanroom_image_id": data["cleanroom"]["Id"],
            "command_image_id": data["command"]["Id"],
            "reference_path": LOCK["reference_path"],
            "baseline_tree": run("git", "-C", target, "rev-parse", "baseline^{tree}", capture=True),
        },
    )
    print(target)


def smoke(args):
    data = images()
    command = r"""set -eu
test -x /opt/programbench-reference/executable
test ! -r /opt/programbench-reference/executable
test -r README.md && test -r assets/flac_rt.wav
test ! -e /workspace/eval
git status --porcelain
/opt/programbench-reference/executable -version | head -4
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
printf '#include <stdio.h>\n#include <zlib.h>\n#include <lzma.h>\nint main(void){puts(zlibVersion());puts(lzma_version_string());return 0;}\n' > "$d/main.c"
gcc "$d/main.c" -lz -llzma -o "$d/hello"
"$d/hello"
g++ -x c++ "$d/main.c" -lz -llzma -o "$d/hello-cpp"
"$d/hello-cpp"
printf 'fn main() { println!("rust-build-ok"); }\n' > "$d/main.rs"
rustc "$d/main.rs" -o "$d/hello-rust"
test "$("$d/hello-rust")" = rust-build-ok
printf 'package main\nimport "fmt"\nfunc main() { fmt.Println("go-build-ok") }\n' > "$d/main.go"
GOCACHE="$d/go-cache" go build -o "$d/hello-go" "$d/main.go"
test "$("$d/hello-go")" = go-build-ok
python3 -c 'import pytest,pytest_timeout,xdist,pytest_dependency,pytest_rerunfailures; print("test-tools-ok")'
python3 -m pip check
/opt/programbench-reference/executable -v error -i assets/flac_rt.flac -f wav "$d/sample.wav"
test -s "$d/sample.wav"
python3 -c 'import urllib.request; r=urllib.request.urlopen("https://pypi.org/simple/pytest-timeout/",timeout=10); assert r.status==200; print("network-enabled-ok")'
echo smoke-ok"""
    run(
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        LOCK["platform"],
        "--network",
        LOCK["inference_network"],
        "--cpus",
        str(LOCK["local_resources"]["cpus"]),
        "--memory",
        LOCK["local_resources"]["memory"],
        "--pids-limit",
        "256",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--mount",
        f"type=bind,source={args.target.resolve()},target=/workspace",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "/bin/bash",
        data["command"]["Id"],
        "-lc",
        command,
    )


def submit(args):
    target = args.target.resolve()
    output = args.output.resolve()
    if output == target or target in output.parents:
        raise ValueError("Submission output must be outside the agent target")
    if not (target / "compile.sh").is_file():
        raise ValueError("Submission requires compile.sh")
    paths = subprocess.run(
        ["git", "-C", str(target), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    output.mkdir(parents=True, exist_ok=False)
    task_dir = output / LOCK["instance_id"]
    task_dir.mkdir()
    archive = task_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=False) as tar:
        for raw in sorted(set(paths.split(b"\0")) - {b""}):
            name = os.fsdecode(raw)
            if name == "executable" or name.split("/", 1)[0] == ".git":
                continue
            path = target / name
            if path.exists() or path.is_symlink():
                tar.add(path, arcname=name, recursive=False)
    write_json(
        output / "submission.json",
        {
            "archive_sha256": digest(archive),
            "target": str(target),
            "head": run("git", "-C", target, "rev-parse", "HEAD", capture=True),
            "git_status": run(
                "git", "--no-optional-locks", "-C", target, "status", "--porcelain", capture=True
            ),
        },
    )
    print(archive)


def dependencies(args):
    data = images()
    script = r"""python3 - <<'PY'
import json, subprocess, sys
def output(argv):
    return subprocess.check_output(argv, text=True).strip()
versions = {}
for tool in ('gcc', 'g++', 'make', 'cmake', 'pkg-config', 'python3', 'git', 'rustc', 'cargo'):
    versions[tool] = output([tool, '--version']).splitlines()[0]
versions['go'] = output(['go', 'version'])
print(json.dumps({
    'python_packages': sorted(json.loads(output([sys.executable, '-m', 'pip', 'list', '--format=json'])), key=lambda p: p['name']),
    'system_packages': output(['dpkg-query', '-W', '-f=${Package}\t${Version}\n']).splitlines(),
    'toolchains': versions,
    'native_libraries': output(['pkg-config', '--modversion', 'zlib', 'liblzma']).splitlines(),
}))
PY"""
    inventories = {}
    for role in ("command", "evaluator"):
        identity = ["--user", f"{os.getuid()}:{os.getgid()}"] if role == "command" else []
        inventories[role] = json.loads(
            run(
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--platform",
                LOCK["platform"],
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                *identity,
                "--env",
                "HOME=/tmp",
                "--entrypoint",
                "/bin/bash",
                data[role]["Id"],
                "-lc",
                script,
                capture=True,
            )
        )
    if inventories["command"] != inventories["evaluator"]:
        raise ValueError("Agent/evaluator dependency mismatch")
    write_json(
        STATE / "dependency-parity.json",
        {
            "matching": True,
            "image_ids": {r: data[r]["Id"] for r in inventories},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **inventories["command"],
        },
    )
    print(
        f"Matching agent/evaluator packages, native libraries, and toolchains: {STATE / 'dependency-parity.json'}"
    )


def grade(args):
    data = images()
    fixtures = check_fixtures()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_dir = output / LOCK["instance_id"]
    task_dir.mkdir()
    archive = task_dir / "submission.tar.gz"
    if args.command == "control":
        with tempfile.TemporaryDirectory(prefix="programbench-control-") as directory:
            source = Path(directory)
            if args.kind == "reference":
                script = "#!/bin/sh\nset -eu\ncp /opt/programbench-control/gold ./executable\nchmod +x ./executable\n"
            else:
                script = "#!/bin/sh\nset -eu\ngcc main.c -o executable\n"
                (source / "main.c").write_text("int main(void) { return 1; }\n")
            (source / "compile.sh").write_text(script)
            with tarfile.open(archive, "w:gz") as tar:
                for path in sorted(source.iterdir()):
                    tar.add(path, arcname=path.name)
    else:
        shutil.copyfile(args.submission.resolve(), archive)
    role = "control" if args.command == "control" and args.kind == "reference" else "evaluator"
    # The upstream API accepts name:tag, so bind an ID-derived local alias and
    # verify it against the immutable receipt before calling the unmodified API.
    alias = "mikoshi-programbench-eval:" + data[role]["Id"].split(":")[1]
    run("docker", "tag", data[role]["Id"], alias)
    if inspect_image(alias)["Id"] != data[role]["Id"]:
        raise ValueError("Evaluator image alias mismatch")
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.command,
        "control": getattr(args, "kind", None),
        "control_branches": getattr(args, "branch", []),
        "image_id": data[role]["Id"],
        "benchmark_commit": LOCK["benchmark_commit"],
        "hf_revision": LOCK["hf_revision"],
        "submission_sha256": digest(archive),
        "recipe_sha256": recipe_digest(),
        "helper_sha256": digest(Path(__file__)),
        "grade_helper_sha256": digest(BASE / "grade.py"),
        "fixture_sha256": fixtures["sha256"],
        "cpus": args.cpus,
        "memory": args.memory,
        "network": LOCK["evaluation_network"],
        "branch_workers": 1,
        "model_requests": 0,
    }
    write_json(output / "execution.json", receipt)
    with (output / "console.log").open("w") as console:
        result = subprocess.run(
            [
                str(BENCHMARK / ".venv/bin/python"),
                str(BASE / "grade.py"),
                "--state",
                str(STATE),
                "--output",
                str(output),
                "--image",
                alias,
                "--cpus",
                str(args.cpus),
                "--memory",
                args.memory,
                *[part for branch in getattr(args, "branch", []) for part in ("--branch", branch)],
            ],
            stdout=console,
            stderr=subprocess.STDOUT,
        )
    receipt.update(finished_at=datetime.now(timezone.utc).isoformat(), returncode=result.returncode)
    write_json(output / "execution.json", receipt)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print((output / "console.log").read_text()[-8000:], file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)


def archive_images(args):
    data = images()
    path = STATE / "images.tar"
    with path.open("xb") as stream:
        run(
            "docker",
            "image",
            "save",
            *dict.fromkeys(
                data[r]["Id"] for r in ("cleanroom", "full", "command", "evaluator", "control")
            ),
            stdout=stream,
        )
    (STATE / "images.tar.sha256").write_text(f"{digest(path)}  images.tar\n")
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=setup)
    p = sub.add_parser("prepare")
    p.add_argument("target", type=Path)
    p.set_defaults(func=prepare)
    p = sub.add_parser("smoke")
    p.add_argument("--target", required=True, type=Path)
    p.set_defaults(func=smoke)
    p = sub.add_parser("submit")
    p.add_argument("target", type=Path)
    p.add_argument("output", type=Path)
    p.set_defaults(func=submit)
    for name in ("grade", "control"):
        p = sub.add_parser(name)
        if name == "grade":
            p.add_argument("submission", type=Path)
        else:
            p.add_argument("kind", choices=("negative", "reference"))
            p.add_argument(
                "--branch",
                action="append",
                default=[],
                help="Explicit control diagnostic subset; default runs all branches",
            )
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--cpus", type=int, default=LOCK["local_resources"]["cpus"])
        p.add_argument("--memory", default=LOCK["local_resources"]["memory"])
        p.set_defaults(func=grade)
    sub.add_parser("archive").set_defaults(func=archive_images)
    sub.add_parser("dependencies").set_defaults(func=dependencies)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
