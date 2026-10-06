"""Run a frozen DDPM budget plan only while the GPU has no other compute owner.

Every attempt and completed checkpoint hash is retained. If another CUDA job
appears, ask this trainer to yield at its next completed epoch, then resume the
same state after an idle grace period. Never stop or signal another process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, path)


def compute_pids():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(line.strip()) for line in result.splitlines() if line.strip()}


def check_inputs(plan):
    for path, expected in plan["hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=90)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(args.plan.read_text())
        state = {"plan_sha256": sha(args.plan), "jobs": {}, "attempts": []}
        if args.ledger.exists():
            state = json.loads(args.ledger.read_text())
            if state["plan_sha256"] != sha(args.plan):
                raise ValueError("Plan differs from the existing execution ledger")
            if state.get("phase") in ["training", "failed"]:
                raise RuntimeError("Inspect the previous attempt before recovery")
        check_inputs(plan)
        state["runner_pid"] = os.getpid()
        environment = {**os.environ, **plan["environment"]}
        for job in plan["jobs"]:
            prior = state["jobs"].get(job["name"])
            if prior is not None:
                for path, expected in prior["sha256"].items():
                    if sha(path) != expected:
                        raise ValueError(f"Completed checkpoint changed: {path}")
                continue
            finished = False
            while not finished:
                state["phase"] = "waiting_for_idle_gpu"
                state["job"] = job["name"]
                idle_since = None
                while (
                    idle_since is None
                    or time.monotonic() - idle_since < args.idle_seconds
                ):
                    others = compute_pids()
                    state["other_compute_pids"] = sorted(others)
                    state["observed_unix"] = time.time()
                    save(args.ledger, state)
                    if others:
                        idle_since = None
                    elif idle_since is None:
                        idle_since = time.monotonic()
                    time.sleep(args.poll_seconds)
                check_inputs(plan)
                output = Path(job["output"])
                stop_file = args.ledger.parent / "yield_gpu.request"
                stop_file.unlink(missing_ok=True)
                command = job["command"] + ["--stop-file", str(stop_file)]
                if (output / "training_state.pt").exists():
                    command.append("--resume")
                history_path = output / "history.json"
                history = (
                    json.loads(history_path.read_text())
                    if history_path.exists()
                    else []
                )
                pilot_epochs = job.get("pilot_epochs", 0)
                if pilot_epochs and (
                    not history or history[-1]["epoch"] < pilot_epochs
                ):
                    command += ["--stop-after-epoch", str(pilot_epochs)]
                log = args.ledger.parent / f"attempt_{len(state['attempts']):03d}.log"
                attempt = {
                    "job": job["name"],
                    "command": command,
                    "started_unix": time.time(),
                    "log": str(log),
                }
                state["attempts"].append(attempt)
                with log.open("x") as handle:
                    child = subprocess.Popen(
                        command,
                        cwd=job["cwd"],
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                    attempt["pid"] = child.pid
                    state["phase"] = "training"
                    save(args.ledger, state)
                    print(
                        json.dumps({"job": job["name"], "pid": child.pid}), flush=True
                    )
                    while child.poll() is None:
                        others = compute_pids() - {child.pid}
                        if others:
                            stop_file.write_text("Another compute job arrived\n")
                            attempt["yield_requested_for_pids"] = sorted(others)
                            save(args.ledger, state)
                        time.sleep(args.poll_seconds)
                    attempt["returncode"] = child.returncode
                attempt["finished_unix"] = time.time()
                attempt["log_sha256"] = sha(log)
                if child.returncode:
                    state["phase"] = "failed"
                    save(args.ledger, state)
                    raise RuntimeError(f"Training failed; inspect {log}")
                history = json.loads((output / "history.json").read_text())
                if pilot_epochs and len(history) >= pilot_epochs:
                    seconds = (
                        history[pilot_epochs - 1]["elapsed_seconds"] / pilot_epochs
                    )
                    attempt["pilot_seconds_per_epoch"] = seconds
                    if seconds > job["max_pilot_seconds_per_epoch"]:
                        state["phase"] = "pilot_cost_review"
                        save(args.ledger, state)
                        raise RuntimeError(
                            "Measured pilot exceeds the declared resource bound"
                        )
                finished = history[-1]["epoch"] == job["epochs"]
                if finished:
                    state["jobs"][job["name"]] = {
                        "completed_unix": time.time(),
                        "sha256": {path: sha(path) for path in job["artifacts"]},
                    }
                state["phase"] = "between_attempts"
                save(args.ledger, state)
        state["phase"] = "complete"
        save(args.ledger, state)
        print("All declared DDPM training budgets are complete", flush=True)


if __name__ == "__main__":
    main()
