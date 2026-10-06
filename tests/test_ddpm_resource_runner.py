import importlib.util
import json
import sys
from pathlib import Path


def test_runner_waits_yields_its_job_and_resumes_without_touching_other_jobs(
    tmp_path, monkeypatch
):
    path = Path(__file__).parents[1] / "research/run_ddpm_budget.py"
    spec = importlib.util.spec_from_file_location("ddpm_resource_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    worker = tmp_path / "worker.py"
    worker.write_text("""import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--stop-file',type=Path)
p.add_argument('--output',type=Path)
p.add_argument('--resume',action='store_true')
p.add_argument('--stop-after-epoch',type=int)
a=p.parse_args();a.output.mkdir(exist_ok=True)
epoch=2 if a.resume else 1
(a.output/'training_state.pt').write_text('fake resumable state')
(a.output/'history.json').write_text(json.dumps(
    [{'epoch':i,'elapsed_seconds':.01*i} for i in range(1,epoch+1)]))
if not a.resume:
    deadline=time.monotonic()+5
    while not a.stop_file.exists():
        if time.monotonic()>deadline:raise RuntimeError('No yield request')
        time.sleep(.01)
""")
    output, plan, ledger = [
        tmp_path / n for n in ["output", "plan.json", "ledger.json"]
    ]
    plan.write_text(
        json.dumps(
            {
                "environment": {},
                "hashes": {str(worker): runner.sha(worker)},
                "jobs": [
                    {
                        "name": "fake",
                        "cwd": str(tmp_path),
                        "output": str(output),
                        "epochs": 2,
                        "pilot_epochs": 1,
                        "max_pilot_seconds_per_epoch": 1,
                        "command": [
                            sys.executable,
                            str(worker),
                            "--output",
                            str(output),
                        ],
                        "artifacts": [str(output / "history.json")],
                    }
                ],
            }
        )
    )
    children = []
    observations = []
    original_popen = runner.subprocess.Popen

    def pids():
        if not children:
            result = {999999} if len(observations) < 2 else set()
        elif len(children) == 1 and children[0].poll() is None:
            result = {children[0].pid, 999999}
        elif len(children) == 2 and children[1].poll() is None:
            result = {children[1].pid}
        else:
            result = set()
        observations.append(result)
        return result

    def spawn(*args, **kwargs):
        assert observations[-1] == set(), "Started while another owner was present"
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner, "compute_pids", pids)
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--plan",
            str(plan),
            "--ledger",
            str(ledger),
            "--idle-seconds",
            ".01",
            "--poll-seconds",
            ".01",
        ],
    )
    runner.main()
    saved = json.loads(ledger.read_text())
    assert saved["phase"] == "complete"
    assert len(saved["attempts"]) == 2
    assert saved["attempts"][0]["yield_requested_for_pids"] == [999999]
    assert "--resume" in saved["attempts"][1]["command"]
    assert "--stop-after-epoch" in saved["attempts"][0]["command"]
    assert saved["attempts"][0]["pilot_seconds_per_epoch"] == 0.01
    assert all(child.returncode == 0 for child in children)
    assert saved["jobs"]["fake"]["sha256"][str(output / "history.json")] == runner.sha(
        output / "history.json"
    )
