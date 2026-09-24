"""Bounded timing-only chunk-4/chunk-8 CUDA run for the frozen G15 winner.

This is a reviewed one-shot runner, not a trainer launcher. Model/schema
preflight runs CPU-only; CUDA starts only in the bounded benchmark subprocess.
"""
import hashlib
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


ROOT = Path(r"C:\Docker\SlitherAI")
CHECKOUT = Path(r"C:\Docker\SlitherAI-perfbench-g15-46de6ca")
RUN = ROOT / "runs" / "20260924-163112-852097"
PYTHON = Path(r"C:\Docker\SlitherAI\.venv\Scripts\python.exe")
FROZEN_PAYLOAD = Path(r"C:\Users\88mat\AppData\Local\Temp\slitherai-g15-best-validation-ref-20260924-163112.pkl")
SOURCE_PAYLOAD = RUN / "best-validation.pkl"
CHECKPOINT = RUN / "checkpoint-14"
API = "http://127.0.0.1:8765"
EXPECTED_HEAD = "46de6ca59f7eecb335b18255eb35d589c7349d8f"
EXPECTED_BENCHMARK_MODULE_SHA256 = "d67cc79500ea9e0613f8854ab7e17e89b8043b0104fca9cc13efeb4f25af8f7c"
EXPECTED_SETTINGS_SHA = "e8d7f1245a5ee5644e28606b8442797cbf3e24091949139917ddd576ef8f401b"
EXPECTED_SOURCE_SHA = "682fd88f3e4c1d8a7adaa57bdb2654c3d7dc9c709289fc043200394c42d70731"
EXPECTED_PAYLOAD_SHA = EXPECTED_SOURCE_SHA
EXPECTED_CHECKPOINT_SHA = "dc3980133e7d9b1a48bbc96cd402d629d0e76a4f811cb7654112c24da7d68ad9"
EXPECTED_GENE_SHA = "12faa26e179b9c7db319f834b60b39579652fb8d5e03947c4b7d3f59f7278d3e"
EXPECTED_GENERATION = 14
EXPECTED_GENOME_ID = 2867
OUTPUT = RUN / "analysis" / "policy-timing-only-G15-id2867-chunk4-8-seed1038282-60s-46de6ca.json"
METADATA = RUN / "analysis" / "policy-timing-only-G15-id2867-chunk4-8-seed1038282-60s-46de6ca-control.json"
BENCHMARK_MODULE = CHECKOUT / "slitherai" / "benchmark_observation.py"


def api(method, path, value=None):
    body = None if value is None else json.dumps(value).encode("utf-8")
    request = urllib.request.Request(API + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=4) as response:
        return json.load(response)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trainer_processes():
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' "
              "-and $_.CommandLine -and $_.CommandLine.Contains('slitherai.train') "
              f"-and $_.CommandLine.Contains('{RUN}') }} "
              "| Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                            capture_output=True, text=True, timeout=5, check=True)
    if not result.stdout.strip():
        return []
    value = json.loads(result.stdout)
    return value if isinstance(value, list) else [value]


def read_control():
    return json.loads((RUN / "control.json").read_text(encoding="utf-8"))


def kill_verified_benchmark_processes(root_pid):
    """Kill only this benchmark PID or its descendants after command-line checks."""
    evidence = {"root_pid": root_pid, "descendants_seen": [], "verified_pids": [],
                "unverified_pids": [], "kill_errors": [], "still_alive": []}
    try:
        root = psutil.Process(root_pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return evidence
    except psutil.Error as exc:
        evidence["kill_errors"].append(f"descendant_lookup:{type(exc).__name__}:{exc}")
        return evidence

    def is_this_benchmark(candidate):
        try:
            args = candidate.cmdline()
        except psutil.NoSuchProcess:
            return False
        except psutil.Error as exc:
            evidence["kill_errors"].append(f"cmdline_pid_{candidate.pid}:{type(exc).__name__}:{exc}")
            return False
        joined = " ".join(args).casefold()
        return "slitherai.benchmark_observation" in joined and str(OUTPUT).casefold() in joined

    targets = []
    for child in descendants:
        evidence["descendants_seen"].append(child.pid)
        if is_this_benchmark(child):
            targets.append(child)
        else:
            evidence["unverified_pids"].append(child.pid)
    if is_this_benchmark(root):
        targets.append(root)
    else:
        evidence["unverified_pids"].append(root_pid)

    # Kill deepest verified descendants first. Never inspect or signal unrelated PIDs.
    for target in reversed(targets):
        try:
            target.kill()
            evidence["verified_pids"].append(target.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            evidence["kill_errors"].append(f"kill_pid_{target.pid}:{type(exc).__name__}:{exc}")
    if targets:
        _, alive = psutil.wait_procs(targets, timeout=3)
        evidence["still_alive"] = [proc.pid for proc in alive]
    evidence["cleanup_warning"] = bool(evidence["unverified_pids"] or evidence["kill_errors"]
                                       or evidence["still_alive"])
    return evidence


def model_preflight():
    if sha256(FROZEN_PAYLOAD) != EXPECTED_PAYLOAD_SHA:
        raise RuntimeError("Frozen G15 payload SHA changed; refusing another model.")
    if sha256(RUN / "settings.json") != EXPECTED_SETTINGS_SHA:
        raise RuntimeError("Reference settings SHA changed; refusing another run config.")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("G15 checkpoint SHA changed; refusing inconsistent provenance.")

    saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(CHECKOUT))
    try:
        from slitherai.benchmark_observation import load_run_champion
        import slitherai.benchmark_observation as benchmark_module
        if Path(benchmark_module.__file__).resolve() != BENCHMARK_MODULE.resolve():
            raise RuntimeError(f"Benchmark module imported from unexpected path: {benchmark_module.__file__}")
        config, genome, _, source = load_run_champion(RUN, FROZEN_PAYLOAD)
    finally:
        if saved_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_visibility
    actual = (source.get("genome_generation"), genome.key,
              source.get("genome_gene_sha256"), config.sensor_version)
    expected = (EXPECTED_GENERATION, EXPECTED_GENOME_ID, EXPECTED_GENE_SHA, "legacy-v1")
    if actual != expected:
        raise RuntimeError(f"Frozen payload identity mismatch: got {actual}, expected {expected}.")
    return source


def main():
    if Path.cwd().resolve() != CHECKOUT:
        raise SystemExit("Refusing: run from the reviewed isolated checkout.")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CHECKOUT,
                          capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    if head != EXPECTED_HEAD:
        raise SystemExit(f"Refusing unexpected worktree SHA: {head}")
    if sha256(BENCHMARK_MODULE) != EXPECTED_BENCHMARK_MODULE_SHA256:
        raise SystemExit("Refusing unexpected timing-only benchmark module SHA.")
    if OUTPUT.exists() or METADATA.exists():
        raise SystemExit("Refusing to overwrite the unique output or metadata path.")
    payload_source = model_preflight()

    state_before = api("GET", "/api/state")
    if (not state_before["active"] or state_before["status"].get("phase") != "training"
            or state_before["status"].get("run") != str(RUN)):
        raise SystemExit("Refusing: service is not training the expected reference run.")
    control_before = read_control()
    process_ids_before = trainer_processes()
    if len(process_ids_before) != 2:
        raise SystemExit(f"Refusing: expected one trainer wrapper/child pair, got {process_ids_before}.")

    started_utc = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + 180.0
    touched_control = False
    pause_acknowledged = False
    restored = False
    restore_error = None
    benchmark_error = None
    benchmark_outcome = "not-started"
    benchmark_return_code = None
    benchmark_timed_out = False
    benchmark_pid = None
    process = None
    cleanup_evidence = None
    state_after = None
    stdout_path = METADATA.with_suffix(".stdout.txt")
    stderr_path = METADATA.with_suffix(".stderr.txt")
    try:
        touched_control = True
        api("POST", "/api/control", {"pause": True})
        while time.monotonic() < deadline:
            current_state = api("GET", "/api/state")
            current_control = read_control()
            if current_state["status"].get("run") != str(RUN):
                raise RuntimeError("Service changed reference run during pause acknowledgement.")
            if (current_control.get("stop") != control_before.get("stop")
                    or current_control.get("arena") != control_before.get("arena")):
                raise RuntimeError("Stop or arena changed; aborting without changing either field.")
            if (current_state["status"].get("phase") == "paused"
                    and current_control.get("pause") is True):
                pause_acknowledged = True
                break
            time.sleep(0.25)
        if not pause_acknowledged:
            raise TimeoutError("Reference pause was not acknowledged within 180 seconds.")

        timeout = min(145.0, deadline - time.monotonic() - 15.0)
        if timeout <= 0:
            raise TimeoutError("Insufficient time remains for a bounded benchmark.")
        command = [str(PYTHON), "-m", "slitherai.benchmark_observation",
                   "--run", str(RUN), "--model-payload", str(FROZEN_PAYLOAD),
                   "--policy-timing-only", "--maps", "64", "--worms", "16",
                   "--seconds", "60", "--seeds", "1038282",
                   "--chunks", "4", "8", "--device", "cuda", "--out", str(OUTPUT)]
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=CHECKOUT, stdout=stdout, stderr=stderr)
            benchmark_pid = process.pid
            try:
                benchmark_return_code = process.wait(timeout=timeout)
                benchmark_outcome = "exited"
            except subprocess.TimeoutExpired:
                benchmark_timed_out = True
                benchmark_outcome = "timeout"
                cleanup_evidence = kill_verified_benchmark_processes(process.pid)
                if cleanup_evidence.get("cleanup_warning"):
                    benchmark_error = "Timeout cleanup left unverified or live benchmark-related PIDs."
                try:
                    benchmark_return_code = process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    benchmark_return_code = None
    except Exception as exc:
        benchmark_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            try:
                cleanup_evidence = kill_verified_benchmark_processes(process.pid)
                if cleanup_evidence.get("cleanup_warning"):
                    benchmark_error += "; timeout cleanup left unverified or live benchmark-related PIDs"
                process.wait(timeout=3)
            except Exception as kill_exc:
                benchmark_error += f"; child_cleanup={type(kill_exc).__name__}: {kill_exc}"
    finally:
        if touched_control:
            try:
                api("POST", "/api/control", {"pause": bool(control_before["pause"])})
                restore_check_deadline = min(deadline, time.monotonic() + 8.0)
                while time.monotonic() < restore_check_deadline:
                    current_control = read_control()
                    state_after = api("GET", "/api/state")
                    if (current_control.get("pause") == bool(control_before.get("pause"))
                            and current_control.get("stop") == control_before.get("stop")
                            and current_control.get("arena") == control_before.get("arena")):
                        restored = True
                        if control_before.get("pause") or state_after["status"].get("phase") == "training":
                            break
                    time.sleep(0.2)
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"

    process_ids_after = trainer_processes()
    final_control = read_control()
    output_sha = sha256(OUTPUT) if OUTPUT.is_file() else None
    output = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.is_file() else None
    output_validation_error = None
    if output is not None:
        policy = output.get("policy_timing_only")
        rows = [] if not isinstance(policy, dict) else policy.get("timed_episodes", [])
        if (output.get("mode") != "policy_timing_only"
                or not isinstance(rows, list) or len(rows) != 2
                or any(not isinstance(row, dict) for row in rows)
                or [row.get("sensor_chunk") for row in rows] != [4, 8]
                or any(row.get("seed") != 1038282
                       or not isinstance(row.get("wall_ms"), (int, float))
                       or row.get("wall_ms") <= 0
                       for row in rows)):
            output_validation_error = "Output did not match the frozen two-row timing-only contract."
    record = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "window_elapsed_seconds": round(time.monotonic() - (deadline - 180.0), 3),
        "worktree_sha": head,
        "benchmark_module": str(BENCHMARK_MODULE.resolve()),
        "benchmark_module_sha256": sha256(BENCHMARK_MODULE),
        "run": str(RUN), "settings_sha256": EXPECTED_SETTINGS_SHA,
        "model_source": str(SOURCE_PAYLOAD),
        "source_payload_sha_at_freeze": EXPECTED_SOURCE_SHA,
        "frozen_model_payload": str(FROZEN_PAYLOAD), "frozen_payload_sha256": EXPECTED_PAYLOAD_SHA,
        "checkpoint14_sha256": EXPECTED_CHECKPOINT_SHA,
        "generation": EXPECTED_GENERATION, "genome_id": EXPECTED_GENOME_ID,
        "gene_fingerprint": EXPECTED_GENE_SHA, "payload_preflight": payload_source,
        "control_before": control_before, "control_after": final_control,
        "pause_acknowledged": pause_acknowledged, "restored": restored,
        "restore_error": restore_error,
        "state_before": state_before["status"],
        "state_after": None if state_after is None else state_after.get("status"),
        "trainer_processes_before": process_ids_before,
        "trainer_processes_after": process_ids_after,
        "command": command if "command" in locals() else None,
        "benchmark_outcome": benchmark_outcome, "benchmark_return_code": benchmark_return_code,
        "benchmark_timed_out": benchmark_timed_out, "benchmark_pid": benchmark_pid,
        "benchmark_error": benchmark_error, "timeout_cleanup_evidence": cleanup_evidence,
        "output": str(OUTPUT), "output_sha256": output_sha,
        "output_validation_error": output_validation_error,
        "policy_result": None if output is None else output.get("policy_timing_only"),
        "stdout_file": str(stdout_path), "stderr_file": str(stderr_path),
    }
    METADATA.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if (not restored or restore_error or benchmark_timed_out or benchmark_error
            or benchmark_return_code != 0 or output is None or output_validation_error):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
