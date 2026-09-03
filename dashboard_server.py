import json
import os
import subprocess
import sys
import threading
from http.server import (
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SCENARIO_PATH = ROOT / "dashboard_logs" / "dashboard_scenario.json"

MAX_OUTPUT_LINES = 2000


class TaskRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.task_name = None
        self.output = []
        self.return_code = None
        self.stopped = False

    def start(self, task_name, command):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "Another task is already running."

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)

            self.task_name = task_name
            self.return_code = None
            self.stopped = False
            self.output = ["$ " + " ".join(map(str, command))]

            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            thread = threading.Thread(
                target=self._collect_output,
                args=(self.process,),
                daemon=True,
            )
            thread.start()

        return True, f"{task_name} started."

    def _collect_output(self, process):
        if process.stdout is not None:
            for line in process.stdout:
                with self.lock:
                    self.output.append(line.rstrip())
                    self.output = self.output[-MAX_OUTPUT_LINES:]

        return_code = process.wait()

        with self.lock:
            self.return_code = return_code

    def stop(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return False, "No task is running."

            task_name = self.task_name
            self.process.terminate()
            self.stopped = True
            self.output.append(f"Stop requested for: {task_name}")

        return True, f"{task_name} is stopping."

    def status(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None

            return {
                "running": running,
                "task": self.task_name,
                "return_code": self.return_code,
                "output": "\n".join(self.output),
                "stopped": self.stopped,
            }


task_runner = TaskRunner()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(ROOT),
            **kwargs,
        )

    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))

        if content_length > 1_000_000:
            raise ValueError("Request is too large.")

        raw_data = self.rfile.read(content_length)

        if not raw_data:
            return {}

        return json.loads(raw_data.decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/status":
            self.send_json(task_runner.status())
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            payload = self.read_json()

            if path == "/api/train":
                self.start_script(
                    "Tactical DQN training",
                    ROOT / "train_tactical_dqn.py",
                )
                return

            if path == "/api/evaluate":
                self.start_script(
                    "Tactical holdout evaluation",
                    ROOT / "evaluate_tactical_holdout.py",
                )
                return

            if path == "/api/run-scenario":
                self.run_scenario(payload)
                return

            if path == "/api/stop":
                self.stop_task()
                return

            self.send_json(
                {"error": "Unknown API endpoint."},
                status=404,
            )

        except (
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.send_json(
                {"error": str(error)},
                status=400,
            )

    def stop_task(self):
        stopped, message = task_runner.stop()

        self.send_json(
            {
                "stopped": stopped,
                "message": message,
            },
            status=202 if stopped else 409,
        )

    def start_script(self, task_name, script_path):
        started, message = task_runner.start(
            task_name,
            [sys.executable, "-u", str(script_path)],
        )

        self.send_json(
            {
                "started": started,
                "message": message,
            },
            status=202 if started else 409,
        )

    def run_scenario(self, payload):
        scenario = payload.get("scenario")
        policy = payload.get("policy", "dqn")

        if not isinstance(scenario, dict):
            raise ValueError("A valid scenario object is required.")

        if policy not in {"dqn", "heuristic"}:
            raise ValueError("Policy must be dqn or heuristic.")

        SCENARIO_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        SCENARIO_PATH.write_text(
            json.dumps(scenario, indent=2),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            "-u",
            str(ROOT / "run_scenario.py"),
            str(SCENARIO_PATH),
            "--policy",
            policy,
        ]

        started, message = task_runner.start(
            f"Custom scenario ({policy})",
            command,
        )

        self.send_json(
            {
                "started": started,
                "message": message,
            },
            status=202 if started else 409,
        )


def main():
    address = ("127.0.0.1", 8000)
    server = ThreadingHTTPServer(
        address,
        DashboardHandler,
    )

    print(
        "TALON dashboard running at:",
        "http://127.0.0.1:8000/tacew_dashboard.html",
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
