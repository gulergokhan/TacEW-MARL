import sys
import time
import unittest

from dashboard_server import TaskRunner


class TestTaskRunner(unittest.TestCase):

    def wait_for_completion(self, runner, timeout=2.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            status = runner.status()

            if status["return_code"] is not None:
                return status

            time.sleep(0.01)

        self.fail("Dashboard task did not finish in time.")

    def test_task_runner_captures_command_output(self):
        runner = TaskRunner()
        started, _ = runner.start(
            "test task",
            [sys.executable, "-c", "print('ready')"],
        )

        self.assertTrue(started)
        status = self.wait_for_completion(runner)
        self.assertEqual(status["return_code"], 0)
        self.assertIn("ready", status["output"])

    def test_task_runner_rejects_parallel_task(self):
        runner = TaskRunner()
        started, _ = runner.start(
            "long task",
            [
                sys.executable,
                "-c",
                "import time; time.sleep(0.5)",
            ],
        )
        second_started, message = runner.start(
            "second task",
            [sys.executable, "-c", "print('second')"],
        )

        self.assertTrue(started)
        self.assertFalse(second_started)
        self.assertIn("already running", message)
        stopped, _ = runner.stop()
        self.assertTrue(stopped)
        self.wait_for_completion(runner)


if __name__ == "__main__":
    unittest.main()
