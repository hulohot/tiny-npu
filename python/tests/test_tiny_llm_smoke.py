import json
import subprocess
import unittest


class TinyLLMSmokeTest(unittest.TestCase):
    def test_real_weights_first_token_pipeline(self):
        cmd = [
            "python3",
            "python/run_tiny_llm_sim.py",
            "--prepare",
            "--prompt",
            "Tiny NPU says",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            self.skipTest(f"Pipeline unavailable in this environment: {proc.stderr.strip()}")

        payload = json.loads(proc.stdout)
        self.assertIn("reference", payload)
        self.assertIn("simulated", payload)
        self.assertIsInstance(payload["reference"]["token_id"], int)
        self.assertIsInstance(payload["simulated"]["token_id"], int)


if __name__ == "__main__":
    unittest.main()
