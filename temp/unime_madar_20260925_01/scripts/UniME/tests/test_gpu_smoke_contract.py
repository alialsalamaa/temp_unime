"""Dependency-free guards for the separate GPU parity preflight's scope."""
import ast
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_inference_gpu.py"
spec = importlib.util.spec_from_file_location("smoke_inference_gpu_contract", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class GPUParityScopeTests(unittest.TestCase):
    def protocol(self):
        selection = [f"validation_{index}" for index in range(80)]
        return {"case_ids": {"selection80": selection, "val": selection + ["development"],
                             "train": ["train"], "test": ["test"]},
                "mapped_ids": {"selection80": ["mapped_first"]},
                "processed_to_case": {"mapped_first": selection[0]}}

    def test_fixed_first_validation_case(self):
        self.assertEqual(smoke.choose_case(self.protocol()), ("validation_0", "mapped_first"))

    def test_rejects_test_or_training_overlap(self):
        for population in ("train", "test"):
            protocol = self.protocol()
            protocol["case_ids"][population].append("validation_0")
            with self.assertRaisesRegex(ValueError, "exclusively"):
                smoke.choose_case(protocol)

    def test_rejects_nonvalidation_or_wrong_mapping(self):
        protocol = self.protocol()
        protocol["case_ids"]["val"].remove("validation_0")
        with self.assertRaises(ValueError):
            smoke.choose_case(protocol)
        protocol = self.protocol()
        protocol["processed_to_case"]["mapped_first"] = "test"
        with self.assertRaises(ValueError):
            smoke.choose_case(protocol)

    def test_rejects_incomplete_selection_population(self):
        protocol = self.protocol()
        protocol["case_ids"]["selection80"].pop()
        with self.assertRaises(ValueError):
            smoke.choose_case(protocol)

    def test_metric_schema_finite_and_range(self):
        smoke.require_scores({"WT": 0., "TC": 1., "ET": .5}, ("WT", "TC", "ET"))
        for value in (float("nan"), float("inf"), -1., 1.1):
            with self.assertRaises(ValueError):
                smoke.require_scores({"WT": value}, ("WT",))
        with self.assertRaises(ValueError):
            smoke.require_scores({"WT": .5}, ("WT", "TC", "ET"))

    def test_script_has_no_training_or_scheduler_calls(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(name.endswith(("backward", "optimizer.step", "subprocess.run",
                                            "subprocess.Popen", "os.system", "torch.save")) for name in calls))
        self.assertIn("write_json_exclusive", calls)
        self.assertIn("build_model", calls)
        self.assertIn("accelerator.prepare", calls)


if __name__ == "__main__":
    unittest.main()
