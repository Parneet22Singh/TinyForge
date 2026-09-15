import unittest

from tinyforge.models import ModelRegistry, ModelSpec, HardwareInfo, select_model, default_registry
from tinyforge.training import TrainingConfig, run_pilot, train, evaluate
from tinyforge.compression import quantize, dequantize
from tinyforge.optimization import grid_search


class BackendTests(unittest.TestCase):
    def test_model_selection_respects_memory(self):
        registry = ModelRegistry([ModelSpec("small", memory_gb=1), ModelSpec("large", memory_gb=4)])
        self.assertEqual(select_model(registry, HardwareInfo(2, 2)).name, "small")

    def test_gpu_is_preferred_for_available_hardware(self):
        hardware = HardwareInfo(2, 16, True, "Test GPU", "cuda", 4)
        registry = ModelRegistry([
            ModelSpec("ram-fit", memory_gb=8),
            ModelSpec("gpu-fit", memory_gb=3),
        ])
        self.assertEqual(select_model(registry, hardware).name, "gpu-fit")

    def test_physical_gpu_without_runtime_prefers_cpu_execution(self):
        hardware = HardwareInfo(2, 16, False, "RTX 4050", "cpu", 6.0, None, "2.14.0+cpu", True, "12.9")
        self.assertEqual(hardware.preferred_device, "cpu")
        self.assertTrue(hardware.physical_gpu)

    def test_training_is_deterministic(self):
        config = TrainingConfig(max_steps=3, seed=7)
        self.assertEqual(train([], config), train([], config))
        self.assertEqual(run_pilot(config, samples=10).estimated_steps, 3)

    def test_compression_and_evaluation(self):
        encoded = quantize([-1.0, 0.0, 1.0])
        self.assertEqual(len(dequantize(encoded)), 3)
        self.assertEqual(evaluate([1, 2], [1, 0])["accuracy"], 0.5)

    def test_search(self):
        result = grid_search({"x": [1, 2]}, lambda p: p["x"])
        self.assertEqual(result.params, {"x": 2})

    def test_default_registry_contains_multiple_families(self):
        registry = default_registry()
        self.assertIn("gpt2", registry)
        self.assertIn("qwen2.5-0.5b", registry)
        self.assertEqual(registry.get("gpt2").default_target_modules, ("c_attn", "c_proj"))


if __name__ == "__main__":
    unittest.main()
