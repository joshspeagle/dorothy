"""
Performance benchmarks for dynamic masking.

These tests measure the overhead of the masking augmentations to ensure
they don't significantly impact training throughput.
"""

import time

import pytest
import torch

from dorothy.data.augmentation import DynamicInputMasking, DynamicLabelMasking


class TestLabelMaskingPerformance:
    """Performance benchmarks for DynamicLabelMasking."""

    @pytest.fixture
    def label_masking(self):
        """Create label masking instance."""
        return DynamicLabelMasking(
            p_labelset_min=0.3,
            p_labelset_max=1.0,
            p_label_min=0.3,
            p_label_max=1.0,
        )

    @pytest.mark.parametrize("batch_size", [64, 256, 512, 1024])
    @pytest.mark.parametrize("n_params", [11, 22])
    def test_label_masking_timing(self, label_masking, batch_size, n_params):
        """Benchmark label masking overhead."""
        y = torch.rand(batch_size, 3, n_params)
        y[:, 2, :] = (torch.rand(batch_size, n_params) > 0.2).float()  # 80% valid

        # Warmup
        for _ in range(10):
            _ = label_masking(y)

        # Timed runs
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = label_masking(y)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_runs) * 1000
        print(
            f"Label masking: batch={batch_size}, params={n_params}, avg={avg_ms:.3f}ms"
        )

        # Assert reasonable overhead (< 5ms per batch for CPU)
        assert avg_ms < 5.0, f"Label masking too slow: {avg_ms:.3f}ms"

    @pytest.mark.parametrize("n_labelsets", [1, 2, 4])
    def test_multi_labelset_timing(self, label_masking, n_labelsets):
        """Benchmark multi-labelset masking."""
        batch_size = 512
        n_params = 11

        y_dict = {
            f"source_{i}": torch.rand(batch_size, 3, n_params)
            for i in range(n_labelsets)
        }
        for v in y_dict.values():
            v[:, 2, :] = (torch.rand(batch_size, n_params) > 0.2).float()

        # Warmup
        for _ in range(10):
            _ = label_masking(y_dict)

        # Timed runs
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = label_masking(y_dict)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_runs) * 1000
        print(f"Label masking ({n_labelsets} sets): avg={avg_ms:.3f}ms")

        # Should scale roughly linearly
        assert avg_ms < 5.0 * n_labelsets, "Multi-labelset masking too slow"


class TestInputMaskingPerformance:
    """Performance benchmarks for DynamicInputMasking."""

    @pytest.fixture
    def input_masking(self):
        """Create input masking instance."""
        return DynamicInputMasking(
            p_survey_min=0.3,
            p_survey_max=1.0,
            f_max=0.5,
            p_block_min=0.3,
            p_block_max=1.0,
        )

    @pytest.mark.parametrize("batch_size", [64, 256, 512])
    @pytest.mark.parametrize("n_wavelengths", [3473, 4506, 7650])
    def test_input_masking_timing(self, input_masking, batch_size, n_wavelengths):
        """Benchmark input masking overhead."""
        X = {"test_survey": torch.rand(batch_size, 3, n_wavelengths)}
        X["test_survey"][:, 2, :] = (
            torch.rand(batch_size, n_wavelengths) > 0.1
        ).float()

        n_wavelengths_dict = {"test_survey": n_wavelengths}

        # Warmup
        for _ in range(10):
            _ = input_masking(X, n_wavelengths_dict)

        # Timed runs
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = input_masking(X, n_wavelengths_dict)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_runs) * 1000
        print(
            f"Input masking: batch={batch_size}, N={n_wavelengths}, avg={avg_ms:.3f}ms"
        )

        # Assert reasonable overhead (< 50ms per batch for CPU)
        assert avg_ms < 50.0, f"Input masking too slow: {avg_ms:.3f}ms"

    @pytest.mark.parametrize("n_surveys", [1, 2, 4])
    def test_multi_survey_timing(self, input_masking, n_surveys):
        """Benchmark multi-survey masking."""
        batch_size = 256
        n_wavelengths = 4506

        X = {
            f"survey_{i}": torch.rand(batch_size, 3, n_wavelengths)
            for i in range(n_surveys)
        }
        for v in X.values():
            v[:, 2, :] = (torch.rand(batch_size, n_wavelengths) > 0.1).float()

        n_wavelengths_dict = {f"survey_{i}": n_wavelengths for i in range(n_surveys)}

        # Warmup
        for _ in range(5):
            _ = input_masking(X, n_wavelengths_dict)

        # Timed runs
        n_runs = 20
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = input_masking(X, n_wavelengths_dict)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_runs) * 1000
        print(f"Input masking ({n_surveys} surveys): avg={avg_ms:.3f}ms")

        # Should scale roughly linearly with number of surveys
        assert avg_ms < 30.0 * n_surveys, "Multi-survey masking too slow"


class TestCombinedMaskingOverhead:
    """Test combined overhead of label and input masking."""

    def test_overhead_vs_batch_loading(self):
        """Compare masking overhead to typical batch loading time."""
        # Typical batch loading time (HDF5 read + preprocessing) is ~5-20ms
        batch_load_time_ms = 10.0

        # Create masking instances
        label_masking = DynamicLabelMasking()
        input_masking = DynamicInputMasking()

        # Realistic batch sizes
        batch_size = 512
        n_params = 11
        n_wavelengths = 4506

        y = torch.rand(batch_size, 3, n_params)
        y[:, 2, :] = 1.0

        X = {"boss": torch.rand(batch_size, 3, n_wavelengths)}
        X["boss"][:, 2, :] = 1.0

        # Warmup
        for _ in range(10):
            _ = label_masking(y)
            _ = input_masking(X, {"boss": n_wavelengths})

        # Measure combined overhead
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = label_masking(y)
            _ = input_masking(X, {"boss": n_wavelengths})
        total_overhead = (time.perf_counter() - start) / n_runs * 1000

        overhead_pct = (total_overhead / batch_load_time_ms) * 100
        print(
            f"Masking overhead: {total_overhead:.2f}ms ({overhead_pct:.1f}% of batch load)"
        )

        # Assert overhead is small relative to batch loading
        # Use 200% threshold to account for CI environment variability
        assert overhead_pct < 200, f"Masking overhead too high: {overhead_pct:.1f}%"

    def test_realistic_training_scenario(self):
        """Test masking in a realistic multi-survey, multi-label scenario."""
        label_masking = DynamicLabelMasking()
        input_masking = DynamicInputMasking()

        batch_size = 512

        # Multi-survey input
        X = {
            "boss": torch.rand(batch_size, 3, 4506),
            "desi": torch.rand(batch_size, 3, 7650),
            "lamost_lrs": torch.rand(batch_size, 3, 3473),
        }
        for v in X.values():
            v[:, 2, :] = (torch.rand_like(v[:, 2, :]) > 0.1).float()

        n_wavelengths = {"boss": 4506, "desi": 7650, "lamost_lrs": 3473}

        # Multi-labelset labels
        y = {
            "apogee": torch.rand(batch_size, 3, 11),
            "galah": torch.rand(batch_size, 3, 11),
        }
        for v in y.values():
            v[:, 2, :] = (torch.rand_like(v[:, 2, :]) > 0.2).float()

        # Warmup
        for _ in range(5):
            _ = input_masking(X, n_wavelengths)
            _ = label_masking(y)

        # Timed runs
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = input_masking(X, n_wavelengths)
            _ = label_masking(y)
        elapsed = (time.perf_counter() - start) / n_runs * 1000

        print(f"Realistic scenario (3 surveys, 2 labelsets): {elapsed:.2f}ms per batch")

        # Should still be reasonable
        assert elapsed < 100, f"Realistic scenario too slow: {elapsed:.2f}ms"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """GPU performance tests for masking."""

    def test_gpu_label_masking(self):
        """Test label masking performance on GPU."""
        label_masking = DynamicLabelMasking()
        batch_size = 512

        y = torch.rand(batch_size, 3, 11, device="cuda")
        y[:, 2, :] = (torch.rand(batch_size, 11, device="cuda") > 0.2).float()

        # Warmup
        for _ in range(20):
            _ = label_masking(y)
            torch.cuda.synchronize()

        # Timed runs
        n_runs = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = label_masking(y)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000

        print(f"GPU label masking: {elapsed:.3f}ms per batch")

    def test_gpu_input_masking(self):
        """Test input masking performance on GPU."""
        input_masking = DynamicInputMasking()
        batch_size = 512

        X = {"desi": torch.rand(batch_size, 3, 7650, device="cuda")}
        X["desi"][:, 2, :] = (torch.rand(batch_size, 7650, device="cuda") > 0.1).float()

        # Warmup
        for _ in range(10):
            _ = input_masking(X, {"desi": 7650})
            torch.cuda.synchronize()

        # Timed runs
        n_runs = 50
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = input_masking(X, {"desi": 7650})
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000

        print(f"GPU input masking: {elapsed:.3f}ms per batch")
