#!/usr/bin/env python
"""
Quick test script to verify all example variants work with the new catalogue.
Runs each variant for 3 epochs to verify data loading and training loop work.
"""

import contextlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def test_variant(config_path: Path, test_epochs: int = 3) -> bool:
    """Test a single variant configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {config_path.name}")
    print(f"{'='*70}")

    try:
        # Load config
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Override epochs for quick test
        raw_config["training"]["epochs"] = test_epochs
        raw_config["output_dir"] = f"outputs/test_{config_path.stem}"

        # Write temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(raw_config, f)
            temp_config = f.name

        # Run training via CLI
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "dorothy.cli.main", "train", temp_config],
            capture_output=True,
            text=True,
            timeout=300,
        )

        elapsed = time.time() - t0

        # Check results
        if result.returncode == 0:
            print("\n  Result: SUCCESS")
            print(f"  Time: {elapsed:.1f}s ({elapsed/test_epochs:.1f}s/epoch)")

            # Extract final loss from output
            for line in result.stdout.split("\n"):
                if "loss" in line.lower():
                    print(f"  {line.strip()}")

            return True
        else:
            print(f"\n  Result: FAILED (exit code {result.returncode})")
            print(f"  STDERR: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("\n  Result: TIMEOUT")
        return False
    except Exception as e:
        print("\n  Result: FAILED")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup temp file
        with contextlib.suppress(OSError):
            Path(temp_config).unlink()


def main():
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"

    variants = [
        "variant1_boss_apogee.yaml",
        "variant2_multi_survey.yaml",
        "variant3_multi_labelset.yaml",
        "variant4_all_surveys.yaml",
        "variant5_all_surveys_masked.yaml",
    ]

    results = {}

    print("=" * 70)
    print("DOROTHY Variant Test Suite")
    print("Testing all example configurations with new super-catalogue")
    print("=" * 70)

    for variant in variants:
        config_path = examples_dir / variant
        if config_path.exists():
            results[variant] = test_variant(config_path, test_epochs=3)
        else:
            print(f"\nSkipping {variant} - file not found")
            results[variant] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for variant, result in results.items():
        status = "PASS" if result is True else ("FAIL" if result is False else "SKIP")
        print(f"  {variant}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
