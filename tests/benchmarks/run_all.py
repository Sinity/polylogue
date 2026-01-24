"""Run all benchmarks and generate a performance report."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmark_caching
import benchmark_hashing
import benchmark_pipeline
import benchmark_search


def format_results(results: dict) -> str:
    """Format benchmark results for display.

    Args:
        results: Dictionary of benchmark results

    Returns:
        Formatted string representation
    """
    lines = []
    for name, data in results.items():
        if isinstance(data, dict):
            parts = [f"{name}:"]
            for key, value in data.items():
                if isinstance(value, float):
                    parts.append(f"{key}={value:.2f}")
                else:
                    parts.append(f"{key}={value}")
            lines.append(" ".join(parts))
        else:
            lines.append(f"{name}: {data}")
    return "\n".join(lines)


def save_results(results: dict, output_path: Path | None = None):
    """Save benchmark results to JSON file.

    Args:
        results: Benchmark results dictionary
        output_path: Optional output path (defaults to timestamped file)
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"results_{timestamp}.json"

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_path}")


def main():
    """Run all benchmarks and generate report."""
    print("=" * 80)
    print("POLYLOGUE PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print("=" * 80)

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
        }
    }

    # Run hashing benchmarks
    print("\n")
    hashing_results = benchmark_hashing.run_all()
    all_results["hashing"] = hashing_results

    # Run search benchmarks
    print("\n")
    search_results = benchmark_search.run_all()
    all_results["search"] = search_results

    # Run pipeline benchmarks
    print("\n")
    pipeline_results = benchmark_pipeline.run_all()
    all_results["pipeline"] = pipeline_results

    # Run caching benchmarks
    print("\n")
    caching_results = benchmark_caching.run_all()
    all_results["caching"] = caching_results

    # Save results
    save_results(all_results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
