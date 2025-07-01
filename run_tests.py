#!/usr/bin/env python3
"""Test runner script for brain-image-implementation."""

import sys
import subprocess
from pathlib import Path


def run_tests(verbose=True, coverage=False, markers=None):
    """Run the test suite with specified options."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])

    if markers:
        cmd.extend(["-m", markers])

    # Add the tests directory
    cmd.append("tests/")

    print(f"Running tests with command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main function to run tests based on command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tests for brain-image-implementation"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Run tests without verbose output"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--slow", action="store_true", help="Include slow tests")

    args = parser.parse_args()

    verbose = not args.no_verbose
    coverage = args.coverage
    markers = []

    if args.unit_only:
        markers.append("unit")
    elif args.integration_only:
        markers.append("integration")

    if not args.slow:
        markers.append("not slow")

    markers_str = " and ".join(markers) if markers else None

    return_code = run_tests(verbose=verbose, coverage=coverage, markers=markers_str)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
