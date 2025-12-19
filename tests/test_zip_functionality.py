"""
Test script to verify the zip functionality works correctly.

This script tests:
1. Running a quick simulated experiment
2. Creating a zip archive
3. Validating zip contents
"""

import subprocess
import sys
import zipfile
from pathlib import Path
import json


def test_zip_functionality():
    """Test the zip results functionality."""
    print("=" * 80)
    print("Testing Kaggle Zip Functionality")
    print("=" * 80)

    # Step 1: Run a quick experiment with zip
    print("\n[1/3] Running simulated experiment with --zip-results...")
    cmd = [
        sys.executable,
        "main.py",
        "run",
        "--dataset",
        "mnist",
        "--budget",
        "2",
        "--simulation",
        "--zip-results",
        "--output-dir",
        ".",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Experiment completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed: {e}")
        print(f"Output: {e.output}")
        return False

    # Step 2: Find the created zip file
    print("\n[2/3] Locating zip file...")
    zip_files = list(Path(".").glob("*_results_*.zip"))

    if not zip_files:
        print("✗ No zip file found!")
        return False

    latest_zip = sorted(zip_files, key=lambda x: x.stat().st_mtime)[-1]
    zip_size_mb = latest_zip.stat().st_size / (1024 * 1024)
    print(f"✓ Found zip file: {latest_zip.name}")
    print(f"  Size: {zip_size_mb:.2f} MB")

    # Step 3: Validate zip contents
    print("\n[3/3] Validating zip contents...")

    with zipfile.ZipFile(latest_zip, "r") as zf:
        file_list = zf.namelist()
        print(f"✓ Zip contains {len(file_list)} files:")

        for file in file_list:
            print(f"  - {file}")

        # Check for required files
        required_patterns = [
            lambda x: x.endswith("_summary.json"),
            lambda x: x.endswith("_history.json"),
            lambda x: x == "README.md",
            lambda x: "checkpoint" in x and x.endswith(".json"),
        ]

        for i, pattern in enumerate(required_patterns):
            if not any(pattern(f) for f in file_list):
                print(f"✗ Missing required file pattern {i+1}")
                return False

        print("\n✓ All required files present")

        # Validate JSON files can be parsed
        print("\n[Extra] Validating JSON files...")
        for file in file_list:
            if file.endswith(".json"):
                try:
                    content = zf.read(file)
                    json.loads(content)
                    print(f"✓ Valid JSON: {file}")
                except json.JSONDecodeError as e:
                    print(f"✗ Invalid JSON in {file}: {e}")
                    return False

    # Summary
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print(f"✓ Zip file ready for download: {latest_zip.name}")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_zip_functionality()
    sys.exit(0 if success else 1)
