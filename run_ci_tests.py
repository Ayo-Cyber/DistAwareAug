#!/usr/bin/env python
"""
Local CI/CD Test Runner
Simulates GitHub Actions workflows locally
"""

import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_header(message):
    """Print a formatted header."""
    print(f"\n{'=' * 50}")
    print(f"🎯 {message}")
    print(f"{'=' * 50}\n")


def print_status(returncode, message):
    """Print colored status message."""
    if returncode == 0:
        print(f"{Colors.GREEN}✅ {message} passed{Colors.NC}\n")
        return True
    else:
        print(f"{Colors.RED}❌ {message} failed{Colors.NC}\n")
        return False


def run_command(cmd, check_name):
    """Run a command and check its return code."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return print_status(result.returncode, check_name)
    except Exception as e:
        print(f"{Colors.RED}Error running {check_name}: {e}{Colors.NC}\n")
        return False


def main():
    """Main CI/CD test runner."""
    print_header("Local CI/CD Test Suite")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(f"{Colors.RED}Error: pyproject.toml not found.{Colors.NC}")
        print("Please run this from the project root directory.")
        sys.exit(1)
    
    all_passed = True
    
    # Install dependencies
    print_header("Installing Dependencies")
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "-q"],
        "Dependency installation"
    ):
        print(f"{Colors.YELLOW}Warning: Some dependencies may not be installed{Colors.NC}")
    
    # 1. Code Quality Checks
    print_header("Code Quality Checks")
    
    print("1️⃣ Checking code formatting with Black...")
    if not run_command(
        ["black", "--check", "distawareaug", "tests", "--line-length=100"],
        "Black formatting"
    ):
        print(f"{Colors.YELLOW}💡 Tip: Run 'black distawareaug tests --line-length=100' to fix{Colors.NC}\n")
        all_passed = False
    
    print("2️⃣ Checking import sorting with isort...")
    if not run_command(
        ["isort", "--check-only", "distawareaug", "tests", "--profile", "black"],
        "isort imports"
    ):
        print(f"{Colors.YELLOW}💡 Tip: Run 'isort distawareaug tests --profile black' to fix{Colors.NC}\n")
        all_passed = False
    
    print("3️⃣ Linting with flake8...")
    if not run_command(
        ["flake8", "distawareaug", "tests", "--max-line-length=100", 
         "--extend-ignore=E203,W503"],
        "flake8 linting"
    ):
        all_passed = False
    
    # 2. Unit Tests
    print_header("Running Unit Tests")
    
    try:
        # Try with coverage first
        if not run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v", 
             "--cov=distawareaug", "--cov-report=term-missing", 
             "--cov-report=html"],
            "Unit tests with coverage"
        ):
            all_passed = False
    except:
        # Fallback without coverage
        print(f"{Colors.YELLOW}Running tests without coverage...{Colors.NC}")
        if not run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            "Unit tests"
        ):
            all_passed = False
    
    # 3. Package Build Test
    print_header("Testing Package Build")
    
    print("Building distribution packages...")
    if not run_command(
        [sys.executable, "-m", "build"],
        "Package build"
    ):
        print(f"{Colors.YELLOW}Note: Install 'build' with: pip install build{Colors.NC}\n")
    
    # Final Summary
    print_header("Test Summary")
    
    if all_passed:
        print(f"{Colors.GREEN}✅ All CI/CD checks passed!{Colors.NC}\n")
        print("Next steps:")
        print("  • Review coverage report: open htmlcov/index.html")
        print("  • Commit your changes: git add . && git commit -m 'your message'")
        print("  • Push to trigger GitHub Actions: git push")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ Some checks failed. Please fix the issues above.{Colors.NC}\n")
        print("Quick fixes:")
        print("  • Format code: make format")
        print("  • Run tests: pytest tests/ -v")
        sys.exit(1)


if __name__ == "__main__":
    main()
