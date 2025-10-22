#!/usr/bin/env python3
"""
Script to automatically fix common linting issues.
Removes unused imports and fixes simple linting errors.
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Main function to fix linting issues."""
    
    print("\n" + "="*60)
    print("🛠️  Auto-fixing Linting Issues")
    print("="*60)
    
    # Install autoflake if not available
    print("\n📦 Installing autoflake...")
    subprocess.run([sys.executable, "-m", "pip", "install", "autoflake"], 
                   capture_output=True)
    
    # Remove unused imports
    success = run_command(
        "autoflake --in-place --remove-all-unused-imports --remove-unused-variables "
        "distawareaug/*.py tests/*.py",
        "Removing unused imports and variables"
    )
    
    if success:
        print("✅ Successfully removed unused imports and variables")
    else:
        print("⚠️  Some files may need manual fixing")
    
    # Format with black
    run_command(
        "black distawareaug tests --line-length=100",
        "Formatting code with Black"
    )
    
    # Sort imports
    run_command(
        "isort distawareaug tests --profile black",
        "Sorting imports with isort"
    )
    
    print("\n" + "="*60)
    print("✅ Auto-fix complete!")
    print("="*60)
    print("\nNow run: sh run_ci_tests.sh")
    print("To verify all issues are fixed.")

if __name__ == "__main__":
    main()
