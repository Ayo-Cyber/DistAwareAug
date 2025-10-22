#!/bin/bash

# Local CI/CD Test Runner
# This script simulates the GitHub Actions workflows locally

set -e  # Exit on error

echo "=========================================="
echo "🧪 Running Local CI/CD Tests"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2 passed${NC}"
    else
        echo -e "${RED}❌ $2 failed${NC}"
        exit 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Run this from the project root.${NC}"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -e ".[dev]" -q
print_status $? "Dependency installation"
echo ""

# 1. Code Quality Checks
echo "=========================================="
echo "🎨 Code Quality Checks"
echo "=========================================="
echo ""

echo "1️⃣ Checking code formatting with Black..."
black --check distawareaug tests --line-length=100
print_status $? "Black formatting check"
echo ""

echo "2️⃣ Checking import sorting with isort..."
isort --check-only distawareaug tests --profile black
print_status $? "isort import check"
echo ""

echo "3️⃣ Linting with flake8..."
flake8 distawareaug tests --max-line-length=100 --extend-ignore=E203,W503
print_status $? "flake8 linting"
echo ""

# 2. Unit Tests
echo "=========================================="
echo "🧪 Running Unit Tests"
echo "=========================================="
echo ""

echo "Running pytest with coverage..."
pytest tests/ -v --cov=distawareaug --cov-report=term-missing --cov-report=html
print_status $? "Unit tests"
echo ""

# 3. Coverage Report
echo "=========================================="
echo "📊 Coverage Summary"
echo "=========================================="
echo ""
pytest --cov=distawareaug --cov-report=term tests/ -q
echo ""

# 4. Package Build Test
echo "=========================================="
echo "📦 Testing Package Build"
echo "=========================================="
echo ""

echo "Building distribution packages..."
python -m build
print_status $? "Package build"
echo ""

# Final Summary
echo "=========================================="
echo -e "${GREEN}✅ All CI/CD checks passed!${NC}"
echo "=========================================="
echo ""
echo "Coverage report available at: htmlcov/index.html"
echo ""
echo "Next steps:"
echo "  • Review coverage report: open htmlcov/index.html"
echo "  • Commit your changes: git add . && git commit -m 'your message'"
echo "  • Push to trigger GitHub Actions: git push"
echo ""
