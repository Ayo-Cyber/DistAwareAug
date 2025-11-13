#!/bin/bash
# Publish DistAwareAug v0.2.0 to PyPI

set -e  # Exit on error

echo "================================================================================"
echo "Publishing DistAwareAug v0.2.0 to PyPI"
echo "================================================================================"

# 1. Clean old builds
echo -e "\n📦 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info/ distawareaug.egg-info/

# 2. Run tests
echo -e "\n🧪 Running tests..."
python -m pytest tests/ -v --tb=short

# 3. Build package
echo -e "\n🔨 Building package..."
python -m build

# 4. Check package
echo -e "\n🔍 Checking package..."
python -m twine check dist/*

# 5. Upload to PyPI
echo -e "\n🚀 Uploading to PyPI..."
echo "⚠️  You will be prompted for your PyPI API token"
python -m twine upload dist/*

echo -e "\n================================================================================"
echo "✅ Published successfully!"
echo "Install with: pip install distawareaug==0.2.0"
echo "================================================================================"
