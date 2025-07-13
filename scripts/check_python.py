#!/usr/bin/env python3
"""Check Python version and installation."""

import sys
import subprocess
import os

def check_python_versions():
    """Check available Python versions."""
    print("🔍 Checking Python installations...")
    
    versions = ['python3.11', 'python3.10', 'python3.9', 'python3', 'python']
    
    for version in versions:
        try:
            result = subprocess.run([version, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {version}: {result.stdout.strip()}")
            else:
                print(f"❌ {version}: Not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {version}: Not found")
    
    print(f"\n📍 Current Python: {sys.executable}")
    print(f"📍 Python version: {sys.version}")
    
    # Check if we meet the requirement
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 11:
        print("✅ Python version meets requirements (3.11+)")
        return True
    else:
        print(f"❌ Python version {major}.{minor} does not meet requirements (need 3.11+)")
        return False

def check_pip():
    """Check pip installation."""
    print("\n📦 Checking pip...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ pip: {result.stdout.strip()}")
            return True
        else:
            print("❌ pip: Not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ pip: Not found")
        return False

def check_venv():
    """Check virtual environment."""
    print("\n🏠 Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        print(f"📍 Virtual env: {sys.prefix}")
        return True
    else:
        print("❌ Not running in virtual environment")
        return False

def main():
    """Main check function."""
    print("🐍 Python Environment Check")
    print("=" * 40)
    
    python_ok = check_python_versions()
    pip_ok = check_pip()
    venv_ok = check_venv()
    
    print("\n" + "=" * 40)
    print("📋 Summary:")
    print(f"   Python 3.11+: {'✅' if python_ok else '❌'}")
    print(f"   pip: {'✅' if pip_ok else '❌'}")
    print(f"   Virtual env: {'✅' if venv_ok else '❌'}")
    
    if not python_ok:
        print("\n💡 To install Python 3.11:")
        print("   brew install python@3.11")
        print("   # or download from https://www.python.org/downloads/")
    
    if not venv_ok:
        print("\n💡 To create virtual environment:")
        print("   python3.11 -m venv venv")
        print("   source venv/bin/activate")
    
    return python_ok and pip_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 