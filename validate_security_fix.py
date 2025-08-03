#!/usr/bin/env python3
"""
Security Fix Validation Script
Amharic Enhanced LLM System

This script validates that all security fixes have been properly implemented
and the environment is correctly configured.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent

def check_exposed_credentials() -> Tuple[bool, List[str]]:
    """Check for any exposed credentials in the codebase."""
    print("🔍 Scanning for exposed credentials...")
    
    project_root = get_project_root()
    issues = []
    
    # Patterns for actual credential formats (not class names)
    patterns = {
        'HuggingFace Token': r'hf_[A-Za-z0-9]{34}',
        'Kaggle API Key': r'[0-9a-f]{32}',
        'AWS Access Key': r'AKIA[0-9A-Z]{16}',
        'GitHub Token': r'ghp_[A-Za-z0-9]{36}',
    }
    
    # Files to check (exclude library/dependency directories)
    file_extensions = ['.py', '.md', '.txt', '.json', '.env', '.env.example', '.env.save']
    exclude_dirs = ['.git', '__pycache__', 'venv', 'node_modules', 'site-packages', '.venv']
    
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and any(file_path.name.endswith(ext) for ext in file_extensions):
            if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Skip obvious placeholders
                        if any(placeholder in match.lower() for placeholder in 
                               ['your_', 'example', 'placeholder', 'token_here', 'key_here']):
                            continue
                        issues.append(f"{file_path}: {pattern_name} - {match[:10]}...")
            except Exception:
                continue
    
    return len(issues) == 0, issues

def check_gitignore() -> Tuple[bool, List[str]]:
    """Check that .gitignore properly excludes .env files."""
    print("📝 Checking .gitignore configuration...")
    
    project_root = get_project_root()
    issues = []
    
    # Check root .gitignore
    root_gitignore = project_root / '.gitignore'
    if not root_gitignore.exists():
        issues.append("Root .gitignore file missing")
    else:
        with open(root_gitignore, 'r') as f:
            content = f.read()
            if '.env' not in content:
                issues.append("Root .gitignore doesn't exclude .env files")
    
    # Check subdirectory .gitignore
    sub_gitignore = project_root / 'amharic-hnet' / '.gitignore'
    if sub_gitignore.exists():
        with open(sub_gitignore, 'r') as f:
            content = f.read()
            if '.env' not in content:
                issues.append("Subdirectory .gitignore doesn't exclude .env files")
    
    return len(issues) == 0, issues

def check_env_structure() -> Tuple[bool, List[str]]:
    """Check the .env file structure and configuration."""
    print("🏗️ Checking environment structure...")
    
    project_root = get_project_root()
    issues = []
    
    # Check for actual .env file (should not exist in repo)
    env_file = project_root / '.env'
    if env_file.exists():
        issues.append("Actual .env file exists in repository (should be user-created only)")
    
    # Check for .env.example
    env_example = project_root / '.env.example'
    if not env_example.exists():
        issues.append("Missing .env.example template file")
    else:
        with open(env_example, 'r') as f:
            content = f.read()
            required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'HUGGINGFACE_TOKEN']
            for var in required_vars:
                if var not in content:
                    issues.append(f"Missing {var} in .env.example")
    
    # Check subdirectory structure (look for actual .env files, not .gitignore entries)
    sub_env_files = list((project_root / 'amharic-hnet').glob('.env'))
    if sub_env_files:
        for env_file in sub_env_files:
            if env_file.is_file():
                issues.append(f"Subdirectory .env file exists: {env_file} (should use root .env only)")
    
    return len(issues) == 0, issues

def check_environment_loading() -> Tuple[bool, List[str]]:
    """Check that environment loading is properly configured."""
    print("⚙️ Checking environment loading configuration...")
    
    project_root = get_project_root()
    issues = []
    
    # Check env_config.py
    env_config_path = project_root / 'amharic-hnet' / 'env_config.py'
    if not env_config_path.exists():
        issues.append("Missing env_config.py file")
    else:
        with open(env_config_path, 'r') as f:
            content = f.read()
            if 'load_dotenv' not in content:
                issues.append("env_config.py doesn't use load_dotenv")
            if 'project_root / ".env"' not in content:
                issues.append("env_config.py doesn't check root .env file")
    
    return len(issues) == 0, issues

def check_security_documentation() -> Tuple[bool, List[str]]:
    """Check that security documentation is present and complete."""
    print("📚 Checking security documentation...")
    
    project_root = get_project_root()
    issues = []
    
    # Check for security fix documentation
    security_doc = project_root / 'SECURITY_FIX_COMPLETE.md'
    if not security_doc.exists():
        issues.append("Missing SECURITY_FIX_COMPLETE.md documentation")
    
    # Check for secure setup script
    secure_setup = project_root / 'secure_setup.py'
    if not secure_setup.exists():
        issues.append("Missing secure_setup.py script")
    
    return len(issues) == 0, issues

def run_validation() -> bool:
    """Run all validation checks."""
    print("🔒 Amharic Enhanced LLM - Security Fix Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    checks = [
        ("Exposed Credentials", check_exposed_credentials),
        ("GitIgnore Configuration", check_gitignore),
        ("Environment Structure", check_env_structure),
        ("Environment Loading", check_environment_loading),
        ("Security Documentation", check_security_documentation),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            passed, issues = check_func()
            results[check_name] = (passed, issues)
            
            if passed:
                print(f"✅ {check_name}: PASSED")
            else:
                print(f"❌ {check_name}: FAILED")
                for issue in issues:
                    print(f"   - {issue}")
                all_checks_passed = False
        except Exception as e:
            print(f"⚠️ {check_name}: ERROR - {e}")
            all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 ALL SECURITY CHECKS PASSED!")
        print("\n✅ Security Fix Status: COMPLETE")
        print("✅ Environment Configuration: SECURE")
        print("✅ Credential Exposure: NONE DETECTED")
        print("\n🚀 Your system is ready for secure development!")
        
        print("\n📋 Next Steps:")
        print("1. Run: python secure_setup.py (to configure credentials)")
        print("2. Test: python amharic-hnet/test_env.py")
        print("3. Start: python amharic-hnet/quick_start.py --status")
    else:
        print("⚠️ SECURITY ISSUES DETECTED!")
        print("\nPlease address the issues above before proceeding.")
        print("\n📚 Documentation:")
        print("- SECURITY_FIX_COMPLETE.md")
        print("- ENVIRONMENT_SETUP.md")
    
    return all_checks_passed

def main():
    """Main validation function."""
    success = run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()