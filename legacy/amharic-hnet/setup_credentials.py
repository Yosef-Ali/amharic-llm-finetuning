#!/usr/bin/env python3
"""
Interactive Setup Script for Amharic Enhanced LLM
Configures Kaggle API and HuggingFace credentials
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n🔹 Step {step_num}: {description}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def check_kaggle_credentials():
    """Check if Kaggle credentials are properly configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        return False, "kaggle.json not found"
    
    # Check permissions
    stat_info = kaggle_json.stat()
    if oct(stat_info.st_mode)[-3:] != '600':
        return False, "Incorrect permissions on kaggle.json"
    
    # Try to load and validate JSON
    try:
        with open(kaggle_json, 'r') as f:
            data = json.load(f)
        if 'username' not in data or 'key' not in data:
            return False, "Invalid kaggle.json format"
        return True, "Kaggle credentials configured correctly"
    except Exception as e:
        return False, f"Error reading kaggle.json: {e}"

def setup_kaggle_credentials():
    """Guide user through Kaggle API setup"""
    print_step(1, "Setting up Kaggle API Credentials")
    
    # Check if already configured
    is_configured, message = check_kaggle_credentials()
    if is_configured:
        print_success(f"Kaggle API already configured: {message}")
        return True
    
    print("\n📋 Kaggle API Setup Instructions:")
    print("1. Visit https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download kaggle.json file")
    
    # Check if user has downloaded the file
    downloads_path = Path.home() / "Downloads" / "kaggle.json"
    
    while True:
        response = input("\n❓ Have you downloaded kaggle.json? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            print("\n⏳ Please download kaggle.json first, then return here.")
            input("Press Enter when ready to continue...")
        else:
            print("Please enter 'y' or 'n'")
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Look for kaggle.json in common locations
    possible_locations = [
        downloads_path,
        Path.cwd() / "kaggle.json",
        Path.home() / "kaggle.json"
    ]
    
    kaggle_json_source = None
    for location in possible_locations:
        if location.exists():
            kaggle_json_source = location
            break
    
    if not kaggle_json_source:
        print("\n❓ Please enter the full path to your kaggle.json file:")
        while True:
            path_input = input("Path: ").strip()
            if path_input:
                kaggle_json_source = Path(path_input)
                if kaggle_json_source.exists():
                    break
                else:
                    print(f"File not found: {path_input}")
            else:
                print("Please enter a valid path")
    
    # Copy and set permissions
    try:
        kaggle_json_dest = kaggle_dir / "kaggle.json"
        
        # Copy file
        import shutil
        shutil.copy2(kaggle_json_source, kaggle_json_dest)
        
        # Set permissions
        os.chmod(kaggle_json_dest, 0o600)
        
        print_success(f"Kaggle credentials installed to {kaggle_json_dest}")
        
        # Verify setup
        is_configured, message = check_kaggle_credentials()
        if is_configured:
            print_success("Kaggle API setup completed successfully!")
            return True
        else:
            print_error(f"Setup verification failed: {message}")
            return False
            
    except Exception as e:
        print_error(f"Error setting up Kaggle credentials: {e}")
        return False

def check_huggingface_token():
    """Check if HuggingFace token is configured"""
    token = os.getenv('HF_TOKEN')
    if token:
        return True, "HuggingFace token found in environment"
    return False, "HF_TOKEN environment variable not set"

def setup_huggingface_token():
    """Guide user through HuggingFace token setup"""
    print_step(2, "Setting up HuggingFace Token")
    
    # Check if already configured
    is_configured, message = check_huggingface_token()
    if is_configured:
        print_success(f"HuggingFace token already configured: {message}")
        return True
    
    print("\n📋 HuggingFace Token Setup Instructions:")
    print("1. Visit https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Choose 'Write' permissions")
    print("4. Copy the generated token")
    
    while True:
        response = input("\n❓ Have you obtained your HuggingFace token? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            print("\n⏳ Please get your HuggingFace token first, then return here.")
            input("Press Enter when ready to continue...")
        else:
            print("Please enter 'y' or 'n'")
    
    # Get token from user
    while True:
        token = input("\n🔑 Please paste your HuggingFace token: ").strip()
        if token:
            break
        print("Please enter a valid token")
    
    # Add to shell profile
    shell_profiles = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".bash_profile"
    ]
    
    # Find existing shell profile
    profile_file = None
    for profile in shell_profiles:
        if profile.exists():
            profile_file = profile
            break
    
    if not profile_file:
        # Default to .zshrc on macOS
        profile_file = Path.home() / ".zshrc"
    
    try:
        # Add export statement
        export_line = f"export HF_TOKEN=\"{token}\"\n"
        
        with open(profile_file, 'a') as f:
            f.write(f"\n# HuggingFace token for Amharic Enhanced LLM\n")
            f.write(export_line)
        
        # Set for current session
        os.environ['HF_TOKEN'] = token
        
        print_success(f"HuggingFace token added to {profile_file}")
        print_warning("Please run 'source ~/.zshrc' or restart your terminal for permanent effect")
        
        return True
        
    except Exception as e:
        print_error(f"Error setting up HuggingFace token: {e}")
        return False

def test_apis():
    """Test both API configurations"""
    print_step(3, "Testing API Configurations")
    
    # Test Kaggle API
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success("Kaggle API test passed")
        else:
            print_error("Kaggle API test failed")
            return False
    except Exception as e:
        print_error(f"Kaggle API test error: {e}")
        return False
    
    # Test HuggingFace (basic check)
    if os.getenv('HF_TOKEN'):
        print_success("HuggingFace token available")
    else:
        print_error("HuggingFace token not available")
        return False
    
    return True

def main():
    """Main setup function"""
    print_header("🇪🇹 Amharic Enhanced LLM - Credential Setup")
    
    print("\n🎯 This script will help you configure:")
    print("   • Kaggle API credentials")
    print("   • HuggingFace authentication token")
    
    # Setup Kaggle
    kaggle_success = setup_kaggle_credentials()
    
    # Setup HuggingFace
    hf_success = setup_huggingface_token()
    
    # Test configurations
    if kaggle_success and hf_success:
        test_success = test_apis()
        
        if test_success:
            print_header("🎉 Setup Complete!")
            print("\n✅ All credentials configured successfully!")
            print("\n🚀 Next steps:")
            print("   1. Run: python quick_start.py --status")
            print("   2. Run: python quick_start.py --phase all")
            print("   3. Monitor: python quick_start.py --phase monitor")
            
            # Ask if user wants to run status check
            response = input("\n❓ Would you like to check system status now? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("\n🔍 Checking system status...")
                try:
                    subprocess.run([sys.executable, 'quick_start.py', '--status'])
                except Exception as e:
                    print_error(f"Error running status check: {e}")
        else:
            print_error("API testing failed. Please check your configurations.")
    else:
        print_error("Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()