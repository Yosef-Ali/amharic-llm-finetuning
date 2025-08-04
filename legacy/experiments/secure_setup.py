#!/usr/bin/env python3
"""
Secure Environment Setup Script
Amharic Enhanced LLM System

This script helps users securely configure their environment variables
without exposing credentials in the codebase.
"""

import os
import sys
from pathlib import Path
import getpass
from typing import Optional

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent

def check_existing_env() -> bool:
    """Check if .env file already exists."""
    env_path = get_project_root() / ".env"
    return env_path.exists()

def create_secure_env_file() -> bool:
    """Create a secure .env file with user input."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"
    
    print("🔒 Secure Environment Setup")
    print("=" * 50)
    
    if env_path.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled.")
            return False
    
    if not example_path.exists():
        print("❌ .env.example template not found!")
        return False
    
    print("\n📝 Please provide your API credentials:")
    print("(Press Enter to skip optional fields)\n")
    
    # Collect credentials securely
    credentials = {}
    
    # Kaggle credentials
    print("🔹 Kaggle API Credentials")
    print("   Get from: https://www.kaggle.com/account")
    kaggle_username = input("   Username: ").strip()
    if kaggle_username:
        kaggle_key = getpass.getpass("   API Key: ").strip()
        if kaggle_key:
            credentials['KAGGLE_USERNAME'] = kaggle_username
            credentials['KAGGLE_KEY'] = kaggle_key
    
    print("\n🔹 HuggingFace Token")
    print("   Get from: https://huggingface.co/settings/tokens")
    hf_token = getpass.getpass("   Token: ").strip()
    if hf_token:
        credentials['HUGGINGFACE_TOKEN'] = hf_token
        credentials['HF_TOKEN'] = hf_token
        credentials['HF_API_TOKEN'] = hf_token
    
    # Read template and create .env file
    try:
        with open(example_path, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders with actual values or keep placeholders
        env_content = template_content
        
        # Replace known placeholders
        replacements = {
            'your_kaggle_username_here': credentials.get('KAGGLE_USERNAME', 'your_kaggle_username_here'),
            'your_kaggle_api_key_here': credentials.get('KAGGLE_KEY', 'your_kaggle_api_key_here'),
            'your_huggingface_token_here': credentials.get('HUGGINGFACE_TOKEN', 'your_huggingface_token_here'),
        }
        
        for placeholder, value in replacements.items():
            env_content = env_content.replace(placeholder, value)
        
        # Write the .env file
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        # Set secure permissions (readable only by owner)
        os.chmod(env_path, 0o600)
        
        print(f"\n✅ Created secure .env file: {env_path}")
        print(f"🔒 File permissions set to 600 (owner read/write only)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_environment() -> bool:
    """Validate the environment setup."""
    print("\n🧪 Validating environment setup...")
    
    try:
        # Import and test environment loading
        sys.path.append(str(get_project_root() / "amharic-hnet"))
        from env_config import EnvironmentConfig
        
        env_config = EnvironmentConfig()
        env_config.load_environment()
        
        # Check for required variables
        required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'HUGGINGFACE_TOKEN']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("   You can add them later to your .env file")
        else:
            print("✅ All required environment variables are set")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def show_next_steps():
    """Show next steps after setup."""
    print("\n🚀 Next Steps:")
    print("=" * 30)
    print("1. Test your setup:")
    print("   python amharic-hnet/test_env.py")
    print("\n2. Check system status:")
    print("   python amharic-hnet/quick_start.py --status")
    print("\n3. Run evaluation (works offline):")
    print("   python amharic-hnet/quick_start.py --phase eval")
    print("\n4. For full pipeline (requires credentials):")
    print("   python amharic-hnet/quick_start.py --phase all")
    print("\n📚 Documentation:")
    print("   - SECURITY_FIX_COMPLETE.md")
    print("   - ENVIRONMENT_SETUP.md")
    print("   - PRODUCTION_SETUP_GUIDE.md")

def main():
    """Main setup function."""
    print("🔒 Amharic Enhanced LLM - Secure Environment Setup")
    print("=" * 60)
    print("This script will help you securely configure your API credentials.")
    print("Your credentials will be stored locally and never committed to git.\n")
    
    # Check if already configured
    if check_existing_env():
        print("ℹ️  Environment file already exists.")
        response = input("   Continue with reconfiguration? (y/N): ")
        if response.lower() != 'y':
            print("✅ Setup skipped. Use existing configuration.")
            show_next_steps()
            return
    
    # Create secure environment file
    if create_secure_env_file():
        validate_environment()
        show_next_steps()
        print("\n🎉 Secure setup complete!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()