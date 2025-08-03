#!/usr/bin/env python3
"""
Test script for environment configuration.

This script tests the environment variable loading from different locations
and validates the configuration setup.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from env_config import env_config, print_env_status, validate_env


def test_environment_loading():
    """Test environment variable loading."""
    print("🧪 Testing Environment Configuration")
    print("=" * 50)
    
    # Test basic loading
    print("\n1. Testing environment loading...")
    env_config.load_environment()
    
    # Test credential validation
    print("\n2. Validating credentials...")
    validation = validate_env()
    
    for credential, is_valid in validation.items():
        status = "✅ Valid" if is_valid else "❌ Missing"
        print(f"   {credential}: {status}")
    
    # Test configuration values
    print("\n3. Testing configuration values...")
    test_vars = [
        ('HUGGINGFACE_TOKEN', 'Hugging Face Token'),
        ('KAGGLE_USERNAME', 'Kaggle Username'),
        ('KAGGLE_KEY', 'Kaggle API Key'),
        ('MODEL_DIR', 'Model Directory'),
        ('DATA_DIR', 'Data Directory'),
        ('BATCH_SIZE', 'Batch Size'),
        ('LEARNING_RATE', 'Learning Rate'),
        ('DEVICE', 'Device'),
    ]
    
    for var_name, description in test_vars:
        value = env_config.get(var_name)
        status = "✅ Set" if value else "❌ Not set"
        display_value = value if value and 'TOKEN' not in var_name and 'KEY' not in var_name else "[HIDDEN]"
        print(f"   {description}: {status} ({display_value})")
    
    # Print full status
    print("\n4. Full environment status:")
    print_env_status()
    
    # Test path creation
    print("\n5. Testing path creation...")
    test_paths = ['MODEL_DIR', 'DATA_DIR', 'LOG_DIR']
    
    for path_var in test_paths:
        path = env_config.get_path(path_var, create=True)
        if path:
            status = "✅ Created" if path.exists() else "❌ Failed"
            print(f"   {path_var}: {status} ({path})")
    
    print("\n" + "=" * 50)
    print("✅ Environment configuration test completed!")


def test_subdirectory_access():
    """Test accessing environment from subdirectory context."""
    print("\n🔍 Testing Subdirectory Access")
    print("=" * 50)
    
    # Show current working directory
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    print(f"Current working directory: {current_dir}")
    print(f"Script directory: {script_dir}")
    
    # Test .env file discovery
    search_paths = [
        script_dir / ".env",
        script_dir.parent / ".env",
        Path.home() / ".env",
    ]
    
    print("\nSearching for .env files:")
    for i, path in enumerate(search_paths, 1):
        exists = "✅ Found" if path.exists() else "❌ Not found"
        print(f"   {i}. {path}: {exists}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    try:
        test_environment_loading()
        test_subdirectory_access()
        
        print("\n🎉 All tests completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Copy .env.example to .env")
        print("   2. Fill in your actual credentials")
        print("   3. Run this test again to verify setup")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)