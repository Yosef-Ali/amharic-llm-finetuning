#!/usr/bin/env python3
"""
Environment Configuration Module for Amharic H-Net

This module handles loading environment variables from .env files
at different directory levels and provides a centralized configuration.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    load_dotenv = None


class EnvironmentConfig:
    """Centralized environment configuration for Amharic H-Net."""
    
    def __init__(self, auto_load: bool = True):
        """Initialize environment configuration.
        
        Args:
            auto_load: Whether to automatically load environment variables
        """
        self._loaded = False
        if auto_load:
            self.load_environment()
    
    def load_environment(self) -> bool:
        """Load environment variables from .env files.
        
        Searches for .env files in the following order:
        1. Current directory
        2. Parent directory (project root)
        3. Home directory
        
        Returns:
            bool: True if any .env file was loaded successfully
        """
        if self._loaded:
            return True
            
        if load_dotenv is None:
            print("Warning: python-dotenv not available. Using system environment variables only.")
            self._loaded = True
            return False
        
        # Get current script directory
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        home_dir = Path.home()
        
        # Search paths for .env files
        search_paths = [
            current_dir / ".env",           # ./amharic-hnet/.env
            project_root / ".env",          # ./.env (project root)
            home_dir / ".env",              # ~/.env
        ]
        
        loaded_any = False
        
        for env_path in search_paths:
            if env_path.exists():
                try:
                    load_dotenv(env_path, override=False)  # Don't override existing vars
                    print(f"✅ Loaded environment from: {env_path}")
                    loaded_any = True
                except Exception as e:
                    print(f"⚠️ Failed to load {env_path}: {e}")
        
        if not loaded_any:
            print("ℹ️ No .env files found. Using system environment variables.")
        
        self._loaded = True
        return loaded_any
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        if not self._loaded:
            self.load_environment()
        return os.getenv(key, default)
    
    def get_required(self, key: str) -> str:
        """Get required environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If environment variable is not set
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Boolean value
        """
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Integer value
        """
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid integer value for {key}: {value}. Using default: {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Float value
        """
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid float value for {key}: {value}. Using default: {default}")
            return default
    
    def get_path(self, key: str, default: Optional[str] = None, create: bool = False) -> Optional[Path]:
        """Get path environment variable.
        
        Args:
            key: Environment variable name
            default: Default path if not found
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Path object or None
        """
        value = self.get(key, default)
        if value is None:
            return None
        
        path = Path(value).expanduser().resolve()
        
        if create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {path}")
            except Exception as e:
                print(f"⚠️ Failed to create directory {path}: {e}")
        
        return path
    
    def validate_credentials(self) -> Dict[str, bool]:
        """Validate that required credentials are available.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'huggingface_token': bool(self.get('HUGGINGFACE_TOKEN') or self.get('HF_TOKEN')),
            'kaggle_username': bool(self.get('KAGGLE_USERNAME')),
            'kaggle_key': bool(self.get('KAGGLE_KEY')),
        }
        return results
    
    def print_status(self) -> None:
        """Print environment configuration status."""
        print("\n🔧 Environment Configuration Status:")
        print("=" * 40)
        
        # Credentials
        validation = self.validate_credentials()
        print("\n🔑 Credentials:")
        for key, valid in validation.items():
            status = "✅" if valid else "❌"
            print(f"  {key}: {status}")
        
        # Paths
        print("\n📁 Paths:")
        paths = {
            'MODEL_DIR': self.get('MODEL_DIR', './models'),
            'DATA_DIR': self.get('DATA_DIR', './data'),
            'RESULTS_DIR': self.get('RESULTS_DIR', '../results'),
            'LOG_DIR': self.get('LOG_DIR', './logs'),
        }
        
        for key, path_str in paths.items():
            path = Path(path_str).expanduser().resolve()
            exists = "✅" if path.exists() else "❌"
            print(f"  {key}: {path} {exists}")
        
        # Settings
        print("\n⚙️ Settings:")
        settings = {
            'DEVICE': self.get('DEVICE', 'cpu'),
            'BATCH_SIZE': self.get('BATCH_SIZE', '16'),
            'LEARNING_RATE': self.get('LEARNING_RATE', '5e-5'),
            'LOG_LEVEL': self.get('LOG_LEVEL', 'INFO'),
        }
        
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("=" * 40)


# Global instance
env_config = EnvironmentConfig()

# Convenience functions
def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value."""
    return env_config.get(key, default)

def get_required_env(key: str) -> str:
    """Get required environment variable."""
    return env_config.get_required(key)

def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return env_config.get_bool(key, default)

def get_int_env(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    return env_config.get_int(key, default)

def get_float_env(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    return env_config.get_float(key, default)

def get_path_env(key: str, default: Optional[str] = None, create: bool = False) -> Optional[Path]:
    """Get path environment variable."""
    return env_config.get_path(key, default, create)

def validate_env() -> Dict[str, bool]:
    """Validate environment configuration."""
    return env_config.validate_credentials()

def print_env_status() -> None:
    """Print environment status."""
    env_config.print_status()


if __name__ == "__main__":
    # Test the configuration
    print("Testing Environment Configuration...")
    env_config.print_status()