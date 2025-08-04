#!/usr/bin/env python3
"""
Amharic LLM Environment Setup Script
Automated setup for local development environment

Features:
- Virtual environment creation
- Dependency installation
- Kaggle API setup
- Directory structure creation
- Initial data collection
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class AmharicLLMSetup:
    """Setup manager for Amharic LLM development environment"""
    
    def __init__(self, project_dir=None):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.venv_dir = self.project_dir / "venv"
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        
    def run_command(self, command, check=True):
        """Run shell command safely"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=check, 
                capture_output=True, 
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {command}")
            print(f"Error: {e.stderr}")
            return None
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        print("\n📦 Setting up virtual environment...")
        
        if self.venv_dir.exists():
            print("✅ Virtual environment already exists")
            return True
        
        # Create virtual environment
        result = self.run_command(f"python -m venv {self.venv_dir}")
        if result is None:
            return False
        
        print("✅ Virtual environment created")
        return True
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("\n📚 Installing dependencies...")
        
        # Get pip path
        if os.name == 'nt':  # Windows
            pip_path = self.venv_dir / "Scripts" / "pip"
        else:  # Unix/Linux/macOS
            pip_path = self.venv_dir / "bin" / "pip"
        
        # Core ML packages
        packages = [
            "torch",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "tokenizers>=0.13.0",
            "accelerate>=0.20.0",
            "gradio>=3.35.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "beautifulsoup4>=4.9.0",
            "lxml>=4.6.0",
            "requests>=2.25.0",
            "tqdm>=4.60.0"
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            result = self.run_command(f"{pip_path} install {package} -q")
            if result is None:
                print(f"⚠️ Failed to install {package}")
            else:
                print(f"✅ {package} installed")
        
        print("✅ All dependencies installed")
        return True
    
    def create_directory_structure(self):
        """Create project directory structure"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "collected",
            self.models_dir / "checkpoints",
            self.models_dir / "final",
            self.project_dir / "notebooks",
            self.project_dir / "scripts",
            self.project_dir / "logs",
            self.project_dir / "outputs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory.relative_to(self.project_dir)}")
        
        # Create .gitkeep files
        for directory in directories:
            gitkeep = directory / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        
        return True
    
    def setup_kaggle_config(self):
        """Setup Kaggle API configuration"""
        print("\n🏆 Setting up Kaggle API...")
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_config = kaggle_dir / "kaggle.json"
        
        if kaggle_config.exists():
            print("✅ Kaggle API already configured")
            return True
        
        print("📋 Kaggle API setup required:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print(f"4. Place it in: {kaggle_config}")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        
        # Create directory
        kaggle_dir.mkdir(exist_ok=True)
        
        return False
    
    def create_sample_data(self):
        """Create sample Amharic data for testing"""
        print("\n📝 Creating sample data...")
        
        sample_texts = [
            "ሰላም ነው። እንዴት ነዎት? ዛሬ ቆንጆ ቀን ነው። የአየር ሁኔታው በጣም ጥሩ ነው።",
            "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት። ታሪካዊ እና ባህላዊ ሀብት የበዛባት ሀገር ናት። በዚህ ሀገር ውስጥ ብዙ ቋንቋዎች ይነገራሉ።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። በዚህ ከተማ ውስጥ ብዙ ሰዎች ይኖራሉ። የአፍሪካ ህብረት ዋና መሥሪያ ቤትም በዚህ ከተማ ይገኛል።",
            "ትምህርት በሰው ልጅ ህይወት ውስጥ በጣም አስፈላጊ ነው። ትምህርት ሰውን ያበቃዋል። ትምህርት የሰውን አስተሳሰብ ያሰፋዋል።",
            "ቴክኖሎጂ በዘመናችን ህይወት ውስጥ ትልቅ ሚና ይጫወታል። ኮምፒውተር እና ስማርት ፎን ህይወታችንን ቀይረውታል። ኢንተርኔት መረጃ ለማግኘት ይረዳናል።",
            "ባህል የሰው ማህበረሰብ መለያ ነው። እያንዳንዱ ሀገር የራሱ ባህል አለው። ባህል ከትውልድ ወደ ትውልድ ይተላለፋል።",
            "ስፖርት ለጤንነት በጣም ጠቃሚ ነው። እግር ኳስ በኢትዮጵያ ውስጥ ተወዳጅ ስፖርት ነው። ሩጫም በሀገራችን ውስጥ ታዋቂ ነው።",
            "ምግብ ለሰው ልጅ ህይወት አስፈላጊ ነው። እያንዳንዱ ሀገር የራሱ ባህላዊ ምግብ አለው። ኢንጀራ የኢትዮጵያ ባህላዊ ምግብ ነው።"
        ]
        
        # Save as JSON
        sample_data = {
            "metadata": {
                "total_texts": len(sample_texts),
                "creation_date": datetime.now().isoformat(),
                "description": "Sample Amharic texts for testing"
            },
            "texts": sample_texts
        }
        
        sample_file = self.data_dir / "collected" / "sample_amharic_data.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Sample data created: {sample_file.relative_to(self.project_dir)}")
        return True
    
    def create_quick_start_script(self):
        """Create quick start script"""
        print("\n🚀 Creating quick start script...")
        
        script_content = '''#!/bin/bash
# Amharic LLM Quick Start Script

echo "🇪🇹 Amharic LLM Development Environment"
echo "======================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup_environment.py first."
    exit 1
fi

# Check if Kaggle is configured
if [ -f "~/.kaggle/kaggle.json" ]; then
    echo "✅ Kaggle API configured"
else
    echo "⚠️ Kaggle API not configured. See setup instructions."
fi

echo ""
echo "📋 Available commands:"
echo "  python amharic_data_collector.py    # Collect Amharic data"
echo "  jupyter notebook                    # Open Jupyter for development"
echo "  python huggingface_spaces_app.py   # Test Gradio interface"
echo ""
echo "📁 Project structure:"
echo "  data/collected/     # Collected Amharic texts"
echo "  models/            # Trained models"
echo "  notebooks/         # Jupyter notebooks"
echo "  logs/              # Training logs"
echo ""
echo "🎯 Next steps:"
echo "1. Collect more Amharic data"
echo "2. Upload to Kaggle for training"
echo "3. Deploy to Hugging Face Spaces"
'''
        
        script_file = self.project_dir / "quick_start.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print(f"✅ Quick start script created: {script_file.name}")
        return True
    
    def run_setup(self):
        """Run complete setup process"""
        print("🇪🇹 Amharic LLM Development Environment Setup")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create directory structure
        if not self.create_directory_structure():
            return False
        
        # Setup Kaggle
        kaggle_ready = self.setup_kaggle_config()
        
        # Create sample data
        if not self.create_sample_data():
            return False
        
        # Create quick start script
        if not self.create_quick_start_script():
            return False
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment: source venv/bin/activate")
        if not kaggle_ready:
            print("2. Configure Kaggle API (see instructions above)")
        print(f"3. Run quick start: ./quick_start.sh")
        print("4. Start collecting data: python amharic_data_collector.py")
        print("5. Open Jupyter: jupyter notebook")
        
        return True

def main():
    """Main setup function"""
    setup = AmharicLLMSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Environment setup completed!")
        return 0
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())