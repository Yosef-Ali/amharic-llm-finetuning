#!/usr/bin/env python3
"""
Quick Start Script for Amharic Enhanced LLM
Runs the complete pipeline from data collection to deployment

Usage:
    python quick_start.py --phase all
    python quick_start.py --phase data
    python quick_start.py --phase train
    python quick_start.py --phase deploy
    python quick_start.py --phase monitor
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmharicLLMQuickStart:
    """Quick start orchestrator for Amharic LLM pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root.parent / "amharic_env"
        self.python_path = self.venv_path / "bin" / "python"
        
        # Check if virtual environment exists
        if not self.python_path.exists():
            self.python_path = "python"  # Fallback to system python
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and return success status"""
        logger.info(f"🚀 {description}")
        logger.info(f"Running: {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            logger.info(f"✅ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'torch', 'transformers', 'requests', 'beautifulsoup4', 
            'kaggle', 'gradio', 'huggingface-hub'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            
            install_cmd = f"{self.python_path} -m pip install {' '.join(missing_packages)}"
            return self.run_command(install_cmd, "Installing dependencies")
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def run_data_collection(self) -> bool:
        """Run Phase 1: Data Collection and Preprocessing"""
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 1: DATA COLLECTION & PREPROCESSING")
        logger.info("="*60)
        
        # Step 1: Enhanced data collection
        if not self.run_command(
            f"{self.python_path} enhanced_data_collector.py",
            "Enhanced data collection"
        ):
            return False
        
        # Step 2: Data preprocessing
        if not self.run_command(
            f"{self.python_path} amharic_preprocessor.py",
            "Data preprocessing"
        ):
            return False
        
        logger.info("✅ Phase 1 completed successfully!")
        return True
    
    def run_evaluation(self) -> bool:
        """Run Phase 3: Evaluation and Benchmarking"""
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 3: EVALUATION & BENCHMARKING")
        logger.info("="*60)
        
        return self.run_command(
            f"{self.python_path} amharic_evaluation_suite.py",
            "Comprehensive model evaluation"
        )
    
    def run_deployment(self) -> bool:
        """Run Phase 4: Production Deployment"""
        logger.info("\n" + "="*60)
        logger.info("🚀 PHASE 4: PRODUCTION DEPLOYMENT")
        logger.info("="*60)
        
        # Check for HuggingFace token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logger.warning("⚠️  HF_TOKEN not found in environment variables")
            logger.info("Please set your HuggingFace token:")
            logger.info("export HF_TOKEN='your_token_here'")
            return False
        
        return self.run_command(
            f"{self.python_path} deploy_huggingface.py",
            "HuggingFace deployment"
        )
    
    def run_monitoring(self) -> bool:
        """Run Phase 5: Monitoring and Analytics"""
        logger.info("\n" + "="*60)
        logger.info("📈 PHASE 5: MONITORING & ANALYTICS")
        logger.info("="*60)
        
        logger.info("Starting monitoring dashboard...")
        logger.info("Press Ctrl+C to stop monitoring")
        
        return self.run_command(
            f"{self.python_path} monitoring_analytics.py",
            "Monitoring and analytics dashboard"
        )
    
    def run_training_setup(self) -> bool:
        """Setup for Phase 2: Model Training (Kaggle)"""
        logger.info("\n" + "="*60)
        logger.info("🧠 PHASE 2: MODEL TRAINING SETUP")
        logger.info("="*60)
        
        # Check if Kaggle is configured
        kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_config.exists():
            logger.warning("⚠️  Kaggle API not configured")
            logger.info("Please configure Kaggle API:")
            logger.info("1. Download kaggle.json from https://www.kaggle.com/account")
            logger.info("2. Place it in ~/.kaggle/kaggle.json")
            logger.info("3. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Create Kaggle dataset
        logger.info("📦 Creating Kaggle dataset...")
        
        # Check if preprocessing report exists
        if not Path("preprocessing_report.json").exists():
            logger.error("Preprocessing report not found. Run data collection first.")
            return False
        
        logger.info("✅ Training setup ready!")
        logger.info("📝 Next steps:")
        logger.info("   1. Upload kaggle_training_notebook.ipynb to Kaggle")
        logger.info("   2. Create a new Kaggle notebook")
        logger.info("   3. Upload the processed data as a dataset")
        logger.info("   4. Run the training notebook")
        
        return True
    
    def generate_status_report(self) -> dict:
        """Generate current project status report"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'project_status': 'active',
            'phases': {
                'phase_1_data': self.check_phase_1_status(),
                'phase_2_training': self.check_phase_2_status(),
                'phase_3_evaluation': self.check_phase_3_status(),
                'phase_4_deployment': self.check_phase_4_status(),
                'phase_5_monitoring': self.check_phase_5_status()
            },
            'next_steps': self.get_next_steps()
        }
        
        return status
    
    def check_phase_1_status(self) -> dict:
        """Check Phase 1 completion status"""
        data_dir = Path("data")
        preprocessing_report = Path("preprocessing_report.json")
        
        return {
            'status': 'completed' if data_dir.exists() and preprocessing_report.exists() else 'pending',
            'data_collected': data_dir.exists(),
            'preprocessing_done': preprocessing_report.exists()
        }
    
    def check_phase_2_status(self) -> dict:
        """Check Phase 2 completion status"""
        notebook_exists = Path("kaggle_training_notebook.ipynb").exists()
        
        return {
            'status': 'ready' if notebook_exists else 'pending',
            'notebook_ready': notebook_exists,
            'kaggle_configured': (Path.home() / ".kaggle" / "kaggle.json").exists()
        }
    
    def check_phase_3_status(self) -> dict:
        """Check Phase 3 completion status"""
        evaluation_suite = Path("amharic_evaluation_suite.py").exists()
        
        return {
            'status': 'ready' if evaluation_suite else 'pending',
            'evaluation_suite_ready': evaluation_suite
        }
    
    def check_phase_4_status(self) -> dict:
        """Check Phase 4 completion status"""
        deployment_script = Path("deploy_huggingface.py").exists()
        hf_token_set = bool(os.getenv('HF_TOKEN'))
        
        return {
            'status': 'ready' if deployment_script and hf_token_set else 'pending',
            'deployment_script_ready': deployment_script,
            'hf_token_configured': hf_token_set
        }
    
    def check_phase_5_status(self) -> dict:
        """Check Phase 5 completion status"""
        monitoring_script = Path("monitoring_analytics.py").exists()
        
        return {
            'status': 'ready' if monitoring_script else 'pending',
            'monitoring_script_ready': monitoring_script
        }
    
    def get_next_steps(self) -> list:
        """Get recommended next steps based on current status"""
        steps = []
        
        # Check each phase and recommend next steps
        if not self.check_phase_1_status()['data_collected']:
            steps.append("Run data collection: python quick_start.py --phase data")
        
        if not self.check_phase_2_status()['kaggle_configured']:
            steps.append("Configure Kaggle API for training")
        
        if not self.check_phase_4_status()['hf_token_configured']:
            steps.append("Set HuggingFace token: export HF_TOKEN='your_token'")
        
        if not steps:
            steps.append("All phases ready! Run: python quick_start.py --phase all")
        
        return steps
    
    def print_status_report(self):
        """Print formatted status report"""
        status = self.generate_status_report()
        
        print("\n" + "="*80)
        print("🇪🇹 AMHARIC ENHANCED LLM - PROJECT STATUS")
        print("="*80)
        
        for phase_name, phase_status in status['phases'].items():
            phase_num = phase_name.split('_')[1]
            phase_desc = phase_name.split('_', 2)[2] if len(phase_name.split('_')) > 2 else 'training'
            
            status_emoji = "✅" if phase_status['status'] == 'completed' else "🔄" if phase_status['status'] == 'ready' else "⏳"
            print(f"{status_emoji} Phase {phase_num}: {phase_desc.title()} - {phase_status['status'].upper()}")
        
        print("\n🎯 Next Steps:")
        for i, step in enumerate(status['next_steps'], 1):
            print(f"   {i}. {step}")
        
        print("\n📚 Available Commands:")
        print("   python quick_start.py --status          # Show this status")
        print("   python quick_start.py --phase data      # Run data collection")
        print("   python quick_start.py --phase train     # Setup training")
        print("   python quick_start.py --phase eval      # Run evaluation")
        print("   python quick_start.py --phase deploy    # Deploy to HuggingFace")
        print("   python quick_start.py --phase monitor   # Start monitoring")
        print("   python quick_start.py --phase all       # Run complete pipeline")
        print("="*80)
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🇪🇹 STARTING COMPLETE AMHARIC LLM PIPELINE")
        logger.info("="*80)
        
        # Check dependencies first
        if not self.check_dependencies():
            return False
        
        # Phase 1: Data Collection
        if not self.run_data_collection():
            logger.error("❌ Pipeline failed at Phase 1")
            return False
        
        # Phase 2: Training Setup
        if not self.run_training_setup():
            logger.error("❌ Pipeline failed at Phase 2 setup")
            return False
        
        # Phase 3: Evaluation
        if not self.run_evaluation():
            logger.error("❌ Pipeline failed at Phase 3")
            return False
        
        # Phase 4: Deployment (optional, requires HF token)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            if not self.run_deployment():
                logger.warning("⚠️  Deployment failed, but continuing...")
        else:
            logger.info("⏭️  Skipping deployment (HF_TOKEN not set)")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Print final status
        self.print_status_report()
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Amharic Enhanced LLM Quick Start')
    parser.add_argument('--phase', choices=['all', 'data', 'train', 'eval', 'deploy', 'monitor'], 
                       help='Phase to run')
    parser.add_argument('--status', action='store_true', help='Show project status')
    
    args = parser.parse_args()
    
    quick_start = AmharicLLMQuickStart()
    
    if args.status:
        quick_start.print_status_report()
        return
    
    if not args.phase:
        quick_start.print_status_report()
        return
    
    success = False
    
    if args.phase == 'all':
        success = quick_start.run_complete_pipeline()
    elif args.phase == 'data':
        success = quick_start.run_data_collection()
    elif args.phase == 'train':
        success = quick_start.run_training_setup()
    elif args.phase == 'eval':
        success = quick_start.run_evaluation()
    elif args.phase == 'deploy':
        success = quick_start.run_deployment()
    elif args.phase == 'monitor':
        success = quick_start.run_monitoring()
    
    if success:
        logger.info("\n✅ Operation completed successfully!")
    else:
        logger.error("\n❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()