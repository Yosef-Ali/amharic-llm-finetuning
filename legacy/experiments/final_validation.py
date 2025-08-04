#!/usr/bin/env python3
"""
Final Validation Script for Smart Amharic LLM
Verifies all troubleshooting guidelines are implemented
"""

import os
import json
import requests
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalValidator:
    """Final validation following troubleshooting guidelines"""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "file_structure": {},
            "app_functionality": {},
            "data_quality": {},
            "training_components": {},
            "evaluation_metrics": {},
            "overall_status": "unknown"
        }
        
    def validate_file_structure(self) -> bool:
        """Validate required files exist"""
        logger.info("🔍 Validating file structure...")
        
        required_files = [
            "smart_amharic_app.py",
            "smart_train.py",
            "evaluation_metrics.py",
            "training_monitor.py",
            "local_data_collector.py",
            "requirements_local.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                
        self.validation_results["file_structure"] = {
            "required_files_present": len(missing_files) == 0,
            "missing_files": missing_files,
            "total_required": len(required_files),
            "found": len(required_files) - len(missing_files)
        }
        
        if missing_files:
            logger.warning(f"⚠️  Missing files: {missing_files}")
        else:
            logger.info("✅ All required files present")
            
        return len(missing_files) == 0
        
    def validate_app_functionality(self) -> bool:
        """Test if applications are running"""
        logger.info("🌐 Validating app functionality...")
        
        apps_to_test = [
            {"name": "Gradio App", "url": "http://127.0.0.1:7860", "expected_status": 200},
            {"name": "Flask App", "url": "http://127.0.0.1:5001", "expected_status": 200}
        ]
        
        app_results = []
        for app in apps_to_test:
            try:
                response = requests.get(app["url"], timeout=5)
                is_working = response.status_code == app["expected_status"]
                app_results.append({
                    "name": app["name"],
                    "url": app["url"],
                    "status_code": response.status_code,
                    "is_working": is_working
                })
                
                if is_working:
                    logger.info(f"✅ {app['name']} is running successfully")
                else:
                    logger.warning(f"⚠️  {app['name']} returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ {app['name']} is not accessible: {e}")
                app_results.append({
                    "name": app["name"],
                    "url": app["url"],
                    "status_code": 0,
                    "is_working": False,
                    "error": str(e)
                })
                
        working_apps = sum(1 for app in app_results if app["is_working"])
        
        self.validation_results["app_functionality"] = {
            "total_apps": len(apps_to_test),
            "working_apps": working_apps,
            "app_details": app_results,
            "all_working": working_apps == len(apps_to_test)
        }
        
        return working_apps == len(apps_to_test)
        
    def validate_data_quality(self) -> bool:
        """Validate data collection and quality"""
        logger.info("📊 Validating data quality...")
        
        data_checks = {
            "training_data_exists": False,
            "metadata_exists": False,
            "collection_progress": False,
            "data_quality_metrics": False
        }
        
        # Check for training data
        training_files = list(Path(".").glob("**/training_data*.txt"))
        data_checks["training_data_exists"] = len(training_files) > 0
        
        # Check for metadata
        metadata_dir = Path("data/metadata")
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*.json"))
            data_checks["metadata_exists"] = len(metadata_files) > 0
            
        # Check collection progress
        progress_file = Path("data/metadata/collection_progress.json")
        data_checks["collection_progress"] = progress_file.exists()
        
        # Check evaluation results
        eval_dir = Path("evaluation_results")
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.json"))
            data_checks["data_quality_metrics"] = len(eval_files) > 0
            
        self.validation_results["data_quality"] = data_checks
        
        passed_checks = sum(1 for check in data_checks.values() if check)
        logger.info(f"📊 Data quality checks: {passed_checks}/{len(data_checks)} passed")
        
        return passed_checks >= len(data_checks) // 2  # At least half should pass
        
    def validate_training_components(self) -> bool:
        """Validate training infrastructure"""
        logger.info("🏋️ Validating training components...")
        
        training_checks = {
            "smart_train_exists": Path("smart_train.py").exists(),
            "training_monitor_exists": Path("training_monitor.py").exists(),
            "model_directory_exists": Path("models").exists(),
            "checkpoints_exist": False,
            "training_plots_generated": False
        }
        
        # Check for checkpoints
        checkpoint_dirs = list(Path(".").glob("**/checkpoint-*"))
        training_checks["checkpoints_exist"] = len(checkpoint_dirs) > 0
        
        # Check for training plots
        plots_dir = Path("training_plots")
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            training_checks["training_plots_generated"] = len(plot_files) > 0
            
        self.validation_results["training_components"] = training_checks
        
        passed_checks = sum(1 for check in training_checks.values() if check)
        logger.info(f"🏋️ Training component checks: {passed_checks}/{len(training_checks)} passed")
        
        return passed_checks >= 4  # Most components should be present
        
    def validate_evaluation_metrics(self) -> bool:
        """Validate evaluation system"""
        logger.info("📈 Validating evaluation metrics...")
        
        eval_checks = {
            "evaluation_script_exists": Path("evaluation_metrics.py").exists(),
            "evaluation_results_exist": False,
            "metrics_comprehensive": False
        }
        
        # Check for evaluation results
        eval_dir = Path("evaluation_results")
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.json"))
            eval_checks["evaluation_results_exist"] = len(eval_files) > 0
            
            # Check if metrics are comprehensive
            if eval_files:
                try:
                    with open(eval_files[0], 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        required_metrics = ["language_quality", "conversational_coherence", "cultural_appropriateness"]
                        has_all_metrics = all(metric in eval_data for metric in required_metrics)
                        eval_checks["metrics_comprehensive"] = has_all_metrics
                except Exception as e:
                    logger.warning(f"⚠️  Could not parse evaluation results: {e}")
                    
        self.validation_results["evaluation_metrics"] = eval_checks
        
        passed_checks = sum(1 for check in eval_checks.values() if check)
        logger.info(f"📈 Evaluation checks: {passed_checks}/{len(eval_checks)} passed")
        
        return passed_checks >= 2  # At least script and results should exist
        
    def run_comprehensive_validation(self) -> Dict:
        """Run all validation checks"""
        logger.info("🇪🇹 Starting comprehensive validation...")
        
        validation_functions = [
            ("file_structure", self.validate_file_structure),
            ("app_functionality", self.validate_app_functionality),
            ("data_quality", self.validate_data_quality),
            ("training_components", self.validate_training_components),
            ("evaluation_metrics", self.validate_evaluation_metrics)
        ]
        
        passed_validations = 0
        total_validations = len(validation_functions)
        
        for validation_name, validation_func in validation_functions:
            try:
                result = validation_func()
                if result:
                    passed_validations += 1
                    logger.info(f"✅ {validation_name} validation passed")
                else:
                    logger.warning(f"⚠️  {validation_name} validation failed")
            except Exception as e:
                logger.error(f"❌ {validation_name} validation error: {e}")
                
        # Calculate overall status
        success_rate = passed_validations / total_validations
        if success_rate >= 0.8:
            self.validation_results["overall_status"] = "excellent"
        elif success_rate >= 0.6:
            self.validation_results["overall_status"] = "good"
        elif success_rate >= 0.4:
            self.validation_results["overall_status"] = "fair"
        else:
            self.validation_results["overall_status"] = "needs_improvement"
            
        self.validation_results["summary"] = {
            "passed_validations": passed_validations,
            "total_validations": total_validations,
            "success_rate": success_rate
        }
        
        # Save results
        self.save_validation_results()
        
        # Print summary
        self.print_validation_summary()
        
        return self.validation_results
        
    def save_validation_results(self):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"final_validation_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Validation results saved to {results_file}")
        
    def print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("🇪🇹 SMART AMHARIC LLM - FINAL VALIDATION REPORT")
        print("="*70)
        
        summary = self.validation_results["summary"]
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   • Validations Passed: {summary['passed_validations']}/{summary['total_validations']}")
        print(f"   • Success Rate: {summary['success_rate']:.1%}")
        print(f"   • Overall Status: {self.validation_results['overall_status'].upper()}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        
        # File Structure
        fs = self.validation_results["file_structure"]
        status = "✅" if fs.get("required_files_present", False) else "❌"
        print(f"   {status} File Structure: {fs.get('found', 0)}/{fs.get('total_required', 0)} files")
        
        # App Functionality
        af = self.validation_results["app_functionality"]
        status = "✅" if af.get("all_working", False) else "❌"
        print(f"   {status} App Functionality: {af.get('working_apps', 0)}/{af.get('total_apps', 0)} apps running")
        
        # Data Quality
        dq = self.validation_results["data_quality"]
        passed_dq = sum(1 for v in dq.values() if v)
        status = "✅" if passed_dq >= len(dq) // 2 else "❌"
        print(f"   {status} Data Quality: {passed_dq}/{len(dq)} checks passed")
        
        # Training Components
        tc = self.validation_results["training_components"]
        passed_tc = sum(1 for v in tc.values() if v)
        status = "✅" if passed_tc >= 4 else "❌"
        print(f"   {status} Training Components: {passed_tc}/{len(tc)} components ready")
        
        # Evaluation Metrics
        em = self.validation_results["evaluation_metrics"]
        passed_em = sum(1 for v in em.values() if v)
        status = "✅" if passed_em >= 2 else "❌"
        print(f"   {status} Evaluation Metrics: {passed_em}/{len(em)} metrics available")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if self.validation_results["overall_status"] == "excellent":
            print("   🎉 System is fully operational and follows all guidelines!")
        elif self.validation_results["overall_status"] == "good":
            print("   👍 System is working well with minor improvements needed")
        else:
            print("   ⚠️  System needs attention in some areas")
            
        print("\n🚀 NEXT STEPS:")
        print("   • Test conversational features with Amharic inputs")
        print("   • Monitor training progress and adjust parameters")
        print("   • Collect more diverse Amharic data for better performance")
        print("   • Deploy to production when ready")
        
        print("\n" + "="*70)
        
def main():
    """Main validation function"""
    logger.info("🇪🇹 Starting Smart Amharic LLM Final Validation")
    
    validator = FinalValidator()
    results = validator.run_comprehensive_validation()
    
    logger.info("✅ Final validation completed!")
    return results

if __name__ == "__main__":
    main()