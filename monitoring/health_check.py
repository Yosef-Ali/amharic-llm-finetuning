#!/usr/bin/env python3
"""
Health Check Script for Amharic Language AI
"""

import requests
import json
import sys
import time
from datetime import datetime

def check_health(base_url="http://localhost:8000"):
    """Check system health."""
    try:
        # Basic health check
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ System Status: {health_data.get('status', 'unknown')}")
            print(f"   Service: {health_data.get('service', 'N/A')}")
            print(f"   Version: {health_data.get('version', 'N/A')}")
            
            components = health_data.get('components', {})
            print("   Components:")
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {component}")
            
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_api_endpoints(base_url="http://localhost:8000"):
    """Check API endpoints functionality."""
    print("\n🔍 Testing API Endpoints:")
    
    # Test generation endpoint
    try:
        test_data = {"prompt": "በአዲስ አበባ ስብሰባ ተካሂዷል", "domain": "news"}
        response = requests.post(f"{base_url}/generate", json=test_data, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ Generation endpoint working")
        else:
            print(f"   ❌ Generation endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Generation endpoint error: {e}")
    
    # Test extraction endpoint
    try:
        test_data = {"text": "ጠቅላይ ሚኒስትር በአዲስ አበባ ስብሰባ አካሂደዋል", "domain": "news"}
        response = requests.post(f"{base_url}/extract", json=test_data, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ Extraction endpoint working")
        else:
            print(f"   ❌ Extraction endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Extraction endpoint error: {e}")

def main():
    """Main health check."""
    print(f"🏥 HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    if check_health(base_url):
        check_api_endpoints(base_url)
        print("\n✅ System appears healthy")
        sys.exit(0)
    else:
        print("\n❌ System health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
