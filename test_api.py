#!/usr/bin/env python3
"""Test script for Amharic H-Net API."""

import requests
import json
import time
from typing import Dict, Any


class AmharicAPITester:
    """Test the Amharic H-Net API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def test_health(self) -> bool:
        """Test health endpoint."""
        print("🏥 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health check passed")
                print(f"   📊 Status: {data.get('status')}")
                print(f"   🤖 Model loaded: {data.get('model_loaded')}")
                print(f"   ⏰ Uptime: {data.get('uptime')}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_generate_single(self) -> bool:
        """Test single text generation."""
        print("📝 Testing single text generation...")
        
        test_cases = [
            {"prompt": "ኢትዮጵያ", "length": 50, "category": "general"},
            {"prompt": "አዲስ አበባ", "length": 40, "category": "news"},
            {"prompt": "ትምህርት", "length": 60, "category": "educational"},
            {"prompt": "", "length": 30, "category": "cultural"}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"   Test {i}: {test_case}")
            try:
                response = self.session.post(
                    f"{self.base_url}/generate",
                    json=test_case
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Generated: '{data['text'][:50]}...'")
                    print(f"   📊 Quality: {data['quality_score']:.3f}")
                    print(f"   ⏱️  Time: {data['generation_time']:.3f}s")
                else:
                    print(f"   ❌ Generation failed: {response.status_code}")
                    print(f"   📄 Response: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Generation error: {e}")
                return False
        
        return True
    
    def test_generate_batch(self) -> bool:
        """Test batch text generation."""
        print("📦 Testing batch text generation...")
        
        batch_request = {
            "requests": [
                {"prompt": "ሰላም", "length": 30, "category": "conversation"},
                {"prompt": "ባህል", "length": 40, "category": "cultural"},
                {"prompt": "ወጣቶች", "length": 35, "category": "educational"}
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate/batch",
                json=batch_request
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Batch generated: {data['batch_size']} texts")
                print(f"   ⏱️  Total time: {data['total_time']:.3f}s")
                print(f"   📊 Avg per request: {data['avg_time_per_request']:.3f}s")
                
                for i, result in enumerate(data['results'], 1):
                    print(f"   Text {i}: '{result['text'][:40]}...' (Q: {result['quality_score']:.3f})")
                
                return True
            else:
                print(f"   ❌ Batch generation failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Batch generation error: {e}")
            return False
    
    def test_evaluate(self) -> bool:
        """Test text evaluation endpoint."""
        print("🔍 Testing text evaluation...")
        
        test_texts = [
            "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ውብ ሀገር ናት። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።",
            "አዲስ አበባ ዋና ከተማ ናት።",
            "hello world this is not amharic text",
            "ሀገር ሀገር ሀገር ሀገር ሀገር"  # Repetitive text
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"   Test {i}: '{text[:50]}...'")
            try:
                response = self.session.post(
                    f"{self.base_url}/evaluate",
                    json={"text": text}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    scores = data['scores']
                    print(f"   ✅ Overall Quality: {scores['overall_quality']:.3f}")
                    print(f"   🔤 Amharic Ratio: {scores['amharic_ratio']:.3f}")
                    print(f"   💬 Fluency: {scores['fluency_score']:.3f}")
                    print(f"   🔗 Coherence: {scores['coherence_score']:.3f}")
                else:
                    print(f"   ❌ Evaluation failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Evaluation error: {e}")
                return False
        
        return True
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality."""
        print("🚦 Testing rate limiting...")
        
        # Make multiple rapid requests
        rapid_requests = 5
        success_count = 0
        
        for i in range(rapid_requests):
            try:
                response = self.session.post(
                    f"{self.base_url}/generate",
                    json={"prompt": "ሰላም", "length": 20}
                )
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    print(f"   ⚠️ Rate limited at request {i+1}")
                    break
                    
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"   ❌ Rate limit test error: {e}")
                return False
        
        print(f"   ✅ {success_count}/{rapid_requests} requests succeeded")
        return True
    
    def test_stats(self) -> bool:
        """Test statistics endpoint."""
        print("📊 Testing statistics endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Stats retrieved")
                print(f"   ⏰ Uptime: {data.get('uptime_hours', 0):.2f} hours")
                print(f"   📈 Total requests: {data.get('total_requests', 0)}")
                print(f"   🚀 Requests/hour: {data.get('requests_per_hour', 0):.1f}")
                return True
            else:
                print(f"   ❌ Stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Stats error: {e}")
            return False
    
    def test_homepage(self) -> bool:
        """Test homepage endpoint."""
        print("🏠 Testing homepage...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200 and "Amharic H-Net API" in response.text:
                print(f"   ✅ Homepage loaded successfully")
                return True
            else:
                print(f"   ❌ Homepage failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Homepage error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests."""
        print("🧪 Running Amharic H-Net API Tests")
        print("=" * 45)
        
        tests = {
            "health": self.test_health,
            "homepage": self.test_homepage,
            "generate_single": self.test_generate_single,
            "generate_batch": self.test_generate_batch,
            "evaluate": self.test_evaluate,
            "stats": self.test_stats,
            "rate_limiting": self.test_rate_limiting
        }
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests.items():
            print(f"\n🔸 {test_name.replace('_', ' ').title()}")
            try:
                results[test_name] = test_func()
                if results[test_name]:
                    passed += 1
                    print(f"   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED")
            except Exception as e:
                results[test_name] = False
                print(f"   ❌ ERROR: {e}")
        
        print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
        
        if passed == len(tests):
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the logs above.")
        
        return results


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Amharic H-Net API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["health", "generate", "evaluate", "batch", "all"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = AmharicAPITester(base_url=args.url)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "health":
        tester.test_health()
    elif args.test == "generate":
        tester.test_generate_single()
    elif args.test == "evaluate":
        tester.test_evaluate()
    elif args.test == "batch":
        tester.test_generate_batch()


if __name__ == "__main__":
    main()