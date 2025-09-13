#!/usr/bin/env python3
"""
Simple test script to verify the frontend interface works correctly.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_static_files():
    """Test that all static files exist and are accessible."""
    print("Testing static file structure...")
    
    static_dir = Path("src/static")
    required_files = [
        "index.html",
        "css/styles.css", 
        "js/app.js"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = static_dir / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
        else:
            print(f"✓ Found: {full_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All static files found")
    return True

def test_html_structure():
    """Test that the HTML file has the required structure."""
    print("\nTesting HTML structure...")
    
    html_file = Path("src/static/index.html")
    if not html_file.exists():
        print("✗ HTML file not found")
        return False
    
    content = html_file.read_text()
    
    required_elements = [
        'id="prediction-tab"',
        'id="batch-tab"', 
        'id="results-tab"',
        'id="docs-tab"',
        'id="predict-btn"',
        'id="sic"',
        'id="b4c"',
        'id="al2o3"',
        'class="tab-button"',
        'class="property-cards"'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
        else:
            print(f"✓ Found: {element}")
    
    if missing_elements:
        print(f"✗ Missing HTML elements: {missing_elements}")
        return False
    
    print("✓ HTML structure is correct")
    return True

def test_css_structure():
    """Test that the CSS file has the required styles."""
    print("\nTesting CSS structure...")
    
    css_file = Path("src/static/css/styles.css")
    if not css_file.exists():
        print("✗ CSS file not found")
        return False
    
    content = css_file.read_text()
    
    required_styles = [
        ':root {',
        '.container {',
        '.header {',
        '.nav-tabs {',
        '.tab-button {',
        '.form-section {',
        '.property-card {',
        '.btn-primary {',
        '@media (max-width: 768px)'
    ]
    
    missing_styles = []
    for style in required_styles:
        if style not in content:
            missing_styles.append(style)
        else:
            print(f"✓ Found: {style}")
    
    if missing_styles:
        print(f"✗ Missing CSS styles: {missing_styles}")
        return False
    
    print("✓ CSS structure is correct")
    return True

def test_js_structure():
    """Test that the JavaScript file has the required functionality."""
    print("\nTesting JavaScript structure...")
    
    js_file = Path("src/static/js/app.js")
    if not js_file.exists():
        print("✗ JavaScript file not found")
        return False
    
    content = js_file.read_text()
    
    required_functions = [
        'class CeramicArmorApp',
        'makePrediction()',
        'collectFormData()',
        'displayResults(',
        'updateCompositionTotal()',
        'setupEventListeners()',
        'callPredictionAPI(',
        'switchTab('
    ]
    
    missing_functions = []
    for func in required_functions:
        if func not in content:
            missing_functions.append(func)
        else:
            print(f"✓ Found: {func}")
    
    if missing_functions:
        print(f"✗ Missing JavaScript functions: {missing_functions}")
        return False
    
    print("✓ JavaScript structure is correct")
    return True

def test_server_startup():
    """Test that the server can start and serve the frontend."""
    print("\nTesting server startup...")
    
    try:
        # Start the server in the background
        print("Starting FastAPI server...")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000",
            "--log-level", "warning"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start (longer timeout for model loading)
        time.sleep(8)
        
        # Test if server is responding
        try:
            response = requests.get("http://127.0.0.1:8000/", timeout=5)
            if response.status_code == 200:
                print("✓ Server started successfully")
                print("✓ Frontend is accessible at http://127.0.0.1:8000/")
                
                # Test static file serving
                css_response = requests.get("http://127.0.0.1:8000/static/css/styles.css", timeout=5)
                js_response = requests.get("http://127.0.0.1:8000/static/js/app.js", timeout=5)
                
                if css_response.status_code == 200:
                    print("✓ CSS file served correctly")
                else:
                    print(f"✗ CSS file not served (status: {css_response.status_code})")
                
                if js_response.status_code == 200:
                    print("✓ JavaScript file served correctly")
                else:
                    print(f"✗ JavaScript file not served (status: {js_response.status_code})")
                
                return True
            else:
                print(f"✗ Server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Could not connect to server: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to start server: {e}")
        return False
        
    finally:
        # Clean up - terminate the server process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

def main():
    """Run all frontend tests."""
    print("=== Ceramic Armor ML Frontend Test ===\n")
    
    tests = [
        test_static_files,
        test_html_structure,
        test_css_structure,
        test_js_structure,
        test_server_startup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"✗ Test failed with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The frontend is ready to use.")
        print("\nTo start the server manually:")
        print("python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nThen visit: http://localhost:8000")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())