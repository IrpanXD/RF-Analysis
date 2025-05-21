"""
Static site builder for the RF-Analysis Streamlit app.
This script creates a static version of the Streamlit app for GitHub Pages.
"""
import os
import subprocess
import shutil

def main():
    # Create build directory if it doesn't exist
    if not os.path.exists('build'):
        os.makedirs('build')
    
    # Install required packages if they don't exist
    try:
        import streamlit_static_io
    except ImportError:
        print("Installing streamlit-static-io...")
        subprocess.check_call(["pip", "install", "streamlit-static-io"])
    
    # Build the static site
    print("Building static site...")
    subprocess.check_call(["streamlit_static", "run", "appv4.py", "--output-dir=build"])
    
    # Copy index.html to the root of the build directory
    print("Copying assets...")
    if os.path.exists('assets'):
        if not os.path.exists('build/assets'):
            os.makedirs('build/assets')
        for item in os.listdir('assets'):
            s = os.path.join('assets', item)
            d = os.path.join('build/assets', item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    
    print("Static site built successfully in the 'build' directory!")

if __name__ == "__main__":
    main() 