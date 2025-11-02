#!/usr/bin/env python3
"""
Groundwater Chatbot Launcher
============================

This script launches the Groundwater Assistant chatbot.
Make sure you have all dependencies installed and data files in place.

Usage:
    python run_chatbot.py

Requirements:
    - All dependencies from requirements.txt
    - Data files in data/ directory
    - Trained model in models/ directory
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required files and directories exist"""
    required_files = [
        "data/rainfall.csv",
        "data/groundwater.csv", 
        "data/regions.geojson",
        "models/groundwater_predictor.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all data files and the trained model are available.")
        return False
    
    print("✅ All required files found!")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def launch_chatbot():
    """Launch the Streamlit chatbot"""
    print("🚀 Launching Groundwater Assistant...")
    try:
        # Change to the frontend directory to run the chatbot
        os.chdir("frontend")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "chatbot.py", "--server.port", "8502"])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")

def main():
    """Main function"""
    print("🤖 Groundwater Assistant Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("groundwater-backend"):
        print("❌ Please run this script from the project root directory")
        print("   Expected structure: groundwater-backend/")
        return
    
    # Change to the groundwater-backend directory
    os.chdir("groundwater-backend")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Ask user if they want to install dependencies
    install_deps = input("\n📦 Install/update dependencies? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        if not install_dependencies():
            return
    
    # Launch chatbot
    print("\n🌐 The chatbot will open in your default web browser")
    print("   URL: http://localhost:8502")
    print("   Press Ctrl+C to stop the chatbot")
    print("\n" + "=" * 40)
    
    launch_chatbot()

if __name__ == "__main__":
    main()
