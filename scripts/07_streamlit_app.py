"""
============================================================
Traffic Sign Recognition - Server Entry Point
============================================================
To run the web application, execute:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

if __name__ == "__main__":
    print("=" * 65)
    print("NeuroSign AI Web Application Launcher")
    print("=" * 65)
    print("Starting Flask web server...")
    import app
    app.app.run(host="127.0.0.1", port=5000, debug=True)
