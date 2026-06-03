import sys
import os
from pathlib import Path
import runpy

# This is a proxy script for Streamlit Cloud to find the main app
# since the rubric requires moving all code into a 'code/' folder.

code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))
os.chdir(code_dir)

runpy.run_path("app.py", run_name="__main__")
