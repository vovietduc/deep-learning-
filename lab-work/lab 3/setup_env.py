"""Create a project-local virtual environment using the invoking Python."""
from pathlib import Path
import os
import platform
import subprocess
import sys
import venv
ROOT=Path(__file__).resolve().parent

def run(*args):
    print("+", " ".join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)

def main():
    print("Python:",sys.version)
    if sys.version_info[:2] != (3,14):
        raise SystemExit("Please run with Python 3.14: py -3.14 setup_env.py")
    if platform.architecture()[0] != "64bit":
        raise SystemExit("64-bit Python is required.")
    venv.EnvBuilder(with_pip=True).create(ROOT/".venv")
    python=ROOT/".venv"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    run(python,"-m","pip","install","--upgrade","pip")
    run(python,"-m","pip","install","torch==2.15.0","--index-url","https://download.pytorch.org/whl/cpu")
    run(python,"-m","pip","install","--only-binary=:all:","-r", "requirements-windows.lock" if os.name == "nt" else "requirements.txt")
    run(python,"check_env.py")
    print("SETUP COMPLETE. Select .venv in VS Code and run Exercise 1/2.")

if __name__ == "__main__":
    try: main()
    except subprocess.CalledProcessError as exc:
        print("Installation failed. Check internet and the error above. Run setup again after fixing it.")
        raise SystemExit(exc.returncode)
