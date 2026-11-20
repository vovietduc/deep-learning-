import importlib
import platform
import sys
print("Executable:",sys.executable)
print("Python:",platform.python_version(),platform.architecture()[0])
failed=[]
for name in ["torch","transformers","datasets","evaluate","accelerate","numpy"]:
    try:
        module=importlib.import_module(name)
        print(name,getattr(module,"__version__","OK"))
    except Exception as exc:
        failed.append(name);print(name,"ERROR",repr(exc))
if failed:
    raise SystemExit("Missing/broken packages: "+", ".join(failed))
from transformers import Trainer,TrainingArguments,AutoTokenizer,AutoModelForSequenceClassification
print("Trainer imports: OK")
