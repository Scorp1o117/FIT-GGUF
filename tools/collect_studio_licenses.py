"""Copy installed dependency license files alongside a distributable build."""
from importlib.metadata import distribution, PackageNotFoundError
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
report = []
python_license = next((Path(sys.base_prefix) / name for name in ("LICENSE_PYTHON.txt", "LICENSE.txt", "LICENSE")
                       if (Path(sys.base_prefix) / name).is_file()), None)
if python_license is None:
    raise RuntimeError("Python runtime license was not found; do not distribute this build")
python_target = destination / "python" / python_license.name
python_target.parent.mkdir(parents=True, exist_ok=True)
python_target.write_bytes(python_license.read_bytes())
report.append({"name": "Python", "version": sys.version.split()[0],
               "licenses": [str(python_target.relative_to(destination))]})
for name in ("fit-gguf", "llmfit", "pywebview", "pythonnet", "clr-loader", "psutil",
             "PyYAML", "bottle", "proxy-tools", "cffi", "pycparser", "typing-extensions",
             "setuptools", "packaging"):
    try:
        package = distribution(name)
    except PackageNotFoundError:
        continue
    copied = []
    for index, file in enumerate(package.files or []):
        if "license" in file.name.lower() or file.name.lower().startswith(("copying", "notice")):
            source = Path(package.locate_file(file))
            if source.is_file():
                target = destination / name / f"{index}-{source.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                copied.append(str(target.relative_to(destination)))
    report.append({"name": name, "version": package.version, "licenses": copied})
(destination / "dependencies.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
