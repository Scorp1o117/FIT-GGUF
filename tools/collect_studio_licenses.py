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
             "setuptools", "packaging", "PyInstaller"):
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
    if name == "proxy-tools" and not copied:
        # The 0.1.0 wheel omits its upstream MIT license file.
        source = Path(__file__).parent / "licenses" / "proxy-tools-LICENSE.txt"
        target = destination / name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        copied.append(str(target.relative_to(destination)))
    report.append({"name": name, "version": package.version, "licenses": copied})

# Conda Python uses separate runtime DLLs; preserve their installed-package
# notices alongside the Python and wheel licenses when building from Conda.
runtime_files = {p.name.lower() for p in (destination.parent / "_internal").glob("*.dll")}
for metadata in (Path(sys.base_prefix) / "conda-meta").glob("*.json"):
    record = json.loads(metadata.read_text(encoding="utf-8"))
    libraries = sorted({Path(name).name for name in record.get("files", [])
                        if Path(name).name.lower() in runtime_files})
    if not libraries or record["name"] == "python":
        continue
    source = Path(record.get("link", {}).get("source", "")) / "info" / "licenses"
    files = [p for p in source.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"Missing runtime notices for {record['name']}: {libraries}")
    copied = []
    for file in files:
        target = destination / ("runtime-" + record["name"]) / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        copied.append(str(target.relative_to(destination)))
    report.append({"name": record["name"], "version": record["version"],
                   "runtime_libraries": libraries, "licenses": copied})
(destination / "dependencies.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
