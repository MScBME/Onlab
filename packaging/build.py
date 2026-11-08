"""Build the Windows installer of Swimmer Tracker.

    uv run --group build python packaging/build.py [--skip-installer] [--skip-smoke-test]

The steps are the following.
1. Stage the models of release_models.json, their .onnx and metadata files with the release display names.
2. PyInstaller onedir build into packaging/dist/SwimmerTracker/SwimmerTracker.exe.
3. Smoke test, which starts the frozen app offscreen on an empty temporary library and checks that the UI and every
   model load.
4. Inno Setup, if ISCC.exe is found, which makes packaging/dist/SwimmerTracker-<version>-setup.exe.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
ROOT = PACKAGING.parent
MODELS_DIR = ROOT / "models"
BUILD = PACKAGING / "build"
DIST = PACKAGING / "dist"
APP_DIR = DIST / "SwimmerTracker"
SMOKE_TIMEOUT_S = 120
ISCC_IN_INSTALL_DIR = Path("Inno Setup 6") / "ISCC.exe"
# Folders Inno Setup 6 installs into, per user or for all users, as environment variable and subfolder
INNO_INSTALL_ROOTS = (("LOCALAPPDATA", "Programs"), ("ProgramFiles(x86)", ""), ("ProgramFiles", ""))


def version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def stage_models() -> list:
    release = json.loads((PACKAGING / "release_models.json").read_text(encoding="utf-8"))["models"]
    out = BUILD / "models"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for entry in release:
        name = entry["name"]
        onnx = MODELS_DIR / f"{name}.onnx"
        if not onnx.exists():
            sys.exit(f"{onnx} is missing. train-model writes it at the end of the training of {name}.")
        metadata_path = onnx.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metadata.update(display_name=entry["display_name"], description=entry.get("description", ""))
        shutil.copy2(onnx, out / onnx.name)
        (out / metadata_path.name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"  model {name} as {entry['display_name']}")
    return [entry["name"] for entry in release]


def pyinstaller():
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller"),
                    str(PACKAGING / "swimmertracker.spec")], check=True)


def smoke_test(expected_models: list):
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "SWIMMERTRACKER_SMOKE_TEST": str(report),
               "SWIMMERTRACKER_LIBRARY": str(Path(tmp) / "library")}
        env.pop("SWIMMERTRACKER_MODELS", None)
        code = subprocess.run([str(APP_DIR / "SwimmerTracker.exe")], env=env, timeout=SMOKE_TIMEOUT_S).returncode
        if not report.exists():
            sys.exit(f"Smoke test failed, the app exited with code {code} and wrote no report")
        result = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))
    found = [m["name"] for m in result.get("models", [])]
    if code != 0 or not result.get("ok") or sorted(found) != sorted(expected_models):
        sys.exit(f"Smoke test failed with exit code {code}, the app found the models {found} "
                 f"instead of {expected_models}")
    print("Smoke test OK")


def find_iscc():
    """ISCC.exe from the ISCC environment variable, from PATH or from the default install folders."""
    candidates = [os.environ.get("ISCC"), shutil.which("iscc")]
    for variable, subfolder in INNO_INSTALL_ROOTS:
        root = os.environ.get(variable)
        if root:  # an unset variable must not turn into a path relative to the working directory
            candidates.append(Path(root) / subfolder / ISCC_IN_INSTALL_DIR)
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def installer(app_version: str):
    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup 6 was not found, so there is no installer. Install it with "
              "winget install JRSoftware.InnoSetup or set ISCC to the path of ISCC.exe, then run this again.")
        return
    subprocess.run([str(iscc), f"/DAppVersion={app_version}", str(PACKAGING / "installer.iss")], check=True)
    print(f"Installer written to {DIST / f'SwimmerTracker-{app_version}-setup.exe'}")


def main():
    parser = argparse.ArgumentParser(description="Build the Swimmer Tracker Windows installer.")
    parser.add_argument("--skip-installer", action="store_true",
                        help="Stop after the PyInstaller build and the smoke test")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Do not start the frozen app")
    args = parser.parse_args()

    app_version = version()
    print(f"Swimmer Tracker {app_version}")
    print("[1/4] Staging release models...")
    models = stage_models()
    print("[2/4] PyInstaller...")
    pyinstaller()
    if args.skip_smoke_test:
        print("[3/4] Smoke test skipped")
    else:
        print("[3/4] Smoke test...")
        smoke_test(models)
    if args.skip_installer:
        print("[4/4] Installer skipped")
    else:
        print("[4/4] Installer...")
        installer(app_version)


if __name__ == "__main__":
    main()
