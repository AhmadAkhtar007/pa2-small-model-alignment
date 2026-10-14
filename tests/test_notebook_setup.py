import json
from pathlib import Path
import zipfile

import pytest


@pytest.mark.parametrize("compressed", [False, True])
def test_kaggle_bootstrap_accepts_archive_and_unpacked_dataset(tmp_path, compressed):
    inputs = tmp_path / "input"
    working = tmp_path / "working"
    inputs.mkdir()
    working.mkdir()
    files = {"submission_adapter.py": "", "alignment/core.py": "", "configs/sft.json": "{}"}
    if compressed:
        with zipfile.ZipFile(inputs / "small-model-alignment.zip", "w") as archive:
            for name, content in files.items():
                archive.writestr("small-model-alignment/" + name, content)
    else:
        for name, content in files.items():
            path = inputs / "project-dataset/small-model-alignment" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/01_training.ipynb").read_text())
    setup = "".join(notebook["cells"][1]["source"]).split("os.chdir(REPO)")[0]
    setup = setup.replace("/kaggle/input", inputs.as_posix()).replace("/kaggle/working", working.as_posix())
    exec(compile(setup, "bootstrap", "exec"), {})
    assert (working / "small-model-alignment/alignment/core.py").exists()
