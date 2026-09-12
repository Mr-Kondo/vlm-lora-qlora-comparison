"""Structural checks on the Colab notebook.

The notebook must orchestrate the shared CLI and visualize its artifacts -- it
must not contain a second copy of the training implementation.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "vlm_lora_qlora_comparison.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_sources(nb):
    # Plain concatenation, exactly how Jupyter and Colab reconstruct a cell.
    # Joining with "\n" instead would paper over missing line terminators in the
    # source arrays and let a notebook that cannot run at all pass these tests.
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _all_source(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_is_valid_nbformat():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)


def test_source_lines_are_newline_terminated():
    """Every line but the last must end with "\\n".

    nbformat stores a cell as a list of lines that the client concatenates
    verbatim. Without the terminators every cell collapses onto one line and
    Colab raises SyntaxError on the first multi-line cell.
    """
    nb = _notebook()
    offenders = []
    for index, cell in enumerate(nb["cells"]):
        for line_no, line in enumerate(cell["source"][:-1]):
            if not line.endswith("\n"):
                offenders.append((index, line_no, line[:60]))
    assert not offenders, f"{len(offenders)} line(s) missing a terminator, e.g. {offenders[:3]}"


def test_multi_line_cells_survive_concatenation():
    """A cell with several statements must still read as several lines."""
    nb = _notebook()
    for index, source in enumerate(_code_sources(nb)):
        if len(source) > 120:
            assert "\n" in source, f"code cell {index} concatenated into a single line"


def test_every_code_cell_parses():
    for index, source in enumerate(_code_sources(_notebook())):
        neutral = "\n".join(
            "pass" if line.lstrip().startswith(("!", "%")) else line for line in source.split("\n")
        )
        try:
            ast.parse(neutral)
        except SyntaxError as exc:
            raise AssertionError(f"code cell {index} has a syntax error: {exc}\n{source[:300]}")


def test_notebook_drives_the_shared_cli():
    source = _all_source(_notebook())
    assert "scripts/train.py --method lora" in source
    assert "scripts/train.py --method qlora" in source
    for variant in ("base", "lora", "qlora"):
        assert f"scripts/evaluate.py --model-variant {variant}" in source
    assert "scripts/download_model.py" in source


def test_notebook_does_not_reimplement_training():
    """The training implementation lives in scripts/ and src/, nowhere else."""
    forbidden = [
        "get_peft_model(",
        "LoraConfig(",
        "TrainingArguments(",
        "Trainer(",
        "prepare_model_for_kbit_training(",
        "BitsAndBytesConfig(",
        "AutoModelForImageTextToText.from_pretrained",
        ".backward()",
        "optimizer.step()",
    ]
    for source in _code_sources(_notebook()):
        for token in forbidden:
            assert token not in source, (
                f"notebook re-implements training ({token!r}); it must call the shared CLI instead"
            )


def test_notebook_reads_artifacts_through_the_report_module():
    source = _all_source(_notebook())
    assert "from vlm_ft import report" in source
    assert "report.build_comparison" in source
    assert "report.load_loss_curves" in source


def test_notebook_covers_every_required_visualization():
    source = _all_source(_notebook())
    for token in ("json_validity_rate", "document_exact_match", "field_accuracy",
                  "field_precision", "field_recall", "field_f1", "cer", "wer", "eval_loss",
                  "peak_training_vram_gib", "training_duration_seconds",
                  "trainable_parameters", "trainable_percent"):
        assert token in source, f"no visualization or table reference for {token}"
    assert "Training loss" in source and "Validation loss" in source
    assert "Quality versus cost" in source or "quality against resource cost" in source


def test_notebook_does_not_fabricate_a_base_loss_curve():
    source = _all_source(_notebook())
    assert "BASE has none" in source or "BASE is absent by design" in source
    # load_loss_curves defaults to the two trained methods only
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from vlm_ft import report

    import inspect

    default = inspect.signature(report.load_loss_curves).parameters["variants"].default
    assert "base" not in tuple(default)


def test_notebook_answers_all_seven_questions():
    source = _all_source(_notebook())
    for number in range(1, 8):
        assert f"{number}." in source
    for phrase in ("How much does LoRA improve", "How much does QLoRA improve",
                   "meaningful quality difference between LoRA and QLoRA",
                   "How much GPU VRAM does QLoRA save", "differ in training time",
                   "similar number of trainable adapter parameters",
                   "justify"):
        assert phrase in source, f"final analysis does not address: {phrase}"
