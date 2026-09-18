"""
Static contract between Notebook 03 and ``src``.

The notebook is never executed in CI, so drift between a cell and a helper
signature is otherwise only discovered mid-training on RunPod. These tests parse
every code cell and check imports, keyword arguments and undefined names.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_asr_baseline_training.ipynb"
MAGIC_LINE = re.compile(r"^\s*[%!]")

# Names the notebook legitimately gets from the runtime rather than a definition.
RUNTIME_GLOBALS = {"__file__", "get_ipython", "display", "In", "Out"}


def _code_cells() -> List[Tuple[int, str]]:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    out = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            out.append((idx, "".join(cell.get("source", []))))
    return out


def _strip_magics(source: str) -> str:
    """IPython magics (``%pip``, ``!python``) are not valid Python syntax."""
    return "\n".join(
        "pass  # ipython magic" if MAGIC_LINE.match(line) else line
        for line in source.splitlines()
    )


def _parsed_cells() -> List[Tuple[int, str, ast.Module]]:
    out = []
    for idx, source in _code_cells():
        out.append((idx, source, ast.parse(_strip_magics(source))))
    return out


def _cell_by_marker(marker: str) -> str:
    for _, source in _code_cells():
        if source.lstrip().startswith(marker):
            return source
    raise AssertionError(f"no code cell starts with {marker!r}")


class TestNotebookCompiles:
    def test_every_code_cell_compiles(self):
        errors = []
        for idx, source in _code_cells():
            try:
                compile(_strip_magics(source), f"<cell {idx}>", "exec")
            except SyntaxError as exc:
                errors.append(f"cell {idx}: {exc}")
        assert not errors, "cells failed to compile: " + "; ".join(errors)


class TestNotebookImportsResolve:
    def test_all_src_imports_exist(self):
        """A notebook importing a helper that no longer exists must fail here."""
        missing = []
        for idx, _source, tree in _parsed_cells():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("src"):
                    continue
                module = importlib.import_module(node.module)
                for alias in node.names:
                    if not hasattr(module, alias.name):
                        missing.append(f"cell {idx}: {node.module}.{alias.name}")
        assert not missing, "notebook imports non-existent symbols: " + "; ".join(missing)

    def test_no_undefined_names(self):
        defined = set(dir(builtins)) | RUNTIME_GLOBALS
        used: Dict[str, int] = {}
        for idx, _source, tree in _parsed_cells():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, ast.arg):
                    defined.add(node.arg)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        defined.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined.add(alias.asname or alias.name)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    defined.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.setdefault(node.id, idx)
        undefined = sorted(name for name in used if name not in defined)
        assert not undefined, f"notebook uses undefined names: {undefined}"


def _missing_required_args(sig: inspect.Signature, node: ast.Call) -> List[str]:
    """
    Required parameters (positional or keyword-only) the call never supplies.

    A missing keyword-only argument is a TypeError at runtime but reads like a
    perfectly valid call, so signature drift here is invisible until the cell
    executes. Calls that splat ``*args``/``**kwargs`` are skipped.
    """
    if any(kw.arg is None for kw in node.keywords):
        return []
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return []
    supplied = {kw.arg for kw in node.keywords}
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for index, param in enumerate(positional):
        if index < len(node.args):
            supplied.add(param.name)
    return [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.name not in supplied
        and p.name != "self"
    ]


class TestNotebookKeywordArguments:
    """Every kwarg the notebook passes to a src helper must exist in its signature."""

    def _src_callables(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for _idx, _source, tree in _parsed_cells():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("src"):
                    continue
                module = importlib.import_module(node.module)
                for alias in node.names:
                    obj = getattr(module, alias.name, None)
                    if callable(obj):
                        out[alias.asname or alias.name] = obj
        return out

    def test_calls_match_signatures(self):
        callables = self._src_callables()
        problems = []
        for idx, _source, tree in _parsed_cells():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                target = callables.get(node.func.id)
                if target is None:
                    continue
                try:
                    sig = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                accepted = set(sig.parameters)
                takes_kwargs = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                for kw in node.keywords:
                    if kw.arg is None or takes_kwargs:
                        continue
                    if kw.arg not in accepted:
                        problems.append(f"cell {idx}: {node.func.id}(..., {kw.arg}=...)")
                positional = len([a for a in node.args if not isinstance(a, ast.Starred)])
                max_positional = sum(
                    1 for p in sig.parameters.values()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
                has_varargs = any(
                    p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
                )
                if positional > max_positional and not has_varargs:
                    problems.append(
                        f"cell {idx}: {node.func.id}() got {positional} positional args, "
                        f"max {max_positional}"
                    )
                missing = _missing_required_args(sig, node)
                if missing:
                    problems.append(
                        f"cell {idx}: {node.func.id}() missing required {missing}"
                    )
        assert not problems, "notebook calls disagree with src signatures: " + "; ".join(problems)


class TestPrepareCellWiring:
    def test_f1_uses_streaming_prepare_and_pinned_parquet_revision(self):
        f1 = _cell_by_marker("# Cell F1")
        assert "run_full_prepare_streaming" in f1
        assert "parquet_revision=EXPECTED_PARQUET_REVISION" in f1
        # One pass over both splits: the old per-split calls read every shard twice.
        assert "run_full_prepare_split" not in f1
        assert f1.count('splits={"train"') == 1
        assert "cleanup_shard=drop_shard_download" in f1
        assert "make_hf_parquet_stream_reader" in f1
        assert "cache_dir=HF_PARQUET_CACHE_DIR" in f1

    def test_config_pins_immutable_parquet_snapshot(self):
        cfg = _cell_by_marker("# Cell 3")
        assert "EXPECTED_PARQUET_REVISION" in cfg
        match = re.search(r'EXPECTED_PARQUET_REVISION\s*=\s*"([0-9a-f]{40})"', cfg)
        assert match, "EXPECTED_PARQUET_REVISION must be a 40-hex commit sha"
        assert "refs/convert/parquet" not in match.group(1)
        assert "HF_PARQUET_CACHE_DIR" in cfg
        assert "resolve_runtime_paths" in cfg
        assert "LOCAL_ROOT" in cfg and "DURABLE_ROOT" in cfg
        assert "MAX_AUDIO_DURATION = 40.0" in cfg


def _config_value(cell: str, name: str) -> str:
    match = re.search(rf"^{name}\s*=\s*(.+)$", cell, flags=re.M)
    assert match, f"{name} not found in config cell"
    return match.group(1)


class TestResumeTestCellWiring:
    def test_single_session_resume_test_is_rejected(self):
        f2 = _cell_by_marker("# Cell F2")
        assert 'if FULL_STAGE == "resume_test":' in f2
        assert "Single-session resume_test is not a valid proof" in f2

    def test_proof_is_captured_not_asserted(self):
        f2 = _cell_by_marker("# Cell F2")
        assert "make_resume_proof_callback" in f2
        assert "summarize_resume_proof" in f2
        assert "assert_cross_session_resume" in f2
        assert "derive_resume_test_status_from_proof" in f2
        # No literal restart claims anywhere in the cell.
        assert "true_restart\"] = True" not in f2
        assert "true_restart=True" not in f2
        assert "prove_trainer_resume_state" not in f2

    def test_phase_a_persists_session_token(self):
        f2 = _cell_by_marker("# Cell F2")
        assert '"session": session_token' in f2
        assert "new_session_token()" in f2


class TestTrainAndEvaluateCellWiring:
    def test_train_binds_contract_to_fingerprint(self):
        f3 = _cell_by_marker("# Cell F3")
        assert "expected_contract=training_contract" in f3
        assert "expected_contract=None" not in f3
        assert "parquet_revision=EXPECTED_PARQUET_REVISION" in f3
        # Train and validation share every physical shard, so they hydrate as one
        # union; two per-split calls would download all 50 shards twice.
        assert "hydrate_union_audio" in f3
        assert "hydrate_rows_audio" not in f3

    def test_evaluate_recomputes_hparams_instead_of_trusting_summary(self):
        f4 = _cell_by_marker("# Cell F4")
        assert "build_full_training_hparams" in f4
        assert "hp_expected" in f4 and "hp_recorded" in f4
        assert "derive_full_evaluate_status" in f4
        assert "resolve_best_checkpoint_from_durable" in f4
        # Status must come from the gate, never from a literal.
        assert '"status": "SUCCESS_FULL_EVALUATE"' not in f4
        assert 'full_status = "SUCCESS_FULL_EVALUATE"' not in f4


class TestExportCell:
    def test_export_uses_run_id_and_paths(self):
        export = None
        for _idx, source in _code_cells():
            if "export_pointer" in source and "RUNTIME_PATHS.export_root" in source:
                export = source
        assert export is not None, "export cell not found"
        assert "RUN_ID" in export
        assert 'PATHS["artifacts"]' in export and 'PATHS["results"]' in export
        assert "experiment_checkpoint_dir" in export or "full_state_dir" in export
        assert not re.search(r'run_id\s*=\s*"[0-9a-f]{8}-', export), "hardcoded run id in export"
        assert "/content/" not in export
        assert "DURABLE_ROOT" in export
