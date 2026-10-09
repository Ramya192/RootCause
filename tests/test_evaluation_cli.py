"""`python -m causal_engine.evaluation` (causal_engine/evaluation/__main__.py): argument handling and output.

The evaluation runs themselves are covered by the test_evaluation_* files. Here they are replaced by stubs
so each test checks only what the command line does with them: which mode runs, what is written where,
which flag combinations are refused and what the exit code is.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from causal_engine.evaluation import __main__ as cli


def _report_module(calls: list, name: str):
    def run(**kwargs):
        calls.append((name, kwargs))
        return {"which": name}

    return SimpleNamespace(
        run=run,
        render_markdown=lambda result: f"# {name} markdown",
        render_json=lambda result: f'{{"which": "{name}"}}',
    )


def _scenario(error=None):
    return SimpleNamespace(runs=[SimpleNamespace(error=error)])


@pytest.fixture
def stubs(monkeypatch):
    """Replace every evaluation module the CLI dispatches to; `calls` records how each was invoked."""
    calls: list = []
    monkeypatch.setattr(cli, "anomalies_eval", SimpleNamespace(
        run=lambda **kw: calls.append(("anomalies", kw)) or {},
        render_markdown=lambda r: "# anomalies markdown",
        render_json=lambda r: '{"which": "anomalies"}',
        run_real=lambda **kw: calls.append(("anomalies_real", kw)) or {},
        render_real_markdown=lambda r: "# anomalies_real markdown",
        render_real_json=lambda r: '{"which": "anomalies_real"}',
    ))
    monkeypatch.setattr(cli, "learners", _report_module(calls, "learners"))
    monkeypatch.setattr(cli, "sensitivity_eval", _report_module(calls, "sensitivity"))
    results = {"value": [_scenario()]}
    monkeypatch.setattr(cli, "harness", SimpleNamespace(
        run_evaluation=lambda **kw: calls.append(("evaluation", kw)) or results["value"],
        run_stress=lambda *a, **kw: calls.append(("stress", (a, kw))) or results["value"],
    ))
    monkeypatch.setattr(cli, "report", SimpleNamespace(
        render_markdown=lambda r, s: "# results markdown", render_json=lambda r, s: '{"which": "results"}',
    ))
    monkeypatch.setattr(cli, "stress_report", SimpleNamespace(render_markdown=lambda r, s: "# stress markdown"))
    monkeypatch.setattr(cli, "baselines_report", SimpleNamespace(render_markdown=lambda r, s: "# baselines markdown"))
    return SimpleNamespace(calls=calls, results=results)


@pytest.mark.parametrize(
    "flag, label, stem",
    [
        ("--anomalies", "anomalies", "anomalies"),
        ("--anomalies-real", "anomalies_real", "anomalies_real"),
        ("--learners", "learners", "learners"),
        ("--sensitivity", "sensitivity", "sensitivity"),
    ],
)
def test_a_single_mode_run_prints_and_writes_its_report(stubs, tmp_path, capsys, flag, label, stem):
    assert cli.main([flag, "--out", str(tmp_path)]) == 0

    assert stubs.calls[0][0] == label
    assert f"# {label} markdown" in capsys.readouterr().out
    assert (tmp_path / f"{stem}.md").read_text(encoding="utf-8") == f"# {label} markdown"
    assert f'"which": "{label}"' in (tmp_path / f"{stem}.json").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "flag, expected", [("--anomalies", 10), ("--anomalies-real", 5), ("--learners", 10), ("--sensitivity", 10)]
)
def test_replicates_default_per_mode_and_can_be_overridden(stubs, flag, expected):
    cli.main([flag, "--no-write"])
    cli.main([flag, "--no-write", "--replicates", "3"])

    assert stubs.calls[0][1]["replicates"] == expected
    assert stubs.calls[1][1]["replicates"] == 3


def test_no_write_prints_without_creating_any_file(stubs, tmp_path, capsys):
    out = tmp_path / "reports"

    assert cli.main(["--learners", "--no-write", "--out", str(out)]) == 0

    assert not out.exists()
    assert "# learners markdown" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--anomalies-real", "--stress"],
        ["--anomalies", "--learners"],
        ["--learners", "--sensitivity"],
        ["--sensitivity", "--baselines"],
        ["--stress", "--baselines"],
        ["--workers", "0"],
        ["--scenario", "small_n"],  # only meaningful with --stress / --baselines
        ["--workers", "2"],  # likewise
        ["--stress", "--replicates", "0"],
    ],
)
def test_conflicting_or_meaningless_flag_combinations_are_refused(stubs, argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv + ["--no-write"])

    assert exit_info.value.code == 2
    assert stubs.calls == []  # refused before any evaluation ran


def test_default_run_scores_every_dataset_and_writes_results(stubs, tmp_path):
    assert cli.main(["--domain", "german_credit", "--dataset", "real", "--replicates", "2", "--out", str(tmp_path)]) == 0

    name, kwargs = stubs.calls[0]
    assert name == "evaluation"
    assert kwargs["domains"] == ["german_credit"] and kwargs["datasets"] == ["real"] and kwargs["replicates"] == 2
    assert (tmp_path / "results.md").exists() and (tmp_path / "results.json").exists()


def test_forcing_an_algorithm_writes_to_a_separate_file_so_defaults_are_not_overwritten(stubs, tmp_path):
    assert cli.main(["--algorithm", "ges", "--out", str(tmp_path)]) == 0

    assert (tmp_path / "results_ges.md").exists() and (tmp_path / "results_ges.json").exists()
    assert not (tmp_path / "results.md").exists()


def test_a_dataset_that_errored_makes_the_command_exit_nonzero(stubs):
    stubs.results["value"] = [_scenario(error="boom")]

    assert cli.main(["--no-write"]) == 1


@pytest.mark.parametrize("flag, stem", [("--stress", "stress"), ("--baselines", "baselines")])
def test_stress_and_baselines_runs_use_their_own_report_and_file_names(stubs, tmp_path, flag, stem):
    assert cli.main([flag, "--replicates", "2", "--scenario", "small_n", "--out", str(tmp_path)]) == 0

    name, (args, kwargs) = stubs.calls[0]
    assert name == "stress" and args == (2,)
    assert kwargs["scenario_ids"] == ["small_n"]
    assert kwargs["with_baselines"] == (flag == "--baselines")
    assert (tmp_path / f"{stem}.md").read_text(encoding="utf-8") == f"# {stem} markdown"
    assert (tmp_path / f"{stem}.json").exists()


def test_stress_replicates_default_to_the_documented_constant(stubs, tmp_path):
    cli.main(["--stress", "--out", str(tmp_path)])

    assert stubs.calls[0][1][0][0] == cli.DEFAULT_STRESS_REPLICATES
