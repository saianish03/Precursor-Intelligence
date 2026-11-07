"""check_release (spec §2.0): the decision table, then the check against the fake CMS, then the CLI as the
release-check DAG runs it. Offline: the fake CMS serves the catalog and the metastore."""

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import requests
import yaml
from cms_fakes import API, FakeResponse, make_zip
from test_pipeline_offline import build_archive

from precursorintelligence.common.config import ReleaseCheckCfg
from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.gold.build import CURRENT_KEY
from precursorintelligence.harmonize import cli as pipeline_cli
from precursorintelligence.ingestion.context import build_context, validate_run_id
from precursorintelligence.ingestion.http import SafeHttpClient
from precursorintelligence.ingestion.pipeline import EXIT_ERROR, EXIT_OK, LOCK_NAME, run_step1
from precursorintelligence.ingestion.release_check import (
    PROCEED,
    Inputs,
    ReleaseCheckError,
    decide,
    run_release_check,
)

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
CFG = ReleaseCheckCfg()
DATASETS = [
    "4pq5-n9py",
    "r5ix-sfxw",
    "svdt-c123",
    "g6vv-u9sr",
    "djen-97ju",
    "ijh5-nb2v",
    "tagd-9999",
]
SHA = "a" * 40
ZIP = "theme:2026-08-06:nursing-homes_2026-08-06.zip@1000"
OTHER_ZIP = "theme:2026-08-26:nursing-homes_2026-08-26.zip@5"


def inputs(**kw):
    base = {
        "busy_run": None,
        "modified": dict.fromkeys(DATASETS, "2026-09"),
        "latest_ingested": "2026-09",
        "latest_published": "2026-09",
        "backlog": [],
        "pending_rows": 0,
        "failure": None,
        "last_run_start": NOW
        - timedelta(minutes=30),  # a pipeline run started after the rows below
    }
    return Inputs(**(base | kw))


def row(decision, target, hours_ago, sha=SHA, backlog=(), **kw):
    at = (NOW - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    return {
        "decision": decision,
        "target": target,
        "checked_at": at,
        "git_sha": sha,
        "proceed": decision in PROCEED,
        "backlog_json": json.dumps(list(backlog)),
    } | kw


def check(inp, history=(), sha=SHA):
    return decide(inp, list(history), now=NOW, git_sha=sha, cfg=CFG)


# ------------------------------------------------------------------------------------------- the decision table


def test_nothing_new_and_published_is_no_change():
    d = check(inputs())
    assert (d.decision, d.proceed, d.target) == ("NO_CHANGE", False, None)


def test_all_seven_at_one_newer_month_is_ready():
    d = check(inputs(modified=dict.fromkeys(DATASETS, "2026-10")))
    assert (d.decision, d.proceed, d.target) == ("READY", True, "2026-10")


def test_an_empty_database_is_ready():
    d = check(inputs(latest_ingested=None, latest_published=None))
    assert (d.decision, d.target) == ("READY", "2026-09")


@pytest.mark.parametrize("published", ["2026-08", None])
def test_an_ingested_vintage_not_published_is_resume(published):
    d = check(inputs(latest_ingested="2026-09", latest_published=published))
    assert (d.decision, d.proceed, d.target) == ("RESUME", True, "2026-09")


def test_a_failure_after_the_last_gold_build_is_resume_even_when_published():
    """Review S3: e.g. validate_gold failed after gold published, or a later run failed in silver."""
    d = check(inputs(failure="task silver failed in run r2"))
    assert (d.decision, d.proceed, d.target) == ("RESUME", True, "2026-09")
    assert "task silver failed" in d.reason


def test_unfinished_landing_work_is_changed():
    d = check(inputs(backlog=[ZIP]))
    assert (d.decision, d.proceed, d.target) == ("CHANGED", True, None)
    assert ZIP in d.reason
    d = check(inputs(pending_rows=3))  # inventoried, not resolved: a run failed in resolve
    assert (d.decision, d.reason) == (
        "CHANGED",
        "unfinished landing work: 3 inventory rows not resolved",
    )


def test_datasets_at_different_months_is_partial():
    d = check(inputs(modified=dict.fromkeys(DATASETS, "2026-09") | {"4pq5-n9py": "2026-10"}))
    assert (d.decision, d.proceed, d.target, d.alert) == ("PARTIAL", False, "2026-10", None)
    assert "2026-10: 4pq5-n9py" in d.reason


def test_a_held_lock_is_busy_whatever_else_holds():
    d = check(inputs(busy_run="manual__x", modified=dict.fromkeys(DATASETS, "2026-10")))
    assert (d.decision, d.proceed) == ("BUSY", False)


def test_precedence_ready_resume_changed_partial():
    new_month = dict.fromkeys(DATASETS, "2026-10")
    partial = dict.fromkeys(DATASETS, "2026-09") | {"4pq5-n9py": "2026-10"}
    unpublished = {"latest_published": "2026-08"}
    failed = {"failure": "task gold failed in run r1"}
    assert check(inputs(modified=new_month, backlog=[ZIP], **unpublished)).decision == "READY"
    assert check(inputs(backlog=[ZIP], **unpublished)).decision == "RESUME"
    assert check(inputs(backlog=[ZIP], **failed)).decision == "RESUME"
    assert check(inputs(modified=partial, backlog=[ZIP])).decision == "PARTIAL"  # see below


def test_landing_work_waits_while_a_release_is_partial():
    """Review S7: any run would download it, and a zip posted mid-release can hold a half-updated month. A RESUME
    with nothing to download still runs."""
    partial = dict.fromkeys(DATASETS, "2026-09") | {"4pq5-n9py": "2026-10"}
    for kw in (
        {"backlog": [ZIP]},
        {"pending_rows": 2},
        {"backlog": [ZIP], "latest_published": "2026-08"},
    ):
        d = check(inputs(modified=partial, **kw))
        assert (d.decision, d.proceed) == ("PARTIAL", False), kw
        assert "landing work waits until all 7 match" in d.reason
    assert check(inputs(modified=partial, latest_published="2026-08")).decision == "RESUME"
    assert (
        check(inputs(modified=dict.fromkeys(DATASETS, "2026-10"), backlog=[ZIP])).decision
        == "READY"
    )


# ---------------------------------------------------------------------------------- BLOCKED: no hourly repeats

PROCEEDS = {
    "READY": inputs(modified=dict.fromkeys(DATASETS, "2026-10")),
    "RESUME": inputs(latest_published="2026-08"),
    "CHANGED": inputs(backlog=[ZIP]),
}


def repeat(decision, hours_ago=1, **kw):
    """The history row of the same proceed, decided earlier on the same inputs."""
    inp = PROCEEDS[decision]
    return row(decision, check(inp).target, hours_ago, backlog=inp.backlog, **kw)


@pytest.mark.parametrize("decision", PROCEEDS)
def test_the_same_proceed_after_its_run_started_on_the_same_commit_is_blocked(decision):
    d = check(PROCEEDS[decision], [repeat(decision)])
    assert (d.decision, d.proceed, d.target) == ("BLOCKED", False, check(PROCEEDS[decision]).target)


@pytest.mark.parametrize("decision", PROCEEDS)
def test_a_new_commit_or_the_window_passing_lets_it_proceed(decision):
    assert check(PROCEEDS[decision], [repeat(decision, sha="b" * 40)]).proceed
    assert check(PROCEEDS[decision], [repeat(decision, hours_ago=25)]).proceed


@pytest.mark.parametrize("decision", PROCEEDS)
@pytest.mark.parametrize("last_run_start", [None, NOW - timedelta(hours=2)])
def test_a_proceed_whose_run_never_started_proceeds_again(decision, last_run_start):
    """Review H1: the pipeline was paused, the VM stopped, or the event was lost; no 24 h silence."""
    inp = replace(PROCEEDS[decision], last_run_start=last_run_start)
    assert check(inp, [repeat(decision)]).decision == decision


def test_new_cms_data_since_the_last_proceed_lets_it_proceed():
    """Review S1/S2: a new month, a new zip, or the same zip re-sized again is work the last run never saw."""
    new_month = dict.fromkeys(DATASETS, "2026-10")
    assert check(PROCEEDS["READY"], [row("READY", "2026-09", 1)]).decision == "READY"
    ready = row("READY", "2026-10", 1, backlog=[ZIP])
    assert check(inputs(modified=new_month, backlog=[ZIP, OTHER_ZIP]), [ready]).proceed
    resized = ZIP.replace("@1000", "@2000")
    assert check(inputs(backlog=[resized]), [row("CHANGED", None, 1, backlog=[ZIP])]).proceed
    # RESUME blocked (gold keeps failing), and CMS re-posts an old archive: run, the landing work is new
    assert check(
        inputs(latest_published="2026-08", backlog=[OTHER_ZIP]), [repeat("RESUME")]
    ).proceed


def test_a_resume_after_the_ready_that_ingested_it_is_blocked():
    """Review S6: READY 2026-10 ingested the month, then gold failed; RESUME 2026-10 would repeat that run."""
    history = [row("READY", "2026-10", 1, backlog=[ZIP])]
    inp = inputs(
        modified=dict.fromkeys(DATASETS, "2026-10"),
        latest_ingested="2026-10",
        latest_published="2026-09",
    )
    assert check(inp, history).decision == "BLOCKED"


def test_landing_work_the_blocked_run_already_saw_stays_blocked():
    """A quarantined zip or a download that keeps failing: still in the backlog, nothing new."""
    history = [row("READY", "2026-10", 1, backlog=[ZIP])]
    inp = inputs(modified=dict.fromkeys(DATASETS, "2026-10"), backlog=[ZIP])
    assert check(inp, history).decision == "BLOCKED"


def test_a_check_with_nothing_to_do_in_between_ends_the_repeat():
    """The situation did not hold: the last run finished (NO_CHANGE), then a new failure appeared."""
    history = [repeat("RESUME", hours_ago=3), row("NO_CHANGE", None, 2)]
    assert check(inputs(failure="task bronze failed in run r3"), history).decision == "RESUME"


def test_only_the_latest_proceed_counts_so_a_rollback_proceeds():
    """Commit A, then B, then back to A: proceeds."""
    history = [repeat("RESUME", hours_ago=3, sha=SHA), repeat("RESUME", hours_ago=2, sha="b" * 40)]
    assert check(PROCEEDS["RESUME"], history, sha=SHA).decision == "RESUME"


def test_blocked_and_busy_checks_do_not_extend_the_window():
    history = [repeat("RESUME", hours_ago=25), row("BLOCKED", "2026-09", 2), row("BUSY", None, 1)]
    assert check(PROCEEDS["RESUME"], history).decision == "RESUME"


def test_an_unknown_commit_counts_as_one_commit():
    """A laptop build without PI_GIT_SHA still does not repeat every hour."""
    d = check(PROCEEDS["RESUME"], [repeat("RESUME", sha="")], sha="")
    assert d.decision == "BLOCKED"


# ------------------------------------------------------------- proceeds that start no run: one alert per streak

NO_RUN = replace(PROCEEDS["READY"], last_run_start=NOW - timedelta(hours=10))


def test_the_third_proceed_in_a_row_without_a_run_alerts_once():
    """Review F4: cms_nh_pipeline paused or failing to import; BUSY and BLOCKED checks do not break the streak."""
    history = [
        repeat("READY", 3),
        row("BUSY", None, 2.5),
        repeat("READY", 2),
        row("BLOCKED", "2026-10", 1),
    ]
    d = check(NO_RUN, history)
    assert (d.decision, d.proceed) == ("READY", True)
    assert "3 times in a row" in d.alert and "paused" in d.alert
    assert d.alert_subject == "[cms_nh_release_check] the pipeline did not start (READY 2026-10)"
    assert check(NO_RUN, [repeat("READY", 2)]).alert is None  # the second: not yet
    assert check(NO_RUN, [repeat("READY", 3), repeat("READY", 2, alerted=True)]).alert is None


def test_a_run_that_started_or_a_check_with_nothing_to_do_ends_the_streak():
    started = replace(NO_RUN, last_run_start=NOW - timedelta(hours=2, minutes=30))
    assert check(started, [repeat("READY", 3), repeat("READY", 2)]).alert is None
    assert (
        check(NO_RUN, [repeat("READY", 3), row("NO_CHANGE", None, 2.5), repeat("READY", 2)]).alert
        is None
    )


# ------------------------------------------------------------------------------- PARTIAL: one alert per streak

PARTIAL = inputs(modified=dict.fromkeys(DATASETS, "2026-09") | {"4pq5-n9py": "2026-10"})


def test_a_partial_streak_alerts_once_it_is_old_enough_and_busy_does_not_break_it():
    history = [row("PARTIAL", "2026-10", hours_ago=80), row("BUSY", None, hours_ago=2)]
    d = check(PARTIAL, history)
    assert d.decision == "PARTIAL" and d.alert and "sent once" in d.alert
    assert d.alert_subject == "[cms_nh_release_check] CMS release still partial (PARTIAL 2026-10)"


def test_a_young_partial_streak_does_not_alert():
    assert check(PARTIAL, [row("PARTIAL", "2026-10", hours_ago=30)]).alert is None


def test_a_partial_streak_alerts_only_once():
    history = [row("PARTIAL", "2026-10", hours_ago=80), row("PARTIAL", "2026-10", 1, alerted=True)]
    assert check(PARTIAL, history).alert is None


def test_another_decision_ends_the_streak():
    history = [row("PARTIAL", "2026-10", hours_ago=200), row("NO_CHANGE", None, hours_ago=10)]
    assert check(PARTIAL, history).alert is None


# --------------------------------------------------------------------------------- against the fake CMS


@pytest.fixture
def ingested(fake_cms, make_ctx, tmp_path, monkeypatch):
    """The fake archive fully ingested (vintages 2021-01 and 2021-03); CMS reports 2021-03 everywhere."""
    monkeypatch.setenv("PI_GIT_SHA", SHA)
    build_archive(fake_cms)
    run_step1(make_ctx(fake_cms, root=tmp_path))
    fake_cms.modified = dict.fromkeys(DATASETS, "2021-03-01")

    def ctx(run_id="check-1"):
        return make_ctx(fake_cms, root=tmp_path, run_id=run_id)

    return ctx


def publish(ctx, v, state="current", logged=True):
    """gold/_current.json as gold writes it, with the gold run's `ok` row in run_log once it is current."""
    pointer = {"state": state, "run_id": "gold-run"} | ({"v": v} if state == "current" else {})
    ctx.object_store.put_bytes(CURRENT_KEY, json.dumps(pointer).encode(), overwrite=True)
    if state == "current" and logged:
        task_ran(ctx, "gold", "ok", datetime.now(UTC), run_id="gold-run")


def task_ran(ctx, task, status, at, run_id="pipeline-1"):
    """A run_log row as task_scope writes it, at a chosen time."""
    stamp = at.isoformat(timespec="seconds")
    ctx.metadata.append(
        "run_log",
        [
            {
                "run_id": run_id,
                "task": task,
                "entry_id": None,
                "status": status,
                "started_at": stamp,
                "finished_at": stamp,
                "duration_s": "0.00",
                "detail": "{}",
            }
        ],
    )


def test_nothing_new_and_gold_published_is_no_change(ingested):
    ctx = ingested()
    publish(ctx, "2021-03")
    out = run_release_check(ctx)
    assert (out["decision"], out["latest_ingested"], out["latest_published"]) == (
        "NO_CHANGE",
        "2021-03",
        "2021-03",
    )


def test_a_new_month_everywhere_is_ready_and_the_decision_is_recorded(ingested, fake_cms):
    ctx = ingested()
    publish(ctx, "2021-03")
    fake_cms.modified = dict.fromkeys(DATASETS, "2021-04-01")
    out = run_release_check(ctx)
    assert (out["decision"], out["proceed"], out["target"]) == ("READY", True, "2021-04")
    (rec,) = ctx.metadata.read("vintage_registry")
    assert (rec["decision"], rec["target"], rec["proceed"], rec["git_sha"]) == (
        "READY",
        "2021-04",
        True,
        SHA,
    )
    assert json.loads(rec["modified_json"]) == dict.fromkeys(DATASETS, "2021-04")


def test_resume_is_blocked_once_its_run_started_and_a_new_commit_resumes(ingested, monkeypatch):
    publish(ingested(), "2021-01")  # a run failed after ingesting 2021-03
    t = datetime.now(UTC)
    assert run_release_check(ingested("c1"), now=t)["decision"] == "RESUME"
    # its event started nothing (cms_nh_pipeline paused): the next check proceeds again
    assert run_release_check(ingested("c2"), now=t + timedelta(hours=1))["decision"] == "RESUME"
    task_ran(ingested(), "gold", "failed", t + timedelta(hours=1, minutes=5))
    assert run_release_check(ingested("c3"), now=t + timedelta(hours=2))["decision"] == "BLOCKED"
    monkeypatch.setenv("PI_GIT_SHA", "c" * 40)  # a fix deployed
    assert run_release_check(ingested("c4"), now=t + timedelta(hours=3))["decision"] == "RESUME"


def test_a_failure_after_the_last_good_gold_build_is_resume(ingested):
    ctx = ingested()
    publish(ctx, "2021-03")
    t = datetime.now(UTC)
    task_ran(
        ctx, "silver", "failed", t - timedelta(minutes=5)
    )  # before the good build: fixed by it
    task_ran(ctx, "gold", "ok", t)
    assert run_release_check(ctx, now=t + timedelta(minutes=1))["decision"] == "NO_CHANGE"
    task_ran(ctx, "bronze", "failed", t + timedelta(minutes=10))
    out = run_release_check(ingested("c2"), now=t + timedelta(hours=1))
    assert (out["decision"], out["target"]) == ("RESUME", "2021-03")
    assert (
        "run pipeline-1 did not reach a good gold build: its last recorded task is bronze (failed"
        in (out["reason"])
    )
    task_ran(ctx, "bronze", "ok", t + timedelta(hours=1, minutes=10), run_id="pipeline-2")
    task_ran(ctx, "gold", "ok", t + timedelta(hours=1, minutes=20), run_id="pipeline-2")
    assert run_release_check(ingested("c3"), now=t + timedelta(hours=2))["decision"] == "NO_CHANGE"


def test_a_run_killed_before_gold_is_resume(ingested):
    """Review F1: a task killed (exit 137, a timeout) writes no row; the run's earlier rows still show it."""
    ctx = ingested()
    publish(ctx, "2021-03")
    t = datetime.now(UTC)
    task_ran(ctx, "discover", "ok", t + timedelta(minutes=1), run_id="changed-run")
    task_ran(
        ctx, "bronze", "ok", t + timedelta(minutes=2), run_id="changed-run"
    )  # then silver was killed
    out = run_release_check(ctx, now=t + timedelta(hours=1))
    assert out["decision"] == "RESUME" and "its last recorded task is bronze (ok" in out["reason"]


def test_a_pointer_without_a_good_gold_build_in_run_log_resumes(ingested):
    """No gold `ok` row at all (e.g. run_log lost): every recorded run counts as unfinished, until one finishes."""
    ctx = ingested()
    publish(ctx, "2021-03", logged=False)
    assert run_release_check(ctx)["decision"] == "RESUME"


def test_validation_errors_after_the_last_good_gold_build_are_resume(ingested):
    """validate writes dq_results, not run_log: e.g. validate_gold failed after gold published."""
    ctx = ingested()
    publish(ctx, "2021-03")
    t = datetime.now(UTC)
    task_ran(ctx, "gold", "ok", t)

    def dq(severity, at):
        row = {
            "run_id": "pipeline-1",
            "layer": "gold",
            "table_name": "labels",
            "severity": severity,
        }
        ctx.metadata.append("dq_results", [row | {"checked_at": at.isoformat(timespec="seconds")}])

    dq("error", t - timedelta(minutes=5))  # before the good build
    dq("warning", t + timedelta(minutes=1))  # a warning does not fail the run
    assert run_release_check(ctx, now=t + timedelta(minutes=2))["decision"] == "NO_CHANGE"
    dq("error", t + timedelta(minutes=3))
    out = run_release_check(ingested("c2"), now=t + timedelta(hours=1))
    assert out["decision"] == "RESUME"
    assert "validation found 1 error(s), the last in run pipeline-1 (gold labels)" in out["reason"]


def test_the_latest_ingested_vintage_is_provider_informations(
    ingested, fake_cms, make_ctx, tmp_path
):
    """Review S4, resolve's rule: a newer month seen only in another table is not the latest vintage."""
    fake_cms.add("2021-05-06", make_zip("2021-04", "Apr2021", tables=["penalties"]))
    run_step1(make_ctx(fake_cms, root=tmp_path))
    assert run_release_check(ingested())["latest_ingested"] == "2021-03"


def test_a_gold_run_that_did_not_finish_promoting_counts_as_unpublished(ingested):
    ctx = ingested()
    publish(ctx, None, state="promoting")
    out = run_release_check(ctx)
    assert (out["decision"], out["latest_published"]) == ("RESUME", None)


def test_a_republished_zip_is_changed(ingested, fake_cms):
    ctx = ingested()
    publish(ctx, "2021-03")
    rel = next(e["url"] for e in fake_cms.entries if e["date"] == "2021-03-29")
    fake_cms.replace(rel, fake_cms.files[rel] + b"\0")  # CMS rewrote it: the catalog size changes
    out = run_release_check(ctx)
    assert out["decision"] == "CHANGED"
    size = len(fake_cms.files[rel])
    assert out["backlog"] == [f"theme:2021-03-29:nursing-homes_2021-03-29.zip@{size}"]


def test_a_zip_that_keeps_failing_is_blocked_once_its_run_started(ingested, fake_cms):
    """E.g. a quarantined re-post: still in the backlog (download_log has no good copy), nothing new at CMS."""
    ctx = ingested()
    publish(ctx, "2021-03")
    rel = next(e["url"] for e in fake_cms.entries if e["date"] == "2021-03-29")
    fake_cms.replace(rel, fake_cms.files[rel] + b"\0")
    t = datetime.now(UTC)
    assert run_release_check(ingested("c1"), now=t)["decision"] == "CHANGED"
    task_ran(ctx, "download", "failed", t + timedelta(minutes=5))
    out = run_release_check(ingested("c2"), now=t + timedelta(hours=1))
    assert (out["decision"], out["backlog_count"]) == ("BLOCKED", 1)


def test_a_new_zip_in_the_catalog_is_changed_but_an_out_of_scope_one_is_not(ingested, fake_cms):
    ctx = ingested()
    publish(ctx, "2021-03")
    fake_cms.add("2020-06-30", make_zip("2020-06", "Jun2020"))  # before scope.min_zip_date
    fake_cms.add("2021-12-31", make_zip("2021-03", "Mar2021"), kind="annual_theme")  # not in scope
    assert run_release_check(ctx)["decision"] == "NO_CHANGE"
    body = make_zip("2021-03", "Mar2021", tables=["provider_info"])
    fake_cms.add("2021-05-06", body)
    out = run_release_check(ingested("check-2"))
    assert (out["decision"], out["backlog"]) == (
        "CHANGED",
        [f"theme:2021-05-06:nursing-homes_2021-05-06.zip@{len(body)}"],
    )


def test_a_new_current_zip_is_changed(ingested, fake_cms):
    ctx = ingested()
    publish(ctx, "2021-03")
    current = fake_cms.current[0]
    current["date"] = "2021-05-06"  # the "download all" zip of a newer release
    fake_cms.replace(current["url"], make_zip("2021-03", "Mar2021", tables=["provider_info"]))
    out = run_release_check(ctx)
    assert out["decision"] == "CHANGED"
    assert [b.split("@")[0] for b in out["backlog"]] == [
        "current:2021-05-06:theme_nursing-homes_current.zip"
    ]


def test_a_current_zip_posted_mid_release_waits_for_all_seven(ingested, fake_cms):
    ctx = ingested()
    publish(ctx, "2021-03")
    fake_cms.modified["4pq5-n9py"] = "2021-04-01"  # Provider Information moved first
    current = fake_cms.current[0]
    current["date"] = "2021-04-20"
    fake_cms.replace(current["url"], make_zip("2021-04", "Apr2021", tables=["provider_info"]))
    out = run_release_check(ctx)
    assert (out["decision"], out["proceed"], out["backlog_count"]) == ("PARTIAL", False, 1)
    fake_cms.modified = dict.fromkeys(DATASETS, "2021-04-01")  # the release is complete
    assert run_release_check(ingested("c2"))["decision"] == "READY"


def test_the_printed_backlog_is_capped_but_the_row_keeps_all(ingested, fake_cms):
    """The JSON is one output line, read back from a stream shared with stderr: keep it a few KB."""
    ctx = ingested()
    publish(ctx, "2021-03")
    for day in range(1, 26):
        fake_cms.add(f"2022-01-{day:02d}", b"zip")
    out = run_release_check(ctx)
    assert (out["decision"], len(out["backlog"]), out["backlog_count"]) == ("CHANGED", 20, 25)
    assert len(json.dumps(out)) < 4096  # one short write: no stderr frame can land inside it
    (rec,) = ctx.metadata.read("vintage_registry")
    assert len(json.loads(rec["backlog_json"])) == 25


def test_a_run_holding_the_lock_is_busy_but_a_stale_lock_is_not(ingested):
    ctx = ingested()
    publish(ctx, "2021-03")
    lock = RunLock(ctx.metadata, LOCK_NAME)
    lock.acquire("manual__running")
    assert run_release_check(ctx)["decision"] == "BUSY"
    later = datetime.now(UTC) + timedelta(hours=13)  # past run_lock.stale_after_hours (12)
    assert run_release_check(ingested("check-2"), now=later)["decision"] == "NO_CHANGE"


def test_a_dataset_the_metastore_does_not_answer_for_is_an_error(ingested, fake_cms):
    ctx = ingested()
    url = f"{API}/metastore/schemas/dataset/items/djen-97ju?show-reference-ids=false"
    fake_cms.overrides[url] = FakeResponse(404, b"gone", {"Content-Type": "text/html"})
    with pytest.raises(ReleaseCheckError, match="djen-97ju"):
        run_release_check(ctx)
    assert ctx.metadata.read("vintage_registry") == []  # nothing recorded for a failed check


@pytest.mark.parametrize("modified", ["soon", "2026-13-01", "2026-00-01"])
def test_a_malformed_modified_date_is_an_error(ingested, fake_cms, modified):
    fake_cms.modified["tagd-9999"] = modified
    with pytest.raises(ReleaseCheckError, match="tagd-9999"):
        run_release_check(ingested())


# ------------------------------------------------------------------------------------- the CLI (the DAG task)


@pytest.fixture
def cli(fake_cms, tmp_path, monkeypatch):
    """Runs `precursor-pipeline` with an env file on the same store, its HTTP served by the fake CMS."""
    env = tmp_path / "env.yaml"
    env.write_text(
        yaml.safe_dump(
            {
                "env": "test-cli",
                "object_store": {"kind": "local", "root": str(tmp_path / "store")},
                "metadata_store": {"url": f"sqlite:///{tmp_path / 'meta.db'}"},
                "logging": {"log_dir": str(tmp_path / "logs")},
            }
        )
    )

    def build(settings, run_id=None, http=None):
        s = settings.sources
        return build_context(
            settings,
            run_id=run_id,
            http=SafeHttpClient(s.source, s.http, s.security, session=fake_cms),
        )

    monkeypatch.setattr(pipeline_cli, "build_context", build)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args, "--env-file", str(env)])
        with pytest.raises(SystemExit) as exc:
            pipeline_cli.main()
        return exc.value.code

    return run


def test_the_cli_prints_the_decision_as_its_last_json_line(cli, ingested, capsys):
    publish(ingested(), "2021-03")
    capsys.readouterr()
    assert cli("check-release", "--run-id", "scheduled__2026-11-07T12:00:00+00:00") == EXIT_OK
    lines = capsys.readouterr().out.strip().splitlines()
    out = json.loads(lines[-1])
    assert (out["decision"], out["proceed"]) == ("NO_CHANGE", False)
    assert out["check_run_id"] == "scheduled__2026-11-07T12:00:00+00:00"
    assert lines[0].startswith("check_release: NO_CHANGE")
    assert len(ingested().metadata.read("vintage_registry")) == 1


def test_a_check_by_hand_is_not_recorded(cli, ingested, fake_cms, capsys):
    """Review S5: only the DAG task (--run-id) records, so a manual READY never blocks the scheduled one."""
    fake_cms.modified = dict.fromkeys(DATASETS, "2021-04-01")
    capsys.readouterr()
    assert cli("check-release") == EXIT_OK
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["decision"] == "READY"
    assert ingested().metadata.read("vintage_registry") == []


def test_the_cli_exits_1_when_cms_cannot_be_read(cli, ingested, fake_cms, monkeypatch, capsys):
    def down(*a, **k):
        raise requests.ConnectionError("CMS is down")

    monkeypatch.setattr(fake_cms, "request", down)
    assert cli("check-release", "--run-id", "r1") == EXIT_ERROR
    assert "CMS is down" in capsys.readouterr().err


def test_the_run_ids_of_the_runs_it_starts_are_valid_pipeline_run_ids():
    """Every cms_nh_pipeline task gets --run-id {{ run_id }}. Airflow 3.1.8 names an asset-triggered run
    `asset_triggered__<run_after>_<8 random characters>` (DagRun.generate_run_id, read in the image): 58."""
    run_id = "asset_triggered__2026-11-07T12:05:00.123456+00:00_AbCd1234"
    assert validate_run_id(run_id) == run_id
