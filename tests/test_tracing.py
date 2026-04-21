"""Unit tests for observability.tracing (no OTel deps required)."""

from __future__ import annotations

import threading

from src.observability.tracing import get_trace_id, new_trace_id, span


def test_new_trace_id_returns_hex():
    tid = new_trace_id()
    assert len(tid) == 16
    int(tid, 16)  # should not raise


def test_get_trace_id_stable():
    new_trace_id()
    a = get_trace_id()
    b = get_trace_id()
    assert a == b


def test_new_trace_id_changes():
    t1 = new_trace_id()
    t2 = new_trace_id()
    assert t1 != t2


def test_span_noop_when_disabled():
    with span("test_span", {"key": "value"}) as s:
        assert s is None


def test_trace_ids_differ_across_threads() -> None:
    """ContextVar storage is per-thread: concurrent pipelines must not share an id."""
    out: list[str] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        new_trace_id()
        out.append(get_trace_id())

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(set(out)) == 2
