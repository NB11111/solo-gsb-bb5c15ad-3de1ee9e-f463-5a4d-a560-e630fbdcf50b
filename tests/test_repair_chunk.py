"""PATCH /sessions/{id}/chunks/repair：已收分片原位修复。"""
from __future__ import annotations

import hashlib
import threading
import time

from fastapi.testclient import TestClient

from app import settings, storage


def _create(client: TestClient, data: bytes, name: str = "data.bin", chunk_size: int = 0):
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "chunk_size": chunk_size,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _repair(
    client: TestClient,
    sid: str,
    start: int,
    end: int,
    content: bytes,
    headers: dict | None = None,
):
    return client.request(
        "PATCH",
        f"/api/v1/sessions/{sid}/chunks/repair?start={start}&end={end}",
        content=content,
        headers=headers,
    )


def _upload_with_bad_chunk(
    client: TestClient, sid: str, data: bytes, start: int, end: int, bad: bytes
) -> None:
    """按 [0,start)+[start,end)(坏内容)+[end,n) 三片上传。"""
    assert len(bad) == end - start
    bounds = [0] + [b for b in (start, end) if 0 < b < len(data)] + [len(data)]
    for lo, hi in zip(bounds, bounds[1:]):
        body = bad if (lo, hi) == (start, end) else data[lo:hi]
        r = client.put(
            f"/api/v1/sessions/{sid}/chunks?start={lo}&end={hi}", content=body
        )
        assert r.status_code == 200, (lo, hi, r.text)


def _no_leftover_tempfiles() -> None:
    tmp = storage.tmp_dir()
    assert list(tmp.glob("recv-*.part")) == []
    assert list(tmp.glob("repair-*.old")) == []


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_repair_chunk_then_full_sha_completes(client: TestClient) -> None:
    data = bytes((i * 17 + 5) % 256 for i in range(8000))
    start, end = 2000, 3500
    bad = bytes((b ^ 0xFF) for b in data[start:end])
    sid = _create(client, data)
    _upload_with_bad_chunk(client, sid, data, start, end, bad)

    # 坏内容下整文件校验必然失败。
    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "sha256 mismatch"

    before = client.get(f"/api/v1/sessions/{sid}").json()
    assert before["bytes_received"] == len(data)
    assert before["missing_ranges"] == []

    # 原位修复：区间与已登记分片完全一致、内容等长。
    r = _repair(client, sid, start, end, data[start:end])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["received"] == {"start": start, "end": end}

    # 修复不改变缺失区间与已收字节数（等长替换）。
    after = client.get(f"/api/v1/sessions/{sid}").json()
    assert after["bytes_received"] == before["bytes_received"] == len(data)
    assert after["missing_ranges"] == before["missing_ranges"] == []
    assert after["fully_received"] is True
    assert body["bytes_received"] == len(data)
    assert body["missing_ranges"] == []
    assert body["fully_received"] is True
    assert "expires_at" in body

    # 磁盘上只有新内容，且随后可重新完成整文件校验并下载。
    assert storage.part_path(sid, start, end).read_bytes() == data[start:end]
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == data
    _no_leftover_tempfiles()


def test_repair_with_chunk_sha256_header_accepts_digest_forms(
    client: TestClient,
) -> None:
    data = bytes(range(256)) * 20
    start, end = 1000, 2000
    bad = b"z" * (end - start)
    good = data[start:end]
    for header_value in (
        hashlib.sha256(good).hexdigest(),
        hashlib.sha256(good).hexdigest().upper(),
        "sha256:" + hashlib.sha256(good).hexdigest(),
    ):
        sid = _create(client, data)
        _upload_with_bad_chunk(client, sid, data, start, end, bad)
        r = _repair(
            client,
            sid,
            start,
            end,
            good,
            headers={"X-Chunk-SHA256": header_value},
        )
        assert r.status_code == 200, (header_value, r.text)
        assert storage.part_path(sid, start, end).read_bytes() == good


def test_repair_keeps_missing_ranges_when_other_chunks_absent(
    client: TestClient,
) -> None:
    # 会话中只有一个已登记分片（其余缺失）：修复后缺失区间与字节数照旧。
    data = b"abcdefghij" * 1000
    sid = _create(client, data)
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=1000&end=2000",
        content=b"x" * 1000,
    )
    assert r.status_code == 200

    r = _repair(client, sid, 1000, 2000, data[1000:2000])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bytes_received"] == 1000
    assert body["missing_ranges"] == [
        {"start": 0, "end": 1000},
        {"start": 2000, "end": len(data)},
    ]
    assert body["fully_received"] is False
    _no_leftover_tempfiles()


def test_repair_is_idempotent(client: TestClient) -> None:
    data = bytes((i * 29 + 1) % 256 for i in range(2000))
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end={len(data)}",
        content=b"q" * len(data),
    )
    # 用相同的正确内容连续修复两次：都成功，状态不变。
    for _ in range(2):
        r = _repair(client, sid, 0, len(data), data)
        assert r.status_code == 200, r.text
        assert r.json()["bytes_received"] == len(data)
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text


# ---------------------------------------------------------------------------
# 失败必须保留原文件 / 登记 / 已收字节数
# ---------------------------------------------------------------------------


def test_repair_digest_mismatch_preserves_old_chunk(client: TestClient) -> None:
    data = bytes((i * 13 + 9) % 256 for i in range(5000))
    start, end = 1000, 2500
    bad = bytes((b ^ 0x0F) for b in data[start:end])
    sid = _create(client, data)
    _upload_with_bad_chunk(client, sid, data, start, end, bad)
    good_digest = hashlib.sha256(data[start:end]).hexdigest()

    # 内容与摘要不符：422，旧分片原样保留。
    r = _repair(
        client,
        sid,
        start,
        end,
        bad,  # 等长但与 good_digest 不符
        headers={"X-Chunk-SHA256": good_digest},
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "chunk sha256 mismatch"
    assert detail["expected"] == good_digest
    assert detail["actual"] == hashlib.sha256(bad).hexdigest()
    assert detail["start"] == start
    assert detail["end"] == end

    assert storage.part_path(sid, start, end).read_bytes() == bad
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["status"] == "uploading"
    assert status["bytes_received"] == len(data)
    assert status["missing_ranges"] == []
    _no_leftover_tempfiles()

    # 用正确内容 + 正确摘要重试：恢复并通过整文件校验。
    r = _repair(
        client,
        sid,
        start,
        end,
        data[start:end],
        headers={"X-Chunk-SHA256": good_digest},
    )
    assert r.status_code == 200, r.text
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


def test_repair_wrong_length_preserves_old_chunk(client: TestClient) -> None:
    data = b"abcdefghij" * 300
    start, end = 500, 1500
    bad = b"!" * (end - start)
    sid = _create(client, data)
    _upload_with_bad_chunk(client, sid, data, start, end, bad)

    # 声明长度（Content-Length 由客户端按 body 长度给出）与区间长度不符 → 400。
    r = _repair(client, sid, start, end, data[start : end - 10])
    assert r.status_code == 400, r.text
    assert "content-length" in r.json()["detail"]["error"]

    # 旧分片、登记、字节数全部不变。
    assert storage.part_path(sid, start, end).read_bytes() == bad
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["bytes_received"] == len(data)

    # 等长正确内容仍可修复。
    r = _repair(client, sid, start, end, data[start:end])
    assert r.status_code == 200, r.text
    _no_leftover_tempfiles()


def test_repair_malformed_sha_header_is_400_before_body(client: TestClient) -> None:
    data = b"abcdefghij" * 100
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=100", content=b"x" * 100
    )
    for bad_header in ("not-hex", "a" * 63, "g" * 64, "sha256:" + "b" * 63):
        r = _repair(
            client,
            sid,
            0,
            100,
            data[:100],
            headers={"X-Chunk-SHA256": bad_header},
        )
        assert r.status_code == 400, (bad_header, r.text)
        assert r.json()["detail"]["error"] == "invalid X-Chunk-SHA256 header"
        status = client.get(f"/api/v1/sessions/{sid}").json()
        assert status["bytes_received"] == 100
    # 原内容未被改动。
    assert storage.part_path(sid, 0, 100).read_bytes() == b"x" * 100


def test_repair_unknown_or_non_matching_range_is_404(client: TestClient) -> None:
    data = bytes(range(256)) * 20  # 5120
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=1000&end=2000",
        content=data[1000:2000],
    )

    # 完全未登记的区间。
    r = _repair(client, sid, 0, 1000, data[:1000])
    assert r.status_code == 404, r.text
    assert "no registered chunk" in r.json()["detail"]["error"]

    # 只与已登记分片部分重叠（不“完全一致”）同样 404。
    for lo, hi in ((1000, 1500), (1500, 2000), (900, 2000), (1000, 2100)):
        r = _repair(client, sid, lo, hi, data[lo:hi])
        assert r.status_code == 404, (lo, hi, r.text)

    # 相邻但未登记也 404。
    r = _repair(client, sid, 2000, 3000, data[2000:3000])
    assert r.status_code == 404

    # 已登记区间本身可修复（证明 404 仅针对不匹配区间）。
    r = _repair(client, sid, 1000, 2000, data[1000:2000])
    assert r.status_code == 200, r.text
    assert client.get(f"/api/v1/sessions/{sid}").json()["bytes_received"] == 1000
    _no_leftover_tempfiles()


def test_repair_validates_range_and_bounds(client: TestClient) -> None:
    data = b"y" * 1000
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=1000", content=data
    )

    # start >= end（半开区间非法）。
    r = _repair(client, sid, 500, 500, b"")
    assert r.status_code == 400

    # 区间越过 total_size（即便该长度的分片曾登记也不可能）。
    r = _repair(client, sid, 900, 1001, data[900:] + b"x")
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "chunk range exceeds total_size"

    assert storage.part_path(sid, 0, 1000).read_bytes() == data


def test_repair_respects_max_chunk_bytes(client: TestClient, monkeypatch) -> None:
    data = b"m" * 10_000
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=5000", content=b"a" * 5000
    )
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=5000&end=10000",
        content=b"b" * 5000,
    )

    monkeypatch.setattr(settings, "max_chunk_bytes", 4096)
    r = _repair(client, sid, 0, 5000, data[:5000])
    assert r.status_code == 413
    # 旧内容保留。
    assert storage.part_path(sid, 0, 5000).read_bytes() == b"a" * 5000

    monkeypatch.setattr(settings, "max_chunk_bytes", 0)
    r = _repair(client, sid, 0, 5000, data[:5000])
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 会话状态门槛
# ---------------------------------------------------------------------------


def test_repair_requires_uploading_session(client: TestClient) -> None:
    data = b"done" * 100
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end={len(data)}", content=data
    )
    client.post(f"/api/v1/sessions/{sid}/complete")

    # 已完成会话拒绝修复。
    r = _repair(client, sid, 0, len(data), data)
    assert r.status_code == 409
    assert client.get(f"/api/v1/sessions/{sid}").json()["status"] == "completed"

    # 未知会话 / 已放弃（硬删除）会话 → 404。
    assert _repair(client, "deadbeef" * 4, 0, 1, b"x").status_code == 404
    other = _create(client, b"hi" * 10)
    client.put(
        f"/api/v1/sessions/{other}/chunks?start=0&end=20", content=b"hi" * 10
    )
    assert client.delete(f"/api/v1/sessions/{other}").status_code == 204
    assert _repair(client, other, 0, 20, b"ih" * 10).status_code == 404


def test_repair_expired_session_is_410_and_preserves_error_semantics(
    client: TestClient, monkeypatch
) -> None:
    data = b"e" * 1000
    monkeypatch.setattr(settings, "ttl_seconds", 1)
    sid = _create(client, data)
    client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=1000", content=b"o" * 1000
    )
    time.sleep(1.1)

    r = _repair(client, sid, 0, 1000, data)
    assert r.status_code == 410
    status = client.get(f"/api/v1/sessions/{sid}")
    assert status.status_code == 410


# ---------------------------------------------------------------------------
# 落盘/登记失败：回滚恢复旧分片，仍可重试
# ---------------------------------------------------------------------------


def test_repair_database_failure_after_swap_restores_old_chunk(
    client: TestClient, monkeypatch
) -> None:
    """新分片已换入、chunks 刷新 SQL 失败：事务回滚，旧分片必须恢复原样，
    临时文件与备份不残留，之后重新修复仍能成功。"""
    import pytest
    from contextlib import contextmanager
    from app import db

    data = b"f" * 3000
    start, end = 500, 1800
    bad = b"#" * (end - start)
    sid = _create(client, data)
    _upload_with_bad_chunk(client, sid, data, start, end, bad)

    class InjectedDBError(RuntimeError):
        pass

    class FailingConn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *params):
            if sql.startswith("UPDATE chunks SET received_at"):
                raise InjectedDBError("injected update failure")
            return self._conn.execute(sql, *params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextmanager
    def flaky_immediate():
        conn = db.get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield FailingConn(conn)
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    monkeypatch.setattr("app.main.db.immediate", flaky_immediate)
    with pytest.raises(InjectedDBError):
        _repair(client, sid, start, end, data[start:end])
    monkeypatch.undo()

    # 旧分片完整恢复（字节与大小），登记/字节数不变，没有备份与临时文件残留。
    assert storage.part_path(sid, start, end).read_bytes() == bad
    assert storage.part_path(sid, start, end).stat().st_size == end - start
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["bytes_received"] == len(data)
    assert status["missing_ranges"] == []
    _no_leftover_tempfiles()
    assert len(list(storage.parts_dir(sid).iterdir())) == 3

    # 旧分片仍可用：重新修复成功并完成整文件校验。
    r = _repair(client, sid, start, end, data[start:end])
    assert r.status_code == 200, r.text
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


# ---------------------------------------------------------------------------
# 并发：与 complete / abort / 修复相互排队，绝不读到半替换内容
# ---------------------------------------------------------------------------


def test_concurrent_repair_and_complete_never_publishes_half_replaced(
    client: TestClient,
) -> None:
    data = bytes((i * 7 + 3) % 256 for i in range(6000))
    start, end = 2000, 4000
    bad = bytes((b + 1) % 256 for b in data[start:end])

    for round_no in range(6):
        sid = _create(client, data, name=f"race-{round_no}.bin")
        _upload_with_bad_chunk(client, sid, data, start, end, bad)

        outcome: dict[str, int] = {}
        barrier = threading.Barrier(2)

        def do_complete() -> None:
            barrier.wait()
            outcome["complete"] = client.post(
                f"/api/v1/sessions/{sid}/complete"
            ).status_code

        def do_repair() -> None:
            barrier.wait()
            outcome["repair"] = _repair(
                client, sid, start, end, data[start:end]
            ).status_code

        threads = [
            threading.Thread(target=do_complete, daemon=True),
            threading.Thread(target=do_repair, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads)

        # 修复要么成功（200），要么撞上完成后的状态（409）；
        # 坏内容的 complete 只能 422，绝不可能 200 发布坏成品。
        assert outcome["repair"] in (200, 409)
        assert outcome["complete"] in (200, 422)
        if outcome["complete"] == 200:
            # 仅当修复先落地，complete 才可能成功——成品必须是修复后的正确内容。
            assert outcome["repair"] == 200

        dl = client.get(f"/api/v1/sessions/{sid}/download")
        if dl.status_code == 200:
            assert dl.content == data  # 永远不可能下到半替换/坏内容
        else:
            assert dl.status_code == 404

        # 未完成时分片文件始终等长（旧或新），不存在截断/半写入文件。
        if dl.status_code == 404:
            part = storage.part_path(sid, start, end)
            assert part.is_file()
            assert part.stat().st_size == end - start
        _no_leftover_tempfiles()

        # 无论竞态结局如何，再修复一次后总能完成并得到正确成品。
        assert _repair(client, sid, start, end, data[start:end]).status_code in (
            200,
            409,
        )
        if client.get(f"/api/v1/sessions/{sid}").json()["status"] == "uploading":
            done = client.post(f"/api/v1/sessions/{sid}/complete")
            assert done.status_code == 200, done.text
            assert client.get(f"/api/v1/sessions/{sid}/download").content == data


def test_concurrent_repair_and_abort_leave_no_half_state(client: TestClient) -> None:
    data = b"A" * 1000 + b"B" * 1000 + b"C" * 1000
    start, end = 1000, 2000
    bad = b"X" * 1000

    for round_no in range(6):
        sid = _create(client, data, name=f"abort-race-{round_no}.bin")
        _upload_with_bad_chunk(client, sid, data, start, end, bad)

        outcome: dict[str, int] = {}
        barrier = threading.Barrier(2)

        def do_abort() -> None:
            barrier.wait()
            outcome["abort"] = client.delete(f"/api/v1/sessions/{sid}").status_code

        def do_repair() -> None:
            barrier.wait()
            outcome["repair"] = _repair(
                client, sid, start, end, data[start:end]
            ).status_code

        threads = [
            threading.Thread(target=do_abort, daemon=True),
            threading.Thread(target=do_repair, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads)

        assert outcome["abort"] == 204
        # 放弃先于修复：404（会话已硬删除）；修复先于放弃：200 后被删。
        assert outcome["repair"] in (200, 404, 410)

        # 两种结局都不残留分片目录/临时文件；不存在半替换文件。
        assert not storage.parts_dir(sid).exists()
        assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
        _no_leftover_tempfiles()


def test_concurrent_repairs_serialize_without_corruption(client: TestClient) -> None:
    data = bytes((i * 11 + 1) % 256 for i in range(4000))
    start, end = 1000, 3000
    sid = _create(client, data)
    _upload_with_bad_chunk(
        client, sid, data, start, end, b"~" * (end - start)
    )

    results: list[int] = []
    barrier = threading.Barrier(4)
    # 两路正确内容、两路等长错误内容并发修复：区间始终精确匹配，
    # 因此全部串行成功（最后一个写者决定内容），登记与字节数始终不变。
    payloads = [data[start:end], data[start:end], b"!" * (end - start), b"?" * (end - start)]

    def worker(payload: bytes) -> None:
        barrier.wait()
        results.append(_repair(client, sid, start, end, payload).status_code)

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not any(t.is_alive() for t in threads)
    assert results == [200, 200, 200, 200]

    part = storage.part_path(sid, start, end)
    assert part.stat().st_size == end - start  # 任何时刻都等长，无半文件
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["bytes_received"] == len(data)
    assert status["missing_ranges"] == []

    # 再用正确内容修复一次即可完成（无论最后写者是谁）。
    assert _repair(client, sid, start, end, data[start:end]).status_code == 200
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data
    _no_leftover_tempfiles()
