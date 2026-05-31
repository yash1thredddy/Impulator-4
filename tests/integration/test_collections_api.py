"""Integration tests for the collections router (Phase 23, plan 23-06).

Endpoints under /api/v1/collections (design §6.3):
  POST   /collections           -> create + submit (returns the collection)
  GET    /collections           -> CollectionListResponse (GLOBAL, D-05)
  GET    /collections/{id}      -> CollectionDetailResponse
  DELETE /collections/{id}      -> soft delete + ZIP removal
  GET    /collections/{id}/download -> stream the ZIP

These exercise the real router against the testcontainers/Postgres harness
(tests/integration/conftest.py -> ``client`` fixture). They require Docker /
``TEST_DATABASE_URL`` to provision the ``pg_engine`` testcontainer; in CI's
``services: postgres`` they run and pass. The ``scheduler.trigger`` the POST
handler calls is patched by the ``client`` fixture, so no real job runs.
"""


def _valid_payload(name="FlavonoidComparison", author="Jane Doe"):
    """Two-member payload satisfying D-06 (min 2) + D-03 (clean name)."""
    return {
        "name": name,
        "author_name": author,
        "description": "Comparing common dietary flavonoids.",
        "members": [
            {"name": "Quercetin", "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"},
            {"name": "Kaempferol", "smiles": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"},
        ],
    }


def test_create_returns_ids(client):
    """POST /collections creates a COLLECTION job + collection row and returns
    their ids (the collection ``id`` and its linked ``job_id``)."""
    resp = client.post("/api/v1/collections", json=_valid_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # The created collection carries both its own id and the linked job_id.
    assert body["id"], "collection id missing"
    assert body["job_id"], "linked job_id missing"
    assert body["name"] == "FlavonoidComparison"
    assert body["author_name"] == "Jane Doe"


def test_collection_job_is_visible_to_its_session(client):
    """The collection's job is stamped with the caller's X-Session-ID so it
    shows up in the session-scoped active-jobs view the sidebar polls.

    Regression for: the POST handler ignored the session header and created the
    job with session=None, so it never appeared in the sidebar (no progress bar).
    """
    sid = "33333333-3333-4333-8333-333333333333"
    resp = client.post(
        "/api/v1/collections",
        json=_valid_payload(name="SessionScoped"),
        headers={"X-Session-ID": sid},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]

    # The sidebar polls GET /jobs/active, filtered by X-Session-ID. The collection
    # job (PENDING — scheduler.trigger is patched) must be visible to its session.
    active = client.get("/api/v1/jobs/active", headers={"X-Session-ID": sid})
    assert active.status_code == 200, active.text
    job_ids = [j.get("id") for j in active.json()]
    assert job_id in job_ids, (
        f"collection job {job_id} not visible to its own session {sid}"
    )

    # And it must NOT leak into a different session's active-jobs view.
    other = client.get(
        "/api/v1/jobs/active",
        headers={"X-Session-ID": "44444444-4444-4444-8444-444444444444"},
    )
    assert other.status_code == 200, other.text
    assert job_id not in [j.get("id") for j in other.json()]


def test_name_rejects_path_traversal(client):
    """D-03: a ``../`` collection name is rejected at the schema boundary (422),
    and nothing is persisted."""
    payload = _valid_payload(name="../etc/passwd")
    resp = client.post("/api/v1/collections", json=payload)
    # Pydantic validation failure surfaces as 422 (request-body validation);
    # the router's own ValueError path would be 400. Either way it is a 4xx
    # rejection and the collection is never created.
    assert resp.status_code in (400, 422), resp.text

    # Confirm nothing leaked into the global list.
    listing = client.get("/api/v1/collections")
    assert listing.status_code == 200, listing.text
    names = [c["name"] for c in listing.json()["items"]]
    assert "../etc/passwd" not in names
    assert "etc/passwd" not in names


def test_list_is_global(client):
    """D-05: GET /collections lists every collection regardless of session
    (no per-session filtering), like the entries page."""
    # Two collections created under DIFFERENT session ids.
    p1 = _valid_payload(name="CollectionAlpha")
    p1["session_id"] = "11111111-1111-1111-1111-111111111111"
    p2 = _valid_payload(name="CollectionBeta")
    p2["session_id"] = "22222222-2222-2222-2222-222222222222"

    r1 = client.post("/api/v1/collections", json=p1)
    r2 = client.post("/api/v1/collections", json=p2)
    assert r1.status_code == 201, r1.text
    assert r2.status_code == 201, r2.text

    # The global list returns BOTH, with no X-Session-ID header supplied -- the
    # handler takes no SessionDep, so session never filters the result.
    listing = client.get("/api/v1/collections")
    assert listing.status_code == 200, listing.text
    names = {c["name"] for c in listing.json()["items"]}
    assert {"CollectionAlpha", "CollectionBeta"}.issubset(names)


def test_detail_surfaces_failed_members(client, db_session):
    """D-PF-6: GET /collections/{id} exposes the linked job's
    ``result_summary.failed_members`` so the frontend can render per-member
    failure + lower-tier cascade hints. The payload lives on the JOB (it must
    survive the D-11 auto-delete of the collection row), so the detail endpoint
    reads it off the linked job, not the collection row.
    """
    import uuid as _uuid

    from backend.repositories.job_repository import job_repo

    resp = client.post("/api/v1/collections", json=_valid_payload(name="CascadeSurface"))
    assert resp.status_code == 201, resp.text
    collection_id = resp.json()["id"]
    job_id = resp.json()["job_id"]

    # Persist a failed_members payload onto the linked job's result_summary.
    job = job_repo.get_by_job_id(db_session, _uuid.UUID(job_id))
    assert job is not None
    failed_members = [
        {"name": "Quercetin", "error": "No bioactivity data found",
         "cascade_results": [{"threshold": 50, "count": 1}, {"threshold": 40, "count": 29}]},
        {"name": "Kaempferol", "error": "boom", "cascade_results": None},
    ]
    job.result_summary = {"failed_members": failed_members}
    db_session.commit()

    detail = client.get(f"/api/v1/collections/{collection_id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["failed_members"] == failed_members


def test_detail_and_list_fold_linked_job_status(client, db_session):
    """403 fix: collections are GLOBAL but their job is session-scoped, so a
    non-owner session can't read /jobs/{id} (403) — the detail/list endpoints
    therefore fold the linked job's status/progress/message onto the (global)
    collection payload so the frontend never makes the session-owned job call.
    """
    import uuid as _uuid

    from backend.models.enums import JobStatus
    from backend.repositories.job_repository import job_repo

    resp = client.post("/api/v1/collections", json=_valid_payload(name="StatusFold"))
    assert resp.status_code == 201, resp.text
    collection_id = resp.json()["id"]
    job_id = resp.json()["job_id"]

    job = job_repo.get_by_job_id(db_session, _uuid.UUID(job_id))
    assert job is not None
    job.status = JobStatus.PROCESSING
    job.progress = 42.0
    job.current_step = "Processed 3/7 members"
    job.error_message = None
    db_session.commit()

    body = client.get(f"/api/v1/collections/{collection_id}").json()
    assert body["status"] == "processing"
    assert body["progress"] == 42.0
    assert body["message"] == "Processed 3/7 members"  # current_step while running

    # A failed job surfaces error_message (not current_step) as `message`.
    job.status = JobStatus.FAILED
    job.error_message = "All 7 collection members failed"
    db_session.commit()
    body = client.get(f"/api/v1/collections/{collection_id}").json()
    assert body["status"] == "failed"
    assert body["message"] == "All 7 collection members failed"

    # The GLOBAL list endpoint folds status onto each summary too.
    listing = client.get("/api/v1/collections").json()
    row = next(r for r in listing["items"] if r["id"] == collection_id)
    assert row["status"] == "failed"


def test_detail_surfaces_failed_members_partial_success(client, db_session):
    """D-PF-6 (partial-success bridge): when SOME members fail but the collection
    still COMPLETES, ``_finalize_job_sync`` must persist ``stats['failed_members']``
    into the linked job's ``result_summary`` so the detail endpoint surfaces it.

    Regression: the partial-success path computed ``stats['failed_members']`` but
    ``_finalize_job_sync`` only wrote the known Collection columns and dropped it,
    so a completed-with-failures collection (the PRIMARY viewable case — fully
    failed ones are auto-soft-deleted, D-11) surfaced nothing.
    """
    import uuid as _uuid

    from backend.services import collection_service as cs

    resp = client.post(
        "/api/v1/collections", json=_valid_payload(name="PartialSurface")
    )
    assert resp.status_code == 201, resp.text
    collection_id = resp.json()["id"]
    job_id = resp.json()["job_id"]

    failed_members = [
        {"name": "Kaempferol", "error": "No bioactivity data found",
         "cascade_results": [{"threshold": 50, "count": 2}]},
    ]
    # Drive the REAL finalize path (no manual result_summary set) with a stats dict
    # carrying failed_members — exactly what process_collection_job builds on a
    # partial-success run. get_db_session() routes to the test DB via the client
    # fixture's SessionLocal rebind.
    cs._finalize_job_sync(
        _uuid.UUID(job_id),
        collection_id,
        {
            "compound_count": 1,
            "member_failed_count": 1,
            "failed_members": failed_members,
        },
    )

    detail = client.get(f"/api/v1/collections/{collection_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["failed_members"] == failed_members
