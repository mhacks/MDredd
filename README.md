# MDredd

This repository is under heavy development and tailored for MHacks.

MDredd is a pairwise judging API. An organizer uploads a CSV of projects. Each judge is given one pair at a time, picks a winner (or reports that someone is absent), and the server folds that outcome into a ranking. Pair selection and the strength model follow [Bayesian Decision Process for Cost-Efficient Dynamic Ranking via Crowdsourcing](https://www.jmlr.org/papers/v17/16-066.html).

- Pair sampling and strength updates run just-in-time through JAX.
- Every accepted change is committed to SQLite (WAL, full fsync) before the response is sent, so a retry after a lost response does not double-count.

## Run

The API listens on port 8000. Set a token of at least 32 characters (for example `openssl rand -hex 32`).

```bash
export MDREDD_API_TOKEN=...
docker compose up --build
```

The database lives in the `mdredd-data` volume at `/app/data/mdredd.db`. `GET /health` needs no token and returns `{"status":"ok"}` while the judge worker is alive.

| Variable | Default | Role |
|---|---|---|
| `MDREDD_API_TOKEN` | required | Bearer token for every route except `/health` |
| `MDREDD_DB_FILE` | `mdredd.db` | SQLite path |
| `MDREDD_CORS_ORIGINS` | `["http://localhost:8000"]` | Allowed browser origins |
| `MDREDD_STRIKE_LIMIT` | `3` | Consecutive absences before a project is removed from the draw |
| `MDREDD_MIN_JUDGMENTS` | `5` | Appearances each active project gets before open sampling |

## Authentication

Every route except `GET /health` requires:

```http
Authorization: Bearer <MDREDD_API_TOKEN>
```

The token is one shared secret for organizers and judges. It does not identify a person. Send the judge's identity as `judge_id` in the body. A missing token is `401` `missing_key`. An unknown token is `401` `unknown_key`.

Errors are JSON: `{"detail":{"code":"..."}}`. Rate limits add `retry_after_ms` and a `Retry-After` header.

## Organizer flow

1. Upload projects with `POST /datasets` as multipart form data, file field `entities_csv`. The CSV must be UTF-8, have unique non-empty headers, and contain at least two data rows. Row ids are the zero-based index in file order. Cell values are returned later as strings.
2. A successful upload is `201` and turns judging on:

   ```json
   { "is_started": true, "headers": ["name", "track"] }
   ```

   Uploading that same CSV again while judging is on succeeds and changes nothing. A different CSV while judging is on is `409` `JUDGING_ALREADY_STARTED`. Stop judging, then upload.
3. `POST /judging/stop` rejects new pairs and comparisons and keeps the dataset, open pairs, strikes, and rankings. `POST /judging/start` and `POST /judging/resume` are the same call: turn judging back on. Repeating the call that matches the current state succeeds. `GET /judging` returns `{ "is_started": true }` or `false`.
4. `GET /pool` lists every project in upload order: `id`, `attributes`, `strikes`, and `removed`. `removed` is true once `strikes` reaches the strike limit. `POST /pool/{id}/restore` clears that project's strikes and returns it to the draw. Restoring a project that is still active succeeds and changes nothing.
5. `GET /projects` lists every project in upload order. The response is ids and attributes only. Before any dataset exists the list is empty. `GET /rankings` returns every row, strongest first, with the same shape. Rankings stay readable after stop. Before any dataset exists they are `409` `JUDGING_NEVER_STARTED`. `GET /columns` lists headers. `GET /rows/{id}` returns one row, or `404` `UNKNOWN_ROW`.
6. `POST /archive` moves the SQLite database and the log file into a new folder under `archive/` and starts empty. Each call keeps the earlier folders. The response `path` is that folder's name. `GET /archives` lists those names, newest first, and `GET /archives/{id}` downloads that folder as a zip. An unknown id is `404` `UNKNOWN_ARCHIVE`. Judging is off. If startup cannot read the file, it logs that and keeps serving. Other routes are `503` `DATABASE_UNREADABLE` until `POST /archive`.

## Judge flow

`GET /projects` lists every project in upload order. The response is ids and attributes only, the same list an organizer receives. Before any dataset exists the list is empty.

A judge holds at most one open pair. The screen loop is: ask for a pair, show the two projects, submit a winner or an absence, then ask again.

```text
POST /pairs  →  show the two projects  →  POST /comparisons  →  POST /pairs
```

**Ask for the current pair.** `POST /pairs` with an empty absence list returns the pair this judge already holds. It draws a new pair only when they have none. Call this on load and after a refresh.

```json
{ "judge_id": "judge-42", "absent": [] }
```

```json
{
  "pair": [
    { "id": 3, "attributes": { "name": "Project A" } },
    { "id": 11, "attributes": { "name": "Project B" } }
  ]
}
```

**Record a winner.** `entity_ids` must be the two ids of that judge's open pair. Order does not matter. `winner_id` must be one of them.

```json
{ "judge_id": "judge-42", "entity_ids": [3, 11], "winner_id": 3 }
```

Success is `{ "ok": true }`. The open pair is cleared, and the next `POST /pairs` draws another. Sending that same comparison again after it has landed returns `{ "ok": true }` and does not count twice. A different pair, or a comparison when this judge has no open pair, is `409` `JUDGE_DOES_NOT_OWN_PAIR`. A winner outside the two ids is `422` `INCORRECT_PAIR_FORMAT`. On `JUDGE_DOES_NOT_OWN_PAIR`, drop the local pair and call `POST /pairs` with `"absent": []`.

**Report an absence.** Call `POST /pairs` again with those ids in `absent`. They must belong to the pair this judge currently holds. Anything else is `409` `ABSENT_NOT_IN_PAIR`.

- One id: the project that is present wins. That is a real comparison. The missing project takes one strike. The response is the next pair.
- Both ids: nobody wins. Both projects take a strike, and the appearance count from drawing that pair is undone. The response is a new pair that includes neither of them.

A project that is shown has its strike streak reset to 0. A strike is recorded only for an absence. At the strike limit the project leaves future draws and stays in `/pool` and `/rankings` with `removed: true`. Retrying the same absence body returns the replacement pair already drawn.

If fewer than two projects are still active, `POST /pairs` is `409` `POOL_EXHAUSTED`. While judging is stopped, `POST /pairs` and `POST /comparisons` are `409` `JUDGING_NOT_STARTED`.

## How pairs and rankings are chosen

Each project has a strength, and every project starts equal. A win raises it and a loss lowers it. `GET /rankings` sorts by that strength.

Drawing a pair is separate from strength. Each draw increments an appearance count for both projects. Until every active project has been drawn `MDREDD_MIN_JUDGMENTS` times, new pairs come from the projects still under that floor. After that, projects shown less often are more likely to be drawn. Removed projects are never drawn. The client displays the pair it is given. It does not choose who is compared.

## Limits

Rate limits are global for the process, shared by every judge:

| Calls | Burst | Refill |
|---|---|---|
| `POST /pairs` | 6 | about 1 every 5 seconds |
| `POST /comparisons` | 2 | about 1 per minute |
| Admin writes (upload, start, stop, restore) | 4 | about 1 every 30 seconds |

`429` is `{"detail":{"code":"RATE_LIMITED","retry_after_ms":...}}` plus `Retry-After`. Retry the same body. `GET /judging`, `/projects`, `/rankings`, `/pool`, `/rows`, and `/columns` are not limited.

If the judge worker is dead, stuck, or its queue is full, the call is `503` `WORKER_UNAVAILABLE`. State already committed is kept. Retry shortly.

| HTTP | `detail.code` | When |
|---|---|---|
| 401 | `missing_key`, `unknown_key` | Token missing or wrong |
| 404 | `UNKNOWN_ROW` | Row id is not in the dataset |
| 409 | `JUDGING_NOT_STARTED` | Judging is paused |
| 409 | `JUDGING_ALREADY_STARTED` | A different CSV was uploaded while judging is on |
| 409 | `JUDGING_NEVER_STARTED` | No dataset has been stored |
| 409 | `JUDGE_DOES_NOT_OWN_PAIR` | Comparison does not match this judge's open pair |
| 409 | `ABSENT_NOT_IN_PAIR` | Absence ids are not the current pair |
| 409 | `POOL_EXHAUSTED` | Fewer than two projects are still active |
| 422 | `TOO_FEW_ENTITIES` | CSV has fewer than two rows |
| 422 | `INVALID_COLUMNS` | Headers are empty, duplicated, or unreadable. `detail.names` lists the bad headers when known |
| 422 | `INCORRECT_PAIR_FORMAT` | Winner is not one of the two ids |
| 429 | `RATE_LIMITED` | Shared bucket is empty |
| 503 | `WORKER_UNAVAILABLE` | Judge worker cannot accept the command |
