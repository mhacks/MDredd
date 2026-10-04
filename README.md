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
| `MDREDD_MIN_JUDGMENTS` | `3` | Appearances each active project gets before open sampling |
| `MDREDD_DEVPOST_COOKIE` | empty | `Cookie` header sent to Devpost when resolving submission URLs |
| `MDREDD_DEVPOST_CONCURRENCY` | `4` | Submission URLs resolved at once during upload |

## Authentication

Every route except `GET /health` requires:

```http
Authorization: Bearer <MDREDD_API_TOKEN>
```

The token is one shared secret for organizers and judges. It does not identify a person. Send the judge's identity as `judge_id` in the body. A missing token is `401` `missing_key`. An unknown token is `401` `unknown_key`.

Errors are JSON: `{"detail":{"code":"..."}}`. Rate limits add `retry_after_ms` and a `Retry-After` header.

## Organizer flow

1. Upload the Devpost projects export with `POST /datasets` as multipart form data, file field `entities_csv`. The CSV must be UTF-8 with unique header names, and must have the columns `Project Title`, `Submission Url`, `M Hacks Main Track`, and `Sponsor Opt In Prizes`; a missing one is `422` `INVALID_COLUMNS` with `detail.names`. Rows with an empty `Submission Url` (drafts) are dropped, and at least two must remain. Columns with a blank header, which spreadsheet apps add when they re-save the export, are dropped. Devpost headers only the first team member, so cells past the last header are dropped and short rows are padded with empty strings. Row ids are the zero-based index among the kept rows. Cell values are returned later as strings.

   Before storing anything, the upload follows each `Submission Url` to the public page it redirects to and stores it on that row as a `Project Url` column, which is appended to the headers. Every hop must stay on `https` `devpost.com`. If any row does not resolve, nothing is stored and the upload is `422` `DEVPOST_UNRESOLVED` with `detail.failures`, one `{ "title", "submission_url", "code" }` per failed row. `code` is `INVALID_DEVPOST_URL`, `DEVPOST_LOGIN_REQUIRED`, `DEVPOST_NOT_FOUND`, `DEVPOST_REDIRECTED_OFFSITE`, `DEVPOST_TOO_MANY_REDIRECTS`, or `DEVPOST_UNAVAILABLE`. While the hackathon's submissions are private, Devpost sends anonymous requests to its login page (`DEVPOST_LOGIN_REQUIRED`); set `MDREDD_DEVPOST_COOKIE` to the `Cookie` header of an organizer's logged-in Devpost session to resolve them anyway. Resolution makes one request per row, `MDREDD_DEVPOST_CONCURRENCY` at a time, so a large upload can take a while.
2. A successful upload is `201` and turns judging on:

   ```json
   { "is_started": true, "headers": ["name", "track"] }
   ```

   Uploading that same CSV again while judging is on succeeds, changes nothing, and does not contact Devpost. A different CSV while judging is on is `409` `JUDGING_ALREADY_STARTED`. Stop judging, then upload.
3. `POST /judging/stop` rejects new pairs and comparisons and keeps the dataset, open pairs, strikes, and rankings. `POST /judging/start` and `POST /judging/resume` are the same call: turn judging back on. Repeating the call that matches the current state succeeds. `GET /judging` returns `{ "is_started": true }` or `false`.
4. `GET /pool` lists every project in upload order: `id`, `attributes`, `strikes`, and `removed`. `removed` is true once `strikes` reaches the strike limit. `POST /pool/{id}/restore` clears that project's strikes and returns it to the draw. Restoring a project that is still active succeeds and changes nothing.
5. `GET /projects` lists every project in upload order. The response is ids and attributes only. Before any dataset exists the list is empty. `GET /rankings` returns every row, strongest first, with the same shape. Rankings stay readable after stop. Before any dataset exists they are `409` `JUDGING_NEVER_STARTED`. `GET /columns` lists headers. `GET /rows/{id}` returns one row, or `404` `UNKNOWN_ROW`.
6. `POST /archive` moves the SQLite database and the log file into a new folder under `archive/` and starts empty. Each call keeps the earlier folders. The response `path` is that folder's name. `GET /archives` lists those names, newest first, and `GET /archives/{id}` downloads that folder as a zip. An unknown id is `404` `UNKNOWN_ARCHIVE`. Judging is off. If startup cannot read the file, it logs that and keeps serving. Other routes are `503` `DATABASE_UNREADABLE` until `POST /archive`.
7. `PUT /tables` with `{"tables": {"https://devpost.com/software/project-a": 12, ...}}` replaces the whole project URL to table number mapping. Build it from each team's saved Devpost link and reserved table. Table numbers must be positive. URLs match a project's `Project Url` ignoring case, `www.`, a trailing slash, the query, and the scheme. The response is `{ "stored": 2, "unknown_urls": [...] }`, where `unknown_urls` are the sent URLs that match no uploaded project. The mapping is kept in SQLite and survives a new upload, so send it again whenever a team changes tables. Send `{"tables": {}}` to clear it. **Only projects with a table are drawn for judges**, so until the mapping is sent, `POST /pairs` is `409` `POOL_EXHAUSTED`. A pair a judge already holds is still returned if one of its projects loses its table.
8. `GET /export` downloads every project in upload order as `projects.csv`: `id`, every stored column (including `Project Url`), then `Table Number`, which is empty when no table is mapped to that project.

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
    {
      "id": 3,
      "url": "https://devpost.com/software/project-a",
      "name": "Project A",
      "tracks": ["Actually Intelligent (AI)", "Figma Best Design"]
    },
    { "id": 11, "url": "https://devpost.com/software/project-b", "name": "Project B", "tracks": [] }
  ],
  "assigned_at": 1790000000.0,
  "server_time": 1790000042.5
}
```

`url` is the row's `Project Url`, `name` its `Project Title`, and `tracks` its `M Hacks Main Track` followed by each prize in `Sponsor Opt In Prizes`. No other CSV column is sent to judges. Match `url` against the Devpost links teams saved to find the team and its table.

`assigned_at` is the Unix time this pair was handed out, and stays the same each time the judge asks for the pair they hold, including across restarts. `server_time` is MDredd's clock when it answered, so a client can time the pair as `server_time - assigned_at` without trusting its own clock.

**Skip a pair.** Call `POST /pairs` with the open pair's two ids in `skip`, for example when the judge's time runs out. Nothing is recorded about either project: no comparison, no strike, and the appearance the draw counted for each is given back. The new pair avoids both skipped projects when enough others are drawable. If `skip` no longer matches the judge's open pair, because it was already replaced, MDredd returns the current pair instead of skipping again, so a retry is safe. `skip` and `absent` cannot be sent together (`422`).

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

`POST /pairs` and `POST /comparisons` are limited per `judge_id`, so judges sharing one client (such as the dashboard) do not slow each other down. Admin writes share one limit for the process:

| Calls | Limit applies to | Burst | Refill |
|---|---|---|---|
| `POST /pairs` | each judge | 6 | about 1 every 5 seconds |
| `POST /comparisons` | each judge | 2 | about 1 per minute |
| Admin writes (upload, start, stop, restore) | everyone | 4 | about 1 every 30 seconds |
| `PUT /tables` | everyone | 30 | about 1 per second |

`429` is `{"detail":{"code":"RATE_LIMITED","retry_after_ms":...}}` plus `Retry-After`. Retry the same body. `GET /judging`, `/projects`, `/rankings`, `/pool`, `/rows`, `/columns`, and `/export` are not limited.

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
| 409 | `POOL_EXHAUSTED` | Fewer than two projects are still active and have a table |
| 422 | `TOO_FEW_ENTITIES` | CSV has fewer than two rows |
| 422 | `INVALID_COLUMNS` | Headers are duplicated, all blank, or unreadable, or a required column is missing. `detail.names` lists the bad headers when known |
| 422 | `DEVPOST_UNRESOLVED` | Some submission URLs did not resolve. `detail.failures` lists them |
| 422 | `INCORRECT_PAIR_FORMAT` | Winner is not one of the two ids |
| 429 | `RATE_LIMITED` | Shared bucket is empty |
| 503 | `WORKER_UNAVAILABLE` | Judge worker cannot accept the command |
