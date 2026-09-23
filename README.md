# MDredd

⚠️ This repository is under heavy development and tailored heavily for MHacks.

This is a realistic pair-wise judging algorithm API based on [Bayesian Decision Process for Cost-Efficient Dynamic Ranking via Crowdsourcing](https://www.jmlr.org/papers/v17/16-066.html).

<u>**Features:**</u>
- Just In Time via JAX
- Crash resistance through a custom Write-Ahead Logging system

## Judge flow

GraphQL is served at `POST /`. Send `Authorization: Bearer <api-key>` on every HTTP and websocket request. The server stores only the SHA-256 hex digest of each key.

```sh
python -c "import hashlib; print(hashlib.sha256(b'your-key').hexdigest())"
```

```sh
MDREDD_API_KEYS='[{"key_hash":"<digest>","user_id":"judge-1","role":"judge"}]'
```

`admin` keys can call `startJudging`, `stopJudging`, `resumeJudging`, and `crash`. `judge` keys can call `columns`, `row`, `session`, `pair`, `submitComparison`, `rankings`, and `rankingsUpdated`. The judge id is the key's user id. `pair` and `submitComparison` do not take a client-supplied judge id.

`columns` is the uploaded CSV header. Row attributes are whichever of those columns you name:

```graphql
query {
  columns
  pair(force: false) {
    id
    attributes(names: ["Project Title"]) {
      name
      value
    }
  }
}
```

Submit the chosen row indexes:

```graphql
mutation {
  submitComparison(entityIds: [0, 1], winnerId: 0)
}
```

Poll `pair` or `rankings` on an interval, or subscribe to `rankingsUpdated` on `ws://<host>/` with the same `Authorization` header and the `graphql-transport-ws` subprotocol. A judge can hold one rankings subscription. Another attempt spends from that judge's `pair` budget and does not open a second stream.

```graphql
subscription {
  rankingsUpdated {
    id
    attributes(names: ["Project Title"]) {
      name
      value
    }
  }
}
```

When a bucket is empty the response is a GraphQL error and the judge state is left unchanged:

```json
{
  "errors": [
    {
      "message": "Rate limit exceeded",
      "extensions": {
        "code": "RATE_LIMITED",
        "retryAfterMs": 5000
      }
    }
  ]
}
```

`pair` refills quickly enough to refresh, and slowly enough that `force: true` cannot be cycled. `submitComparison` refills more slowly. Admin mutations share a smaller bucket. Per-key `pair_capacity`, `pair_refill_per_second`, `submit_capacity`, `submit_refill_per_second`, `admin_capacity`, and `admin_refill_per_second` override the defaults.
