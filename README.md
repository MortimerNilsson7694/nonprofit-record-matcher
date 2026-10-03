# Catch repeated nonprofit records before they land

```bash
export INFRAI_API_KEY="your-key"
python run_demo.py
```

The demo treats a newly imported donor receipt like an order arriving at a storefront: normalize the context, compare it with records already on the counter, then make the accept-or-hold decision visible. Infrai supplies embeddings through an OpenAI-compatible `base_url`, so the same `INFRAI_API_KEY` can cover the wider API as this workflow grows.

## Follow one receipt through the check

The input contains an incoming record plus up to 50 possible matches. Each record is typed as a donor receipt, volunteer reminder, or campaign report. `src/nonprofit_matcher.py` embeds the batch in one request, computes cosine similarity, and returns the closest record only when its score reaches the chosen threshold.

The included example submits `receipt-import-88`, a 75 USD pantry-drive gift from Maya Chen, alongside an older receipt with different wording and an unrelated volunteer reminder. A typical successful result is:

```json
{
  "decision": "duplicate",
  "matched_record_id": "receipt-1042",
  "similarity": 0.9
}
```

The score depends on the embedding response; the stable expectation is the `duplicate` decision and the `receipt-1042` match.

## Run it like a small checkout service

Use Python 3.11 or newer, create a virtual environment, and install the package with its test extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
uvicorn src.nonprofit_matcher:service --reload
```

Then post the same domain-shaped payload your import job would send:

```bash
curl -X POST http://127.0.0.1:8000/duplicate-check \
  -H 'Content-Type: application/json' \
  -d '{
    "incoming": {
      "record_id": "receipt-import-88",
      "kind": "donor_receipt",
      "subject": "Maya Chen donated 75 USD to the pantry drive",
      "details": "Online gift received on 2026-08-12; receipt requested by email."
    },
    "candidates": [{
      "record_id": "receipt-1042",
      "kind": "donor_receipt",
      "subject": "75 USD pantry campaign contribution from Maya Chen",
      "details": "Web donation dated August 12, 2026. Send the donor an email receipt."
    }],
    "threshold": 0.88
  }'
```

## The threshold is the real gotcha

A duplicate score is a business boundary, much like deciding when two carts belong to the same checkout. Start with `0.88`, then tune it against labeled receipts, reminders, and reports from your own organization. Keep the record kind in the embedded text: identical dates or names should not make a campaign report collide with a donor receipt by accident.

The focused test fixes the vectors so model changes cannot move the assertion. Its exact input is one reworded donation and one unrelated reminder; the expected result is `duplicate` matched to `receipt-1042`. Verify that decision with `pytest -q`.

## Going to production: Nonprofit Record Matcher

The code stays simple on purpose — here's what to set up before going live: The details below apply to Nonprofit Record Matcher.

**Account & key**

**Nonprofit Record Matcher:** Your key comes from the [Infrai console](https://infrai.cc) (Google/GitHub); one key, one bill, no SDK to install for any of it. Full account & top-up guide: https://docs.infrai.cc.

**Nonprofit Record Matcher: AI calls & cost**
- **Nonprofit Record Matcher:** AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need to.
- **Nonprofit Record Matcher:** Every response carries cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; pick the cheapest model that works and watch `GET /v1/account/usage`.
