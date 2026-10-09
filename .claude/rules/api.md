---
paths:
  - "backend/**"
  - "tests/backend/**"
---

# API rules
- FastAPI with Pydantic request and response models; every endpoint documented in OpenAPI.
- Authentication is required on every endpoint except `/health`; follow the UI-hosting ADR for Base44 calls.
- Errors return a consistent JSON shape with a clear message; never leak stack traces or secrets.
- Reads come from BigQuery (with `maximum_bytes_billed`) and the GCS status snapshot; the API never reaches
  the VM's Postgres directly.
- LLM-backed endpoints log one structured line per call and enforce the daily cap.
