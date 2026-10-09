# frontend/: dashboard (Streamlit) → Cloud Run `precursor-ui`

Reads data only through the backend API, not directly from BigQuery.

```
frontend/
  app.py           # Streamlit entry point, login + email allowlist   (to be added)
  pages/           # Facility, Account and Ops views
  components/      # reusable charts, tables, brief renderer
```
Tests: `tests/frontend/`. Image: `docker/frontend.Dockerfile` (planned).
