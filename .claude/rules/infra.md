---
paths:
  - "infra/**"
  - "docker/**"
  - "docker/airflow-local/docker-compose.yaml"
  - ".github/workflows/**"
---

# Infrastructure rules
- Load the `mlops-deployment` skill before changing infrastructure, images or workflows.
- Locally run only `terraform fmt`, `validate` and `plan`; `apply` happens in CI after review.
- No service-account JSON keys; GitHub Actions authenticate through Workload Identity Federation.
- Pin base images, Terraform providers and third-party actions. Keep workflow `permissions:` minimal.
