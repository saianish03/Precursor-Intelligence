---
paths:
  - "src/precursorintelligence/training/**"
  - "src/precursorintelligence/evaluation/**"
  - "src/precursorintelligence/serving/**"
  - "src/precursorintelligence/monitoring/**"
  - "configs/splits.yaml"
  - "tests/training/**"
  - "tests/evaluation/**"
---

# ML pipeline rules
- Load the `ml-engineering` skill before changing training, evaluation, gates, scoring or monitoring.
- Time-based, facility-grouped splits only; calibrate before evaluating; report bootstrap CIs.
- Promotion logic stays deterministic in `src/precursorintelligence/evaluation/gates.py`; agents never promote.
- Log every run to MLflow with the dataset manifest ID and git SHA.
