# precursor-airflow: the official Airflow 3 image plus the pinned Docker provider and the DAGs themselves.
# It runs the scheduler, API server, DAG processor and triggerer; pipeline work runs in precursor-pipeline
# containers that the DockerOperator starts.
#
#   docker build -f docker/airflow.Dockerfile -t precursor-airflow:<tag> .
# Base images by tag and digest (L10): rebuilds start from the same bytes; Dependabot proposes digest updates.
FROM apache/airflow:3.1.8-python3.12@sha256:740da888bf8367e75e6db8c8cd62c4d7b4e6744cdf0e9b53bf4467458f8dadab

ARG AIRFLOW_VERSION=3.1.8
ARG DOCKER_PROVIDER_VERSION=4.6.1
# The commit the tag constraints-3.1.8 points to: a tag can be moved, a commit can't (L10). Change it with the version.
ARG AIRFLOW_CONSTRAINTS_COMMIT=6b42e8020f6d21ca892b0d1dc21c60469fef6e5f

# The base image ships apache-airflow-providers-docker 4.5.2. 4.6.1 is the version PipelineDockerOperator's
# exit-code detection was checked against (dags/utils/pipeline_operator.py; tests/dags guards it). Installed
# under Airflow's own constraints for this release (with that one provider line swapped), so no other
# dependency moves between builds. pip, not uv: the official image installs into the airflow user's
# site-packages (uv has no --user mode). pytest is only for tests/dags.
RUN curl -fsSL -o /tmp/constraints.txt \
      "https://raw.githubusercontent.com/apache/airflow/${AIRFLOW_CONSTRAINTS_COMMIT}/constraints-3.12.txt" \
    && sed -i "s/^apache-airflow-providers-docker==.*/apache-airflow-providers-docker==${DOCKER_PROVIDER_VERSION}/" /tmp/constraints.txt \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt \
      "apache-airflow==${AIRFLOW_VERSION}" \
      "apache-airflow-providers-docker==${DOCKER_PROVIDER_VERSION}" \
      "pytest==8.4.2" \
    && rm /tmp/constraints.txt

# The code the Airflow services run, so one image (one git sha) is one release: the DAGs, the src package they import
# (only precursorintelligence.alerts, which needs nothing outside the standard library) and the compose bundle the VM
# deploys from. On a laptop, docker/airflow-local/docker-compose.override.yaml mounts the working tree over the first two.
ENV PYTHONPATH=/opt/airflow/src
COPY dags /opt/airflow/dags
COPY src/precursorintelligence /opt/airflow/src/precursorintelligence
COPY docker/airflow-local/docker-compose.yaml docker/airflow-local/docker-compose.vm.yaml \
     docker/airflow-local/init-db.sh /opt/precursor/compose/
# The base image's DAGs folder belongs to airflow (775) and COPY keeps a folder's owner, so the airflow user could add
# or delete DAGs. Root-owned and read-only, like the files in it and the old :ro mount.
USER root
RUN chown root:root /opt/airflow/dags && chmod 755 /opt/airflow/dags
USER airflow

# The commit the image was built from, so a pulled or retagged image still says which one it is. CD always passes it
# (and checks it before pushing); a laptop build may leave it empty. Both labels replace the base image's, which name
# Apache's repository and commit. Last, so changing the sha rebuilds no other layer.
ARG PI_GIT_SHA=""
LABEL org.opencontainers.image.source=https://github.com/saianish03/Precursor-Intelligence \
      org.opencontainers.image.revision=${PI_GIT_SHA}
