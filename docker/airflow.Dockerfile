# precursor-airflow: the official Airflow 3 image plus the pinned Docker provider the DAGs use.
# It runs the scheduler, API server, DAG processor and triggerer; pipeline work runs in precursor-pipeline
# containers that the DockerOperator starts.
#
#   docker build -f docker/airflow.Dockerfile -t precursor-airflow:<tag> .
FROM apache/airflow:3.1.8-python3.12

ARG AIRFLOW_VERSION=3.1.8
ARG DOCKER_PROVIDER_VERSION=4.6.1

# The base image ships apache-airflow-providers-docker 4.5.2. 4.6.1 is the version PipelineDockerOperator's
# exit-code detection was checked against (dags/utils/pipeline_operator.py; tests/dags guards it). Installed
# under Airflow's own constraints for this release (with that one provider line swapped), so no other
# dependency moves between builds. pip, not uv: the official image installs into the airflow user's
# site-packages (uv has no --user mode). pytest is only for tests/dags.
RUN curl -fsSL -o /tmp/constraints.txt \
      "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt" \
    && sed -i "s/^apache-airflow-providers-docker==.*/apache-airflow-providers-docker==${DOCKER_PROVIDER_VERSION}/" /tmp/constraints.txt \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt \
      "apache-airflow==${AIRFLOW_VERSION}" \
      "apache-airflow-providers-docker==${DOCKER_PROVIDER_VERSION}" \
      "pytest==8.4.2" \
    && rm /tmp/constraints.txt
