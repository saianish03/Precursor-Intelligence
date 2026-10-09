"""Storage ports (interfaces) and their adapters.

Pipeline tasks only ever talk to these three ports:
  * ObjectStore    - blobs/files (zips, extracted CSVs, raw API payloads, reports)
  * MetadataStore  - pipeline bookkeeping tables (catalog, download_log, inventory, ...)
  * TableSink      - analytical tables (bronze/silver; used from Step 2 onward)
Which adapter backs each port is chosen by configs/env/<env>.yaml through io.registry.
"""
