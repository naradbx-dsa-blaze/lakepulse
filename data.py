"""
LakePulse — data layer.

All queries run against real Databricks system tables.
Pass warehouse_id explicitly from the UI — no env var dependency at runtime.

Pricing sources:
  DBU pricing  → databricks.com/product/pricing (Enterprise, pay-as-you-go, 2024)
  Spot discount → AWS Spot Advisor / Azure Spot VMs: 70-90% typical (70% used — conservative)
  Job cluster  → Databricks docs: job clusters have zero idle cost vs always-on all-purpose
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

# ── DBU list prices by cloud (databricks.com/product/pricing, Enterprise, 2024) ─
# These are public list prices. Customers should override with their contracted rates.
CLOUD_DBU_DEFAULTS = {
    "AWS":   {"ALL_PURPOSE": 0.55, "JOBS": 0.20, "DLT": 0.36, "SQL": 0.22},
    "Azure": {"ALL_PURPOSE": 0.55, "JOBS": 0.20, "DLT": 0.36, "SQL": 0.22},
    "GCP":   {"ALL_PURPOSE": 0.55, "JOBS": 0.20, "DLT": 0.36, "SQL": 0.22},
}
DBU_PRICE         = CLOUD_DBU_DEFAULTS["AWS"]  # default; overridden at runtime by sidebar
DBU_PRICE_DEFAULT = 0.40                        # blended fallback for unknown product types


def _sql(warehouse_id: str, stmt: str) -> pd.DataFrame:
    import time
    w    = WorkspaceClient()
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=stmt, wait_timeout="50s"
    )
    # API max wait is 50s; poll until terminal state for longer queries
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(resp.status.error.message)
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array or []
    return pd.DataFrame([[v.str for v in r.data] for r in rows], columns=cols)


def _is_live(warehouse_id: str) -> bool:
    return bool(warehouse_id) and _HAS_SDK


def get_workspace_info(warehouse_id: str = "") -> dict:
    """
    Returns the workspace host URL and detected cloud provider.
    Cloud is inferred from the host pattern:
      - *.azuredatabricks.net  → Azure
      - *.gcp.databricks.com   → GCP
      - everything else        → AWS
    Falls back to {"host": "", "cloud": "AWS"} on any error.
    """
    if _is_live(warehouse_id):
        try:
            w    = WorkspaceClient()
            host = (w.config.host or "").rstrip("/")
            if "azuredatabricks" in host:
                cloud = "Azure"
            elif "gcp.databricks" in host:
                cloud = "GCP"
            else:
                cloud = "AWS"
            return {"host": host, "cloud": cloud}
        except Exception:
            pass
    return {"host": "", "cloud": "AWS"}


# ── DBU Waste ─────────────────────────────────────────────────────────────────
# system.billing.usage × system.compute.clusters
# estimated_cost_usd in SQL uses list prices; the app layer recalculates with user prices.

def get_dbu_waste(warehouse_id: str = "") -> pd.DataFrame:
    if _is_live(warehouse_id):
        return _sql(warehouse_id, """
            SELECT
                COALESCE(u.usage_metadata.cluster_id, 'unknown')        AS cluster_id,
                COALESCE(c.cluster_name, 'Unnamed')                     AS cluster_name,
                COALESCE(c.owned_by, 'unknown')                         AS owner,
                u.sku_name,
                u.billing_origin_product                                AS product,
                ROUND(SUM(u.usage_quantity), 2)                        AS total_dbu,
                ROUND(DATEDIFF(HOUR,
                    MIN(u.usage_start_time), MAX(u.usage_end_time)), 1) AS lifetime_hours,
                ROUND(SUM(u.usage_quantity) * 0.55, 2)                 AS estimated_cost_usd
            FROM system.billing.usage u
            LEFT JOIN system.compute.clusters c
                   ON u.usage_metadata.cluster_id = c.cluster_id
            WHERE u.usage_start_time >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
              AND u.billing_origin_product = 'ALL_PURPOSE'
            GROUP BY 1,2,3,4,5
            ORDER BY total_dbu DESC
            LIMIT 50
        """)

    rng    = np.random.default_rng(42)
    owners = ["alice@corp.com","bob@corp.com","charlie@corp.com","data-team@corp.com","ml-team@corp.com"]
    skus   = ["r5.2xlarge","r5.4xlarge","m5d.2xlarge","i3.xlarge","c5.4xlarge"]
    names  = ["etl-prod","ml-training","adhoc-analysis","dlt-pipeline","feature-eng",
              "data-science-ws","batch-scoring","report-refresh"]
    rows   = []
    for i in range(25):
        dbu = float(rng.uniform(15, 600))
        rows.append({
            "cluster_id":         f"cluster-{i:04d}",
            "cluster_name":       f"{random.choice(names)}-{i}",
            "owner":              random.choice(owners),
            "sku_name":           random.choice(skus),
            "product":            "ALL_PURPOSE",
            "total_dbu":          round(dbu, 2),
            "lifetime_hours":     round(float(rng.uniform(2, 168)), 1),
            "estimated_cost_usd": round(dbu * DBU_PRICE["ALL_PURPOSE"], 2),
        })
    return pd.DataFrame(rows)


# ── Bottlenecks ────────────────────────────────────────────────────────────────
# system.query.history
# shuffle_read_bytes / shuffle_write_bytes from the metrics struct

def get_bottlenecks(warehouse_id: str = "") -> pd.DataFrame:
    if _is_live(warehouse_id):
        return _sql(warehouse_id, """
            SELECT
                query_id,
                SUBSTR(statement_text, 1, 120)                              AS query_snippet,
                executed_by,
                start_time,
                ROUND(total_duration_ms / 60000.0, 2)                      AS duration_min,
                ROUND(COALESCE(metrics.shuffle_read_bytes,  0) / 1e9, 3)   AS shuffle_read_gb,
                ROUND(COALESCE(metrics.shuffle_write_bytes, 0) / 1e9, 3)   AS shuffle_write_gb,
                ROUND(COALESCE(metrics.peak_memory_bytes,   0) / 1e9, 3)   AS peak_memory_gb
            FROM system.query.history
            WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
              AND total_duration_ms > 60000
            ORDER BY shuffle_read_gb DESC
            LIMIT 50
        """)

    rng      = np.random.default_rng(7)
    users    = ["alice@corp.com","bob@corp.com","etl-svc@corp.com","analyst@corp.com"]
    snippets = [
        "SELECT /*+ REPARTITION(200) */ * FROM sales s JOIN customers c ON s.id = c.id ...",
        "INSERT OVERWRITE delta.`/mnt/gold/orders` SELECT date, SUM(amount) FROM raw GROUP BY ...",
        "SELECT customer_id, COUNT(*), SUM(amount) FROM transactions GROUP BY 1",
        "MERGE INTO target t USING source s ON t.id = s.id WHEN MATCHED THEN UPDATE ...",
        "SELECT * FROM events LATERAL VIEW EXPLODE(items) t AS item WHERE event_date ...",
    ]
    now  = datetime.now()
    rows = []
    for i in range(18):
        shuffle = float(rng.uniform(0.1, 60))
        rows.append({
            "query_id":         f"qry-{i:06d}",
            "query_snippet":    random.choice(snippets),
            "executed_by":      random.choice(users),
            "start_time":       (now - timedelta(hours=float(rng.uniform(1, 168)))).isoformat(),
            "duration_min":     round(float(rng.uniform(1, 180)), 2),
            "shuffle_read_gb":  round(shuffle, 3),
            "shuffle_write_gb": round(shuffle * float(rng.uniform(0.3, 1.2)), 3),
            "peak_memory_gb":   round(float(rng.uniform(1, 64)), 3),
        })
    return pd.DataFrame(rows)


# ── Job History (SLA Oracle) ──────────────────────────────────────────────────
# system.lakeflow.job_run_timeline
# run_duration is in milliseconds per Databricks docs

def get_job_history(warehouse_id: str = "") -> pd.DataFrame:
    if _is_live(warehouse_id):
        return _sql(warehouse_id, """
            SELECT
                j.job_id,
                j.run_id,
                COALESCE(j.run_name, CONCAT('Job-', j.job_id))   AS job_name,
                j.creator_user_name,
                j.trigger_time,
                j.result_state,
                ROUND(j.run_duration / 60000.0, 2)               AS duration_min,
                ROUND(COALESCE(j.queued_time, 0) / 60000.0, 2)   AS queue_min
            FROM system.lakeflow.job_run_timeline j
            WHERE j.period_start_time >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
              AND j.result_state IS NOT NULL
            ORDER BY j.trigger_time DESC
            LIMIT 500
        """)

    rng      = np.random.default_rng(13)
    base_dur = {"etl-daily":45,"ml-training":120,"feature-pipeline":30,
                "dbt-run":20,"data-quality":15,"batch-scoring":60}
    states   = ["SUCCEEDED"]*7 + ["FAILED","TIMED_OUT"]
    users    = ["alice@corp.com","etl-svc@corp.com","ml-team@corp.com"]
    now      = datetime.now()
    rows     = []
    for i in range(120):
        name = random.choice(list(base_dur))
        dur  = base_dur[name] * float(rng.uniform(0.6, 3.0))
        rows.append({
            "job_id":            f"{abs(hash(name)) % 9999:04d}",
            "run_id":            f"run-{i:05d}",
            "job_name":          name,
            "creator_user_name": random.choice(users),
            "trigger_time":      (now - timedelta(hours=float(rng.uniform(1, 720)))).isoformat(),
            "result_state":      random.choice(states),
            "duration_min":      round(dur, 2),
            "queue_min":         round(float(rng.uniform(0, 12)), 2),
        })
    return pd.DataFrame(rows)


# ── Data Popularity ────────────────────────────────────────────────────────────
# system.access.audit

def get_data_popularity(warehouse_id: str = "") -> pd.DataFrame:
    if _is_live(warehouse_id):
        return _sql(warehouse_id, """
            SELECT
                request_params.full_name_arg                               AS table_name,
                COUNT(*)                                                    AS access_count,
                MAX(event_time)                                             AS last_accessed,
                COUNT(DISTINCT user_identity.email)                         AS unique_users,
                DATEDIFF(DAY, MAX(event_time), CURRENT_TIMESTAMP)          AS days_since_access
            FROM system.access.audit
            WHERE action_name IN ('getTable','selectFromTable','createTableAsSelect')
              AND event_time >= CURRENT_TIMESTAMP - INTERVAL 90 DAYS
              AND request_params.full_name_arg IS NOT NULL
            GROUP BY 1
            ORDER BY access_count DESC
            LIMIT 100
        """)

    rng      = np.random.default_rng(99)
    catalogs = ["prod_catalog","dev_catalog","ml_catalog"]
    schemas  = ["sales","marketing","finance","raw","silver","gold","features"]
    tables   = ["customers","orders","transactions","events","sessions","products","predictions"]
    rows     = []
    for i in range(50):
        days = int(rng.integers(0, 91))
        rows.append({
            "table_name":        f"{random.choice(catalogs)}.{random.choice(schemas)}.{random.choice(tables)}_{i}",
            "access_count":      int(rng.integers(0, 600)),
            "last_accessed":     (datetime.now() - timedelta(days=days)).isoformat(),
            "unique_users":      int(rng.integers(1, 25)),
            "days_since_access": days,
        })
    return pd.DataFrame(rows)


# ── Billing Trend ──────────────────────────────────────────────────────────────
# system.billing.usage
# estimated_cost_usd here uses list prices; app layer recalculates with user prices.

def get_billing_trend(warehouse_id: str = "") -> pd.DataFrame:
    if _is_live(warehouse_id):
        return _sql(warehouse_id, """
            SELECT
                DATE_TRUNC('day', usage_start_time)  AS date,
                billing_origin_product                AS product,
                ROUND(SUM(usage_quantity), 2)         AS total_dbu,
                ROUND(SUM(
                    usage_quantity * CASE billing_origin_product
                        WHEN 'ALL_PURPOSE' THEN 0.55
                        WHEN 'JOBS'        THEN 0.20
                        WHEN 'DLT'         THEN 0.36
                        WHEN 'SQL'         THEN 0.22
                        ELSE 0.40
                    END
                ), 2)                                 AS estimated_cost_usd
            FROM system.billing.usage
            WHERE usage_start_time >= CURRENT_TIMESTAMP - INTERVAL 90 DAYS
            GROUP BY 1, 2
            ORDER BY 1
        """)

    rng      = np.random.default_rng(55)
    products = ["ALL_PURPOSE","JOBS","DLT","SQL"]
    base     = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows     = []
    for d in range(90):
        day = base - timedelta(days=89 - d)
        for prod in products:
            trend    = 1 + 0.004 * d
            seasonal = 0.85 + 0.15 * abs((d % 7) - 3) / 3
            dbu      = float(rng.uniform(60, 280)) * trend * seasonal
            price    = DBU_PRICE.get(prod, DBU_PRICE_DEFAULT)
            rows.append({
                "date":               day.date().isoformat(),
                "product":            prod,
                "total_dbu":          round(dbu, 2),
                "estimated_cost_usd": round(dbu * price, 2),
            })
    return pd.DataFrame(rows)
