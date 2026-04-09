"""
LakePulse data layer.
Queries Databricks system tables when connected; returns synthetic demo data otherwise.

Set env vars to connect to real data:
  DATABRICKS_HOST       = https://<workspace>.azuredatabricks.net
  DATABRICKS_TOKEN      = dapi...
  DATABRICKS_WAREHOUSE_ID = <sql-warehouse-id>

Set LAKEPULSE_DEMO=1 to force demo mode.
"""
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
DEMO_MODE = (
    os.getenv("LAKEPULSE_DEMO", "0") == "1"
    or not WAREHOUSE_ID
    or not _HAS_SDK
    or not os.getenv("DATABRICKS_HOST")
)

# ── SQL execution ─────────────────────────────────────────────────────────────

def _sql(statement: str) -> pd.DataFrame:
    w = WorkspaceClient()
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        wait_timeout="60s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(resp.status.error.message)
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array or []
    return pd.DataFrame([[v.str for v in r.data] for r in rows], columns=cols)


# ── DBU Waste ─────────────────────────────────────────────────────────────────

_DBU_WASTE_SQL = """
SELECT
    COALESCE(u.usage_metadata.cluster_id, 'unknown')          AS cluster_id,
    COALESCE(c.cluster_name, 'Unnamed Cluster')               AS cluster_name,
    COALESCE(c.owned_by, 'unknown@example.com')               AS owner,
    u.sku_name,
    u.billing_origin_product                                   AS product,
    ROUND(SUM(u.usage_quantity), 2)                           AS total_dbu,
    MIN(u.usage_start_time)                                    AS first_seen,
    MAX(u.usage_end_time)                                      AS last_seen,
    ROUND(
        DATEDIFF(HOUR, MIN(u.usage_start_time), MAX(u.usage_end_time)), 1
    )                                                          AS lifetime_hours,
    ROUND(SUM(u.usage_quantity) * 0.55, 2)                   AS estimated_cost_usd
FROM system.billing.usage u
LEFT JOIN system.compute.clusters c
       ON u.usage_metadata.cluster_id = c.cluster_id
WHERE u.usage_start_time >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
  AND u.billing_origin_product = 'ALL_PURPOSE'
GROUP BY 1, 2, 3, 4, 5
ORDER BY total_dbu DESC
LIMIT 50
"""

def get_dbu_waste() -> pd.DataFrame:
    if not DEMO_MODE:
        return _sql(_DBU_WASTE_SQL)

    rng = np.random.default_rng(42)
    owners = ["alice@corp.com", "bob@corp.com", "charlie@corp.com",
              "data-team@corp.com", "ml-platform@corp.com"]
    skus = ["Standard_DS3_v2", "Standard_DS4_v2", "i3.xlarge", "r5.2xlarge", "m5d.4xlarge"]
    products = ["ALL_PURPOSE", "ALL_PURPOSE", "ALL_PURPOSE", "JOBS"]
    names = ["etl-prod", "ml-training", "adhoc-analysis", "dlt-pipeline",
             "feature-eng", "data-science-ws", "batch-scoring", "report-refresh"]
    now = datetime.now()
    rows = []
    for i in range(25):
        dbu = float(rng.uniform(15, 600))
        lifetime = float(rng.uniform(2, 168))
        rows.append({
            "cluster_id":         f"cluster-{i:04d}",
            "cluster_name":       f"{random.choice(names)}-{i}",
            "owner":              random.choice(owners),
            "sku_name":           random.choice(skus),
            "product":            random.choice(products),
            "total_dbu":          round(dbu, 2),
            "first_seen":         (now - timedelta(hours=lifetime + rng.uniform(0, 72))).isoformat(),
            "last_seen":          (now - timedelta(hours=float(rng.uniform(0, 24)))).isoformat(),
            "lifetime_hours":     round(lifetime, 1),
            "estimated_cost_usd": round(dbu * 0.55, 2),
        })
    return pd.DataFrame(rows)


# ── Bottleneck Detector ────────────────────────────────────────────────────────

_BOTTLENECK_SQL = """
SELECT
    query_id,
    SUBSTR(statement_text, 1, 120)                              AS query_snippet,
    executed_by,
    start_time,
    ROUND(total_duration_ms / 60000.0, 2)                      AS duration_min,
    ROUND(COALESCE(metrics.shuffle_read_bytes,  0) / 1e9, 3)   AS shuffle_read_gb,
    ROUND(COALESCE(metrics.shuffle_write_bytes, 0) / 1e9, 3)   AS shuffle_write_gb,
    ROUND(COALESCE(metrics.peak_memory_bytes,   0) / 1e9, 3)   AS peak_memory_gb,
    COALESCE(metrics.rows_read_count,    0)                     AS rows_read,
    COALESCE(metrics.rows_written_count, 0)                     AS rows_written
FROM system.query.history
WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
  AND total_duration_ms > 60000
ORDER BY shuffle_read_gb DESC
LIMIT 50
"""

def get_bottlenecks() -> pd.DataFrame:
    if not DEMO_MODE:
        return _sql(_BOTTLENECK_SQL)

    rng = np.random.default_rng(7)
    users = ["alice@corp.com", "bob@corp.com", "etl-svc@corp.com", "analyst@corp.com"]
    snippets = [
        "SELECT /*+ REPARTITION(200) */ * FROM sales s JOIN customers c ON s.cust_id = c.id WHERE ...",
        "INSERT OVERWRITE delta.`/mnt/gold/orders` SELECT date, region, SUM(amount) FROM raw GROUP BY ...",
        "SELECT customer_id, COUNT(*), SUM(amount) FROM transactions WHERE dt >= '2024-01' GROUP BY 1",
        "MERGE INTO target t USING (SELECT * FROM source) s ON t.id = s.id WHEN MATCHED THEN UPDATE ...",
        "SELECT * FROM events LATERAL VIEW EXPLODE(items) t AS item WHERE event_date BETWEEN ... AND ...",
        "CREATE TABLE silver.features AS SELECT *, LAG(amount,1) OVER (PARTITION BY id ORDER BY ts) ...",
    ]
    now = datetime.now()
    rows = []
    for i in range(18):
        shuffle = float(rng.uniform(0.1, 60))
        rows.append({
            "query_id":       f"qry-{i:06d}",
            "query_snippet":  random.choice(snippets),
            "executed_by":    random.choice(users),
            "start_time":     (now - timedelta(hours=float(rng.uniform(1, 168)))).isoformat(),
            "duration_min":   round(float(rng.uniform(1, 180)), 2),
            "shuffle_read_gb":  round(shuffle, 3),
            "shuffle_write_gb": round(shuffle * float(rng.uniform(0.3, 1.2)), 3),
            "peak_memory_gb":   round(float(rng.uniform(1, 64)), 3),
            "rows_read":        int(rng.integers(int(1e5), int(1e9))),
            "rows_written":     int(rng.integers(int(1e4), int(1e7))),
        })
    return pd.DataFrame(rows)


# ── SLA Oracle ────────────────────────────────────────────────────────────────

_JOB_HISTORY_SQL = """
SELECT
    j.job_id,
    j.run_id,
    COALESCE(j.run_name, CONCAT('Job-', j.job_id))  AS job_name,
    j.creator_user_name,
    j.trigger_time,
    j.result_state,
    ROUND(j.run_duration / 60.0, 2)                 AS duration_min,
    ROUND(COALESCE(j.queued_time, 0) / 60.0, 2)     AS queue_min
FROM system.lakeflow.job_run_timeline j
WHERE j.period_start_time >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
  AND j.result_state IS NOT NULL
ORDER BY j.trigger_time DESC
LIMIT 500
"""

def get_job_history() -> pd.DataFrame:
    if not DEMO_MODE:
        return _sql(_JOB_HISTORY_SQL)

    rng = np.random.default_rng(13)
    base_durations = {
        "etl-daily":        45,
        "ml-training":     120,
        "feature-pipeline": 30,
        "dbt-run":          20,
        "data-quality":     15,
        "batch-scoring":    60,
    }
    states = ["SUCCEEDED"] * 7 + ["FAILED", "TIMED_OUT"]
    users = ["alice@corp.com", "etl-svc@corp.com", "ml-team@corp.com"]
    now = datetime.now()
    rows = []
    for i in range(120):
        name = random.choice(list(base_durations))
        dur = base_durations[name] * float(rng.uniform(0.6, 3.0))
        rows.append({
            "job_id":             f"{abs(hash(name)) % 9999:04d}",
            "run_id":             f"run-{i:05d}",
            "job_name":           name,
            "creator_user_name":  random.choice(users),
            "trigger_time":       (now - timedelta(hours=float(rng.uniform(1, 720)))).isoformat(),
            "result_state":       random.choice(states),
            "duration_min":       round(dur, 2),
            "queue_min":          round(float(rng.uniform(0, 12)), 2),
        })
    return pd.DataFrame(rows)


# ── Data Popularity ───────────────────────────────────────────────────────────

_DATA_POPULARITY_SQL = """
SELECT
    request_params.full_name_arg                                             AS table_name,
    COUNT(*)                                                                  AS access_count,
    MAX(event_time)                                                           AS last_accessed,
    COUNT(DISTINCT user_identity.email)                                       AS unique_users,
    DATEDIFF(DAY, MAX(event_time), CURRENT_TIMESTAMP)                        AS days_since_access
FROM system.access.audit
WHERE action_name IN ('getTable','selectFromTable','createTableAsSelect','describeTable')
  AND event_time >= CURRENT_TIMESTAMP - INTERVAL 90 DAYS
  AND request_params.full_name_arg IS NOT NULL
GROUP BY 1
ORDER BY access_count DESC
LIMIT 100
"""

def get_data_popularity() -> pd.DataFrame:
    if not DEMO_MODE:
        return _sql(_DATA_POPULARITY_SQL)

    rng = np.random.default_rng(99)
    catalogs = ["prod_catalog", "dev_catalog", "ml_catalog"]
    schemas  = ["sales", "marketing", "finance", "raw", "silver", "gold", "features"]
    tables   = ["customers", "orders", "transactions", "events", "sessions",
                "products", "inventory", "returns", "predictions", "metrics"]
    rows = []
    for i in range(50):
        days_ago = int(rng.integers(0, 91))
        rows.append({
            "table_name":       f"{random.choice(catalogs)}.{random.choice(schemas)}.{random.choice(tables)}_{i}",
            "access_count":     int(rng.integers(0, 600)),
            "last_accessed":    (datetime.now() - timedelta(days=days_ago)).isoformat(),
            "unique_users":     int(rng.integers(1, 25)),
            "days_since_access": days_ago,
        })
    return pd.DataFrame(rows)


# ── Billing Trend ─────────────────────────────────────────────────────────────

_BILLING_TREND_SQL = """
SELECT
    DATE_TRUNC('day', usage_start_time)  AS date,
    billing_origin_product               AS product,
    ROUND(SUM(usage_quantity), 2)        AS total_dbu,
    ROUND(SUM(usage_quantity) * 0.55, 2) AS estimated_cost_usd
FROM system.billing.usage
WHERE usage_start_time >= CURRENT_TIMESTAMP - INTERVAL 90 DAYS
GROUP BY 1, 2
ORDER BY 1
"""

def get_billing_trend() -> pd.DataFrame:
    if not DEMO_MODE:
        return _sql(_BILLING_TREND_SQL)

    rng = np.random.default_rng(55)
    products = ["ALL_PURPOSE", "JOBS", "DLT", "SQL"]
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for d in range(90):
        day = base - timedelta(days=89 - d)
        for prod in products:
            # gradual upward trend with weekly seasonality
            trend   = 1 + 0.004 * d
            seasonal = 0.85 + 0.15 * abs((d % 7) - 3) / 3
            dbu = float(rng.uniform(60, 280)) * trend * seasonal
            rows.append({
                "date":               day.date().isoformat(),
                "product":            prod,
                "total_dbu":          round(dbu, 2),
                "estimated_cost_usd": round(dbu * 0.55, 2),
            })
    return pd.DataFrame(rows)


# ── ESG ───────────────────────────────────────────────────────────────────────
# 1 DBU ≈ 0.14 kWh  (rough industry estimate)
# US grid avg: 0.386 kg CO₂/kWh

KWH_PER_DBU = 0.14
KG_CO2_PER_KWH = 0.386
TREES_ABSORB_KG_CO2_PER_YEAR = 21.77

def get_esg_metrics() -> pd.DataFrame:
    df = get_billing_trend().copy()
    df["date"] = pd.to_datetime(df["date"])
    df["total_dbu"] = pd.to_numeric(df["total_dbu"], errors="coerce")
    df["kwh"]      = df["total_dbu"] * KWH_PER_DBU
    df["kg_co2"]   = df["kwh"] * KG_CO2_PER_KWH
    df["trees_equiv"] = df["kg_co2"] / (TREES_ABSORB_KG_CO2_PER_YEAR / 365)
    return df
