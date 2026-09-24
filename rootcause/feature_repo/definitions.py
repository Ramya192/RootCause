"""Feast feature repo definitions for the employee_attrition domain (Stage 2).

The feature set here is exactly the causal_discovery.variables list in
rootcause/configs/employee_attrition.yaml -- this file is data-source
plumbing, not a place to redefine what the domain's variables are.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

DATA_PATH = Path(__file__).resolve().parent / "data" / "features.parquet"

employee = Entity(name="employee", join_keys=["employee_id"], value_type=ValueType.INT64)

employee_source = FileSource(
    name="employee_source",
    path=str(DATA_PATH),
    timestamp_field="event_timestamp",
)

employee_causal_features = FeatureView(
    name="employee_causal_features",
    entities=[employee],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="compensation", dtype=Float32),
        Field(name="manager_quality", dtype=Float32),
        Field(name="workload", dtype=Float32),
        Field(name="job_satisfaction", dtype=Float32),
        Field(name="burnout", dtype=Float32),
        Field(name="attrition", dtype=Int64),
    ],
    source=employee_source,
    online=True,
)
