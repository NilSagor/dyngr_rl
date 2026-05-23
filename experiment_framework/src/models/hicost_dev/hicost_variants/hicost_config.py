from dataclasses import dataclass, field
from typing import Optional, Any, Literal
from enum import Enum
import warnings

class ModelVersion(str, Enum):
    V0 = "v0_tawrmac_baseline"
    V1 = "v1_plus_co_gnn"
    V2 = "v2_causal_time_delta"
    V3 = "v3_sam_prototypes"
    V4 = "v4_hicost_r_full"
    V5 = "v5_ablate_co_gnn"
    V6 = "v6_ablate_temporal"
    V7 = "v7_ablate_hct"

