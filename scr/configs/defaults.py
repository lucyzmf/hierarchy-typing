"""Paths and compute settings, read from environment variables (see .env.example).

The PINET_* variables are machine specific; none of them enters an exca cache key, so
changing them never invalidates cached results. DEVICE and RUN_MODE select which
experiments run.

    PINET_DATA_PATH         root of the dataset (required to build events / read raw data)
    PINET_CACHE_PATH        exca cache root                          [./cache/pinet_archive]
    PINET_CLUSTER           auto | slurm | local | processpool | none   [auto]
    PINET_SLURM_PARTITION   slurm partition(s)                       [unset]
    PINET_CPUS              cpus per job                             [4]
    PINET_MEM_GB            memory per decoding job (GB)             [12]
    PINET_FEATURE_MEM_GB    memory per feature-extraction job (GB)   [32]
    PINET_FEATURE_MAX_JOBS  parallel feature-extraction jobs                [1]
    PINET_MAX_JOBS          parallel decoding jobs (one per subject) [24]
    PINET_DEVICE_MAX_JOBS   parallel preprocessing jobs (one per recording) [84]
    PINET_TIMEOUT_MIN       job timeout in minutes                   [4320]
    DEVICE                  Meg | Eeg                                [Meg]
    RUN_MODE                all | test (one recording, tiny window)  [all]
"""

import os
import pathlib

workdir = os.getcwd()

PROJECT_NAME = "pinet_archive"
DATA_PATH = os.getenv("PINET_DATA_PATH", "")
CACHE_PATH = os.getenv("PINET_CACHE_PATH", f"./cache/{PROJECT_NAME}")
_cluster = os.getenv("PINET_CLUSTER", "auto")
CLUSTER: str | None = None if _cluster.lower() in {"", "none", "null"} else _cluster
SLURM_PARTITION: str | None = os.getenv("PINET_SLURM_PARTITION") or None
NUM_CPUS = int(os.getenv("PINET_CPUS", "4"))
MEM_GB = int(os.getenv("PINET_MEM_GB", "12"))  # measured peak 8.1 GB (ButtonClass, largest subject)
FEATURE_MEM_GB = int(os.getenv("PINET_FEATURE_MEM_GB", "32"))  # otherwise slurm reserves the whole node
MAX_JOBS = int(os.getenv("PINET_MAX_JOBS", "24"))  # decoding: one job per subject
DEVICE_MAX_JOBS = int(os.getenv("PINET_DEVICE_MAX_JOBS", "84"))  # preprocessing: one job per recording
FEATURE_MAX_JOBS = int(os.getenv("PINET_FEATURE_MAX_JOBS", "1"))  # few hundred items, computed once then cached; each job loads the model (fastText: 7 GB)
TIMEOUT_MIN = int(os.getenv("PINET_TIMEOUT_MIN", str(60 * 24 * 3)))

infra_config: dict[str, dict[str, object]] = {
    "infra": {
        "cluster": CLUSTER,
        "folder": CACHE_PATH,
        "timeout_min": TIMEOUT_MIN,
        "gpus_per_node": 0,
        "cpus_per_task": NUM_CPUS,
        "job_name": PROJECT_NAME,
        "min_samples_per_job": 1,
        "max_jobs": MAX_JOBS,
        "mem_gb": MEM_GB,
    },
}
if SLURM_PARTITION:
    infra_config["infra"]["slurm_partition"] = SLURM_PARTITION

# check if cache path exists
pathlib.Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

# get device from env
DEVICE = os.getenv("DEVICE", "Meg")
RUN_MODE = os.getenv("RUN_MODE")

default_config = {
    "study": {
        "name": f"Pinet2024{DEVICE}",
        "path": DATA_PATH,
        "infra": {"folder": CACHE_PATH},
        "enhancers": [
            {"name": "RemoveTrainingTrials"},
            {"name": "CleanButtons"},
            {"name": "AddContextPinet"},
            {"name": "AddSyllableEvents"},
            {"name": "ReassignSequenceid"},
            {"name": "AddSeqOffsetWordId"},
            {"name": "AddCharIdOffset"},
            {"name": "AddSeqCharIdOnsetOffset"},
        ],
        "query": "~subject.str.contains('S23', na=False)",
    },
    "device": {
        "name": DEVICE,
        "mne_cpus": 1,
        "baseline": None,
        "filter": (0.5, 45),
        "frequency": 50.0,
        "infra": {
            "folder": CACHE_PATH,
            "cluster": CLUSTER,
            "max_jobs": DEVICE_MAX_JOBS,
            "timeout_min": 60 * 24 * 3,
            "cpus_per_task": NUM_CPUS,
            "mem_gb": 16,
            "min_samples_per_job": 1,
        },
    },
    "feature_class": {
        "name": "ContextWordEmbeddingPinet",
        "aggregation": "trigger",
        "infra": {
            "folder": CACHE_PATH,
            "cpus_per_task": NUM_CPUS,
            "cluster": CLUSTER,
            "mem_gb": FEATURE_MEM_GB,
            "max_jobs": FEATURE_MAX_JOBS,
        },
    },
    "is_image": False,
    "start": -1.0,
    "duration": 3.0,
    "pipeline": {
        "model": "RidgeCV",
        "alpha_per_target": True,
        "alphas": (-1, 10),
        "eval_metric": "correlate",
        "decode_function_name": "temporal_decode",
    },
    "n_splits": 5,
}

default_config.update(infra_config)

# the nested per-recording device infra does not inherit slurm_partition
if CLUSTER == "slurm" and SLURM_PARTITION:
    default_config["device"]["infra"]["slurm_partition"] = SLURM_PARTITION

test_config = {
    "start": 0.0,
    "duration": 0.1,
    "infra.cluster": None,
    "device.infra.cluster": None,
    "feature_class.infra.cluster": None,
    "device.mne_cpus": 1,
    "study.query": "timeline_index==0",
    "num_subj": 1,
}
