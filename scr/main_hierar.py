# %%
import concurrent.futures

from scr.utils.config import update_config

from .configs.defaults import CLUSTER, RUN_MODE, default_config, test_config
from .configs.hierarchy_exp_list import hierarchy_configs, hierarchy_feature_configs
from .experiments.word_feature_decode import WordFeatureDecode

print(f"running hierarchy production with cluster {CLUSTER}")
exps = []

param = {"group_by": "sequence_id"}

# create list of exp for hierarchy analysis
for h_conf in hierarchy_configs:
    for feat_name, feat_conf in hierarchy_feature_configs.items():
        full_config = update_config(default_config, h_conf)
        if RUN_MODE == "test":
            full_config = update_config(full_config, test_config)
        full_config = update_config(full_config, feat_conf)
        full_config = update_config(full_config, param)

        assert (
            full_config["is_image"] == False
        ), "is_image should be False for main hierarchy exp"

        exp = WordFeatureDecode(**full_config)

        exps.append(exp)

#############################
# feature hierarchy
#############################
if __name__ == "__main__":
    jobs = []
    max_workers = 1 if RUN_MODE == "test" else 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
        for exp in exps:
            jobs.append(exc.submit(exp.run))
        for job in concurrent.futures.as_completed(jobs):
            if job.exception() is not None:
                job.result()
                print(job.exception())


# %%
