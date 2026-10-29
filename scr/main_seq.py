# %%

import concurrent.futures

from scr.configs.defaults import RUN_MODE, default_config, test_config
from scr.configs.seq_exp_list import seq_config, seq_feature_configs
from scr.experiments.sequential_features import SequentialDecoding

from scr.utils.config import update_config

seq_exp = []

param = {"group_by": "sequence_id"}

# create list of experiments
for feat_name, feat_conf in seq_feature_configs.items():
    full_config = update_config(default_config, seq_config)
    if RUN_MODE == "test":
        full_config = update_config(full_config, test_config)
    full_config = update_config(full_config, feat_conf)
    full_config = update_config(full_config, param)

    exp = SequentialDecoding(**full_config)

    seq_exp.append(exp)

#############################
# run
#############################
if __name__ == "__main__":
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exc:
        for exp in seq_exp:
            jobs.append(exc.submit(exp.run))
        for job in concurrent.futures.as_completed(jobs):
            if job.exception() is not None:
                job.result()
                print(job.exception())


# %%
