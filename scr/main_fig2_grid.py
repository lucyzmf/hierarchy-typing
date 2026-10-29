# %%

import concurrent.futures

from scr.utils.config import update_config
from scr.configs.defaults import CLUSTER, RUN_MODE, default_config, test_config
from scr.configs.word_feat_seq_exp_list import feature_configs, word_seq_configs
from scr.experiments.word_feature_decode_seq_onset import WordFeatureDecodeSeqOnset

params = {
    "sent_onset": False,
}


exps = []

# create list of experiments
for seq_config in word_seq_configs:
    for feat_name, feat_conf in feature_configs.items():
        full_config = update_config(default_config, seq_config)
        full_config = update_config(full_config, feat_conf)
        full_config = update_config(full_config, params)
        if RUN_MODE == "test":
            full_config = update_config(full_config, test_config)

        exp = WordFeatureDecodeSeqOnset(**full_config)

        exps.append(exp)


#############################
# run
#############################
if __name__ == "__main__":
    print(f"running word feat onset with cluster {CLUSTER}")

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exc:
        for exp in exps:
            jobs.append(exc.submit(exp.run))
        for job in concurrent.futures.as_completed(jobs):
            if job.exception() is not None:
                job.result()
                print(job.exception())


# %%
