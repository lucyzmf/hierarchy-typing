word_seq_configs = [
    {
        "timelock_onset": True,
        "start": -1,
        "duration": 2.5,
        "onset": True,
    },
    {
        "timelock_onset": False,
        "start": -1.5,
        "duration": 2.5,
        "onset": False,
    },
]

feature_configs = {
    "ButtonClass": {
        "feature_class.feature_type": "class",
        "feature_class.name": "ButtonFeat",
        "pipeline.model": "RidgeClassifierCV",
        "pipeline.eval_metric": "balanced_accuracy",
        "pipeline.decode_function_name": "temporal_decode_shifted_onset_df_classify",
        "subscore_func": "null",
        "n_relative_pos": 4,
    },
}
