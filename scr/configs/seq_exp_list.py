seq_config = {"start": -1.5, "duration": 3.0, "is_image": False, "relative_to_seq": False}

seq_feature_configs = {
    "ContextEmbedding": {
        "feature_class.model_name": "gpt2",
        "feature_class.name": "ContextWordEmbeddingPinet",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "sequential_decode_unmasked_df",
        "shifts": 1,
        "cv": "GroupKFold",
    },
    "SpacyEmbedding": {
        "feature_class.language": "spanish_lg",
        "feature_class.name": "SpacyEmbedding",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "sequential_decode_unmasked_df",
        "shifts": 1,
        "cv": "GroupKFold",
    },
    "SyllableEmbedding": {
        "feature_class.model_name": "wiki_all_minn1",
        "feature_class.name": "SyllableEmbedding",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "sequential_decode_unmasked_df",
        "shifts": 2,
        "cv": "GroupKFold",
    },
    "ButtonClass": {
        "feature_class.feature_type": "class",
        "feature_class.name": "ButtonFeat",
        "pipeline.model": "RidgeClassifierCV",
        "pipeline.eval_metric": "balanced_accuracy",
        "pipeline.decode_function_name": "sequential_decode_unmasked_df_classify",
        "shifts": 4,
        "cv": "GroupKFold",
    },
}
