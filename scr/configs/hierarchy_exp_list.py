hierarchy_configs = [
    {
        "onset": True,
        "start": -1.5,
        "duration": 2.0,
    },
    {
        "onset": False,
        "start": -0.5,
        "duration": 2.0,
    },
]

hierarchy_feature_configs = {
    "SpacyEmbedding": {
        "feature_class.language": "spanish_lg",
        "feature_class.name": "SpacyEmbedding",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "temporal_decode",
    },
    "ContextEmbedding": {
        "feature_class.model_name": "gpt2",
        "feature_class.name": "ContextWordEmbeddingPinet",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "temporal_decode",
    },
    "SyllableEmbedding": {
        "feature_class.model_name": "wiki_all_minn1",
        "feature_class.name": "SyllableEmbedding",
        "pipeline.model": "RidgeCV",
        "pipeline.eval_metric": "correlate",
        "pipeline.decode_function_name": "temporal_decode_subscore_byindex",
        "subscore_func": "index",
    },
    "ButtonClass": {
        "feature_class.feature_type": "class",
        "feature_class.name": "ButtonFeat",
        "pipeline.model": "RidgeClassifierCV",
        "pipeline.eval_metric": "balanced_accuracy",
        "pipeline.decode_function_name": "temporal_decode_subscore_byindex_classify_dummy",
        "subscore_func": "index",
    },
}


