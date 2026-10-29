# decoding functions

import numpy as np
from sklearn.dummy import DummyClassifier


def temporal_decode(data_tuple: tuple) -> np.array:
    X, y, _, train, test, model, eval_metric = data_tuple

    n_samples, n_channels, n_times = X.shape

    scores = np.zeros(n_times)

    for t in range(n_times):
        model.fit(X[train, :, t], y[train])
        y_pred = model.predict(X[test, :, t])

        scores[t] = eval_metric(y[test], y_pred)

    return scores


def temporal_decode_subscore_byindex(data_tuple) -> np.array:
    X, y, subword_idx, train, test, model, eval_metric = data_tuple
    max_idx = int(np.max(subword_idx))

    n_samples, n_channels, n_times = X.shape

    uniq_idx = np.unique(subword_idx)
    scores = np.zeros((int(max_idx) + 1, n_times))

    for t in range(n_times):
        model.fit(X[train, :, t], y[train])
        y_pred = model.predict(X[test, :, t])

        for i in range(len(uniq_idx)):
            select = (subword_idx == uniq_idx[i])[test]

            if select.sum() <= 1:  # only 1 test sample
                scores[i, t] = 0
            else:
                scores[i, t] = eval_metric(y[test][select], y_pred[select])

    return scores


def temporal_decode_subscore_byindex_classify_dummy(data_tuple) -> np.array:
    X, y, subword_idx, train, test, model, eval_metric = data_tuple
    max_idx = int(np.max(subword_idx))

    n_samples, n_channels, n_times = X.shape

    uniq_idx = np.unique(subword_idx)
    scores = np.zeros((int(max_idx) + 1, n_times))
    baselines = np.zeros((int(max_idx) + 1, n_times))

    dummy = DummyClassifier()

    for t in range(n_times):
        model.fit(X[train, :, t], y[train])
        dummy.fit(X[train, :, t], y[train])

        y_pred = model.predict(X[test, :, t])
        y_pred_dummy = dummy.predict(X[test, :, t])

        for i in range(len(uniq_idx)):
            select = (subword_idx == uniq_idx[i])[test]

            if select.sum() <= 1:  # only 1 test sample
                scores[i, t] = 0
                baselines[i, t] = 0
            else:
                scores[i, t] = eval_metric(y[test][select], y_pred[select])
                baselines[i, t] = eval_metric(y[test][select], y_pred_dummy[select])

    return scores - baselines


def sequential_decode_unmasked_df(data_tuple: tuple) -> tuple:
    X, y, _, shift, train, test, model, eval_metric, _, feature_df = data_tuple

    shifts = np.arange(-shift, shift + 1)
    n_samples, n_ch, n_t = X.shape

    scores = np.zeros((len(shifts), n_t, n_t))

    # seq_char_id, seq_subword_id
    feature_col_name_map = {
        "Word": "word",
        "Button": "seq_char",
        "Syllable": "seq_subword",
    }
    feat_type = feature_df.type.unique()[0]

    index_col_name = feature_col_name_map[feat_type]

    for t in range(n_t):
        # fit on current word
        model.fit(X[train, :, t], y[train])

        for s, shift_size in enumerate(shifts):
            if (
                shifts[s] < 0
            ):  # relative to onset, keep all words that are not the last word
                keeps = (
                    feature_df[f"{index_col_name}_id_offset"] >= np.abs(shift_size)
                ).values.astype(bool)
            elif shifts[s] > 0:  # relative to offset
                keeps = (feature_df[f"{index_col_name}_id"] >= shift_size).values.astype(
                    bool
                )
            else:
                keeps = np.ones(len(X)).astype(bool)

            goods = np.arange(len(X))[keeps]

            Y_shifted = y[feature_df[f"shifted_idx{shift_size}"].values.astype(int)]
            this_test = np.intersect1d(test, goods)

            for tt in range(n_t):
                y_pred = model.predict(X[this_test, :, tt])
                scores[s, t, tt] = eval_metric(Y_shifted[this_test], y_pred)

    return scores


def sequential_decode_unmasked_df_classify(data_tuple: tuple) -> tuple:
    X, y, _, shift, train, test, model, eval_metric, _, feature_df = data_tuple

    shifts = np.arange(-shift, shift + 1)
    n_samples, n_ch, n_t = X.shape

    scores = np.zeros((len(shifts), n_t, n_t))
    baseline = np.zeros((len(shifts), n_t, n_t))

    dummy = DummyClassifier()

    # seq_char_id, seq_subword_id
    feature_col_name_map = {
        "Word": "word",
        "Button": "seq_char",
        "Syllable": "seq_subword",
    }
    feat_type = feature_df.type.unique()[0]

    index_col_name = feature_col_name_map[feat_type]

    for t in range(n_t):
        # fit on current word
        model.fit(X[train, :, t], y[train])
        dummy.fit(X[train, :, t], y[train])

        for s, shift_size in enumerate(shifts):
            if (
                shifts[s] < 0
            ):  # relative to onset, keep all words that are not the last word
                keeps = (
                    feature_df[f"{index_col_name}_id_offset"] >= np.abs(shift_size)
                ).values.astype(bool)
            elif shifts[s] > 0:  # relative to offset
                keeps = (feature_df[f"{index_col_name}_id"] >= shift_size).values.astype(
                    bool
                )
            else:
                keeps = np.ones(len(X)).astype(bool)

            goods = np.arange(len(X))[keeps]

            Y_shifted = y[feature_df[f"shifted_idx{shift_size}"].values.astype(int)]
            this_test = np.intersect1d(test, goods)

            for tt in range(n_t):
                y_pred = model.predict(X[this_test, :, tt])
                y_pred_dummy = dummy.predict(X[this_test, :, tt])

                scores[s, t, tt] = eval_metric(Y_shifted[this_test], y_pred)
                baseline[s, t, tt] = eval_metric(Y_shifted[this_test], y_pred_dummy)

    return scores - baseline


def temporal_decode_shifted_onset_df_classify(data_tuple: tuple) -> tuple:

    X, y, onset_epoch_sel, shift, train, test, model, eval_metric, _, feature_df = (
        data_tuple
    )

    assert len(X) == len(feature_df)

    if shift < 0:  # relative to onset
        shift_size = np.flip(
            np.arange(shift, 1)
        )  # shift -1 gives the next label, hence actual index diff = 1
    else:
        shift_size = np.flip(np.arange(0, shift + 1))

    n_samples, n_ch, n_t = X.shape
    scores = np.zeros((len(shift_size), n_t))
    baseline = np.zeros((len(shift_size), n_t))

    dummy = DummyClassifier()

    # seq_char_id, seq_subword_id
    feature_col_name_map = {
        "Word": "word",
        "Button": "seq_char",
        "Syllable": "seq_subword",
    }
    feat_type = feature_df.type.unique()[0]
    index_col_name = feature_col_name_map[feat_type]

    for s, shift_s in enumerate(shift_size):

        if shift < 0:  # relative to onset, keep all words that are not the last word
            keeps = (
                feature_df[f"{index_col_name}_id_offset"] >= np.abs(shift_s)
            ).values.astype(
                bool
            )  # shift -1 gives the next label, hence actual index diff = 1
        elif shift > 0:  # relative to offset
            keeps = (feature_df[f"{index_col_name}_id"] >= shift_s).values.astype(bool)
        else:
            keeps = np.ones(len(X)).astype(bool)

        goods = np.arange(len(X))[keeps]

        Y = y[feature_df[f"shifted_idx{shift_s}"].values.astype(int)]
        this_train = np.intersect1d(train, goods)

        epoch_sel = onset_epoch_sel[test]
        test_set = test[epoch_sel]
        this_eval_set = np.intersect1d(test_set, goods)

        for t in range(n_t):
            # fit on shifted
            model.fit(X[this_train, :, t], Y[this_train])
            dummy.fit(X[this_train, :, t], Y[this_train])

            y_pred = model.predict(X[this_eval_set, :, t])
            y_pred_dummy = dummy.predict(X[this_eval_set, :, t])

            scores[s, t] = eval_metric(Y[this_eval_set], y_pred)
            baseline[s, t] = eval_metric(Y[this_eval_set], y_pred_dummy)

    return scores - baseline


