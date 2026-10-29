import copy


def update_config(config: dict, update: dict) -> dict:
    """
    This enables to modify a nested dictionary, either by updating an individual entry, or by overwriting full sub-dictionaries.
    Note that the behavior is different from that of ConfDict.update, which only updates individual entries (see examples below).

    Copied verbatim from neuraltrain.utils.update_config (the only neuraltrain function this project uses).

    Parameters
    ----------
    config : dict
        The dictionary to update.
    update : dict
        The dictionary containing the updates.

    Returns
    -------
    dict: The updated dictionary.


    Example
    -------
        config  = {"a": 1, "b": {"c": 2, "d": 3}}
        update1 = {"b.c": 4}
        update2 = {"b": {"e": 5}}
        With update_config, the following holds:
        update_config(config, update1) -> {"a": 1, "b": {"c": 4, "d": 3}}
        update_config(config, update2) -> {"a": 1, "b": {"e": 5}}
        With ConfDict.update, the following holds:
        ConfDict(config).update(update2) -> {"a": 1, "b": {"c": 2, "d": 3, "e": 5}}
    """
    new_config = copy.deepcopy(config)
    for k, v in update.items():
        path = k.split(".")
        c = new_config
        for p in path[:-1]:
            c = c[p]
        c[path[-1]] = v
    return new_config
