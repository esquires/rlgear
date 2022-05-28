import rlgear.utils

try:
    import ray  # NOQA
except ImportError as e:
    import warnings
    warnings.warn((
        "could not import ray, skipping rlgear.models, rllib_utils, "
        "and models"))
    print(e)
else:
    # pylint: disable=ungrouped-imports
    import rlgear.agents
    import rlgear.rllib_utils  # NOQA

    try:
        import torch  # NOQA
        import rlgear.models  # NOQA
    except ImportError as e:
        import warnings
        warnings.warn("could not import torch, skipping rlgear.models")
        print(e)
