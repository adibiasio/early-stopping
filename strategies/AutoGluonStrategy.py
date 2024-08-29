from .FeaturePatienceStrategy import FeaturePatienceStrategy


class AutoGluonStrategy(FeaturePatienceStrategy):
    """
    Patience equation is influenced by features of the dataset currently being "trained" on.
    """

    _name = "ag_patience"
    _short_name = "AG"

    def __init__(
        self,
        metadata: dict,
        a=0.3,
        b=20,
        min_offset=None,
        max_offset=300,
        max_patience=10000,
        **kwargs,
    ):
        super().__init__(metadata, a=a, b=b, min_offset=min_offset, max_offset=max_offset, max_patience=max_patience, **kwargs)
