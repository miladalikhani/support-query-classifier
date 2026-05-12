import importlib

PACKAGES = [
    "src.data",
    "src.evaluation",
    "src.labeling",
    "src.pii",
    "src.pipelines",
    "src.serving",
    "src.training",
]


def test_packages_import() -> None:
    for name in PACKAGES:
        importlib.import_module(name)
