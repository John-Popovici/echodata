# models/model_registry.py

# from models.model_base import BaseModel
from models.tabfms.tabularfm_protocol import TabularFMProtocol
from models.tabfms.tabicl_impl import TabICLImpl
from models.tabfms.tabpfn_impl import TabPFNImpl, TabPFNRegImpl
from models.tabfms.autotabpfn_impl import AutoTabPFNImpl, AutoTabPFNRegImpl
from models.tabfms.tabstar_impl import TabSTARImpl, TabSTARRegImpl
from models.tabfms.mitra_impl import MitraImpl, MitraRegImpl

# Map models
MODEL_REGISTRY: dict[str, dict[str, str | type[TabularFMProtocol]]] = {
    "TabICL_Classification": {
        "model": TabICLImpl,
        "model_name": "TabICL",
        "model_type": "Classification",
    },
    "TabPFN_Classification": {
        "model": TabPFNImpl,
        "model_name": "TabPFN",
        "model_type": "Classification",
    },
    "TabPFN_Regression": {
        "model": TabPFNRegImpl,
        "model_name": "TabPFN",
        "model_type": "Regression",
    },
    "AutoTabPFN_Classification": {
        "model": AutoTabPFNImpl,
        "model_name": "AutoTabPFN",
        "model_type": "Classification",
    },
    "AutoTabPFN_Regression": {
        "model": AutoTabPFNRegImpl,
        "model_name": "AutoTabPFN",
        "model_type": "Regression",
    },
    "TabSTAR_Classification": {
        "model": TabSTARImpl,
        "model_name": "TabSTAR",
        "model_type": "Classification",
    },
    "TabSTAR_Regression": {
        "model": TabSTARRegImpl,
        "model_name": "TabSTAR",
        "model_type": "Regression",
    },
    "Mitra_Classification": {
        "model": MitraImpl,
        "model_name": "Mitra",
        "model_type": "Classification",
    },
    "Mitra_Regression": {
        "model": MitraRegImpl,
        "model_name": "Mitra",
        "model_type": "Regression",
    },
}
