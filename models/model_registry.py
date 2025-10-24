# models/model_registry.py

# from models.model_base import BaseModel
from models.tabfms.tabularfm_protocol import TabularFMProtocol
from models.tabfms.tabicl_impl import TabICLImpl
from models.tabfms.tabpfn_impl import TabPFNImpl, TabPFNRegImpl
from models.tabfms.autotabpfn_impl import AutoTabPFNImpl, AutoTabPFNRegImpl
from models.tabfms.tabstar_impl import TabSTARImpl, TabSTARRegImpl

# Map models

MODEL_REGISTRY: dict[str, type[TabularFMProtocol]] = {
    "TabICL_Classification": TabICLImpl,
    "TabPFN_Classification": TabPFNImpl,
    "TabPFN_Regression": TabPFNRegImpl,
    "AutoTabPFN_Classification": AutoTabPFNImpl,
    "AutoTabPFN_Regression": AutoTabPFNRegImpl,
    "TabSTAR_Classification": TabSTARImpl,
    "TabSTAR_Regression": TabSTARRegImpl,
}
