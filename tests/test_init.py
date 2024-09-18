import pandas as pd
from pandorafits import NIRDAFORMATS

def test_formats():
    assert isinstance(NIRDAFORMATS, dict)
    assert isinstance(NIRDAFORMATS[0], pd.ExcelFile)