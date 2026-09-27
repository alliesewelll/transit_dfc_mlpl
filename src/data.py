"""Read one NTD agency/mode/service series without changing the workbook."""
import re
from pathlib import Path
import pandas as pd


def load_monthly(path, ntd_id="00001", mode="MB", tos="DO"):
    def select(frame):
        return frame.loc[
            frame["NTD ID"].astype("string").str.strip().eq(str(ntd_id))
            & frame["Mode"].eq(mode) & frame["TOS"].eq(tos)
        ]

    series = []
    with pd.ExcelFile(path) as book:
        for sheet, metric in [("UPT", "ridership"), ("VRM", "service_miles"), ("VRH", "service_hours")]:
            frame = pd.read_excel(book, sheet_name=sheet, dtype={"NTD ID": "string"})
            row = select(frame)
            if len(row) != 1:
                raise ValueError(f"{sheet}: expected one matching row, got {len(row)} for {ntd_id}/{mode}/{tos}.")
            agency = str(row["Agency"].iloc[0])
            months = [c for c in frame if re.fullmatch(r"\d{1,2}/\d{4}", str(c))]
            long = row.melt(value_vars=months, var_name="date", value_name=metric)
            long["date"] = pd.to_datetime(long["date"], format="%m/%Y")
            long[metric] = pd.to_numeric(long[metric], errors="raise")
            if long["date"].duplicated().any():
                raise ValueError(f"Duplicate dates in {sheet}.")
            series.append(long.set_index("date")[metric])
        monthly = pd.concat(series, axis=1).sort_index().asfreq("MS")
        for sheet, flag in [("UPT Estimates", "upt_listed_estimate"), ("VRM Estimates", "vrm_listed_estimate")]:
            rows = select(pd.read_excel(book, sheet_name=sheet, dtype={"NTD ID": "string"}))
            monthly[flag] = monthly.index.isin(pd.to_datetime(rows["Month"], format="%m/%Y"))
    monthly.index.name = "date"
    metadata = dict(agency=agency, ntd_id=str(ntd_id), mode=mode, tos=tos, source=Path(path).name)
    return monthly, metadata


def validate_target(y):
    if len(y) < 72:
        raise ValueError("Use at least 72 monthly observations for training and evaluation.")
    if not y.index.equals(pd.date_range(y.index.min(), y.index.max(), freq="MS", name=y.index.name)):
        raise ValueError("Target must have a continuous month-start index.")
    if y.isna().any() or not y.map(lambda v: float('-inf') < v < float('inf')).all() or y.lt(0).any():
        raise ValueError("Ridership contains missing, infinite or negative values. Investigate before modeling; no automatic filling is performed.")
