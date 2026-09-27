from pathlib import Path
import json
from types import SimpleNamespace

class Notebook:
    def __init__(self):
        self.cells = []
        self.metadata = {}

def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result

def write(nb, path):
    path.write_text(json.dumps(dict(nbformat=4, nbformat_minor=4, cells=nb.cells, metadata=nb.metadata), indent=2))

nbf = SimpleNamespace(v4=SimpleNamespace(new_notebook=Notebook, new_markdown_cell=lambda s: cell("markdown", s), new_code_cell=lambda s: cell("code", s)), write=write)

out = Path('/Users/alliesewell/Documents/Codex/2026-09-26/oka/outputs')
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# Public transit ridership: initial exploration

Place this notebook in your repository's `notebooks/` folder and the unchanged workbook in `data/raw/`. Select your existing Python environment and run cells in order. Dependencies: pandas, matplotlib, openpyxl.

We start with **COTA motorbus, directly operated** as an editable example, not a final agency decision. UPT counts unlinked passenger trips (boardings), not unique people. Each source row represents an agency, mode and type of service. We keep these distinctions explicit.

Source: FTA NTD, **July 2026 Complete Monthly Ridership (with adjustments and estimates)_260901.xlsx**, especially `Read Me`, `UPT`, `VRM`, `VRH` and the estimate sheets. No changes are made to the source workbook.''')
code('''from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

FILENAME = "July 2026 Complete Monthly Ridership (with adjustments and estimates)_260901.xlsx"
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

# Prefer the project's raw data; the Downloads fallback lets you try it now.
DATA_PATH = ROOT / "data" / "raw" / FILENAME
if not DATA_PATH.exists():
    DATA_PATH = Path.home() / "Downloads" / FILENAME
if not DATA_PATH.is_file():
    raise FileNotFoundError("Set DATA_PATH to the downloaded workbook.")

NTD_ID = "00001"  # King County, Seattle area
MODE = "MB"       # Motorbus
TOS = "DO"        # Directly operated
print("Reading:", DATA_PATH)
''')
md('''## 1. Inspect the available series

`UPT` is the target; `VRM` and `VRH` describe service supplied. The monthly columns use labels such as `1/2002`. The workbook also has annual totals and industry summary rows; those are not individual monthly agency series.''')
code('''with pd.ExcelFile(DATA_PATH) as workbook:
    print("Sheets:", workbook.sheet_names)
    upt = pd.read_excel(workbook, sheet_name="UPT", dtype={"NTD ID": "string"})

keys = ["NTD ID", "Agency", "Mode", "TOS"]
display(upt.loc[upt["NTD ID"].eq(NTD_ID), keys])
print("UPT dimensions:", upt.shape)
''')
md('''## 2. Reshape one series and align service measures

`melt` converts the month columns to rows. We require exactly one match per sheet, parse values strictly and retain missing observations. Joining by date avoids relying on identical row positions across sheets.''')
code('''def extract_series(frame, value_name):
    selected = frame.loc[
        frame["NTD ID"].eq(NTD_ID)
        & frame["Mode"].eq(MODE)
        & frame["TOS"].eq(TOS)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one {value_name} row, found {len(selected)}. Check ID, mode and TOS.")

    months = [c for c in frame.columns if re.fullmatch(r"\d{1,2}/\d{4}", str(c))]
    if not months:
        raise ValueError("No monthly columns found.")
    result = selected.melt(value_vars=months, var_name="date", value_name=value_name)
    result["date"] = pd.to_datetime(result["date"], format="%m/%Y")
    result[value_name] = pd.to_numeric(result[value_name], errors="raise")
    if result["date"].duplicated().any():
        raise ValueError("Duplicate monthly dates found.")
    return result.set_index("date").sort_index()

monthly = extract_series(upt, "ridership")
with pd.ExcelFile(DATA_PATH) as workbook:
    for sheet, name in [("VRM", "vehicle_revenue_miles"), ("VRH", "vehicle_revenue_hours")]:
        frame = pd.read_excel(workbook, sheet_name=sheet, dtype={"NTD ID": "string"})
        monthly = monthly.join(extract_series(frame, name), how="outer", validate="one_to_one")

monthly = monthly.sort_index().asfreq("MS")
monthly.index.name = "date"
display(monthly.head())
display(monthly.tail())
''')
md('''## 3. Preserve estimate provenance and check quality

The workbook's `Read Me` explains that UPT and VRM estimates are already incorporated into their main sheets. **Do not add the estimates again.** These flags indicate a match in this release's estimate tables. A false flag does not prove a value has never been adjusted, revised or sampled.''')
code('''with pd.ExcelFile(DATA_PATH) as workbook:
    for sheet, flag in [("UPT Estimates", "upt_listed_estimate"), ("VRM Estimates", "vrm_listed_estimate")]:
        estimates = pd.read_excel(workbook, sheet_name=sheet, dtype={"NTD ID": "string"})
        selected = estimates.loc[
            estimates["NTD ID"].eq(NTD_ID)
            & estimates["Mode"].eq(MODE)
            & estimates["TOS"].eq(TOS)
        ]
        estimate_dates = pd.to_datetime(selected["Month"], format="%m/%Y")
        monthly[flag] = monthly.index.isin(estimate_dates)

measures = ["ridership", "vehicle_revenue_miles", "vehicle_revenue_hours"]
quality = pd.DataFrame({
    "missing": monthly[measures].isna().sum(),
    "zero": monthly[measures].eq(0).sum(),
    "negative": monthly[measures].lt(0).sum(),
})
print(f"{len(monthly)} months: {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}")
display(quality)
display(monthly[["upt_listed_estimate", "vrm_listed_estimate"]].sum().rename("flagged_months"))
display(monthly[measures].describe())
''')
md('''## 4. Plot the history

Look for seasonality, unusual values, the 2020 disruption and recovery. A break is not automatically a data error. Keep missing values visible instead of replacing them with zero.''')
code('''agency = upt.loc[upt["NTD ID"].eq(NTD_ID), "Agency"].iloc[0]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly.index, monthly["ridership"], linewidth=1.4)
flagged = monthly.loc[monthly["upt_listed_estimate"]]
ax.scatter(flagged.index, flagged["ridership"], color="darkorange", label="Listed estimate", zorder=3)
ax.set(title=f"{agency}: {MODE}/{TOS} monthly ridership", xlabel="Month", ylabel="Unlinked passenger trips")
ax.ticklabel_format(axis="y", style="plain")
ax.grid(alpha=0.2)
if len(flagged):
    ax.legend()
fig.tight_layout()
plt.show()
''')
md('''## 5. Explore seasonality using complete years

Compare recent complete years separately so the partial 2026 year and long-term growth do not distort a single monthly average. This is descriptive exploration; reserve a future test period before choosing model features.''')
code('''recent = monthly.assign(year=monthly.index.year, month=monthly.index.month)
counts = recent.groupby("year")["ridership"].count()
complete_years = counts[counts.eq(12)].index[-5:]
seasonality = recent.loc[recent["year"].isin(complete_years)].pivot(
    index="month", columns="year", values="ridership"
)
ax = seasonality.plot(figsize=(10, 4), marker="o")
ax.set(title="Ridership by month: latest five complete years", xlabel="Month", ylabel="Unlinked passenger trips", xticks=range(1, 13))
ax.ticklabel_format(axis="y", style="plain")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
''')
md('''## 6. Optional processed-data export

Enable this after placing the notebook in your repository. The raw workbook stays untouched. The filename records the selected series and source release.''')
code('''SAVE_PROCESSED = False
if SAVE_PROCESSED:
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ntd_{NTD_ID}_{MODE}_{TOS}_260901.csv"
    export = monthly.assign(ntd_id=NTD_ID, mode=MODE, tos=TOS, source_file=DATA_PATH.name)
    export.to_csv(output_path, index_label="date")
    print("Saved:", output_path)
''')
md('''## Next steps

1. Confirm the agency/mode/service scope and investigate missing values or changes in reporting.
2. Reserve a chronological test period and define the forecast horizon before model tuning.
3. Add a seasonal-naive baseline (same month last year), then lagged and shifted rolling features. For example, `ridership.shift(1).rolling(3).mean()` excludes the target month.
4. Compare regression and SARIMA using the same forecast horizon and information availability.
5. Join external data by month once its geographic scope and publication delay are clear.

**Forecast timing matters:** realized service miles/hours for a future month are not normally known at prediction time. Use available lags, separately supplied plans or forecasts. Even last month's NTD data may not yet be published: the source notes a reporting delay and later revisions. Historical tests with this revised release are retrospective tests, not a reconstruction of exactly what was known in real time.

Keep the 2020 disruption visible and document how the modeling period handles it. Do not interpolate across the test boundary or use future observations to fill model inputs.''')
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, out / '01_eda.ipynb')
print(out / '01_eda.ipynb')
