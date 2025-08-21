import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("../data/raw_data/Nationally Determined Contributions")

# Load the NDC data

ndcs = pd.read_excel("CW_NDC_tracker_visual.xlsx", 
                     sheet_name="Country",
                     header=2)


ndcs = ndcs[['Country', 'ISO', 'Unconditional.3', 'Conditional.3']]
ndcs.rename(columns={'Unconditional.3': 'Unconditional',
                     'Conditional.3': 'Conditional'}, inplace=True)


# Historical emissions data
historical_emissions = pd.read_csv("CW_HistoricalEmissions_ClimateWatch.csv")
historical_emissions = historical_emissions[(historical_emissions["Sector"] == "Total including LUCF") & 
                      (historical_emissions["Gas"] == "All GHG")]
emissions_10 = historical_emissions[["Country", "2020"]]

ndc_ref = ndcs.merge(emissions_10, left_on="ISO", right_on="Country", how="left")
ndc_ref.rename(columns={"Country_x": "Country"}, inplace=True)
ndc_ref.drop(columns=["Country_y"], inplace=True)
ndc_ref["Unconditional_b20"] = ndc_ref["Unconditional"] / ndc_ref["2020"] - 1
ndc_ref["Conditional_b20"] = ndc_ref["Conditional"] / ndc_ref["2020"] - 1 

ndc_ref.to_csv("../../processed_data/ndc_reference.csv", index=False)

## Brief Analysis
bins = np.arange(-1, 1.1, 0.1)
ndc_ref["bin_uncd"] = pd.cut(ndc_ref["Unconditional_b20"], bins, right=True, include_lowest=True)
ndc_ref["bin_cond"] = pd.cut(ndc_ref["Conditional_b20"], bins, right=True, include_lowest=True)

uncd_grp = ndc_ref.groupby("bin_uncd")\
      .agg(Countries=("Country", list), Count=("Country", "size"))\
      .reset_index()

cond_grp = ndc_ref.groupby("bin_cond")\
      .agg(Countries=("Country", list), Count=("Country", "size"))\
      .reset_index()

# Plots

plt.figure(figsize=(10, 5))
plt.bar(uncd_grp["bin_uncd"].astype(str), uncd_grp["Count"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("bin_uncd range")
plt.ylabel("Number of Countries")
plt.title("Countries in each bin of Unconditional NDCs relative to 2020 emissions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(cond_grp["bin_cond"].astype(str), cond_grp["Count"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("bin_uncd range")
plt.ylabel("Number of Countries")
plt.title("Countries in each bin of Conditional NDCs relative to 2020 emissions")
plt.tight_layout()
plt.show()