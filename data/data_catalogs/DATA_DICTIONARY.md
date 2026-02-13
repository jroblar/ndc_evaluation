# Data Dictionary

This document describes every variable available in the master panel (`data/processed/panel_country_year.parquet`) and the five filtered balanced panels. Variables are organized by source database.

In the master panel, column names are prefixed by their MFA dimension:
- `cap_` = Capabilities (from `out/capabilities/`)
- `con_` = Constraints (from `out/constraints/`)
- `inc_` = Incentives (from `out/incentives/`)

The **source name** column shows the original column name before prefixing.

Columns marked with **A**, **B**, **C**, **D**, or **E** appear in the corresponding filtered panel:
- **A** = `panel_1965_2022_50countries_32vars` (50 countries, 58 years, 32 vars)
- **B** = `panel_1992_2021_140countries_30vars` (140 countries, 30 years, 30 vars)
- **C** = `panel_2000_2022_136countries_52vars` (136 countries, 23 years, 52 vars)
- **D** = `panel_2000_2022_136countries_62vars` (136 countries, 23 years, 62 vars) — Panel C + CCLW + Climate Policy DB (zero-filled)
- **E** = `panel_2002_2022_136countries_68vars` (136 countries, 21 years, 68 vars) — Panel D + WGI governance

Panels D and E use **zero-filling** for CCLW and Climate Policy DB variables: a country-year with no record in those sources is treated as 0 (no law/policy adopted), which is semantically correct for count and binary indicator variables.

---

## 1. OWID CO2 & Greenhouse Gas Emissions

**Source:** Our World in Data, compiled from Global Carbon Project, CAIT, and EPA.
**Coverage:** 218 countries, 1750–2024.
**What it measures:** Fossil-fuel and land-use CO2, methane, nitrous oxide, and aggregate GHG emissions by country-year. The primary source for historical emissions trajectories.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_co2_mt` | `co2_mt` | CO2 emissions | Total CO2 emissions excluding land use change | Mt | A B C D E |
| `con_co2_per_capita` | `co2_per_capita` | CO2 per capita | CO2 emissions per person | t/person | A B C D E |
| `con_total_ghg_mt` | `total_ghg_mt` | Total GHG emissions | All greenhouse gases incl. CH4, N2O (AR5 GWP100) | Mt CO2eq | A |
| `con_ghg_per_capita` | `ghg_per_capita` | GHG per capita | Total GHG emissions per person | t CO2eq/person | A B C D E |
| `con_methane_mt` | `methane_mt` | Methane emissions | CH4 emissions | Mt CO2eq | A B C D E |
| `con_nitrous_oxide_mt` | `nitrous_oxide_mt` | Nitrous oxide emissions | N2O emissions | Mt CO2eq | A B C D E |
| `con_co2_incl_luc_mt` | `co2_incl_luc_mt` | CO2 incl. land use change | CO2 emissions including land use change and forestry | Mt | A B C D E |
| `con_share_global_co2_pct` | `share_global_co2_pct` | Share of global CO2 | Country share of global CO2 emissions | % | A B C D E |
| `con_population` | `population` | Population (OWID) | Country population from OWID CO2 dataset | persons | A B C D E |
| `con_gdp` | `gdp` | GDP (OWID) | Gross domestic product | int. $ | A |

---

## 2. OWID Energy

**Source:** Our World in Data, compiled from BP Statistical Review, EIA, Ember, and IRENA.
**Coverage:** 219 countries, 1965–2024.
**What it measures:** Primary energy consumption, electricity generation, and energy mix shares by fuel type. The broadest energy dataset with consistent long-run series.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_fossil_share_energy_pct` | `fossil_share_energy_pct` | Fossil share of energy | Fossil fuels as share of primary energy | % | A |
| `con_coal_share_energy_pct` | `coal_share_energy_pct` | Coal share of energy | Coal as share of primary energy | % | A |
| `con_oil_share_energy_pct` | `oil_share_energy_pct` | Oil share of energy | Oil as share of primary energy | % | A |
| `con_gas_share_energy_pct` | `gas_share_energy_pct` | Gas share of energy | Natural gas as share of primary energy | % | A |
| `con_renewables_share_energy_pct` | `renewables_share_energy_pct` | Renewables share of energy | Renewables as share of primary energy | % | A |
| `con_nuclear_share_energy_pct` | `nuclear_share_energy_pct` | Nuclear share of energy | Nuclear as share of primary energy | % | A |
| `con_low_carbon_share_energy_pct` | `low_carbon_share_energy_pct` | Low-carbon share of energy | Low-carbon sources as share of primary energy | % | A |
| `con_solar_share_energy_pct` | `solar_share_energy_pct` | Solar share of energy | Solar as share of primary energy | % | A |
| `con_wind_share_energy_pct` | `wind_share_energy_pct` | Wind share of energy | Wind as share of primary energy | % | A |
| `con_hydro_share_energy_pct` | `hydro_share_energy_pct` | Hydro share of energy | Hydro as share of primary energy | % | A |
| `con_primary_energy_twh` | `primary_energy_twh` | Primary energy consumption | Total primary energy consumption | TWh | A B C D E |
| `con_energy_per_capita_kwh` | `energy_per_capita_kwh` | Energy per capita | Primary energy consumption per person | kWh/person | A B C D E |
| `con_fossil_share_elec_pct` | `fossil_share_elec_pct` | Fossil share of electricity | Fossil fuels as share of electricity generation | % | C D E |
| `con_renewables_share_elec_pct` | `renewables_share_elec_pct` | Renewables share of electricity | Renewables as share of electricity generation | % | C D E |
| `con_electricity_generation_twh` | `electricity_generation_twh` | Electricity generation | Total electricity generation | TWh | C D E |
| `con_energy_intensity` | `energy_intensity` | Energy intensity | Energy consumed per unit of GDP | kWh/$ | A |
| `con_net_elec_imports_twh` | `net_elec_imports_twh` | Net electricity imports | Net electricity imports (negative = exports) | TWh | C D E |

---

## 3. World Bank — World Development Indicators (WDI)

**Source:** World Bank API, series from national accounts, household surveys, and administrative data.
**Coverage:** 221 countries, 1960–2024.
**What it measures:** Broad socioeconomic indicators spanning income, poverty, education, trade, land use, and infrastructure access. The most widely used cross-country development dataset.

### Capabilities split (`out/capabilities/wdi.parquet` → `cap_`)

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `cap_gdp_per_capita_ppp_current` | `gdp_per_capita_ppp_current` | GDP per capita PPP (current) | GDP per capita at PPP, current international dollars | current int. $ | B C D E |
| `cap_gdp_per_capita_constant_2015` | `gdp_per_capita_constant_2015` | GDP per capita (constant 2015) | GDP per capita in constant 2015 US dollars | constant 2015 USD | B C D E |
| `cap_gdp_current_usd` | `gdp_current_usd` | GDP (current USD) | Gross domestic product, current US dollars | current USD | A B C D E |
| `cap_literacy_rate` | `literacy_rate` | Literacy rate | Adult literacy rate (15+ years) | % | — |
| `cap_tertiary_enrollment` | `tertiary_enrollment` | Tertiary enrollment | Gross enrollment ratio, tertiary education | % | — |
| `cap_education_expenditure_pct_gdp` | `education_expenditure_pct_gdp` | Education expenditure | Government expenditure on education as share of GDP | % of GDP | — |
| `cap_rd_spending_pct_gdp` | `rd_spending_pct_gdp` | R&D spending | Research and development expenditure as share of GDP | % of GDP | — |
| `cap_trade_openness_pct_gdp` | `trade_openness_pct_gdp` | Trade openness | Trade (imports + exports) as share of GDP | % of GDP | — |
| `cap_fdi_net_inflows_pct_gdp` | `fdi_net_inflows_pct_gdp` | FDI net inflows | Foreign direct investment, net inflows as share of GDP | % of GDP | B C D E |
| `cap_population` | `population` | Population (WDI) | Total population | persons | A B C D E |
| `cap_government_debt_pct_gdp` | `government_debt_pct_gdp` | Government debt | Central government debt as share of GDP | % of GDP | — |

### Constraints split (`out/constraints/wdi.parquet` → `con_`)

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_poverty_headcount_215` | `poverty_headcount_215` | Poverty headcount ($2.15/day) | Share of population below $2.15/day poverty line | % | — |
| `con_poverty_headcount_365` | `poverty_headcount_365` | Poverty headcount ($3.65/day) | Share of population below $3.65/day poverty line | % | — |
| `con_poverty_headcount_685` | `poverty_headcount_685` | Poverty headcount ($6.85/day) | Share of population below $6.85/day poverty line | % | — |
| `con_agricultural_land_pct` | `agricultural_land_pct` | Agricultural land | Agricultural land as share of total land area | % | A B C D E |
| `con_forest_area_pct` | `forest_area_pct` | Forest area | Forest area as share of total land area | % | B C D E |
| `con_fuel_exports_pct` | `fuel_exports_pct` | Fuel exports | Fuel exports as share of merchandise exports | % | — |
| `con_resource_rents_pct_gdp` | `resource_rents_pct_gdp` | Resource rents | Total natural resource rents as share of GDP | % of GDP | — |
| `con_renewable_energy_consumption_pct` | `renewable_energy_consumption_pct` | Renewable energy consumption | Renewable energy as share of total final energy consumption | % | B |
| `con_electricity_access_pct` | `electricity_access_pct` | Electricity access | Share of population with access to electricity | % | C D E |

---

## 4. Ember — Global Electricity Review

**Source:** Ember Climate, compiled from national statistics offices, ENTSO-E, EIA, and IEA.
**Coverage:** 215 countries, 2000–2025.
**What it measures:** Detailed electricity generation and demand by fuel type (coal, gas, nuclear, hydro, solar, wind, bioenergy), plus CO2 intensity of the power sector. The most granular electricity-specific dataset.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_bioenergy` | `bioenergy` | Bioenergy generation | Electricity generated from bioenergy | TWh | C D E |
| `con_co2_intensity` | `co2_intensity` | CO2 intensity of electricity | Carbon intensity of electricity generation | gCO2/kWh | C D E |
| `con_coal` | `coal` | Coal generation | Electricity generated from coal | TWh | C D E |
| `con_demand` | `demand` | Electricity demand | Total electricity demand | TWh | C D E |
| `con_demand_per_capita` | `demand_per_capita` | Electricity demand per capita | Electricity demand per person | MWh/person | C D E |
| `con_gas` | `gas` | Gas generation | Electricity generated from gas | TWh | C D E |
| `con_gas_and_other_fossil` | `gas_and_other_fossil` | Gas & other fossil generation | Electricity from gas and other fossil sources | TWh | C D E |
| `con_hydro` | `hydro` | Hydro generation | Electricity generated from hydropower | TWh | C D E |
| `con_hydro,_bioenergy_and_other_renewables` | `hydro,_bioenergy_and_other_renewables` | Hydro+bio+other renewables | Electricity from hydro, bioenergy, and other renewables | TWh | C D E |
| `con_nuclear` | `nuclear` | Nuclear generation | Electricity generated from nuclear | TWh | C D E |
| `con_other_fossil` | `other_fossil` | Other fossil generation | Electricity from other fossil sources | TWh | C D E |
| `con_other_renewables` | `other_renewables` | Other renewables generation | Electricity from other renewable sources | TWh | — |
| `con_solar` | `solar` | Solar generation | Electricity generated from solar | TWh | C D E |
| `con_total_generation` | `total_generation` | Total generation | Total electricity generated | TWh | C D E |
| `con_total_emissions` | `total_emissions` | Power sector emissions | Total emissions from electricity generation | MtCO2 | C D E |
| `con_wind` | `wind` | Wind generation | Electricity generated from wind | TWh | — |
| `con_wind_and_solar` | `wind_and_solar` | Wind + solar generation | Electricity from wind and solar combined | TWh | C D E |

---

## 5. Penn World Table (PWT 10.01)

**Source:** University of Groningen, Feenstra, Inklaar & Timmer (2015).
**Coverage:** 183 countries, 1950–2019.
**What it measures:** PPP-adjusted GDP, capital stocks, labor inputs, total factor productivity, and price levels. The standard dataset for cross-country macroeconomic comparisons at purchasing power parity. Excluded from filtered panels because it ends in 2019.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `cap_rgdp_expenditure_ppp` | `rgdp_expenditure_ppp` | Real GDP (expenditure, PPP) | Real GDP at chained PPPs, expenditure-side | M 2017 USD | — |
| `cap_rgdp_output_ppp` | `rgdp_output_ppp` | Real GDP (output, PPP) | Real GDP at chained PPPs, output-side | M 2017 USD | — |
| `cap_cgdp_expenditure_current` | `cgdp_expenditure_current` | Current GDP (expenditure, PPP) | Current-price GDP at PPPs, expenditure-side | M current USD | — |
| `cap_cgdp_output_current` | `cgdp_output_current` | Current GDP (output, PPP) | Current-price GDP at PPPs, output-side | M current USD | — |
| `cap_population_millions` | `population_millions` | Population (PWT) | Country population | millions | — |
| `cap_human_capital_index` | `human_capital_index` | Human capital index | Index based on years of schooling and returns to education | index | — |
| `cap_labor_share_income` | `labor_share_income` | Labor share of income | Share of labor compensation in GDP | share (0–1) | — |
| `cap_tfp_welfare` | `tfp_welfare` | TFP (welfare) | Total factor productivity at current PPPs | index (USA=1) | — |
| `cap_tfp_national_accounts` | `tfp_national_accounts` | TFP (national accounts) | Total factor productivity at constant national prices | index (2017=1) | — |
| `cap_capital_stock_current` | `capital_stock_current` | Capital stock (current) | Capital stock at current PPPs | M current USD | — |
| `cap_capital_stock_constant` | `capital_stock_constant` | Capital stock (constant) | Capital stock at constant national prices | M 2017 nat. currency | — |
| `cap_employment_millions` | `employment_millions` | Employment | Number of persons engaged | millions | — |
| `cap_avg_hours_worked` | `avg_hours_worked` | Average hours worked | Average annual hours worked per engaged person | hours/year | — |
| `cap_price_level_gdp` | `price_level_gdp` | Price level of GDP | Price level of output-side GDP | index (USA=1) | — |
| `cap_price_level_consumption` | `price_level_consumption` | Price level of consumption | Price level of household consumption | index (USA=1) | — |
| `cap_price_level_investment` | `price_level_investment` | Price level of investment | Price level of capital formation | index (USA=1) | — |
| `cap_gdp_per_capita_ppp` | `gdp_per_capita_ppp` | GDP per capita PPP (PWT) | Real GDP per capita at PPP (derived: rgdpe/pop) | k 2017 USD/person | — |
| `cap_employment_rate` | `employment_rate` | Employment rate | Employment as share of population (derived: emp/pop) | ratio | — |
| `cap_labor_productivity` | `labor_productivity` | Labor productivity | Real GDP per worker (derived: rgdpe/emp) | M 2017 USD/M workers | — |

---

## 6. SWIID — Standardized World Income Inequality Database

**Source:** Frederick Solt, compiled and standardized from Luxembourg Income Study, OECD, Eurostat, and national surveys.
**Coverage:** 191 countries, 1960–2024 (sparse: ~15.5% coverage).
**What it measures:** Comparable Gini coefficients for market income (pre-tax) and disposable income (post-tax), plus the redistributive effect of taxes and transfers. Excluded from filtered panels due to sparse coverage.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_gini_disposable` | `gini_disposable` | Gini (disposable income) | Gini coefficient for post-tax, post-transfer income | index (0–100) | — |
| `con_gini_disposable_se` | `gini_disposable_se` | Gini disposable (std. error) | Standard error of disposable income Gini estimate | index points | — |
| `con_gini_market` | `gini_market` | Gini (market income) | Gini coefficient for pre-tax, pre-transfer income | index (0–100) | — |
| `con_gini_market_se` | `gini_market_se` | Gini market (std. error) | Standard error of market income Gini estimate | index points | — |
| `con_redistribution` | `redistribution` | Redistribution | Absolute Gini reduction through taxes and transfers | index points | — |
| `con_redistribution_pct` | `redistribution_pct` | Redistribution (%) | Percentage Gini reduction through taxes and transfers | % | — |

---

## 7. EDGAR v8.0 — Emissions Database for Global Atmospheric Research

**Source:** European Commission Joint Research Centre (JRC).
**Coverage:** 221 countries, 1970–2022.
**What it measures:** GHG emissions by country based on activity data and emission factors, independent of national self-reported inventories. Provides an alternative emissions estimate to cross-validate OWID/GCP data.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_edgar_ghg_mt` | `edgar_ghg_mt` | EDGAR GHG (Mt) | Total GHG emissions, all substances and sectors summed | Mt CO2eq (AR5) | B C D E |
| `con_edgar_ghg_gg` | `edgar_ghg_gg` | EDGAR GHG (Gg) | Total GHG emissions, all substances and sectors summed | Gg CO2eq (AR5) | B C D E |

---

## 8. V-Dem v15 — Varieties of Democracy

**Source:** V-Dem Institute, University of Gothenburg. Expert-coded indices from ~3,700 country experts.
**Coverage:** 202 countries, 1789–2024.
**What it measures:** Democracy along multiple conceptual dimensions — electoral, liberal, participatory, deliberative, egalitarian — plus governance quality indicators (corruption, rule of law, civil liberties, accountability). The most comprehensive democracy measurement project.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `cap_electoral_democracy_idx` | `electoral_democracy_idx` | Electoral democracy index | Free/fair elections, suffrage, elected officials (v2x_polyarchy) | index (0–1) | A B C D E |
| `cap_liberal_democracy_idx` | `liberal_democracy_idx` | Liberal democracy index | Electoral democracy + rule of law, judicial and legislative constraints | index (0–1) | A B C D E |
| `cap_participatory_democracy_idx` | `participatory_democracy_idx` | Participatory democracy index | Active citizen participation beyond elections | index (0–1) | — |
| `cap_deliberative_democracy_idx` | `deliberative_democracy_idx` | Deliberative democracy index | Decisions through reasoned public deliberation | index (0–1) | — |
| `cap_egalitarian_democracy_idx` | `egalitarian_democracy_idx` | Egalitarian democracy index | Equal political rights across social groups | index (0–1) | — |
| `cap_corruption_idx` | `corruption_idx` | Corruption index | Degree of political corruption (higher = more corrupt) | index (0–1) | A B C D E |
| `cap_rule_of_law_idx` | `rule_of_law_idx` | Rule of law index | Transparent, enforced, fairly adjudicated laws | index (0–1) | A B C D E |
| `cap_civil_liberties_idx` | `civil_liberties_idx` | Civil liberties index | Freedom from government interference in private life | index (0–1) | A B C D E |
| `cap_freedom_expression_idx` | `freedom_expression_idx` | Freedom of expression index | Media freedom, academic freedom, freedom of discussion | index (0–1) | — |
| `cap_civil_society_participation_idx` | `civil_society_participation_idx` | Civil society participation index | Extent of civil society engagement | index (0–1) | — |
| `cap_gender_equality_idx` | `gender_equality_idx` | Gender equality index | Women's political empowerment and civil liberties | index (0–1) | — |
| `cap_accountability_idx` | `accountability_idx` | Accountability index | Vertical, horizontal, and diagonal accountability | index (0–1) | A B C D E |

---

## 9. EM-DAT — International Disaster Database

**Source:** Centre for Research on the Epidemiology of Disasters (CRED), Université catholique de Louvain.
**Coverage:** 231 countries, 1900–2025 (sporadic).
**What it measures:** Natural and technological disaster events — deaths, affected population, economic damages — plus event counts by disaster type. Excluded from filtered panels because disaster occurrence is inherently sporadic (many country-years have zero events recorded as NaN).

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_n_disaster_events` | `n_disaster_events` | Disaster events | Total natural disaster events recorded | count | — |
| `con_disaster_deaths` | `disaster_deaths` | Disaster deaths | Total deaths from natural disasters | persons | — |
| `con_disaster_affected` | `disaster_affected` | Disaster affected | Total people affected by natural disasters | persons | — |
| `con_disaster_damage_000usd` | `disaster_damage_000usd` | Disaster damage | Total economic damage from disasters | k USD | — |
| `con_disaster_damage_adj_000usd` | `disaster_damage_adj_000usd` | Disaster damage (adjusted) | Total economic damage, inflation-adjusted | k USD | — |
| `con_n_flood` | `n_flood` | Flood events | Number of flood events | count | — |
| `con_n_storm` | `n_storm` | Storm events | Number of storm events | count | — |
| `con_n_earthquake` | `n_earthquake` | Earthquake events | Number of earthquake events | count | — |
| `con_n_drought` | `n_drought` | Drought events | Number of drought events | count | — |
| `con_n_wildfire` | `n_wildfire` | Wildfire events | Number of wildfire events | count | — |
| `con_n_epidemic` | `n_epidemic` | Epidemic events | Number of epidemic events | count | — |
| `con_n_extreme_temperature` | `n_extreme_temperature` | Extreme temperature events | Number of extreme temperature events | count | — |

---

## 10. World Bank — Worldwide Governance Indicators (WGI)

**Source:** World Bank, compiled from 30+ data sources (expert assessments, surveys, NGO reports).
**Coverage:** 214 countries, 1996–2023 (with year gaps: missing 1997, 1999, 2001).
**What it measures:** Six aggregate governance dimensions. Scored on a standardized scale where 0 is the global mean and values range roughly from -2.5 (weak) to +2.5 (strong). Year gaps in 1997/1999/2001 prevent inclusion in panels starting before 2002. Included in Panel E (2002–2022).

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `cap_control_corruption` | `control_corruption` | Control of corruption | Perceptions of public power exercised for private gain | score (~-2.5 to 2.5) | E |
| `cap_govt_effectiveness` | `govt_effectiveness` | Government effectiveness | Quality of public services, civil service, policy formulation | score (~-2.5 to 2.5) | E |
| `cap_political_stability` | `political_stability` | Political stability | Likelihood of political instability or politically-motivated violence | score (~-2.5 to 2.5) | E |
| `cap_regulatory_quality` | `regulatory_quality` | Regulatory quality | Ability to formulate sound policies for private sector development | score (~-2.5 to 2.5) | E |
| `cap_rule_of_law` | `rule_of_law` | Rule of law (WGI) | Confidence in rules of society, contract enforcement, property rights | score (~-2.5 to 2.5) | E |
| `cap_voice_accountability` | `voice_accountability` | Voice and accountability | Citizen participation in government selection, freedom of expression | score (~-2.5 to 2.5) | E |

---

## 11. ND-GAIN — Notre Dame Global Adaptation Initiative

**Source:** Notre Dame Global Adaptation Initiative, University of Notre Dame.
**Coverage:** 192 countries, 1995–2023.
**What it measures:** A country's vulnerability to climate change (exposure, sensitivity, adaptive capacity) and its readiness to leverage investments for adaptation (economic, governance, social). Higher score = better positioned to adapt.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `con_ndgain_score` | `ndgain_score` | ND-GAIN score | Overall adaptation readiness minus vulnerability | index (0–100) | C D E |
| `con_ndgain_vulnerability` | `ndgain_vulnerability` | ND-GAIN vulnerability | Exposure, sensitivity, adaptive capacity to climate hazards | index (0–1) | C D E |
| `con_ndgain_readiness` | `ndgain_readiness` | ND-GAIN readiness | Economic, governance, social readiness to leverage investment | index (0–1) | C D E |

---

## 12. CCLW — Climate Change Laws of the World

**Source:** Grantham Research Institute on Climate Change (LSE) and Sabin Center for Climate Change Law (Columbia).
**Coverage:** 203 countries, 1990–2026 (event-based, sporadic).
**What it measures:** Climate-related legislation, executive policies, and UNFCCC submissions. Tracks cumulative legal frameworks and whether a country has adopted a framework climate law. Three key variables are included in Panels D and E via zero-filling (NaN → 0, meaning no law adopted that year).

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `inc_n_climate_laws` | `n_climate_laws` | Climate laws/documents (year) | Climate-related laws and policies adopted in that year | count | D E (zero-filled) |
| `inc_n_climate_legislative` | `n_climate_legislative` | Legislative climate documents | Legislative-origin climate documents | count | — |
| `inc_n_climate_executive` | `n_climate_executive` | Executive climate documents | Executive-origin climate documents | count | — |
| `inc_n_climate_unfccc` | `n_climate_unfccc` | UNFCCC climate documents | UNFCCC-related climate documents | count | — |
| `inc_n_laws` | `n_laws` | Formal laws (year) | Number of formal laws adopted in that year | count | — |
| `inc_n_policies` | `n_policies` | Policies/plans (year) | Number of policies, plans, or strategies adopted | count | — |
| `inc_cumulative_climate_laws` | `cumulative_climate_laws` | Cumulative climate laws | Running total of climate laws/documents ever adopted | count (cumul.) | D E (zero-filled) |
| `inc_has_framework_law` | `has_framework_law` | Has framework law | Whether country has ever adopted a climate framework law | binary (0/1) | D E (zero-filled) |

---

## 13. DESTA — Design of Trade Agreements

**Source:** DESTA project (Dür, Baccini & Elsig), University of Bern.
**Coverage:** 45 countries, 1948–2023 (limited country coverage due to COW code mapping).
**What it measures:** International trade agreements and their institutional depth (dispute settlement, tariff provisions). Captures a country's economic integration and international cooperation. Excluded from filtered panels because only 45 countries have mapped data.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `cap_n_trade_agreements` | `n_trade_agreements` | Cumulative trade agreements | Total trade agreements signed up to that year | count (cumul.) | — |
| `cap_avg_agreement_depth` | `avg_agreement_depth` | Average agreement depth | Avg. depth score across treaties (base=1, protocol=0.5) | index (0–1) | — |
| `cap_n_new_agreements` | `n_new_agreements` | New agreements (year) | Number of new trade agreements signed in that year | count | — |

---

## 14. RFF WCPD — World Carbon Pricing Database

**Source:** Resources for the Future, distributed via Dryad. Compiled from ICAP, World Bank, national legislation.
**Coverage:** 182 countries, 1989–2022.
**What it measures:** Whether countries have carbon taxes or emissions trading systems (ETS) in force, the price levels, and the share of national emissions covered by carbon pricing instruments.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `inc_has_carbon_tax` | `has_carbon_tax` | Has carbon tax | Whether country has a carbon tax in force | binary (0/1) | B C D E |
| `inc_has_ets` | `has_ets` | Has ETS | Whether country has an emissions trading system | binary (0/1) | B C D E |
| `inc_has_carbon_pricing` | `has_carbon_pricing` | Has carbon pricing | Whether country has any carbon pricing (tax or ETS) | binary (0/1) | — |
| `inc_carbon_tax_rate_lcu` | `carbon_tax_rate_lcu` | Carbon tax rate | Average carbon tax rate (excl. exemptions) | LCU/tCO2 | — |
| `inc_ets_price_lcu` | `ets_price_lcu` | ETS price | Average ETS allowance price | LCU/tCO2 | — |
| `inc_carbon_pricing_coverage` | `carbon_pricing_coverage` | Carbon pricing coverage | Fraction of IPCC sectors covered by carbon pricing | share (0–1) | B C D E |

---

## 15. NewClimate Institute — Climate Policy Database

**Source:** NewClimate Institute, compiled from national policy documents, IEA, IRENA, and expert review.
**Coverage:** 198 countries, 1980–2025 (event-based, sparse for most technology combos).
**What it measures:** National climate and energy policy adoption — total counts, sector coverage, and binary indicators for specific policy types (carbon pricing, renewable targets, efficiency standards, EV policies, coal phaseout). Also produces ~110 dynamic `n_policies_<technology_combo>` columns counting policies by technology combination. Seven key variables are included in Panels D and E via zero-filling (NaN → 0, meaning no policy adopted that year). The binary dummies are regression-ready.

| Panel name | Source name | Real name | Description | Unit | Filtered |
|---|---|---|---|---|---|
| `inc_n_policies_total` | `n_policies_total` | Total climate policies | Number of climate policies adopted in that country-year | count | D E (zero-filled) |
| `inc_n_sectors_covered` | `n_sectors_covered` | Sectors covered by policy | Number of distinct economic sectors addressed by policies | count | D E (zero-filled) |
| `inc_has_carbon_pricing` | `has_carbon_pricing` | Has carbon pricing (CPD) | Whether any carbon pricing policy exists | binary (0/1) | D E (zero-filled) |
| `inc_has_renewable_target` | `has_renewable_target` | Has renewable target | Whether any renewable energy target policy exists | binary (0/1) | D E (zero-filled) |
| `inc_has_efficiency_standard` | `has_efficiency_standard` | Has efficiency standard | Whether any energy efficiency standard policy exists | binary (0/1) | D E (zero-filled) |
| `inc_has_ev_policy` | `has_ev_policy` | Has EV policy | Whether any electric vehicle/transport policy exists | binary (0/1) | D E (zero-filled) |
| `inc_has_coal_phaseout` | `has_coal_phaseout` | Has coal phaseout | Whether any coal phaseout policy exists | binary (0/1) | D E (zero-filled) |
| `inc_n_policies_<combo>` | `n_policies_<combo>` | Policies by technology combo | Count of policies targeting a specific technology combination (dynamic, ~110 columns) | count | — |

---

## Filtered Panel Summary

| Panel | Time window | Countries | Variables | Key additions over base |
|---|---|---|---|---|
| **A** | 1965–2022 | 50 | 32 | Deepest time series. OWID energy shares, V-Dem democracy. |
| **B** | 1992–2021 | 140 | 30 | Broadest country coverage. V-Dem + WCPD carbon pricing. |
| **C** | 2000–2022 | 136 | 52 | Ember electricity detail + ND-GAIN + WCPD. |
| **D** | 2000–2022 | 136 | 62 | Panel C + CCLW climate laws (3) + Climate Policy DB dummies/counts (7). Zero-filled. |
| **E** | 2002–2022 | 136 | 68 | Panel D + WGI governance (6). Costs 2 years vs. Panel D. |

All five panels are **fully balanced** — every country has every variable for every year in the window, with zero missing values. Panels D and E use zero-filling for CCLW and Climate Policy DB variables (NaN → 0 for count/binary indicators where absence of a record means no law or policy was adopted).
