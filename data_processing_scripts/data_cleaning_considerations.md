## Cleaning
* Dropped duplicated rows using keep-first.
* Dropped rows with null values in year field.
* Dropped rows with null values in iso3 field.
* Filter the dataset to 2000 >= year < 2023 to match the available years in EDGAR db.
* Dropped records with Announced, Planned, and Ended 'status'.


## Data Insights
* Several policy titles are repeated accross different years.
* Several policy titles are repeated accross different status although countries might be different and years too. 