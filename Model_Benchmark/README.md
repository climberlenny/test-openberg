# Description
The purpose of this test is to choose the best wind/oecan model to use in the simulation using Openberg model.
We'll compare 3 wind model : No wind , CARRA and ERA 5.

- Ocean model : GLOB CURRENT, GLORYS, TOPAZ 4, TOPAZ 5 and TOPAZ 6
- Observation : Iceberg observation (Baffin Bay in 2011,2012,2013 and 2021) + satellite tracked icebergs in 2018 and 2019
- The observations files needs to be in a csv format with at least 3 columns (Longitude, Latitude and Time)
- We apply a preprocessing routine to the observation files which has the following steps : 

  - Formatting the file to have (column name, date format) and converting it to a dataframe
  - Splitting the dataframe regarding time gaps( < 7 days )
  - Splitting the dataframe regarding the distance to the nearest coastline ( > 50km )
  - Interpolating the data to a desire timestep (1h)

- In output of the test :
