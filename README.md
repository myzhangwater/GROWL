# GROWL Dataset (Version 1.0)

**License:** CC-BY 4.0
**Repository:** [Zenodo](https://zenodo.org/)

The GROWL dataset provides comprehensive information on reservoirs, including both static attributes and time series data of water levels or storage. It is designed to support hydrological research, water resources management, and environmental studies.

## Dataset Components

The dataset is organized into three main components for ease of use:

1. **GROWL_metadata.csv**

   - Contains static attributes for each of the 3,612 reservoirs in the dataset.

   - Attributes include:

     |       **Field name**        |                   **Description / Notes**                    |
     | :-------------------------: | :----------------------------------------------------------: |
     |           RES_ID            | Primary key of the database; ensures unambiguous  reference to each station or reservoir. |
     |           Station           |     Original identifier assigned by the data  provider.      |
     |名字| Designated name of the station or reservoir, as  provided in authoritative records. |
     |          Latitude           | Geographic coordinate in decimal degrees (WGS84  reference system). |
     |          Longitude          | Geographic coordinate in decimal degrees (WGS84  reference system). |
     |     Temporal_Resolution     | Nominal time step of the observations (e.g., *Daily*,  *Monthly*). |
     |        Level_Periods        | Temporal extent of the water level time series  (e.g., 1980–2020). |
     |       Storage_Periods       |    Temporal extent of the storage (volume) time  series.     |
     |  Level_Record_Length_Years  | Total number of years with valid water level  observations.  |
     | Storage_Record_Length_Years |   Total number of years with valid storage  observations.    |
     |     In-situ_Level_Ratio     | Fraction of water level records derived from  direct (in-situ) measurements, relative to all available data. |
     |    In-situ_Storage_Ratio    | Fraction of storage records derived from in-situ  measurements, relative to all available data. |
     |     Station_Elevation_m     |           Elevation above mean sea level (meters).           |
     |       Vertical_Datum        | Geodetic reference used for elevation  measurements (e.g., EGM96, WGS84). |
     |    Hydrolakes_Match_Type    | Method of correspondence between the station and  HydroLAKES database (e.g., ID match, nearest spatial neighbor). |
     |     Hydrolakes_Hylak_id     | Unique identifier of the corresponding  lake/reservoir in HydroLAKES. |
     | Hydrolakes_Match_Distance_m | Spatial separation (meters) between the station  and the matched HydroLAKES feature. |
     |    Reservoirs_Match_Type    | Method of correspondence with the Reservoirs  database (spatial or attribute-based). |
     |      Reservoirs_fid_1       | Unique identifier of the corresponding feature in  the Reservoirs database. |
     | Reservoirs_Match_Distance_m | Spatial separation (meters) between the station  and the matched Reservoirs feature. |
     |       GDW_Match_Type        | Method of correspondence with the Global  Dam/Reservoir (GDW) database. |
     |         GDW_GDW_ID          | Unique identifier of the corresponding feature in  the GDW database. |
     |    GDW_Match_Distance_m     | Spatial separation (meters) between the station  and the matched GDW feature. |
     |类型| Nature of the data source (e.g., *in-situ  station*, *remote sensing*). |
     |           Country           | Sovereign state in which the station or reservoir  is located. |
     |           Source            |   Organization or project responsible for data  provision.   |
     |         Source_Link         |  Persistent link to the dataset or institutional  webpage.   |
     |       Other_Metadata        | Additional descriptive attributes not covered by  the structured fields.  **** |

   - Purpose: Provides reference information and metadata for each reservoir.

2. **GROWL_timeseries/**

   - Directory containing long-term time series of water level or storage for each reservoir.

   - Each reservoir has a separate file.

   - Attributes include:

     | **Field Name** |                       **描述**                        |     **Data Type / Unit**      |
     | :------------: | :----------------------------------------------------------: | :---------------------------: |
     |     RES_ID     | Unique identifier assigned to each reservoir,  consistent with GROWL_metadata.csv |            Integer            |
     |      Date      |                Observation date in ISO format                |          YYYY-MM-DD           |
     |   Level_Raw    | Original in situ water level observation prior to  QC. For remote-sensing products, corresponds to water surface elevation |            Meters             |
     |   Flag_Level   | Quality flag for water level:  0 = good; 1 = error; 2 = interpolated; 3 =  suspect |     Integer (categorical)     |
     |     Level      | Quality-controlled long-term water level time  series. For remote-sensing products, corresponds to water surface elevation |           Meters(m)           |
     |  Storage_Raw   | Original in situ reservoir storage observation  prior to QC. For DAHITI-derived records, represents volume variation rather  than absolute storage values | Million cubic meters (10⁶ m³) |
     |  Flag_Storage  | Quality flag for reservoir storage:  0 = good; 1 = error; 2 = interpolated; 3 =  suspect |     Integer (categorical)     |
     |    Storage     | Quality-controlled long-term reservoir storage  time series. For DAHITI-derived records, represents volume variation rather  than absolute storage values | Million cubic meters (10⁶ m³) |

   - Purpose: Supports analysis of long-term trends and patterns in reservoir water level and storage.

3. **GROWL_shortterm/**

   - Directory containing in-situ observations of water level or storage for reservoirs with records shorter than one year.
   - Data are provided in raw form without processing.

## Usage Notes

- The dataset is provided under the [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Users are free to share and adapt the data as long as appropriate credit is given.
- Time series files may have different lengths depending on the available observations for each reservoir.
