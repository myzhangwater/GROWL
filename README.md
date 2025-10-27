# **GROWL Example Code Repository**

This repository provides example scripts for spatial matching and data processing of the **GROWL (Global Reservoir and Lake Water Level)** dataset. It includes workflows for **spatial association** between hydrological stations and reservoir polygons, as well as procedures for **quality control and flagging protocol** of water level and storage time series.

### **Contents**

#### **1. 37.CSV**
Sample data provided by the original institution or agency, which has not undergone any preprocessing or cleaning.

#### **2. HydroPointMatcher.py**

Demonstrates spatial matching between hydrological stations and reservoir polygons, including:

- Buffer-based and within-polygon matching between points and polygons
- Calculation of the shortest boundary distance (m) for buffer matches

#### **3. DataProcess.py**

Demonstrates time-series quality control and interpolation for water level and storage datasets, including:

- Timeline regularization and gap segmentation
- Adaptive robust Z 
- Jump detection
- Monotonic consistency between Level & Storage
- Cross-variable completion via monotonic mapping

### **Resources**

- **Interactive visualization of the GROWL dataset:** https://ee-zmy18888536368.projects.earthengine.app/view/growl
- **Download the complete GROWL dataset:** https://figshare.com/s/e993f2563e914ebb9121
