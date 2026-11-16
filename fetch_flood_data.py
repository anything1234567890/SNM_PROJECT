import ee
ee.Initialize(project="snm-flood-research")

# India boundary
india = (
    ee.FeatureCollection("FAO/GAUL/2015/level0")
    .filter(ee.Filter.eq("ADM0_NAME", "India"))
)

# Global Flood Database
floods = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

# Filter flood events for India
india_floods = floods.filter(
    ee.Filter.eq("dfo_country", "India")
)

print("Flood events:", india_floods.size().getInfo())

# Combine historical flood extents
flood_extent = india_floods.select("flooded").sum()

# Remove permanent water
permanent_water = india_floods.select("jrc_perm_water").max().unmask(0)

historical_floods = (
    flood_extent
    .updateMask(permanent_water.eq(0))
    .clip(india.geometry())
)

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=historical_floods,
    description="India_Historical_Floods",
    folder="Flood_Data",
    fileNamePrefix="India_Historical_Floods_2000_2018",
    region=india.geometry(),
    scale=250,
    maxPixels=1e13
)

task.start()

print("Export started. Check Google Earth Engine Tasks.")