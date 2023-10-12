#Time Interpolation:

#1-It reads time values from both the observation (obs_time) and model (model_time) datasets.
#2-For each observation time (obs_time), it finds the nearest model times within a specified number of closest neighbors.
#3-It then interpolates the model data at these nearest model times to the observation time using linear interpolation, effectively 
#performing time interpolation.

#Spatial Interpolation:

#1-It reads latitude (obs_lat, model_lat) and longitude (obs_lon, model_lon) values from both the observation and model datasets.
#2-For each observation location (obs_lat, obs_lon), it identifies the nearest grid points in the model grid (model_lat_grid, model_lon_grid) 
#within a specified number of closest neighbors.
#3-It then interpolates the model data at these nearest grid points to the observation location using linear interpolation, effectively 
#performing spatial interpolation.


#How it does the time interpolation:

#The time interpolation in the code occurs within the interpolate_model_to_observation function.
#Here's a breakdown of the relevant parts of the code that perform time interpolation:


#1- It reads time values from both the observation (obs_time) and model (model_time) datasets:

#obs_time = obs_nc.variables['TIME'][:]
#model_time.append(model_nc.variables['time'][:])


#2- For each observation time (obs_time), it finds the nearest model times within a specified number of closest neighbors:

#for i in range(len(obs_time)):
#    distances = np.sqrt((obs_lat[i] - model_points[:, 0]) ** 2 + (obs_lon[i] - model_points[:, 1]) ** 2)
#    closest_indices = np.argsort(distances)[:10]  # Adjust the number of neighbors as needed
#3- It then interpolates the model data at these nearest model times to the observation time using linear interpolation, 
#effectively performing time interpolation:

#interpolated_values = []
#for var in model_data:
#    var_values = np.array(var).ravel()[closest_indices]
#  interp = LinearNDInterpolator(model_points[closest_indices], var_values, fill_value=999)
#   interpolated = interp(obs_lat[i], obs_lon[i])
#    interpolated_values.append(interpolated)

#So, the interpolate_model_to_observation function handles the entire process of reading and interpolating model data in time to match
#the observation times.


#How it does the spatial sinterpolation:

#The part of the code that performs spatial interpolation is mainly within the function interpolate_model_to_observation.
# Here's a breakdown of the steps involved in spatial interpolation:

#Reading Latitude and Longitude Values:

#It reads latitude values from both the observation (obs_lat) and model (model_lat) datasets.
#It reads longitude values from both the observation (obs_lon) and model (model_lon) datasets.
#Identification of Nearest Grid Points:

#For each observation location defined by obs_lat and obs_lon, the code identifies the nearest grid points in the model grid.
#It does this by computing the distances between the observation location and all grid points in the model grid
# (model_lat_grid, model_lon_grid).
#The code then selects the specified number of closest neighbors (in this case, up to 10) based on the computed distances.
#Spatial Interpolation:

#After identifying the nearest grid points, the code performs spatial interpolation to estimate model data at the observation location.
#It uses linear interpolation to estimate values at the observation location based on the values at the nearest grid points.
#The result is effectively spatial interpolation, where model data is estimated at each observation location based on the surrounding
#grid points.

#Author = Ghazal Mohammadpour

#-------------------------------------------------------------------------------


import netCDF4 as nc
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator

# Function to create a structured mesh grid for the model data
def create_model_grid(model_lat, model_lon):
    lon, lat = np.meshgrid(model_lon, model_lat)
    return lat, lon

# Function to interpolate model data to observation space
def interpolate_model_to_observation(obs_lat, obs_lon, obs_time, model_grid, model_data):
    model_lat_grid, model_lon_grid = model_grid
    model_points = np.column_stack((model_lat_grid.ravel(), model_lon_grid.ravel()))

    interpolated_model_data = []

    for i in range(len(obs_time)):
        distances = np.sqrt((obs_lat[i] - model_points[:, 0]) ** 2 + (obs_lon[i] - model_points[:, 1]) ** 2)
        closest_indices = np.argsort(distances)[:10]  # Adjust the number of neighbors as needed

        interpolated_values = []
        for var in model_data:
            var_values = np.array(var).ravel()[closest_indices]
            interp = LinearNDInterpolator(model_points[closest_indices], var_values, fill_value=999)
            interpolated = interp(obs_lat[i], obs_lon[i])
            interpolated_values.append(interpolated)

        interpolated_model_data.append(interpolated_values)

    return interpolated_model_data

# Specify the observation file path
obs_file = './JASON2_time_averaged_data.nc'

# Load observation NetCDF file
obs_nc = nc.Dataset(obs_file, 'r')

obs_lat = obs_nc.variables['LATITUDE'][:]
obs_lon = obs_nc.variables['LONGITUDE'][:]
obs_time = obs_nc.variables['TIME'][:]
# Add other observation variables as needed

obs_nc.close()

# Specify the directory containing model files
model_directory = '/scratch2/NCEPDEV/marine/Jessica.Meixner/Data/HR1/Winter/gfs.20200225/00/wave/gridded'

# Detect model files with the specified format
model_files = [f for f in os.listdir(model_directory) if f.startswith("gfswave.t00z.global.0p25.f") and f.endswith(".grib2.nc")]

# Initialize lists to store model data
model_lat = []
model_lon = []
model_time = []
model_data = []

# Loop through sorted model files
for model_file in sorted(model_files):
    # Create the model file path
    model_file_path = os.path.join(model_directory, model_file)

    # Load model gridded output NetCDF file
    model_nc = nc.Dataset(model_file_path, 'r')

    # Extract model data and append to lists
    model_lat.append(model_nc.variables['latitude'][:])
    model_lon.append(model_nc.variables['longitude'][:])
    model_time.append(model_nc.variables['time'][:])
    model_data.append([model_nc.variables['SPC_surface'][:], model_nc.variables['WIND_surface'][:]])
    # Add other model variables as needed

    model_nc.close()

# Create a structured mesh grid for the model data using the first model file's latitude and longitude
model_mesh_lat, model_mesh_lon = create_model_grid(model_lat[0], model_lon[0])

# Interpolate model data to observation space
interpolated_model_data = interpolate_model_to_observation(obs_lat, obs_lon, obs_time, (model_mesh_lat, model_mesh_lon), model_data)

# Create a new NetCDF file to store the interpolated data (replace 'output.nc' with your desired output file path)
output_file = 'output.nc'
with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
    # Define dimensions
    ncfile.createDimension('time', len(obs_time))

    # Create variables
    time_var = ncfile.createVariable('time', 'f8', ('time',))
    lat_var = ncfile.createVariable('latitude', 'f4', ('time',))
    lon_var = ncfile.createVariable('longitude', 'f4', ('time',))
    spc_surface_var = ncfile.createVariable('SPC_surface', 'f4', ('time',))
    wind_surface_var = ncfile.createVariable('WIND_surface', 'f4', ('time',))
    # Add other model variables as needed

    # Fill variables with data
    time_var[:] = obs_time
    lat_var[:] = obs_lat
    lon_var[:] = obs_lon
    spc_surface_var[:] = [interp[0] for interp in interpolated_model_data]
    wind_surface_var[:] = [interp[1] for interp in interpolated_model_data]
    # Fill other model variables

print(f"Interpolated model data saved in {output_file}")


