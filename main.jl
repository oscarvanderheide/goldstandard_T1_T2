## Activate the current environment
using Pkg
Pkg.activate(".")

# Install dependencies: should only be done once, can be commented out afterwards
# Pkg.instantiate()

# Load external packages
using Revise

using FFTW
using HDF5
using LinearAlgebra
using MAT
using NamedDims
using PhilipsDataList
using PhilipsWorkspace
using ProgressBars
using Varpro

# Load utility functions
includet("utils.jl")

## T1 mapping
export_folder_T₁ = "/smb/user/oheide/RT-Temp/oheide/T1";
T1_map, _ = goldstandard_T₁(export_folder_T₁);

## T2 mapping
export_folder_T₂ = "/smb/user/oheide/RT-Temp/oheide/T2";
T2_map, _ = goldstandard_T₂(export_folder_T₂);

## Save T1 and T2 maps to .mat file
MAT.matwrite("T1_and_T2_maps.mat", Dict("T1_map" => T1_map, "T2_map" => T2_map));

# # Save input data to HDF5 files
# h5open("data_T1.h5", "w") do file

#     images, mask, inversion_times = prepare_data_T₁(export_folder_T₁)
#     file["images"] = images
#     file["mask"] = Int.(mask)
#     file["inversion_times"] = inversion_times

# end

# h5open("data_T2.h5", "w") do file

#     images, mask, echo_times = prepare_data_T₁(export_folder_T₂)
#     file["images"] = images
#     file["mask"] = Int.(mask)
#     file["echo_times"] = echo_times

# end