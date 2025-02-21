"""
    goldstandard_T₁(export_folder::String)

Fit T₁ to inversion-recovery spin-echo data with multiple inversion times using the VarPro algorithm.

Args: 
- `export_folder::String`: Path to the export folder containing the raw.{data,list} files and a workspace.json file.

Returns:
- `T1_map::Array{Float64,1}`: T₁ map in ms.
- `PD_map::Array{ComplexF64,1}`: Proton density map.
"""
function goldstandard_T₁(export_folder::String)

    # Load kspace data, compress to single virtual coil, and go to image space
    images, mask, inversion_times = prepare_data_T₁(export_folder)
    # Perform T₁ mapping with VarPro algorithm
    T1_map, PD_map = varpro_T₁(images, mask, inversion_times)

    return T1_map, PD_map
end

"""
    goldstandard_T₂(export_folder::String)
    
Fit T₂ to inversion-recovery spin-echo data with multiple echo times using the VarPro algorithm.

Args: 
- `export_folder::String`: Path to the export folder containing the raw.{data,list} files and a workspace.json file.

Returns:
- `T2_map::Array{Float64,1}`: T₂ map in ms.
- `PD_map::Array{ComplexF64,1}`: Proton density map.
"""
function goldstandard_T₂(export_folder::String)

    images, mask, echo_times = prepare_data_T₂(export_folder)

    # Fit T₂
    T2_map, PD_map = varpro_T₂(images, mask, echo_times)

    return T2_map, PD_map
end

"""
    prepare_data_T₁(export_folder::String)

Prepare the data for T₁ mapping.

Args:
- `export_folder::String`: Path to the export folder containing the raw.{data,list} files and a workspace.json file.

Returns:
- `images::Array{ComplexF64,3}`: Complex image data.
- `mask::Array{Bool,2}`: Mask of the image data.
- `inversion_times::Array{Float64,1}`: Inversion times in ms.
"""
function prepare_data_T₁(export_folder::String)

    # Load in kspace from raw.{data,list} files
    kspace = PhilipsDataList.data_list_to_kspace(joinpath(export_folder, "raw.data"), remove_readout_oversampling=true)

    # For 1D acquisition, re-add ky dimension otherwise we need to write separate code for 1D and 2D acquisitions
    if :ky ∉ dimnames(kspace)
        kx, dyn, chan = size(kspace)
        kspace = reshape(kspace, kx, 1, dyn, chan)
        kspace = NamedDimsArray(kspace, (:kx, :ky, :dyn, :chan))
    end

    # Read sequence parameters from workspace.json file
    workspace = read(joinpath(export_folder, "workspace.json"), String)

    # Extract the inversion times from the workspace
    TI_increments = extract_EX_PROTO_scan_int_array(workspace)
    base_TI = extract_base_inversion_delay(workspace)
    inversion_times = base_TI .+ TI_increments


    max_dyn = min(size(kspace, :dyn), length(inversion_times))
    inversion_times = inversion_times[1:max_dyn]

    # Sort inversion times and kspace data in ascending order
    idx = sortperm(inversion_times)
    inversion_times = inversion_times[idx]
    kspace = kspace[:, :, :, idx]

    println("Inversion times [ms]: $(inversion_times)")

    # Compress to single virtual coil with SVD
    kspace = compress_to_single_virtual_coil(kspace)

    # Go to image space
    images = ifftc_2d(kspace)

    # Generate mask by thresholding
    image_sum = sum(abs.(images), dims=3)
    mask = image_sum .> 0.05 * maximum(image_sum)
    mask = dropdims(mask, dims=3)

    return images, mask, inversion_times

end

"""
    prepare_data_T₂(export_folder::String)

Prepare the data for T₂ mapping.

Args:
- `export_folder::String`: Path to the export folder containing the raw.{data,list} files and a workspace.json file.
    
Returns:
- `images::Array{ComplexF64,3}`: Complex image data.
- `mask::Array{Bool,2}`: Mask of the image data.
- `echo_times::Array{Float64,1}`: Echo times in ms.
"""
function prepare_data_T₂(export_folder::String)

    # Load in kspace from raw.{data,list} files
    kspace = PhilipsDataList.data_list_to_kspace(joinpath(export_folder, "raw.data"), remove_readout_oversampling=true)

    # Read sequence parameters from workspace.json file
    workspace = read(joinpath(export_folder, "workspace.json"), String)

    # For 1D acquisition, re-add ky dimension otherwise we need to write separate code for 1D and 2D acquisitions
    if :ky ∉ dimnames(kspace)
        kx, dyn, chan = size(kspace)
        kspace = reshape(kspace, kx, 1, dyn, chan)
        kspace = NamedDimsArray(kspace, (:kx, :ky, :dyn, :chan))
    end

    # Extract the inversion times from the workspace
    TE_increments = extract_EX_PROTO_scan_int_array(workspace)
    base_TE = extract_base_echo_time(workspace) # [ms]
    echo_times = base_TE .+ TE_increments
    echo_times = echo_times[1:size(kspace, :dyn)]

    max_dyn = min(size(kspace, :dyn), length(echo_times))
    echo_times = echo_times[1:max_dyn]

    # Sort the echo times and kspace data in ascending order
    idx = sortperm(echo_times)
    echo_times = echo_times[idx]
    kspace = kspace[:, :, :, idx]

    println("Echo times [ms]: $(echo_times)")

    # Compress to single virtual coil with SVD
    kspace = compress_to_single_virtual_coil(kspace)

    # Go to image space
    images = ifftc_2d(kspace)

    # Generate mask by thresholding
    image_sum = sum(abs.(images), dims=3)
    mask = image_sum .> 0.05 * maximum(image_sum)
    mask = dropdims(mask, dims=3)

    return images, mask, echo_times
end



ifftc_1d(x) = fftshift(ifft(ifftshift(x, 1), 1), 1);
ifftc_2d(x) = fftshift(ifft(ifftshift(x, (1, 2)), (1, 2)), (1));

function compress_to_single_virtual_coil(kspace::AbstractArray{T,4}) where {T}

    # Put the channel dimension last
    kspace = permutedims(kspace, (:kx, :ky, :dyn, :chan))
    nkx, nky, ndyn, nchan = size(kspace)

    # Reshape into a matrix with "data per channel" as columns
    kspace = reshape(kspace, nkx * nky * ndyn, nchan)

    # Appyl SVD and extract first singular vector
    U, _, _ = svd(kspace)
    kspace = U[:, 1]

    # Reshape back to original shape (without chan dimension)
    kspace = reshape(kspace, nkx, nky, ndyn)

    return kspace
end

function compress_to_single_virtual_coil(kspace::AbstractArray{T,3}) where {T}

    # Put the channel dimension last
    kspace = permutedims(kspace, (:kx, :dyn, :chan))
    nkx, ndyn, nchan = size(kspace)

    # Reshape into a matrix with "data per channel" as columns
    kspace = reshape(kspace, nkx * ndyn, nchan)

    # Apply SVD and extract first singular vector
    U, _, _ = svd(kspace)
    kspace = U[:, 1]

    # Reshape back to original shape (without chan dimension)
    kspace = reshape(kspace, nkx, ndyn)

    return kspace
end

function extract_EX_PROTO_scan_int_array(workspace_string::String)

    pattern = r"\"EX_PROTO_scan_int_array\":\s*\{[^}]*\}"
    m = match(pattern, workspace_string)

    if m === nothing
        error("EX_PROTO_scan_int_array entry not found")
    end

    EX_PROTO_scan_int_array_string = m.match

    pattern = r"\"Value\":\s*\[([^\]]*)\]"
    m = match(pattern, EX_PROTO_scan_int_array_string)

    if m === nothing
        error("Value array not found in EX_PROTO_scan_int_array")
    end

    value_string = m.captures[1]
    value_array = [parse(Int, x) for x in split(value_string, ",")]
    
    # Remove trailing zeros, but keep the first one
    while value_array[end] == 0
        pop!(value_array)
    end

    return value_array
end

""" Extract the base inversion delay from the workspace"""
function extract_base_inversion_delay(workspace_string::String)

    pattern = r"\"VAL01_INV_delay\":\s*\{[^}]*\}"
    m = match(pattern, workspace_string)

    VAL01_INV_delay = m.match

    if m === nothing
        error("VAL01_INV_delay entry not found")
    end

    pattern = r"\"Value\":\s*\[([^\]]*)\]"
    m = match(pattern, VAL01_INV_delay)

    if m === nothing
        error("Value array not found in VAL01_INV_delay")
    end

    value_string = m.captures[1]
    value = parse(Float64, value_string)

    return value
end

""" Extract the base echo time from the workspace"""
function extract_base_echo_time(workspace_string::String)

    pattern = r"\"VAL03_RFE_act_first_echo_time\":\s*\{[^}]*\}"
    m = match(pattern, workspace_string)

    VAL03_RFE_act_first_echo_time = m.match

    if m === nothing
        error("VAL03_RFE_act_first_echo_time entry not found")
    end

    pattern = r"\"Value\":\s*\[([^\]]*)\]"
    m = match(pattern, VAL03_RFE_act_first_echo_time)

    if m === nothing
        error("Value array not found in VAL01_INV_delay") 
    end

    value_string = m.captures[1]
    value = parse(Float64, value_string)

    return value
end

"""
    varpro_T₁(complex_image_data::Array{T, N}, mask, inversion_times) where {T,N}

Reconstruct T₁ from single-echo, inversion-recovery data. 

- Uses the "VarPro" algorithm to separete the linear (proton density) and non-linear (T₁) parameters.
- Assumes TR >> T₁ s.t. there is full recovery in between RF pulses (otherwise a model with additional exp(-TR/T₁) term is needed).
- Inversion time dimension is the third dimension of the `image_data` array
- The input data is complex but VarPro only works with real values. Therefore, work with `[real(complex_voxel_data); imag(complex_voxel_data)]` instead and double the number of linear parameters.
"""
function varpro_T₁(complex_image_data::Array{T,N}, mask, inversion_times) where {T<:Complex,N}

    nx, ny = size(mask)
    
    # Initialize output arrays
    T₁_map = zeros(real(T), nx, ny)
    PD_map = zeros(T, nx, ny)

    # Loop over all voxels, fit T₁ and PD, and store in output arrays
    for voxel in ProgressBar(CartesianIndices(mask))

        if !mask[voxel]
            continue
        end

        complex_voxel_data = complex_image_data[voxel, :]
        varpro_data = [real(complex_voxel_data); imag(complex_voxel_data)]

        try
            T₁, PD = varpro_T₁_single_voxel(varpro_data, inversion_times)
            T₁_map[voxel] = T₁
            PD_map[voxel] = PD
        catch err
            println("Skipping voxel $(voxel) due to error: $(err)")
            continue
        end
    end

    return T₁_map, PD_map
end

"Fit T₁ in a single voxel using VarPro"
function varpro_T₁_single_voxel(data::Vector{T}, inversion_times) where {T<:Real}

    w = ones(2 * length(inversion_times))
    nlinpars = 2 # PDre and PDim

    ind = [1 2; 1 1]

    x_init = [1000.0] # < needs to be reset each time, bug in Varpro implementation!
    ctx = FitContext(Float64.(data), Float64.([inversion_times; inversion_times]), w, x_init, nlinpars, ind, ϕ_T₁, ∂ϕ_T₁)
    ctx.verbose = false

    (alpha, c, wresid, resid_norm, y_est, regression) = varpro(ctx)

    T₁ = alpha[1]
    PD = complex(c[1], c[2])

    return T₁, PD
end

"The VarPro model matrix for T₁ fitting"
function ϕ_T₁(T₁, ctx)

    # Because of the [real; imag] stuff, the model matrix is a bit weird
    @. ctx.phi[1:end÷2, 1] = (1 - 2 * exp(-ctx.t[1:end÷2] / T₁))
    @. ctx.phi[1:end÷2, 2] = 0
    @. ctx.phi[end÷2+1:end, 1] = 0
    @. ctx.phi[end÷2+1:end, 2] = (1 - 2 * exp(-ctx.t[1:end÷2] / T₁))

    ctx.phi
end

"Partial derivatives of the VarPro model matrix ϕ w.r.t T₁"
function ∂ϕ_T₁(T₁, ctx)

    # Because of the [real; imag] stuff, the model derivatives matrix is a bit weird
    @. ctx.dphi[1:end÷2, 1] = 2 * -exp(-ctx.t[1:end÷2] / T₁) * (ctx.t[1:end÷2] * T₁^-2)
    @. ctx.dphi[1:end÷2, 2] = 0
    @. ctx.dphi[end÷2+1:end, 1] = 0
    @. ctx.dphi[end÷2+1:end, 2] = 2 * -exp(-ctx.t[1:end÷2] / T₁) * (ctx.t[1:end÷2] * T₁^-2)

    ctx.dphi
end

"""
    varpro_T₂(complex_image_data::Array{ComplexF64, 3}, mask, echo_times)

Reconstruct T₂ from single-echo, spin-echo data.

- Uses the "VarPro" algorithm to separete the linear (proton density) and non-linear (T₁) parameters.
- Assumes TR >> T₁ s.t. there is full recovery in between RF pulses (otherwise a model with additional exp(-TR/T₁) term is needed).
- Echo time dimension is the third dimension of the`image_data`` array
- The input data is complex but VarPro only works with real values. Therefore, work with `[real(image_data); imag(image_data)]` instead and double the number of linear parameters.
"""
function varpro_T₂(complex_image_data::Array{T,N}, mask, echo_times) where {T<:Complex,N}

    nx, ny = size(mask)

    # Initialize output arrays
    T₂_map = zeros(real(T), nx, ny)
    PD_map = zeros(T, nx, ny)

    # Loop over all voxels, fit T₂ and PD, and store in output arrays
    for voxel in ProgressBar(CartesianIndices(mask))
        if !mask[voxel]
            continue
        end

        complex_voxel_data = complex_image_data[voxel, :]
        varpro_data = abs.([real(complex_voxel_data); imag(complex_voxel_data)])

        try
            T₂, PD = varpro_T₂_single_voxel(varpro_data, echo_times)
            T₂_map[voxel] = T₂
            PD_map[voxel] = PD
        catch err
            println("Skipping voxel $(voxel) due to error: $(err)")
            continue
        end
    end

    return T₂_map, PD_map
end

"Fit T₂ in a single voxel using VarPro"
function varpro_T₂_single_voxel(data::Vector{T}, echo_times) where {T<:Real}

    w = ones(2 * length(echo_times))
    nlinpars = 2 # PDre and PDim

    ind = [1 2; 1 1]

    x_init = [100.0] # < needs to be reset each time, bug in Varpro implementation!
    ctx = FitContext(Float64.(data), Float64.([echo_times; echo_times]), w, x_init, nlinpars, ind, ϕ_T₂, ∂ϕ_T₂)
    ctx.verbose = false

    (alpha, c, wresid, resid_norm, y_est, regression) = varpro(ctx)

    T₂ = alpha[1]
    PD = complex(c[1], c[2])

    return T₂, PD
end

"The VarPro model matrix for T₂ fitting"
function ϕ_T₂(T₂, ctx)

    # Because of the [real; imag] stuff, the model matrix is a bit weird
    @. ctx.phi[1:end÷2, 1] = exp(-ctx.t[1:end÷2] / T₂)
    @. ctx.phi[1:end÷2, 2] = 0
    @. ctx.phi[end÷2+1:end, 1] = 0
    @. ctx.phi[end÷2+1:end, 2] = exp(-ctx.t[1:end÷2] / T₂)

    return ctx.phi
end

"Partial derivatives of the VarPro model matrix ϕ w.r.t T₂"
function ∂ϕ_T₂(T₂, ctx)

    # Because of the [real; imag] stuff, the model derivatives matrix is a bit weird
    @. ctx.dphi[1:end÷2, 1] = exp(-ctx.t[1:end÷2] / T₂) * (ctx.t[1:end÷2] * T₂^-2)
    @. ctx.dphi[1:end÷2, 2] = 0
    @. ctx.dphi[end÷2+1:end, 1] = 0
    @. ctx.dphi[end÷2+1:end, 2] = exp(-ctx.t[1:end÷2] / T₂) * (ctx.t[1:end÷2] * T₂^-2)

    return ctx.dphi
end
