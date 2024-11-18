# functions for fitting T1 and T2 to gold standard spin echo (inversion recovery) measurements using VarPro

function prepare_data_T₁(export_folder::String)

    kspace = PhilipsDataList.data_list_to_kspace(joinpath(export_folder, "raw.data"), remove_readout_oversampling=true)

    workspace = read(joinpath(export_folder, "workspace.json"), String)

    # Extract the inversion times from the workspace
    TI_increments = extract_EX_PROTO_scan_int_array(workspace)
    base_TI = extract_base_inversion_delay(workspace)
    inversion_times = base_TI .+ TI_increments

    # Sort inversion times and kspace data in ascending order
    idx = sortperm(inversion_times)
    inversion_times = inversion_times[idx]
    kspace = kspace[:, :, idx]

    println("Inversion times [ms]: $(inversion_times)")

    # Compress to single virtual coil with SVD
    kspace = compress_to_single_virtual_coil(kspace)

    # Go to image space
    images = ifftc_1d(kspace)

    # Generate mask by thresholding
    image_sum = sum(abs.(images), dims=2)
    mask = image_sum .> 0.1 * maximum(image_sum)

    return images, mask, inversion_times

end

function goldstandard_T₁(export_folder::String)

    images, mask, inversion_times = prepare_data_T₁(export_folder)

    T1_map, PD_map = varpro_T₁(images, mask, inversion_times)

    return T1_map, PD_map
end

function prepare_data_T₂(export_folder::String)

    kspace = PhilipsDataList.data_list_to_kspace(joinpath(export_folder, "raw.data"), remove_readout_oversampling=true)

    workspace = read(joinpath(export_folder, "workspace.json"), String)

    # Extract the inversion times from the workspace
    TE_increments = extract_EX_PROTO_scan_int_array(workspace)
    base_TE = extract_base_echo_time(workspace) # [ms]
    echo_times = base_TE .+ TE_increments

    # Sort the echo times and kspace data in ascending order
    idx = sortperm(echo_times)
    echo_times = echo_times[idx]
    kspace = kspace[:, :, idx]

    println("Echo times [ms]: $(echo_times)")

    # Compress to single virtual coil with SVD
    kspace = compress_to_single_virtual_coil(kspace)

    # Go to image space
    images = ifftc_1d(kspace)

    # Generate mask by thresholding
    image_sum = sum(abs.(images), dims=2)
    mask = image_sum .> 0.1 * maximum(image_sum)

    return images, mask, echo_times
end
    
function goldstandard_T₂(export_folder::String)

    images, mask, echo_times = prepare_data_T₂(export_folder)

    # Fit T₂
    T2_map, PD_map = varpro_T₂(images, mask, echo_times)

    return T2_map, PD_map
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
    U, S, V = svd(kspace)
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

    # Appyl SVD and extract first singular vector
    U, S, V = svd(kspace)
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
    @show value_array
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

    num_voxels = length(mask)

    # Initialize output arrays
    T₁_map = zeros(real(T), num_voxels)
    PD_map = zeros(T, num_voxels)

    # Loop over all voxels, fit T₁ and PD, and store in output arrays
    for voxel in 1:num_voxels
        if !mask[voxel]
            continue
        end

        complex_voxel_data = complex_image_data[voxel, :]
        varpro_data = [real(complex_voxel_data); imag(complex_voxel_data)]

        T₁, PD = varpro_T₁_single_voxel(varpro_data, inversion_times)
        T₁_map[voxel] = T₁
        PD_map[voxel] = PD
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

    num_voxels = length(mask)

    # Initialize output arrays
    T₂_map = zeros(real(T), num_voxels)
    PD_map = zeros(T, num_voxels)

    # Loop over all voxels, fit T₂ and PD, and store in output arrays
    for voxel in 1:num_voxels
        if !mask[voxel]
            continue
        end

        complex_voxel_data = complex_image_data[voxel, :]
        varpro_data = abs.([real(complex_voxel_data); imag(complex_voxel_data)])

        T₂, PD = varpro_T₂_single_voxel(varpro_data, echo_times)
        T₂_map[voxel] = T₂
        PD_map[voxel] = PD
    end

    return T₂_map, PD_map
end

"Fit T₂ in a single voxel using VarPro"
function varpro_T₂_single_voxel(data::Vector{T}, echo_times) where {T<:Real}

    w = ones(2 * length(echo_times))
    nlinpars = 2 # PDre and PDim

    ind = [1 2; 1 1]

    x_init = [1000.0] # < needs to be reset each time, bug in Varpro implementation!
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
