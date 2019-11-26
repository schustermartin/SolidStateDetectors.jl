abstract type SOR_BACKEND end
struct CUDA_BACKEND <: SOR_BACKEND end



@fastmath function update!(   gpu_rbpot::CuDeviceArray{T}, gpu_ϵ, gpu_volume_weights, gpu_pointtypes, 
                        gpu_ρ, gw_r, gw_φ, gw_z, 
                        evenodd::Val{_evenodd} ) where {T, _evenodd}
    rbi_tar, rbi_src = _evenodd ? (2, 1) : (1, 2) 
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    nrb1 = size(gpu_rbpot, 3) - 2 # r
    nrb2 = size(gpu_rbpot, 2) - 2 # phi
    nrb1_nrb2 = nrb1 * nrb2
    nrb3 = size(gpu_rbpot, 1) - 2 # z
    N = nrb1_nrb2 * nrb3 

    @inbounds for i = index:stride:N # ni <--> normal index ( not red black)
        in3 = div(i - 1, nrb1_nrb2) + 1 # z
        r   = mod(i - 1, nrb1_nrb2) + 1 
        in2 = div(r - 1, nrb1) + 1 # phi
        in1 = mod(r - 1, nrb1) + 1 # r
        if _evenodd # if is true
            if iseven(in1 + in2)
                in3 = in3 * 2 
            else
                in3 = (in3 - 1) * 2 + 1
            end
        else
            if iseven(in1 + in2)
                in3 = (in3 - 1) * 2 + 1
            else
                in3 = in3 * 2 
            end
        end
        sync_threads()
        

        irb1 = in1 + 1 # second +1 due to extended grid
        irb2 = in2 + 1
        irb3 = div(in3, 2) + mod(in3, 2) + 1 
        
        rbidx_l = _evenodd ? (iseven(irb1 + irb2) ? (irb3) : (irb3 - 1)) : (iseven(irb1 + irb2) ? (irb3 - 1) : (irb3))
        rbidx_r = rbidx_l + 1

        begin
            # @cuprintf("%i \n", in3)
            pwwrr               = gw_r[1, in1]
            pwwrl               = gw_r[2, in1]
            r_inv_pwΔmpr        = gw_r[3, in1]
            Δr_ext_inv_r_pwmprr = gw_r[4, in1] 
            Δr_ext_inv_l_pwmprl = gw_r[5, in1] 
            Δmpr_squared        = gw_r[6, in1]  

            pwwφr        = gw_φ[1, in2]
            pwwφl        = gw_φ[2, in2]
            pwΔmpφ       = gw_φ[3, in2]
            Δφ_ext_inv_r = gw_φ[4, in2 + 1]
            Δφ_ext_inv_l = gw_φ[4, in2]
            
            if in1 == 1
                pwwφr = T(0.5)
                pwwφl = T(0.5)
                pwΔmpφ = T(2π)
                Δφ_ext_inv_r = inv(pwΔmpφ)
                Δφ_ext_inv_l = Δφ_ext_inv_r
            end
            sync_threads()

            pwwrr_pwwφr = pwwrr * pwwφr
            pwwrr_pwwφl = pwwrr * pwwφl
            pwwrl_pwwφr = pwwrl * pwwφr
            pwwrl_pwwφl = pwwrl * pwwφl

            Δr_ext_inv_r_pwmprr_pwΔmpφ = Δr_ext_inv_r_pwmprr * pwΔmpφ
            Δr_ext_inv_l_pwmprl_pwΔmpφ = Δr_ext_inv_l_pwmprl * pwΔmpφ
            pwΔmpφ_Δmpr_squared = pwΔmpφ * Δmpr_squared
            r_inv_pwΔmpr_Δφ_ext_inv_r = r_inv_pwΔmpr * Δφ_ext_inv_r
            r_inv_pwΔmpr_Δφ_ext_inv_l = r_inv_pwΔmpr * Δφ_ext_inv_l

            pwwzr        = gw_z[1, in3]
            pwwzl        = gw_z[2, in3]
            pwΔmpz       = gw_z[3, in3]
            Δz_ext_inv_r = gw_z[4, in3 + 1]
            Δz_ext_inv_l = gw_z[4, in3]

            ϵ_rrr = gpu_ϵ[  in1, in2 + 1, in3 + 1]
            ϵ_rlr = gpu_ϵ[  in1,     in2, in3 + 1]
            ϵ_rrl = gpu_ϵ[  in1, in2 + 1, in3 ]
            ϵ_rll = gpu_ϵ[  in1,     in2, in3 ]
            ϵ_lrr = gpu_ϵ[ in1, in2 + 1, in3 + 1]
            ϵ_llr = gpu_ϵ[ in1,     in2, in3 + 1]
            ϵ_lrl = gpu_ϵ[ in1, in2 + 1, in3 ] 
            ϵ_lll = gpu_ϵ[ in1,     in2, in3 ] 

            pwwφr_pwwzr = pwwφr * pwwzr
            pwwφl_pwwzr = pwwφl * pwwzr
            pwwφr_pwwzl = pwwφr * pwwzl
            pwwφl_pwwzl = pwwφl * pwwzl
            pwwrl_pwwzr = pwwrl * pwwzr
            pwwrr_pwwzr = pwwrr * pwwzr
            pwwrl_pwwzl = pwwrl * pwwzl
            pwwrr_pwwzl = pwwrr * pwwzl

                            # right weight in r: wrr
            wrr = ϵ_rrr * pwwφr_pwwzr
            wrr    = muladd(ϵ_rlr, pwwφl_pwwzr, wrr)   
            wrr    = muladd(ϵ_rrl, pwwφr_pwwzl, wrr)    
            wrr    = muladd(ϵ_rll, pwwφl_pwwzl, wrr)
            # left weight in r: wrr
            wrl = ϵ_lrr * pwwφr_pwwzr
            wrl    = muladd(ϵ_llr, pwwφl_pwwzr, wrl)   
            wrl    = muladd(ϵ_lrl, pwwφr_pwwzl, wrl)    
            wrl    = muladd(ϵ_lll, pwwφl_pwwzl, wrl) 
            # right weight in φ: wφr
            wφr = ϵ_lrr * pwwrl_pwwzr 
            wφr    = muladd(ϵ_rrr, pwwrr_pwwzr, wφr)  
            wφr    = muladd(ϵ_lrl, pwwrl_pwwzl, wφr)    
            wφr    = muladd(ϵ_rrl, pwwrr_pwwzl, wφr) 
            # left weight in φ: wφl
            wφl = ϵ_llr * pwwrl_pwwzr 
            wφl    = muladd(ϵ_rlr, pwwrr_pwwzr, wφl)  
            wφl    = muladd(ϵ_lll, pwwrl_pwwzl, wφl)    
            wφl    = muladd(ϵ_rll, pwwrr_pwwzl, wφl) 
            # right weight in z: wzr
            wzr = ϵ_rrr * pwwrr_pwwφr  
            wzr    = muladd(ϵ_rlr, pwwrr_pwwφl, wzr)     
            wzr    = muladd(ϵ_lrr, pwwrl_pwwφr, wzr)     
            wzr    = muladd(ϵ_llr, pwwrl_pwwφl, wzr)
            # left weight in z: wzr
            wzl = ϵ_rrl * pwwrr_pwwφr 
            wzl    = muladd(ϵ_rll, pwwrr_pwwφl, wzl)    
            wzl    = muladd(ϵ_lrl, pwwrl_pwwφr, wzl)    
            wzl    = muladd(ϵ_lll, pwwrl_pwwφl, wzl)

            wrr *= Δr_ext_inv_r_pwmprr_pwΔmpφ * pwΔmpz
            wrl *= Δr_ext_inv_l_pwmprl_pwΔmpφ * pwΔmpz
            wφr *= r_inv_pwΔmpr_Δφ_ext_inv_r * pwΔmpz
            wφl *= r_inv_pwΔmpr_Δφ_ext_inv_l * pwΔmpz
            wzr *= Δz_ext_inv_r * pwΔmpφ_Δmpr_squared
            wzl *= Δz_ext_inv_l * pwΔmpφ_Δmpr_squared
        end

        v1l = gpu_rbpot[   irb3,     irb2, irb1 - 1, rbi_src]
        v1r = gpu_rbpot[   irb3,     irb2, irb1 + 1, rbi_src]
        v2l = gpu_rbpot[   irb3, irb2 - 1,     irb1, rbi_src]
        v2r = gpu_rbpot[   irb3, irb2 + 1,     irb1, rbi_src]
        v3l = gpu_rbpot[rbidx_l,     irb2,     irb1, rbi_src] # red black axis 
        v3r = gpu_rbpot[rbidx_r,     irb2,     irb1, rbi_src] # red black axis

        old_potential = gpu_rbpot[irb3, irb2, irb1, rbi_tar]

        new_potential = gpu_ρ[irb3, irb2, irb1, rbi_tar]
        new_potential = muladd( wzl, v1l, new_potential)
        new_potential = muladd( wzr, v1r, new_potential)
        new_potential = muladd( wφl, v2l, new_potential)
        new_potential = muladd( wφr, v2r, new_potential)
        new_potential = muladd( wrl, v3l, new_potential)
        new_potential = muladd( wrr, v3r, new_potential)
        new_potential *= gpu_volume_weights[irb3, irb2, irb1, rbi_tar]

        new_potential -= old_potential
        new_potential = muladd(new_potential, T(1), old_potential)

        gpu_rbpot[irb3, irb2, irb1, rbi_tar] = ifelse(gpu_pointtypes[irb3, irb2, irb1, rbi_tar] & update_bit > 0, new_potential, old_potential)
    end        
    return nothing
end

@fastmath function update_boundaries!( gpu_rbpot::CuDeviceArray{T}, evenodd::Val{_evenodd} ) where {T, _evenodd}
    rbi_tar, rbi_src = _evenodd ? (2, 1) : (1, 2) 
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    nrb1 = size(gpu_rbpot, 3) - 2 # r
    nrb2 = size(gpu_rbpot, 2) - 2 # phi
    nrb1_nrb2 = nrb1 * nrb2
    nrb3 = size(gpu_rbpot, 1) - 2 # z
    N = nrb1_nrb2 * nrb3 

    @inbounds for i = index:stride:N # ni <--> normal index ( not red black)
        in3 = div(i - 1, nrb1_nrb2) + 1 # z
        r   = mod(i - 1, nrb1_nrb2) + 1 
        in2 = div(r - 1, nrb1) + 1 # phi
        in1 = mod(r - 1, nrb1) + 1 # r
        if _evenodd # if is true
            if iseven(in1 + in2)
                in3 = in3 * 2 
            else
                in3 = (in3 - 1) * 2 + 1
            end
        else
            if iseven(in1 + in2)
                in3 = (in3 - 1) * 2 + 1
            else
                in3 = in3 * 2 
            end
        end
        # sync_threads()  

        irb1 = in1 + 1 # second +1 due to extended grid
        irb2 = in2 + 1
        irb3 = div(in3, 2) + mod(in3, 2) + 1 
        
        rbidx_l = _evenodd ? (iseven(irb1 + irb2) ? (irb3) : (irb3 - 1)) : (iseven(irb1 + irb2) ? (irb3 - 1) : (irb3))
        rbidx_r = rbidx_l + 1
    end        
    return nothing
end

function update!(   fssrb::PotentialSimulationSetupRB{T}, ::Type{CUDA_BACKEND}; n_times::Int = 100, use_nthreads::Int = Base.Threads.nthreads(), 
                    depletion_handling::Val{depletion_handling_enabled} = Val{false}(), only2d::Val{only_2d} = Val{false}(),
                    is_weighting_potential::Val{_is_weighting_potential} = Val{false}())::Nothing where {T, depletion_handling_enabled, only_2d, _is_weighting_potential}
    # First step:
    # Convert Arrays to CuArrays
    gw1 = CuArray(fssrb.geom_weights[1].weights);  # r or x 
    gw2 = CuArray(fssrb.geom_weights[2].weights);  # φ or y
    gw3 = CuArray(fssrb.geom_weights[3].weights);  # z or z
    gw_r = gw1;
    gw_φ = gw2;
    gw_z = gw3;

    gpu_rbpot = CuArray( fssrb.potential ); 
    gpu_ϵ = CuArray(fssrb.ϵ);
    gpu_volume_weights = CuArray(fssrb.volume_weights);
    gpu_pointtypes = CuArray(fssrb.pointtypes);
    gpu_ρ = CuArray(fssrb.ρ);

    for i in 1:n_times
        @cuda threads=512 blocks=512 update!(gpu_rbpot, gpu_ϵ, gpu_volume_weights, gpu_pointtypes, gpu_ρ, gw_r, gw_φ, gw_z, Val(true) )
        # @cuda threads=512 blocks=512 update_boundaries!(gpu_rbpot, Val(true) )
        synchronize()
        @cuda threads=512 blocks=512 update!(gpu_rbpot, gpu_ϵ, gpu_volume_weights, gpu_pointtypes, gpu_ρ, gw_r, gw_φ, gw_z, Val(false) )
        # @cuda threads=512 blocks=512 update_boundaries!(gpu_rbpot, Val(false) )
        synchronize()
    end

    fssrb.potential[:] = Array(gpu_rbpot);
    fssrb.pointtypes[:] = Array(gpu_pointtypes);

    nothing
end
   

