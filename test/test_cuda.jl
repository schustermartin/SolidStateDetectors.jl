using SolidStateDetectors
SSD = SolidStateDetectors
# using Plots
using CUDAdrv
using CuArrays
using CUDAnative

T = Float32

sim = Simulation(SolidStateDetector{T}(SSD_examples[:Coax]));
# sim = Simulation(SolidStateDetector{T}(SSD_examples[:InvertedCoax]));

potential_type = ElectricPotential
CS = SSD.get_coordinate_system(sim.detector)
grid = Grid(sim.detector, init_grid_size = (40, 20, 100))
sor_consts = (T(1), T(1))
refine = false
only_2d = false
convergence_limit = T(1e-6)
is_weighting_potential = Val(false)
use_nthreads = 1
only2d = Val(only_2d)
depletion_handling = Val(false)

begin # CPU
    sim = Simulation(SolidStateDetector{T}(SSD_examples[:Coax]));
    apply_initial_state!(sim, potential_type, grid)
    plot( grid.r, grid.z, sim.electric_potential[:,1,:]', st=:heatmap )
    begin
        fssrb = SSD.PotentialSimulationSetupRB(sim.detector, sim.electric_potential.grid, 
                    sim.electric_potential.data, sor_consts = T.(sor_consts));

        @time begin
            for i in 1:500
                SSD.update!(fssrb, 
                    use_nthreads = use_nthreads, 
                    depletion_handling = depletion_handling, 
                    only2d = only2d, 
                    is_weighting_potential = is_weighting_potential)
            end
        end

        sim.ρ = ChargeDensity(SSD.ChargeDensityArray(fssrb), grid);
        sim.ρ_fix = ChargeDensity(SSD.FixedChargeDensityArray(fssrb), grid);
        sim.ϵ = DielectricDistribution(SSD.DielektrikumDistributionArray(fssrb), grid);
        sim.electric_potential = ElectricPotential(SSD.ElectricPotentialArray(fssrb), grid);
        sim.point_types = PointTypes(SSD.PointTypeArray(fssrb), grid);
    end

    plot( grid.r, grid.z, sim.electric_potential[:,1,:]', st=:heatmap, aspect_ratio = 1 )
end

gpuid = 0
dev = CuDevice(gpuid) # My graphics card
dev_capability  = capability(dev)
max_threads_per_block = dev_capability >= v"2.0.0" ? 1024 : 512


begin # GPU
    sim = Simulation(SolidStateDetector{T}(SSD_examples[:Coax]));
    apply_initial_state!(sim, potential_type, grid)
    # plot( grid.r, grid.z, sim.electric_potential[:,1,:]', st=:heatmap )
    begin
        fssrb = SSD.PotentialSimulationSetupRB(sim.detector, sim.electric_potential.grid, 
                        sim.electric_potential.data, sor_consts = T.(sor_consts));

        @time SSD.update!(fssrb, SSD.CUDA_BACKEND, n_times = 500)

        sim.ρ = ChargeDensity(SSD.ChargeDensityArray(fssrb), grid);
        sim.ρ_fix = ChargeDensity(SSD.FixedChargeDensityArray(fssrb), grid);
        sim.ϵ = DielectricDistribution(SSD.DielektrikumDistributionArray(fssrb), grid);
        sim.electric_potential = ElectricPotential(SSD.ElectricPotentialArray(fssrb), grid);
        sim.point_types = PointTypes(SSD.PointTypeArray(fssrb), grid);
    end;

    # plot( grid.r, grid.z, (sim.electric_potential[:,1,:]'), st=:heatmap )

end