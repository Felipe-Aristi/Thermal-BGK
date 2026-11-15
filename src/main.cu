
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "output.cuh"

#include "utilities/cudaUtilities.cuh"

// ============================================================
// Main
// ============================================================

int main()
{
    Memory memory{};

    try
    {
        // ====================================================
        // Simulation information
        // ====================================================

        std::cout << std::fixed << std::setprecision(12);

        std::cout << "\n";
        std::cout << "============================================\n";
        std::cout << "     TBGK - Thermal BGK Lattice Boltzmann   \n";
        std::cout << "       Rayleigh-Benard Convection            \n";
        std::cout << "============================================\n";

        std::cout << "\nLattice:\n";
        std::cout << "Q                    = " << D2V37::Q << "\n";
        std::cout << "D                    = " << D2V37::D << "\n";

        std::cout << "\nComputational domain:\n";
        std::cout << "NX physical          = " << NXPhysical << "\n";
        std::cout << "NY physical          = " << NYPhysical << "\n";
        std::cout << "NX allocated         = " << NX << "\n";
        std::cout << "NY allocated         = " << NY << "\n";
        std::cout << "Ghost layers         = " << Domain::ghostLayers << "\n";
        std::cout << "Aspect ratio         = " << aspectRatio << "\n";

        std::cout << "\nThermodynamic parameters:\n";
        std::cout << "Reference density    = " << rho0 << "\n";
        std::cout << "Reference temperature= " << referenceTemperature << "\n";
        std::cout << "Lower temperature    = " << lowerWallTemperature << "\n";
        std::cout << "Upper temperature    = " << upperWallTemperature << "\n";
        std::cout << "Temperature diff.    = " << temperatureDifference << "\n";
        std::cout << "cv                   = " << cv << "\n";
        std::cout << "cp                   = " << cp << "\n";
        std::cout << "Prandtl              = " << prandtl << "\n";

        std::cout << "\nRayleigh-Benard parameters:\n";
        std::cout << "Rayleigh             = " << rayleigh << "\n";
        std::cout << "Calculated Rayleigh  = " << calculatedRayleigh << "\n";
        std::cout << "Polytropic index     = " << polytropicIndex << "\n";
        std::cout << "Gravity              = " << gravity << "\n";
        std::cout << "Superadiabatic frac. = " << superadiabaticFraction << "\n";

        std::cout << "\nTransport coefficients:\n";
        std::cout << "Reference viscosity  = " << nuReference << "\n";
        std::cout << "Reference diffusivity= " << alphaReference << "\n";
        std::cout << "Dynamic viscosity    = " << dynamicViscosityReference << "\n";
        std::cout << "Thermal conductivity = " << thermalConductivityReference << "\n";
        std::cout << "Reference tau        = " << tauReference << "\n";
        std::cout << "Reference omega      = " << omegaReference << "\n";

        std::cout << "\nSimulation parameters:\n";
        std::cout << "Steps                = " << steps << "\n";
        std::cout << "Output every         = " << outputEvery << "\n";
        std::cout << "Perturbation         = " << perturbationAmplitude << "\n";
        std::cout << "Population memory MB = " << static_cast<double>(bytesTotal) / 1.0e6 << "\n";

        // ====================================================
        // Allocate GPU memory
        // ====================================================

        allocateMemory(memory);

        // ====================================================
        // Initialize hydrostatic Rayleigh-Benard state
        // ====================================================

        launchRBInitialization(
            memory,
            rho0,
            referenceTemperature,
            temperatureDifference,
            polytropicIndex,
            perturbationAmplitude,
            accelerationY
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Start performance timer
        // ====================================================

        const auto start = std::chrono::steady_clock::now();

        // ====================================================
        // Main time integration
        //
        // 1. Apply diffuse-reflection boundary conditions.
        // 2. Optionally reconstruct and save macroscopic fields.
        // 3. Pull stream and collide with thermal BGK.
        // 4. Swap population buffers.
        //
        // The output routine reconstructs incoming populations.
        // Therefore, output is taken after preparing boundaries
        // and before the BGK collision for that step.
        // ====================================================

        for (label_t step = 1; step <= steps; ++step)
        {
            // ------------------------------------------------
            // Thermal diffuse-reflection boundary conditions
            // ------------------------------------------------

            launchRBDiffuseBoundary(
                memory,
                lowerWallTemperature,
                upperWallTemperature
            );

            // ------------------------------------------------
            // Save VTI results
            // ------------------------------------------------

            if (step == 1 || step % outputEvery == 0)
            {
                writeVTI(memory, step);

                std::cout << "Step " << step << " / " << steps
                          << " | VTI saved\n";
            }

            // ------------------------------------------------
            // Pull streaming + thermal BGK collision
            // ------------------------------------------------

            launchThermalBGK(
                memory,
                dynamicViscosityReference,
                accelerationX,
                accelerationY
            );

            // ------------------------------------------------
            // Swap A and B
            // ------------------------------------------------

            swapMemory(memory);
        }

        // ====================================================
        // Synchronize GPU
        // ====================================================

        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Performance measurements
        // ====================================================

        const auto finish = std::chrono::steady_clock::now();

        const double elapsed = std::chrono::duration<double>(finish - start).count();

        const double mlups = static_cast<double>(Nphysical) *
                             static_cast<double>(steps) /
                             (elapsed * 1.0e6);

        std::cout << "\n";
        std::cout << "============================================\n";
        std::cout << "              Simulation finished           \n";
        std::cout << "============================================\n";

        std::cout << "Elapsed time (s) = " << elapsed << "\n";
        std::cout << "MLUPS            = " << mlups << "\n";
        std::cout << "Results folder   = " << vtiOutputDirectory().generic_string() << "/\n";

        // ====================================================
        // Free GPU memory
        // ====================================================

        freeMemory(memory);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "\nTBGK error: " << error.what() << "\n";

        freeMemory(memory);

        return EXIT_FAILURE;
    }
}
