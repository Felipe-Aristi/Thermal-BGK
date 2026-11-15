// From the project root:
// nvcc -O2 -std=c++20 -arch=sm_89 --expt-extended-lambda tests/initialization.cu -o /tmp/tbgk_initialization_test
// /tmp/tbgk_initialization_test
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/launch.cuh"

int main()
{
    Memory memory{};
    try
    {
        allocateMemory(memory);
        launchRBInitialization(memory, rho0, referenceTemperature,
                               temperatureDifference, polytropicIndex,
                               0.0, accelerationY);
        launchRBDiffuseBoundary(memory, lowerWallTemperature, upperWallTemperature);

        std::vector<real_t> populations(Npopulations);
        CUDA_CHECK(cudaMemcpy(populations.data(), memory.A, bytesPopulations,
                              cudaMemcpyDeviceToHost));

        real_t maxSpeed = 0.0;
        for (label_t y = Domain::bulkYBegin; y <= Domain::bulkYEnd; ++y)
        {
            for (label_t x = 0; x < NXPhysical; ++x)
            {
                real_t rho = 0.0, jx = 0.0, jy = 0.0;
                for (label_t i = 0; i < D2V37::Q; ++i)
                {
                    int xs = static_cast<int>(x) - D2V37::cxTable[i];
                    xs = (xs + NXPhysical) % NXPhysical;
                    const int ys = static_cast<int>(y) - D2V37::cyTable[i];
                    const real_t f = populations[i * Ncells + idx(xs, ys)];
                    rho += f;
                    jx += D2V37::cxTable[i] * f;
                    jy += D2V37::cyTable[i] * f;
                }
                const real_t speed = std::hypot(jx / rho + 0.5 * accelerationX,
                                                jy / rho + 0.5 * accelerationY);
                if (!std::isfinite(speed))
                    throw std::runtime_error("Non-finite initial velocity");
                maxSpeed = std::max(maxSpeed, speed);
            }
        }
        freeMemory(memory);

        // Allow hydrostatic spatial-discretization error, but reject an
        // erroneous startup velocity of order one gravity timestep.
        const real_t tolerance = 1.0e-3 * std::abs(accelerationY);
        std::cout << "First reconstructed bulk max speed: " << maxSpeed
                  << " (tolerance " << tolerance << ")\n";
        if (maxSpeed > tolerance)
            return EXIT_FAILURE;
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        freeMemory(memory);
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
