
#ifndef OUTPUT_CUH
#define OUTPUT_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.cuh"
#include "memory.cuh"
#include "utilities/indexing.cuh"
#include "utilities/cudaUtilities.cuh"

// ============================================================
// Save Rayleigh-Benard results in VTI format
//
// A contains post-collision populations.
//
// Before calling this function, the ghost populations
// in A must be updated using the wall boundary kernel.
//
// Macroscopic quantities are recovered from the incoming
// populations after pull streaming.
//
// Only physical fluid nodes are exported.
// ============================================================

inline std::string rayleighOutputTag()
{
    std::ostringstream tag;
    tag << std::setprecision(std::numeric_limits<real_t>::max_digits10) << rayleigh;
    return tag.str();
}

inline std::filesystem::path vtiOutputDirectory()
{
    return std::filesystem::path("results/rb") /
           ("NX_" + std::to_string(NXPhysical) + "_NY_" + std::to_string(NYPhysical)) /
           ("Ra_" + rayleighOutputTag()) / "vti";
}

inline void writeVTI(const Memory &memory, const label_t step)
{
    // ========================================================
    // Copy populations from GPU to CPU
    // ========================================================

    std::vector<real_t> populations(Npopulations);

    CUDA_CHECK(cudaMemcpy(populations.data(), memory.A, bytesPopulations, cudaMemcpyDeviceToHost));

    // ========================================================
    // Output fields
    // ========================================================

    std::vector<real_t> density(Nphysical);
    std::vector<real_t> temperature(Nphysical);
    std::vector<real_t> pressure(Nphysical);
    std::vector<real_t> ux(Nphysical);
    std::vector<real_t> uy(Nphysical);
    std::vector<real_t> speed(Nphysical);

    // ========================================================
    // Recover hydrodynamic quantities
    //
    // Output processing is performed on the CPU.
    // ========================================================

    for (label_t y = 0; y < NYPhysical; ++y)
    {
        for (label_t x = 0; x < NXPhysical; ++x)
        {
            const label_t allocatedY = y + Domain::fluidYBegin;
            const std::size_t id = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * NXPhysical;

            real_t rho = 0.0;
            real_t jx = 0.0;
            real_t jy = 0.0;
            real_t m2 = 0.0;

            for (label_t i = 0; i < D2V37::Q; ++i)
            {
                const int cx = D2V37::cxTable[i];
                const int cy = D2V37::cyTable[i];

                int xs = static_cast<int>(x) - cx;

                if (xs < 0) xs += static_cast<int>(NXPhysical);
                else if (xs >= static_cast<int>(NXPhysical)) xs -= static_cast<int>(NXPhysical);

                const label_t ys = static_cast<label_t>(static_cast<int>(allocatedY) - cy);

                const std::size_t source = static_cast<std::size_t>(i) * Ncells + idx(static_cast<label_t>(xs), ys);

                const real_t fi = populations[source];

                rho += fi;
                jx += static_cast<real_t>(cx) * fi;
                jy += static_cast<real_t>(cy) * fi;
                m2 += static_cast<real_t>(cx * cx + cy * cy) * fi;
            }

            const real_t uxL = jx / rho;
            const real_t uyL = jy / rho;

            const real_t TL = (m2 / rho - uxL * uxL - uyL * uyL) / static_cast<real_t>(D2V37::D);

            const real_t uxH = uxL + 0.5 * accelerationX;
            const real_t uyH = uyL + 0.5 * accelerationY;

            const real_t TH = TL + accelerationSquared / (4.0 * static_cast<real_t>(D2V37::D));

            density[id] = rho;
            temperature[id] = TH;
            pressure[id] = rho * TH;
            ux[id] = uxH;
            uy[id] = uyH;
            speed[id] = std::sqrt(uxH * uxH + uyH * uyH);
        }
    }

    // ========================================================
    // Create output folder
    // ========================================================

    const std::filesystem::path directory = vtiOutputDirectory();

    std::filesystem::create_directories(directory);

    const std::string filename = "rb_Ra_" + rayleighOutputTag() +
                                 "_NX_" + std::to_string(NXPhysical) +
                                 "_NY_" + std::to_string(NYPhysical) +
                                 "_" + [&]()
    {
        std::ostringstream name;
        name << std::setw(8) << std::setfill('0') << step;
        return name.str();
    }() + ".vti";

    const std::filesystem::path path = directory / filename;

    std::ofstream file(path);

    if (!file)
        throw std::runtime_error("Unable to create VTI output file.");

    file << std::setprecision(17);

    // ========================================================
    // VTK XML header
    //
    // First physical node is half a lattice unit above
    // the lower wall.
    // ========================================================

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

    file << "<ImageData WholeExtent=\"0 " << NXPhysical - 1
         << " 0 " << NYPhysical - 1
         << " 0 0\" Origin=\"0 0.5 0\" Spacing=\"1 1 1\">\n";

    file << "<Piece Extent=\"0 " << NXPhysical - 1
         << " 0 " << NYPhysical - 1
         << " 0 0\">\n";

    file << "<PointData Scalars=\"Temperature\" Vectors=\"Velocity\">\n";

    // ========================================================
    // Scalar field writer
    // ========================================================

    const auto writeScalar = [&](const char *name, const std::vector<real_t> &field)
    {
        file << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";

        for (const real_t value : field)
            file << value << '\n';

        file << "</DataArray>\n";
    };

    writeScalar("Density", density);
    writeScalar("Temperature", temperature);
    writeScalar("Pressure", pressure);
    writeScalar("Speed", speed);

    // ========================================================
    // Velocity vector
    // ========================================================

    file << "<DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (std::size_t id = 0; id < Nphysical; ++id)
        file << ux[id] << ' ' << uy[id] << " 0\n";

    file << "</DataArray>\n";

    // ========================================================
    // Close VTK XML
    // ========================================================

    file << "</PointData>\n";
    file << "<CellData/>\n";
    file << "</Piece>\n";
    file << "</ImageData>\n";
    file << "</VTKFile>\n";
}

#endif // OUTPUT_CUH
