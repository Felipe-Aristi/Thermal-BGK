
#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <cstddef>

#include "utilities/types.cuh"
#include "utilities/bound.cuh"
#include "../stencil/d2v37.cuh"

// ============================================================
// Simulation configuration
// ============================================================

inline constexpr label_t steps = 150000;
inline constexpr label_t outputEvery = 1000;

// ============================================================
// CUDA configuration
// ============================================================

inline constexpr label_t blockX = 32;
inline constexpr label_t blockY = 8;

// ============================================================
// Physical domain
// ============================================================

inline constexpr label_t NXPhysical = 200;
inline constexpr label_t NYPhysical = 100;

// ============================================================
// Domain geometry
// ============================================================

using Domain = Bounds<NXPhysical, NYPhysical, D2V37>;

inline constexpr label_t NX = Domain::NX;
inline constexpr label_t NY = Domain::NY;

// ============================================================
// Memory sizes
// ============================================================

inline constexpr std::size_t Ncells = static_cast<std::size_t>(NX) * NY;
inline constexpr std::size_t Nphysical = static_cast<std::size_t>(NXPhysical) * NYPhysical;

inline constexpr std::size_t Npopulations = static_cast<std::size_t>(D2V37::Q) * Ncells;

inline constexpr std::size_t bytesPopulations = Npopulations * sizeof(real_t);
inline constexpr std::size_t bytesTotal = 2 * bytesPopulations;

// ============================================================
// Mathematical constants
// ============================================================

inline constexpr real_t pi = 3.1415926535897932384626433832795;

// ============================================================
// Reference thermodynamic state
//
// Ideal gas: p = rho * R * T
//
// Lattice units: R = 1
// ============================================================

inline constexpr real_t gasConstant = 1.0;

inline constexpr real_t rho0 = 1.0;
inline constexpr real_t referenceTemperature = D2V37::cs2;

inline constexpr real_t referencePressure = rho0 * gasConstant * referenceTemperature;

// ============================================================
// Heat capacities
//
// cv = D / 2
// cp = cv + R
// ============================================================

inline constexpr real_t cv = static_cast<real_t>(D2V37::D) * 0.5;
inline constexpr real_t cp = cv + gasConstant;

inline constexpr real_t heatCapacityRatio = cp / cv;

// ============================================================
// Prandtl number
//
// Thermal BGK: Pr = 1
// ============================================================

inline constexpr real_t prandtl = 1.0;

// ============================================================
// Rayleigh-Benard temperature configuration
//
// Lower wall: hot
// Upper wall: cold
//
// DeltaT / T0 = 0.6
// ============================================================

inline constexpr real_t temperatureDifferenceRatio = 0.6;

inline constexpr real_t temperatureDifference = temperatureDifferenceRatio * referenceTemperature;

inline constexpr real_t lowerWallTemperature = referenceTemperature + 0.5 * temperatureDifference;
inline constexpr real_t upperWallTemperature = referenceTemperature - 0.5 * temperatureDifference;

// ============================================================
// Initial thermal perturbation
//
// T' = epsilon * T0 * cos(kx*x) * sin(pi*eta)
//
// epsilon is dimensionless.
// ============================================================

inline constexpr real_t perturbationAmplitude = 1.0e-4; //1.0e-4

// ============================================================
// Hydrostatic polytropic equilibrium
//
// T(y) = T0 + DeltaT * (0.5 - eta)
//
// rho(y) = rho0 * (T(y) / T0)^m
//
// g = (m + 1) * DeltaT / H
// ============================================================

inline constexpr real_t polytropicIndex = -0.9806;

inline constexpr real_t gravity = (polytropicIndex + 1.0) * temperatureDifference / Domain::height;

// ============================================================
// External acceleration
//
// Gravity acts in the negative y direction.
// ============================================================

inline constexpr real_t accelerationX = 0.0;
inline constexpr real_t accelerationY = -gravity;

inline constexpr real_t accelerationSquared =  accelerationX * accelerationX + accelerationY * accelerationY;

// ============================================================
// Reference thermal expansion coefficient
//
// Ideal gas: beta = 1 / T0
// ============================================================

inline constexpr real_t betaReference = 1.0 / referenceTemperature;

// ============================================================
// Temperature gradients
//
// Imposed gradient: DeltaT / H
// Adiabatic gradient: g / cp
//
// Superadiabatic gradient:
//
// GammaSA = DeltaT / H - g / cp
// ============================================================

inline constexpr real_t temperatureGradient = temperatureDifference / Domain::height;

inline constexpr real_t adiabaticGradient = gravity / cp;

inline constexpr real_t superadiabaticGradient = temperatureGradient - adiabaticGradient;

inline constexpr real_t superadiabaticTemperatureDifference = temperatureDifference - gravity * Domain::height / cp;

inline constexpr real_t superadiabaticFraction = 1.0 - (polytropicIndex + 1.0) / cp;

// ============================================================
// Target superadiabatic Rayleigh number
//
// Ra = g * beta * DeltaT_SA * H^3 / (nu * alpha)
//
// alpha = nu / Pr
//
// Therefore:
//
// nu = sqrt(Pr * g * beta * DeltaT_SA * H^3 / Ra)
// ============================================================

inline constexpr real_t rayleigh = 1590.0;

// ============================================================
// Compile-time square root
//
// Newton iteration
// ============================================================

[[nodiscard]] constexpr real_t constexprSqrt(const real_t value) noexcept
{
    if (value <= 0.0) return 0.0;

    real_t root = value >= 1.0 ? value : 1.0;

    for (int i = 0; i < 64; ++i)
    {
        root = 0.5 * (root + value / root);
    }

    return root;
}

// ============================================================
// Reference transport coefficients
// ============================================================

inline constexpr real_t nuReference = constexprSqrt(
    prandtl * gravity * betaReference *
    superadiabaticTemperatureDifference *
    Domain::height * Domain::height * Domain::height / rayleigh
);

inline constexpr real_t alphaReference = nuReference / prandtl;

// ============================================================
// Dynamic viscosity and thermal conductivity
//
// mu = rho0 * nu0
//
// k = rho0 * cp * alpha0
//
// For Pr = 1:
// k = cp * mu
// ============================================================

inline constexpr real_t dynamicViscosityReference = rho0 * nuReference;

inline constexpr real_t thermalConductivityReference = rho0 * cp * alphaReference;

// ============================================================
// Reference BGK relaxation parameters
//
// nu0 = T0 * (tau0 - 0.5)
//
// These parameters are evaluated at rho0 and T0.
//
// During simulation:
//
// tau(x,y) = 0.5 + mu0 / (rho * TH)
// ============================================================

inline constexpr real_t tauReference = 0.5 + nuReference / referenceTemperature;

inline constexpr real_t omegaReference = 1.0 / tauReference;

// ============================================================
// Reference Rayleigh diagnostics
// ============================================================

inline constexpr real_t calculatedRayleigh =
    gravity * betaReference * superadiabaticTemperatureDifference *
    Domain::height * Domain::height * Domain::height /
    (nuReference * alphaReference);

inline constexpr real_t nominalRayleigh =
    gravity * betaReference * temperatureDifference *
    Domain::height * Domain::height * Domain::height /
    (nuReference * alphaReference);

// ============================================================
// Geometric parameters
// ============================================================

inline constexpr real_t aspectRatio = static_cast<real_t>(NXPhysical) / Domain::height;

#endif // CONSTANTS_CUH
