// ============================================================================
// WoStKernel.hpp
//
// 3-D Walk on Stars (WoSt) Monte Carlo PDE kernel.
//
// Solves three equation variants on a closed 3-D domain Ω described
// by a WoStGeometryBackend:
//
//   (1)  Δu = 0          pure Dirichlet   u = g  on ∂Ω
//   (2)  Δu = f          Dirichlet + volume source (Poisson)
//   (3)  Δu = 0          mixed  u = g on ΓD,  ∂u/∂n = h on ΓN
//
// Reference: Sawhney, Seyb, Jarosz, Crane – "Walk on Stars", SIGGRAPH 2023.
//
// Algorithm overview
// ------------------
//   At each walk step from x:
//     R* ← StarRadius(x)          (min of dist-to-∂Ω and dist-to-silhouette)
//     d  ← uniform unit vector
//     y  ← x + R* · d
//
//     if IsInside(y)  →  x ← y   (take the step)
//     else            →  ray-cast x→d to find boundary hit p
//                        Dirichlet hit: accumulate g(p), terminate
//                        Neumann hit:   accumulate h(p)·t, reflect, continue
//
//   Poisson source term: at each step add f(x)·R*²/6  (3-D sphere estimator)
//                        (see § "Mean-value formula" below)
//
// Mean-value formula (3-D)
// -------------------------
//   u(x) = (1/|∂B|) ∫_{∂B(x,R)} u dσ  −  (R²/6) · Δu(x)
//   →  for Δu = f:
//          u(x) = E[u(y)] − (R²/6)·f(x)
//   Source functor f should return the RHS of Δu = f (sign included by user).
// ============================================================================
#pragma once

#include "WoStGeometryBackend.hpp"
#include <functional>
#include <cstdint>

namespace wost {

// ===========================================================================
// Callback types
// ===========================================================================

/// Dirichlet BC:  boundary-point → scalar value g(p)
using DirichletFn   = std::function<float(const BoundaryPoint&)>;

/// Neumann BC:    boundary-point → outward normal derivative h(p)
using NeumannFn     = std::function<float(const BoundaryPoint&)>;

/// Volume source: interior point  → f(x)   (right-hand side of Δu = f)
using SourceFn      = std::function<float(const vec3&)>;

/// Predicate – returns true if a boundary point lies on the Neumann region ΓN
using NeumannPredFn = std::function<bool(const BoundaryPoint&)>;

// ===========================================================================
// Walk parameters
// ===========================================================================
struct WoStParams {
    int      numSamples = 256;   ///< independent random walks per query point
    int      maxSteps   = 512;   ///< safety cap on walk length
    float    eps        = 1e-4f; ///< absorption radius near ∂Ω
    uint64_t seed       = 0xDEADBEEF; ///< base RNG seed (varied per sample)
};

// ===========================================================================
// Per-query result
// ===========================================================================
struct WalkResult {
    float value       = 0.f;  ///< Monte Carlo estimate  E[u(x)]
    float stdErr      = 0.f;  ///< standard error  σ / sqrt(N)
    float meanSteps   = 0.f;  ///< average walk length in steps
    bool  anyDiverged = false;///< true if any walk hit maxSteps limit
};

// ===========================================================================
// PCG32 – tiny, fast, good-quality PRNG
//   O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good
//   Algorithms for Random Number Generation", 2014.
// ===========================================================================
struct PCG32 {
    uint64_t state = 0;
    uint64_t inc   = 1;

    /// Seed with (initState, sequence_id).  Two generators with different
    /// sequence IDs produce independent streams.
    PCG32(uint64_t initState, uint64_t initSeq = 1) {
        inc   = (initSeq << 1u) | 1u;
        state = 0u;
        next();
        state += initState;
        next();
    }

    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot        = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
    }

    /// Uniform float in [0, 1)
    float nextF() { return static_cast<float>(next() >> 8) * (1.f / (1u << 24)); }
};

// ===========================================================================
// Utility – uniform direction on S² via rejection sampling (Marsaglia 1972).
// E[cost] = 6/π ≈ 1.91 uniform samples per call.
// ===========================================================================
inline vec3 sampleUnitSphere(PCG32& rng) {
    float x, y, z, r2;
    do {
        x  = 2.f * rng.nextF() - 1.f;
        y  = 2.f * rng.nextF() - 1.f;
        z  = 2.f * rng.nextF() - 1.f;
        r2 = x*x + y*y + z*z;
    } while (r2 < 1e-12f || r2 > 1.f);
    float inv = 1.f / std::sqrt(r2);
    return { x * inv, y * inv, z * inv };
}

// ===========================================================================
// WoStKernel
// ===========================================================================
class WoStKernel {
public:
    explicit WoStKernel(const WoStGeometryBackend& geo);

    // -----------------------------------------------------------------------
    // (1) Laplace  Δu = 0,  u = g on ∂Ω
    // -----------------------------------------------------------------------
    WalkResult SolveLaplace(const vec3&        x,
                            const DirichletFn& g,
                            const WoStParams&  p = {}) const;

    // -----------------------------------------------------------------------
    // (2) Poisson  Δu = f,  u = g on ∂Ω
    //     Source term:  f(x) contributes −R²/6 · f(x) per step (3-D formula).
    //     Pass f = [](auto&){ return 0.f; } to recover Laplace.
    // -----------------------------------------------------------------------
    WalkResult SolvePoisson(const vec3&        x,
                            const DirichletFn& g,
                            const SourceFn&    f,
                            const WoStParams&  p = {}) const;

    // -----------------------------------------------------------------------
    // (3) Mixed BCs  Δu = 0,  u = g on ΓD,  ∂u/∂n = h on ΓN
    //     'isNeumann' classifies boundary hits; false → Dirichlet.
    //     Set h = [](auto&){ return 0.f; } for a zero-flux (insulating) ΓN.
    // -----------------------------------------------------------------------
    WalkResult SolveMixed(const vec3&           x,
                          const DirichletFn&    g,
                          const NeumannFn&      h,
                          const NeumannPredFn&  isNeumann,
                          const WoStParams&     p = {}) const;

private:
    const WoStGeometryBackend& geo_;
};

} // namespace wost