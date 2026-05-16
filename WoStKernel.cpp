// ============================================================================
// WoStKernel.cpp
//
// Implementation of the Walk on Stars Monte Carlo PDE kernel.
// See WoStKernel.hpp for the mathematical background.
// ============================================================================

#include "WoStKernel.hpp"

#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>

namespace wost {

// ---------------------------------------------------------------------------
// Local math helpers (keep these file-local; the backend has its own copies)
// ---------------------------------------------------------------------------
namespace {

inline float dot3(const vec3& a, const vec3& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline vec3 scale3(const vec3& a, float s) noexcept {
    return { a.x * s, a.y * s, a.z * s };
}
// x + s * d
inline vec3 madd(const vec3& x, const vec3& d, float s) noexcept {
    return { x.x + d.x * s, x.y + d.y * s, x.z + d.z * s };
}
// reflect incident direction `d` about outward normal `n`
inline vec3 reflect(const vec3& d, const vec3& n) noexcept {
    float dn = dot3(d, n);
    return { d.x - 2.f * dn * n.x,
             d.y - 2.f * dn * n.y,
             d.z - 2.f * dn * n.z };
}

// -----------------------------------------------------------------------
// Assemble a BoundaryPoint from a successful IntersectRay result.
// -----------------------------------------------------------------------
inline BoundaryPoint makeBP(const vec3& origin, const vec3& dir,
                             float t, const vec3& normal, uint32_t prim) {
    BoundaryPoint bp;
    bp.position = madd(origin, dir, t);
    bp.normal   = normal;
    bp.dist     = t;
    bp.triIdx   = prim;
    return bp;
}

// -----------------------------------------------------------------------
// Finalise statistics from accumulated sums.
// -----------------------------------------------------------------------
inline void finalise(WalkResult& r, double sumV, double sumV2,
                     int sumSteps, int N) {
    r.value     = static_cast<float>(sumV / N);
    r.meanSteps = static_cast<float>(sumSteps) / static_cast<float>(N);

    double mean  = sumV / N;
    double var   = std::max(0.0, sumV2 / N - mean * mean);
    double sigma = std::sqrt(var);                     // population std-dev
    r.stdErr     = static_cast<float>(sigma / std::sqrt(static_cast<double>(N)));
}

} // anonymous namespace

// ===========================================================================
// WoStKernel constructor
// ===========================================================================
WoStKernel::WoStKernel(const WoStGeometryBackend& geo) : geo_(geo) {}

// ===========================================================================
// Internal walk-step dispatcher
//
// The three public solvers share the same stepping loop; they differ only in
// what they do when the walk hits a boundary or crosses the epsilon shell.
// We inline the logic directly into each solver to keep the hot path clear.
// ===========================================================================


// ===========================================================================
// (1) SolveLaplace  –  Δu = 0,  pure Dirichlet
//
// Walk until the walk exits Ω or the star radius falls below eps.
// No source accumulation needed.
// ===========================================================================
WalkResult WoStKernel::SolveLaplace(const vec3&        x0,
                                     const DirichletFn& g,
                                     const WoStParams&  params) const
{
    WalkResult result;
    double sumV = 0.0, sumV2 = 0.0;
    int    sumSteps = 0;

    for (int s = 0; s < params.numSamples; ++s) {
        // Per-sample stream: combine base seed with sample index so each
        // walk is independently seeded (no correlation between samples).
        PCG32 rng(params.seed, static_cast<uint64_t>(s));

        vec3  x    = x0;
        float val  = 0.f;
        int   steps = 0;
        bool  done  = false;

        for (int step = 0; step < params.maxSteps; ++step) {
            ++steps;

            // ── Query geometry ───────────────────────────────────────────
            BoundaryPoint bndBP, silBP;
            float R = geo_.StarRadius(x, bndBP, silBP);

            // ── Absorption shell ─────────────────────────────────────────
            //   If the star ball is tiny we are essentially on ∂Ω.
            //   Use the closest Dirichlet boundary point as the terminus.
            if (R < params.eps) {
                val  = g(bndBP);
                done = true;
                break;
            }

            // ── Sample a point on S(x, R) ────────────────────────────────
            vec3 dir = sampleUnitSphere(rng);

            // ── Ray-trace up to star radius R ────────────────────────────
            float t;
            vec3 hitN;
            uint32_t prim;
            bool hit = geo_.IntersectRay(x, dir, R, t, hitN, prim);

            if (hit) {
                // We hit the boundary before reaching R
                val = g(makeBP(x, dir, t, hitN, prim));
                done = true;
                break;
            } else {
                // Free path - no boundary intersected, take full step
                x = madd(x, dir, R); 
            }
        } // walk steps

        if (!done) {
            // maxSteps reached: use closest boundary as best estimate
            BoundaryPoint bp;
            geo_.ClosestPoint(x, bp);
            val               = g(bp);
            result.anyDiverged = true;
        }

        sumV  += val;
        sumV2 += static_cast<double>(val) * val;
        sumSteps += steps;
    }

    finalise(result, sumV, sumV2, sumSteps, params.numSamples);
    return result;
}

// ===========================================================================
// (2) SolvePoisson  –  Δu = f,  pure Dirichlet
//
// At each walk step, accumulate the volume source contribution via the
// 3-D mean-value formula:
//
//   u(x) = E[u(y)] − (R²/6) · f(x)   (on sphere of radius R, in 3D)
//
// So the *path estimator* accumulates:
//   value ← g(y_terminal)  +  Σ_k [ −(R_k²/6) · f(x_k) ]
//
// This is exact for constant f; higher-order corrections vanish for
// slowly varying f under refinement.
// ===========================================================================
WalkResult WoStKernel::SolvePoisson(const vec3&        x0,
                                     const DirichletFn& g,
                                     const SourceFn&    f,
                                     const WoStParams&  params) const
{
    WalkResult result;
    double sumV = 0.0, sumV2 = 0.0;
    int    sumSteps = 0;

    for (int s = 0; s < params.numSamples; ++s) {
        PCG32 rng(params.seed, static_cast<uint64_t>(s));

        vec3  x    = x0;
        float acc  = 0.f;  // accumulated walk value
        int   steps = 0;
        bool  done  = false;

        for (int step = 0; step < params.maxSteps; ++step) {
            ++steps;

            BoundaryPoint bndBP, silBP;
            float R = geo_.StarRadius(x, bndBP, silBP);

            // ── Absorption shell ─────────────────────────────────────────
            if (R < params.eps) {
                acc += g(bndBP);
                done = true;
                break;
            }

            // ── Walk step ────────────────────────────────────────────────
            vec3 dir = sampleUnitSphere(rng);

            // Ray-trace up to star radius R
            float t;
            vec3 hitN;
            uint32_t prim;
            bool hit = geo_.IntersectRay(x, dir, R, t, hitN, prim);

            // If we hit, the step length is t. Otherwise, it's R.
            float stepDist = hit ? t : R;

            // Correct star-shaped mean-value estimator for 3D
            acc -= (stepDist * stepDist / 6.f) * f(x);

            if (hit) {
                // We hit the boundary before reaching R
                acc += g(makeBP(x, dir, t, hitN, prim));
                done = true;
                break;
            } else {
                // Free path - no boundary intersected, take full step
                x = madd(x, dir, R);
            }
        }

        if (!done) {
            BoundaryPoint bp;
            geo_.ClosestPoint(x, bp);
            acc               += g(bp);
            result.anyDiverged = true;
        }

        sumV  += acc;
        sumV2 += static_cast<double>(acc) * acc;
        sumSteps += steps;
    }

    finalise(result, sumV, sumV2, sumSteps, params.numSamples);
    return result;
}

// ===========================================================================
// (3) SolveMixed  –  Δu = 0,  mixed Dirichlet / Neumann BCs
//
// Boundary behaviour at a hit point p:
//
//   Dirichlet (isNeumann(p) == false):
//       Accumulate g(p), terminate.  Same as pure-Dirichlet kernel.
//
//   Neumann   (isNeumann(p) == true):
//       Accumulate contribution  h(p) · t  where t is the distance
//       travelled to the boundary (= the "last free-path length").
//       This weights h by the same harmonic measure that WoSt assigns to
//       the Neumann term (Sawhney et al. 2023, §4).
//       Then reflect the walk direction about the outward normal and
//       restart from just inside the reflection point.
//
// Walk terminates when it hits a Dirichlet boundary or exhausts maxSteps.
// ===========================================================================
WalkResult WoStKernel::SolveMixed(const vec3&           x0,
                                   const DirichletFn&    g,
                                   const NeumannFn&      h,
                                   const NeumannPredFn&  isNeumann,
                                   const WoStParams&     params) const
{
    WalkResult result;
    double sumV = 0.0, sumV2 = 0.0;
    int    sumSteps = 0;

    for (int s = 0; s < params.numSamples; ++s) {
        PCG32 rng(params.seed, static_cast<uint64_t>(s));

        vec3  x    = x0;
        float acc  = 0.f;
        int   steps = 0;
        bool  done  = false;

        for (int step = 0; step < params.maxSteps; ++step) {
            ++steps;

            BoundaryPoint bndBP, silBP;
            float R = geo_.StarRadius(x, bndBP, silBP);

            // ── Absorption shell ─────────────────────────────────────────
            if (R < params.eps) {
                if (isNeumann(bndBP)) {
                    // On ΓN: accumulate Neumann term and reflect inward
                    acc += h(bndBP) * R;
                    // Sample a direction in the inward hemisphere
                    vec3 n = bndBP.normal;
                    vec3 d = sampleUnitSphere(rng);
                    if (dot3(d, n) > 0.f) d = scale3(d, -1.f);  // flip outward
                    x = madd(bndBP.position, d, params.eps * 2.f);
                    continue;  // restart walk from just inside ΓN
                } else {
                    // On ΓD: terminate
                    acc += g(bndBP);
                    done = true;
                    break;
                }
            }

            // ── Walk step ────────────────────────────────────────────────
            vec3 dir = sampleUnitSphere(rng);

            // Ray-trace up to star radius R
            float t;
            vec3 hitN;
            uint32_t prim;
            bool hit = geo_.IntersectRay(x, dir, R, t, hitN, prim);

            if (hit) {
                // We hit the boundary before reaching R
                BoundaryPoint bp = makeBP(x, dir, t, hitN, prim);

                if (isNeumann(bp)) {
                    // ── Neumann: accumulate + reflect ────────────────────
                    acc += h(bp) * t;

                    // Reflect direction about outward normal
                    vec3 n    = bp.normal;
                    vec3 refl = reflect(dir, n);
                    // Restart from just past the reflection point (inward side)
                    x = madd(bp.position, refl, params.eps * 2.f);
                } else {
                    // ── Dirichlet: accumulate + terminate ────────────────
                    acc += g(bp);
                    done = true;
                    break;
                }
            } else {
                // Free path - no boundary intersected, take full step
                x = madd(x, dir, R);
            }
        } // walk steps

        if (!done) {
            BoundaryPoint bp;
            geo_.ClosestPoint(x, bp);
            if (!isNeumann(bp)) acc += g(bp);
            result.anyDiverged = true;
        }

        sumV  += acc;
        sumV2 += static_cast<double>(acc) * acc;
        sumSteps += steps;
    }

    finalise(result, sumV, sumV2, sumSteps, params.numSamples);
    return result;
}

} // namespace wost