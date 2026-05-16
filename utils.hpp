
#include "WoStKernel.hpp"
#include <fstream>
#include <iostream>

using namespace wost;

// ---------------------------------------------------------------------------
// Local math helpers
// ---------------------------------------------------------------------------
namespace {

inline float dot3(const vec3& a, const vec3& b) noexcept {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline vec3 madd(const vec3& x, const vec3& d, float s) noexcept {
    return { x.x + d.x*s, x.y + d.y*s, x.z + d.z*s };
}
inline vec3 reflect(const vec3& d, const vec3& n) noexcept {
    float dn = dot3(d, n);
    return { d.x - 2.f*dn*n.x, d.y - 2.f*dn*n.y, d.z - 2.f*dn*n.z };
}
inline BoundaryPoint makeBP(const vec3& o, const vec3& d,
                             float t, const vec3& n, uint32_t prim) noexcept
{
    BoundaryPoint bp;
    bp.position = madd(o, d, t);
    bp.normal   = n;
    bp.dist     = t;
    bp.triIdx   = prim;
    return bp;
}
inline void finalise(WalkResult& r, double sumV, double sumV2,
                     int sumSteps, int N) noexcept
{
    r.value     = static_cast<float>(sumV / N);
    r.meanSteps = static_cast<float>(sumSteps) / N;
    double mean = sumV / N;
    double var  = std::max(0.0, sumV2/N - mean*mean);
    r.stdErr    = static_cast<float>(std::sqrt(var / N));
}

} // anonymous namespace

// ===========================================================================
// SolveLaplaceNeumann
//
// WoSt estimator for  Δu = 0,  u = g on ΓD,  ∂u/∂n = h on ΓN.
//
// Walk logic:
//   • Compute star radius R at current position x.
//   • If R < eps (we are in the absorption shell on ∂Ω):
//       – Neumann hit → accumulate h(p)·R, bounce inward, continue.
//       – Dirichlet hit → accumulate g(p), terminate.
//   • Sample a random direction d; step to y = x + R·d.
//   • If y is inside Ω → advance the walk.
//   • If y is outside Ω → ray-cast x→d to find boundary point p:
//       – Neumann hit → accumulate h(p)·t, reflect about p.normal, continue.
//       – Dirichlet hit → accumulate g(p), terminate.
// ===========================================================================
static WalkResult SolveLaplaceNeumann(
        const WoStGeometryBackend& geo,
        const vec3&                x0,
        const DirichletFn&         g,
        const NeumannFn&           h,
        const NeumannPredFn&       isNeumann,
        const WoStParams&          params)
{
    WalkResult result;
    double sumV = 0.0, sumV2 = 0.0;
    int    sumSteps = 0;

    const float eps = params.eps;

    for (int s = 0; s < params.numSamples; ++s) {
        // Each sample uses an independent PCG32 stream (different 'inc').
        PCG32 rng(params.seed, static_cast<uint64_t>(s) + 1u);

        vec3  x     = x0;
        float acc   = 0.f;
        int   steps = 0;
        bool  done  = false;

        for (int step = 0; step < params.maxSteps; ++step) {
            ++steps;

            // ── Star-shaped ball radius at x ──────────────────────────────
            BoundaryPoint bndBP, silBP;
            float R = geo.StarRadius(x, bndBP, silBP);

            // ── Absorption shell ──────────────────────────────────────────
            if (R < eps) {
                if (isNeumann(bndBP)) {
                    // Accumulate Neumann contribution weighted by shell radius.
                    acc += h(bndBP) * R;
                    // Sample a direction in the inward hemisphere and continue.
                    vec3 n = bndBP.normal;
                    vec3 d = sampleUnitSphere(rng);
                    if (dot3(d, n) > 0.f) { d.x=-d.x; d.y=-d.y; d.z=-d.z; }
                    x = madd(bndBP.position, d, eps * 2.f);
                    continue;
                } else {
                    // Dirichlet shell: absorb.
                    acc += g(bndBP);
                    done = true;
                    break;
                }
            }

            // ── Random step on ∂B(x, R) ───────────────────────────────────
            vec3 dir = sampleUnitSphere(rng);
            vec3 y   = madd(x, dir, R);

            if (geo.IsInside(y)) {
                x = y;   // valid interior step
            } else {
                // Overstepped the boundary: ray-cast to get the exact hit.
                float    t;
                vec3     hitN;
                uint32_t prim;
                bool hit = geo.IntersectRay(x, dir, R * 1.1f, t, hitN, prim);

                BoundaryPoint bp = hit ? makeBP(x, dir, t, hitN, prim) : bndBP;
                if (!hit) t = R;

                if (isNeumann(bp)) {
                    // Accumulate h(p)·t and reflect the walk direction.
                    acc += h(bp) * t;
                    vec3 refl = reflect(dir, bp.normal);
                    x = madd(bp.position, refl, eps * 2.f);
                } else {
                    // Dirichlet: accumulate and terminate.
                    acc += g(bp);
                    done = true;
                    break;
                }
            }
        } // walk steps

        if (!done) {
            // Safety: walk exhausted maxSteps without terminating.
            BoundaryPoint bp;
            geo.ClosestPoint(x, bp);
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

// ===========================================================================
// WriteVTK
//
// Writes a legacy ASCII VTK structured-points file.  Exterior voxels have
// NaN in solution/exact so ParaView automatically masks them; the "inside"
// scalar (0/1) can be used with the Threshold filter for explicit masking.
// ===========================================================================
static void WriteVTK(const std::string&        filename,
                     int nx, int ny, int nz,
                     float ox, float oy, float oz,
                     float dx, float dy, float dz,
                     const std::vector<float>& solution,
                     const std::vector<float>& exact,
                     const std::vector<float>& abs_err,
                     const std::vector<float>& std_err,
                     const std::vector<float>& inside)
{
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "ERROR: cannot open " << filename << "\n";
        return;
    }

    const int N = nx * ny * nz;

    f << "# vtk DataFile Version 3.0\n"
      << "WoSt Laplace+Neumann on spot_triangulated\n"
      << "ASCII\n"
      << "DATASET STRUCTURED_POINTS\n"
      << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n"
      << "ORIGIN "     << ox << " " << oy << " " << oz << "\n"
      << "SPACING "    << dx << " " << dy << " " << dz << "\n"
      << "\nPOINT_DATA " << N << "\n";

    auto writeScalar = [&](const char* name, const std::vector<float>& v) {
        f << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < N; ++i) f << v[i] << "\n";
    };

    writeScalar("solution",  solution);
    writeScalar("exact",     exact);
    writeScalar("abs_error", abs_err);
    writeScalar("std_error", std_err);
    writeScalar("inside",    inside);

    std::cout << "Wrote " << filename
              << "  (" << nx << "×" << ny << "×" << nz
              << " = " << N << " voxels)\n";
}
