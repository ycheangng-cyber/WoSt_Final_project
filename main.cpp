// ============================================================================
// wost_test.cpp
//
// Walk on Stars – solution field test on spot_triangulated.obj
//
// Problem (manufactured, exactly verifiable):
//
//   Δu = 0  in Ω  (Laplace, steady-state heat conduction)
//
//   Neumann BC  ∂u/∂n = h  on ΓN = { p ∈ ∂Ω : p.y >= Y_DIRICHLET_THRESH }
//       h(p) = p.normal.y          ← y-component of the outward unit normal
//
//   Dirichlet BC  u = g  on ΓD = { p ∈ ∂Ω : p.y <  Y_DIRICHLET_THRESH }
//       g(p) = p.position.y        ← fixes the hooves / feet of Spot
//
//   Exact solution:  u(x) = x.y   for all x ∈ Ω
//
//   Verification:
//     Δ(y) = 0                      ✓   (linear functions are harmonic)
//     y|_{ΓD} = p.y                 ✓   (Dirichlet satisfied exactly)
//     ∂y/∂n = ∇y · n = ê_y · n = ny ✓  (Neumann flux = ny)
//
// Why a Dirichlet anchor is needed
// ---------------------------------
//   Pure Neumann Laplace is uniquely solvable only up to an additive
//   constant.  In WoSt terms the walk has no absorbing boundary and runs
//   until maxSteps.  Pinning a small patch ΓD (the hooves, y < threshold)
//   makes the walk terminate while leaving the entire body surface as Neumann.
//
// Output:  wost_solution.vtk – STRUCTURED_POINTS with five scalar arrays:
//   solution   Monte Carlo estimate of u(x)
//   exact      analytic value x.y
//   abs_error  |solution − exact|
//   std_error  σ / √N  (Monte Carlo standard error)
//   inside     1 = interior voxel, 0 = exterior
//
//   Open in ParaView or VisIt.  Apply Threshold (inside == 1) to mask
//   exterior voxels, then colour by "solution" or "abs_error".
//
// Build
// -----
//   g++ -O2 -std=c++17 -I. \
//       wost_test.cpp WoStGeometryBackend.cpp WoStKernel.cpp \
//       -o wost_test
//   ./wost_test [path/to/spot_triangulated.obj]
// ============================================================================

#include "src/WoStKernel.hpp"   // pulls in WoStGeometryBackend.hpp + tiny_bvh.h
#include "src/utils.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <limits>

using namespace wost;

// ---------------------------------------------------------------------------
// ── Tunable parameters ──────────────────────────────────────────────────────
// ---------------------------------------------------------------------------

// Grid resolution (NX × NY × NZ cells, each centred in the mesh AABB+PAD).
static constexpr int   GRID_NX  = 20;
static constexpr int   GRID_NY  = 25;
static constexpr int   GRID_NZ  = 20;
static constexpr float GRID_PAD = 0.04f;   // world-space padding around AABB

// Monte Carlo quality
static constexpr int      NUM_SAMPLES = 64;    // walks per interior voxel
static constexpr int      MAX_STEPS   = 1024;  // safety cap on walk length
static constexpr float    EPS_SHELL   = 3e-4f; // absorption radius near ∂Ω
static constexpr uint64_t RNG_SEED    = 0xBADC0FFEULL;

// Boundary split  (see problem description above)
// Spot's y range is ~ [-0.74, +0.95].
// y < Y_DIRICHLET_THRESH → Dirichlet anchor (hooves/lower legs).
// y ≥ Y_DIRICHLET_THRESH → Neumann  (the entire body of Spot).
static constexpr float Y_DIRICHLET_THRESH = -0.30f;

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char* argv[])
{
    const std::string objPath = (argc > 1) ? argv[1] : "spot/spot_triangulated.obj";

    std::cout
        << "=================================================================\n"
        << "  Walk on Stars – Laplace + Neumann BCs\n"
        << "  Mesh: " << objPath << "\n"
        << "=================================================================\n\n";

    // ── Build geometry backend ────────────────────────────────────────────────
    std::cout << "Loading mesh ...\n";
    WoStGeometryBackend geo(objPath);

    vec3 bmin, bmax;
    geo.MeshBounds(bmin, bmax);
    std::cout << "  Triangles  : " << geo.TriangleCount() << "\n"
              << "  AABB       : x[" << bmin.x << ", " << bmax.x << "]"
              <<              "  y[" << bmin.y << ", " << bmax.y << "]"
              <<              "  z[" << bmin.z << ", " << bmax.z << "]\n\n";

    // ── Boundary conditions ───────────────────────────────────────────────────
    //   isNeumann : true  for the body of Spot (y >= Y_DIRICHLET_THRESH)
    //               false for the hooves/feet  (y <  Y_DIRICHLET_THRESH)
    //
    //   The manufactured solution u = y satisfies:
    //     Δu = 0,  u|_ΓD = p.y,  ∂u/∂n|_ΓN = (0,1,0)·n = ny
    //
    NeumannPredFn isNeumann = [](const BoundaryPoint& bp) -> bool {
        return bp.position.y >= Y_DIRICHLET_THRESH;
    };
    DirichletFn g = [](const BoundaryPoint& bp) -> float {
        return bp.position.y;   // u = y at the Dirichlet anchor
    };
    NeumannFn h = [](const BoundaryPoint& bp) -> float {
        return bp.normal.y;     // ∂u/∂n = ny  (matches exact solution u=y)
    };

    auto exact_fn = [](const vec3& x) -> float { return x.y; };

    // ── WoSt parameters ───────────────────────────────────────────────────────
    WoStParams params;
    params.numSamples = NUM_SAMPLES;
    params.maxSteps   = MAX_STEPS;
    params.eps        = EPS_SHELL;
    params.seed       = RNG_SEED;

    std::cout
        << "Problem\n"
        << "  PDE          : Δu = 0  (Laplace)\n"
        << "  Neumann ΓN   : y >= " << Y_DIRICHLET_THRESH
        <<                 "   →  ∂u/∂n = ny   (body of Spot)\n"
        << "  Dirichlet ΓD : y <  " << Y_DIRICHLET_THRESH
        <<                 "   →  u = y         (feet anchor)\n"
        << "  Exact u(x)   : x.y\n"
        << "  Grid         : " << GRID_NX<<"x"<<GRID_NY<<"x"<<GRID_NZ << "\n"
        << "  Samples      : " << NUM_SAMPLES << "  max_steps=" << MAX_STEPS << "\n\n";

    // ── Grid setup ────────────────────────────────────────────────────────────
    const float gox = bmin.x - GRID_PAD,  gex = bmax.x + GRID_PAD;
    const float goy = bmin.y - GRID_PAD,  gey = bmax.y + GRID_PAD;
    const float goz = bmin.z - GRID_PAD,  gez = bmax.z + GRID_PAD;

    const float dx = (gex - gox) / GRID_NX;
    const float dy = (gey - goy) / GRID_NY;
    const float dz = (gez - goz) / GRID_NZ;

    const int N = GRID_NX * GRID_NY * GRID_NZ;

    std::vector<float> sol (N, 0.f), exc (N, 0.f), aerr(N, 0.f),
                       serr(N, 0.f), ins (N, 0.f);

 // ── Solve on grid ────────────────────────────────────────────────────────────
std::cout << "Solving...\n";

auto t_start = std::chrono::steady_clock::now();

int    numInside    = 0;
int    numDiverged  = 0;
double sumAbsErr    = 0.0;
double sumMeanSteps = 0.0;

#pragma omp parallel for collapse(3) schedule(dynamic,4) \
    reduction(+:numInside,numDiverged,sumAbsErr,sumMeanSteps)
for (int iz = 0; iz < GRID_NZ; ++iz)
for (int iy = 0; iy < GRID_NY; ++iy)
for (int ix = 0; ix < GRID_NX; ++ix)
{
    const int idx = ix + GRID_NX * (iy + GRID_NY * iz);

    // Cell-center position
    const vec3 x {
        gox + (ix + 0.5f) * dx,
        goy + (iy + 0.5f) * dy,
        goz + (iz + 0.5f) * dz
    };

    if (!geo.IsInside(x))
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();

        sol[idx]  = nan;
        exc[idx]  = nan;
        aerr[idx] = 0.f;
        serr[idx] = 0.f;
        ins[idx]  = 0.f;

        continue;
    }

    ins[idx] = 1.f;
    numInside++;

    // Thread-local parameters
    WoStParams localParams = params;

    // decorrelated RNG stream
    localParams.seed =
        RNG_SEED +
        static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL;

    WalkResult r =
        SolveLaplaceNeumann(
            geo,
            x,
            g,
            h,
            isNeumann,
            localParams);

    const float ex_val = exact_fn(x);
    const float err    = std::abs(r.value - ex_val);

    sol[idx]  = r.value;
    exc[idx]  = ex_val;
    aerr[idx] = err;
    serr[idx] = r.stdErr;

    sumAbsErr    += err;
    sumMeanSteps += r.meanSteps;

    if (r.anyDiverged)
        numDiverged++;
}

double elapsed =
    std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start
    ).count();

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n\nResults\n"
              << std::fixed << std::setprecision(4)
              << "  Interior voxels : " << numInside << " / " << N << "\n"
              << "  Diverged walks  : " << numDiverged << "\n"
              << "  Mean |u-u_ex|   : "
              << (numInside>0 ? sumAbsErr/numInside : 0.0) << "\n"
              << "  Mean walk steps : "
              << (numInside>0 ? sumMeanSteps/numInside : 0.0) << "\n"
              << "  Elapsed         : "
              << std::setprecision(2) << elapsed << " s\n\n";

    // ── Write VTK ─────────────────────────────────────────────────────────────
    WriteVTK("wost_solution.vtk",
             GRID_NX, GRID_NY, GRID_NZ,
             gox, goy, goz, dx, dy, dz,
             sol, exc, aerr, serr, ins);

    std::cout
        << "\nTips for ParaView / VisIt:\n"
        << "  1. Threshold on 'inside == 1' to hide exterior voxels.\n"
        << "  2. Colour by 'solution' (MC estimate) – should look like\n"
        << "     a smooth y-gradient from feet (blue) to head (red).\n"
        << "  3. Colour by 'abs_error' to inspect Monte Carlo error.\n"
        << "     Expected magnitude ~ std_error for 64 samples.\n";

    return 0;
}