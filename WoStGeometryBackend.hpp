// ============================================================================
// WoStGeometryBackend.hpp
//
// Geometry backend for Walk on Stars (WoSt) – a Monte Carlo PDE solver.
//
// Reference: "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs
//             with Neumann Boundary Conditions"
//             Sawhney, Seyb, Jarosz, Crane – SIGGRAPH 2023.
//
// WoSt needs exactly four geometric queries from the scene:
//
//   (1)  ClosestPoint(x)     nearest point on ∂Ω; gives sphere radius for WoS
//   (2)  StarRadius(x)       radius of the largest *star-shaped* ball at x
//                            = min(dist to ∂Ω,  dist to nearest silhouette)
//                            This is the key WoSt ingredient.
//   (3)  IntersectRay(…)     segment vs ∂Ω; used to sample on-boundary points
//   (4)  IsInside(x)         inside/outside test for domain Ω
//
// Dependency: tiny_bvh (single-header, MIT).
// Define TINYBVH_IMPLEMENTATION in exactly one translation unit before
// including tiny_bvh.h.
// ============================================================================
#pragma once

#include "tiny_bvh.h"
#include <string>
#include <vector>

namespace wost {

using vec3 = tinybvh::bvhvec3;
using vec4 = tinybvh::bvhvec4;

// ---------------------------------------------------------------------------
// BoundaryPoint – result of closest-point / ray-hit queries
// ---------------------------------------------------------------------------
struct BoundaryPoint {
    vec3     position;        // point on ∂Ω
    vec3     normal;          // outward unit normal
    float    dist   = 0.f;   // distance from the query point x
    uint32_t triIdx = ~0u;   // flat triangle index (vertex = triangles[triIdx*3+k])
};

// ---------------------------------------------------------------------------
// SilhouetteEdge – precomputed for the mesh topology
//
// An edge shared by two triangles with outward normals n0, n1.
// From query point x the edge is a silhouette when the signs of
//   dot(n0, x - v0)  and  dot(n1, x - v0)
// differ, i.e. one face points toward x and the other points away.
// The WoSt star radius is min(closest-boundary, closest-silhouette).
// ---------------------------------------------------------------------------
struct SilhouetteEdge {
    vec3 v0, v1;   // endpoints
    vec3 n0, n1;   // face normals of the two adjacent triangles
};

// ===========================================================================
// WoStGeometryBackend
// ===========================================================================
class WoStGeometryBackend {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    explicit WoStGeometryBackend(const std::string& objFile);
    ~WoStGeometryBackend();

    WoStGeometryBackend(const WoStGeometryBackend&)            = delete;
    WoStGeometryBackend& operator=(const WoStGeometryBackend&) = delete;

    // -----------------------------------------------------------------------
    // (1) Closest point on ∂Ω
    //     BVH traversal: O(log N) expected.
    //     Fill bp; return the Euclidean distance.
    // -----------------------------------------------------------------------
    float ClosestPoint(const vec3& x, BoundaryPoint& bp) const;

    // -----------------------------------------------------------------------
    // (2) Star-shaped ball radius
    //     = min( closest-boundary-dist, closest-silhouette-dist )
    //
    //     The two-argument overload also returns the two contributing points,
    //     which is useful for debugging and for reading Dirichlet BCs at
    //     termination.
    // -----------------------------------------------------------------------
    float StarRadius(const vec3& x) const;
    float StarRadius(const vec3& x,
                     BoundaryPoint& closestBoundary,
                     BoundaryPoint& closestSilhouette) const;

    // -----------------------------------------------------------------------
    // (3) Ray–boundary intersection
    //     Returns true if the ray [origin, origin + tMax*dir] hits ∂Ω.
    //     On hit, sets t (parametric), hitNormal (outward), primIdx.
    //     Wrap of BVH::Intersect with normal reconstruction from barycentric
    //     coordinates.
    // -----------------------------------------------------------------------
    bool IntersectRay(const vec3& origin, const vec3& dir, float tMax,
                      float& t, vec3& hitNormal, uint32_t& primIdx) const;

    // -----------------------------------------------------------------------
    // (4) Inside/outside test
    //     Ray-casting along a fixed direction; odd crossing count → inside.
    //     Uses BVH::Intersect in a loop until tMax is exhausted.
    // -----------------------------------------------------------------------
    bool IsInside(const vec3& x) const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    uint32_t TriangleCount() const { return numTriangles; }
    const vec4* Vertices()   const { return triangles;    }
    const vec3& TriNormal(uint32_t i) const { return triNormals[i]; }
    const std::vector<SilhouetteEdge>& Silhouettes() const { return silhouettes; }

    void MeshBounds(vec3& outMin, vec3& outMax) const {
        outMin = bvh.aabbMin; outMax = bvh.aabbMax;
    }

private:
    // --- geometry primitives ------------------------------------------------
    static vec3  ClosestPtOnTriangle(const vec3& p,
                                     const vec3& a, const vec3& b, const vec3& c);
    static float PointAABBDist2     (const vec3& p,
                                     const vec3& bmin, const vec3& bmax);
    static float PointSegDist2      (const vec3& p,
                                     const vec3& a, const vec3& b, vec3& closest);
    // --- BVH traversals -----------------------------------------------------
    float ClosestPointBVH   (const vec3& x, BoundaryPoint& out) const;
    float ClosestSilhouette (const vec3& x, BoundaryPoint& out) const;
    // --- build-time helpers -------------------------------------------------
    void LoadOBJ              (const std::string& path);
    void ComputeNormals       ();
    void BuildSilhouetteEdges ();

    // --- data ---------------------------------------------------------------
    tinybvh::BVH bvh;

    // Flat SOA: triangle i has vertices at triangles[i*3 + {0,1,2}]
    vec4*    triangles    = nullptr;
    uint32_t numTriangles = 0;
    uint32_t numVertices  = 0;

    std::vector<vec3>           triNormals;
    std::vector<SilhouetteEdge> silhouettes;
};

} // namespace wost