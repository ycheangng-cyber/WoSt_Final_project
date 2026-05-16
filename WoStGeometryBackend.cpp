// ============================================================================
// WoStGeometryBackend.cpp
//
// Implementation of WoStGeometryBackend.
// Add  #define TINYBVH_IMPLEMENTATION  in this file (or another .cpp in the
// project) before tiny_bvh.h is first included.
// ============================================================================

#define TINYBVH_IMPLEMENTATION
#include "WoStGeometryBackend.hpp"

#include <cassert>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <map>
#include <array>
#include <utility>
#include <stdexcept>
#include <cstdio>

namespace wost {

// ============================================================================
// Internal math aliases (avoid repeating the tinybvh_ prefix everywhere)
// ============================================================================
static inline float  dot3  (const vec3& a, const vec3& b) { return tinybvh::tinybvh_dot(a, b); }
static inline float  len3  (const vec3& a)                { return tinybvh::tinybvh_length(a); }
static inline vec3   norm3 (const vec3& a)                { return tinybvh::tinybvh_normalize(a); }
static inline vec3   cross3(const vec3& a, const vec3& b) { return tinybvh::tinybvh_cross(a, b); }

static inline vec3 sub(const vec3& a, const vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
static inline vec3 add(const vec3& a, const vec3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
static inline vec3 scale(const vec3& a, float s) {
    return { a.x * s, a.y * s, a.z * s };
}
static inline float dist2(const vec3& a, const vec3& b) {
    vec3 d = sub(a, b);
    return dot3(d, d);
}

// ============================================================================
// (A) Closest point on a triangle – Ericson §5.1.5
// ============================================================================
vec3 WoStGeometryBackend::ClosestPtOnTriangle(
        const vec3& p,
        const vec3& a, const vec3& b, const vec3& c)
{
    vec3 ab = sub(b, a);
    vec3 ac = sub(c, a);
    vec3 ap = sub(p, a);

    float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;   // vertex A

    vec3  bp = sub(p, b);
    float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b;   // vertex B

    vec3  cp = sub(p, c);
    float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return c;   // vertex C

    // Edge AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    // Edge AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        float w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    // Edge BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    // Interior
    float inv = 1.f / (va + vb + vc);
    float v   = vb * inv, w = vc * inv;
    return add(a, add(scale(ab, v), scale(ac, w)));
}

// ============================================================================
// (B) Squared distance from a point to an AABB
//     Used to prune BVH nodes during closest-point traversal.
// ============================================================================
float WoStGeometryBackend::PointAABBDist2(
        const vec3& p, const vec3& bmin, const vec3& bmax)
{
    float dx = std::max(0.f, std::max(bmin.x - p.x, p.x - bmax.x));
    float dy = std::max(0.f, std::max(bmin.y - p.y, p.y - bmax.y));
    float dz = std::max(0.f, std::max(bmin.z - p.z, p.z - bmax.z));
    return dx*dx + dy*dy + dz*dz;
}

// ============================================================================
// (C) Squared distance from a point to a line segment  [a, b]
//     Also returns the closest point on the segment.
// ============================================================================
float WoStGeometryBackend::PointSegDist2(
        const vec3& p, const vec3& a, const vec3& b, vec3& closest)
{
    vec3  ab = sub(b, a);
    float t  = dot3(sub(p, a), ab);
    float d  = dot3(ab, ab);

    if (d > 1e-12f) t /= d; else t = 0.f;
    t = std::max(0.f, std::min(1.f, t));

    closest = add(a, scale(ab, t));
    return dist2(p, closest);
}

// ============================================================================
// (1) Closest point on ∂Ω  –  BVH traversal
//
// We traverse the BVH with a stack, pruning any node whose minimum possible
// distance to x (= distance to its AABB) exceeds the running best.
// ============================================================================

float WoStGeometryBackend::ClosestPointBVH(
        const vec3& x, BoundaryPoint& out) const
{
    float bestD2 = std::numeric_limits<float>::max();

    // Stack of (node_index, lower_bound_dist2).
    // We use a simple array-backed stack for speed.
    constexpr int STACK_SIZE = 64;
    uint32_t stk[STACK_SIZE];
    int      top = 0;
    stk[top++]   = 0;   // root

    const auto* nodes = bvh.bvhNode;
    const auto& vslice = bvh.verts;

    while (top > 0) {
        uint32_t nodeIdx = stk[--top];
        const auto& node = nodes[nodeIdx];

        if (PointAABBDist2(x, node.aabbMin, node.aabbMax) >= bestD2)
            continue;   // can't improve current best

        if (node.isLeaf()) {
            // Test every primitive in this leaf
            for (uint32_t i = 0; i < node.triCount; ++i) {
                uint32_t prim = bvh.primIdx[node.leftFirst + i];

                vec3 a(vslice[prim * 3 + 0]);
                vec3 b(vslice[prim * 3 + 1]);
                vec3 c(vslice[prim * 3 + 2]);

                vec3  cp = ClosestPtOnTriangle(x, a, b, c);
                float d2 = dist2(x, cp);

                if (d2 < bestD2) {
                    bestD2 = d2;
                    out.position = cp;
                    out.normal   = triNormals[prim];
                    out.triIdx   = prim;
                    out.dist     = std::sqrt(d2);
                }
            }
        } else {
            // Push children; heuristic: push the farther child first so the
            // nearer child is processed next (better pruning in practice).
            uint32_t left  = node.leftFirst;
            uint32_t right = node.leftFirst + 1;
            float    dLeft  = PointAABBDist2(x, nodes[left].aabbMin,  nodes[left].aabbMax);
            float    dRight = PointAABBDist2(x, nodes[right].aabbMin, nodes[right].aabbMax);

            if (dLeft <= dRight) {
                if (top + 1 < STACK_SIZE) { stk[top++] = right; stk[top++] = left; }
            } else {
                if (top + 1 < STACK_SIZE) { stk[top++] = left;  stk[top++] = right; }
            }
        }
    }
    return out.dist;
}

float WoStGeometryBackend::ClosestPoint(const vec3& x, BoundaryPoint& bp) const
{
    return ClosestPointBVH(x, bp);
}

// ============================================================================
// (2a) Closest silhouette edge from x  –  linear scan
//
// An edge (v0, v1) with face normals n0, n1 is a *silhouette* from x when
//   sign( dot(n0, x - v0) )  !=  sign( dot(n1, x - v0) )
// For every such edge we compute the closest point on the segment and keep
// the minimum.
//
// Complexity: O(E).  For very large meshes build a BVH over edges too.
// ============================================================================
float WoStGeometryBackend::ClosestSilhouette(
        const vec3& x, BoundaryPoint& out) const
{
    float bestD2 = std::numeric_limits<float>::max();
    out.dist = std::numeric_limits<float>::max();

    for (const auto& e : silhouettes) {
        // Is this edge a silhouette as seen from x?
        float s0 = dot3(e.n0, sub(x, e.v0));
        float s1 = dot3(e.n1, sub(x, e.v0));
        if (s0 * s1 >= 0.f) continue;   // same sign → not a silhouette

        vec3  cp;
        float d2 = PointSegDist2(x, e.v0, e.v1, cp);

        if (d2 < bestD2) {
            bestD2       = d2;
            out.position = cp;
            // Approximate normal: average of the two face normals
            out.normal   = norm3(add(e.n0, e.n1));
            out.dist     = std::sqrt(d2);
            out.triIdx   = ~0u;   // edge, not a full triangle
        }
    }
    return out.dist;
}

// ============================================================================
// (2b) StarRadius  =  min( ClosestPoint, ClosestSilhouette )
// ============================================================================
float WoStGeometryBackend::StarRadius(
        const vec3& x,
        BoundaryPoint& closestBoundary,
        BoundaryPoint& closestSilhouette) const
{
    float db = ClosestPointBVH(x, closestBoundary);
    float ds = ClosestSilhouette(x, closestSilhouette);
    return std::min(db, ds);
}

float WoStGeometryBackend::StarRadius(const vec3& x) const
{
    BoundaryPoint bp, sp;
    return StarRadius(x, bp, sp);
}

// ============================================================================
// (3) Ray–boundary intersection
//
// Wraps BVH::Intersect and reconstructs the outward normal from the stored
// per-triangle normals using the Intersection.prim index.
// ============================================================================
bool WoStGeometryBackend::IntersectRay(
        const vec3& origin, const vec3& dir, float tMax,
        float& t, vec3& hitNormal, uint32_t& primIdx) const
{
    tinybvh::Ray ray(origin, dir, tMax);
    bvh.Intersect(ray);

    if (ray.hit.t >= tMax)
        return false;

    t        = ray.hit.t;
    primIdx  = ray.hit.prim;
    hitNormal = triNormals[primIdx];

    // Flip if the ray is coming from outside the surface (back-face)
    if (dot3(hitNormal, dir) > 0.f)
        hitNormal = scale(hitNormal, -1.f);

    return true;
}

// ============================================================================
// (4) IsInside – ray-casting with randomized direction
//
// Count how many times a ray from x crosses ∂Ω.
// Odd count → x is inside Ω.
//
// We randomize the ray direction on every call to avoid aligning with mesh
// grids or coplanar edges, which can cause parity errors.
// ============================================================================
bool WoStGeometryBackend::IsInside(const vec3& x) const
{
    // Randomize ray direction to avoid alignment issues with mesh geometry
    // Use a pseudo-random direction based on the point position with proper spherical mapping
    float seed = x.x * 12.9898f + x.y * 78.233f + x.z * 37.719f;
    float u = std::fmod(seed * 43758.5453f, 1.0f); // [0, 1]
    float v = std::fmod(seed * 67108.864f, 1.0f);  // [0, 1]
    float theta = 2.0f * 3.14159265359f * u;
    float phi = std::acos(1.0f - 2.0f * v);
    
    vec3 RAY_DIR;
    RAY_DIR.x = std::sin(phi) * std::cos(theta);
    RAY_DIR.y = std::sin(phi) * std::sin(theta);
    RAY_DIR.z = std::cos(phi);
    
    static const float INF    = 1e30f;

    int    crossings = 0;
    float  offset    = 1e-5f;          // small offset to avoid re-hitting same prim
    vec3   origin    = x;

    for (;;) {
        tinybvh::Ray ray(origin, RAY_DIR, INF);
        int hitCount = bvh.Intersect(ray);
        if (hitCount == 0 || ray.hit.t >= INF)
            break;

        ++crossings;
        // Advance origin past this hit
        origin.x += ray.hit.t * RAY_DIR.x + offset;
        origin.y += ray.hit.t * RAY_DIR.y + offset;
        origin.z += ray.hit.t * RAY_DIR.z + offset;
    }
    return (crossings & 1) != 0;   // odd crossing count → inside
}

// ============================================================================
// Build-time: per-triangle outward normals
// ============================================================================
void WoStGeometryBackend::ComputeNormals()
{
    triNormals.resize(numTriangles);
    for (uint32_t i = 0; i < numTriangles; ++i) {
        vec3 a(triangles[i * 3 + 0]);
        vec3 b(triangles[i * 3 + 1]);
        vec3 c(triangles[i * 3 + 2]);

        vec3 n = cross3(sub(b, a), sub(c, a));
        float l = len3(n);
        triNormals[i] = (l > 1e-12f) ? scale(n, 1.f / l) : vec3(0, 1, 0);
    }
}

// ============================================================================
// Build-time: silhouette edge list via half-edge adjacency
//
// For every pair of triangles that share an edge (manifold assumption), we
// store the edge endpoints and both face normals.  The silhouette-from-x test
// is then O(1) per edge (two dot products + sign check).
// ============================================================================
void WoStGeometryBackend::BuildSilhouetteEdges()
{
    // Key: sorted vertex-index pair  (v_lo, v_hi) → [tri0, tri1]
    using EdgeKey  = std::pair<uint32_t, uint32_t>;
    using AdjEntry = std::pair<int, int>;   // first=-1 means empty
    std::map<EdgeKey, AdjEntry> adj;

    // We use vertex position equality to identify shared vertices.
    // Build a position → canonical index map.
    // For indexed geometry this would use index buffers directly; since we
    // have a flat (non-indexed) array we hash positions.
    // Simple approach: sort vertices and assign IDs.
    // For brevity we hash the raw bit pattern (exact float equality works for
    // meshes loaded from a file where shared vertices have identical bits).

    auto packVec = [](const vec3& v) -> std::array<float,3> {
        return { v.x, v.y, v.z };
    };

    struct VecCmp {
        bool operator()(const std::array<float,3>& a,
                        const std::array<float,3>& b) const {
            for (int i = 0; i < 3; ++i)
                if (a[i] != b[i]) return a[i] < b[i];
            return false;
        }
    };
    std::map<std::array<float,3>, uint32_t, VecCmp> vertexID;
    uint32_t nextID = 0;

    auto getID = [&](const vec3& v) -> uint32_t {
        auto key = packVec(v);
        auto it  = vertexID.find(key);
        if (it != vertexID.end()) return it->second;
        vertexID[key] = nextID;
        return nextID++;
    };

    // Build adjacency
    for (uint32_t tri = 0; tri < numTriangles; ++tri) {
        uint32_t vi[3];
        for (int k = 0; k < 3; ++k)
            vi[k] = getID(vec3(triangles[tri * 3 + k]));

        for (int k = 0; k < 3; ++k) {
            uint32_t a = vi[k], b = vi[(k + 1) % 3];
            EdgeKey  key = (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};

            auto it = adj.find(key);
            if (it == adj.end())
                adj[key] = { (int)tri, -1 };
            else if (it->second.second == -1)
                it->second.second = (int)tri;
            // (manifold: at most 2 triangles per edge)
        }
    }

    // Populate silhouettes from manifold edges (those with two adjacent tris)
    // AND open boundary edges (those with only one adjacent tri)
    silhouettes.clear();
    silhouettes.reserve(adj.size());

    for (auto& [key, tris] : adj) {
        if (tris.first == -1 || tris.second == -1) {
            // This is an open boundary edge. It is ALWAYS a silhouette!
            uint32_t t0 = (tris.first != -1) ? (uint32_t)tris.first : (uint32_t)tris.second;
            
            vec3 va, vb;
            bool gotA = false, gotB = false;
            for (int k = 0; k < 3; ++k) {
                vec3 vk(triangles[t0 * 3 + k]);
                uint32_t id = getID(vk);
                if (id == key.first  && !gotA) { va = vk; gotA = true; }
                if (id == key.second && !gotB) { vb = vk; gotB = true; }
            }
            if (gotA && gotB) {
                SilhouetteEdge se;
                se.v0 = va;
                se.v1 = vb;
                se.n0 = triNormals[t0];
                se.n1 = scale(triNormals[t0], -1.f); // Invert the second normal!
                silhouettes.push_back(se);
            }
            continue;
        }

        uint32_t t0 = (uint32_t)tris.first;
        uint32_t t1 = (uint32_t)tris.second;

        // Recover actual 3-D positions for the shared vertices.
        // Walk through t0's vertices and pick the two that match the key.
        vec3 va, vb;
        bool gotA = false, gotB = false;
        for (int k = 0; k < 3; ++k) {
            vec3 vk(triangles[t0 * 3 + k]);
            uint32_t id = getID(vk);
            if (id == key.first  && !gotA) { va = vk; gotA = true; }
            if (id == key.second && !gotB) { vb = vk; gotB = true; }
        }
        if (!gotA || !gotB) continue;

        SilhouetteEdge se;
        se.v0 = va;
        se.v1 = vb;
        se.n0 = triNormals[t0];
        se.n1 = triNormals[t1];
        silhouettes.push_back(se);
    }
}

// ============================================================================
// OBJ loader
//
// Minimal subset: v (vertices) and f (faces, triangles only, no materials).
// Negative indices, quads, and polygon faces are not supported.
// ============================================================================
void WoStGeometryBackend::LoadOBJ(const std::string& path)
{
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("WoStGeometryBackend: cannot open " + path);

    std::vector<vec3>     positions;
    std::vector<uint32_t> faces;   // triples of position indices

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back({ x, y, z });
        } else if (token == "f") {
            // Parse up to 4 indices (handle quads by splitting into two tris)
            uint32_t idx[4]; int cnt = 0;
            std::string fv;
            while (ss >> fv && cnt < 4) {
                // Handle v, v/vt, v/vt/vn, v//vn
                uint32_t vi = (uint32_t)std::stoi(fv);
                if (vi > 0) --vi; else vi = (uint32_t)positions.size() + vi;
                idx[cnt++] = vi;
            }
            if (cnt >= 3) {
                faces.push_back(idx[0]); faces.push_back(idx[1]); faces.push_back(idx[2]);
            }
            if (cnt == 4) {   // split quad into two triangles
                faces.push_back(idx[0]); faces.push_back(idx[2]); faces.push_back(idx[3]);
            }
        }
    }

    numVertices  = (uint32_t)positions.size();
    numTriangles = (uint32_t)(faces.size() / 3);

    // Build the flat interleaved array expected by tiny_bvh:
    //   triangle i  →  triangles[i*3 + {0,1,2}]
    triangles = new vec4[numTriangles * 3];
    for (uint32_t i = 0; i < numTriangles; ++i) {
        for (int k = 0; k < 3; ++k) {
            const vec3& p = positions[faces[i * 3 + k]];
            triangles[i * 3 + k] = vec4(p.x, p.y, p.z, 0.f);
        }
    }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================
WoStGeometryBackend::WoStGeometryBackend(const std::string& objFile)
{
    LoadOBJ(objFile);
    ComputeNormals();

    // Build the BVH.  primCount = numTriangles (not numTriangles * 3).
    // tiny_bvh expects 3 consecutive bvhvec4 per triangle.
    bvh.Build(triangles, numTriangles);

    BuildSilhouetteEdges();

    std::printf("[WoSt] Loaded %u triangles, %u silhouette edges.\n",
                numTriangles, (unsigned)silhouettes.size());
}

WoStGeometryBackend::~WoStGeometryBackend()
{
    delete[] triangles;
}

} // namespace wost