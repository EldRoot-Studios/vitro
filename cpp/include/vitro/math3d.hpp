// Shared 3D perspective camera, deterministic spatial jitter, and top-K index helpers.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace vitro {

struct Camera3D {
    float cy = 1, sy = 0, cp = 1, sp = 0;
    float focal = 1, dist = 1, ox = 0, oy = 0, vx = 0, vy = 0, zoom = 1, half_span = 0;

    void map(float wx, float wy, float wz, float* out_x, float* out_y, float* out_scale) const {
        float x = wx - half_span;
        float x1 = x * cy + wz * sy, z1 = -x * sy + wz * cy;
        float y2 = wy * cp - z1 * sp, z2 = wy * sp + z1 * cp;
        float f = focal / std::max(1e-3f, dist + z2);
        *out_x = vx + (ox + x1 * f - vx) * zoom;
        *out_y = vy + (oy - y2 * f - vy) * zoom;
        *out_scale = f * zoom;
    }
};

inline float hash01(uint32_t a, uint32_t b) {
    uint32_t h = (a * 0x9E3779B1u) ^ ((b + 0x7F4A7C15u) * 0x85EBCA6Bu);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return static_cast<float>(h & 0xFFFFFF) / 16777216.0f;
}

inline void top_k_indices(const float* v, int n, int k, float floor_w, std::vector<int>& idx, std::vector<int>& out) {
    out.clear();
    if (n <= 0 || k <= 0) return;
    idx.resize(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    int take = std::min(k, n);
    auto by_weight = [&](int p, int q) { return v[p] > v[q]; };
    if (take < n) std::nth_element(idx.begin(), idx.begin() + take, idx.end(), by_weight);
    std::sort(idx.begin(), idx.begin() + take, by_weight);
    for (int i = 0; i < take; ++i)
        if (v[idx[static_cast<size_t>(i)]] > floor_w) out.push_back(idx[static_cast<size_t>(i)]);
}

}  // namespace vitro
