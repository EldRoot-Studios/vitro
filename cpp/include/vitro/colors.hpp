// Colour ramps and the Vitro dark theme palette.
#pragma once
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>

#include "canvas.hpp"

namespace vitro {

namespace pal {
inline const Color bg = rgb(0x181b1f), panel = rgb(0x20242a), panel2 = rgb(0x292e36), border = rgb(0x3a414b);
inline const Color text = rgb(0xe3e6ea), dim = rgb(0x97a0ac), faint = rgb(0x5d6672);
inline const Color accent = rgb(0x60a5fa), good = rgb(0x52c478), bad = rgb(0xeb5454), warn = rgb(0xf0b43c);
inline const Color select = rgb(0xffe14d), hover = rgb(0xffffff);
}  // namespace pal

struct Stop { float at; Color c; };

// Piecewise-linear ramp over arbitrary monotonically increasing stop positions.
inline Color ramp(std::initializer_list<Stop> stops, float t) {
    const Stop* a = stops.begin();
    if (std::isnan(t) || t <= a->at) return a->c;
    for (const Stop* s = stops.begin() + 1; s != stops.end(); ++s) {
        if (t <= s->at) return mix(a->c, s->c, (t - a->at) / (s->at - a->at));
        a = s;
    }
    return a->c;
}

// Attention weights: dark to bright (magma-like); t in [0, 1].
inline Color magma(float t) {
    return ramp({{0.0f, rgb(0x050510)}, {0.2f, rgb(0x3b0f70)}, {0.45f, rgb(0x8c2981)}, {0.7f, rgb(0xde4968)},
                 {0.88f, rgb(0xfe9f6d)}, {1.0f, rgb(0xfcfdbf)}}, t);
}

// Latent / signed vector values: negative blue, zero neutral, positive orange-red; t in [-1, 1].
inline Color diverging(float t) {
    t = std::clamp(t, -1.0f, 1.0f);
    return t < 0 ? mix(rgb(0x20262e), rgb(0x3d8bfd), -t) : mix(rgb(0x20262e), rgb(0xff7a3d), t);
}

// Cosine similarity in [-1, 1]: bright warm for aligned, dark for orthogonal, purple for opposite.
inline Color similarity(float c) {
    c = std::clamp(c, -1.0f, 1.0f);
    return c >= 0 ? ramp({{0.0f, rgb(0x1b2028)}, {0.35f, rgb(0x2b4a7a)}, {0.7f, rgb(0x4fb0c6)}, {1.0f, rgb(0xf4f1bb)}}, c)
                  : mix(rgb(0x1b2028), rgb(0xa855f7), -c);
}

// Token surprisal in bits (-log2 p); calm cool tones for high certainty, hot orange/white for surprise.
inline Color surprisal(double bits) {
    float t = static_cast<float>(bits);
    return ramp({{0.0f, rgb(0x26313f)}, {0.1f, rgb(0x2b5d8f)}, {0.5f, rgb(0x3fa3ad)}, {1.0f, rgb(0xe6c84a)},
                 {3.0f, rgb(0xeb6e32)}, {6.0f, rgb(0xdc2840)}, {10.0f, rgb(0xffffff)}}, t);
}

// Scalar / numeric prediction distance ramp; error_at scales where the crimson ceiling starts (default 12 steps).
inline Color delta_error(double steps, double red_at = 12.0) {
    const float k = static_cast<float>(std::max(1.0, red_at) / 12.0);
    const float t = static_cast<float>(steps);
    return ramp({{0.0f * k, rgb(0x275a52)}, {1.0f * k, rgb(0x3f7d6a)}, {3.0f * k, rgb(0x93ad4f)},
                 {11.0f * k, rgb(0xe6a937)}, {12.0f * k, rgb(0xe0452f)}, {200.0f * k, rgb(0xff2a55)}}, t);
}

inline Color text_on(Color bg) {
    int lum = (bg.r * 299 + bg.g * 587 + bg.b * 114) / 1000;
    return lum > 140 ? rgb(0x11151a) : rgb(0xf5f7fa);
}

// Parses "#RRGGBB" or named 3D edge tint tags ("violet", "gold", "magenta", "blue", "cyan", "emerald").
inline Color parse_tint(const std::string& s, Color fallback = rgb(0x7b6df0)) {
    if (s.empty()) return fallback;
    if (s[0] == '#' && s.size() == 7) {
        char* end = nullptr;
        unsigned long v = std::strtoul(s.c_str() + 1, &end, 16);
        if (end == s.c_str() + 7) return rgb(static_cast<uint32_t>(v));
    }
    if (s == "violet") return rgb(0x7b6df0);
    if (s == "gold") return rgb(0xffc94a);
    if (s == "magenta") return rgb(0xb06cf0);
    if (s == "blue") return rgb(0x5f8cf2);
    if (s == "cyan") return rgb(0x4fb0c6);
    if (s == "emerald") return rgb(0x52c478);
    if (s == "coral") return rgb(0xeb6e52);
    return fallback;
}

}  // namespace vitro
