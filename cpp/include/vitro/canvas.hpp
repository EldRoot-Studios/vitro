// Software RGB canvas: direct pixel memory, vector primitives, max-pooled heat maps,
// anti-aliased Segoe UI / Consolas glyphs via GDI GetGlyphOutlineW, and atomic PNG export.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <emmintrin.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vitro {

struct Color { uint8_t r = 0, g = 0, b = 0; };
inline Color rgb(uint32_t v) { return {static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v)}; }
inline Color rgb(int r, int g, int b) { return {static_cast<uint8_t>(std::clamp(r, 0, 255)), static_cast<uint8_t>(std::clamp(g, 0, 255)), static_cast<uint8_t>(std::clamp(b, 0, 255))}; }
inline Color mix(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return {static_cast<uint8_t>(a.r + (b.r - a.r) * t + 0.5f), static_cast<uint8_t>(a.g + (b.g - a.g) * t + 0.5f), static_cast<uint8_t>(a.b + (b.b - a.b) * t + 0.5f)};
}
inline uint32_t pack(Color c) { return (static_cast<uint32_t>(c.r) << 16) | (static_cast<uint32_t>(c.g) << 8) | c.b; }
inline Color unpack(uint32_t v) { return {static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v)}; }

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    int right() const { return x + w; }
    int bottom() const { return y + h; }
    bool contains(int px, int py) const { return px >= x && py >= y && px < x + w && py < y + h; }
    bool empty() const { return w <= 0 || h <= 0; }
    Rect inset(int d) const { return {x + d, y + d, w - 2 * d, h - 2 * d}; }
    Rect intersect(const Rect& o) const {
        int x0 = std::max(x, o.x), y0 = std::max(y, o.y), x1 = std::min(right(), o.right()), y1 = std::min(bottom(), o.bottom());
        return {x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
    }
};

enum FontId { F_UI = 0, F_BOLD, F_SMALL, F_MONO, F_BIG, F_TINY, F_COUNT };

struct Glyph { int w = 0, h = 0, ox = 0, oy = 0, adv = 0; std::vector<uint8_t> a; };

inline std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        if (c < 0x80) cp = c;
        else if ((c >> 5) == 6 && i + 1 < s.size()) { cp = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu); len = 2; }
        else if ((c >> 4) == 14 && i + 2 < s.size()) { cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu); len = 3; }
        else if ((c >> 3) == 30 && i + 3 < s.size()) { cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu); len = 4; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

class Fonts {
public:
    explicit Fonts(float scale) {
        dc_ = CreateCompatibleDC(nullptr);
        build(scale);
    }
    ~Fonts() {
        destroy();
        DeleteDC(dc_);
    }
    Fonts(const Fonts&) = delete;
    Fonts& operator=(const Fonts&) = delete;

    void rescale(float scale) {
        destroy();
        cache_.clear();
        build(scale);
    }

    int ascent(FontId f) const { return ascent_[f]; }
    int height(FontId f) const { return height_[f]; }
    float scale() const { return scale_; }

    const Glyph& glyph(FontId f, uint32_t cp) {
        uint64_t key = (static_cast<uint64_t>(f) << 32) | cp;
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
        Glyph g;
        HGDIOBJ old = SelectObject(dc_, font_[f]);
        GLYPHMETRICS gm;
        MAT2 mat = {{0, 1}, {0, 0}, {0, 0}, {0, 1}};
        WCHAR wc[2] = {0, 0};
        int nwc = 1;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            wc[0] = static_cast<WCHAR>(0xD800 + (cp >> 10));
            wc[1] = static_cast<WCHAR>(0xDC00 + (cp & 0x3FF));
            nwc = 2;
        } else {
            wc[0] = static_cast<WCHAR>(cp);
        }
        WORD index = 0;
        GetGlyphIndicesW(dc_, wc, nwc, &index, GGI_MARK_NONEXISTING_GLYPHS);
        if (index == 0xFFFF) {
            wc[0] = L'?';
            nwc = 1;
        }
        DWORD size = GetGlyphOutlineW(dc_, wc[0], GGO_GRAY8_BITMAP, &gm, 0, nullptr, &mat);
        if (size == GDI_ERROR) {
            SelectObject(dc_, old);
            throw std::runtime_error("GetGlyphOutlineW failed for code point " + std::to_string(cp));
        }
        g.adv = gm.gmCellIncX;
        if (size > 0) {
            std::vector<uint8_t> buf(size);
            GetGlyphOutlineW(dc_, wc[0], GGO_GRAY8_BITMAP, &gm, size, buf.data(), &mat);
            g.w = static_cast<int>(gm.gmBlackBoxX);
            g.h = static_cast<int>(gm.gmBlackBoxY);
            g.ox = gm.gmptGlyphOrigin.x;
            g.oy = gm.gmptGlyphOrigin.y;
            size_t pitch = (static_cast<size_t>(g.w) + 3) & ~static_cast<size_t>(3);
            g.a.resize(static_cast<size_t>(g.w) * g.h);
            for (int y = 0; y < g.h; ++y)
                for (int x = 0; x < g.w; ++x) {
                    int v = buf[y * pitch + x];
                    g.a[static_cast<size_t>(y) * g.w + x] = static_cast<uint8_t>(std::min(255, v * 255 / 64));
                }
        }
        SelectObject(dc_, old);
        return cache_.emplace(key, std::move(g)).first->second;
    }

    int width(FontId f, const std::string& s) {
        int w = 0;
        for (uint32_t cp : utf8_codepoints(s)) w += glyph(f, cp).adv;
        return w;
    }

private:
    void build(float scale) {
        scale_ = scale;
        const struct { const wchar_t* face; int px; int weight; } spec[F_COUNT] = {
            {L"Segoe UI", 13, FW_NORMAL}, {L"Segoe UI", 13, FW_SEMIBOLD}, {L"Segoe UI", 11, FW_NORMAL},
            {L"Consolas", 12, FW_NORMAL}, {L"Segoe UI", 19, FW_SEMIBOLD}, {L"Segoe UI", 9, FW_NORMAL}};
        for (int i = 0; i < F_COUNT; ++i) {
            int px = std::max(6, static_cast<int>(spec[i].px * scale + 0.5f));
            font_[i] = CreateFontW(-px, 0, 0, 0, spec[i].weight, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   ANTIALIASED_QUALITY, DEFAULT_PITCH, spec[i].face);
            if (!font_[i]) throw std::runtime_error("cannot create a font");
            HGDIOBJ old_font = SelectObject(dc_, font_[i]);
            TEXTMETRICW tm;
            GetTextMetricsW(dc_, &tm);
            ascent_[i] = tm.tmAscent;
            height_[i] = tm.tmHeight;
            SelectObject(dc_, old_font);
        }
    }
    void destroy() {
        for (int i = 0; i < F_COUNT; ++i) {
            if (font_[i]) DeleteObject(font_[i]);
            font_[i] = nullptr;
        }
    }

    float scale_ = 1.0f;
    HDC dc_ = nullptr;
    HFONT font_[F_COUNT] = {};
    int ascent_[F_COUNT] = {}, height_[F_COUNT] = {};
    std::unordered_map<uint64_t, Glyph> cache_;
};

class Canvas {
public:
    int W, H;
    std::vector<uint32_t> px;
    Fonts fonts;

    Canvas(int w, int h, float scale) : W(w), H(h), px(static_cast<size_t>(w) * h, 0), fonts(scale), clip_{0, 0, w, h} {
        if (w < 16 || h < 16) throw std::runtime_error("a canvas smaller than 16 pixels");
    }
    float scale() const { return fonts.scale(); }
    void set_scale(float scale) { fonts.rescale(scale); }
    int s(int v) const { return static_cast<int>(v * fonts.scale() + 0.5f); }

    void resize(int w, int h) {
        W = w; H = h;
        px.assign(static_cast<size_t>(w) * h, 0);
        stack_.clear();
        clip_ = {0, 0, w, h};
    }
    void clear(Color c) { std::fill(px.begin(), px.end(), pack(c)); }

    void push_clip(Rect r) { stack_.push_back(clip_); clip_ = clip_.intersect(r); }
    void pop_clip() { clip_ = stack_.back(); stack_.pop_back(); }
    Rect clip() const { return clip_; }

    void put(int x, int y, Color c, float a = 1.0f) {
        if (!clip_.contains(x, y)) return;
        uint32_t& d = px[static_cast<size_t>(y) * W + x];
        if (a >= 0.999f) { d = pack(c); return; }
        if (a <= 0.0f) return;
        Color o = unpack(d);
        d = pack(mix(o, c, a));
    }
    void fill(Rect r, Color c, float a = 1.0f) {
        Rect q = r.intersect(clip_);
        if (q.empty()) return;
        uint32_t v = pack(c);
        for (int y = q.y; y < q.bottom(); ++y) {
            uint32_t* row = &px[static_cast<size_t>(y) * W];
            if (a >= 0.999f) std::fill(row + q.x, row + q.right(), v);
            else for (int x = q.x; x < q.right(); ++x) row[x] = pack(mix(unpack(row[x]), c, a));
        }
    }
    void frame(Rect r, Color c, int t = 1) {
        fill({r.x, r.y, r.w, t}, c);
        fill({r.x, r.bottom() - t, r.w, t}, c);
        fill({r.x, r.y, t, r.h}, c);
        fill({r.right() - t, r.y, t, r.h}, c);
    }
    void hline(int x0, int x1, int y, Color c, float a = 1.0f) { fill({std::min(x0, x1), y, std::abs(x1 - x0) + 1, 1}, c, a); }
    void vline(int x, int y0, int y1, Color c, float a = 1.0f) { fill({x, std::min(y0, y1), 1, std::abs(y1 - y0) + 1}, c, a); }
    void line(int x0, int y0, int x1, int y1, Color c, float a = 1.0f) {
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
        while (true) {
            put(x0, y0, c, a);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }
    void add_px(int x, int y, Color c, float a) {
        if (!clip_.contains(x, y) || a <= 0.0f) return;
        uint32_t& d = px[static_cast<size_t>(y) * W + x];
        int r = static_cast<int>((d >> 16) & 255) + static_cast<int>(c.r * a + 0.5f);
        int g = static_cast<int>((d >> 8) & 255) + static_cast<int>(c.g * a + 0.5f);
        int b = static_cast<int>(d & 255) + static_cast<int>(c.b * a + 0.5f);
        d = (static_cast<uint32_t>(std::min(r, 255)) << 16) | (static_cast<uint32_t>(std::min(g, 255)) << 8) | static_cast<uint32_t>(std::min(b, 255));
    }
    // Liang-Barsky clip + fixed-point Wu anti-aliased additive line in SSE2.
    void line_add(float x0, float y0, float x1, float y1, Color c, float a) {
        if (a <= 0.0f || clip_.w < 2 || clip_.h < 2) return;
        float dx = x1 - x0, dy = y1 - y0, t0 = 0.0f, t1 = 1.0f;
        const float lo[4] = {static_cast<float>(clip_.x), static_cast<float>(clip_.x + clip_.w - 2), static_cast<float>(clip_.y), static_cast<float>(clip_.y + clip_.h - 2)};
        const float p[4] = {-dx, dx, -dy, dy}, q[4] = {x0 - lo[0], lo[1] - x0, y0 - lo[2], lo[3] - y0};
        for (int i = 0; i < 4; ++i) {
            if (p[i] == 0.0f) { if (q[i] < 0.0f) return; continue; }
            float t = q[i] / p[i];
            if (p[i] < 0.0f) { if (t > t1) return; t0 = std::max(t0, t); }
            else { if (t < t0) return; t1 = std::min(t1, t); }
        }
        float ax = x0 + t0 * dx, ay = y0 + t0 * dy, bx = x0 + t1 * dx, by = y0 + t1 * dy;
        const bool steep = std::fabs(by - ay) > std::fabs(bx - ax);
        if (steep) { std::swap(ax, ay); std::swap(bx, by); }
        if (ax > bx) { std::swap(ax, bx); std::swap(ay, by); }
        const float ddx = bx - ax, grad = ddx < 1e-6f ? 0.0f : (by - ay) / ddx;
        const int xs = static_cast<int>(std::floor(ax)), xe = static_cast<int>(std::floor(bx));
        int32_t yfix = static_cast<int32_t>((ay + grad * (static_cast<float>(xs) - ax)) * 65536.0f);
        const int32_t gfix = static_cast<int32_t>(grad * 65536.0f);
        const float k = a * 256.0f;
        const __m128i col = _mm_set_epi16(0, 0, 0, 0, 0, static_cast<short>(std::min(65535.0f, c.r * k)), static_cast<short>(std::min(65535.0f, c.g * k)),
                                          static_cast<short>(std::min(65535.0f, c.b * k)));
        const __m128i half = _mm_set1_epi16(0x80);
        uint32_t* base = px.data();
        const size_t major = steep ? static_cast<size_t>(W) : 1u, minor = steep ? 1u : static_cast<size_t>(W);
        for (int x = xs; x <= xe; ++x, yfix += gfix) {
            const int yi = yfix >> 16;
            const uint32_t f = static_cast<uint32_t>(yfix) & 0xFFFFu;
            uint32_t* p0 = base + static_cast<size_t>(x) * major + static_cast<size_t>(yi) * minor;
            uint32_t* p1 = p0 + minor;
            const __m128i c0 = _mm_srli_epi16(_mm_add_epi16(_mm_mulhi_epu16(col, _mm_set1_epi16(static_cast<short>(65535u - f))), half), 8);
            const __m128i c1 = _mm_srli_epi16(_mm_add_epi16(_mm_mulhi_epu16(col, _mm_set1_epi16(static_cast<short>(f))), half), 8);
            *p0 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_adds_epu8(_mm_cvtsi32_si128(static_cast<int>(*p0)), _mm_packus_epi16(c0, c0))));
            *p1 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_adds_epu8(_mm_cvtsi32_si128(static_cast<int>(*p1)), _mm_packus_epi16(c1, c1))));
        }
    }
    void glow(float cx, float cy, float r, Color c, float a) {
        int x0 = static_cast<int>(std::floor(cx - r)), x1 = static_cast<int>(std::ceil(cx + r)), y0 = static_cast<int>(std::floor(cy - r)), y1 = static_cast<int>(std::ceil(cy + r));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                float ex = static_cast<float>(x) + 0.5f - cx, ey = static_cast<float>(y) + 0.5f - cy, d2 = (ex * ex + ey * ey) / (r * r);
                if (d2 < 1.0f) {
                    float fall = 1.0f - std::sqrt(d2);
                    add_px(x, y, c, a * fall * fall * 1.6f);
                }
            }
    }
    void hatch(Rect r, Color c, int step = 6, float a = 1.0f) {
        Rect q = r.intersect(clip_);
        for (int y = q.y; y < q.bottom(); ++y)
            for (int x = q.x; x < q.right(); ++x)
                if (((x + y) % step) == 0) put(x, y, c, a);
    }

    int text(int x, int y, const std::string& str, Color c, FontId f = F_UI, int w = 0, int align = 0) {
        int total = fonts.width(f, str);
        if (align == 1) x += (w - total) / 2;
        else if (align == 2) x += w - total;
        int base = y + fonts.ascent(f), pen = x;
        for (uint32_t cp : utf8_codepoints(str)) {
            const Glyph& g = fonts.glyph(f, cp);
            int gx = pen + g.ox, gy = base - g.oy;
            for (int yy = 0; yy < g.h; ++yy)
                for (int xx = 0; xx < g.w; ++xx) {
                    uint8_t cov = g.a[static_cast<size_t>(yy) * g.w + xx];
                    if (cov) put(gx + xx, gy + yy, c, cov / 255.0f);
                }
            pen += g.adv;
        }
        return total;
    }
    int text_width(const std::string& str, FontId f = F_UI) { return fonts.width(f, str); }
    int text_fit(int x, int y, int w, const std::string& str, Color c, FontId f = F_UI, int align = 0) {
        if (fonts.width(f, str) <= w) return text(x, y, str, c, f, w, align);
        std::vector<uint32_t> cps = utf8_codepoints(str);
        std::string out;
        int used = 0, dots = fonts.width(f, "\xE2\x80\xA6");
        for (uint32_t cp : cps) {
            int adv = fonts.glyph(f, cp).adv;
            if (used + adv + dots > w) break;
            used += adv;
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
            else { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        }
        return text(x, y, out + "\xE2\x80\xA6", c, f, w, 0);
    }

    void heat(Rect dst, int rows, int cols, const std::function<float(int, int)>& value, const std::function<Color(float)>& cmap) {
        Rect q = dst.intersect(clip_);
        if (q.empty() || rows <= 0 || cols <= 0) return;
        const size_t cells = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        heat_values_.resize(cells);
        heat_colors_.resize(cells);
        const uint32_t none = pack(rgb(0x14181c));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                float v = value(r, c);
                size_t k = static_cast<size_t>(r) * cols + c;
                heat_values_[k] = v;
                heat_colors_[k] = std::isnan(v) ? none : pack(cmap(v));
            }
        heat_col0_.resize(static_cast<size_t>(q.w));
        heat_col1_.resize(static_cast<size_t>(q.w));
        for (int x = q.x; x < q.right(); ++x) {
            int c0 = static_cast<int>(static_cast<long long>(x - dst.x) * cols / dst.w);
            int c1 = std::min(cols, std::max(c0 + 1, static_cast<int>(static_cast<long long>(x - dst.x + 1) * cols / dst.w)));
            heat_col0_[static_cast<size_t>(x - q.x)] = c0;
            heat_col1_[static_cast<size_t>(x - q.x)] = c1;
        }
        for (int y = q.y; y < q.bottom(); ++y) {
            const int r0 = static_cast<int>(static_cast<long long>(y - dst.y) * rows / dst.h);
            const int r1 = std::min(rows, std::max(r0 + 1, static_cast<int>(static_cast<long long>(y - dst.y + 1) * rows / dst.h)));
            uint32_t* row = &px[static_cast<size_t>(y) * W];
            for (int x = q.x; x < q.right(); ++x) {
                const int c0 = heat_col0_[static_cast<size_t>(x - q.x)], c1 = heat_col1_[static_cast<size_t>(x - q.x)];
                if (r1 - r0 == 1 && c1 - c0 == 1) {
                    row[x] = heat_colors_[static_cast<size_t>(r0) * cols + c0];
                    continue;
                }
                float best = -INFINITY;
                size_t at = 0;
                bool any = false;
                for (int r = r0; r < r1; ++r)
                    for (int c = c0; c < c1; ++c) {
                        size_t k = static_cast<size_t>(r) * cols + c;
                        float v = heat_values_[k];
                        if (std::isnan(v) || v <= best) continue;
                        any = true;
                        best = v;
                        at = k;
                    }
                row[x] = any ? heat_colors_[at] : none;
            }
        }
    }

    void save_png(const std::string& path) const {
        std::vector<uint8_t> raw;
        raw.reserve((static_cast<size_t>(W) * 3 + 1) * H);
        for (int y = 0; y < H; ++y) {
            raw.push_back(0);
            for (int x = 0; x < W; ++x) {
                uint32_t p = px[static_cast<size_t>(y) * W + x];
                raw.push_back(static_cast<uint8_t>(p >> 16)); raw.push_back(static_cast<uint8_t>(p >> 8)); raw.push_back(static_cast<uint8_t>(p));
            }
        }
        uLongf cl = compressBound(static_cast<uLong>(raw.size()));
        std::vector<uint8_t> comp(cl);
        int zr = compress2(comp.data(), &cl, raw.data(), static_cast<uLong>(raw.size()), 6);
        if (zr != Z_OK) throw std::runtime_error("zlib compress2 failed with " + std::to_string(zr));
        comp.resize(cl);
        std::vector<uint8_t> png = {137, 80, 78, 71, 13, 10, 26, 10};
        auto be32 = [](std::vector<uint8_t>& v, uint32_t x) { for (int i = 3; i >= 0; --i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); };
        auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
            be32(png, static_cast<uint32_t>(data.size()));
            std::vector<uint8_t> td(type, type + 4);
            td.insert(td.end(), data.begin(), data.end());
            png.insert(png.end(), td.begin(), td.end());
            be32(png, static_cast<uint32_t>(crc32(0, td.data(), static_cast<uInt>(td.size()))));
        };
        std::vector<uint8_t> ihdr;
        be32(ihdr, static_cast<uint32_t>(W)); be32(ihdr, static_cast<uint32_t>(H));
        ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
        chunk("IHDR", ihdr);
        chunk("IDAT", comp);
        chunk("IEND", {});
        std::string tmp = path + ".part";
        FILE* fh = std::fopen(tmp.c_str(), "wb");
        if (!fh) throw std::runtime_error("cannot write " + tmp);
        size_t put = std::fwrite(png.data(), 1, png.size(), fh);
        bool ok = std::fflush(fh) == 0 && put == png.size();
        std::fclose(fh);
        if (!ok) throw std::runtime_error("short write to " + tmp);
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp + " to " + path);
    }

private:
    Rect clip_;
    std::vector<Rect> stack_;
    std::vector<float> heat_values_;
    std::vector<uint32_t> heat_colors_;
    std::vector<int> heat_col0_, heat_col1_;
};

class StaggeredLabels {
public:
    explicit StaggeredLabels(int rows = 2) : rows_(static_cast<size_t>(rows)) {}
    int place(int cx, int w, int gap) {
        const int left = cx - w / 2, right = cx + w / 2;
        for (size_t k = 0; k < rows_.size(); ++k) {
            bool free_here = true;
            for (const std::pair<int, int>& iv : rows_[k])
                if (left - gap < iv.second && right + gap > iv.first) {
                    free_here = false;
                    break;
                }
            if (free_here || k + 1 == rows_.size()) {
                rows_[k].push_back({left, right});
                return static_cast<int>(k);
            }
        }
        return 0;
    }

private:
    std::vector<std::vector<std::pair<int, int>>> rows_;
};

}  // namespace vitro
