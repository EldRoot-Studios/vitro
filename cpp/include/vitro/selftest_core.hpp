// Reusable core C++ self-tests shared across Vitro binaries and downstream tools.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "canvas.hpp"
#include "colors.hpp"
#include "container.hpp"
#include "json.hpp"
#include "math3d.hpp"

namespace vitro {

struct SelftestResult {
    int checks = 0;
    int failed = 0;

    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failed;
            std::fprintf(stderr, "selftest FAILED: %s\n", what.c_str());
        }
    }
};

inline bool test_throws(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

inline void run_core_selftests(SelftestResult& r) {
    // JSON parser
    {
        Json j = parse_json(R"({"a":[1,2.5,-3e2],"s":"x\u00e9\ud83d\ude00\n","t":true,"n":null,"o":{"k":7}})");
        r.check(j["a"].size() == 3 && j["a"].at(1).num() == 2.5 && j["a"].at(2).num() == -300.0, "json numbers");
        r.check(j["s"].str() == "x\xC3\xA9\xF0\x9F\x98\x80\n", "json string escapes and surrogate pair");
        r.check(j["t"].boolean() && j["n"].is_null() && j["o"]["k"].i() == 7, "json literals and nesting");
        r.check(test_throws([&] { (void)j["missing"]; }), "json missing key throws");
        r.check(test_throws([&] { (void)j["a"].str(); }), "json wrong type throws");
        r.check(test_throws([&] { (void)j["a"].at(1).i(); }), "json float as int throws");
        r.check(test_throws([] { parse_json("{\"a\":1} x"); }), "json trailing text throws");
        r.check(test_throws([] { parse_json("{\"a\":}"); }), "json missing value throws");
        r.check(test_throws([] { parse_json("\"\\q\""); }), "json bad escape throws");
        r.check(test_throws([] { parse_json("[1,2"); }), "json unterminated array throws");
        r.check(test_throws([] { parse_json("\"\\ud800\""); }), "json lone surrogate throws");
    }
    // Half float
    {
        r.check(half_to_float(0x3C00) == 1.0f && half_to_float(0xC000) == -2.0f && half_to_float(0x0000) == 0.0f, "half: 1, -2, 0");
        r.check(half_to_float(0x7BFF) == 65504.0f, "half: max finite");
        float tiny = half_to_float(0x0001);
        r.check(tiny > 5.9e-8f && tiny < 6.0e-8f, "half: smallest subnormal");
        r.check(std::isinf(half_to_float(0x7C00)) && std::isnan(half_to_float(0x7E00)), "half: inf and NaN");
        r.check(half_to_float(0x3555) > 0.3332f && half_to_float(0x3555) < 0.3334f, "half: one third");
    }
    // UTF-8
    {
        std::vector<uint32_t> cp = utf8_codepoints("a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
        r.check(cp.size() == 4 && cp[0] == 'a' && cp[1] == 0xE9 && cp[2] == 0x20AC && cp[3] == 0x1F600, "utf-8 decoding");
        std::vector<uint32_t> bad = utf8_codepoints("\xFF");
        r.check(bad.size() == 1 && bad[0] == 0xFFFD, "malformed utf-8 becomes U+FFFD");
    }
    // Canvas, fonts, clipping
    {
        Canvas cv(200, 120, 1.0f);
        cv.clear(rgb(0x101010));
        cv.fill({10, 10, 20, 20}, rgb(0xff0000));
        r.check(cv.px[10 * 200 + 10] == 0xff0000 && cv.px[29 * 200 + 29] == 0xff0000 && cv.px[30 * 200 + 30] == 0x101010, "fill covers exact rect");
        cv.push_clip({0, 0, 15, 15});
        cv.fill({0, 0, 100, 100}, rgb(0x00ff00));
        cv.pop_clip();
        r.check(cv.px[5 * 200 + 5] == 0x00ff00 && cv.px[20 * 200 + 20] == 0xff0000, "clip stops fill");
        r.check(cv.px[50 * 200 + 50] == 0x101010, "nothing drawn outside clip");
        cv.fill({40, 40, 10, 10}, rgb(0xffffff), 0.5f);
        Color mid = unpack(cv.px[45 * 200 + 45]);
        r.check(mid.r > 130 && mid.r < 140, "alpha blend mixes");
        int adv = cv.text(5, 60, "Token \xD0\xA2\xD0\xBE\xD0\xBA\xD0\xB5\xD0\xBD", rgb(0xffffff), F_UI);
        r.check(adv > 40 && adv < 140, "UTF-8 mixed text width plausible");
        bool lit = false;
        for (int y = 60; y < 80 && !lit; ++y)
            for (int x = 5; x < 5 + adv; ++x)
                if (cv.px[static_cast<size_t>(y) * 200 + x] != 0x101010) { lit = true; break; }
        r.check(lit, "text writes pixels");
        r.check(cv.fonts.glyph(F_UI, 0x422).w > 0, "Cyrillic glyph renders");
        r.check(cv.fonts.width(F_MONO, "iiii") == cv.fonts.width(F_MONO, "WWWW"), "monospace font is monospaced");
        cv.text_fit(5, 90, 40, "a long label that cannot fit", rgb(0xffffff), F_UI);
        bool beyond = false;
        for (int y = 90; y < 110; ++y)
            for (int x = 46; x < 200; ++x)
                if (cv.px[static_cast<size_t>(y) * 200 + x] != 0x101010) beyond = true;
        r.check(!beyond, "text_fit stays within width");
        Canvas h(200, 100, 1.0f);
        h.clear(rgb(0));
        h.heat({10, 10, 100, 10}, 1, 1000, [](int, int c) { return c == 503 ? 1.0f : 0.0f; }, [](float v) { return v > 0.5f ? rgb(0xffffff) : rgb(0x000000); });
        int bright = 0;
        for (int x = 10; x < 110; ++x) bright += h.px[10 * 200 + x] == 0xffffff;
        r.check(bright == 1, "heat preserves single-column peak");
    }
    // Heat call cost & max-pooling
    {
        Canvas cv(500, 500, 1.0f);
        cv.clear(rgb(0));
        int value_calls = 0, cmap_calls = 0;
        cv.heat({20, 20, 400, 400}, 32, 32, [&](int r0, int c0) { ++value_calls; return static_cast<float>(r0 * 32 + c0); },
                [&](float v) { ++cmap_calls; return rgb(static_cast<int>(v) / 4, static_cast<int>(v) % 256, 7); });
        r.check(value_calls == 32 * 32 && cmap_calls == 32 * 32, "heat asks each cell once");
        r.check(unpack(cv.px[static_cast<size_t>(88) * 500 + 113]).g == (5 * 32 + 7) % 256 && cv.px[19 * 500 + 19] == 0, "heat draws exact cell block");
        Canvas p(100, 20, 1.0f);
        p.clear(rgb(0));
        p.heat({0, 0, 10, 1}, 2, 40, [](int r0, int c0) { return r0 == 1 && c0 == 17 ? 5.0f : (r0 == 0 && c0 == 3 ? 2.0f : 0.0f); },
               [](float v) { return rgb(static_cast<int>(v * 40), 0, 0); });
        r.check(unpack(p.px[4]).r == 200 && unpack(p.px[0]).r == 80 && unpack(p.px[7]).r == 0, "heat max-pools per pixel");
    }
    // SSE2 line_add parity
    {
        const int W = 160, H = 120;
        Canvas fast(W, H, 1.0f), slow(W, H, 1.0f);
        fast.clear(rgb(0x101820)); slow.clear(rgb(0x101820));
        uint32_t seed = 12345;
        auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return static_cast<float>(seed >> 8) / 16777216.0f; };
        auto ref_line = [&](float x0, float y0, float x1, float y1, Color c, float a) {
            float dx = x1 - x0, dy = y1 - y0;
            bool steep = std::fabs(dy) > std::fabs(dx);
            if (steep) { std::swap(x0, y0); std::swap(x1, y1); }
            if (x0 > x1) { std::swap(x0, x1); std::swap(y0, y1); }
            float ddx = x1 - x0, grad = ddx < 1e-6f ? 0.0f : (y1 - y0) / ddx;
            int xs = static_cast<int>(std::floor(x0)), xe = static_cast<int>(std::floor(x1));
            float y = y0 + grad * (static_cast<float>(xs) - x0);
            for (int x = xs; x <= xe; ++x, y += grad) {
                int yi = static_cast<int>(std::floor(y));
                float f = y - static_cast<float>(yi);
                if (steep) { slow.add_px(yi, x, c, a * (1.0f - f)); slow.add_px(yi + 1, x, c, a * f); }
                else { slow.add_px(x, yi, c, a * (1.0f - f)); slow.add_px(x, yi + 1, c, a * f); }
            }
        };
        for (int i = 0; i < 40; ++i) {
            float x0 = 4 + rnd() * (W - 9), y0 = 4 + rnd() * (H - 9), x1 = 4 + rnd() * (W - 9), y1 = 4 + rnd() * (H - 9);
            if (i % 7 == 0) y1 = y0;
            if (i % 7 == 1) x1 = x0;
            Color c = rgb(static_cast<int>(rnd() * 255), static_cast<int>(rnd() * 255), static_cast<int>(rnd() * 255));
            float a = 0.02f + rnd() * 0.9f;
            fast.line_add(x0, y0, x1, y1, c, a);
            ref_line(x0, y0, x1, y1, c, a);
        }
        int worst = 0; double sum = 0;
        for (int y = 0; y < H - 1; ++y)
            for (int x = 0; x < W - 1; ++x) {
                uint32_t p = fast.px[static_cast<size_t>(y) * W + x], q = slow.px[static_cast<size_t>(y) * W + x];
                for (int sh = 0; sh <= 16; sh += 8) {
                    int diff = std::abs(static_cast<int>((p >> sh) & 255) - static_cast<int>((q >> sh) & 255));
                    worst = std::max(worst, diff); sum += diff;
                }
            }
        r.check(worst <= 4 && sum / (3.0 * (W - 1) * (H - 1)) < 0.05, "line_add matches float reference");
        Canvas sat(40, 20, 1.0f); sat.clear(rgb(0));
        for (int i = 0; i < 5; ++i) sat.line_add(5.0f, 10.0f, 30.0f, 10.0f, rgb(0xffffff), 1.0f);
        r.check(sat.px[10 * 40 + 15] == 0xffffff && sat.px[9 * 40 + 15] == 0, "line_add saturates cleanly");
        Canvas cut(60, 40, 1.0f); cut.clear(rgb(0));
        cut.push_clip({10, 10, 30, 20});
        cut.line_add(-30.0f, -10.0f, 100.0f, 70.0f, rgb(0xffffff), 1.0f);
        cut.line_add(25.0f, -50.0f, 25.0f, 90.0f, rgb(0xff00ff), 1.0f);
        cut.pop_clip();
        bool outside = false, inside = false;
        for (int y = 0; y < 40; ++y)
            for (int x = 0; x < 60; ++x) {
                const bool in = x >= 10 && x < 40 && y >= 10 && y < 30;
                if (cut.px[static_cast<size_t>(y) * 60 + x] != 0) (in ? inside : outside) = true;
            }
        r.check(inside && !outside, "line_add clips cleanly");
    }
    // PNG export
    {
        Canvas cv(64, 48, 1.0f);
        cv.clear(rgb(0x203040));
        cv.fill({5, 5, 20, 10}, rgb(0xff8000));
        cv.line(0, 0, 63, 47, rgb(0xffffff));
        std::string path = "selftest_core.png";
        cv.save_png(path);
        FILE* fh = std::fopen(path.c_str(), "rb");
        r.check(fh != nullptr, "PNG file written");
        if (fh) {
            std::vector<uint8_t> raw;
            uint8_t buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, fh)) > 0) raw.insert(raw.end(), buf, buf + n);
            std::fclose(fh);
            std::remove(path.c_str());
            auto be32 = [&](size_t at) { return (static_cast<uint32_t>(raw[at]) << 24) | (static_cast<uint32_t>(raw[at + 1]) << 16) | (static_cast<uint32_t>(raw[at + 2]) << 8) | raw[at + 3]; };
            r.check(raw.size() > 60 && be32(16) == 64 && be32(20) == 48, "PNG header valid");
            std::vector<uint8_t> idat;
            size_t at = 8; bool crc_ok = true, iend = false;
            while (at + 12 <= raw.size()) {
                uint32_t len = be32(at);
                std::string type(reinterpret_cast<const char*>(&raw[at + 4]), 4);
                crc_ok = crc_ok && crc32(0, &raw[at + 4], static_cast<uInt>(len + 4)) == be32(at + 8 + len);
                if (type == "IDAT") idat.insert(idat.end(), raw.begin() + static_cast<long>(at) + 8, raw.begin() + static_cast<long>(at) + 8 + len);
                if (type == "IEND") iend = true;
                at += 12 + len;
            }
            std::vector<uint8_t> pix(static_cast<size_t>(64 * 3 + 1) * 48);
            uLongf out = static_cast<uLongf>(pix.size());
            r.check(crc_ok && iend && uncompress(pix.data(), &out, idat.data(), static_cast<uLong>(idat.size())) == Z_OK && out == pix.size(), "PNG CRCs and zlib inflate");
        }
    }
    // Colors & 3D helpers
    {
        r.check(pack(surprisal(0)) == 0x26313f && pack(surprisal(100)) == 0xffffff, "surprisal ramp ends");
        r.check(pack(delta_error(0)) == 0x275a52 && pack(delta_error(1000)) == 0xff2a55, "delta_error ramp ends");
        r.check(pack(magma(0)) == 0x050510 && pack(magma(1)) == 0xfcfdbf && pack(magma(std::nan(""))) == 0x050510, "magma ramp ends and NaN");
        r.check(pack(diverging(0)) == 0x20262e && pack(diverging(-5)) == 0x3d8bfd && pack(diverging(5)) == 0xff7a3d, "diverging ramp");
        r.check(pack(similarity(1)) == 0xf4f1bb && pack(similarity(0)) == 0x1b2028, "similarity ramp");
        r.check(pack(parse_tint("#12ab34")) == 0x12ab34 && pack(parse_tint("cyan")) == 0x4fb0c6, "parse_tint hex and named");
    }
}

}  // namespace vitro
