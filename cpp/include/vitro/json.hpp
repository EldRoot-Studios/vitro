// Strict JSON reader for trace dump headers.
//
//   vitro::Json j = vitro::parse_json(text);   // throws std::runtime_error with byte offset on malformed input
//   j["stages"][0]["count"].i()                // checked access: missing keys or wrong types throw explicitly
//
// Numbers are doubles (integers exact below 2^53), strings decode UTF-8 and \uXXXX surrogate pairs, objects
// preserve key order. Comments, trailing commas, and NaN/Inf literals are rejected.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vitro {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj };
    Type t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    static const char* type_name(Type tp) {
        static const char* names[] = {"null", "bool", "number", "string", "array", "object"};
        return names[tp];
    }
    void need(Type want, const char* what) const {
        if (t != want) throw std::runtime_error(std::string("json: ") + what + " is " + type_name(t) + ", expected " + type_name(want));
    }
    bool has(const char* key) const {
        if (t != Obj) return false;
        for (const auto& kv : o) if (kv.first == key) return true;
        return false;
    }
    const Json& operator[](const char* key) const {
        need(Obj, key);
        for (const auto& kv : o) if (kv.first == key) return kv.second;
        throw std::runtime_error(std::string("json: no key '") + key + "'");
    }
    const Json& at(size_t i) const {
        need(Arr, "an element");
        if (i >= a.size()) throw std::runtime_error("json: index " + std::to_string(i) + " past " + std::to_string(a.size()));
        return a[i];
    }
    size_t size() const {
        if (t == Arr) return a.size();
        if (t == Obj) return o.size();
        throw std::runtime_error(std::string("json: size of a ") + type_name(t));
    }
    double num() const { need(Num, "a value"); return n; }
    int i() const {
        need(Num, "a value");
        if (n != static_cast<double>(static_cast<long long>(n))) throw std::runtime_error("json: " + std::to_string(n) + " is not an integer");
        return static_cast<int>(n);
    }
    bool boolean() const { need(Bool, "a value"); return b; }
    const std::string& str() const { need(Str, "a value"); return s; }
    bool is_null() const { return t == Null; }

    int i_or(const char* key, int fallback) const { return has(key) && !(*this)[key].is_null() ? (*this)[key].i() : fallback; }
    double num_or(const char* key, double fallback) const { return has(key) && !(*this)[key].is_null() ? (*this)[key].num() : fallback; }
    bool bool_or(const char* key, bool fallback) const { return has(key) && !(*this)[key].is_null() ? (*this)[key].boolean() : fallback; }
    std::string str_or(const char* key, const std::string& fallback = "") const { return has(key) && !(*this)[key].is_null() ? (*this)[key].str() : fallback; }
};

namespace detail {

struct Parser {
    const char* p;
    const char* begin;
    const char* end;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("json: " + why + " at offset " + std::to_string(p - begin));
    }
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    void expect(char c) { if (!eat(c)) fail(std::string("expected '") + c + "'"); }

    static void put_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    }
    unsigned hex4() {
        if (end - p < 4) fail("short \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= end) fail("unterminated string");
            char c = *p++;
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (p >= end) fail("unterminated escape");
            char e = *p++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (end - p < 6 || p[0] != '\\' || p[1] != 'u') fail("lone high surrogate");
                        p += 2;
                        unsigned lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("bad low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail(std::string("bad escape \\") + e);
            }
        }
        return out;
    }
    Json value() {
        ws();
        if (p >= end) fail("unexpected end");
        Json j;
        char c = *p;
        if (c == '{') {
            ++p;
            j.t = Json::Obj;
            if (eat('}')) return j;
            do {
                ws();
                std::string key = string();
                expect(':');
                j.o.emplace_back(std::move(key), value());
            } while (eat(','));
            expect('}');
        } else if (c == '[') {
            ++p;
            j.t = Json::Arr;
            if (eat(']')) return j;
            do { j.a.push_back(value()); } while (eat(','));
            expect(']');
        } else if (c == '"') {
            j.t = Json::Str;
            j.s = string();
        } else if (c == 't' || c == 'f' || c == 'n') {
            auto lit = [&](const char* w) {
                size_t n = std::char_traits<char>::length(w);
                if (static_cast<size_t>(end - p) < n || std::string(p, n) != w) fail(std::string("expected ") + w);
                p += n;
            };
            if (c == 't') { lit("true"); j.t = Json::Bool; j.b = true; }
            else if (c == 'f') { lit("false"); j.t = Json::Bool; j.b = false; }
            else { lit("null"); j.t = Json::Null; }
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            char* stop = nullptr;
            j.n = std::strtod(p, &stop);
            if (stop == p || stop > end) fail("bad number");
            j.t = Json::Num;
            p = stop;
        } else {
            fail(std::string("unexpected character '") + c + "'");
        }
        return j;
    }
};

}  // namespace detail

inline Json parse_json(const char* text, size_t len) {
    detail::Parser ps{text, text, text + len};
    Json j = ps.value();
    ps.ws();
    if (ps.p != ps.end) ps.fail("text after the value");
    return j;
}
inline Json parse_json(const std::string& text) { return parse_json(text.data(), text.size()); }

inline std::vector<int> json_ints(const Json& j) {
    std::vector<int> v;
    v.reserve(j.size());
    for (size_t k = 0; k < j.size(); ++k) v.push_back(j.at(k).i());
    return v;
}
inline std::vector<std::string> json_strings(const Json& j) {
    std::vector<std::string> v;
    v.reserve(j.size());
    for (size_t k = 0; k < j.size(); ++k) v.push_back(j.at(k).str());
    return v;
}

}  // namespace vitro
