// Binary container reader for Vitro / aligned tensor bundles:
//   bytes 0..7    8-byte magic (e.g. "VITRO01\n" or custom magic)
//   bytes 8..15   little-endian uint64 JSON header length
//   bytes 16..    UTF-8 JSON header, zero-padded to 16-byte alignment, then raw C-order tensors
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

namespace vitro {

inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;
        else {
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            bits = (sign << 31) | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct Arr {
    const uint8_t* p = nullptr;
    char dtype[3] = {0, 0, 0};
    std::vector<size_t> shape;
    size_t n = 0;

    bool empty() const { return p == nullptr; }
    size_t dim(size_t k) const { return k < shape.size() ? shape[k] : 1; }
    float f(size_t i) const {
        switch (dtype[0]) {
            case 'f':
                if (dtype[1] == '4') { float v; std::memcpy(&v, p + 4 * i, 4); return v; }
                else { uint16_t h; std::memcpy(&h, p + 2 * i, 2); return half_to_float(h); }
            case 'i':
                if (dtype[1] == '4') { int32_t v; std::memcpy(&v, p + 4 * i, 4); return static_cast<float>(v); }
                else if (dtype[1] == '2') { int16_t v; std::memcpy(&v, p + 2 * i, 2); return static_cast<float>(v); }
                else return static_cast<float>(static_cast<int8_t>(p[i]));
            default: return static_cast<float>(p[i]);
        }
    }
    int iv(size_t i) const {
        switch (dtype[0]) {
            case 'i':
                if (dtype[1] == '4') { int32_t v; std::memcpy(&v, p + 4 * i, 4); return v; }
                else if (dtype[1] == '2') { int16_t v; std::memcpy(&v, p + 2 * i, 2); return v; }
                else return static_cast<int8_t>(p[i]);
            case 'u': return p[i];
            default: return static_cast<int>(f(i));
        }
    }
};

inline Arr make_arr(const Json& d, const uint8_t* data, size_t data_size) {
    Arr a;
    const std::string& dt = d["dtype"].str();
    if (dt.size() != 2) throw std::runtime_error("bad dtype " + dt);
    a.dtype[0] = dt[0]; a.dtype[1] = dt[1];
    size_t item = 0;
    if (dt == "f4" || dt == "i4") item = 4;
    else if (dt == "f2" || dt == "i2") item = 2;
    else if (dt == "i1" || dt == "u1") item = 1;
    else throw std::runtime_error("unknown dtype " + dt);
    a.n = 1;
    for (size_t k = 0; k < d["shape"].size(); ++k) {
        size_t s = static_cast<size_t>(d["shape"].at(k).num());
        a.shape.push_back(s);
        a.n *= s;
    }
    double off = d["off"].num();
    if (off < 0 || static_cast<size_t>(off) % 16 || static_cast<size_t>(off) + a.n * item > data_size)
        throw std::runtime_error("array " + dt + " at " + std::to_string(static_cast<long long>(off)) + " lies outside the data section");
    a.p = data + static_cast<size_t>(off);
    return a;
}

class FileMap {
public:
    FileMap() = default;
    FileMap(const FileMap&) = delete;
    FileMap& operator=(const FileMap&) = delete;
    FileMap(FileMap&& o) noexcept { swap(o); }
    FileMap& operator=(FileMap&& o) noexcept {
        if (this != &o) { close(); swap(o); }
        return *this;
    }
    ~FileMap() { close(); }

    void open(const std::string& p) {
        close();
        int wl = MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, nullptr, 0);
        std::wstring wp(static_cast<size_t>(wl), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, &wp[0], wl);
        file_ = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + p + " (Windows error " + std::to_string(GetLastError()) + ")");
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(file_, &sz)) throw std::runtime_error("cannot size " + p);
        size_ = static_cast<size_t>(sz.QuadPart);
        if (size_ < 16) throw std::runtime_error(p + ": too short to be a trace container");
        map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_) throw std::runtime_error("cannot map " + p + " (Windows error " + std::to_string(GetLastError()) + ")");
        base_ = static_cast<const uint8_t*>(MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0));
        if (!base_) throw std::runtime_error("cannot map view of " + p + " (Windows error " + std::to_string(GetLastError()) + ")");
    }
    void close() {
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        base_ = nullptr; map_ = nullptr; file_ = INVALID_HANDLE_VALUE; size_ = 0;
    }
    const uint8_t* data() const { return base_; }
    size_t size() const { return size_; }

private:
    void swap(FileMap& o) noexcept {
        std::swap(file_, o.file_); std::swap(map_, o.map_);
        std::swap(base_, o.base_); std::swap(size_, o.size_);
    }
    HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
};

struct MappedContainer {
    FileMap file;
    Json head;
    const uint8_t* data = nullptr;
    size_t data_size = 0;

    void open(const std::string& p, const char* magic8, const std::string& wrong_magic_msg) {
        file.open(p);
        const uint8_t* base = file.data();
        const size_t size = file.size();
        if (std::memcmp(base, magic8, 8) != 0) throw std::runtime_error(p + ": " + wrong_magic_msg);
        uint64_t hl = 0;
        std::memcpy(&hl, base + 8, 8);
        if (hl == 0 || 16 + hl > size)
            throw std::runtime_error(p + ": the header says " + std::to_string(hl) + " bytes, the file has " + std::to_string(size - 16));
        head = parse_json(std::string(reinterpret_cast<const char*>(base + 16), static_cast<size_t>(hl)));
        if (head["version"].i() != 1)
            throw std::runtime_error(p + ": container version " + std::to_string(head["version"].i()) + ", expected 1");
        size_t pad = (16 - hl % 16) % 16;
        data = base + 16 + hl + pad;
        data_size = size - (16 + hl + pad);
    }

    Arr arr(const Json& d) const { return make_arr(d, data, data_size); }
};

}  // namespace vitro
