// The list of trace files the app knows, the same one `vitro web` serves at /api/library: the files named on the
// command line, then the sample traces built into the program. A file that cannot be read is listed with its error,
// never left out.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "app_assets.hpp"
#include "vitro/container.hpp"

namespace vitro::app {

struct Entry {
    std::string name;     // what the person sees: the file name
    std::string shown;    // how a message names the file: the path as it was given
    std::string source;   // "argument" or "sample"
    std::wstring path;    // on disk; empty for a sample built into the program
    int resource_id = 0;  // the sample's resource; 0 for a file on disk
};

inline std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

inline std::wstring wide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

// The facts of a trace file that the library shows, read from its container header.
struct Info {
    std::string kind, trace_id, title, model, arch;
    uint64_t size = 0;
};

// Reads the header of a container held in memory and refuses it in the words of python/vitro/container.py.
inline Info describe_bytes(const uint8_t* data, size_t size, const std::string& shown, const std::string& stem) {
    if (size < 16) throw std::runtime_error(shown + ": truncated file (less than 16 bytes)");
    if (std::memcmp(data, "VITRO01\n", 8) != 0) throw std::runtime_error(shown + ": not a Vitro trace file");
    uint64_t hl = 0;
    std::memcpy(&hl, data + 8, 8);
    if (hl == 0 || 16 + hl > size)
        throw std::runtime_error(shown + ": header says " + std::to_string(hl) + " bytes, file has " + std::to_string(size - 16));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, reinterpret_cast<const char*>(data + 16), static_cast<int>(hl), nullptr, 0) == 0)
        throw std::runtime_error(shown + ": header is not valid UTF-8");
    Json head;
    try {
        head = parse_json(reinterpret_cast<const char*>(data + 16), static_cast<size_t>(hl));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(shown + ": header is not valid JSON (" + e.what() + ")");
    }
    if (head.t != Json::Obj) throw std::runtime_error(shown + ": header must be a JSON object");
    if (!head.has("version") || head["version"].t != Json::Num || head["version"].n != 1) {
        const std::string got = head.has("version") && head["version"].t == Json::Num ? std::to_string(static_cast<long long>(head["version"].n)) : "missing";
        throw std::runtime_error(shown + ": container version " + got + ", expected 1");
    }
    Info info;
    info.size = size;
    info.kind = head.str_or("kind", "dump");
    const Json* trace = head.has("trace") && head["trace"].t == Json::Obj ? &head["trace"] : nullptr;
    info.trace_id = trace ? trace->str_or("id", "") : "";
    info.title = trace ? trace->str_or("title", stem) : stem;
    info.model = trace ? trace->str_or("model_name", "") : "";
    info.arch = trace ? trace->str_or("arch", "") : "";
    return info;
}

// The header facts of an entry, from its file on disk or from the program's own resources.
inline Info describe(const Entry& e) {
    const size_t dot = e.name.rfind('.');
    const std::string stem = dot == std::string::npos ? e.name : e.name.substr(0, dot);
    if (e.resource_id != 0) {
        Blob b = resource(e.resource_id);
        return describe_bytes(b.data, b.size, e.shown, stem);
    }
    WIN32_FILE_ATTRIBUTE_DATA attrs;
    if (GetFileAttributesExW(e.path.c_str(), GetFileExInfoStandard, &attrs) && attrs.nFileSizeHigh == 0 && attrs.nFileSizeLow < 16)
        throw std::runtime_error(e.shown + ": truncated file (less than 16 bytes)");  // too small to map
    FileMap map;
    map.open(utf8(e.path));
    return describe_bytes(map.data(), map.size(), e.shown, stem);
}

inline void put_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

// The body of /api/library: {"files": [...]} with the same fields as `vitro web`.
inline std::string library_json(const std::vector<Entry>& entries) {
    std::string out = "{\"files\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i) out += ',';
        const Entry& e = entries[i];
        out += "{\"id\":" + std::to_string(i) + ",\"name\":";
        put_string(out, e.name);
        try {
            const Info info = describe(e);
            out += ",\"kind\":";
            put_string(out, info.kind);
            out += ",\"size\":" + std::to_string(info.size) + ",\"traceId\":";
            put_string(out, info.trace_id);
            out += ",\"title\":";
            put_string(out, info.title);
            out += ",\"model\":";
            put_string(out, info.model);
            out += ",\"arch\":";
            put_string(out, info.arch);
            out += ",\"source\":";
            put_string(out, e.source);
        } catch (const std::runtime_error& err) {
            out += ",\"source\":";
            put_string(out, e.source);
            out += ",\"error\":";
            put_string(out, err.what());
        }
        out += '}';
    }
    return out + "]}";
}

}  // namespace vitro::app
