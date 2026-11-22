// What is built into vitro.exe: the web app, the sample traces, the WebView2 loader and the licence notices.
//
// The build (tools/make_app_resources.py) writes them as RCDATA resources, and resource 1 is a table that says what
// each of the others is: one line per item, "id <TAB> kind <TAB> mime <TAB> name", where kind is
//   asset   a file of the web app, served at /<name>
//   sample  a trace file that is listed in the library
//   loader  WebView2Loader.dll (the name is the SDK version)
//   notice  a text of licence notices
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vitro::app {

struct Blob {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

struct Item {
    int id = 0;
    std::string kind, mime, name;
};

// The bytes of resource `id`; an empty blob when there is no such resource. They live as long as the program.
inline Blob resource(int id) {
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(id), RT_RCDATA);
    if (!r) return {};
    HGLOBAL g = LoadResource(nullptr, r);
    if (!g) return {};
    return {static_cast<const uint8_t*>(LockResource(g)), static_cast<size_t>(SizeofResource(nullptr, r))};
}

// Every item the build embedded. A missing or malformed table means a broken build, so it throws.
inline std::vector<Item> table() {
    Blob b = resource(1);
    if (!b.data) throw std::runtime_error("this vitro.exe has no resource table: the build is broken");
    std::vector<Item> items;
    const std::string text(reinterpret_cast<const char*>(b.data), b.size);
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        if (line.empty()) continue;
        std::vector<std::string> f;
        size_t from = 0;
        for (int k = 0; k < 3; ++k) {
            size_t tab = line.find('\t', from);
            if (tab == std::string::npos) throw std::runtime_error("malformed line in the resource table: " + line);
            f.push_back(line.substr(from, tab - from));
            from = tab + 1;
        }
        f.push_back(line.substr(from));
        Item it;
        it.id = std::stoi(f[0]);
        it.kind = f[1];
        it.mime = f[2];
        it.name = f[3];
        items.push_back(it);
    }
    return items;
}

}  // namespace vitro::app
