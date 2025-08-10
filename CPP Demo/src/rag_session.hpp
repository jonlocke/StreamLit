#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

// Use nlohmann JSON; either via system package <nlohmann/json.hpp> or vendored include/json.hpp
#if __has_include(<nlohmann/json.hpp>)
  #include <nlohmann/json.hpp>
  using json = nlohmann::json;
#elif __has_include("json.hpp")
  #include "json.hpp"
  using json = nlohmann::json;
#else
  #error "nlohmann/json.hpp not found. Install nlohmann-json or place json.hpp into include/."
#endif

struct Chunk {
    std::string id;
    std::string text;
    std::vector<float> embedding;
};

struct SessionIndex {
    std::string session_id;
    std::vector<Chunk> chunks;
};

class RAGSessionManager {
public:
    explicit RAGSessionManager(std::string base_dir = "chroma_cpp",
                               std::string ollama_url = "http://localhost:11434",
                               std::string embed_model = "mxbai-embed-large",
                               std::string llm_model = "deepseek-r1:latest");

    // Create a session by ingesting all PDFs under folder (recursively).
    // Requires `pdftotext` to be installed on the system.
    // Returns session_id.
    std::string createSessionFromFolder(const std::string& folder_path);

    // Chat against an existing session.
    std::string chat(const std::string& session_id,
                     const std::string& message,
                     int k = 5,
                     double score_threshold = 0.2);

private:
    std::string base_dir_;
    std::string ollama_url_;
    std::string embed_model_;
    std::string llm_model_;

    static std::string uuid4();
    static std::string sanitizePath(const std::string& s);
    static std::vector<std::string> findPDFs(const std::string& folder_path);
    static std::string run_pdftotext(const std::string& pdf_path);
    static std::vector<std::string> split_chunks(const std::string& text, size_t chunk_size=1024, size_t overlap=100);

    // Networking
    std::vector<float> embed(const std::string& text);
    std::string ollama_chat(const std::string& prompt);

    // Persistence
    std::string sessionDir(const std::string& session_id) const;
    void save_index(const SessionIndex& idx) const;
    std::optional<SessionIndex> load_index(const std::string& session_id) const;

    // Retrieval
    static double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<size_t> topk(const std::vector<double>& scores, int k);

    static std::string build_prompt(const std::string& context, const std::string& question);
};
