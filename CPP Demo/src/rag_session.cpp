#include "rag_session.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <curl/curl.h>

namespace fs = std::filesystem;

RAGSessionManager::RAGSessionManager(std::string base_dir,
                                     std::string ollama_url,
                                     std::string embed_model,
                                     std::string llm_model)
: base_dir_(std::move(base_dir)),
  ollama_url_(std::move(ollama_url)),
  embed_model_(std::move(embed_model)),
  llm_model_(std::move(llm_model)) {
    fs::create_directories(base_dir_);
}

std::string RAGSessionManager::uuid4() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    auto to_hex = [](uint64_t x, int bytes){
        std::ostringstream oss; oss<<std::hex;
        for(int i=bytes-1;i>=0;--i){ oss<<((x>>(i*8))&0xFFu); }
        return oss.str();
    };
    uint64_t a=dis(gen), b=dis(gen);
    std::ostringstream oss;
    oss<<to_hex(a,8)<<to_hex(b,8);
    return oss.str().substr(0,32);
}

std::string RAGSessionManager::sanitizePath(const std::string& s) {
    std::string out = s;
    for(char& c: out){ if(c=='\\' || c=='/') c='_'; }
    return out;
}

std::vector<std::string> RAGSessionManager::findPDFs(const std::string& folder_path) {
    std::vector<std::string> pdfs;
    for (auto& p : fs::recursive_directory_iterator(folder_path)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension()==".pdf" || p.path().extension()==".PDF") {
            pdfs.push_back(p.path().string());
        }
    }
    return pdfs;
}

std::string RAGSessionManager::run_pdftotext(const std::string& pdf_path) {
    // Requires poppler-utils (pdftotext) installed in PATH.
    std::string tmp = fs::temp_directory_path() / fs::path(sanitizePath(pdf_path) + ".txt");
    std::ostringstream cmd;
#ifdef _WIN32
    cmd << "pdftotext \"" << pdf_path << "\" \"" << tmp << "\"";
#else
    cmd << "pdftotext \"" << pdf_path << "\" \"" << tmp << "\"";
#endif
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        throw std::runtime_error("pdftotext failed for: " + pdf_path);
    }
    std::ifstream ifs(tmp);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    fs::remove(tmp);
    return buffer.str();
}

std::vector<std::string> RAGSessionManager::split_chunks(const std::string& text, size_t chunk_size, size_t overlap) {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = std::min(start + chunk_size, text.size());
        chunks.emplace_back(text.substr(start, end - start));
        if (end == text.size()) break;
        start = end - std::min(overlap, end);
    }
    return chunks;
}

// CURL helpers
static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size*nmemb);
    return size*nmemb;
}

std::vector<float> RAGSessionManager::embed(const std::string& text) {
    CURL* curl = curl_easy_init();
    if(!curl) throw std::runtime_error("curl_easy_init failed");

    std::string url = ollama_url_ + "/api/embeddings";
    json payload = {
        {"model", embed_model_},
        {"prompt", text}
    };
    std::string payload_str = payload.dump();

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if(res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL error: ") + curl_easy_strerror(res));
    }

    auto j = json::parse(response);
    if (!j.contains("embedding")) {
        throw std::runtime_error("Ollama embeddings response missing 'embedding'");
    }
    std::vector<float> vec = j["embedding"].get<std::vector<float>>();
    return vec;
}

std::string RAGSessionManager::ollama_chat(const std::string& prompt) {
    CURL* curl = curl_easy_init();
    if(!curl) throw std::runtime_error("curl_easy_init failed");

    std::string url = ollama_url_ + "/api/chat";
    json payload = {
        {"model", llm_model_},
        {"messages", json::array({
            json{{"role","system"},{"content","You are a helpful assistant answering questions based on provided context."}},
            json{{"role","user"},{"content", prompt}}
        })},
        {"stream", false}
    };
    std::string payload_str = payload.dump();

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if(res != CURLE_OK) {
        throw std::runtime_error(std::string("CURL error: ") + curl_easy_strerror(res));
    }

    auto j = json::parse(response);
    if (!j.contains("message") || !j["message"].contains("content")) {
        throw std::runtime_error("Ollama chat response missing 'message.content'");
    }
    return j["message"]["content"].get<std::string>();
}

std::string RAGSessionManager::sessionDir(const std::string& session_id) const {
    return (fs::path(base_dir_) / session_id).string();
}

void RAGSessionManager::save_index(const SessionIndex& idx) const {
    fs::create_directories(sessionDir(idx.session_id));
    std::ofstream ofs(fs::path(sessionDir(idx.session_id)) / "index.json");
    json j;
    j["session_id"] = idx.session_id;
    j["chunks"] = json::array();
    for (const auto& c : idx.chunks) {
        j["chunks"].push_back({
            {"id", c.id},
            {"text", c.text},
            {"embedding", c.embedding}
        });
    }
    ofs << j.dump(2);
}

std::optional<SessionIndex> RAGSessionManager::load_index(const std::string& session_id) const {
    auto path = fs::path(sessionDir(session_id)) / "index.json";
    if (!fs::exists(path)) return std::nullopt;
    std::ifstream ifs(path);
    json j; ifs >> j;
    SessionIndex idx;
    idx.session_id = j.value("session_id", session_id);
    for (auto& cj : j["chunks"]) {
        Chunk c;
        c.id = cj.value("id","");
        c.text = cj.value("text","");
        c.embedding = cj.value("embedding", std::vector<float>{});
        idx.chunks.push_back(std::move(c));
    }
    return idx;
}

double RAGSessionManager::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size()!=b.size()) return -1.0;
    double dot=0.0, na=0.0, nb=0.0;
    for (size_t i=0;i<a.size();++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na==0 || nb==0) return -1.0;
    return dot / (std::sqrt(na)*std::sqrt(nb));
}

std::vector<size_t> RAGSessionManager::topk(const std::vector<double>& scores, int k) {
    std::vector<size_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+std::min(k,(int)idx.size()), idx.end(),
        [&](size_t i, size_t j){ return scores[i] > scores[j]; });
    if ((int)idx.size() > k) idx.resize(k);
    return idx;
}

std::string RAGSessionManager::build_prompt(const std::string& context, const std::string& question) {
    std::ostringstream oss;
    oss << "Answer the question based only on the context.\n\nContext:\n"
        << context << "\n\nQuestion:\n" << question
        << "\n\nAnswer concisely and accurately in three sentences or less.";
    return oss.str();
}

std::string RAGSessionManager::createSessionFromFolder(const std::string& folder_path) {
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        throw std::runtime_error("Folder does not exist: " + folder_path);
    }
    std::vector<std::string> pdfs = findPDFs(folder_path);
    if (pdfs.empty()) throw std::runtime_error("No PDFs found in: " + folder_path);

    SessionIndex idx;
    idx.session_id = uuid4();

    for (const auto& pdf : pdfs) {
        std::string text = run_pdftotext(pdf);
        auto chunks = split_chunks(text, 1024, 100);
        for (size_t i=0; i<chunks.size(); ++i) {
            Chunk c;
            c.id = sanitizePath(pdf) + "#" + std::to_string(i);
            c.text = std::move(chunks[i]);
            c.embedding = embed(c.text);
            idx.chunks.push_back(std::move(c));
        }
    }

    save_index(idx);
    return idx.session_id;
}

std::string RAGSessionManager::chat(const std::string& session_id,
                                    const std::string& message,
                                    int k,
                                    double score_threshold) {
    auto opt = load_index(session_id);
    if (!opt) throw std::runtime_error("Invalid or unknown session_id");

    auto& idx = *opt;
    // Embed query
    auto qvec = embed(message);

    // Score
    std::vector<double> scores(idx.chunks.size());
    for (size_t i=0;i<idx.chunks.size();++i) {
        scores[i] = cosine_similarity(qvec, idx.chunks[i].embedding);
    }
    // Filter by threshold
    std::vector<size_t> order(idx.chunks.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){ return scores[a] > scores[b]; });

    std::string context;
    int added = 0;
    for (size_t i : order) {
        if (scores[i] < score_threshold) break;
        context += idx.chunks[i].text + "\n\n";
        if (++added >= k) break;
    }
    if (context.empty()) {
        return "No relevant context found in the document to answer your question.";
    }

    std::string prompt = build_prompt(context, message);
    return ollama_chat(prompt);
}
