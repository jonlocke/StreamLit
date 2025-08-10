#include "rag_session.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  rag_demo ingest <folder>\n"
                  << "  rag_demo chat <session_id> <question>\n";
        return 1;
    }

    try {
        RAGSessionManager mgr; // defaults to localhost:11434 Ollama
        std::string cmd = argv[1];
        if (cmd == "ingest") {
            if (argc < 3) { std::cerr << "Provide folder path\n"; return 1; }
            std::string folder = argv[2];
            std::string sid = mgr.createSessionFromFolder(folder);
            std::cout << "Session ID: " << sid << std::endl;
        } else if (cmd == "chat") {
            if (argc < 4) { std::cerr << "Provide session_id and question\n"; return 1; }
            std::string sid = argv[2];
            std::string question;
            for (int i=3;i<argc;++i) {
                if (i>3) question += " ";
                question += argv[i];
            }
            std::string answer = mgr.chat(sid, question);
            std::cout << "Answer: " << answer << std::endl;
        } else {
            std::cerr << "Unknown command\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}
