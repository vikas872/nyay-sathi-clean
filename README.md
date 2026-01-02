---
title: Nyay Sathi
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# ⚖️ Nyay Sathi - Indian Legal Assistant

Nyay Sathi is an AI-powered legal assistant designed to interpret Indian laws and provide accurate, citation-backed answers.

## Features

- **📚 Agentic RAG**: Intelligently switches between local database search and web search.
- **🇮🇳 Local Knowledge**: Embedded knowledge of Indian Penal Code (IPC), BNS, and Constitution.
- **🌐 Web Search**: Access to trusted `.gov.in` sources via stealth browser automation.
- **✅ Citations**: All answers are grounded in trusted sources with citations.

## API Usage

This Space provides a backend API.

### Endpoint: `/ask`
```bash
curl -X POST "https://huggingface.co/spaces/YOUR_USERNAME/nyay-sathi/ask" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -d '{"question": "How to file an RTI?"}'
```

---
*Disclaimer: For educational purposes only. Always consult a qualified lawyer for legal advice.*
