<div align="center">
  <img src="frontend/public/window.svg" alt="Lexora AI Logo" width="120" height="120" />
  <h1>Lexora AI</h1>
  <p><strong>The Agentic Legal Copilot for Indian Startups</strong></p>
</div>

<br />

## 📖 Overview

The legal landscape for early-stage startups is fraught with complexity, expensive billable hours, and repetitive manual paperwork. **Lexora AI** is an advanced, agent-driven platform designed to democratize legal intelligence for founders. It shifts the paradigm from "filling out giant legal forms" to engaging in a dynamic, conversational workflow that intelligently structures and generates robust legal instruments tailored to Indian statutory requirements.

Whether it is drafting an airtight NDA, a complex ESOP policy, or analyzing a third-party vendor contract for hidden liabilities, Lexora AI serves as your always-available, tirelessly accurate digital legal counsel.

---

## 🤖 Why "Agentic"?

Traditional legal tech platforms are simply glorified typewriters—they offer static templates and expect the user to manually fill in dozens of repetitive fields. 

Lexora AI is built on an **Agentic Workflow**. We believe users shouldn't have to decipher legal jargon to protect their company. Instead of a linear form, Lexora utilizes a stateful orchestrator (powered by LangGraph) that:
1. **Understands Context:** Analyzes the startup's current state and jurisdiction.
2. **Adapts Dynamically:** Asks only the necessary, missing questions via a conversational interface.
3. **Applies Smart Defaults:** Infers non-critical information based on industry standards to minimize user friction.
4. **Iterates & Refines:** Allows users to interactively tweak clauses before finalizing the document.

This creates a "Human-in-the-Loop" copilot experience rather than a "Human-as-a-Data-Entry-Clerk" chore.

---

## ✨ Core Features & Capabilities

### 1. Multi-Instrument Document Generation
Lexora currently supports the generation of 5 critical startup documents, ensuring full parity and compliance with Indian corporate law:
*   **Non-Disclosure Agreements (NDAs)**
*   **Founder/Co-Founder Agreements**
*   **ESOP Policies (Employee Stock Ownership Plans)**
*   **Vendor / Service Agreements**
*   **Employment Agreements**

### 2. Intelligent Red Flag Analyser
Upload any third-party contract (PDF). Lexora extracts the text, chunks the clauses, and utilizes advanced LLMs to identify high-severity risks (e.g., infinite liability, asymmetric termination clauses, or IP assignment loopholes). It provides a severity score and plain-English mitigation advice.

### 3. RAG-Powered Clause Library
Documents are not generated via risky "zero-shot" hallucinations. Lexora relies on a **Retrieval-Augmented Generation (RAG)** pipeline. It retrieves vetted, standard legal clauses from a centralized ChromaDB vector store and seamlessly injects them into the generation context, ensuring the output is legally sound and contextually appropriate.

---

## 🏗️ Architecture & Advanced Technology

Lexora is built using a modern, decoupled architecture designed for scalability, speed, and AI integration.

### Frontend (Client-Side)
*   **Framework:** Next.js (App Router) + React
*   **Styling:** Tailwind CSS (Premium, Brutalist-inspired UI with dynamic micro-animations)
*   **Purpose:** Delivers a lightning-fast, highly interactive dashboard and conversational interface for founders.

### Backend (Server-Side)
*   **Framework:** FastAPI (Python)
*   **Database:** MongoDB (for stateful user data, document history, and audit logging)
*   **Vector Store:** ChromaDB (Local persistent RAG clause retrieval)
*   **LLM Orchestration:** LangChain & LangGraph (Stateful, cyclical agent routing)
*   **Document Processing:** PyMuPDF (Text extraction) & python-docx (Document generation)
*   **AI Models:** Supports flexible providers (Ollama for local privacy, OpenAI for cloud intelligence)

---

## 🔄 The System Workflow

1.  **Intake & Authentication:** Users securely log in (JWT) and select their desired legal objective (Generate or Analyze).
2.  **Context Gathering (The Agent):** The backend initializes a `LegalState` Pydantic model. The agent evaluates missing data and prompts the user via the frontend chat interface.
3.  **Retrieval (RAG):** Once sufficient context is gathered, the system queries the local ChromaDB to find the most appropriate legal clauses based on the user's answers.
4.  **Synthesis:** The LLM integrates the retrieved clauses with the user's specific state to draft a bespoke document.
5.  **Review & Export:** The generated document is presented to the user for review alongside a "Clause Guide" explaining complex terms. The user can export the final instrument as a `.pdf` or `.docx`.

---

## 🌍 Impact & Usefulness

Lexora AI directly impacts the bottom line and operational velocity of startups:
*   **Cost Reduction:** Saves thousands of dollars in preliminary legal drafting and review fees.
*   **Speed to Execution:** Reduces the time to draft an ESOP policy or Founder Agreement from weeks to minutes.
*   **Risk Mitigation:** The Red Flag Analyser acts as a crucial safety net, preventing founders from signing predatory contracts before escalating to human counsel.
*   **Empowerment:** Translates opaque legal jargon into actionable business intelligence.

---

## 🚀 Getting Started

### Prerequisites
*   Node.js (v18+)
*   Python (3.10+)
*   MongoDB (Running locally or Atlas)

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env      # Update your LLM and MongoDB credentials
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The platform will now be accessible at `http://localhost:3000`.

---
<div align="center">
  <i>Built with precision for the modern founder.</i>
</div>
