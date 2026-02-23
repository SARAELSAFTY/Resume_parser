# Resume Parser and Job Proposal Comparison System

A smart system that automatically extracts and structures key information from resumes including skills, experience, and education. It then compares parsed candidate data against job proposals, scoring compatibility and highlighting strengths and gaps. This enables recruiters to quickly shortlist the best-fit candidates with minimal manual effort.

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [API Endpoints](#api-endpoints)
5. [File Support](#file-support)
6. [Models & AI Components](#models--ai-components)
7. [Scoring & Matching Logic](#scoring--matching-logic)
8. [Configuration & Limits](#configuration--limits)
9. [Error Handling](#error-handling)
10. [Setup & Installation](#setup--installation)

---

## Overview

This application is a **FastAPI-based backend service** that provides intelligent resume (CV) analysis and job matching capabilities. It combines **OCR**, **Natural Language Processing (NLP)**, and **semantic similarity** to help recruiters evaluate how well a candidate's resume aligns with a given job proposal.

The system exposes a REST API consumed by three HTML front-end pages:
- `/` — Home / landing page
- `/ats` — ATS (Applicant Tracking System) scoring interface
- `/job_proposal` — Job proposal comparison interface

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI |
| OCR Engine | Microsoft TrOCR (`trocr-base-printed`) |
| Semantic Similarity | Sentence Transformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | PyPDF2 |
| Word Document Parsing | python-docx |
| Image Processing | Pillow (PIL) |
| ML Backend | Hugging Face Transformers + PyTorch |

---

## System Architecture

```
Client (Browser)
      │
      ▼
  FastAPI App
      │
      ├──► /ats_score       → ATS Keyword Scoring
      ├──► /compare         → CV vs Job Document Similarity
      ├──► /match_job       → Structured Job Matching
      ├──► /extract_cv      → CV Text Extraction
      └──► /extract_job     → Job Description Text Extraction
                │
                ▼
     ┌──────────────────────┐
     │   Document Parser    │
     │  PDF | DOCX | Image  │
     └──────────────────────┘
                │
     ┌──────────┴──────────┐
     │                     │
  OCR Engine        Sentence Transformer
  (TrOCR)           (all-MiniLM-L6-v2)
     │                     │
  Image Text         Semantic Embeddings
  Extraction         & Cosine Similarity
```

### Model Lifecycle

Models are loaded **once at startup** using FastAPI's `lifespan` context manager and cleaned up on shutdown, ensuring efficient memory usage:

- `SentenceTransformer` — for semantic similarity scoring
- `TrOCRProcessor` + `VisionEncoderDecoderModel` — for image-based OCR

---

## API Endpoints

### `GET /`
Serves the main `index.html` home page.

---

### `GET /ats`
Serves the ATS scoring front-end page (`ats.html`).

---

### `GET /job_proposal`
Serves the job proposal comparison page (`job_proposal.html`).

---

### `POST /ats_score`

**Description:** Computes an ATS (Applicant Tracking System) score for a given CV text by measuring its semantic similarity to a set of ATS-relevant keywords.

**Form Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `cv_text` | `string` | Yes | The full text of the candidate's CV |

**ATS Keywords Used:** `experience`, `skills`, `education`, `certifications`, `achievements`

**Sample Response:**
```json
{
  "status": "success",
  "ats_score": 72.45,
  "max_score": 100,
  "keywords_matched": 5
}
```

---

### `POST /compare`

**Description:** Compares a CV text against an uploaded job description document (PDF, Word, or Image) and returns a semantic similarity score.

**Form Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `cv_text` | `string` | Yes | Full text of the CV |
| `image` | `file` | Yes | Job description file (PDF, DOCX, or image) |

**Sample Response:**
```json
{
  "status": "success",
  "similarity_score": 68.32,
  "max_score": 100,
  "job_text": "We are looking for a software engineer...",
  "cv_length": 1240,
  "job_text_length": 850
}
```

---

### `POST /match_job`

**Description:** Performs a structured multi-dimensional match between a CV and specific job criteria including job title, experience level, and required skills. Optionally accepts a job description file for an overall similarity score.

**Form Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `cv_text` | `string` | Yes | Full text of the CV |
| `job_title` | `string` | No | Target job title |
| `experience_level` | `string` | No | One of: `entry`, `junior`, `mid`, `senior`, `lead` |
| `required_skills` | `string` | No | Comma-separated list of required skills |
| `image` | `file` | No | Optional job description document |

**Experience Level Keywords:**

| Level | Matched Keywords |
|---|---|
| entry | entry, junior, 0-2, beginner, fresh |
| junior | junior, 2-4, entry, associate |
| mid | mid, 4-7, intermediate, senior |
| senior | senior, 7-10, lead, principal, expert |
| lead | lead, principal, 10+, senior, architect, director |

**Sample Response:**
```json
{
  "status": "success",
  "cv_length": 1240,
  "matches": {
    "job_title": { "required": "Backend Developer", "score": 100, "found": true },
    "experience_level": { "required": "senior", "score": 80.0, "found": true },
    "skills": {
      "required": ["python", "fastapi", "docker"],
      "found": ["python", "fastapi"],
      "score": 66.7,
      "matched_count": 2,
      "total_count": 3
    }
  },
  "overall_match_score": 82.2
}
```

---

### `POST /extract_cv`

**Description:** Extracts and returns text from an uploaded CV file.

**Form Parameters:**

| Parameter | Type | Required |
|---|---|---|
| `file` | `file` | Yes |

**Sample Response:**
```json
{
  "status": "success",
  "filename": "resume.pdf",
  "file_type": "application/pdf",
  "cv_text": "John Doe | Software Engineer...",
  "text_length": 1540
}
```

---

### `POST /extract_job`

**Description:** Extracts and returns text from an uploaded job description file.

**Form Parameters:**

| Parameter | Type | Required |
|---|---|---|
| `file` | `file` | Yes |

**Sample Response:**
```json
{
  "status": "success",
  "filename": "job_description.docx",
  "file_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "job_text": "We are hiring a Senior Python Developer...",
  "text_length": 920
}
```

---

## File Support

| File Type | MIME Type | Extraction Method |
|---|---|---|
| PDF | `application/pdf` | PyPDF2 — page-by-page text extraction |
| DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | python-docx — paragraph iteration |
| DOC | `application/msword` | python-docx |
| JPEG | `image/jpeg` | Microsoft TrOCR OCR |
| PNG | `image/png` | Microsoft TrOCR OCR |
| WebP | `image/webp` | Microsoft TrOCR OCR |

---

## Models & AI Components

### Sentence Transformer — `all-MiniLM-L6-v2`

Used for computing **semantic similarity** between two pieces of text (e.g., CV vs job description). It encodes texts into dense vector embeddings and computes cosine similarity, producing a score between 0–100.

- Lightweight and fast
- Suitable for sentence-level and short-paragraph comparison
- Runs entirely locally

### TrOCR — `microsoft/trocr-base-printed`

Used for **optical character recognition** on image-based resumes or job descriptions. It converts image files into text using a Vision Encoder-Decoder Transformer architecture optimized for printed text.

- Handles clear, printed documents well
- Best suited for scanned or photographed files

---

## Scoring & Matching Logic

### ATS Score
The CV text is encoded into an embedding vector, and the cosine similarity against each ATS keyword embedding is computed. The mean similarity score across all keywords is multiplied by 100 to produce a 0–100 score.

### Similarity Score (`/compare`)
Both the CV text and the extracted job description text are encoded. The cosine similarity between the two embeddings is computed and scaled to 0–100.

### Structured Match Score (`/match_job`)
Three independent sub-scores are computed:

- **Job Title Match** — `100` if exact title found in CV, `50` if any word matches, `0` otherwise
- **Experience Level Match** — Keyword frequency scoring based on predefined keyword lists per level
- **Skills Match** — Percentage of required skills found (case-insensitive substring match)

The **overall match score** is the arithmetic mean of all computed sub-scores.

---

## Configuration & Limits

| Setting | Value |
|---|---|
| Max file size | 10 MB |
| Min CV text length | 10 characters |
| Max CV text length | 50,000 characters |
| Allowed image types | JPEG, PNG, JPG, WebP |
| Allowed document types | PDF, DOCX, DOC |

---

## Error Handling

The API uses standard HTTP status codes for error responses:

| Code | Meaning |
|---|---|
| `400 Bad Request` | Invalid file type, empty file, text too short/long |
| `413 Request Entity Too Large` | File exceeds 10MB limit |
| `404 Not Found` | HTML file not found on server |
| `500 Internal Server Error` | Model inference error or unexpected exception |

All error responses follow the FastAPI standard format:
```json
{
  "detail": "Error description here"
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js (optional, for front-end tooling)

### Install Dependencies

```bash
pip install fastapi uvicorn transformers sentence-transformers pillow PyPDF2 python-docx torch
```

### Run the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Required HTML Files

Place the following HTML files in the same directory as `app.py`:

- `index.html` — Home page
- `ats.html` — ATS scoring page
- `job_proposal.html` — Job comparison page

### Model Downloads

On first run, the following models will be automatically downloaded from Hugging Face:

- `sentence-transformers/all-MiniLM-L6-v2` (~90MB)
- `microsoft/trocr-base-printed` (~400MB)

Ensure internet access is available on the first launch.

---

*Documentation generated for `app.py` — CV/ATS Analyzer v1.0.0*
