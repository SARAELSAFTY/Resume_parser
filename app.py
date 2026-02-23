from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document
except ImportError:
    Document = None

# Global variables for models
ats_model = None
ocr_processor = None
ocr_model = None

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_PDF_TYPES = {"application/pdf"}
ALLOWED_WORD_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"}
ALLOWED_DOCUMENT_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES | ALLOWED_WORD_TYPES
MIN_CV_LENGTH = 10
MAX_CV_LENGTH = 50000
ATS_KEYWORDS = ["experience", "skills", "education", "certifications", "achievements"]

def validate_cv_text(cv_text: str):
    if not cv_text or len(cv_text.strip()) < MIN_CV_LENGTH:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"CV text must be at least {MIN_CV_LENGTH} characters")
    if len(cv_text) > MAX_CV_LENGTH:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"CV text cannot exceed {MAX_CV_LENGTH} characters")

def validate_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file provided")
    if file.content_type not in ALLOWED_DOCUMENT_TYPES:
        allowed = ", ".join(ALLOWED_DOCUMENT_TYPES)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid file type. Allowed types: {allowed}")
    file_data = file.file.read()
    if len(file_data) > MAX_FILE_SIZE:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File size exceeds {MAX_FILE_SIZE / (1024*1024):.0f}MB limit")
    return file_data

def serve_html(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(404, f"{filename} not found")
    with open(filename, "r") as f:
        return f.read()

def compute_similarity(text1: str, text2: str):
    try:
        emb1 = ats_model.encode(text1, convert_to_tensor=True)
        emb2 = ats_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item() * 100
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Error computing similarity: {str(e)}")

def extract_text(file: UploadFile, key: str) -> Dict[str, Any]:
    file_data = validate_file(file)
    text = extract_text_from_document(file_data, file.content_type)
    if not text.strip():
        raise ValueError("No text extracted from file")
    return {
        "status": "success",
        "filename": file.filename,
        "file_type": file.content_type,
        key: text,
        "text_length": len(text)
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ats_model, ocr_processor, ocr_model
    try:
        ats_model = SentenceTransformer('all-MiniLM-L6-v2')
        ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    except Exception as e:
        raise
    
    yield
    
    del ats_model
    del ocr_processor
    del ocr_model

app = FastAPI(title="CV/ATS Analyzer", version="1.0.0", lifespan=lifespan)

def extract_text_from_pdf(pdf_data: bytes) -> str:
    """Extract text from PDF file"""
    if PdfReader is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF support not installed. Install PyPDF2: pip install PyPDF2"
        )
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting text from PDF: {str(e)}"
        )

def extract_text_from_word(word_data: bytes) -> str:
    """Extract text from Word file (.docx or .doc)"""
    if Document is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Word support not installed. Install python-docx: pip install python-docx"
        )
    try:
        doc = Document(io.BytesIO(word_data))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting text from Word file: {str(e)}"
        )

def extract_text_from_image(image_data: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        pixel_values = ocr_processor(img, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {str(e)}"
        )

def extract_text_from_document(file_data: bytes, content_type: str) -> str:
    """Extract text from any supported document type (PDF, Word, Image)"""
    if content_type in ALLOWED_PDF_TYPES:
        return extract_text_from_pdf(file_data)
    elif content_type in ALLOWED_WORD_TYPES:
        return extract_text_from_word(file_data)
    elif content_type in ALLOWED_IMAGE_TYPES:
        return extract_text_from_image(file_data)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type}"
        )

@app.get("/", response_class=HTMLResponse)
def root():
    return serve_html("index.html")

@app.get("/ats", response_class=HTMLResponse)
def get_ats():
    return serve_html("ats.html")

@app.get("/job_proposal", response_class=HTMLResponse)
def get_job_proposal():
    return serve_html("job_proposal.html")

@app.post("/ats_score")
def ats_score(cv_text: str = Form(...)) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        
        # Encode and compute similarities
        cv_embedding = ats_model.encode(cv_text, convert_to_tensor=True)
        keyword_embeddings = ats_model.encode(ATS_KEYWORDS, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(cv_embedding, keyword_embeddings)
        score = similarities.mean().item() * 100
        
        return {
            "status": "success",
            "ats_score": round(score, 2),
            "max_score": 100,
            "keywords_matched": len(ATS_KEYWORDS)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing CV: {str(e)}"
        )

@app.post("/compare")
def compare(cv_text: str = Form(...), image: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        
        file_data = validate_file(image)
        
        # Extract text from document (PDF, Word, or Image)
        job_text = extract_text_from_document(file_data, image.content_type)
        
        if not job_text.strip():
            raise ValueError("No text extracted from file")
        
        # Compute similarity
        similarity = compute_similarity(cv_text, job_text)
        
        return {
            "status": "success",
            "similarity_score": round(similarity, 2),
            "max_score": 100,
            "job_text": job_text,
            "cv_length": len(cv_text),
            "job_text_length": len(job_text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/match_job")
def match_job(
    cv_text: str = Form(...),
    job_title: str = Form(""),
    experience_level: str = Form(""),
    required_skills: str = Form(""),
    image: UploadFile = File(None)
) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        
        cv_lower = cv_text.lower()
        results = {
            "status": "success",
            "cv_length": len(cv_text),
            "matches": {}
        }
        
        # Job title matching
        if job_title.strip():
            job_title_lower = job_title.lower().strip()
            title_score = 0
            if job_title_lower in cv_lower:
                title_score = 100
            elif any(word in cv_lower for word in job_title_lower.split()):
                title_score = 50  # Partial match
            results["matches"]["job_title"] = {
                "required": job_title,
                "score": title_score,
                "found": title_score > 0
            }
        
        # Experience level matching (basic keyword analysis)
        if experience_level:
            exp_keywords = {
                "entry": ["entry", "junior", "0-2", "beginner", "fresh"],
                "junior": ["junior", "2-4", "entry", "associate"],
                "mid": ["mid", "4-7", "intermediate", "senior"],
                "senior": ["senior", "7-10", "lead", "principal", "expert"],
                "lead": ["lead", "principal", "10+", "senior", "architect", "director"]
            }
            
            exp_score = 0
            if experience_level in exp_keywords:
                keywords = exp_keywords[experience_level]
                matches = sum(1 for keyword in keywords if keyword in cv_lower)
                exp_score = min(100, (matches / len(keywords)) * 100)
            
            results["matches"]["experience_level"] = {
                "required": experience_level,
                "score": round(exp_score, 1),
                "found": exp_score > 30
            }
        
        # Skills matching
        if required_skills.strip():
            skills_list = [skill.strip().lower() for skill in required_skills.split(",") if skill.strip()]
            skills_found = []
            skills_score = 0
            
            for skill in skills_list:
                if skill in cv_lower:
                    skills_found.append(skill)
            
            if skills_list:
                skills_score = (len(skills_found) / len(skills_list)) * 100
            
            results["matches"]["skills"] = {
                "required": skills_list,
                "found": skills_found,
                "score": round(skills_score, 1),
                "matched_count": len(skills_found),
                "total_count": len(skills_list)
            }
        
        # Overall similarity score (if image provided)
        if image and image.filename:
            file_data = validate_file(image)
            
            job_text = extract_text_from_document(file_data, image.content_type)
            if job_text.strip():
                similarity = compute_similarity(cv_text, job_text)
                results["similarity_score"] = round(similarity, 2)
                results["job_text"] = job_text
                results["job_text_length"] = len(job_text)
        
        # Calculate overall match score
        match_scores = [match["score"] for match in results["matches"].values() if "score" in match]
        if match_scores:
            results["overall_match_score"] = round(sum(match_scores) / len(match_scores), 1)
        else:
            results["overall_match_score"] = 0
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/extract_cv")
def extract_cv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        return extract_text(file, "cv_text")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Error extracting CV: {str(e)}")

@app.post("/extract_job")
def extract_job(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        return extract_text(file, "job_text")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Error extracting job description: {str(e)}")