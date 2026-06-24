"""
ingestor.py
-----------
Handles loading and ingesting content into ChromaDB.

Supported sources:
  - Plain text strings
  - PDF folders  (with optional VLM-based OCR for tables/charts)
  - Images       (VLM OCR)
  - URLs         (web scraping)
"""

import re
import time
import base64
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image

from src.config import vectorstore, text_splitter, vlm


# ── VLM OCR ─────────────────────────────────────────────────────────────────

def vlm_extract_text(image_source, source_name: str = "image", max_retries: int = 2) -> str:
    """Convert an image (path or PIL Image) to text using the vision model."""
    if isinstance(image_source, (str, Path)):
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_source, Image.Image):
        buffered = BytesIO()
        image_source.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
    else:
        return ""

    image_data = base64.b64encode(image_bytes).decode("utf-8")

    from langchain_core.messages import HumanMessage
    prompt = (
        "You are a precise document OCR engine. Extract ALL text from this image.\n"
        "RULES:\n"
        "- For tables: output valid Markdown tables with | separators and header rows\n"
        "- For charts/graphs: describe axes, labels, data points, legends, and trends\n"
        "- Preserve reading order (top-to-bottom, left-to-right)\n"
        "- Do NOT add commentary, summaries, or conversational text\n"
        "- Output ONLY the extracted content"
    )
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
    ])

    for attempt in range(max_retries + 1):
        try:
            response = vlm.invoke([message])
            if response.content and len(response.content.strip()) > 20:
                return response.content
            print(f"⚠️  VLM short response for {source_name} (attempt {attempt + 1})")
        except Exception as e:
            print(f"⚠️  VLM failed for {source_name} (attempt {attempt + 1}): {e}")
        if attempt < max_retries:
            time.sleep(2)

    print(f"❌ VLM extraction failed after {max_retries + 1} attempts for {source_name}")
    return ""


def _detect_table_or_chart(text: str, page) -> bool:
    """Heuristic: decide whether a PDF page likely contains a table or chart."""
    if hasattr(page, "images") and len(page.images) > 0:
        return True
    words = text.split()
    if not words:
        return False
    num_count   = sum(1 for w in words if re.match(r'^[\d\.\,\$\%\-\+]+$', w))
    short_count = sum(1 for w in words if len(w) <= 3)
    if num_count / len(words) > 0.2 and len(words) > 15:
        return True
    if short_count / len(words) > 0.6 and len(words) > 20:
        return True
    return False


# ── Public ingestion functions ───────────────────────────────────────────────

def ingest_texts(texts: list[str]) -> None:
    """Ingest a list of plain-text strings directly into the vector store."""
    chunks = text_splitter.create_documents(texts)
    vectorstore.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} text chunks into ChromaDB")


def ingest_folder(root_folder: str, ocr_threshold: int = 50) -> None:
    """
    Recursively ingest all PDFs in a folder.
    Pages with little text or detected tables/charts are re-extracted via VLM.
    """
    root_path = Path(root_folder).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"⚠️  Folder not found: {root_folder}")
        return

    pdf_files = list(root_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {root_folder}")
        return

    print(f"📂 Found {len(pdf_files)} PDF files in {root_folder}")
    all_chunks: list[Document] = []

    for pdf_path in pdf_files:
        rel_dir     = pdf_path.resolve().parent.relative_to(root_path)
        folder_name = str(rel_dir) if str(rel_dir) != "." else root_path.name

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"⚠️  Could not read {pdf_path.name}: {e}")
            continue

        pages: list[Document] = []
        ocr_count = 0

        for i, page in enumerate(reader.pages):
            text             = page.extract_text() or ""
            extraction_method = "text"

            needs_ocr = len(text.strip()) < ocr_threshold or _detect_table_or_chart(text, page)
            if needs_ocr:
                if _detect_table_or_chart(text, page):
                    print(f"   📊 Table/chart on {pdf_path.name} p{i + 1} → VLM")
                try:
                    images   = convert_from_path(str(pdf_path), first_page=i + 1, last_page=i + 1, dpi=300)
                    vlm_text = vlm_extract_text(images[0], source_name=f"{pdf_path.name} p{i + 1}")
                    if vlm_text and len(vlm_text.strip()) > 20:
                        text              = vlm_text
                        extraction_method = "vlm_ocr"
                        ocr_count        += 1
                    else:
                        print(f"   ⚠️  VLM empty for {pdf_path.name} p{i + 1}, keeping standard text")
                except Exception as e:
                    print(f"⚠️  VLM failed for {pdf_path.name} p{i + 1}: {e}")

            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={
                        "source":            pdf_path.name,
                        "page":              i + 1,
                        "extraction_method": extraction_method,
                        "folder":            folder_name,
                    },
                ))

        if pages:
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"   📄 {pdf_path.name}: {len(reader.pages)} pages → {len(chunks)} chunks ({ocr_count} via VLM)")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} total chunks into ChromaDB")
    else:
        print("⚠️  No extractable text found.")


def ingest_images(image_paths: list[str]) -> None:
    """Ingest images by extracting their text via VLM OCR."""
    all_chunks: list[Document] = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print(f"⚠️  Missing file: {img_path}")
            continue
        text = vlm_extract_text(path, source_name=path.name)
        if text.strip():
            chunks = text_splitter.split_documents(
                [Document(page_content=text, metadata={"source": path.name, "extraction_method": "vlm_ocr"})]
            )
            all_chunks.extend(chunks)
            print(f"   🖼️  {path.name}: {len(chunks)} chunks via VLM")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} image chunks into ChromaDB")


def ingest_urls(urls: list[str], delay: float = 2.0) -> None:
    """Scrape and ingest web pages."""
    all_chunks: list[Document] = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url], header_template={"User-Agent": "Mozilla/5.0"})
            docs   = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"   🌐 {url}: {len(docs)} pages → {len(chunks)} chunks")
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️  Failed to scrape {url}: {e}")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} web chunks into ChromaDB")
