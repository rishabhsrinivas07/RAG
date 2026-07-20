import re
import base64
import time
import pandas as pd
from datetime import datetime
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, UnstructuredExcelLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.config import vectorstore, text_splitter, vlm

# ==========================================================
# 📝 SHARED LOGGING HELPER
# ==========================================================
LOG_FILE_PATH = Path("extraction_log.txt")

def log_extraction(text: str, source: str, page_info: str, method: str):
    """Appends extracted text to the shared extraction log file."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"SOURCE: {source}\n")
        f.write(f"PAGE/ROW: {page_info}\n")
        f.write(f"METHOD: {method.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(text.strip() + "\n\n\n")


# ==========================================================
# 📄 PDF & VLM UTILITIES
# ==========================================================
def detect_table_or_chart(text: str, page) -> bool:
    if hasattr(page, 'images') and len(page.images) > 0:
        return True
    words = text.split()
    if not words:
        return False
    num_count = sum(1 for w in words if re.match(r'^[\d\.\,\$\%\-\+]+$', w))
    num_ratio = num_count / len(words)
    if num_ratio > 0.2 and len(words) > 15:
        return True
    short_word_count = sum(1 for w in words if len(w) <= 3)
    short_ratio = short_word_count / len(words)
    if short_ratio > 0.6 and len(words) > 20:
        return True
    return False


def vlm_extract_text(image_source, source_name="image", max_retries=2):
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
    prompt = (
        "You are a precise document OCR engine. Extract ALL text from this image.\n"
        "RULES:\n"
        "- For tables: output valid Markdown tables with | separators and header rows\n"
        "- For charts/graphs: describe axes, labels, data points, legends, and trends\n"
        "- Preserve reading order (top-to-bottom, left-to-right)\n"
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
        except Exception as e:
            print(f"⚠️ VLM attempt {attempt+1} failed: {e}", flush=True)
        if attempt < max_retries:
            time.sleep(2)
    return ""


# ==========================================================
# 📊 EXCEL INGESTION (UPGRADED FOR MULTI-SHEET & RAG)
# ==========================================================
def load_excel_with_fallback(file_path: str) -> list[Document]:
    """
    Reads ALL sheets in an Excel file. Formats each row into a 
    context-rich string so the embedding model can understand it.
    """
    try:
        # Ensure the engine is available
        import openpyxl 
        
        xls = pd.ExcelFile(file_path)
        documents = []
        
        # ✅ FIX 1: Loop through EVERY sheet in the Excel file
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Clean data types to prevent ChromaDB crashes
            df = df.fillna("N/A")
            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
                df[col] = df[col].astype(str)
                
            if df.empty:
                continue

            # ✅ FIX 2: Format rows so Embeddings understand them
            for index, row in df.iterrows():
                # Start with the sheet name for context
                row_parts = [f"Sheet: {sheet_name}"]
                
                # Add non-empty columns as Key-Value pairs
                for col in df.columns:
                    val = row[col]
                    if str(val).strip() and str(val) != "N/A":
                        row_parts.append(f"{col}: {val}")
                
                # Example output: "Sheet: Q3_Sales | Region: US | Revenue: 4.2M | Growth: 12%"
                row_text = " | ".join(row_parts)
                
                doc = Document(
                    page_content=row_text,
                    metadata={
                        "source": Path(file_path).name, 
                        "sheet": sheet_name, 
                        "row_index": index + 1,
                        "extraction_method": "excel_pandas"
                    }
                )
                documents.append(doc)
                
        print(f"✅ Success: Loaded {len(documents)} rows across {len(xls.sheet_names)} sheets.", flush=True)
        return documents

    except ImportError:
        print("❌ Missing 'openpyxl'. Run: pip install openpyxl", flush=True)
        return []
    except Exception as e:
        print(f"⚠️ Pandas failed ({e}). Excel file might be corrupted or heavily formatted.", flush=True)
        return []


def ingest_excel(file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ Skipping missing file: {file_path}", flush=True)
        return 0

    print(f"📊 Processing Excel file: {path.name}", flush=True)
    documents = load_excel_with_fallback(str(path))

    if documents:
        for doc in documents:
            log_extraction(
                text=doc.page_content,
                source=path.name,
                page_info=f"Sheet: {doc.metadata.get('sheet')} | Row: {doc.metadata.get('row_index')}",
                method="excel"
            )
        
        # Note: We DO NOT use text_splitter on Excel rows. 
        # Each row is already a perfect, self-contained "chunk".
        vectorstore.add_documents(documents)
        print(f"✅ Ingested {len(documents)} rows from {path.name}", flush=True)
        return len(documents)
    else:
        print(f"⚠️ No data extracted from {path.name}", flush=True)
        return 0


def ingest_excel(file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ Skipping missing file: {file_path}", flush=True)
        return 0

    print(f"📊 Processing Excel file: {path.name}", flush=True)
    documents = load_excel_with_fallback(str(path))

    if documents:
        for i, doc in enumerate(documents):
            doc.metadata["source"] = path.name
            doc.metadata["extraction_method"] = "excel"
            
            # Write to shared log
            log_extraction(
                text=doc.page_content,
                source=path.name,
                page_info=f"Row {i + 1}",
                method="excel"
            )
        
        chunks = text_splitter.split_documents(documents)
        
        # ✅ THE FIX: Automatically strip/convert datetime and other complex metadata
        chunks = filter_complex_metadata(chunks)
        
        vectorstore.add_documents(chunks)
        print(f"✅ Ingested {len(chunks)} chunks from {path.name}", flush=True)
        return len(chunks)
    else:
        print(f"⚠️ No documents extracted from {path.name}", flush=True)
        return 0

# ==========================================================
#  FOLDER (PDF) INGESTION
# ==========================================================
def ingest_texts(texts: list[str]):
    chunks = text_splitter.create_documents(texts)
    vectorstore.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} text chunks", flush=True)


def ingest_folder(root_folder: str, ocr_threshold: int = 50):
    root_path = Path(root_folder).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"⚠️ Folder not found: {root_folder}", flush=True)
        return

    pdf_files = list(root_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"️ No PDFs found in {root_folder}", flush=True)
        return

    print(f" Found {len(pdf_files)} PDFs", flush=True)
    all_chunks = []

    # Add a timestamp header to the log file
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"# NEW PDF INGESTION RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'#'*80}\n\n")

    for pdf_path in pdf_files:
        rel_dir = pdf_path.resolve().parent.relative_to(root_path)
        folder_name = str(rel_dir) if str(rel_dir) != "." else root_path.name
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"⚠️ Failed to read {pdf_path.name}: {e}", flush=True)
            continue

        pages = []
        ocr_count = 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extraction_method = "text"
            force_ocr = len(text.strip()) < ocr_threshold or detect_table_or_chart(text, page)

            if force_ocr:
                try:
                    images = convert_from_path(str(pdf_path), first_page=i+1, last_page=i+1, dpi=300)
                    vlm_text = vlm_extract_text(images[0], source_name=f"{pdf_path.name} page {i+1}")
                    if vlm_text and len(vlm_text.strip()) > 20:
                        text = vlm_text
                        extraction_method = "vlm_ocr"
                        ocr_count += 1
                except Exception as e:
                    print(f"⚠️ VLM failed on {pdf_path.name} page {i+1}: {e}", flush=True)

            if text and text.strip():
                # Write to shared log
                log_extraction(
                    text=text,
                    source=pdf_path.name,
                    page_info=f"Page {i + 1}",
                    method=extraction_method
                )
                
                pages.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": i + 1, "extraction_method": extraction_method, "folder": folder_name}
                ))

        if pages:
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"   📄 {pdf_path.name}: {len(chunks)} chunks ({ocr_count} via VLM)", flush=True)

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Total ingested: {len(all_chunks)} chunks", flush=True)
        print(f"📝 Logs appended to: {LOG_FILE_PATH.resolve()}", flush=True)


# ==========================================================
# 🖼️ IMAGE & WEB INGESTION
# ==========================================================
def ingest_images(image_paths: list[str]):
    all_chunks = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists(): 
            continue
        text = vlm_extract_text(path, source_name=path.name)
        if text and text.strip():
            doc = Document(page_content=text, metadata={"source": path.name, "extraction_method": "vlm_ocr"})
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} image chunks", flush=True)


def ingest_urls(urls: list[str], delay: float = 2.0):
    all_chunks = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url], header_template={"User-Agent": "Mozilla/5.0"})
            docs = loader.load()
            for doc in docs: 
                doc.metadata["source"] = url
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Failed to scrape {url}: {e}", flush=True)
    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} web chunks", flush=True)