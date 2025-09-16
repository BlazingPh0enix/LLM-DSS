import pymupdf
import pymupdf4llm
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count

def process_single_pdf(pdf_file: Path, output_dir: Path):
    output_txt_path = output_dir / f"{pdf_file.stem}.txt"
    
    if output_txt_path.exists():
        print(f"Text for '{pdf_file.name}' already extracted. Skipping.")
        return

    print(f"--- Starting extraction for: {pdf_file.name} ---")
    start_time = time.time()
    
    doc = pymupdf.open(pdf_file)
    if doc.page_count == 0:
        doc.close()
        print(f"No pages in '{pdf_file.name}'. Skipping.")
        return

    all_markdown_text = []
    page_batch_size = 50
    
    print(f"Starting batched markdown conversion for {doc.page_count} pages...")
    for start_page in range(0, doc.page_count, page_batch_size):
        end_page = min(start_page + page_batch_size, doc.page_count)
        print(f"  - Converting pages {start_page + 1} to {end_page}...")
        try:
            batch_markdown = pymupdf4llm.to_markdown(doc, pages=list(range(start_page, end_page)))
            all_markdown_text.append(batch_markdown)
        except Exception as e:
            print(f"    - Failed to convert pages {start_page + 1}-{end_page}: {e}")
    
    full_text = "\n\n".join(all_markdown_text)
    
    # Save the extracted text to a file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    doc.close()
    end_time = time.time()
    print(f"-> Finished and saved to '{output_txt_path}'. Time taken: {end_time - start_time:.2f} seconds.\n")

def extract_and_save_text():
    corpus_dir = Path("./physics_corpus")
    output_dir = Path("./processed_text")
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in ./physics_corpus. Run data_load.py first.")

    num_processes = max(1, cpu_count() // 2)
    print(f"Starting parallel processing of {len(pdf_files)} PDFs with {num_processes} processes...")

    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_single_pdf, [(pdf_file, output_dir) for pdf_file in pdf_files])
    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    extract_and_save_text()
    print("All PDFs have been extracted and saved as text files.")