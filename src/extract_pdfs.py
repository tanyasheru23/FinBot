import fitz  # PyMuPDF
import os
from embed_image import is_relevant_image

def extract_pdf_content(pdf_path, img_dir):
    doc = fitz.open(pdf_path)
    page_chunks = []

    # To filter out very small images such as logos, icons, etc.
    min_size_bytes = 30000  # 30KB
    min_width = 750         # Minimum pixel width
    min_height = 750        # Minimum pixel height
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        images = []

        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            img_obj = doc.extract_image(xref)

            img_bytes = img_obj['image']
            width = img_obj.get('width', 0)
            height = img_obj.get('height', 0)

            # Skip images that are too small—likely doodles/icons/decorations
            if len(img_bytes) < min_size_bytes or (width < min_width and height < min_height):
                continue  # Skip: Don't save or use this image
            
            img_ext = img_obj["ext"]
            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num}_img{img_index}.{img_ext}"
            temp_img_path = os.path.join(img_dir, f"temp_{img_filename}")
            with open(temp_img_path, "wb") as img_file:
                img_file.write(img_bytes)
            
            # Content-aware filtering with CLIP
            if is_relevant_image(temp_img_path):
                # Final path to keep
                final_path = os.path.join(img_dir, img_filename)
                os.rename(temp_img_path, final_path)
                images.append(temp_img_path)
                print(f"Accepted image: {final_path}")
            else:
                os.remove(temp_img_path)
                print(f"Filtered out: {temp_img_path}")

        page_chunks.append({
            "text": text,
            "images": images,
            "source": os.path.basename(pdf_path),
            "page_num": page_num
        })
    return page_chunks

def batch_extract(directory, img_dir):
    all_chunks = []
    for fname in os.listdir(directory):
        if fname.lower().endswith('.pdf'):
            fpath = os.path.join(directory, fname)
            curr_chunks = extract_pdf_content(fpath, img_dir)
            all_chunks.extend(curr_chunks)
    return all_chunks
