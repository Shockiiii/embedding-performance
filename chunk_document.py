# chunk_document.py

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split the input text into overlapping chunks.
    chunk_size: the target length of each chunk (in characters)
    overlap: number of characters that overlap between consecutive chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
