"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from urllib.parse import urlparse
import re
import time

from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel, embed_content

load_dotenv()

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in environment variables.")
configure(api_key=GEMINI_API_KEY)

# Remove fallback for psycopg2: require it for runtime, only fallback for type checkers
import psycopg2
from psycopg2.extras import execute_batch, Json, RealDictCursor

def get_postgres_conn():
    """
    Get a Postgres connection using environment variables.
    Returns:
        psycopg2 connection instance
    """
    import os
    host = os.getenv("POSTGRES_HOST", "172.20.34.142")
    port = os.getenv("POSTGRES_PORT", "5999")
    db = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password,
        cursor_factory=RealDictCursor
    )

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call using Gemini.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = embed_content(
                model="models/embedding-001",
                content=texts,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings with Gemini (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings with Gemini after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        response = embed_content(
                            model="models/embedding-001",
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(response['embedding'])
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback (Gemini embedding-001 has 768 dimensions)
                        embeddings.append([0.0] * 768)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                # Ensure fallback always returns a list of lists
                if 'embeddings' in locals():
                    return embeddings
                else:
                    return [[0.0] * 768 for _ in texts]
    return []

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using Gemini's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 768
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 768

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE") or "gemini-pro"
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the Gemini API to generate contextual information
        model = GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        
        # Extract the generated context
        context = response.text.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_postgres(
    conn, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Postgres crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        conn: psycopg2 connection instance
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs in a single operation
    try:
        with conn, conn.cursor() as cur:
            if unique_urls:
                cur.execute("DELETE FROM crawled_pages WHERE url = ANY(%s)", (unique_urls,))
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM crawled_pages WHERE url = %s", (url,))
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])
            
            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "source_id": source_id,  # Add source_id field
                "embedding": batch_embeddings[j]  # Use embedding from contextual content
            }
            
            batch_data.append(data)
        
        # Insert batch into Postgres with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                with conn, conn.cursor() as cur:
                    execute_batch(cur, """
                        INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                        VALUES (%(url)s, %(chunk_number)s, %(content)s, %(metadata)s, %(source_id)s, %(embedding)s)
                        ON CONFLICT (url, chunk_number) DO UPDATE SET
                            content=EXCLUDED.content,
                            metadata=EXCLUDED.metadata,
                            source_id=EXCLUDED.source_id,
                            embedding=EXCLUDED.embedding
                    """, batch_data)
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Postgres (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            with conn, conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                                    VALUES (%(url)s, %(chunk_number)s, %(content)s, %(metadata)s, %(source_id)s, %(embedding)s)
                                    ON CONFLICT (url, chunk_number) DO UPDATE SET
                                        content=EXCLUDED.content,
                                        metadata=EXCLUDED.metadata,
                                        source_id=EXCLUDED.source_id,
                                        embedding=EXCLUDED.embedding
                                """, record)
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")

def search_documents(
    conn,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Postgres using vector similarity.
    
    Args:
        conn: psycopg2 connection instance
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_crawled_pages function
    try:
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': json.dumps(filter_metadata) if filter_metadata else None
        }
        
        with conn, conn.cursor() as cur:
            # Use ::jsonb to cast the filter parameter
            cur.execute(
                "SELECT * FROM match_crawled_pages(%(query_embedding)s, %(match_count)s, %(filter)s::jsonb)",
                params
            )
            result = cur.fetchall()
        
        return result
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE") or "gemini-pro"
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        model = GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_postgres(
    conn,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the Postgres code_examples table in batches.
    
    Args:
        conn: psycopg2 connection instance
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
    unique_urls = list(set(urls))
    with conn, conn.cursor() as cur:
        for url in unique_urls:
            cur.execute("DELETE FROM code_examples WHERE url = %s", (url,))
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)
        embeddings = create_embeddings_batch(batch_texts)
        batch_data = []
        for j, embedding in enumerate(embeddings):
            idx = i + j
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path
            batch_data.append((
                urls[idx],
                chunk_numbers[idx],
                code_examples[idx],
                summaries[idx],
                json.dumps(metadatas[idx]),
                source_id,
                embedding
            ))
        with conn, conn.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url, chunk_number) DO UPDATE SET
                    content=EXCLUDED.content,
                    summary=EXCLUDED.summary,
                    metadata=EXCLUDED.metadata,
                    source_id=EXCLUDED.source_id,
                    embedding=EXCLUDED.embedding
            """, batch_data)

def extract_source_summary(source_id: str, content: str) -> str:
    """
    Generate a summary for a source given its content.
    
    Args:
        source_id: The ID of the source
        content: The content to summarize
        
    Returns:
        A summary of the source content
    """
    model_choice = os.getenv("MODEL_CHOICE") or "gemini-pro"
    
    prompt = f"""Based on the following content from the source '{source_id}', provide a concise summary (2-3 sentences) that describes the main topics and purpose of the documentation.

<content>
{content[:5000]}
</content>
"""
    
    try:
        model = GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating source summary for {source_id}: {e}")
        return f"Summary for {source_id}"

def update_source_info_postgres(conn, source_id: str, summary: str, total_words: int) -> None:
    """
    Update the summary and word count for a source in the sources table.
    
    Args:
        conn: psycopg2 connection instance
        source_id: The ID of the source to update
        summary: The new summary
        total_words: The new total word count
    """
    with conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sources (source_id, summary, total_words)
            VALUES (%s, %s, %s)
            ON CONFLICT (source_id) DO UPDATE SET
                summary = EXCLUDED.summary,
                total_words = EXCLUDED.total_words,
                updated_at = NOW()
        """, (source_id, summary, total_words))

def search_code_examples(
    conn,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Postgres using vector similarity.
    
    Args:
        conn: psycopg2 connection instance
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching code examples
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_code_examples function
    try:
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': json.dumps(filter_metadata) if filter_metadata else None
        }
        
        with conn, conn.cursor() as cur:
            # Use ::jsonb to cast the filter parameter
            cur.execute(
                "SELECT * FROM match_code_examples(%(query_embedding)s, %(match_count)s, %(filter)s::jsonb)",
                params
            )
            result = cur.fetchall()
        
        return result
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []