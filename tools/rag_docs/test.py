from tools.rag_docs.rag_retriver import query_documents, client

# ==========================================
# STEP 1: Check if data exists in Weaviate
# ==========================================
print("=" * 80)
print("STEP 1: Checking if data exists in Weaviate...")
print("=" * 80)

try:
    collection = client.collections.get("InternationalLawDocument")
    
    # Get total count
    result = collection.aggregate.over_all(total_count=True)
    print(f"✓ Total documents in DB: {result.total_count}")
    
    if result.total_count == 0:
        print("❌ DATABASE IS EMPTY! You need to load data first.")
        exit()
    
except Exception as e:
    print(f"❌ Error checking database: {e}")
    exit()

# ==========================================
# STEP 2: Check sample document structure
# ==========================================
print("\n" + "=" * 80)
print("STEP 2: Checking sample document structure...")
print("=" * 80)

try:
    sample = collection.query.fetch_objects(limit=1)
    
    for obj in sample.objects:
        print("\nSample document properties:")
        for key, value in obj.properties.items():
            print(f"  {key}: {value}")
        
except Exception as e:
    print(f"❌ Error fetching sample: {e}")

# ==========================================
# STEP 3: Query WITHOUT filters
# ==========================================
print("\n" + "=" * 80)
print("STEP 3: Query WITHOUT metadata filters...")
print("=" * 80)

query = "What are the purposes of the United Nations?"

results_no_filter = query_documents(
    query=query,
    metadata_filters=None,  # NO FILTERS
    strict=False,
    top_k=5
)

print(f"Results without filters: {len(results_no_filter)}")

if results_no_filter:
    print("\nFirst result:")
    print(f"  Text: {results_no_filter[0]['text'][:200]}...")
    print(f"  Score: {results_no_filter[0]['similarity_score']}")
    print(f"  Metadata: {results_no_filter[0]['metadata']}")
else:
    print("❌ NO RESULTS EVEN WITHOUT FILTERS!")
    print("   This means either:")
    print("   1. No embeddings were generated during upload")
    print("   2. Query embeddings aren't working")
    print("   3. Vector search configuration issue")

# ==========================================
# STEP 4: Query WITH filters
# ==========================================
print("\n" + "=" * 80)
print("STEP 4: Query WITH metadata filters...")
print("=" * 80)

results_with_filter = query_documents(
    query=query,
    metadata_filters={"document_type": "article"},
    strict=False,
    top_k=5
)

print(f"Results with filters: {len(results_with_filter)}")

if results_with_filter:
    for i, result in enumerate(results_with_filter[:3], 1):
        print(f"\nResult {i}:")
        print(f"  Text: {result['text'][:200]}...")
        print(f"  Score: {result['similarity_score']}")