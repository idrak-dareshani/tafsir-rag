import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

@dataclass
class TafsirEntry:
    surah_number: int
    surah_name_arabic: str
    surah_name_english: str
    ayah_number: int
    tafsir_author: str
    url: str
    tafsir_text: str
    extraction_timestamp: str

class ArabicTafsirRAG:
    def __init__(self, collection_name: str = "arabic_tafsir"):
        # Initialize embedding model (multilingual for Arabic support)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path="./tafsir_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Arabic Tafsir Knowledge Base"}
        )
    
    def load_tafsir_data(self, data_directory: str) -> List[TafsirEntry]:
        """Load tafsir data from directory structure"""
        tafsir_entries = []
        
        for author_folder in os.listdir(data_directory):
            author_path = os.path.join(data_directory, author_folder)
            if not os.path.isdir(author_path):
                continue
                
            # Process each surah
            for file in os.listdir(author_path):
                if file.endswith('.json'):
                    surah_number = file.replace('.json', '')
                    json_path = os.path.join(author_path, file)
                    
                    # Load JSON metadata
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Load corresponding text files
                    for entry in json_data:
                        text_filename = f"{surah_number}_{entry['ayah_number']}.txt"
                        text_path = os.path.join(author_path, text_filename)
                        
                        if os.path.exists(text_path):
                            with open(text_path, 'r', encoding='utf-8') as f:
                                tafsir_text = f.read().strip()
                            
                            tafsir_entry = TafsirEntry(
                                surah_number=entry['surah_number'],
                                surah_name_arabic=entry['surah_name_arabic'],
                                surah_name_english=entry['surah_name_english'],
                                ayah_number=entry['ayah_number'],
                                tafsir_author=entry['tafsir_author'],
                                url=entry['url'],
                                tafsir_text=tafsir_text,
                                extraction_timestamp=entry['extraction_timestamp']
                            )
                            tafsir_entries.append(tafsir_entry)
        
        return tafsir_entries
    
    def index_tafsir_data(self, tafsir_entries: List[TafsirEntry]):
        """Index tafsir data into vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for entry in tafsir_entries:
            # Create searchable document combining Arabic and English names
            document = f"{entry.surah_name_arabic} {entry.surah_name_english} آية {entry.ayah_number} {entry.tafsir_text}"
            
            documents.append(document)
            metadatas.append({
                "surah_number": entry.surah_number,
                "surah_name_arabic": entry.surah_name_arabic,
                "surah_name_english": entry.surah_name_english,
                "ayah_number": entry.ayah_number,
                "tafsir_author": entry.tafsir_author,
                "url": entry.url,
                "extraction_timestamp": entry.extraction_timestamp
            })
            ids.append(str(uuid.uuid4()))
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(documents)} tafsir entries")
    
    def search_tafsir(self, query: str, filters: Optional[Dict] = None, n_results: int = 5) -> List[Dict]:
        """Search tafsir based on query with optional filters"""
        where_clause = None
        
        if filters:
            conditions = []
            
            if 'surah_number' in filters:
                conditions.append({"surah_number": {"$eq": filters['surah_number']}})
            if 'author' in filters:
                conditions.append({"tafsir_author": {"$eq": filters['author']}})
            if 'ayah_number' in filters:
                conditions.append({"ayah_number": {"$eq": filters['ayah_number']}})
            
            # Combine conditions with $and operator
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        return results
    
    def get_tafsir_by_reference(self, surah_number: int, ayah_number: int, author: Optional[str] = None) -> List[Dict]:
        """Get tafsir by specific surah and ayah reference"""
        conditions = [
            {"surah_number": {"$eq": surah_number}},
            {"ayah_number": {"$eq": ayah_number}}
        ]
        
        if author:
            conditions.append({"tafsir_author": {"$eq": author}})
        
        where_clause = {"$and": conditions}
        
        results = self.collection.query(
            query_texts=[f"surah {surah_number} ayah {ayah_number}"],
            n_results=10,
            where=where_clause
        )
        
        return results

# Example usage
def main():
    # Initialize RAG system
    rag_system = ArabicTafsirRAG()
    
    # Load and index data
    print("Loading tafsir data...")
    tafsir_entries = rag_system.load_tafsir_data("./data")
    
    print("Indexing tafsir data...")
    rag_system.index_tafsir_data(tafsir_entries)
    
    # Example searches
    print("\n=== Example Searches ===")
    
    # Search by content
    results = rag_system.search_tafsir("guidance and mercy", n_results=3)
    print(f"Content search results: {len(results['documents'][0])}")
    
    # Search by reference
    fatihah_results = rag_system.get_tafsir_by_reference(1, 1)  # Al-Fatihah, verse 1
    print(f"Al-Fatihah verse 1 results: {len(fatihah_results['documents'][0])}")
    
    # Search with author filter
    alusi_results = rag_system.search_tafsir("الحمد لله", filters={'author': 'Al-Alusi'})
    print(f"Al-Alusi specific results: {len(alusi_results['documents'][0])}")

if __name__ == "__main__":
    main()