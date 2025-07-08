import json
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime

@dataclass
class SearchResult:
    document: str
    metadata: Dict
    distance: float
    relevance_score: float

class EnhancedTafsirRAG:
    def __init__(self, collection_name: str = "arabic_tafsir"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path="./tafsir_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Arabic Tafsir Knowledge Base"}
        )
        
        # Surah name mappings for better search
        self.surah_mappings = self._load_surah_mappings()
    
    def _load_surah_mappings(self) -> Dict:
        """Load surah name mappings for better query processing"""
        return {
            # English names
            'al-fatihah': 1, 'fatihah': 1, 'opening': 1,
            'al-baqarah': 2, 'baqarah': 2, 'cow': 2,
            'al-imran': 3, 'imran': 3, 'family of imran': 3,
            # Add more mappings as needed
            # Arabic names (you can expand this)
            'الفاتحة': 1, 'البقرة': 2, 'آل عمران': 3
        }
    
    def preprocess_query(self, query: str) -> Dict:
        """Enhanced query preprocessing to extract references and keywords"""
        query_info = {
            'original_query': query,
            'processed_query': query.lower().strip(),
            'surah_number': None,
            'ayah_number': None,
            'author_preference': None,
            'keywords': []
        }
        
        # Extract surah:ayah references (e.g., "1:1", "2:255")
        reference_pattern = r'(\d+):(\d+)'
        reference_match = re.search(reference_pattern, query)
        if reference_match:
            query_info['surah_number'] = int(reference_match.group(1))
            query_info['ayah_number'] = int(reference_match.group(2))
        
        # Extract surah names
        for name, number in self.surah_mappings.items():
            if name in query_info['processed_query']:
                query_info['surah_number'] = number
                break
        
        # Extract author preferences
        author_keywords = ['alusi', 'al-alusi', 'tabari', 'ibn kathir', 'qurtubi']
        for author in author_keywords:
            if author in query_info['processed_query']:
                query_info['author_preference'] = author
                break
        
        # Extract keywords (remove common words)
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query_info['processed_query'])
        query_info['keywords'] = [w for w in words if w not in stop_words and len(w) > 2]
        
        return query_info
    
    def smart_search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Enhanced search with query preprocessing and ranking"""
        query_info = self.preprocess_query(query)
        
        # Build filters based on query analysis
        filters = {}
        if query_info['surah_number']:
            filters['surah_number'] = query_info['surah_number']
        if query_info['ayah_number']:
            filters['ayah_number'] = query_info['ayah_number']
        if query_info['author_preference']:
            filters['author'] = query_info['author_preference']
        
        # Search with filters
        results = self.search_tafsir(
            query_info['processed_query'], 
            filters=filters if filters else None,
            n_results=n_results * 2  # Get more results for re-ranking
        )
        
        # Convert to SearchResult objects with enhanced scoring
        search_results = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                doc, metadata, query_info, distance
            )
            
            search_results.append(SearchResult(
                document=doc,
                metadata=metadata,
                distance=distance,
                relevance_score=relevance_score
            ))
        
        # Sort by relevance score and return top results
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results[:n_results]
    
    def _calculate_relevance_score(self, document: str, metadata: Dict, 
                                 query_info: Dict, distance: float) -> float:
        """Calculate relevance score based on multiple factors"""
        score = 1.0 - distance  # Base similarity score
        
        # Boost score for exact reference matches
        if (query_info['surah_number'] and 
            metadata['surah_number'] == query_info['surah_number']):
            score += 0.3
        
        if (query_info['ayah_number'] and 
            metadata['ayah_number'] == query_info['ayah_number']):
            score += 0.2
        
        # Boost score for author preference
        if (query_info['author_preference'] and 
            query_info['author_preference'].lower() in metadata['tafsir_author'].lower()):
            score += 0.15
        
        # Boost score for keyword presence
        doc_lower = document.lower()
        keyword_matches = sum(1 for keyword in query_info['keywords'] 
                            if keyword in doc_lower)
        if query_info['keywords']:
            keyword_score = keyword_matches / len(query_info['keywords'])
            score += keyword_score * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
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
    
    def get_comparative_tafsir(self, surah_number: int, ayah_number: int) -> Dict:
        """Get tafsir from all available authors for comparison"""
        conditions = [
            {"surah_number": {"$eq": surah_number}},
            {"ayah_number": {"$eq": ayah_number}}
        ]
        
        where_clause = {"$and": conditions}
        
        results = self.collection.query(
            query_texts=[f"surah {surah_number} ayah {ayah_number}"],
            n_results=50,  # Get all available authors
            where=where_clause
        )
        
        # Group by author
        authors_tafsir = {}
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            author = metadata['tafsir_author']
            
            if author not in authors_tafsir:
                authors_tafsir[author] = {
                    'document': doc,
                    'metadata': metadata,
                    'distance': results['distances'][0][i]
                }
        
        return authors_tafsir
    
    def evaluate_search_quality(self, test_queries: List[Dict]) -> Dict:
        """Evaluate search quality using test queries"""
        evaluation_results = {
            'total_queries': len(test_queries),
            'successful_retrievals': 0,
            'average_relevance': 0.0,
            'precision_at_k': {},
            'detailed_results': []
        }
        
        total_relevance = 0.0
        
        for query_data in test_queries:
            query = query_data['query']
            expected_surah = query_data.get('expected_surah')
            expected_ayah = query_data.get('expected_ayah')
            
            # Perform search
            results = self.smart_search(query, n_results=5)
            
            # Calculate relevance
            relevance_scores = []
            for result in results:
                relevance = 0.0
                if expected_surah and result.metadata['surah_number'] == expected_surah:
                    relevance += 0.5
                if expected_ayah and result.metadata['ayah_number'] == expected_ayah:
                    relevance += 0.5
                relevance_scores.append(relevance)
            
            # Calculate metrics
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
            total_relevance += avg_relevance
            
            if avg_relevance > 0.3:  # Consider successful if average relevance > 0.3
                evaluation_results['successful_retrievals'] += 1
            
            evaluation_results['detailed_results'].append({
                'query': query,
                'avg_relevance': avg_relevance,
                'results_count': len(results),
                'relevance_scores': relevance_scores
            })
        
        evaluation_results['average_relevance'] = total_relevance / len(test_queries)
        evaluation_results['success_rate'] = evaluation_results['successful_retrievals'] / len(test_queries)
        
        return evaluation_results
    
    def generate_search_report(self, query: str, results: List[SearchResult]) -> str:
        """Generate a formatted search report"""
        report = f"""
=== Tafsir Search Report ===
Query: {query}
Results Found: {len(results)}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        for i, result in enumerate(results, 1):
            report += f"""
--- Result {i} ---
Author: {result.metadata['tafsir_author']}
Reference: {result.metadata['surah_name_english']} ({result.metadata['surah_name_arabic']}) - Verse {result.metadata['ayah_number']}
Relevance Score: {result.relevance_score:.3f}
Distance: {result.distance:.3f}
Source: {result.metadata['url']}

Preview: {result.document[:200]}...

"""
        
        return report

# Example usage and testing
def main():
    # Initialize enhanced RAG system
    rag_system = EnhancedTafsirRAG()
    
    # Test queries
    test_queries = [
        "guidance and mercy",
        "1:1",  # Reference search
        "Al-Fatihah verse 1",
        "الحمد لله",  # Arabic search
        "Al-Alusi commentary on opening",  # Author-specific search
    ]
    
    print("=== Testing Enhanced Search ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag_system.smart_search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.metadata['tafsir_author']} - "
                  f"Surah {result.metadata['surah_number']}:{result.metadata['ayah_number']} "
                  f"(Score: {result.relevance_score:.3f})")
    
    # Test comparative analysis
    print("\n=== Comparative Analysis: Al-Fatihah 1:1 ===")
    comparative_results = rag_system.get_comparative_tafsir(1, 1)
    for author, data in comparative_results.items():
        print(f"\n{author}:")
        print(f"  Preview: {data['document'][:150]}...")
    
    # Generate detailed report
    results = rag_system.smart_search("guidance and mercy", n_results=3)
    report = rag_system.generate_search_report("guidance and mercy", results)
    print("\n" + report)

    # Evaluate search quality
    print("\n=== Evaluating Search Quality ===")
    evaluation_results = rag_system.evaluate_search_quality([
        {'query': 'guidance and mercy', 'expected_surah': 1, 'expected_ayah': 1},
        {'query': '1:1', 'expected_surah': 1, 'expected_ayah': 1},
        {'query': 'Al-Fatihah verse 1', 'expected_surah': 1, 'expected_ayah': 1},
        {'query': 'الحمد لله', 'expected_surah': 1, 'expected_ayah': 1},
        {'query': 'Al-Alusi commentary on opening', 'expected_surah': 1, 'expected_ayah': 1},
    ])
    print(f"Evaluation Success Rate: {evaluation_results['success_rate']:.2%}")

if __name__ == "__main__":
    main()