import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessedText:
    original: str
    normalized: str
    without_diacritics: str
    keywords: List[str]
    language: str  # 'arabic', 'english', 'mixed'

class ArabicTextProcessor:
    def __init__(self):
        # Arabic diacritics (tashkeel)
        self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        
        # Arabic letters range
        self.arabic_letters = re.compile(r'[\u0600-\u06FF]')
        
        # Common Arabic stop words
        self.arabic_stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'إن', 'أن', 'كان', 'كانت', 
            'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'الله', 'رب', 'عبد',
            'قال', 'قالت', 'يقول', 'تقول', 'قد', 'لا', 'ما', 'لم', 'لن',
            'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'نحن', 'أنا'
        }
        
        # Arabic root patterns (simplified)
        self.common_patterns = {
            'الـ': '',  # Remove definite article
            'وال': 'و',  # And the -> and
            'بال': 'ب',  # With the -> with
            'كال': 'ك',  # Like the -> like
            'فال': 'ف',  # So the -> so
        }
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic, English, or mixed"""
        arabic_chars = len(self.arabic_letters.findall(text))
        total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))
        
        if total_chars == 0:
            return 'unknown'
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.8:
            return 'arabic'
        elif arabic_ratio < 0.2:
            return 'english'
        else:
            return 'mixed'
    
    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text"""
        # Convert to NFD (Normalized Form Decomposed)
        text = unicodedata.normalize('NFD', text)
        
        # Normalize common variations
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')  # Ta marbuta to ha
        text = text.replace('ي', 'ى')  # Ya to alif maksura
        text = text.replace('ئ', 'ي').replace('ؤ', 'و')
        
        # Remove extra spaces and clean
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel)"""
        return self.arabic_diacritics.sub('', text)
    
    def extract_keywords(self, text: str, language: str) -> List[str]:
        """Extract keywords based on language"""
        if language == 'arabic':
            # Remove diacritics first
            clean_text = self.remove_diacritics(text)
            
            # Split into words
            words = re.findall(r'[\u0600-\u06FF]+', clean_text)
            
            # Filter out stop words and short words
            keywords = [word for word in words 
                       if word not in self.arabic_stopwords and len(word) > 2]
            
            return keywords[:20]  # Limit to top 20
        
        elif language == 'english':
            # Standard English processing
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            english_stopwords = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
            keywords = [word for word in words 
                       if word not in english_stopwords and len(word) > 2]
            return keywords[:20]
        
        else:  # mixed
            # Process both languages
            arabic_keywords = self.extract_keywords(text, 'arabic')
            english_keywords = self.extract_keywords(text, 'english')
            return arabic_keywords + english_keywords
    
    def process_text(self, text: str) -> ProcessedText:
        """Complete text processing pipeline"""
        language = self.detect_language(text)
        normalized = self.normalize_arabic(text) if language in ['arabic', 'mixed'] else text
        without_diacritics = self.remove_diacritics(normalized)
        keywords = self.extract_keywords(text, language)
        
        return ProcessedText(
            original=text,
            normalized=normalized,
            without_diacritics=without_diacritics,
            keywords=keywords,
            language=language
        )

class EnhancedArabicTafsirRAG:
    def __init__(self, collection_name: str = "arabic_tafsir_enhanced"):
        # Initialize text processor
        self.text_processor = ArabicTextProcessor()
        
        # Initialize embedding model (better for Arabic)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path="./tafsir_db_enhanced")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Enhanced Arabic Tafsir Knowledge Base"}
        )
        
        # Performance metrics
        self.search_metrics = {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'language_distribution': {'arabic': 0, 'english': 0, 'mixed': 0}
        }
    
    def enhanced_index_tafsir(self, tafsir_entries: List) -> Dict:
        """Enhanced indexing with Arabic text processing"""
        indexing_stats = {
            'total_entries': len(tafsir_entries),
            'processed_entries': 0,
            'language_distribution': {'arabic': 0, 'english': 0, 'mixed': 0},
            'avg_keywords_per_entry': 0,
            'processing_errors': []
        }
        
        documents = []
        metadatas = []
        ids = []
        total_keywords = 0
        
        for entry in tafsir_entries:
            try:
                # Process the tafsir text
                processed_text = self.text_processor.process_text(entry.tafsir_text)
                
                # Create enhanced searchable document
                searchable_content = f"""
                {entry.surah_name_arabic} {entry.surah_name_english} 
                آية {entry.ayah_number} 
                {processed_text.original}
                {processed_text.normalized}
                {' '.join(processed_text.keywords)}
                """
                
                documents.append(searchable_content.strip())
                
                # Enhanced metadata
                metadatas.append({
                    "surah_number": entry.surah_number,
                    "surah_name_arabic": entry.surah_name_arabic,
                    "surah_name_english": entry.surah_name_english,
                    "ayah_number": entry.ayah_number,
                    "tafsir_author": entry.tafsir_author,
                    "url": entry.url,
                    "extraction_timestamp": entry.extraction_timestamp,
                    "language": processed_text.language,
                    "keywords": processed_text.keywords[:10],  # Store top 10 keywords
                    "text_length": len(processed_text.original),
                    "normalized_text": processed_text.without_diacritics[:500]  # Store first 500 chars
                })
                
                ids.append(f"{entry.surah_number}_{entry.ayah_number}_{entry.tafsir_author}")
                
                # Update stats
                indexing_stats['language_distribution'][processed_text.language] += 1
                total_keywords += len(processed_text.keywords)
                indexing_stats['processed_entries'] += 1
                
            except Exception as e:
                indexing_stats['processing_errors'].append({
                    'entry': f"Surah {entry.surah_number}:{entry.ayah_number} - {entry.tafsir_author}",
                    'error': str(e)
                })
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        # Calculate final stats
        indexing_stats['avg_keywords_per_entry'] = total_keywords / max(indexing_stats['processed_entries'], 1)
        
        return indexing_stats
    
    def enhanced_search(self, query: str, n_results: int = 5) -> Dict:
        """Enhanced search with Arabic processing and metrics"""
        start_time = datetime.now()
        
        # Process query
        processed_query = self.text_processor.process_text(query)
        
        # Update metrics
        self.search_metrics['total_searches'] += 1
        self.search_metrics['language_distribution'][processed_query.language] += 1
        
        # Create search variations
        search_queries = [
            processed_query.original,
            processed_query.normalized,
            processed_query.without_diacritics,
            ' '.join(processed_query.keywords)
        ]
        
        # Remove duplicates and empty queries
        search_queries = list(set([q.strip() for q in search_queries if q.strip()]))
        
        # Perform searches with different query variations
        all_results = []
        
        for search_query in search_queries:
            try:
                results = self.collection.query(
                    query_texts=[search_query],
                    n_results=n_results * 2  # Get more for ranking
                )
                
                # Process results
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Calculate enhanced relevance score
                    relevance_score = self._calculate_enhanced_relevance(
                        doc, metadata, processed_query, distance
                    )
                    
                    all_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'relevance_score': relevance_score,
                        'search_query_used': search_query
                    })
                    
            except Exception as e:
                print(f"Search error with query '{search_query}': {e}")
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in all_results:
            key = f"{result['metadata']['surah_number']}_{result['metadata']['ayah_number']}_{result['metadata']['tafsir_author']}"
            if key not in unique_results or result['relevance_score'] > unique_results[key]['relevance_score']:
                unique_results[key] = result
        
        # Sort and limit results
        final_results = sorted(unique_results.values(), key=lambda x: x['relevance_score'], reverse=True)[:n_results]
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        self.search_metrics['avg_response_time'] = (
            (self.search_metrics['avg_response_time'] * (self.search_metrics['total_searches'] - 1) + response_time) /
            self.search_metrics['total_searches']
        )
        
        return {
            'query_info': {
                'original': processed_query.original,
                'language': processed_query.language,
                'keywords': processed_query.keywords,
                'search_variations': search_queries
            },
            'results': final_results,
            'metadata': {
                'total_results': len(final_results),
                'response_time': response_time,
                'search_variations_used': len(search_queries)
            }
        }
    
    def _calculate_enhanced_relevance(self, document: str, metadata: Dict, 
                                    processed_query: ProcessedText, distance: float) -> float:
        """Enhanced relevance calculation with Arabic text considerations"""
        base_score = 1.0 - distance
        
        # Language matching bonus
        doc_language = metadata.get('language', 'unknown')
        if doc_language == processed_query.language:
            base_score += 0.1
        
        # Keyword matching in metadata
        doc_keywords = set(metadata.get('keywords', []))
        query_keywords = set(processed_query.keywords)
        
        if doc_keywords and query_keywords:
            keyword_overlap = len(doc_keywords.intersection(query_keywords))
            keyword_score = keyword_overlap / max(len(query_keywords), 1)
            base_score += keyword_score * 0.2
        
        # Text length consideration (prefer detailed explanations)
        text_length = metadata.get('text_length', 0)
        if text_length > 200:  # Prefer longer, more detailed tafsir
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def get_search_analytics(self) -> Dict:
        """Get search analytics and performance metrics"""
        return {
            'search_metrics': self.search_metrics,
            'collection_stats': {
                'total_documents': self.collection.count(),
                'collection_name': self.collection.name
            },
            'performance_insights': {
                'avg_response_time_ms': self.search_metrics['avg_response_time'] * 1000,
                'most_common_language': max(self.search_metrics['language_distribution'], 
                                          key=self.search_metrics['language_distribution'].get),
                'total_searches': self.search_metrics['total_searches']
            }
        }
    
    def benchmark_search_improvements(self, test_queries: List[str]) -> Dict:
        """Benchmark search improvements with Arabic processing"""
        benchmark_results = {
            'test_queries': len(test_queries),
            'language_breakdown': {'arabic': 0, 'english': 0, 'mixed': 0},
            'avg_relevance_scores': [],
            'processing_details': []
        }
        
        for query in test_queries:
            result = self.enhanced_search(query, n_results=3)
            
            # Track language distribution
            query_lang = result['query_info']['language']
            benchmark_results['language_breakdown'][query_lang] += 1
            
            # Calculate average relevance for this query
            if result['results']:
                avg_relevance = np.mean([r['relevance_score'] for r in result['results']])
                benchmark_results['avg_relevance_scores'].append(avg_relevance)
            
            # Store processing details
            benchmark_results['processing_details'].append({
                'query': query,
                'language': query_lang,
                'keywords_extracted': len(result['query_info']['keywords']),
                'search_variations': len(result['query_info']['search_variations']),
                'results_found': len(result['results']),
                'response_time': result['metadata']['response_time']
            })
        
        # Calculate summary statistics
        if benchmark_results['avg_relevance_scores']:
            benchmark_results['overall_avg_relevance'] = np.mean(benchmark_results['avg_relevance_scores'])
            benchmark_results['relevance_std'] = np.std(benchmark_results['avg_relevance_scores'])
        
        return benchmark_results

# Example usage
def main():
    # Initialize enhanced system
    enhanced_rag = EnhancedArabicTafsirRAG()
    
    # Test Arabic text processing
    processor = ArabicTextProcessor()
    
    test_texts = [
        "الحمد لله رب العالمين",
        "guidance and mercy",
        "الرحمن الرحيم with English mixed",
        "بسم الله الرحمن الرحيم"
    ]
    
    print("=== Arabic Text Processing Tests ===")
    for text in test_texts:
        processed = processor.process_text(text)
        print(f"\nOriginal: {processed.original}")
        print(f"Language: {processed.language}")
        print(f"Normalized: {processed.normalized}")
        print(f"Without diacritics: {processed.without_diacritics}")
        print(f"Keywords: {processed.keywords}")
    
    # Test enhanced search (assuming data is loaded)
    print("\n=== Enhanced Search Tests ===")
    test_queries = [
        "الحمد لله",
        "guidance",
        "1:1",
        "mercy of Allah الرحمة"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        # result = enhanced_rag.enhanced_search(query, n_results=2)
        # print(f"Language detected: {result['query_info']['language']}")
        # print(f"Keywords: {result['query_info']['keywords']}")
        # print(f"Results found: {result['metadata']['total_results']}")

if __name__ == "__main__":
    main()