from transformers import pipeline
from newsapi import NewsApiClient
from config import Config
import json
from urllib.parse import urlparse

class RAGPipeline:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification",
                                 model=Config.MODEL_ID,
                                 api_key=Config.HUGGINGFACE_API_KEY)
        self.news_fetcher = NewsApiClient(api_key=Config.NEWS_API_KEY)
    
    def fetch_article_by_url(self, url: str):
        """Fetch a specific article by URL"""
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            response = self.news_fetcher.get_everything(
                q='',
                domains=domain,
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
            
            if response['status'] == 'ok' and response['articles']:
                for article in response['articles']:
                    if url in article.get('url', ''):
                        return article
                return response['articles'][0]
            return None
            
        except Exception as e:
            print(f"Error fetching article by URL: {str(e)}")
            return None
    
    def detect_fake_news(self, article_text: str):
        """Detect if news article is fake using Hugging Face"""
        try:
            candidate_labels = ["real news", "fake news"]
            result = self.classifier(
                article_text,
                candidate_labels,
                hypothesis_template="This text is {}."
            )
            
            is_fake = result['labels'][0] == "fake news"
            confidence = result['scores'][0]
            characteristics = self._analyze_characteristics(article_text)
            
            response = {
                "classification": "FAKE" if is_fake else "REAL",
                "confidence_score": confidence,
                "reasoning": f"""
                Classification based on:
                1. Content Analysis: {characteristics['content']}
                2. Style Analysis: {characteristics['style']}
                3. Source Credibility: {characteristics['source']}
                Overall confidence: {confidence:.2%}
                """
            }
            
            return json.dumps(response)
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return json.dumps({
                "classification": "ERROR",
                "confidence_score": 0,
                "reasoning": f"Error during analysis: {str(e)}"
            })
    
    def _analyze_characteristics(self, text):
        """Analyze various characteristics of the text"""
        style_labels = ["professional writing", "informal writing", "clickbait"]
        content_labels = ["factual content", "opinion", "sensationalized"]
        source_labels = ["credible source", "questionable source"]
        
        style = self.classifier(text, style_labels)
        content = self.classifier(text, content_labels)
        source = self.classifier(text, source_labels)
        
        return {
            "style": f"Writing style appears {style['labels'][0]} ({style['scores'][0]:.2%} confidence)",
            "content": f"Content seems {content['labels'][0]} ({content['scores'][0]:.2%} confidence)",
            "source": f"Source appears {source['labels'][0]} ({source['scores'][0]:.2%} confidence)"
        }
