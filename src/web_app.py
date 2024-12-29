from flask import Flask, render_template, request, jsonify
from rag_pipeline import RAGPipeline

app = Flask(__name__)
rag_pipeline = RAGPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.json.get('url', '')
        if not url:
            return jsonify({'error': 'Please provide a URL'}), 400

        article = rag_pipeline.fetch_article_by_url(url)
        if not article:
            return jsonify({'error': 'Could not fetch article content. Please make sure the URL is from a supported news source.'}), 400

        article_text = f"""
        Title: {article.get('title', '')}
        Content: {article.get('content', '')}
        Description: {article.get('description', '')}
        Source: {article.get('source', {}).get('name', '')}
        """

        result = rag_pipeline.detect_fake_news(article_text)
        return result

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
