class SubQueryColumnMapper:
    def __init__(self, model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

        