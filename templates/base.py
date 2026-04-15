class BaseTemplate:
    """Interface for rendering video scenes."""
    def make_scene(self, sentence_text, idx, res, kw=None):
        raise NotImplementedError("Each template must implement the make_scene method.")
