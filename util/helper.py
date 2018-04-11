def extract_words(document):
    """Summary

    Args:
        document (str): document instance

    Returns:
        list: list of words
    """
    words = document.lower().strip().split(' ')
    return words
