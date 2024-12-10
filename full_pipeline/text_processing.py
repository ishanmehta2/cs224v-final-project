import string

def clean_text(input_text: str) -> str:
    """
    Strips all punctuation from the input string, keeping only letters and digits,
    and converts the text to lowercase.
    """
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = input_text.translate(translator).lower()
    cleaned_text = ''.join(char for char in cleaned_text if char.isalnum() or char.isspace())
    return cleaned_text

