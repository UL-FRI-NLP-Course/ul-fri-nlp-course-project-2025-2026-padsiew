import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer

@dataclass
class PipelineOutput:
    original_text: str
    cleaned_text: str
    tokens: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    num_tokens: int
    was_truncated: bool
    metadata: dict = field(default_factory=dict)

class SloveneTextPreprocessor:
    # Pogoste slovenske okrajšave
    LEGAL_ABBREVS = [
        r"čl\.",        # člen
        r"odst\.",      # odstavek
        r"al\.",        # alineja
        r"št\.",        # številka
        r"str\.",       # stran
        r"npr\.",       # na primer
        r"itd\.",       # in tako dalje
        r"idr\.",       # in drugi
        r"oz\.",        # oziroma
        r"t\.i\.",      # tako imenovani
        r"t\.j\.",      # to je
        r"gl\.",        # glej
        r"prim\.",      # primerjaj
        r"zač\.",       # začasno
        r"sl\.",        # slovensko
        r"angl\.",      # angleško
        r"RS\.",        # Republika Slovenija
        r"Ur\.\s?l\.",  # Uradni list
        r"ZPP\b",       # Zakon o pravdnem postopku
        r"KZ\b",        # Kazenski zakonik
        r"ZKP\b",       # Zakon o kazenskem postopku
        r"ZOR\b",       # Zakon o obligacijskih razmerjih
        r"OZ\b",        # Obligacijski zakonik
        r"ZDR\b",       # Zakon o delovnih razmerjih
    ]

    # Citation patterns common in Slovene law
    CITATION_PATTERNS = [
        # "5. člen" or "5. odstavek" → normalize spacing
        (r"(\d+)\.\s+(člen|odstavek|alineja|točka|poglavje)", r"\1. \2"),
        # Article ranges: "5.-7. člen" → "5.-7. člen"
        (r"(\d+)\s*[-–]\s*(\d+)\.\s+(člen|odstavek)", r"\1-\2. \3"),
        # Law gazette refs: "Ur. l. RS, št. 26/99" → normalize
        (r"Ur\.\s*l\.\s*RS\s*,\s*št\.\s*(\d+)", r"Ur. l. RS, št. \1"),
        # Paragraph refs like "(1)" at start of sentence
        (r"^\((\d+)\)\s+", r"(\1) "),
    ]

    # Pick which preprocessing steps we want to include
    def __init__(
        self,
        lowercase: bool = False, 
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        normalize_punctuation: bool = True,
        normalize_citations: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = False,
    ):
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.normalize_punctuation = normalize_punctuation
        self.normalize_citations = normalize_citations
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails

    def preprocess(self, text: str) -> tuple[str, dict]:
        metadata = {"steps_applied": []}
        metadata["original_length"] = len(text)

        if self.normalize_unicode:
            text = self._normalize_unicode(text)
            metadata["steps_applied"].append("unicode_normalization")

        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
            metadata["steps_applied"].append("url_removal")

        if self.remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
            metadata["steps_applied"].append("email_removal")

        if self.normalize_punctuation:
            text = self._normalize_punctuation(text)
            metadata["steps_applied"].append("punctuation_normalization")

        if self.normalize_citations:
            text = self._normalize_citations(text)
            metadata["steps_applied"].append("citation_normalization")

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
            metadata["steps_applied"].append("whitespace_normalization")

        if self.lowercase:
            text = text.lower()
            metadata["steps_applied"].append("lowercasing")

        metadata["cleaned_length"] = len(text)
        return text, metadata

    def _normalize_unicode(self, text: str) -> str:
        # NFC normalization — important for š, č, ž
        text = unicodedata.normalize("NFC", text)
        # Replace common lookalikes
        replacements = {
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00ad": "",
            "\u00a0": " ",
        }
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        # Normalize punctuation, but preserve legal abbreviations
        # Remove multiple punctuation
        text = re.sub(r"([!?]){2,}", r"\1", text)
        # Remove invalid characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def _normalize_citations(self, text: str) -> str:
        # Normalize the legal citations
        for pattern, replacement in self.CITATION_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        # Normalize all whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text) 
        text = text.strip()
        return text

# Main pipeline

if __name__ == "__main__":
    preprocessor = SloveneTextPreprocessor()

    samples = [
        "Določba 5.  člena  ZPP določa, da mora stranka vložiti tožbo v roku 30 dni.",
        "Sodišče  je v skladu z Ur. l. RS, št. 26/99 odločilo, da se postopek ustavi.",
        "Tožena stranka  je  kršila  1. odst. 7. člena pogodbe.",
    ]

    for sample in samples:
        cleaned, meta = preprocessor.preprocess(sample)
        print(f"Original : {sample}")
        print(f"Cleaned  : {cleaned}")
        print()

class SloveneLegalPipeline:
    model = "cjvt/GaMS-2B"
    GEMMA2_MAX_TOKENS = 8192

    def __init__(
        self,
        model_id: str = model,
        max_length: int = 2048,
        padding: str | bool = False,
        truncation: bool = True,
        preprocessor: Optional[SloveneTextPreprocessor] = None,
        prompt_template: Optional[str] = None,
    ):
        print(f"[Pipeline] Loading tokenizer from: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = min(max_length, self.GEMMA2_MAX_TOKENS)
        self.padding = padding
        self.truncation = truncation
        self.preprocessor = preprocessor or SloveneTextPreprocessor()
        self.prompt_template = prompt_template
        print(f"[Pipeline] Ready. Max tokens: {self.max_length}")

    def process(self, text: str) -> PipelineOutput:
        # 1. Preprocess
        cleaned_text, meta = self.preprocessor.preprocess(text)

        # 2. Apply prompt template if we use rag
        model_input = self._apply_template(cleaned_text)

        # 3. Tokenize
        encoding = self.tokenizer(
            model_input,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=None,   # Trenutno navadni python listi
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        was_truncated = len(input_ids) == self.max_length

        return PipelineOutput(
            original_text=text,
            cleaned_text=cleaned_text,
            tokens=tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_tokens=len(input_ids),
            was_truncated=was_truncated,
            metadata=meta,
        )

    # Process a list of inputs one by one
    def process_batch(self, texts: list[str]) -> list[PipelineOutput]:
        return [self.process(t) for t in texts]
    
    # Apply prompt template if we ise rag
    def _apply_template(self, text: str) -> str:
        if self.prompt_template:
            return self.prompt_template.format(text=text)
        return text

    # Decode token IDs back to text
    def decode(self, input_ids: list[int]) -> str:
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

# Testing

if __name__ == "__main__":
    model = "cjvt/GaMS-2B"
    # Basic pipeline 
    pipeline = SloveneLegalPipeline(
        model_id=model,
        max_length=512,
    )

    sample = (
        "Določba 5.  člena  ZPP določa, da mora stranka vložiti tožbo v roku 30 dni. "
        "Sodišče  je v skladu z Ur. l. RS, št. 26/99 odločilo, da se postopek ustavi. "
        "Tožena stranka  je  kršila  1. odst. 7. člena pogodbe."
    )

    result = pipeline.process(sample)

    print("=== PIPELINE OUTPUT ===")
    print(f"Original  : {result.original_text}")
    print(f"Cleaned   : {result.cleaned_text}")
    print(f"Tokens    : {result.tokens[:20]}{'...' if len(result.tokens) > 20 else ''}")
    print(f"Input IDs : {result.input_ids[:20]}{'...' if len(result.input_ids) > 20 else ''}")
    print(f"Num tokens: {result.num_tokens}")
    print(f"Truncated : {result.was_truncated}")
    print(f"Metadata  : {result.metadata}")

    # RAG pipeline
    rag_pipeline = SloveneLegalPipeline(
        model_id=model,
        max_length=1024,
        prompt_template=(     # Tu dodas prompt
            "Si pravni asistent za slovensko pravo. "
            "Na podlagi spodnjega besedila odgovori na vprašanje.\n\n"
            "Besedilo: {text}\n\n"
            "Odgovor:"
        ),
    )

    question = "Katera določba ureja rok za vložitev tožbe v civilnih sporih?"

    rag_result = rag_pipeline.process(question)
    print("\n=== RAG PIPELINE OUTPUT ===")
    print(f"Model input (first 200 chars): {rag_result.cleaned_text[:200]}...")
    print(f"Num tokens: {rag_result.num_tokens}")

# Preprocessing testing

# if __name__ == "__main__":
#     preprocessor = SloveneTextPreprocessor()

#     samples = [
#         "Določba 5.  člena  ZPP določa, da mora stranka vložiti tožbo v roku 30 dni.",
#         "Sodišče  je v skladu z Ur. l. RS, št. 26/99 odločilo, da se postopek ustavi.",
#         "Tožena stranka  je  kršila  1. odst. 7. člena pogodbe.",
#     ]

#     for sample in samples:
#         cleaned, meta = preprocessor.preprocess(sample)
#         print(f"Original : {sample}")
#         print(f"Cleaned  : {cleaned}")
#         print()