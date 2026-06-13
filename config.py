import os
import re
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if _script_dir.name == "signalhire":
    ROOT = _script_dir.parent
else:
    ROOT = _script_dir

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "output"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"
JD_PATH = DATA_DIR / "job_description.txt"
JD_DOCX_PATH = DATA_DIR / "job_description.docx"
SAMPLE_PATH = DATA_DIR / "sample_candidates.json"
VALIDATE_SCRIPT = DATA_DIR / "validate_submission.py"
SCHEMA_PATH = DATA_DIR / "candidate_schema.json"

CURRENT_YEAR = 2026

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_DEVICE = "cuda"
EMBEDDING_BATCH_SIZE = 2048

CHUNK_SIZE = 1000
PROFILE_BLOB_MAX_CHARS = 1024
TOP_K = 100
SEED = 42

# Composite scores are computed in a ~[0, 1] range. For display and CSV
# export they are rescaled onto a 0..SCORE_SCALE scale (set to 10 -> "x/10",
# set to 100 -> "x/100"). SCORE_DECIMALS controls the emitted precision.
SCORE_SCALE = 10
SCORE_DECIMALS = 2

WEIGHTS = {
    "technical_fit": 0.35,
    "career_quality": 0.25,
    "availability_signal": 0.20,
    "seniority_fit": 0.12,
    "semantic_similarity": 0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"Weights must sum to 1.0: {sum(WEIGHTS.values())}"

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "capgemini", "cognizant", "accenture",
    "hcl", "mphasis", "tech mahindra", "hexaware", "mindtree", "persistent",
]

PROFICIENCY_MAP = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.4,
    "beginner": 0.1,
}

JD_MUST_HAVES = {
    "embeddings / retrieval": 0.25,
    "vector databases": 0.20,
    "python": 0.15,
    "eval frameworks / ndcg / mrr": 0.15,
}

JD_MUST_KEYWORDS = {
    "embeddings / retrieval": [
        "embedding", "sentence-transformer", "bge", "e5", "openai embedding",
        "retrieval", "dense retrieval", "hybrid retrieval",
    ],
    "vector databases": [
        "vector", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "faiss",
    ],
    "python": ["python"],
    "eval frameworks / ndcg / mrr": [
        "ndcg", "mrr", "map", "offline", "eval", "a/b test", "ab test",
        "ranking", "retrieval quality", "online eval",
    ],
}

JD_NICE_TO_HAVES = {
    "llm fine-tuning / lora": 0.10,
    "learning-to-rank / xgboost": 0.10,
    "hr-tech": 0.05,
}

JD_NICE_KEYWORDS = {
    "llm fine-tuning / lora": [
        "fine-tun", "lora", "qlora", "peft", "llm", "llms",
    ],
    "learning-to-rank / xgboost": [
        "learning to rank", "xgboost", "ltr", "lambdamart", "ranklib",
    ],
    "hr-tech": [
        "hr-tech", "recruiting", "talent", "hiring", "redrob", "ats",
    ],
}

PRODUCTION_SIGNALS = [
    "production", "deployed", "shipped", "serving", "at scale",
    "a/b test", "ab test", "ndcg", "mrr", "benchmark",
]

RETRIEVAL_SIGNALS = [
    "embedding", "vector", "semantic search", "rag", "retrieval",
    "faiss", "pinecone", "milvus", "qdrant", "weaviate",
    "bm25", "dense retrieval", "sentence-transformer",
]

ML_AI_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ml ops", "mlops",
    "data scientist", "data science",
    "ai engineer", "ai ml", "artificial intelligence",
    "nlp engineer", "nlp",
    "deep learning", "llm", "gen ai", "generative ai",
    "applied scientist", "research scientist",
    "recommendation", "recommender", "ranking", "search engineer",
    "computer vision", "cv engineer",
]

TIER_1_INSTITUTIONS = [
    "iit ", "iisc", "nit ", "bits pilani", "bits",
    "iiit ", "iiit", "indian institute of technology",
    "indian institute of science",
]

TIER_2_INSTITUTIONS = [
    "dtu", "delhi technological", "nsit", "iiitm", "iiit",
    "nit", "vit", "vellore institute", "srm",
    "college of engineering", "psg", "thapar",
    "birla institute", "bitsat",
]

HONEYPOT_YEAR_BUFFER = 5
GHOST_COMPLETENESS_THRESHOLD = 5

DATE_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
REFERENCE_DATE = "2026-06-01"

RESEARCHER_TITLE_KW = ["researcher", "research scientist", "research engineer"]
NO_CODE_MONTHS = 18
NO_CODE_DAYS = 548

CONSULTING_PENALTY = 0.15
NO_CODE_PENALTY = 0.80
CV_SPEECH_ROBOTICS_PENALTY = 0.85

# Matched with word boundaries (see disqualify._matches_any_kw) to avoid
# false positives like "version control" -> "control" or "RPA robot" -> "robot".
CV_SPEECH_ROBOTICS_KW = [
    "computer vision", "object detection", "image classification", "image segmentation",
    "yolo", "cnn", "resnet", "efficientnet", "opencv",
    "speech recognition", "asr", "speech-to-text", "text-to-speech", "whisper",
    "robotics", "ros", "slam", "motion planning", "control systems", "autonomous",
]
