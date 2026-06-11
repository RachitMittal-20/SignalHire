import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    ARTIFACTS_DIR,
    CANDIDATES_PATH,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    JD_DOCX_PATH,
    JD_PATH,
    PROFILE_BLOB_MAX_CHARS,
)
from disqualify import is_honeypot, should_hard_penalize
from signals import (
    compute_availability_signal,
    compute_career_quality,
    compute_seniority_fit,
    compute_technical_fit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def extract_jd_text() -> str:
    if JD_PATH.exists():
        return JD_PATH.read_text(encoding="utf-8")
    if JD_DOCX_PATH.exists():
        try:
            from docx import Document
            doc = Document(str(JD_DOCX_PATH))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            log.warning("python-docx not installed, copying docx to txt manually")
            raise
    raise FileNotFoundError(f"Neither {JD_PATH} nor {JD_DOCX_PATH} found")


def build_profile_blob(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]

    career = candidate.get("career_history", [])
    for role in career:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))

    skills = candidate.get("skills", [])
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        if prof in ("expert", "advanced"):
            parts.append(skill.get("name", ""))

    blob = " ".join(p for p in parts if p)
    return blob[:PROFILE_BLOB_MAX_CHARS]


def compute_subscores(candidate: dict) -> dict:
    penalty_mult, _ = should_hard_penalize(candidate)
    return {
        "technical_fit": compute_technical_fit(candidate),
        "career_quality": compute_career_quality(candidate),
        "availability_signal": compute_availability_signal(candidate),
        "seniority_fit": compute_seniority_fit(candidate),
        "penalty_multiplier": penalty_mult,
    }


def main():
    log.info("Loading embedding model: %s (device: %s)", EMBEDDING_MODEL, EMBEDDING_DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

    log.info("Extracting JD text")
    jd_text = extract_jd_text()
    jd_embedding = model.encode(jd_text, normalize_embeddings=True).astype(np.float32)
    np.save(str(ARTIFACTS_DIR / "jd_embedding.npy"), jd_embedding)
    log.info("JD embedding saved (shape: %s)", jd_embedding.shape)

    total = 0
    disqualified = []
    honeypot_count = 0
    ghost_count = 0

    all_embeddings = []
    all_candidate_ids = []
    subscores_dict = {}

    blob_batch = []
    id_batch = []

    t0 = time.time()

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed JSON line: %s", e)
                total += 1
                continue
            total += 1

            honeypot, reason = is_honeypot(candidate)
            if honeypot:
                cid = candidate.get("candidate_id", "unknown")
                hp_type = "HONEYPOT" if "HONEYPOT" in reason else "GHOST"
                if hp_type == "HONEYPOT":
                    honeypot_count += 1
                else:
                    ghost_count += 1
                disqualified.append({"id": cid, "type": hp_type, "reason": reason})
                continue

            cid = candidate.get("candidate_id", "")
            all_candidate_ids.append(cid)

            subscores_dict[cid] = compute_subscores(candidate)

            blob = build_profile_blob(candidate)
            blob_batch.append(blob)
            id_batch.append(cid)

            if len(blob_batch) >= EMBEDDING_BATCH_SIZE:
                emb = model.encode(blob_batch, normalize_embeddings=True).astype(np.float32)
                all_embeddings.append(emb)
                blob_batch = []
                id_batch = []

            if total % CHUNK_SIZE == 0:
                elapsed = time.time() - t0
                log.info(
                    "Processed %d candidates | disqualified: %d | honeypot: %d | ghost: %d | %.1f sec",
                    total,
                    len(disqualified),
                    honeypot_count,
                    ghost_count,
                    elapsed,
                )

    if blob_batch:
        emb = model.encode(blob_batch, normalize_embeddings=True).astype(np.float32)
        all_embeddings.append(emb)

    log.info("Stacking embeddings ...")
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    candidate_ids = np.array(all_candidate_ids, dtype=object)

    log.info(
        "Saving artifacts: embeddings=%s, ids=%d, subscores=%d, disqualified=%d",
        embeddings.shape,
        len(candidate_ids),
        len(subscores_dict),
        len(disqualified),
    )

    np.save(str(ARTIFACTS_DIR / "embeddings.npy"), embeddings)
    np.save(str(ARTIFACTS_DIR / "candidate_ids.npy"), candidate_ids)

    with open(ARTIFACTS_DIR / "subscores.pkl", "wb") as f:
        pickle.dump(subscores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(ARTIFACTS_DIR / "disqualified.json", "w") as f:
        json.dump(disqualified, f, indent=2)

    elapsed = time.time() - t0
    log.info("Done. Total: %d | Disqualified: %d | Honeypot: %d | Ghost: %d | Time: %.1f sec",
             total, len(disqualified), honeypot_count, ghost_count, elapsed)


if __name__ == "__main__":
    main()
