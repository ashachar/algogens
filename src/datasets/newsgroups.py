"""
20 Newsgroups dataset loader for clustering benchmark v2.

Provides two settings:
  - 20NG-diverse: 5 well-separated categories (easy)
  - 20NG-similar: 5 confusable comp.* categories (hard)

Both available via sklearn (no extra dependencies).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.datasets import fetch_20newsgroups


@dataclass
class Document:
    doc_id: int
    text: str
    category: str
    category_id: int
    title: Optional[str] = None


# Well-separated categories for the "diverse" setting
DIVERSE_CATEGORIES = [
    'rec.sport.baseball',
    'sci.space',
    'comp.graphics',
    'talk.politics.misc',
    'soc.religion.christian',
]

# Confusable categories for the "similar" setting
SIMILAR_CATEGORIES = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
]


def _clean_newsgroup_text(text: str) -> str:
    """Remove headers, footers, and quotes from newsgroup posts."""
    lines = text.split('\n')
    cleaned = []
    in_header = True
    for line in lines:
        if in_header:
            if line.strip() == '':
                in_header = False
            continue
        # Skip quoted lines
        if line.startswith('>') or line.startswith('|'):
            continue
        # Skip signature markers
        if line.strip() == '--':
            break
        cleaned.append(line)
    result = '\n'.join(cleaned).strip()
    # If cleaning removed everything, fall back to raw text
    return result if len(result) > 50 else text.strip()


def load_newsgroups(setting: str = "diverse",
                    max_docs_per_category: int = 200,
                    random_state: int = 42) -> list[Document]:
    """Load 20 Newsgroups documents for the specified setting.

    Args:
        setting: "diverse" (5 well-separated) or "similar" (5 confusable comp.*)
        max_docs_per_category: max documents to load per category
        random_state: random seed for shuffling

    Returns:
        list of Document objects
    """
    categories = DIVERSE_CATEGORIES if setting == "diverse" else SIMILAR_CATEGORIES

    data = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state,
        shuffle=True,
    )

    # Group by category, take max_docs_per_category from each
    rng = np.random.RandomState(random_state)
    cat_docs = {i: [] for i in range(len(categories))}
    for text, target in zip(data.data, data.target):
        cat_docs[target].append(text)

    documents = []
    doc_id = 0
    for cat_id, category in enumerate(categories):
        texts = cat_docs[cat_id]
        rng.shuffle(texts)
        for text in texts[:max_docs_per_category]:
            cleaned = _clean_newsgroup_text(text)
            if len(cleaned) < 30:
                continue
            documents.append(Document(
                doc_id=doc_id,
                text=cleaned,
                category=category,
                category_id=cat_id,
            ))
            doc_id += 1

    return documents


def split_dev_test(documents: list[Document],
                   dev_fraction: float = 0.3,
                   random_state: int = 42) -> tuple[list[Document], list[Document]]:
    """Split documents into DEV and TEST sets, stratified by category.

    Args:
        documents: list of Document objects
        dev_fraction: fraction of documents for DEV set
        random_state: random seed

    Returns:
        (dev_documents, test_documents)
    """
    rng = np.random.RandomState(random_state)

    # Group by category
    by_category = {}
    for doc in documents:
        by_category.setdefault(doc.category_id, []).append(doc)

    dev_docs, test_docs = [], []
    for cat_id in sorted(by_category.keys()):
        cat_list = by_category[cat_id]
        rng.shuffle(cat_list)
        n_dev = max(1, int(len(cat_list) * dev_fraction))
        dev_docs.extend(cat_list[:n_dev])
        test_docs.extend(cat_list[n_dev:])

    return dev_docs, test_docs
