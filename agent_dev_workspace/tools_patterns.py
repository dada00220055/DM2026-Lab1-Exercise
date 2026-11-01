"""Tools for frequent pattern mining using PAMI algorithms."""

import os
import tempfile
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth as FPGrowthModule
from PAMI.frequentPattern.topk import FAE as FAEModule
from PAMI.frequentPattern.maximal import MaxFPGrowth as MaxFPGrowthModule


def make_tools(session):
    """Build and return pattern mining tools bound to the given session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent term patterns for a specific document category using PAMI.

        Fits a fresh category-local CountVectorizer(stop_words='english'), applies percentile term
        filtering, converts to a transactional database, and mines patterns using FPGrowth, Top-K,
        or MaxFPGrowth.

        Args:
            category_name: Target category label to mine.
            filtering_method: Vocabulary filter technique: 'variance', 'tfidf', or 'term_frequency'.
            algorithm: Mining algorithm: 'fpgrowth', 'topk', or 'maxfpgrowth'.
            min_sup: Minimum support count threshold for fpgrowth and maxfpgrowth.
            k: Number of highest-support patterns to keep for topk.

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, patterns list,
            and note string if no terms/patterns survived.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe must be loaded before pattern mining.")

        # Identify category column
        cat_col = "category_name" if "category_name" in session.dataframe.columns else ("label" if "label" in session.dataframe.columns else None)
        if not cat_col:
            raise ValueError("No category or label column found in session.dataframe.")

        # Filter dataframe to target category
        cat_df = session.dataframe[session.dataframe[cat_col].astype(str) == str(category_name)].copy()
        if len(cat_df) == 0:
            raise ValueError(f"No documents found for category '{category_name}'.")

        texts = cat_df["text"].fillna("").astype(str).tolist()

        # Step 1: Fit fresh per-category CountVectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        try:
            local_dtm = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Vectorizer fails if all terms are English stop words or empty
            feature_names = np.array([])
            local_dtm = None

        if local_dtm is None or len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        dense_local = local_dtm.toarray()
        n_docs, n_terms = dense_local.shape

        # Step 2: Apply chosen vocabulary filtering method
        kept_mask = np.ones(n_terms, dtype=bool)

        if filtering_method == "variance":
            scores = np.var(dense_local, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > p5) & (scores < p95)

        elif filtering_method == "term_frequency":
            scores = np.sum(dense_local, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > p5) & (scores < p95)

        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(local_dtm).toarray()
            scores = np.mean(tfidf, axis=0)
            p20 = np.percentile(scores, 20)
            kept_mask = scores >= p20

        surviving_terms = feature_names[kept_mask]
        surviving_matrix = dense_local[:, kept_mask]

        # Step 3: Handle 0 surviving terms cleanly
        if len(surviving_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            args = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            }
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        # Step 4: Convert surviving terms into transactional DataFrame
        df_trans = pd.DataFrame(surviving_matrix, columns=surviving_terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "transactions.txt")
            out_path = os.path.join(tmpdir, "patterns.txt")

            # Initialize DF2DB(df_trans, "dense") and convert with condition ">=" and thresholdValue 1
            converter = db_converter.DF2DB(df_trans, "dense")
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            # Mine with requested PAMI algorithm class
            if algorithm == "fpgrowth":
                miner = FPGrowthModule.FPGrowth(iFile=db_path, minSup=min_sup)
                miner.mine()
                miner.save(out_path)

            elif algorithm == "topk":
                miner = FAEModule.FAE(iFile=db_path, k=k)
                miner.mine()
                miner.save(out_path)

            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowthModule.MaxFPGrowth(iFile=db_path, minSup=min_sup)
                miner.mine()
                miner.save(out_path)
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}'.")

            # Read mined patterns
            patterns = []
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(":")
                        if len(parts) == 2:
                            itemset_str, sup_str = parts[0].strip(), parts[1].strip()
                            items = [item.strip() for item in itemset_str.split("\t") if item.strip()]
                            if not items:
                                items = [item.strip() for item in itemset_str.split(" ") if item.strip()]
                            try:
                                sup_val = int(float(sup_str))
                            except ValueError:
                                sup_val = 0
                            patterns.append({"pattern": items, "support": sup_val})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns),
            "patterns": patterns,
        }

        session.store_result("mine_patterns_tool", args, summary)
        return summary

    return [mine_patterns_tool]
