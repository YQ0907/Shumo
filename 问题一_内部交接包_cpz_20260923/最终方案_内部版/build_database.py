"""重建问题一内部 SQLite 证据库；只写本目录 data/q1_evidence.sqlite。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q1 = HERE.parent
DB = HERE / "data" / "q1_evidence.sqlite"


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source(con, path: Path, role: str, state: str):
    assert path.is_file(), path
    con.execute("INSERT INTO source_file VALUES (?,?,?,?,?)", (str(path.relative_to(ROOT)), role, state, path.stat().st_size, sha256(path)))


def build():
    DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB.with_suffix(".building.sqlite")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE source_file(path TEXT PRIMARY KEY, role TEXT NOT NULL, evidence_state TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE domain_mapping(mixture_domain TEXT PRIMARY KEY, quality_domain TEXT, mapping_type TEXT NOT NULL, note TEXT NOT NULL);
        CREATE TABLE quality_record(source TEXT NOT NULL, id TEXT NOT NULL, domain TEXT NOT NULL, eligible INTEGER NOT NULL,
          q_group REAL, q_flat REAL, q_median REAL, q_pca_structural REAL, conflict_four_initial INTEGER NOT NULL,
          PRIMARY KEY(source,id));
        CREATE TABLE quality_recheck(source TEXT NOT NULL, id TEXT NOT NULL, q_group_alt REAL, q_flat_alt REAL,
          semantic_conflict_three INTEGER NOT NULL, ngram_disagreement INTEGER NOT NULL,
          PRIMARY KEY(source,id), FOREIGN KEY(source,id) REFERENCES quality_record(source,id));
        CREATE TABLE ai_blind_rating(id TEXT PRIMARY KEY, domain TEXT NOT NULL, char_count INTEGER, snippet TEXT,
          information_0_3 INTEGER, readability_0_3 INTEGER, noise_0_3 INTEGER, note TEXT, rater_type TEXT NOT NULL);
        CREATE TABLE mixture_run(split TEXT NOT NULL, scale TEXT NOT NULL, sample_index INTEGER NOT NULL,
          observed INTEGER NOT NULL, overlaps_train INTEGER NOT NULL, raw_sum REAL NOT NULL,
          PRIMARY KEY(split,scale,sample_index));
        CREATE TABLE mixture_weight(split TEXT NOT NULL, scale TEXT NOT NULL, sample_index INTEGER NOT NULL, domain TEXT NOT NULL,
          raw_weight REAL NOT NULL, normalized_weight REAL NOT NULL,
          PRIMARY KEY(split,scale,sample_index,domain), FOREIGN KEY(split,scale,sample_index) REFERENCES mixture_run(split,scale,sample_index));
        CREATE TABLE validation_loss(split TEXT NOT NULL, scale TEXT NOT NULL, sample_index INTEGER NOT NULL, domain TEXT NOT NULL,
          loss REAL NOT NULL, PRIMARY KEY(split,scale,sample_index,domain),
          FOREIGN KEY(split,scale,sample_index) REFERENCES mixture_run(split,scale,sample_index));
        CREATE TABLE model_route(route TEXT PRIMARY KEY, role TEXT NOT NULL, settings TEXT NOT NULL, evidence TEXT NOT NULL);
        CREATE TABLE model_metric(route TEXT NOT NULL, sample TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL,
          PRIMARY KEY(route,sample,metric), FOREIGN KEY(route) REFERENCES model_route(route));
        CREATE TABLE transfer_effect(target TEXT NOT NULL, model TEXT NOT NULL, mean_delta REAL NOT NULL,
          p10 REAL NOT NULL, p90 REAL NOT NULL, n_supported INTEGER NOT NULL, PRIMARY KEY(target,model));
        CREATE TABLE pair_effect(target_a TEXT NOT NULL, target_b TEXT NOT NULL, model TEXT NOT NULL,
          interaction REAL NOT NULL, n_supported INTEGER NOT NULL, PRIMARY KEY(target_a,target_b,model));
        CREATE TABLE bibliography(ref_id INTEGER PRIMARY KEY, title TEXT NOT NULL, year INTEGER, local_pdf TEXT, url TEXT, q1_role TEXT NOT NULL);
        CREATE VIEW v_quality_summary AS SELECT source,domain,COUNT(*) n,AVG(q_group) mean_q,
          SUM(conflict_four_initial) initial_four_conflicts FROM quality_record GROUP BY source,domain;
        CREATE VIEW v_conflict_summary AS SELECT q.source,q.domain,COUNT(*) n,SUM(r.semantic_conflict_three) semantic_three,
          SUM(r.ngram_disagreement) ngram_disagreement FROM quality_record q JOIN quality_recheck r USING(source,id)
          GROUP BY q.source,q.domain;
        CREATE VIEW v_mixture_summary AS SELECT split,scale,observed,COUNT(*) n,MAX(ABS(raw_sum-1)) max_raw_sum_error,
          SUM(overlaps_train) overlaps_train FROM mixture_run GROUP BY split,scale,observed;
        """)

        for relative, role, state in [
            ("real_attachments/source_manifest.json","original attachment manifest","provenance"),
            ("real_attachments/A_data_value/domain_mapping_guide.csv","A16 quality-to-mixture mapping guide","original"),
            ("real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz","A1 original quality texts/signals","original"),
            ("real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz","A2 original quality texts/signals","original"),
            ("real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz","A3 original quality texts/signals","original"),
            ("问题一研究/cross_session_results/quality_conflict_reconciliation.json","cross-session quality/conflict reconciliation","derived"),
            ("问题一研究/conflict_results/policy_summary.json","conflict policies experiment","derived"),
            ("问题一研究/quality_results/AI盲评核验摘要.json","AI snippet rating comparison","AI judgment"),
            ("问题一研究/mixture_results/effect_protocol.json","local effect scenario protocol","derived"),
        ]:
            source(con,ROOT/relative,role,state)
        for r in rows(ROOT/"real_attachments/A_data_value/domain_mapping_guide.csv"):
            qdomain=None if r["quality_domain"]=="(none)" else r["quality_domain"]
            con.execute("INSERT INTO domain_mapping VALUES (?,?,?,?)",(r["mixture_domain"],qdomain,r["mapping_type"],r["note"]))

        # 质量：两次独立流水线逐条留存；原文只在原始 xz 中，不重复入库。
        for code in ("A1", "A2", "A3"):
            p = Q1 / "quality_results" / "sample_scores.csv"
            if code == "A1":
                source(con,p,"quality scores F_Q1-1","derived")
            batch = []
            for r in rows(p):
                if r["source"] != code:
                    continue
                batch.append((r["source"],r["id"],r["domain"],int(r["eligible"]),float(r["group_equal"]),
                              float(r["flat_equal"]),float(r["group_median"]),float(r["pca_structural"]),int(r["conflict"])))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO quality_record VALUES (?,?,?,?,?,?,?,?,?)",batch); batch.clear()
            con.executemany("INSERT INTO quality_record VALUES (?,?,?,?,?,?,?,?,?)",batch)
            alt = ROOT / "问题一质量实验" / f"{code}_record_scores.csv"
            source(con,alt,f"quality independent recheck {code}","derived")
            batch = []
            for r in rows(alt):
                semantic3=int(any(int(r[k]) for k in ("fineweb_vs_qurater_education","fluency_vs_readability","professionalism_vs_expertise")))
                batch.append((code,r["id"],float(r["Q"]),float(r["Q_equal22"]),semantic3,int(r["two_vs_three_gram"])))
                if len(batch) >= 10000:
                    con.executemany("INSERT INTO quality_recheck VALUES (?,?,?,?,?,?)",batch); batch.clear()
            con.executemany("INSERT INTO quality_recheck VALUES (?,?,?,?,?,?)",batch)

        p = Q1 / "quality_results" / "原文盲评样本_子代理AI评分.csv"
        source(con,p,"AI-only snippet ratings; not human ground truth","AI judgment")
        con.executemany("INSERT INTO ai_blind_rating VALUES (?,?,?,?,?,?,?,?,?)", [
            (r["id"],r["domain"],int(r["原文字符数"]),r["原文片段_前1500字符"],int(r["AI信息价值_0到3"]),
             int(r["AI可读性_0到3"]),int(r["AI噪声_0到3"]),r["AI备注"],r["评分来源"]) for r in rows(p)])

        tables = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"
        specs = [("train","1m","train_mixture_1m.csv","train_pile_loss_1m.csv",1),
                 ("test","1m","test_mixture_1m.csv","test_pile_loss_1m.csv",1),
                 ("test","60m","test_mixture_60m.csv","test_pile_loss_60m.csv",1),
                 ("test","1B","test_mixture_1B.csv","test_pile_loss_1B.csv",1),
                 ("estimated","10b","est_mixture_10b.csv","est_pile_loss_10b.csv",0),
                 ("estimated","70b","est_mixture_70b.csv","est_pile_loss_70b.csv",0)]
        train_recipes = set()
        for split,scale,mfile,lfile,observed in specs:
            mp,lp=tables/mfile,tables/lfile
            source(con,mp,"17-domain mixture input", "observed" if observed else "estimated")
            source(con,lp,"13-domain validation loss", "observed" if observed else "estimated")
            mixes={int(r["index"]):r for r in rows(mp)}
            losses={int(r["index"]):r for r in rows(lp)}
            assert mixes.keys()==losses.keys(), (mfile,lfile)
            for idx,r in mixes.items():
                w={k:float(v) for k,v in r.items() if k!="index"}
                total=sum(w.values())
                assert total>0 and abs(total-1)<0.005, (mfile,idx,total)
                recipe=tuple(round(v/total,6) for v in w.values())
                if split=="train": train_recipes.add(recipe)
                overlap=int(recipe in train_recipes)
                con.execute("INSERT INTO mixture_run VALUES (?,?,?,?,?,?)",(split,scale,idx,observed,overlap,total))
                con.executemany("INSERT INTO mixture_weight VALUES (?,?,?,?,?,?)",[
                    (split,scale,idx,k,v,v/total) for k,v in w.items()])
                con.executemany("INSERT INTO validation_loss VALUES (?,?,?,?,?)",[
                    (split,scale,idx,k,float(v)) for k,v in losses[idx].items() if k!="index"])

        model_json=Q1/"mixture_results"/"model_comparison.json"
        kernel_json=Q1/"cross_session_results"/"harmonized_kernel_results.json"
        source(con,model_json,"F_Q1-1 model experiment","derived")
        source(con,kernel_json,"harmonized kernel five-fold experiment","derived")
        models=json.loads(model_json.read_text(encoding="utf-8"))["models"]
        selected=json.loads(kernel_json.read_text(encoding="utf-8"))["selected"]
        roles={"linear_ridge":"linear baseline","quadratic_ridge":"interpretable local comparison",
               "rbf_kernel":"raw-weight kernel comparator","extra_trees":"equal-weight mean-loss selection",
               "log_excess_law":"smooth scale-law comparator"}
        for key,v in models.items():
            con.execute("INSERT INTO model_route VALUES (?,?,?,?)",(key,roles[key],json.dumps(v["best_param"],ensure_ascii=False),"F_Q1-1"))
            for sample,metrics in [("train_cv",v["train_cv"]),*v["checks"].items()]:
                for metric,val in metrics.items():
                    if isinstance(val,(int,float)) and not isinstance(val,bool):
                        con.execute("INSERT INTO model_metric VALUES (?,?,?,?)",(key,sample,metric,float(val)))
        for key,v in selected.items():
            route="kernel_"+key
            con.execute("INSERT INTO model_route VALUES (?,?,?,?)",(route,"13-domain prediction" if key=="ilr_eps_0.001" else "kernel sensitivity",
                        json.dumps({k:v[k] for k in ("eps","alpha","gamma")},ensure_ascii=False),"harmonized same-five-fold"))
            for sample,metrics in [("train_cv",v["train_cv"]),*v["checks"].items()]:
                for metric,val in metrics.items():
                    if isinstance(val,(int,float)) and not isinstance(val,bool):
                        con.execute("INSERT INTO model_metric VALUES (?,?,?,?)",(route,sample,metric,float(val)))

        p=Q1/"mixture_results"/"one_point_transfers.csv"
        source(con,p,"local 1pp transfer scenarios","derived")
        for r in rows(p):
            assert r["source"]=="train_the_pile_pile_cc"
            for m in ("linear_ridge","quadratic_ridge","log_excess_law","extra_trees"):
                con.execute("INSERT INTO transfer_effect VALUES (?,?,?,?,?,?)",(r["target"],m,float(r[f"{m}_mean_delta_loss"]),
                            float(r[f"{m}_p10_delta_loss"]),float(r[f"{m}_p90_delta_loss"]),int(r["n_local_supported"])))
        p=Q1/"mixture_results"/"two_domain_interactions.csv"
        source(con,p,"local two-domain interactions; observational model scenarios","derived")
        for r in rows(p):
            assert r["source"]=="train_the_pile_pile_cc"
            for m in ("linear_ridge","quadratic_ridge","log_excess_law","extra_trees"):
                con.execute("INSERT INTO pair_effect VALUES (?,?,?,?,?)",(r["target_a"],r["target_b"],m,
                            float(r[f"{m}_mean_interaction"]),int(r["n_local_supported"])))

        papers=[
          (1,"Explaining neural scaling laws",2024,"01_Bahri_Explaining_neural_scaling_laws.pdf","https://doi.org/10.1073/pnas.2311878121","跨规模机制边界"),
          (2,"Training Compute-Optimal Large Language Models",2022,"02_Hoffmann_Training_compute_optimal_LLMs.pdf","https://arxiv.org/abs/2203.15556","算力约束与问题二衔接"),
          (3,"RegMix: Data Mixture as Regression for Language Model Pre-training",2025,"03_Liu_RegMix.pdf","https://proceedings.iclr.cc/paper_files/paper/2025/hash/5f67d864aae6115374fed7beddd119e0-Abstract-Conference.html","配比回归主文献；原题作者/会议有误"),
          (4,"Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling",2023,"04_Biderman_Pythia.pdf","https://arxiv.org/abs/2304.01373","受控跨规模比较"),
          (5,"Synthesizing scientific literature with retrieval-augmented language models",2026,"05_Asai_Synthesizing_scientific_literature.pdf","https://doi.org/10.1038/s41586-025-10072-4","能力目标边界"),
          (6,"Are Emergent Abilities of Large Language Models a Mirage?",2023,"06_Schaeffer_Emergent_abilities_mirage.pdf","https://arxiv.org/abs/2304.15004","连续损失与阈值指标解释")]
        for i,title,year,pdf,url,role in papers:
            pdfpath=ROOT/"原题参考文献"/pdf
            source(con,pdfpath,f"reference [{i}]","local PDF")
            con.execute("INSERT INTO bibliography VALUES (?,?,?,?,?,?)",(i,title,year,str(pdfpath.relative_to(ROOT)),url,role))
        con.execute("CREATE INDEX ix_quality_domain ON quality_record(source,domain)")
        con.execute("CREATE INDEX ix_loss_domain ON validation_loss(domain,scale)")
        con.commit()
        assert con.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        assert con.execute("PRAGMA foreign_key_check").fetchall()==[]
        assert con.execute("SELECT COUNT(*) FROM quality_record").fetchone()[0]==272505
        assert con.execute("SELECT COUNT(*) FROM quality_recheck").fetchone()[0]==272505
        assert con.execute("SELECT COUNT(*) FROM ai_blind_rating").fetchone()[0]==135
        assert con.execute("SELECT COUNT(*) FROM mixture_run").fetchone()[0]==1214
        assert con.execute("SELECT COUNT(*) FROM mixture_weight").fetchone()[0]==1214*17
        assert con.execute("SELECT COUNT(*) FROM validation_loss").fetchone()[0]==1214*13
        assert con.execute("SELECT COUNT(*) FROM source_file").fetchone()[0]==36
        assert con.execute("SELECT mapping_type,COUNT(*) FROM domain_mapping GROUP BY mapping_type ORDER BY mapping_type").fetchall()==[
            ("direct",3),("inferred",11),("near_direct",3)]
        assert con.execute("SELECT source,SUM(semantic_conflict_three) FROM quality_recheck GROUP BY source ORDER BY source").fetchall()==[("A1",965),("A2",0),("A3",4173)]
        assert con.execute("SELECT split,scale,SUM(overlaps_train) FROM mixture_run GROUP BY split,scale ORDER BY split,scale").fetchall()==[
            ("estimated","10b",63),("estimated","70b",63),("test","1B",0),("test","1m",0),("test","60m",0),("train","1m",512)]
        print("quality",con.execute("SELECT source,COUNT(*) FROM quality_record GROUP BY source").fetchall())
        print("mixture",con.execute("SELECT split,scale,COUNT(*) FROM mixture_run GROUP BY split,scale").fetchall())
        print("tables",con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())
    finally:
        con.close()
    os.replace(tmp,DB)
    print("database",DB,"bytes",DB.stat().st_size)


if __name__=="__main__":
    build()
