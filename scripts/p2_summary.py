"""P2 summary: CER/WER per OCR source x degradation level, per book, and leave-one-out.

Reads every data/ocr/*.parquet that exists, writes results/p2_cer_by_level.csv,
results/p2_cer_by_book.csv and results/p2_leave_one_out.csv.

    uv run python scripts/p2_summary.py
"""

from pathlib import Path

import duckdb

OCR = Path("data/ocr")
CORPUS = Path("data/corpus")


def main() -> None:
    files = sorted(OCR.glob("*.parquet"))
    if not files:
        raise SystemExit("no OCR outputs in data/ocr/")
    con = duckdb.connect()
    con.sql(f"""
        create view runs as
        select r.source, r.cer, r.wer, r.seconds, i.set, i.level, i.dropped_effect, p.title
        from read_parquet({[str(f) for f in files]}, union_by_name = true) r
        join '{CORPUS}/images.parquet' i using (image_id)
        join '{CORPUS}/pages.parquet' p on p.page_id = i.page_id
    """)

    by_level = con.sql("""
        select source, level, count(*) as pages,
               round(median(cer), 5) as median_cer, round(avg(cer), 5) as mean_cer,
               round(quantile_cont(cer, 0.9), 5) as p90_cer,
               round(median(wer), 5) as median_wer, round(avg(seconds), 3) as mean_seconds
        from runs where set = 'level' group by all order by source, level
    """)
    by_book = con.sql("""
        select source, title, level, round(median(cer), 5) as median_cer, round(avg(cer), 5) as mean_cer
        from runs where set = 'level' group by all order by source, title, level
    """)
    loo = con.sql("""
        with full_ as (select source, median(cer) as full_cer from runs
                       where set = 'level' and level = 5 group by source)
        select r.source, r.dropped_effect, count(*) as pages,
               round(median(r.cer), 5) as median_cer, round(any_value(f.full_cer), 5) as all_effects_cer
        from runs r join full_ f using (source)
        where r.set = 'loo' group by all order by r.source, median_cer
    """)

    Path("results").mkdir(exist_ok=True)
    by_level.write_csv("results/p2_cer_by_level.csv")
    by_book.write_csv("results/p2_cer_by_book.csv")
    loo.write_csv("results/p2_leave_one_out.csv")
    print(by_level)
    print(con.sql("""pivot (select source, title, level, median(cer) as m from runs where set = 'level' group by all)
                     on level using round(first(m), 4) order by source, title"""))
    print(loo)


if __name__ == "__main__":
    main()
