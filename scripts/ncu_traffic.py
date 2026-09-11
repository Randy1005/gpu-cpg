"""Read actual byte counters from Nsight's wide/raw or long CSV formats."""
import csv
import io

METRICS = ('dram__bytes_op_read.sum', 'dram__bytes_op_write.sum')

def parse_dram_csv(text):
    if '==ERROR==' in text:
        raise ValueError('Nsight reported an error')
    start = text.index('"ID",')
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    totals = {m: 0.0 for m in METRICS}
    ids = {m: set() for m in METRICS}
    wide = bool(rows) and all(m in rows[0] for m in METRICS)
    if wide:
        assert all(rows[0][m] == 'byte' for m in METRICS), 'Unexpected units'
    for row in rows:
        if not row.get('ID', '').isdigit():
            continue
        for m in METRICS:
            if wide:
                value = row[m]
            elif row.get('Metric Name') == m:
                assert row['Metric Unit'] == 'byte'
                value = row['Metric Value']
            else:
                continue
            assert row['ID'] not in ids[m], 'Duplicate kernel metric'
            ids[m].add(row['ID'])
            totals[m] += float(value.replace(',', ''))
    assert ids[METRICS[0]] and ids[METRICS[0]] == ids[METRICS[1]], 'Missing metrics'
    return dict(kernels=len(ids[METRICS[0]]),
                dram_read_bytes=totals[METRICS[0]],
                dram_write_bytes=totals[METRICS[1]])
