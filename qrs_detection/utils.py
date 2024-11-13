import wfdb

def read_record(record_name):
    record = wfdb.rdrecord(record_name)
    return record

def read_annotation(record_name):
    annotation = wfdb.rdann(record_name, 'atr')
    return annotation
