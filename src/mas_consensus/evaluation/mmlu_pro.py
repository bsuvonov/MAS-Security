from .csqa import evaluate_csqa


def evaluate_mmlu_pro(dataset_path, output_path, attacker_num, auditor_num, type):
    # MMLU-Pro shares the same structure as CSQA, so reuse the CSQA evaluator.
    return evaluate_csqa(dataset_path, output_path, attacker_num, auditor_num, type)
