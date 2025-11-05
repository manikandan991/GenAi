from ragas.metrics import DiscreteMetric
from typing import Dict, Any

# A simple discrete metric to check completeness of the TA report
ta_metric = DiscreteMetric(
    name="ta_report_checks",
    prompt=(
        "You are checking a technical analysis report for completeness based on facts given.\n"
        "Pass if ALL are true: (1) mentions trend, (2) proposes an entry with rationale, (3) includes stop-loss, "
        "(4) gives 6-month scenario, (5) mentions sentiment from news if available, (6) includes a disclaimer. "
        "Otherwise Fail.\n\n"
        "Question: {question}\nExpected Answer: Provide a complete TA plan.\nModel Response: {response}\nEvaluation:"
    ),
    allowed_values=["pass","fail"],
)
