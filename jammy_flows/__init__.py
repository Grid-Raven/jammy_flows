from .main.default import pdf
from .main.fully_amortized import fully_amortized_pdf
from ._precision_policy import apply_s2_float64_policy


apply_s2_float64_policy(pdf)
del apply_s2_float64_policy
