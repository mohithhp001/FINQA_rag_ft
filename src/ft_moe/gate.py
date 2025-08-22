NUMERIC_HINTS = [
    "₹"," rs","inr","%","ratio","deposits","advances","pat","profit","eps",
    "gnpa","nnpa","car","crar","roa","roe","income","margin","assets","liabilities"
]

def pick_expert(question: str) -> str:
    ql = question.lower()
    return "num" if any(h in ql for h in [h.lower() for h in NUMERIC_HINTS]) else "nar"
