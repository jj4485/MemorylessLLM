import pandas as pd
from datasets import dataset

data = [
    {
        "prompt": "Explain the differences between supervised and unsupervised learning.",
        "good_response": "Supervised learning uses labeled data...",
        "bad_response": "Unsupervised learning is the same as supervised..."
    },
    {
        "prompt": "What are the health benefits of green tea?",
        "good_response": "Green tea may help with heart health...",
        "bad_response": "Green tea cures cancer instantly."
    },
]