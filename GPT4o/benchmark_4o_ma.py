# Benchmark GPT-4o on MA dataset

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
import re
import openai