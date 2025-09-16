import pandas as pd
import numpy as np
from datetime import datetime
import pymupdf
import pymupdf4llm
from PIL import Image
import os
import io
import spacy
from sentence_transformers import SentenceTransformer
import openai
from transformers import pipeline, AutoTokenizer