# Lint as: python3
"""
Use logistic regression to prect the politeness score
"""

import logging
logging.basicConfig(
    filename="main/train_polite_score.log", filemode="w", level=logging.INFO)
logging.basicConfig(level=logging.INFO)
from src import download_data
download_data.download_data()
from src import convo_politeness
from src import receive_data

[training_data, _] = receive_data.receive_data()
training_data = training_data.dropna()
convo_politeness.cross_validate(training_data)
