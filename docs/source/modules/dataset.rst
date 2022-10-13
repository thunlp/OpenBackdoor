Dataset
===================================

OpenBackdoor integrates 5 tasks and 11 datasets, which can be downloaded from bash scripts in ``datasets``. We list the tasks and datasets below:

- **Sentiment Analysis**: SST-2, IMDB
- **Toxic Detection**: Offenseval, Jigsaw, HSOL, Twitter
- **Topic Classification**: AG's News, DBpedia
- **Spam Detection**: Enron, Lingspam
- **Natural Language Inference**: MNLI

APIs
--------------------------------

Base class of data processor

.. autoclass:: openbackdoor.DataProcessor
   :members: